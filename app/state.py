import operator
from typing import TypedDict, List, Dict, Any, Annotated, Union
import json
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from app.sandbox import run_command
from app.llm import get_chat_model
from app.template import RepoNavigatorResponse, SearchResultItem
from app.helper_tools import parse_ripgrep_output
from app.llm_tools import NAVIGATOR_TOOLS, run_tree, run_ripgrep, find_files, read_file_snippet

class MonorepoState(TypedDict):
    # Initial Inputs
    issue_title: str
    issue_description: str
    config: Dict[str, Any]

    # Discovery Phase
    project_root: str
    target_packages: List[str]       # Affected sub-packages
    filesystem_map: str              # Visual structure via `tree`
    search_results: Union[List[SearchResultItem], Dict[str, Any]]   # Outputs from ripgrep
    relevant_files: List[str]        # Resolved absolute/relative file paths

    # Execution Phase
    proposed_plan: str               # Step-by-step logic drafted by Planner
    file_contents: Dict[str, str]    # Current code snippets being modified
    diffs_to_apply: List[Dict[str, str]] # Format: {"file": "...", "search": "...", "replace": "..."}

    # Verification Phase
    test_command: str                # e.g., "pytest packages/core/tests"
    test_stdout: str                 # Captured standard output
    test_stderr: str                 # Captured error logs
    is_resolved: bool                # Pass/Fail status

    # Guardrails
    iteration_count: Annotated[int, operator.add] # Monotonically increasing counter


TOOL_MAP = {
    "run_tree": run_tree,
    "run_ripgrep": run_ripgrep,
    "find_files": find_files,
    "read_file_snippet": read_file_snippet
}


def convert_langchain_messages_to_responses_input(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    Converts LangChain message objects into the payload format 
    expected by the OpenAI / LiteLLM Responses API.
    """
    formatted_input = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            formatted_input.append({"role": "developer", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            formatted_input.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            item = {"role": "assistant"}
            if msg.content:
                item["content"] = msg.content
            if getattr(msg, "tool_calls", None):
                item["tool_calls"] = msg.tool_calls
            formatted_input.append(item)
        elif isinstance(msg, ToolMessage):
            formatted_input.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content
            })
    return formatted_input


def repo_navigator_node(state: MonorepoState) -> MonorepoState:
    """
    LangGraph Node: Autonomously explores the repository using litellm.responses(),
    executes tree/ripgrep, and updates MonorepoState.
    """
    config = state.get("config", {})
    
    system_prompt = (
        "You are RepoNavigator, an autonomous codebase exploration agent.\n"
        "Your task is to explore the repository using tools (run_tree, run_ripgrep, find_files, read_file_snippet) "
        "to discover affected sub-packages and pinpoint relevant source files for the reported issue.\n"
        "Keep your tool calls minimal and focused."
    )
    
    user_prompt = f"Issue Title: {state['issue_title']}\nDescription: {state['issue_description']}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # Tracking raw tool outputs across iterations
    captured_tree_outputs: List[str] = []
    captured_search_results: List[SearchResultItem] = []
    
    max_search_turns = config.get("max_search_turns", 4)
    
    # 1. Autonomous Discovery Loop
    for _ in range(max_search_turns):
        formatted_messages = convert_langchain_messages_to_responses_input(messages)
        
        # Invoke LiteLLM Responses API directly with tools
        response = get_chat_model(config, messages=formatted_messages, tools=NAVIGATOR_TOOLS)
        
        choice = response.choices[0]
        message_output = choice.message
        tool_calls = getattr(message_output, "tool_calls", None)
        
        ai_msg = AIMessage(content=message_output.content or "", tool_calls=tool_calls or [])
        messages.append(ai_msg)
        
        # Break if the model did not request additional tool calls
        if not tool_calls:
            break
            
        for tool_call in tool_calls:
            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.function.name
            tool_args = tool_call.get("args") if isinstance(tool_call, dict) else json.loads(tool_call.function.arguments)
            tool_id = tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id
            
            if tool_name in TOOL_MAP:
                raw_output = TOOL_MAP[tool_name].invoke(tool_args)
                
                # Record evidence for graph state
                if tool_name == "run_tree":
                    captured_tree_outputs.append(str(raw_output))
                elif tool_name == "run_ripgrep":
                    captured_search_results.extend(parse_ripgrep_output(str(raw_output)))
            else:
                raw_output = f"Error: Tool '{tool_name}' not recognized."
                
            messages.append(ToolMessage(content=str(raw_output), tool_call_id=tool_id))

    # 2. Enforce Structured Consolidation Output
    final_prompt = (
        "Consolidate your findings from the search tool results above.\n"
        "Return a valid JSON object matching the requested schema keys: "
        "target_packages (list of strings), filesystem_map (string), search_results (list), relevant_files (list of strings)."
    )
    messages.append(HumanMessage(content=final_prompt))
    
    formatted_messages = convert_langchain_messages_to_responses_input(messages)
    final_response = get_chat_model(config, messages=formatted_messages, tools=None)
    
    content_text = final_response.choices[0].message.content
    try:
        parsed_data = json.loads(content_text)
    except Exception:
        parsed_data = {}

    # 3. Update MonorepoState with the schema fields
    state["target_packages"] = parsed_data.get("target_packages", [])
    
    # Fallback to captured tool execution if summary trimmed the raw tree
    state["filesystem_map"] = parsed_data.get("filesystem_map") or (
        captured_tree_outputs[-1] if captured_tree_outputs else ""
    )
    
    # Merge search results
    state["search_results"] = parsed_data.get("search_results") or captured_search_results
    state["relevant_files"] = parsed_data.get("relevant_files", [])

    return state


from app.template import PlannerCoderResponse


def fetch_file_contents(file_paths: List[str]) -> Dict[str, str]:
    """
    Reads the target files from the Docker sandbox.
    Adds line numbers for context clarity and saves raw content.
    """
    file_contents: Dict[str, str] = {}
    
    for path in file_paths:
        cmd = f"cat {path}"
        result = run_command(cmd)
        
        if result["exit_code"] == 0:
            file_contents[path] = result["stdout"]
        else:
            file_contents[path] = f"// Error reading file: {result['stderr']}"
            
    return file_contents


def planner_node(state: MonorepoState) -> MonorepoState:
    """
    LangGraph Node: Analyzes relevant files and issue context using litellm.responses(),
    drafts a deterministic fix, and generates search/replace diff blocks.
    """
    config = state.get("config", {})
    
    # Step 1: Read the current content of relevant files from sandbox
    relevant_files = state.get("relevant_files", [])
    file_contents = fetch_file_contents(relevant_files)
    
    # Step 2: Assemble System Prompt with Strict Patching Guardrails
    system_prompt = (
        "You are an expert Monorepo Software Engineer and Planner.\n"
        "Your task is to analyze the issue and target files, formulate a step-by-step fix, "
        "and generate exact search-and-replace code diffs in JSON format.\n\n"
        "STRICT PATCHING RULES:\n"
        "1. NEVER rewrite the entire file. Only output minimal search-and-replace blocks.\n"
        "2. The `search` block MUST exist verbatim in the target file, including exact whitespace and indentation.\n"
        "3. Keep `search` blocks unique enough (3–10 lines) to match only the target location.\n"
        "4. Provide a targeted test command (e.g., `pytest packages/core/tests/test_auth.py`) to verify the fix."
    )
    
    # Format current file contents for prompt context
    formatted_code_context = ""
    for path, content in file_contents.items():
        formatted_code_context += f"\n--- FILE: {path} ---\n{content}\n"

    # Step 3: Check if this is a Self-Healing Retry iteration
    retry_context = ""
    if state.get("test_stderr"):
        retry_context = f"""
🚨 PREVIOUS TEST FAILED (Iteration {state.get('iteration_count', 0)}):
Command: {state.get('test_command')}
Stderr / Stacktrace:
{state.get('test_stderr')}

Please analyze the failure stack trace above, adjust your plan, and output corrected diffs.
"""

    user_prompt = f"""
Issue Title: {state['issue_title']}
Issue Description: {state['issue_description']}
Target Packages: {state.get('target_packages', [])}

=== TARGET FILE CONTENTS ===
{formatted_code_context}
{retry_context}
"""

    messages = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Step 4: Call Model via Responses API
    response = get_chat_model(config, messages=messages, tools=None)
    content_text = response.choices[0].message.content
    
    try:
        parsed_response = json.loads(content_text)
    except Exception:
        parsed_response = {}
        
    # Step 5: Update Graph State
    state["file_contents"] = file_contents
    state["proposed_plan"] = parsed_response.get("proposed_plan", "")
    state["test_command"] = parsed_response.get("test_command", "")
    state["diffs_to_apply"] = parsed_response.get("diffs_to_apply", [])
    
    return state