import operator
from typing import TypedDict, List, Dict, Any, Annotated
import json
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from app.sandbox import run_command
from app.state import MonorepoState
from app.llm import get_chat_model
from app.template import RepoNavigatorResponse, SearchResultItem
from app.helper_tools import parse_ripgrep_output
from app.llm_tools import NAVIGATOR_TOOLS, run_tree, run_ripgrep, find_files, read_file_snippet

class MonorepoState(TypedDict):
    # Initial Inputs
    issue_title: str
    issue_description: str

    # Discovery Phase
    project_root: str
    target_packages: List[str]       # Affected sub-packages
    filesystem_map: str              # Visual structure via `tree`
    search_results: Dict[str, str]   # Outputs from ripgrep
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

def repo_navigator_node(state: MonorepoState) -> MonorepoState:
    """
    LangGraph Node: Autonomously explores the repository, executes tree/ripgrep,
    and updates MonorepoState with RepoNavigatorResponse attributes.
    """
    config = state.get("config", {})
    llm = get_chat_model(config)
    
    # Bind navigation tools to the model slot
    llm_with_tools = llm.bind_tools(NAVIGATOR_TOOLS)
    
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
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Break if the model did not request additional tool calls
        if not response.tool_calls:
            break
            
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
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

    # 2. Enforce Structured Output using RepoNavigatorResponse
    structured_llm = llm.with_structured_output(RepoNavigatorResponse)
    
    final_prompt = (
        "Consolidate your findings from the search tool results above.\n"
        "Return the target packages, filesystem map, key search results, and exact relevant files."
    )
    
    decision: RepoNavigatorResponse = structured_llm.invoke(messages + [HumanMessage(content=final_prompt)])
    
    # 3. Update MonorepoState with the schema fields
    state["target_packages"] = decision.target_packages
    
    # Fallback to captured tool execution if LLM summary trimmed the raw tree
    state["filesystem_map"] = decision.filesystem_map or (
        captured_tree_outputs[-1] if captured_tree_outputs else ""
    )
    
    # Merge LLM-selected search results with parsed tool results
    state["search_results"] = decision.search_results or captured_search_results
    state["relevant_files"] = decision.relevant_files

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
    LangGraph Node: Analyzes relevant files and issue context, drafts a deterministic
    fix, generates precise search/replace diff blocks, and specifies verification tests.
    """
    config = state.get("config", {})
    llm = get_chat_model(config)
    
    # Step 1: Read the current content of relevant files from sandbox
    relevant_files = state.get("relevant_files", [])
    file_contents = fetch_file_contents(relevant_files)
    
    # Step 2: Assemble System Prompt with Strict Patching Guardrails
    system_prompt = (
        "You are an expert Monorepo Software Engineer and Planner.\n"
        "Your task is to analyze the issue and target files, formulate a step-by-step fix, "
        "and generate exact search-and-replace code diffs.\n\n"
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

    # Step 4: Call Model with Structured Output
    structured_llm = llm.with_structured_output(PlannerCoderResponse)
    
    response: PlannerCoderResponse = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    # Step 5: Update Graph State
    state["file_contents"] = file_contents
    state["proposed_plan"] = response.proposed_plan
    state["test_command"] = response.test_command
    state["diffs_to_apply"] = [diff.model_dump() for diff in response.diffs_to_apply]
    
    return state