import litellm
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool


def format_tool_for_completion(tool: Any) -> Dict[str, Any]:
    """
    Standardizes tools (LangChain BaseTool/StructuredTool or dicts)
    into the OpenAI Chat Completion tool schema.
    """
    if isinstance(tool, BaseTool):
        return convert_to_openai_tool(tool)
    elif isinstance(tool, dict):
        if tool.get("type") == "function" and "function" in tool:
            return tool
        if "name" in tool and "parameters" in tool:
            return {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                }
            }
        return tool
    raise ValueError(f"Unsupported tool format: {tool}")


def get_chat_model(
    config: Dict[str, Any], 
    messages: List[Dict[str, Any]], 
    tools: Optional[List[Any]] = None,
    **extra_kwargs
) -> Any:
    """
    Invokes LiteLLM Chat Completions (litellm.completion) with support for
    reasoning effort, structured tools, and multi-turn message histories.
    """
    model_name = config.get("llm_model", "gpt-4o")
    api_key = config.get("api_key") or config.get("api_token")
    api_base = config.get("api_base")
    reasoning_effort = config.get("reasoning_effort")

    call_kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        **extra_kwargs
    }

    if api_key:
        call_kwargs["api_key"] = api_key
    if api_base:
        call_kwargs["api_base"] = api_base
    if reasoning_effort:
        call_kwargs["reasoning_effort"] = reasoning_effort

    # Format tools into OpenAI standard format
    if tools:
        call_kwargs["tools"] = [format_tool_for_completion(t) for t in tools]

    # Standard Chat Completion call
    response = litellm.completion(**call_kwargs)
    
    return response