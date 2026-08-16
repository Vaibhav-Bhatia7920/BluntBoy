import litellm
from typing import Any, Dict, List

def get_chat_model(config: Dict[str, Any], messages: List[Dict[str, Any]], tools: List[Any] = None) -> Any:
    """
    Calls OpenAI's Responses API directly using LiteLLM, 
    automatically serializing LangChain StructuredTools into JSON schemas.
    """
    model_name = config.get("llm_model", "openai/gpt-5.6-luna")
    api_key = config.get("api_key") or config.get("api_token")
    reasoning_effort = config.get("reasoning_effort", "medium")
    
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if config.get("api_base"):
        kwargs["api_base"] = config.get("api_base")

    # Convert LangChain StructuredTools to OpenAI function JSON schemas if needed
    formatted_tools = None
    if tools:
        formatted_tools = []
        for tool in tools:
            # If it's a LangChain StructuredTool object, use its built-in schema generator
            if hasattr(tool, "args_schema") and hasattr(tool, "name") and hasattr(tool, "description"):
                schema = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.args_schema.schema() if tool.args_schema else {}
                    }
                }
                formatted_tools.append(schema)
            elif isinstance(tool, dict):
                # Already a dictionary schema
                formatted_tools.append(tool)

    # Call litellm.responses with JSON-serializable tools
    response = litellm.responses(
        model=model_name,
        input=messages,
        tools=formatted_tools,
        reasoning={
            "effort": reasoning_effort
        },
        **kwargs
    )
    
    return response