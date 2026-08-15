import os
from typing import Dict, Any
from langchain_community.chat_models import ChatLiteLLM

def get_chat_model(config: Dict[str, Any]) -> ChatLiteLLM:
    """
    Instantiates a ChatLiteLLM instance based on dynamic user runtime flags.
    Supports: ollama/qwen2.5-coder, gpt-4o, claude-3-5-sonnet, groq/llama-3.1-70b
    """
    model_name = config.get("llm_model", "ollama/qwen2.5-coder")
    api_base = config.get("api_base", None)
    
    return ChatLiteLLM(
        model=model_name,
        api_base=api_base,
        temperature=0.0,  # Deterministic output for tool navigation
        max_tokens=2000
    )