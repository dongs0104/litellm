"""
LiteLLM Web Fetch Tool Definition

This module defines the standard web fetch tool used across LiteLLM.
Native provider tools (like Anthropic's web_fetch_20250910) are converted
to this format for consistent interception and execution.
"""

from typing import Any, Dict

from litellm.constants import LITELLM_WEB_FETCH_TOOL_NAME

_DESCRIPTION = (
    "Fetch the contents of a URL. Use this when you need to read a specific "
    "web page, document, or article to answer the user's question."
)

_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The absolute URL to fetch (http or https).",
        },
        "max_content_tokens": {
            "type": "integer",
            "description": (
                "Optional soft cap on returned content length, measured in "
                "model tokens. The provider may truncate to honor this."
            ),
        },
        "allowed_domains": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional allowlist of domains. If set, a fetch to a domain "
                "not in this list must be refused."
            ),
        },
    },
    "required": ["url"],
}


def get_litellm_web_fetch_tool() -> Dict[str, Any]:
    """
    Get the standard LiteLLM web fetch tool definition (Anthropic style).

    Native fetch tools (Anthropic's ``web_fetch_20250910``, Claude Code's
    ``web_fetch``, etc.) are converted to this format for interception.
    """
    return {
        "name": LITELLM_WEB_FETCH_TOOL_NAME,
        "description": _DESCRIPTION,
        "input_schema": _INPUT_SCHEMA,
    }


def get_litellm_web_fetch_tool_openai() -> Dict[str, Any]:
    """
    Get the standard LiteLLM web fetch tool definition (OpenAI style).

    Used by the chat-completions path where tools must be in the
    ``{"type": "function", "function": {...}}`` shape.
    """
    return {
        "type": "function",
        "function": {
            "name": LITELLM_WEB_FETCH_TOOL_NAME,
            "description": _DESCRIPTION,
            "parameters": _INPUT_SCHEMA,
        },
    }


def is_web_fetch_tool_chat_completion(tool: Dict[str, Any]) -> bool:
    """
    Strict check for the LiteLLM web fetch tool on the Chat Completions API.

    Only matches the exact ``litellm_web_fetch`` name to avoid false positives
    with user-defined tools. Use the non-strict :func:`is_web_fetch_tool` when
    inspecting inbound Anthropic-style tools that need rewriting.
    """
    tool_type = tool.get("type", "")

    if tool_type == "function" and "function" in tool:
        function_def = tool.get("function", {}) or {}
        if function_def.get("name") == LITELLM_WEB_FETCH_TOOL_NAME:
            return True

    if tool.get("name") == LITELLM_WEB_FETCH_TOOL_NAME:
        return True

    return False


def is_web_fetch_tool(tool: Dict[str, Any]) -> bool:
    """
    Detect any form of a "web fetch" tool.

    Matches:
    - LiteLLM standard (Anthropic shape): ``name == "litellm_web_fetch"``
    - LiteLLM standard (OpenAI shape): ``type == "function"`` with
      ``function.name == "litellm_web_fetch"``
    - Anthropic native: ``type`` starting with ``"web_fetch_"``
      (covers ``web_fetch_20250910`` and any future-dated variants)
    - Claude Code: ``name == "web_fetch"`` paired with a ``type`` field
    - Legacy: ``name == "WebFetch"``
    """
    tool_name = tool.get("name", "")
    tool_type = tool.get("type", "")

    if tool_type == "function" and "function" in tool:
        function_def = tool.get("function", {}) or {}
        if function_def.get("name") == LITELLM_WEB_FETCH_TOOL_NAME:
            return True

    if tool_name == LITELLM_WEB_FETCH_TOOL_NAME:
        return True

    if isinstance(tool_type, str) and tool_type.startswith("web_fetch_"):
        return True

    if tool_name == "web_fetch" and tool_type:
        return True

    if tool_name == "WebFetch":
        return True

    return False
