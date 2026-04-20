"""
WebFetch Interception Module

Provides server-side WebFetch tool execution for models that don't natively
support server-side tool calling (e.g., Bedrock/Claude, Vertex, OpenAI).
"""

from litellm.integrations.webfetch_interception.handler import (
    WebFetchInterceptionLogger,
)
from litellm.integrations.webfetch_interception.tools import (
    get_litellm_web_fetch_tool,
    get_litellm_web_fetch_tool_openai,
    is_web_fetch_tool,
    is_web_fetch_tool_chat_completion,
)

__all__ = [
    "WebFetchInterceptionLogger",
    "get_litellm_web_fetch_tool",
    "get_litellm_web_fetch_tool_openai",
    "is_web_fetch_tool",
    "is_web_fetch_tool_chat_completion",
]
