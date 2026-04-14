"""
WebFetch Interception Module

Provides server-side WebFetch tool execution for models that don't natively
support server-side tool calling (e.g., Bedrock/Claude, Vertex, OpenAI).

Handler is intentionally not re-exported here (it depends on afetch/fetch
providers which land in a later sub-issue). Import it from
`litellm.integrations.webfetch_interception.handler` once available.
"""

from litellm.integrations.webfetch_interception.tools import (
    get_litellm_web_fetch_tool,
    get_litellm_web_fetch_tool_openai,
    is_web_fetch_tool,
    is_web_fetch_tool_chat_completion,
)

__all__ = [
    "get_litellm_web_fetch_tool",
    "get_litellm_web_fetch_tool_openai",
    "is_web_fetch_tool",
    "is_web_fetch_tool_chat_completion",
]
