"""Tests for litellm.integrations.webfetch_interception.tools."""

import pytest

from litellm.constants import LITELLM_WEB_FETCH_TOOL_NAME
from litellm.integrations.webfetch_interception.tools import (
    get_litellm_web_fetch_tool,
    get_litellm_web_fetch_tool_openai,
    is_web_fetch_tool,
    is_web_fetch_tool_chat_completion,
)


class TestLitellmWebFetchToolDefinition:
    def test_anthropic_shape(self):
        tool = get_litellm_web_fetch_tool()
        assert tool["name"] == LITELLM_WEB_FETCH_TOOL_NAME
        assert "input_schema" in tool
        props = tool["input_schema"]["properties"]
        assert props["url"]["type"] == "string"
        assert props["max_content_tokens"]["type"] == "integer"
        assert props["allowed_domains"]["type"] == "array"
        assert tool["input_schema"]["required"] == ["url"]

    def test_openai_shape(self):
        tool = get_litellm_web_fetch_tool_openai()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == LITELLM_WEB_FETCH_TOOL_NAME
        assert tool["function"]["parameters"]["required"] == ["url"]


class TestIsWebFetchTool:
    @pytest.mark.parametrize(
        "tool",
        [
            {"name": LITELLM_WEB_FETCH_TOOL_NAME},
            {
                "type": "function",
                "function": {"name": LITELLM_WEB_FETCH_TOOL_NAME},
            },
            {"type": "web_fetch_20250910", "name": "web_fetch"},
            {"type": "web_fetch_20260101", "name": "web_fetch"},  # future-dated
            {"type": "web_fetch_20250910", "name": "anything_else"},  # type only
            {"type": "web_fetch_20250910"},  # type only, no name
            {"name": "web_fetch", "type": "custom"},  # Claude Code shape
            {"name": "WebFetch"},  # legacy
        ],
    )
    def test_positive_cases(self, tool):
        assert is_web_fetch_tool(tool) is True

    @pytest.mark.parametrize(
        "tool",
        [
            {"name": "calculator"},
            {"name": "web_search"},
            {"name": "litellm_web_search"},
            {"name": "web_fetch"},  # missing type field — not Claude Code
            {"type": "function", "function": {"name": "calculator"}},
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "webfetch_20250910"},  # wrong underscore placement
            {},
        ],
    )
    def test_negative_cases(self, tool):
        assert is_web_fetch_tool(tool) is False


class TestIsWebFetchToolChatCompletion:
    @pytest.mark.parametrize(
        "tool",
        [
            {"name": LITELLM_WEB_FETCH_TOOL_NAME},
            {
                "type": "function",
                "function": {"name": LITELLM_WEB_FETCH_TOOL_NAME},
            },
        ],
    )
    def test_positive_cases(self, tool):
        assert is_web_fetch_tool_chat_completion(tool) is True

    @pytest.mark.parametrize(
        "tool",
        [
            # strict — native Anthropic / Claude Code / legacy must NOT match
            {"type": "web_fetch_20250910", "name": "web_fetch"},
            {"name": "web_fetch", "type": "custom"},
            {"name": "WebFetch"},
            {"name": "calculator"},
            {"type": "function", "function": {"name": "calculator"}},
        ],
    )
    def test_negative_cases(self, tool):
        assert is_web_fetch_tool_chat_completion(tool) is False
