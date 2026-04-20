"""Tests for WebFetchInterceptionLogger.async_pre_request_hook."""

import pytest

from litellm.constants import LITELLM_WEB_FETCH_TOOL_NAME
from litellm.integrations.webfetch_interception.handler import (
    WebFetchInterceptionLogger,
)
from litellm.types.utils import LlmProviders


def _kwargs(tools, provider="bedrock"):
    return {
        "tools": tools,
        "litellm_params": {"custom_llm_provider": provider},
    }


@pytest.mark.asyncio
async def test_converts_native_web_fetch_tool():
    logger = WebFetchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])
    kwargs = _kwargs([{"type": "web_fetch_20250910", "name": "web_fetch"}])

    result = await logger.async_pre_request_hook("model", [], kwargs)

    assert result is not None
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == LITELLM_WEB_FETCH_TOOL_NAME


@pytest.mark.asyncio
async def test_converts_claude_code_and_legacy():
    logger = WebFetchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])

    for tool in (
        {"name": "web_fetch", "type": "custom"},
        {"name": "WebFetch"},
    ):
        kwargs = _kwargs([tool])
        result = await logger.async_pre_request_hook("model", [], kwargs)
        assert result is not None
        assert result["tools"][0]["name"] == LITELLM_WEB_FETCH_TOOL_NAME


@pytest.mark.asyncio
async def test_preserves_non_fetch_tools_in_order():
    logger = WebFetchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])
    kwargs = _kwargs(
        [
            {"name": "calculator"},
            {"type": "web_fetch_20250910", "name": "web_fetch"},
            {"name": "weather"},
        ]
    )

    result = await logger.async_pre_request_hook("model", [], kwargs)

    assert [t.get("name") for t in result["tools"]] == [
        "calculator",
        LITELLM_WEB_FETCH_TOOL_NAME,
        "weather",
    ]


@pytest.mark.asyncio
async def test_dedupes_multiple_web_fetch_tools():
    logger = WebFetchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])
    kwargs = _kwargs(
        [
            {"type": "web_fetch_20250910", "name": "web_fetch"},
            {"name": "WebFetch"},
            {"name": "calculator"},
        ]
    )

    result = await logger.async_pre_request_hook("model", [], kwargs)

    names = [t.get("name") for t in result["tools"]]
    assert names.count(LITELLM_WEB_FETCH_TOOL_NAME) == 1
    assert "calculator" in names


@pytest.mark.asyncio
async def test_skips_when_provider_not_enabled():
    logger = WebFetchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])
    kwargs = _kwargs(
        [{"type": "web_fetch_20250910", "name": "web_fetch"}], provider="anthropic"
    )

    result = await logger.async_pre_request_hook("model", [], kwargs)

    assert result is None


@pytest.mark.asyncio
async def test_skips_when_no_tools():
    logger = WebFetchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])
    kwargs = _kwargs(None)

    result = await logger.async_pre_request_hook("model", [], kwargs)

    assert result is None


@pytest.mark.asyncio
async def test_skips_when_no_web_fetch_tool_present():
    logger = WebFetchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])
    kwargs = _kwargs(
        [{"name": "calculator"}, {"type": "web_search_20250305", "name": "web_search"}]
    )

    result = await logger.async_pre_request_hook("model", [], kwargs)

    assert result is None


@pytest.mark.asyncio
async def test_default_enabled_providers_is_bedrock_only():
    logger = WebFetchInterceptionLogger()
    assert logger.enabled_providers == [LlmProviders.BEDROCK.value]


@pytest.mark.asyncio
async def test_does_not_touch_websearch_tools():
    logger = WebFetchInterceptionLogger(enabled_providers=[LlmProviders.BEDROCK])
    kwargs = _kwargs(
        [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "web_fetch_20250910", "name": "web_fetch"},
        ]
    )

    result = await logger.async_pre_request_hook("model", [], kwargs)

    names_types = [(t.get("name"), t.get("type")) for t in result["tools"]]
    assert ("web_search", "web_search_20250305") in names_types
    assert any(n == LITELLM_WEB_FETCH_TOOL_NAME for n, _ in names_types)
