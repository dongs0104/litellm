"""Tests for litellm.integrations.webfetch_interception.transformation."""

import json

from litellm.constants import LITELLM_WEB_FETCH_TOOL_NAME
from litellm.integrations.webfetch_interception.transformation import (
    WebFetchTransformation,
)


def _anthropic_tool_use(name=LITELLM_WEB_FETCH_TOOL_NAME, url="https://example.com/a"):
    return {
        "content": [
            {"type": "text", "text": "Let me fetch that."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": name,
                "input": {"url": url, "max_content_tokens": 2048},
            },
        ]
    }


class TestDetectAnthropic:
    def test_detects_standard_tool(self):
        has, calls = WebFetchTransformation.transform_request(
            _anthropic_tool_use(), stream=False
        )
        assert has is True
        assert len(calls) == 1
        assert calls[0]["name"] == LITELLM_WEB_FETCH_TOOL_NAME
        assert calls[0]["input"]["url"] == "https://example.com/a"

    def test_detects_legacy_names(self):
        for name in ("WebFetch", "web_fetch"):
            has, calls = WebFetchTransformation.transform_request(
                _anthropic_tool_use(name=name), stream=False
            )
            assert has is True
            assert calls[0]["name"] == name

    def test_ignores_non_fetch_tool_use(self):
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "x",
                    "name": "calculator",
                    "input": {"a": 1},
                }
            ]
        }
        has, calls = WebFetchTransformation.transform_request(response, stream=False)
        assert has is False
        assert calls == []

    def test_empty_content(self):
        has, calls = WebFetchTransformation.transform_request(
            {"content": []}, stream=False
        )
        assert has is False
        assert calls == []

    def test_skips_when_streaming(self):
        has, calls = WebFetchTransformation.transform_request(
            _anthropic_tool_use(), stream=True
        )
        assert has is False
        assert calls == []


class TestDetectOpenAI:
    def test_detects_function_call(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": LITELLM_WEB_FETCH_TOOL_NAME,
                                    "arguments": json.dumps(
                                        {"url": "https://example.com/b"}
                                    ),
                                },
                            }
                        ]
                    }
                }
            ]
        }
        has, calls = WebFetchTransformation.transform_request(
            response, stream=False, response_format="openai"
        )
        assert has is True
        assert calls[0]["input"] == {"url": "https://example.com/b"}
        assert calls[0]["function"]["name"] == LITELLM_WEB_FETCH_TOOL_NAME

    def test_invalid_json_arguments_becomes_empty(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": LITELLM_WEB_FETCH_TOOL_NAME,
                                    "arguments": "{not json",
                                },
                            }
                        ]
                    }
                }
            ]
        }
        has, calls = WebFetchTransformation.transform_request(
            response, stream=False, response_format="openai"
        )
        assert has is True
        assert calls[0]["input"] == {}


class TestExtractFetchArgs:
    def test_valid_url(self):
        args = WebFetchTransformation.extract_fetch_args(
            {"input": {"url": "https://example.com", "max_content_tokens": 512}}
        )
        assert args == {"url": "https://example.com", "max_content_tokens": 512}

    def test_passes_allowed_domains(self):
        args = WebFetchTransformation.extract_fetch_args(
            {
                "input": {
                    "url": "https://example.com",
                    "allowed_domains": ["example.com"],
                }
            }
        )
        assert args["allowed_domains"] == ["example.com"]

    def test_missing_url_returns_empty(self):
        assert WebFetchTransformation.extract_fetch_args({"input": {}}) == {}
        assert WebFetchTransformation.extract_fetch_args({"input": {"url": ""}}) == {}
        assert WebFetchTransformation.extract_fetch_args({"input": {"url": 123}}) == {}

    def test_non_dict_input_returns_empty(self):
        assert WebFetchTransformation.extract_fetch_args({"input": "bogus"}) == {}


class TestTransformResponseAnthropic:
    def _calls(self):
        return [
            {
                "id": "toolu_1",
                "name": LITELLM_WEB_FETCH_TOOL_NAME,
                "input": {"url": "https://example.com/a"},
            }
        ]

    def test_builds_assistant_and_user_messages(self):
        assistant, user = WebFetchTransformation.transform_response(
            self._calls(),
            fetch_results=["Title: Example\nURL: https://example.com/a\n\nbody"],
        )
        assert assistant["role"] == "assistant"
        assert assistant["content"][0]["type"] == "tool_use"
        assert user["role"] == "user"
        assert user["content"][0]["type"] == "tool_result"
        assert user["content"][0]["tool_use_id"] == "toolu_1"
        assert "body" in user["content"][0]["content"]

    def test_prepends_thinking_blocks(self):
        thinking = [{"type": "thinking", "thinking": "hmm"}]
        assistant, _ = WebFetchTransformation.transform_response(
            self._calls(),
            fetch_results=["ok"],
            thinking_blocks=thinking,
        )
        assert assistant["content"][0]["type"] == "thinking"
        assert assistant["content"][1]["type"] == "tool_use"

    def test_formats_dict_result(self):
        _, user = WebFetchTransformation.transform_response(
            self._calls(),
            fetch_results=[
                {
                    "title": "Example",
                    "url": "https://example.com/a",
                    "content": "hello world",
                }
            ],
        )
        text = user["content"][0]["content"]
        assert "Title: Example" in text
        assert "URL: https://example.com/a" in text
        assert "hello world" in text

    def test_formats_error_result(self):
        _, user = WebFetchTransformation.transform_response(
            self._calls(),
            fetch_results=[{"error": "timeout", "status_code": 504}],
        )
        text = user["content"][0]["content"]
        assert "Error fetching URL" in text
        assert "504" in text
        assert "timeout" in text

    def test_prefers_final_url_over_requested_url(self):
        _, user = WebFetchTransformation.transform_response(
            self._calls(),
            fetch_results=[
                {
                    "title": "T",
                    "url": "https://example.com/a",
                    "final_url": "https://example.com/a/after-redirect",
                    "content": "c",
                }
            ],
        )
        assert "after-redirect" in user["content"][0]["content"]


class TestTransformResponseOpenAI:
    def test_emits_tool_calls_and_tool_messages(self):
        tool_calls = [
            {
                "id": "call_1",
                "name": LITELLM_WEB_FETCH_TOOL_NAME,
                "input": {"url": "https://example.com/a"},
            }
        ]
        assistant, tool_messages = WebFetchTransformation.transform_response(
            tool_calls,
            fetch_results=["ok"],
            response_format="openai",
        )
        assert assistant["role"] == "assistant"
        assert (
            assistant["tool_calls"][0]["function"]["name"]
            == LITELLM_WEB_FETCH_TOOL_NAME
        )
        parsed = json.loads(assistant["tool_calls"][0]["function"]["arguments"])
        assert parsed == {"url": "https://example.com/a"}
        assert tool_messages[0]["role"] == "tool"
        assert tool_messages[0]["tool_call_id"] == "call_1"
        assert tool_messages[0]["content"] == "ok"
