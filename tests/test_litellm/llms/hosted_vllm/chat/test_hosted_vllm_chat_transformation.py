import json
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

from litellm.llms.hosted_vllm.chat.transformation import (
    HostedVLLMChatConfig,
    HostedVLLMChatCompletionStreamingHandler,
)


def test_hosted_vllm_chat_transformation_file_url():
    config = HostedVLLMChatConfig()
    video_url = "https://example.com/video.mp4"
    video_data = f"data:video/mp4;base64,{video_url}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_data": video_data,
                    },
                }
            ],
        }
    ]
    transformed_response = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert transformed_response["messages"] == [
        {
            "role": "user",
            "content": [{"type": "video_url", "video_url": {"url": video_data}}],
        }
    ]


def test_hosted_vllm_chat_transformation_with_audio_url():
    from litellm import completion

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama-3.1-70b-instruct",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_response.text = json.dumps(mock_response.json.return_value)
    mock_client.post.return_value = mock_response

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
        return_value=mock_client,
    ):
        try:
            completion(
                model="hosted_vllm/llama-3.1-70b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {"url": "https://example.com/audio.mp3"},
                            },
                        ],
                    },
                ],
                api_base="https://test-vllm.example.com/v1",
            )
        except Exception:
            pass

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args[1]
        request_data = json.loads(call_kwargs["data"])
        assert request_data["messages"] == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": "https://example.com/audio.mp3"},
                    }
                ],
            }
        ]


def test_hosted_vllm_supports_reasoning_effort():
    config = HostedVLLMChatConfig()
    supported_params = config.get_supported_openai_params(model="hosted_vllm/gpt-oss-120b")
    assert "reasoning_effort" in supported_params
    optional_params = config.map_openai_params(
        non_default_params={"reasoning_effort": "high"},
        optional_params={},
        model="hosted_vllm/gpt-oss-120b",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "high"


def test_hosted_vllm_supports_thinking():
    """
    Test that hosted_vllm supports the 'thinking' parameter.

    Anthropic-style thinking is converted to OpenAI-style reasoning_effort
    since vLLM is OpenAI-compatible.

    Related issue: https://github.com/BerriAI/litellm/issues/19761
    """
    config = HostedVLLMChatConfig()
    supported_params = config.get_supported_openai_params(model="hosted_vllm/GLM-4.6-FP8")
    assert "thinking" in supported_params

    # Test thinking with low budget_tokens -> "minimal" (for < 2000)
    optional_params = config.map_openai_params(
        non_default_params={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert "thinking" not in optional_params  # thinking should NOT be passed
    assert optional_params["reasoning_effort"] == "minimal"

    # Test thinking with high budget_tokens -> "high"
    optional_params = config.map_openai_params(
        non_default_params={"thinking": {"type": "enabled", "budget_tokens": 15000}},
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "high"

    # Test that existing reasoning_effort is not overwritten
    optional_params = config.map_openai_params(
        non_default_params={
            "thinking": {"type": "enabled", "budget_tokens": 15000},
            "reasoning_effort": "low",
        },
        optional_params={},
        model="hosted_vllm/GLM-4.6-FP8",
        drop_params=False,
    )
    assert optional_params["reasoning_effort"] == "low"


def test_hosted_vllm_thinking_blocks_converted_to_reasoning_content():
    """
    Test that thinking_blocks on assistant messages are converted to
    reasoning_content field (OpenAI-compatible format for vLLM).

    Thinking blocks should NOT be added to the content array since vLLM
    only accepts standard OpenAI content types (text, image_url, etc.).
    """
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "user",
            "content": "Hello",
        },
        {
            "role": "assistant",
            "content": "Here is my answer.",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Let me reason about this...",
                    "signature": "abc123",
                }
            ],
        },
        {
            "role": "user",
            "content": "Follow up question",
        },
    ]
    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assistant_msg = transformed["messages"][1]
    assert assistant_msg["role"] == "assistant"
    # Content stays as-is (not modified by thinking_blocks)
    assert assistant_msg["content"] == "Here is my answer."
    # Thinking is extracted to reasoning_content
    assert assistant_msg.get("reasoning_content") == "Let me reason about this..."
    assert "thinking_blocks" not in assistant_msg


def test_hosted_vllm_thinking_blocks_with_list_content():
    """
    Test thinking_blocks converted to reasoning_content when assistant
    content is already a list. Content should remain unchanged.
    """
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Response text"}],
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Step 1 reasoning",
                    "signature": "sig1",
                },
                {
                    "type": "thinking",
                    "thinking": "Step 2 reasoning",
                    "signature": "sig2",
                },
            ],
        },
    ]
    transformed = config.transform_request(
        model="hosted_vllm/llama-3.1-70b-instruct",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assistant_msg = transformed["messages"][0]
    # Content stays as-is
    assert assistant_msg["content"] == [{"type": "text", "text": "Response text"}]
    # Thinking blocks are joined into reasoning_content
    assert assistant_msg.get("reasoning_content") == "Step 1 reasoning\nStep 2 reasoning"
    assert "thinking_blocks" not in assistant_msg


def test_hosted_vllm_redacted_thinking_blocks_dropped():
    """
    Redacted thinking blocks should be ignored — they contain opaque
    Anthropic-specific data with no value for vLLM.
    """
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "assistant",
            "content": "Answer.",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "Let me think step by step...",
                    "signature": "sig1",
                },
                {"type": "redacted_thinking", "data": "opaque_data"},
            ],
        },
    ]
    transformed = config.transform_request(
        model="hosted_vllm/deepseek-r1",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assistant_msg = transformed["messages"][0]
    assert assistant_msg.get("reasoning_content") == "Let me think step by step..."
    assert "thinking_blocks" not in assistant_msg


def test_hosted_vllm_existing_reasoning_content_not_overwritten():
    """
    If reasoning_content is already set on the message, it should not be
    overwritten by thinking_blocks extraction.
    """
    config = HostedVLLMChatConfig()
    messages = [
        {
            "role": "assistant",
            "content": "Answer.",
            "reasoning_content": "Original reasoning.",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "New thought.", "signature": "s1"},
            ],
        },
    ]
    transformed = config.transform_request(
        model="hosted_vllm/deepseek-r1",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    assistant_msg = transformed["messages"][0]
    assert assistant_msg.get("reasoning_content") == "Original reasoning."


def test_hosted_vllm_streaming_handler_promotes_reasoning_content_to_thinking_blocks():
    """
    The vLLM streaming handler should convert reasoning_content to
    thinking_blocks so the Anthropic pass-through adapter can detect
    the correct block type.
    """
    handler = HostedVLLMChatCompletionStreamingHandler(
        streaming_response=iter([]),
        sync_stream=True,
        json_mode=False,
    )
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "deepseek-r1",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "reasoning_content": "Let me think about this...",
                },
                "finish_reason": None,
            }
        ],
    }
    response = handler.chunk_parser(chunk)
    delta = response.choices[0].delta

    assert hasattr(delta, "thinking_blocks")
    assert delta.thinking_blocks is not None
    assert len(delta.thinking_blocks) == 1
    assert delta.thinking_blocks[0]["type"] == "thinking"
    assert delta.thinking_blocks[0]["thinking"] == "Let me think about this..."


def test_hosted_vllm_streaming_handler_no_thinking_blocks_without_reasoning():
    """
    When there's no reasoning_content in the chunk, thinking_blocks
    should not be added.
    """
    handler = HostedVLLMChatCompletionStreamingHandler(
        streaming_response=iter([]),
        sync_stream=True,
        json_mode=False,
    )
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "deepseek-r1",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "Hello!",
                },
                "finish_reason": None,
            }
        ],
    }
    response = handler.chunk_parser(chunk)
    delta = response.choices[0].delta

    assert not getattr(delta, "thinking_blocks", None)
    assert delta.content == "Hello!"


def test_hosted_vllm_streaming_handler_existing_thinking_blocks_not_overwritten():
    """
    If thinking_blocks already exist on the delta, reasoning_content
    should not overwrite them.
    """
    handler = HostedVLLMChatCompletionStreamingHandler(
        streaming_response=iter([]),
        sync_stream=True,
        json_mode=False,
    )
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "deepseek-r1",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "reasoning_content": "From reasoning_content field",
                    "thinking_blocks": [
                        {"type": "thinking", "thinking": "From thinking_blocks", "signature": ""},
                    ],
                },
                "finish_reason": None,
            }
        ],
    }
    response = handler.chunk_parser(chunk)
    delta = response.choices[0].delta

    assert delta.thinking_blocks is not None
    assert len(delta.thinking_blocks) == 1
    assert delta.thinking_blocks[0]["thinking"] == "From thinking_blocks"
