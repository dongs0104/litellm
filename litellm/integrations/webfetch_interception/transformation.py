"""
WebFetch Tool Transformation

Transforms between Anthropic/OpenAI tool_use format and LiteLLM fetch format.

Mirrors the structure of ``websearch_interception.transformation`` so the
agentic-loop handler (next sub-issue) can reuse the same control flow.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from litellm._logging import verbose_logger
from litellm.constants import LITELLM_WEB_FETCH_TOOL_NAME

_RECOGNIZED_FETCH_TOOL_NAMES = (
    LITELLM_WEB_FETCH_TOOL_NAME,
    "WebFetch",
    "web_fetch",
)


class WebFetchTransformation:
    """
    Transformation class for WebFetch tool interception.

    Handles transformation between:
    - Anthropic ``tool_use`` blocks → LiteLLM fetch requests
    - OpenAI ``tool_calls`` → LiteLLM fetch requests
    - Fetch results → Anthropic/OpenAI ``tool_result`` format
    """

    @staticmethod
    def transform_request(
        response: Any,
        stream: bool,
        response_format: str = "anthropic",
    ) -> Tuple[bool, List[Dict]]:
        """
        Extract WebFetch tool calls from a model response.

        Returns ``(has_web_fetch, tool_calls)``. Each ``tool_call`` carries
        ``id``, ``name``, and ``input`` (parsed dict with at minimum ``url``).

        Streaming responses are expected to already have been downgraded to
        non-streaming by the handler's pre-request hook. If we're still called
        with ``stream=True`` we skip — same behavior as WebSearch.
        """
        if stream:
            verbose_logger.warning(
                "WebFetchInterception: Unexpected streaming response, "
                "skipping interception"
            )
            return False, []

        if response_format == "openai":
            return WebFetchTransformation._detect_from_openai_response(response)
        return WebFetchTransformation._detect_from_non_streaming_response(response)

    @staticmethod
    def _detect_from_non_streaming_response(
        response: Any,
    ) -> Tuple[bool, List[Dict]]:
        """Parse an Anthropic-shape response for WebFetch ``tool_use`` blocks."""
        if isinstance(response, dict):
            content = response.get("content", [])
        else:
            if not hasattr(response, "content"):
                return False, []
            content = response.content or []

        if not content:
            return False, []

        tool_calls: List[Dict] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                block_name = block.get("name")
                block_id = block.get("id")
                block_input = block.get("input", {})
            else:
                block_type = getattr(block, "type", None)
                block_name = getattr(block, "name", None)
                block_id = getattr(block, "id", None)
                block_input = getattr(block, "input", {})

            if block_type == "tool_use" and block_name in _RECOGNIZED_FETCH_TOOL_NAMES:
                tool_calls.append(
                    {
                        "id": block_id,
                        "type": "tool_use",
                        "name": block_name,
                        "input": block_input or {},
                    }
                )
                verbose_logger.debug(
                    f"WebFetchInterception: found {block_name} tool_use id={block_id}"
                )

        return len(tool_calls) > 0, tool_calls

    @staticmethod
    def _detect_from_openai_response(
        response: Any,
    ) -> Tuple[bool, List[Dict]]:
        """Parse an OpenAI-shape response for WebFetch ``tool_calls``."""
        if isinstance(response, dict):
            choices = response.get("choices", [])
        else:
            if not hasattr(response, "choices"):
                return False, []
            choices = response.choices or []

        if not choices:
            return False, []

        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message", {})
        else:
            message = getattr(first_choice, "message", None)

        if not message:
            return False, []

        if isinstance(message, dict):
            openai_tool_calls = message.get("tool_calls") or []
        else:
            openai_tool_calls = getattr(message, "tool_calls", None) or []

        if not openai_tool_calls:
            return False, []

        tool_calls: List[Dict] = []
        for tool_call in openai_tool_calls:
            if isinstance(tool_call, dict):
                tool_id = tool_call.get("id")
                tool_type = tool_call.get("type")
                function = tool_call.get("function", {}) or {}
                function_name = (
                    function.get("name")
                    if isinstance(function, dict)
                    else getattr(function, "name", None)
                )
                function_arguments = (
                    function.get("arguments")
                    if isinstance(function, dict)
                    else getattr(function, "arguments", None)
                )
            else:
                tool_id = getattr(tool_call, "id", None)
                tool_type = getattr(tool_call, "type", None)
                function = getattr(tool_call, "function", None)
                function_name = getattr(function, "name", None) if function else None
                function_arguments = (
                    getattr(function, "arguments", None) if function else None
                )

            if (
                tool_type == "function"
                and function_name in _RECOGNIZED_FETCH_TOOL_NAMES
            ):
                if isinstance(function_arguments, str):
                    try:
                        arguments = json.loads(function_arguments)
                    except json.JSONDecodeError:
                        verbose_logger.warning(
                            "WebFetchInterception: failed to parse function "
                            f"arguments: {function_arguments!r}"
                        )
                        arguments = {}
                else:
                    arguments = function_arguments or {}

                tool_calls.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "name": function_name,
                        "function": {
                            "name": function_name,
                            "arguments": arguments,
                        },
                        "input": arguments,
                    }
                )

        return len(tool_calls) > 0, tool_calls

    @staticmethod
    def extract_fetch_args(tool_call: Dict) -> Dict[str, Any]:
        """
        Pull the fetch arguments out of a tool_call in a provider-agnostic way.

        Returns a dict with at least ``url`` when the tool call is valid, or
        ``{}`` if the URL is missing. Callers should surface a structured error
        ``tool_result`` rather than raise in the latter case.
        """
        args = tool_call.get("input") or {}
        if not isinstance(args, dict):
            return {}

        url = args.get("url")
        if not isinstance(url, str) or not url:
            return {}

        result: Dict[str, Any] = {"url": url}
        if "max_content_tokens" in args:
            result["max_content_tokens"] = args["max_content_tokens"]
        if "allowed_domains" in args:
            result["allowed_domains"] = args["allowed_domains"]
        return result

    @staticmethod
    def transform_response(
        tool_calls: List[Dict],
        fetch_results: List[Union[str, Dict, Any]],
        response_format: str = "anthropic",
        thinking_blocks: Optional[List[Dict]] = None,
    ) -> Tuple[Dict, Union[Dict, List[Dict]]]:
        """
        Build the follow-up assistant + user/tool messages for the agentic loop.

        ``fetch_results`` may be a pre-formatted string, a dict with
        ``title`` / ``url`` / ``content`` keys, or any object exposing those
        attributes (see :func:`format_fetch_response`).
        """
        formatted = [
            r if isinstance(r, str) else WebFetchTransformation.format_fetch_response(r)
            for r in fetch_results
        ]

        if response_format == "openai":
            return WebFetchTransformation._transform_response_openai(
                tool_calls, formatted
            )
        return WebFetchTransformation._transform_response_anthropic(
            tool_calls, formatted, thinking_blocks=thinking_blocks
        )

    @staticmethod
    def _transform_response_anthropic(
        tool_calls: List[Dict],
        fetch_results: List[str],
        thinking_blocks: Optional[List[Dict]] = None,
    ) -> Tuple[Dict, Dict]:
        assistant_content: List[Dict] = []
        if thinking_blocks:
            assistant_content.extend(thinking_blocks)

        assistant_content.extend(
            {
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["name"],
                "input": tc["input"],
            }
            for tc in tool_calls
        )

        assistant_message = {"role": "assistant", "content": assistant_content}
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_calls[i]["id"],
                    "content": fetch_results[i],
                }
                for i in range(len(tool_calls))
            ],
        }
        return assistant_message, user_message

    @staticmethod
    def _transform_response_openai(
        tool_calls: List[Dict],
        fetch_results: List[str],
    ) -> Tuple[Dict, List[Dict]]:
        assistant_message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": (
                            json.dumps(tc["input"])
                            if isinstance(tc["input"], dict)
                            else str(tc["input"])
                        ),
                    },
                }
                for tc in tool_calls
            ],
        }
        tool_messages = [
            {
                "role": "tool",
                "tool_call_id": tool_calls[i]["id"],
                "content": fetch_results[i],
            }
            for i in range(len(tool_calls))
        ]
        return assistant_message, tool_messages

    @staticmethod
    def format_fetch_response(result: Any) -> str:
        """
        Format a fetch result as text for a ``tool_result`` block.

        Accepts:
        - ``str`` (returned as-is)
        - ``dict`` with ``title`` / ``url`` / ``final_url`` / ``content`` /
          ``status_code`` / ``error`` keys
        - Any object exposing those attributes

        Error shape (produced by the handler when the provider fails or the
        URL is invalid) is preserved verbatim so the model sees the reason.
        """
        if isinstance(result, str):
            return result

        def _get(key: str, default: Any = None) -> Any:
            if isinstance(result, dict):
                return result.get(key, default)
            return getattr(result, key, default)

        error = _get("error")
        if error:
            status_code = _get("status_code")
            status_suffix = f" (status={status_code})" if status_code else ""
            return f"Error fetching URL{status_suffix}: {error}"

        title = _get("title") or ""
        url = _get("final_url") or _get("url") or ""
        content = _get("content") or ""
        if not content and not url and not title:
            return str(result)

        header_parts = []
        if title:
            header_parts.append(f"Title: {title}")
        if url:
            header_parts.append(f"URL: {url}")
        header = "\n".join(header_parts)
        return f"{header}\n\n{content}" if header else content

    @staticmethod
    def build_error_tool_result(tool_call: Dict, error: str) -> str:
        """
        Convenience wrapper used by callers that want a uniform error shape
        without constructing a dict each time.
        """
        return WebFetchTransformation.format_fetch_response({"error": error})
