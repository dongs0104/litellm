"""
WebFetch Interception Handler

CustomLogger that rewrites native Anthropic ``web_fetch_*`` tools to the
LiteLLM-standard ``litellm_web_fetch`` tool *before* the request reaches the
backend. This prevents backends (Bedrock / Vertex / OpenAI) from trying to
execute a server-side tool they don't support.

Agentic-loop execution (detecting ``tool_use`` in the response and running
``litellm.afetch()``) lands in a follow-up sub-issue — this module only wires
the pre-request conversion today.
"""

from typing import Dict, List, Optional, Union

from litellm._logging import verbose_logger
from litellm.constants import LITELLM_WEB_FETCH_TOOL_NAME
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.webfetch_interception.tools import (
    get_litellm_web_fetch_tool,
    is_web_fetch_tool,
)
from litellm.types.utils import LlmProviders


class WebFetchInterceptionLogger(CustomLogger):
    """
    CustomLogger that rewrites native web fetch tools to the LiteLLM standard.

    The agentic-loop hooks (``async_should_run_agentic_loop`` /
    ``async_run_agentic_loop``) are added in a later sub-issue. This class
    currently implements only ``async_pre_request_hook`` so that backends
    never receive a native ``web_fetch_*`` tool they cannot execute.
    """

    def __init__(
        self,
        enabled_providers: Optional[List[Union[LlmProviders, str]]] = None,
        fetch_tool_name: Optional[str] = None,
    ):
        """
        Args:
            enabled_providers: Providers to intercept for. ``None`` defaults to
                ``[LlmProviders.BEDROCK]`` to match the WebSearch default and
                avoid rewriting tools destined for Anthropic itself (which
                can execute ``web_fetch_*`` natively).
            fetch_tool_name: Reserved for the agentic-loop phase — selects
                which entry in ``llm_router.fetch_tools`` to execute. Not yet
                consumed.
        """
        super().__init__()
        if enabled_providers is None:
            self.enabled_providers: List[str] = [LlmProviders.BEDROCK.value]
        else:
            self.enabled_providers = [
                p.value if isinstance(p, LlmProviders) else p for p in enabled_providers
            ]
        self.fetch_tool_name = fetch_tool_name

    async def async_pre_request_hook(
        self, model: str, messages: List[Dict], kwargs: Dict
    ) -> Optional[Dict]:
        """
        Rewrite any native ``web_fetch_*`` / Claude Code ``web_fetch`` / legacy
        ``WebFetch`` tool to the LiteLLM standard ``litellm_web_fetch`` tool.

        Mirrors the shape of ``WebSearchInterceptionLogger.async_pre_request_hook``:
        returns a modified ``kwargs`` dict, or ``None`` when nothing needs to
        change.
        """
        custom_llm_provider = kwargs.get("litellm_params", {}).get(
            "custom_llm_provider", ""
        )

        if (
            self.enabled_providers is not None
            and custom_llm_provider not in self.enabled_providers
        ):
            return None

        tools = kwargs.get("tools")
        if not tools:
            return None

        if not any(is_web_fetch_tool(t) for t in tools):
            return None

        converted_tools: List[Dict] = []
        standard_tool_seen = False
        for tool in tools:
            if is_web_fetch_tool(tool):
                # Collapse multiple web_fetch tools into a single standard tool.
                if standard_tool_seen:
                    verbose_logger.debug(
                        "WebFetchInterception: dropping duplicate web_fetch tool "
                        f"{tool.get('name') or tool.get('type')}"
                    )
                    continue
                converted_tools.append(get_litellm_web_fetch_tool())
                standard_tool_seen = True
                verbose_logger.debug(
                    "WebFetchInterception: converted "
                    f"name={tool.get('name', 'none')} type={tool.get('type', 'none')} "
                    f"-> {LITELLM_WEB_FETCH_TOOL_NAME}"
                )
            else:
                converted_tools.append(tool)

        kwargs["tools"] = converted_tools
        return kwargs
