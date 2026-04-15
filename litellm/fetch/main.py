"""
``litellm.afetch()`` — sandboxed/managed URL fetching for WebFetch interception.

Security default: a fetch provider must be configured. No provider -> explicit
``ValueError`` (not a silent raw fetch). The raw-httpx provider is available
only when the caller explicitly asks for ``provider="direct_httpx"`` and
opts in.
"""

from typing import List, Optional
from urllib.parse import urlparse

from litellm._logging import verbose_logger
from litellm.fetch.providers.base import BaseFetchProvider
from litellm.fetch.providers.direct_httpx import DirectHttpxFetchProvider
from litellm.fetch.providers.firecrawl import FirecrawlFetchProvider
from litellm.fetch.providers.jina import JinaFetchProvider
from litellm.fetch.providers.tavily import TavilyFetchProvider
from litellm.types.fetch import FetchResponse

_PROVIDER_REGISTRY = {
    "jina": JinaFetchProvider,
    "firecrawl": FirecrawlFetchProvider,
    "tavily": TavilyFetchProvider,
    "direct_httpx": DirectHttpxFetchProvider,
}


def _resolve_provider(provider: Optional[str]) -> BaseFetchProvider:
    if not provider:
        raise ValueError(
            "litellm.afetch() requires an explicit `provider`. Configure a "
            "sandboxed fetch provider (jina, firecrawl, tavily) or opt in to "
            "`direct_httpx` with allow_direct_fetch=True."
        )
    try:
        provider_cls = _PROVIDER_REGISTRY[provider]
    except KeyError:
        raise ValueError(
            f"Unknown fetch provider: {provider!r}. "
            f"Valid providers: {sorted(_PROVIDER_REGISTRY)}"
        )
    return provider_cls()


async def afetch(
    url: str,
    *,
    provider: Optional[str] = None,
    max_content_tokens: Optional[int] = None,
    allowed_domains: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    allow_direct_fetch: bool = False,
    extra_headers: Optional[dict] = None,
) -> FetchResponse:
    """
    Fetch the contents of a URL and return a normalized :class:`FetchResponse`.

    Args:
        url: Absolute http/https URL to fetch.
        provider: Fetch provider identifier. One of ``jina``, ``firecrawl``,
            ``tavily``, or ``direct_httpx`` (opt-in only).
        max_content_tokens: Soft cap on returned content length (in tokens).
            Providers that support server-side truncation honor this; others
            fall back to a char-based approximation.
        allowed_domains: Optional allowlist of **exact** hostnames (no subdomain
            matching). If provided and the URL's host is not in the list, the
            fetch is refused. ``example.com`` does *not* match ``sub.example.com`` —
            list each subdomain explicitly if you need it.
        api_key / api_base / timeout / extra_headers: Provider connection
            options.
        allow_direct_fetch: Required to use ``provider="direct_httpx"``. Every
            other provider ignores it.
    """
    if provider == "direct_httpx" and not allow_direct_fetch:
        raise ValueError(
            "provider='direct_httpx' requires allow_direct_fetch=True. "
            "direct_httpx bypasses sandboxed extraction and has a larger "
            "SSRF surface — only enable it if you understand the risks."
        )

    if allowed_domains is not None:
        host = (urlparse(url).hostname or "").lower()
        if host not in {d.lower() for d in allowed_domains}:
            raise ValueError(f"URL host {host!r} is not in the allowed_domains list")

    verbose_logger.debug(f"litellm.afetch: provider={provider} url={url}")
    fetcher = _resolve_provider(provider)
    return await fetcher.fetch(
        url=url,
        max_content_tokens=max_content_tokens,
        api_key=api_key,
        api_base=api_base,
        timeout=timeout,
        extra_headers=extra_headers,
    )
