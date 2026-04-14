"""
Firecrawl ``/scrape`` fetch provider.

Firecrawl runs the fetch in a sandboxed headless browser and returns
model-ready markdown with optional JS rendering. API docs:
https://docs.firecrawl.dev/api-reference/endpoint/scrape
"""

import os
from typing import Any, Dict, Optional

import httpx

from litellm.fetch.providers.base import BaseFetchProvider
from litellm.types.fetch import FetchError, FetchResponse

_DEFAULT_API_BASE = "https://api.firecrawl.dev/v1"
_DEFAULT_TIMEOUT = 45.0


class FirecrawlFetchProvider(BaseFetchProvider):
    name = "firecrawl"

    async def fetch(
        self,
        *,
        url: str,
        max_content_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> FetchResponse:
        base = (
            api_base or os.environ.get("FIRECRAWL_API_BASE") or _DEFAULT_API_BASE
        ).rstrip("/")
        key = api_key or os.environ.get("FIRECRAWL_API_KEY")
        if not key:
            raise FetchError(
                "Firecrawl requires FIRECRAWL_API_KEY (or api_key=) to be set",
                url=url,
            )

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        body: Dict[str, Any] = {"url": url, "formats": ["markdown"]}
        if max_content_tokens is not None:
            # Firecrawl doesn't take a token cap directly; pass through as a
            # hint the client can use for later truncation.
            body["max_content_tokens"] = max_content_tokens

        async with httpx.AsyncClient(timeout=timeout or _DEFAULT_TIMEOUT) as client:
            try:
                resp = await client.post(f"{base}/scrape", headers=headers, json=body)
            except httpx.HTTPError as exc:
                raise FetchError(str(exc), url=url) from exc

        if resp.status_code >= 400:
            raise FetchError(
                f"Firecrawl returned {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
                url=url,
            )

        payload = resp.json()
        data = payload.get("data") or {}
        metadata = data.get("metadata") or {}
        return FetchResponse(
            url=url,
            final_url=metadata.get("sourceURL") or metadata.get("url") or url,
            title=metadata.get("title"),
            content=data.get("markdown") or data.get("content") or "",
            content_type="text/markdown",
            status_code=resp.status_code,
        )
