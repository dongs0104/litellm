"""
Tavily ``/extract`` fetch provider.

Tavily extract runs an isolated fetch + HTML->text extraction server-side.
Docs: https://docs.tavily.com/api-reference/endpoint/extract
"""

import os
from typing import Any, Dict, Optional

import httpx

from litellm.fetch.providers.base import BaseFetchProvider
from litellm.types.fetch import FetchError, FetchResponse

_DEFAULT_API_BASE = "https://api.tavily.com"
_DEFAULT_TIMEOUT = 45.0


class TavilyFetchProvider(BaseFetchProvider):
    name = "tavily"

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
            api_base or os.environ.get("TAVILY_API_BASE") or _DEFAULT_API_BASE
        ).rstrip("/")
        key = api_key or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise FetchError(
                "Tavily requires TAVILY_API_KEY (or api_key=) to be set", url=url
            )

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        body: Dict[str, Any] = {"api_key": key, "urls": [url]}
        if max_content_tokens is not None:
            body["max_content_tokens"] = max_content_tokens

        async with httpx.AsyncClient(timeout=timeout or _DEFAULT_TIMEOUT) as client:
            try:
                resp = await client.post(f"{base}/extract", headers=headers, json=body)
            except httpx.HTTPError as exc:
                raise FetchError(str(exc), url=url) from exc

        if resp.status_code >= 400:
            raise FetchError(
                f"Tavily returned {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
                url=url,
            )

        payload = resp.json()
        results = payload.get("results") or []
        if not results:
            failed = payload.get("failed_results") or []
            reason = failed[0].get("error") if failed else "empty results"
            raise FetchError(
                f"Tavily extract failed: {reason}",
                status_code=resp.status_code,
                url=url,
            )
        first = results[0]
        return FetchResponse(
            url=url,
            final_url=first.get("url") or url,
            title=None,  # Tavily extract does not return title
            content=first.get("raw_content") or first.get("content") or "",
            content_type="text/plain",
            status_code=resp.status_code,
        )
