"""
Jina Reader fetch provider.

Jina Reader (https://r.jina.ai/<url>) renders a target page in a sandboxed
environment and returns model-ready markdown. This is the recommended default
because JS rendering + content extraction + isolation all live on their side.
"""

import os
from typing import Dict, Optional

import httpx

from litellm.fetch.providers.base import BaseFetchProvider
from litellm.types.fetch import FetchError, FetchResponse

_DEFAULT_API_BASE = "https://r.jina.ai"
_DEFAULT_TIMEOUT = 30.0


class JinaFetchProvider(BaseFetchProvider):
    name = "jina"

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
            api_base or os.environ.get("JINA_API_BASE") or _DEFAULT_API_BASE
        ).rstrip("/")
        key = api_key or os.environ.get("JINA_API_KEY")

        headers: Dict[str, str] = {"Accept": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if max_content_tokens is not None:
            # Jina accepts an X-Max-Tokens hint for server-side truncation.
            headers["X-Max-Tokens"] = str(max_content_tokens)
        if extra_headers:
            headers.update(extra_headers)

        request_url = f"{base}/{url}"

        async with httpx.AsyncClient(timeout=timeout or _DEFAULT_TIMEOUT) as client:
            try:
                resp = await client.get(request_url, headers=headers)
            except httpx.HTTPError as exc:
                raise FetchError(str(exc), url=url) from exc

        if resp.status_code >= 400:
            raise FetchError(
                f"Jina Reader returned {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
                url=url,
            )

        # Reader returns JSON when Accept: application/json is sent.
        try:
            payload = resp.json()
        except ValueError:
            payload = {"data": {"content": resp.text}}

        data = payload.get("data") or {}
        return FetchResponse(
            url=url,
            final_url=data.get("url") or url,
            title=data.get("title"),
            content=data.get("content") or data.get("text") or "",
            content_type="text/markdown",
            status_code=resp.status_code,
        )
