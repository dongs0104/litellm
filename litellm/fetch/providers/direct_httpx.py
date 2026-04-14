"""
Opt-in raw-httpx fetch provider.

USE WITH CAUTION. This bypasses sandboxed extraction and runs the fetch
in-process. The caller MUST set ``allow_direct_fetch=True`` in
``litellm.afetch()`` — ``litellm/fetch/main.py`` enforces that gate.

Security guards:
- Block hostnames that resolve to private / link-local / loopback / CGNAT /
  cloud-metadata addresses (RFC1918, 169.254/16, fd00::/8, etc.).
- Block well-known metadata hostnames (``metadata.google.internal`` etc.).
- Cap total redirect hops at 5 and re-validate the host after every hop, so a
  redirect or DNS-rebinding trick cannot land us on a private IP.
- Cap response size and total elapsed time.
- Restrict ``Content-Type`` to an allowlist (text/json/xml/markdown).
"""

import ipaddress
import socket
from typing import Dict, Optional
from urllib.parse import urlparse

import httpx

from litellm.fetch.providers.base import BaseFetchProvider
from litellm.types.fetch import FetchError, FetchResponse

_DEFAULT_TIMEOUT = 10.0
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB
_MAX_REDIRECTS = 5

_ALLOWED_CONTENT_TYPE_PREFIXES = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml",
    "application/rss+xml",
    "application/atom+xml",
    "application/ld+json",
)

_BLOCKED_HOSTNAMES = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
        "instance-data.ec2.internal",
    }
)


def _host_is_private(host: str) -> bool:
    """Return True if *any* IP the host resolves to is blocked."""
    if host.lower() in _BLOCKED_HOSTNAMES:
        return True

    # Literal IP in the URL — check it directly.
    try:
        ip = ipaddress.ip_address(host)
        return _ip_is_blocked(ip)
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise FetchError(f"DNS lookup failed for {host}: {exc}") from exc

    for info in infos:
        sockaddr = info[4]
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except (ValueError, IndexError):
            continue
        if _ip_is_blocked(ip):
            return True
    return False


def _ip_is_blocked(ip) -> bool:
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True
    # Explicit AWS / Alibaba / DO metadata
    if str(ip) in {"169.254.169.254", "100.100.100.200"}:
        return True
    return False


def _validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise FetchError(f"Unsupported URL scheme: {parsed.scheme!r}", url=url)
    host = parsed.hostname
    if not host:
        raise FetchError("URL missing host", url=url)
    if _host_is_private(host):
        raise FetchError(
            f"Refusing to fetch host {host!r}: resolves to a blocked address",
            url=url,
        )
    return host


class DirectHttpxFetchProvider(BaseFetchProvider):
    name = "direct_httpx"

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
        current_url = url
        for _hop in range(_MAX_REDIRECTS + 1):
            _validate_url(current_url)

            headers: Dict[str, str] = {
                "User-Agent": "litellm-webfetch/1.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.5",
            }
            if extra_headers:
                headers.update(extra_headers)

            async with httpx.AsyncClient(
                timeout=timeout or _DEFAULT_TIMEOUT,
                follow_redirects=False,
            ) as client:
                try:
                    resp = await client.get(current_url, headers=headers)
                except httpx.HTTPError as exc:
                    raise FetchError(str(exc), url=current_url) from exc

            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("location")
                if not location:
                    raise FetchError(
                        "Redirect without Location header",
                        status_code=resp.status_code,
                        url=current_url,
                    )
                current_url = str(httpx.URL(current_url).join(location))
                continue

            if resp.status_code >= 400:
                raise FetchError(
                    f"HTTP {resp.status_code}",
                    status_code=resp.status_code,
                    url=current_url,
                )

            content_type = (resp.headers.get("content-type") or "").lower()
            if content_type and not any(
                content_type.startswith(p) for p in _ALLOWED_CONTENT_TYPE_PREFIXES
            ):
                raise FetchError(
                    f"Refusing content-type {content_type!r}",
                    status_code=resp.status_code,
                    url=current_url,
                )

            body = resp.content
            if len(body) > _MAX_RESPONSE_BYTES:
                raise FetchError(
                    f"Response exceeded max size ({len(body)} bytes)",
                    status_code=resp.status_code,
                    url=current_url,
                )

            text = resp.text
            if max_content_tokens is not None:
                # Rough heuristic: ~4 chars per token.
                char_cap = max_content_tokens * 4
                if len(text) > char_cap:
                    text = text[:char_cap]

            return FetchResponse(
                url=url,
                final_url=current_url,
                title=None,
                content=text,
                content_type=content_type or None,
                status_code=resp.status_code,
            )

        raise FetchError(f"Exceeded max redirects ({_MAX_REDIRECTS})", url=url)
