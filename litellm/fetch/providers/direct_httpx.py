"""
Opt-in raw-httpx fetch provider.

USE WITH CAUTION. This bypasses sandboxed extraction and runs the fetch
in-process. The caller MUST set ``allow_direct_fetch=True`` in
``litellm.afetch()`` — ``litellm/fetch/main.py`` enforces that gate.

Security guards:
- Block hostnames that resolve to private / link-local / loopback / CGNAT /
  cloud-metadata addresses (RFC1918, 169.254/16, fd00::/8, etc.).
- Block well-known metadata hostnames (``metadata.google.internal`` etc.).
- Pin the resolved IP into the connect URL (with ``Host`` header + ``sni_hostname``
  extension) so a DNS rebinding trick cannot return a private IP between the
  validation lookup and the actual connect.
- Cap total redirect hops at 5 and re-validate the host after every hop.
- Stream the response body with a hard size cap — no ``resp.content`` buffering.
- Restrict ``Content-Type`` to an allowlist (text/json/xml/markdown).
- Reuse a module-level ``AsyncClient`` so high-concurrency callers don't
  exhaust sockets.
"""

import asyncio
import ipaddress
import socket
from typing import Dict, Optional, Tuple, Union
from urllib.parse import urlparse, urlunparse

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

_shared_client: Optional[httpx.AsyncClient] = None
_shared_client_lock = asyncio.Lock()


async def _get_client(timeout: float) -> httpx.AsyncClient:
    """Return a module-level ``AsyncClient`` so repeated fetches reuse the pool."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        async with _shared_client_lock:
            if _shared_client is None or _shared_client.is_closed:
                _shared_client = httpx.AsyncClient(
                    timeout=timeout,
                    follow_redirects=False,
                    limits=httpx.Limits(
                        max_connections=50, max_keepalive_connections=20
                    ),
                )
    return _shared_client


def _ip_is_blocked(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> bool:
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True
    if str(ip) in {"169.254.169.254", "100.100.100.200"}:
        return True
    return False


def _resolve_and_validate(host: str) -> str:
    """Return a safe IP for ``host`` or raise ``FetchError``.

    Every IP the host resolves to must pass ``_ip_is_blocked``. The first
    IP is returned for pinning into the connect URL (defeats DNS rebinding).
    """
    if host.lower() in _BLOCKED_HOSTNAMES:
        raise FetchError(f"Refusing to fetch blocked hostname {host!r}")

    try:
        literal_ip = ipaddress.ip_address(host)
        if _ip_is_blocked(literal_ip):
            raise FetchError(f"Refusing literal blocked IP {host!r}")
        return str(literal_ip)
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise FetchError(f"DNS lookup failed for {host}: {exc}") from exc

    resolved_ips = []
    for info in infos:
        sockaddr = info[4]
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except (ValueError, IndexError):
            continue
        if _ip_is_blocked(ip):
            raise FetchError(
                f"Refusing to fetch host {host!r}: resolves to blocked {ip}"
            )
        resolved_ips.append(ip)

    if not resolved_ips:
        raise FetchError(f"No usable address for host {host!r}")
    return str(resolved_ips[0])


def _pin_url_to_ip(url: str) -> Tuple[str, str, Dict[str, str], Dict[str, str]]:
    """Return ``(connect_url, host, extra_headers, extensions)`` for the request.

    Validates scheme/host, resolves the host to a safe IP, and returns a URL
    with that IP substituted in — the actual HTTP connect happens against the
    pinned IP, while ``Host`` header and ``sni_hostname`` preserve TLS cert
    validation and vhost routing against the original hostname.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise FetchError(f"Unsupported URL scheme: {parsed.scheme!r}", url=url)
    host = parsed.hostname
    if not host:
        raise FetchError("URL missing host", url=url)

    ip = _resolve_and_validate(host)

    if ":" in ip:
        ip_netloc = f"[{ip}]"
    else:
        ip_netloc = ip
    port = parsed.port
    if port is not None:
        ip_netloc = f"{ip_netloc}:{port}"
    connect_url = urlunparse(parsed._replace(netloc=ip_netloc))

    host_header = host if port is None else f"{host}:{port}"
    extensions = {"sni_hostname": host}
    return connect_url, host, {"Host": host_header}, extensions


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
            connect_url, _host, pin_headers, extensions = _pin_url_to_ip(current_url)

            headers: Dict[str, str] = {
                "User-Agent": "litellm-webfetch/1.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.5",
            }
            if extra_headers:
                headers.update(extra_headers)
            headers.update(pin_headers)

            client = await _get_client(timeout or _DEFAULT_TIMEOUT)
            try:
                async with client.stream(
                    "GET", connect_url, headers=headers, extensions=extensions
                ) as resp:
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
                        content_type.startswith(p)
                        for p in _ALLOWED_CONTENT_TYPE_PREFIXES
                    ):
                        raise FetchError(
                            f"Refusing content-type {content_type!r}",
                            status_code=resp.status_code,
                            url=current_url,
                        )

                    body = bytearray()
                    async for chunk in resp.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > _MAX_RESPONSE_BYTES:
                            raise FetchError(
                                f"Response exceeded max size ({_MAX_RESPONSE_BYTES} bytes)",
                                status_code=resp.status_code,
                                url=current_url,
                            )
                    status_code = resp.status_code
            except httpx.HTTPError as exc:
                raise FetchError(str(exc), url=current_url) from exc

            if resp.status_code in (301, 302, 303, 307, 308):
                continue

            text = body.decode("utf-8", errors="replace")
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
                status_code=status_code,
            )

        raise FetchError(f"Exceeded max redirects ({_MAX_REDIRECTS})", url=url)
