"""Tests for litellm.afetch dispatcher + providers."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

import litellm
from litellm.fetch.main import afetch
from litellm.types.fetch import FetchError, FetchResponse


# ---------- dispatcher ----------


@pytest.mark.asyncio
async def test_missing_provider_raises():
    with pytest.raises(ValueError, match="requires an explicit"):
        await afetch("https://example.com")


@pytest.mark.asyncio
async def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown fetch provider"):
        await afetch("https://example.com", provider="bogus")


@pytest.mark.asyncio
async def test_direct_httpx_requires_opt_in():
    with pytest.raises(ValueError, match="allow_direct_fetch=True"):
        await afetch("https://example.com", provider="direct_httpx")


@pytest.mark.asyncio
async def test_allowed_domains_rejects_offhost():
    with pytest.raises(ValueError, match="allowed_domains"):
        await afetch(
            "https://evil.example.com",
            provider="jina",
            allowed_domains=["example.com"],
        )


@pytest.mark.asyncio
async def test_afetch_is_exposed_on_litellm_module():
    assert litellm.afetch is afetch


# ---------- jina provider ----------


def _mock_httpx_get(
    body: bytes = b'{"data": {"title": "Example", "url": "https://example.com", "content": "body"}}',
    status_code: int = 200,
    headers=None,
):
    mock_response = httpx.Response(
        status_code=status_code,
        content=body,
        headers=headers or {"content-type": "application/json"},
        request=httpx.Request("GET", "https://r.jina.ai/https://example.com"),
    )
    return AsyncMock(return_value=mock_response)


@pytest.mark.asyncio
async def test_jina_happy_path():
    with patch("httpx.AsyncClient.get", new=_mock_httpx_get()):
        result = await afetch("https://example.com", provider="jina")
    assert isinstance(result, FetchResponse)
    assert result.title == "Example"
    assert result.content == "body"
    assert result.content_type == "text/markdown"


@pytest.mark.asyncio
async def test_jina_http_error_raises_fetch_error():
    with patch(
        "httpx.AsyncClient.get",
        new=_mock_httpx_get(body=b"{}", status_code=500),
    ):
        with pytest.raises(FetchError) as excinfo:
            await afetch("https://example.com", provider="jina")
    assert excinfo.value.status_code == 500


# ---------- firecrawl / tavily minimal ----------


@pytest.mark.asyncio
async def test_firecrawl_missing_api_key():
    with pytest.raises(FetchError, match="FIRECRAWL_API_KEY"):
        await afetch("https://example.com", provider="firecrawl")


@pytest.mark.asyncio
async def test_tavily_missing_api_key():
    with pytest.raises(FetchError, match="TAVILY_API_KEY"):
        await afetch("https://example.com", provider="tavily")


# ---------- direct_httpx SSRF ----------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://metadata.google.internal/",
        "ftp://example.com/",
    ],
)
async def test_direct_httpx_blocks_private_and_metadata(bad_url):
    with pytest.raises(FetchError):
        await afetch(bad_url, provider="direct_httpx", allow_direct_fetch=True)


class _FakeStreamResponse:
    def __init__(self, status_code, headers=None, body=b""):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self):
        yield self._body


def _stream_sequence(*responses):
    it = iter(responses)

    def fake_stream(self, method, url, headers=None, extensions=None):
        return next(it)

    return fake_stream


@pytest.mark.asyncio
async def test_direct_httpx_blocks_redirect_to_private(monkeypatch):
    """Redirect chain must re-validate after each hop."""
    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        _stream_sequence(
            _FakeStreamResponse(
                status_code=302, headers={"location": "http://127.0.0.1/"}
            ),
            _FakeStreamResponse(status_code=200, body=b"should not reach"),
        ),
    )

    with pytest.raises(FetchError, match="blocked"):
        await afetch(
            "https://example.com/redirect",
            provider="direct_httpx",
            allow_direct_fetch=True,
        )


@pytest.mark.asyncio
async def test_direct_httpx_rejects_non_allowlisted_content_type(monkeypatch):
    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        _stream_sequence(
            _FakeStreamResponse(
                status_code=200,
                headers={"content-type": "application/octet-stream"},
                body=b"\x00\x01binary",
            )
        ),
    )
    with pytest.raises(FetchError, match="content-type"):
        await afetch(
            "https://example.com/file",
            provider="direct_httpx",
            allow_direct_fetch=True,
        )


@pytest.mark.asyncio
async def test_direct_httpx_enforces_streaming_size_cap(monkeypatch):
    """Oversized response must raise without buffering the full body."""
    from litellm.fetch.providers import direct_httpx as dh

    class _HugeStream(_FakeStreamResponse):
        async def aiter_bytes(self):
            chunk = b"x" * 1024
            # yield until we exceed the cap
            total = 0
            while total <= dh._MAX_RESPONSE_BYTES + 1024:
                yield chunk
                total += len(chunk)

    monkeypatch.setattr(
        httpx.AsyncClient,
        "stream",
        _stream_sequence(
            _HugeStream(status_code=200, headers={"content-type": "text/plain"})
        ),
    )
    with pytest.raises(FetchError, match="exceeded max size"):
        await afetch(
            "https://example.com/huge",
            provider="direct_httpx",
            allow_direct_fetch=True,
        )
