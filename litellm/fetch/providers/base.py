"""Base class for fetch providers."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from litellm.types.fetch import FetchResponse


class BaseFetchProvider(ABC):
    """
    Minimal contract every fetch provider must implement.

    Providers are stateless — a fresh instance is constructed per
    ``litellm.afetch()`` call. This keeps it easy to plug in alternative
    providers from the router's ``fetch_tools`` config (sub-issue #7).
    """

    name: str = ""

    @abstractmethod
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
        """Fetch ``url`` and return a normalized :class:`FetchResponse`."""
