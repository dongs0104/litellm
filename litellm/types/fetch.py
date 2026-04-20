"""
Types for the LiteLLM fetch API (``litellm.afetch`` + fetch providers).
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class FetchResponse(BaseModel):
    """
    Normalized fetch result returned by every fetch provider.

    The ``url`` field holds the URL the caller asked for; ``final_url`` is the
    URL actually fetched (after any redirects). Providers that do not expose
    redirect information set ``final_url = None``.
    """

    model_config = ConfigDict(extra="allow")

    url: str
    final_url: Optional[str] = None
    title: Optional[str] = None
    content: str = ""
    content_type: Optional[str] = None
    status_code: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class FetchError(Exception):
    """Raised by fetch providers when a fetch fails in a recoverable way.

    The handler converts this into a ``tool_result`` error block for the model
    to see, rather than propagating it to the caller.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        url: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url


FETCH_PROVIDERS: List[str] = ["jina", "firecrawl", "tavily", "direct_httpx"]
