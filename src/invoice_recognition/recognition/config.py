"""Settings of the recognition flow: models, limits, and which steps are on."""

from dataclasses import dataclass

FAST = "google/gemini-3.1-flash-lite"
"""The cheap model that reads every page, the header and the layout."""


@dataclass
class Config:
    # One entry per attempt at a page: a page that fails its checks is read again by the
    # next model, up to the end of the tuple.
    page_models: tuple[str, ...] = (FAST, FAST)
    header_model: str = FAST
    concurrency: int = 12  # requests in flight at once
    page_timeout_s: float = 120.0
    page_max_tokens: int = 16_000
