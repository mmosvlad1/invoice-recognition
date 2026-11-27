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
    # A PDF with a text layer: the model describes the layout of the line-item table once and
    # code reads the rows of every page from the text layer. False: every page by the model.
    use_layout: bool = True
    # Also read one page by the model and compare it with the rows found by code.
    layout_witness: bool = True
    layout_model: str = FAST
    layout_max_tokens: int = 900
    # A PDF of more than one page: a page read by the model is asked with the "carry_in"
    # field (carryover.py) and code joins the tail to the last line item of the previous
    # page. A photo, a one-page PDF and the witness page of the layout road are asked with
    # EXTRACT_PAGE as before. False: EXTRACT_PAGE everywhere, no join.
    carry_in: bool = True
