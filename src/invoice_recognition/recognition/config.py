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
    # Photo: the frame is measured before any request (quality.py). A reason of severity
    # "reject" ends the document as "failed" with no request sent; a "warn" is read as
    # usual and flagged.
    photo_quality: bool = True  # False: no measurement, every photo goes to the model
    # The limits below are quality.Limits under the same names; their basis is stated there.
    # Sharpness is Quality.text_sharpness: on the text only, divided by paper brightness,
    # frame as taken. 0 switches a limit off.
    photo_sharpness_reject: float = 140.0
    photo_sharpness_warn: float = 450.0
    photo_paper_level: float = 150.0  # paper darker than this (0-255): warn "dark"
    photo_glare: float = 0.01  # burnt-out spot larger than this share of the sheet: warn
    # The sheet covers more than this share of a frame side: it runs out of the frame there ...
    photo_sheet_touch: float = 0.005
    # ... and text is closer to that side than this many text heights: reject, else warn.
    photo_text_margin_cut: float = 1.5
    # No sheet edge visible at all: text this close to the frame border rejects.
    photo_text_margin_unseen: float = 0.5
