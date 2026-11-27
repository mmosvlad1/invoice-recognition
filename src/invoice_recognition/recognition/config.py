"""Settings of the recognition flow: models, limits, and which steps are on."""

from dataclasses import dataclass

FAST = "google/gemini-3.1-flash-lite"
"""The cheap model that reads every page, the header and the layout."""

MAX_IMAGE_PIXELS = 50_000_000
"""The most pixels an image may have to be opened: an uploaded photo (the upload checks)
and the image of a scanned page. Held in memory, 50 million pixels take 50 MB in gray and
150 MB in colour, before any copy; a PDF of under 1 MB can declare a 30000 x 30000 image,
which would take 900 MB."""


@dataclass
class Config:
    # One entry per attempt at a page: a page that fails its checks is read again by the
    # next model, up to the end of the tuple.
    page_models: tuple[str, ...] = (FAST, FAST)
    header_model: str = FAST
    concurrency: int = 12  # requests in flight at once
    # Seconds from the start of process() after which no request is started and a request
    # in flight is cut short; what was read by then is returned (Result.issues
    # "deadline_exceeded"). Local work already running is not interrupted. None: no deadline.
    deadline_s: float | None = None
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
    # Scan (a PDF read page by page because it has no text layer of its own): every page
    # that is nothing but a scan - images, and at most an invisible OCR text layer - is
    # measured before any request (quality.judge_page), at the resolution of the scan
    # itself. Other pages are read as before.
    #   no text and no ink   - the page is not sent; a note "page_without_text", no issue
    #   "reject"             - the page is not sent; issue scan_blurred (severity "error")
    #   "warn"               - the page is read as usual; issue scan_<code> ("warning")
    #   no page left to read - "failed", no request at all (not even for the header)
    scan_quality: bool = True  # False: no measurement, every page goes to the model
    # The limits below are quality.ScanLimits under the same names; their basis is stated
    # there. 0 switches a limit off.
    scan_sharpness_reject: float = 100.0  # the 20 pages of four good scans: 734-987
    scan_sharpness_warn: float = 200.0  # pages at 342 and 223 were still read without a mistake
    scan_paper_level: float = 150.0  # paper darker than this (0-255): warn "dark"
    # Fainter "text" on a page with no ink is the other side showing through: not text.
    scan_min_contrast: float = 0.10
    scan_blank_ink: float = 0.001  # less than this share of the page is clearly dark: blank
    # A page whose images have more pixels than this together is neither opened nor sent: an
    # issue scan_too_large (severity "error"), as for a page too blurred to read. Rendering a
    # page is capped at the same number of pixels.
    max_image_pixels: int = MAX_IMAGE_PIXELS
