"""What the flow returns: every attempt at a request, every page, the document."""

from dataclasses import asdict, dataclass, field

from invoice_recognition.recognition.boxes import LineBoxes, PageFrame
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.shapes import Checkpoint, Header, JSONObject, Line


@dataclass
class Attempt:
    """One request to the model and what came of it."""

    model: str
    duration_s: float
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    lines: int
    issues: int
    error: str | None = None
    # False: the request ended without a reported cost; cost_usd is then 0.0, not the real price.
    cost_known: bool = True


@dataclass
class PageResult:
    page: int
    lines: list[Line] = field(default_factory=list)
    totals: list[Checkpoint] = field(default_factory=list)  # after_line is local to the page
    issues: list[Issue] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    repaired: int = 0
    # "model": read by the model; "layout": rows found by code (attempts, if any, are the
    # witness or a re-read by the model that was not kept); "skipped": a page of a scan that
    # was not sent to the model - blank, or too blurred to read (Result.quality says which).
    method: str = "model"
    # Text at the top of the page that belongs to the last line item of the previous page;
    # "" when not asked for.
    carry_in: str = ""
    # The page still failed its checks after every attempt by the model, at least one
    # request got an answer, and only one model is configured: a stronger model would read
    # it again here (not called; Result.notes says so).
    stronger_model: bool = False
    # The overall deadline passed before any attempt at this page ran to its end: none was
    # sent, or every one was cut short. The page has a "page_read" issue saying so and the
    # document is "failed". Not in Result.to_json().
    deadline_unread: bool = False
    # The request that read this page of a scan a second time, by another model
    # (Config.second_reading_model); None when none was sent. Not one of `attempts`: those
    # are the attempts at the page, and the result keeps their lines.
    second_reading: Attempt | None = None
    # The model of the attempt whose lines are kept; None for a page found by code or not
    # read by the model. Not in Result.to_json().
    model: str | None = None
    # The attempt whose lines are kept was sent to the backup model (Config.fallback_model)
    # because the provider of the first one failed. Not in Result.to_json().
    by_backup: bool = False


@dataclass
class Result:
    """One document as read. status: "ok" - no issue; "needs_review" - read, with issues;
    "failed" - no line items, no header, nothing sent, or a page left unread at the
    overall deadline (Config.deadline_s). When that deadline cut the reading short, one
    issue "deadline_exceeded" says so and names the pages not read."""

    status: str  # ok | needs_review | failed
    header: Header
    lines: list[Line]
    checkpoints: list[Checkpoint]
    issues: list[Issue]
    pages: list[PageResult]
    evidence_source: str | None
    duration_s: float
    calls: int
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    header_attempts: list[Attempt] = field(default_factory=list)
    cost_unknown_calls: int = 0  # requests whose cost was not reported; cost_usd leaves them out
    layout: JSONObject = field(
        default_factory=dict
    )  # how the layout road went: status, reason, ...
    layout_attempts: list[Attempt] = field(default_factory=list)
    # What changed no number and is not an issue. Every note has "kind", "page" and "text".
    #   rows the layout did not explain, and text the layout parser joined across a page
    #   break: + "row" (the text line on the page)
    #   a tail returned in "carry_in" (source: "carry_in"), "page" = the page it is printed on:
    #     joined_across_pages  - appended to the description of Result.lines[line]:
    #                            + "line", "line_no", "sku", "to_page"
    #     carry_already_joined - that row was read by code and already ends with it;
    #                            nothing was appended: the same keys
    #     carry_without_owner  - no row to append it to; nothing was changed: + "reason"
    #     carry_join_failed    - the join broke ("page" is None); nothing was changed
    #   a page that still fails its checks after every attempt by the model (source:
    #   "retry"): stronger_model_not_called - a stronger model would read it again here;
    #   none is called, to save cost
    #   a page of a scan that has no text and was not sent to the model
    #   (source: "scan_quality"): page_without_text
    #   pages of a scan that were to be read a second time by another model and were not
    #   compared with it (source: "second_reading"): second_reading_missing, "page" is
    #   None, + "pages"; one note for the document
    #   readings kept from the backup model because the provider of the first one failed
    #   (source: "fallback"): fallback_model, "page" is None, + "model", "pages" (the pages
    #   whose line items it read), "header", "layout" (whether it read those); one note for
    #   the document, after every other note
    notes: list[JSONObject] = field(default_factory=list)
    # {} when nothing was measured: a PDF with a text layer, a PDF none of whose pages
    # carries an image, or the check switched off (Config.photo_quality / scan_quality).
    # Photo:
    #   verdict  - ok | warn | reject | unchecked (the measurement itself broke; "error" says how)
    #   reasons  - [{code, message (for the user), severity: reject | warn, value?, side?}]
    #   measures - the numbers (quality.Quality.to_json); limits - the limits they were held
    #              against
    # Every reason is also in `issues` as photo_<code>.
    # Scan:
    #   verdict  - reject (no page was sent: the document is "failed") | warn (a page was
    #              flagged or not read for blur) | ok | unchecked (nothing could be measured;
    #              "error" says how when it broke)
    #   pages    - one entry per page: {page, verdict, reasons, measures?, error?, note?};
    #              the verdict of a page is ok | warn | reject (not sent) | empty (no text:
    #              not sent) | unchecked (read as before: the measurement broke - "error";
    #              the page has no image, or something visible is drawn besides its image -
    #              "note"; no lines of text on a page that is not blank - reason
    #              "no_text_lines")
    #   reasons  - the reasons of all pages, each with "page"; severity: reject | warn |
    #              skip (empty page) | info
    #   limits   - quality.ScanLimits they were held against
    # Every reject and warn reason is also in `issues` as scan_<code> with the page number.
    quality: JSONObject = field(default_factory=dict)
    # Where each line is printed (boxes.py), parallel to `lines`: the rectangles of its
    # printed values on its own page `lines[i]["page"]`, or None when it was not located (a
    # scan, a photo, a page without a usable box, a line whose values are printed nowhere
    # on its page). Rectangles are in points of the page as displayed, top-left origin.
    # Its "row" is the whole printed item on a page found by code (the row and its wrapped
    # description), and the item's own printed row only on a page the model read.
    boxes: list[LineBoxes | None] = field(default_factory=list)
    # How each page of a PDF is displayed (one per page; None for a page without a usable
    # box): the frame the rectangles are given in. [] for a photo and a file not read.
    frames: list[PageFrame | None] = field(default_factory=list)
    # The document has its own printed text that values can be located in: a PDF with a
    # text layer (not a scan, even one with somebody's OCR text over it, nor a photo).
    text_layer: bool = False
    # boxes, frames and text_layer are not in to_json(): the API shows them.

    @property
    def method(self) -> str:
        """How the document was read: "layout" (all pages by code), "model", or "mixed".
        Pages of a scan that were not sent to the model do not count."""
        kinds = {p.method for p in self.pages if p.method != "skipped"}
        return "mixed" if len(kinds) > 1 else next(iter(kinds), "model")

    def to_json(self) -> JSONObject:
        return {
            "status": self.status,
            "header": self.header,
            "lines": self.lines,
            "checkpoints": self.checkpoints,
            "issues": [i.to_json() for i in self.issues],
            "evidence_source": self.evidence_source,
            "duration_s": self.duration_s,
            "calls": self.calls,
            "cost_usd": self.cost_usd,
            "cost_unknown_calls": self.cost_unknown_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "header_attempts": [asdict(a) for a in self.header_attempts],
            "method": self.method,
            "layout": self.layout,
            "notes": self.notes,
            "quality": self.quality,
            "layout_attempts": [asdict(a) for a in self.layout_attempts],
            "pages": [_page_json(p) for p in self.pages],
        }


def _page_json(p: PageResult) -> JSONObject:
    out: JSONObject = {
        "page": p.page,
        "lines": len(p.lines),
        "repaired": p.repaired,
        "method": p.method,
        "issues": [i.to_json() for i in p.issues],
        "attempts": [asdict(a) for a in p.attempts],
    }
    # Only when the page was read a second time: a page that was not has the keys it
    # always had.
    if p.second_reading is not None:
        out["second_reading"] = asdict(p.second_reading)
    return out
