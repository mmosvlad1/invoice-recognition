"""What the flow returns: every attempt at a request, every page, the document."""

from dataclasses import asdict, dataclass, field

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
    # witness or a re-read by the model that was not kept).
    method: str = "model"
    # Text at the top of the page that belongs to the last line item of the previous page;
    # "" when not asked for.
    carry_in: str = ""


@dataclass
class Result:
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
    notes: list[JSONObject] = field(default_factory=list)
    # {} when nothing was measured: a PDF, or the check switched off (Config.photo_quality).
    # Photo:
    #   verdict  - ok | warn | reject | unchecked (the measurement itself broke; "error" says how)
    #   reasons  - [{code, message (for the user), severity: reject | warn, value?, side?}]
    #   measures - the numbers (quality.Quality.to_json); limits - the limits they were held
    #              against
    # Every reason is also in `issues` as photo_<code>.
    quality: JSONObject = field(default_factory=dict)

    @property
    def method(self) -> str:
        """How the document was read: "layout" (all pages by code), "model", or "mixed"."""
        kinds = {p.method for p in self.pages}
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
            "pages": [
                {
                    "page": p.page,
                    "lines": len(p.lines),
                    "repaired": p.repaired,
                    "method": p.method,
                    "issues": [i.to_json() for i in p.issues],
                    "attempts": [asdict(a) for a in p.attempts],
                }
                for p in self.pages
            ],
        }
