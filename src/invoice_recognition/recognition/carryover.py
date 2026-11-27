"""A line item cut by a page break, on the page-by-page road: give the tail a place
and join it by code.

A scan is read one page per request (prompts.EXTRACT_PAGE). When a row does not fit
at the bottom of page N, the rest of its description is printed at the top of page
N+1, without a position number, sku or numbers of its own. The answer to EXTRACT_PAGE
has no place for that text, so the model drops it and the description of the row
stays cut, with nothing to flag it.

Here the model is not asked to assemble anything. The page prompt gets one more
top-level field, "carry_in", for such orphan text; join_carry() then appends it to
the last line item of the previous page. Pure functions, no network; pipeline uses
the prompt and the join for a PDF of more than one page (Config.carry_in).
"""

from typing import Any

from invoice_recognition.recognition.prompts import EXTRACT_PAGE
from invoice_recognition.shapes import Checkpoint, JSONObject, Line

# The paragraph that explains the new field. It names no word of any test document.
CARRY_RULE = """\
"carry_in" is the place for text left over from the previous page. When a line item does
not fit at the bottom of a page, the rest of its description is printed at the top of the
next page: above the first line item of that page (below the repeated table header, if
there is one), and without a position number, sku, quantity, price or amount of its own.
If THIS page starts with such text, copy it into "carry_in" exactly as printed, joined into
one string. Do NOT add it to the description of any line item on this page and do not
return it as a row: it belongs to the last line item of the previous page, which is read
separately. The table header, section headings, "carried forward" rows, subtotals and
totals are never "carry_in". If the page has no such text, "carry_in" is "".

"""

# EXTRACT_PAGE is changed in three places and nowhere else: (old, new).
# "carry_in" comes first in the answer: the text is at the top of the page, so the model
# has to decide about it before it writes the first row, not after the last one.
_EDITS = (
    ('{{\n  "rows": [\n', '{{\n  "carry_in": str,\n  "rows": [\n'),
    (
        '"rows" lists, in the order printed, two kinds of entries:',
        CARRY_RULE + '"rows" lists, in the order printed, two kinds of entries:',
    ),
    (
        'If the page has no line items, return {{"rows": []}}.',
        'If the page has no line items, "rows" is [].',
    ),
)


def _carry_template() -> str:
    out = EXTRACT_PAGE
    for old, new in _EDITS:
        if out.count(old) != 1:  # the page prompt was reworded: do not send a half-edited one
            raise ValueError(f"carryover: EXTRACT_PAGE no longer has exactly one {old!r}")
        out = out.replace(old, new)
    return out


EXTRACT_PAGE_CARRY = _carry_template()


def carry_prompt(page: int, total: int) -> str:
    """The page prompt of the flow plus the "carry_in" field."""
    return EXTRACT_PAGE_CARRY.format(page=page, total=total)


# --------------------------------------------------------------------------- #
# One answer
# --------------------------------------------------------------------------- #


def squash(s: object) -> str:
    return " ".join(str(s or "").split())


def carry_text(value: object) -> str:
    """What the model put into "carry_in" as one line of text; "" for null and for
    anything that is not text."""
    if isinstance(value, str):
        return squash(value)
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return squash(" ".join(value))
    return ""


# --------------------------------------------------------------------------- #
# Joining
# --------------------------------------------------------------------------- #

JOINED = "joined_across_pages"  # the name layout.py uses for the same thing on a text-layer PDF
ORPHAN = "carry_without_owner"
PRESENT = "carry_already_joined"  # the row was read by code, which had appended this very text


def is_total(row: object) -> bool:
    return isinstance(row, dict) and "total_label" in row


def is_line_item(row: object) -> bool:
    """The rule of pipeline.rows_to_page: not a control total, and it has a sku or an amount."""
    return (
        isinstance(row, dict)
        and "total_label" not in row
        and (row.get("sku") is not None or row.get("amount") is not None)
    )


def flow_rows(lines: list[Line], totals: list[Checkpoint]) -> list[JSONObject]:
    """The rows of one page in the order printed, from the two lists the pipeline keeps
    (rows_to_page: line items, and control totals with "after_line" = how many line items
    of the page precede each). The line items are the very dicts that were passed in; a
    control total comes back as {"total_label", "amount"}."""
    before: dict[int, list[JSONObject]] = {}
    for t in totals:
        at = t.get("after_line")
        at = at if isinstance(at, int) and not isinstance(at, bool) else len(lines)
        before.setdefault(min(max(at, 0), len(lines)), []).append(
            {"total_label": t.get("label"), "amount": t.get("amount")}
        )
    rows: list[JSONObject] = []
    for k, item in enumerate(lines):
        rows += before.get(k, [])
        rows.append(item)
    return rows + before.get(len(lines), [])


def _has_tail(description: object, text: str) -> bool:
    return bool(text) and squash(text).casefold() in squash(description).casefold()


def _owner(prev: JSONObject | None, page: int) -> tuple[int | None, str]:
    """Index of the row on the previous page that the tail belongs to, or why there is none."""
    if page <= 1:
        return None, "This is the first page of the document."
    if prev is None:
        return None, f"Page {page - 1} was not read."
    if prev.get("skipped"):
        return None, f"Page {page - 1} was not sent to the model."
    if prev.get("unread"):
        return None, f"Page {page - 1} could not be read in full."
    rows = prev.get("rows") or []
    if not rows:
        return None, f"Page {page - 1} has no rows."
    if is_total(rows[-1]):
        return None, f"Page {page - 1} ends with a control total."
    if not is_line_item(rows[-1]):
        return None, f"The last row of page {page - 1} is not a line item."
    return len(rows) - 1, ""


def join_carry(pages: list[JSONObject]) -> tuple[list[JSONObject], list[JSONObject]]:
    """pages: [{"page": int, "rows": [...], "carry_in": str}, ...] - one entry per page
    that was read, "rows" as the model returned them (line items and control totals, in
    order).

    For every page N+1 with a non-empty "carry_in": when the LAST entry of the rows of
    page N is a line item, the text is appended to its description after a space; when
    it is a control total (or page N is missing, empty, or N+1 is the first page),
    nothing is changed. Either way a note says what happened.

    Optional keys of a page serve the flow (pipeline), where a page is not always an
    answer of the model. "unread": True - the page was not read to its end (a failed
    request, a cut-off answer): its last row is not known to be the last row printed, so
    it owns nothing. "skipped": True - the page was not sent to the model at all (a blank
    or unreadable page of a scan): it owns nothing either. "joined_by_code": True - the rows were read by code from the text
    layer (layout.py), which appends text carried over to the next page by itself: when
    the description of the owner already contains the text, nothing is appended and the
    note is "carry_already_joined".

    Returns (pages, notes). The input is not changed: the returned pages are copies, a
    joined page carries "carry_joined": True and is not joined again. A note is {"page":
    the page the text is printed on, "kind", "text", "reason"} plus, for a join,
    "to_page", "row" (index in the rows of that page), "line_no", "sku"."""
    out = sorted(({**p, "rows": list(p.get("rows") or [])} for p in pages), key=lambda p: p["page"])
    by_page = {p["page"]: p for p in out}
    notes: list[JSONObject] = []
    for p in out:
        text = carry_text(p.get("carry_in"))
        if not text or p.get("carry_joined"):
            continue
        prev = by_page.get(p["page"] - 1)
        idx, why = _owner(prev, p["page"])
        if idx is None or prev is None:
            notes.append({"page": p["page"], "kind": ORPHAN, "text": text, "reason": why})
            continue
        row: dict[str, Any] = dict(prev["rows"][idx])
        head = row.get("description")
        found = {
            "to_page": prev["page"],
            "row": idx,
            "line_no": row.get("line_no"),
            "sku": row.get("sku"),
        }
        if prev.get("joined_by_code") and _has_tail(head, text):
            notes.append({"page": p["page"], "kind": PRESENT, "text": text, "reason": ""} | found)
            continue
        row["description"] = (
            f"{head.rstrip()} {text}" if isinstance(head, str) and head.strip() else text
        )
        prev["rows"][idx] = row
        p["carry_joined"] = True
        notes.append({"page": p["page"], "kind": JOINED, "text": text, "reason": ""} | found)
    return out, notes
