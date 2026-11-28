"""Every text of the API that a person reads, in one place.

Written for the accountant who uploads an invoice: full sentences, no internal names.
Machine-readable codes stay where they are used; only the words live here.
"""

import json
import logging
from collections.abc import Callable
from typing import assert_never

from invoice_recognition.recognition.checks import Issue, ranges
from invoice_recognition.recognition.issue_values import (
    AttemptCounts,
    ControlTotalOff,
    CutByDeadline,
    DeadlineStop,
    Difference,
    DocumentCoverage,
    FrameSide,
    Glare,
    HeaderNotPrinted,
    IssueForm,
    LayoutRow,
    LineProduct,
    Measured,
    MissingNumbers,
    PageCoverage,
    RateUnreadable,
    RepeatedNumbers,
    RowsCompared,
    RowsDropped,
    SecondColumn,
    SecondPage,
    SecondRow,
    SubtotalOff,
    TaxNoRow,
    TaxRateOff,
    TaxTotalOff,
    TooLarge,
    TotalOff,
)
from invoice_recognition.shapes import Line

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# HTTP errors: Problem.detail (the code says the same for a program)
# --------------------------------------------------------------------------- #

BAD_REQUEST = "The request could not be read."
UNAUTHORIZED = "The access code is missing or wrong."
NOT_FOUND = "There is nothing at this address."
JOB_NOT_FOUND = (
    "There is no document with this id. The server keeps only the most recent documents, "
    "and none from before it was restarted."
)
METHOD_NOT_ALLOWED = "This address does not accept {method} requests."
# Also the upload's own check (uploads.py).
FILE_TOO_LARGE = "The file is larger than the limit of {limit}."
INVALID_REQUEST = (
    "The request is missing something or has a value that is not valid. "
    "The list of errors says which."
)
# A body over the limit of a route that takes no file.
BODY_TOO_LARGE = "The request body is too large."
HTTP_ERROR = "The request could not be completed."
INTERNAL_ERROR = "Something went wrong on the server. The error has been logged."


def size(n: int) -> str:
    """A size in bytes as people read it: "10 MB" (1 MB = 1024 x 1024 bytes)."""
    mb = 1024 * 1024
    return f"{n / mb:g} MB" if n >= mb else f"{n:,} bytes"


# --------------------------------------------------------------------------- #
# Upload refusals (uploads.py; FILE_TOO_LARGE above)
# --------------------------------------------------------------------------- #

UNSUPPORTED_FILE = (
    "Only PDF, JPEG and PNG files can be read, and this file is none of them. "
    "Photos in HEIC format (iPhone) must be saved as JPEG first."
)
PDF_NEEDS_PASSWORD = "The PDF is protected by a password. Upload a copy that opens without one."
PDF_DAMAGED = "The PDF could not be opened. It may be damaged or incomplete."
PDF_NO_PAGES = "The PDF has no pages."
TOO_MANY_PAGES = (
    "The PDF has {pages} pages; the limit is {limit}. Split the file and upload the parts."
)
IMAGE_DAMAGED = "The image could not be opened. It may be damaged or incomplete."
TOO_MANY_DOCUMENTS = (
    "The server is busy with other documents: at most {limit} can wait or be read at once. "
    "Please try again shortly."
)
IMAGE_TOO_LARGE = (
    "The image has more pixels than can be read; the limit is {limit}. "
    "Save the image at a lower resolution."
)

# --------------------------------------------------------------------------- #
# Decisions on issues (review.py)
# --------------------------------------------------------------------------- #

ISSUE_NOT_FOUND = "This document has no issue with this id."
# By why the document cannot be reviewed: still being read, or not read at all.
NOT_REVIEWABLE = {
    "unfinished": "The document is still being read. Its issues can be decided once it is "
    "finished.",
    "failed": "The document could not be read, so it has no issues to decide.",
}
NO_SUGGESTIONS = "This issue has no suggested value to accept. Keep the value as read instead."
NO_SUCH_SUGGESTION = (
    "This issue has {count} suggested {values}, numbered from 0: there is no number {k}."
)


def resolution_conflict(other: int, field: str, theirs: object, ours: object) -> str:
    """Why an accepted value cannot be taken: another accepted decision set the same cell
    (field: "line.amount", "header.total") to a different value."""
    group, _, key = field.partition(".")
    names = _COLUMNS if group == "line" else _HEADER_FIELDS
    cell = f"the {names.get(key, key.replace('_', ' '))}"
    if group == "line":
        cell += " of this line"
    return (
        f"Issue {other} already sets {cell} to {_shown(theirs)}; this suggestion would set it "
        f"to {_shown(ours)}. Take that decision back first, or keep the value as read."
    )


# --------------------------------------------------------------------------- #
# The result (views.py)
# --------------------------------------------------------------------------- #

# Notes the API adds to the result, by kind. The flow's own notes keep their text.
NOTE_TEXTS = {
    # read: as printed, with three decimals ("8.200"); corrected: 8200
    "quantity_repaired": (
        "The quantity was read as {read} and corrected to {corrected}: the point in it is a "
        "thousands separator, as the amount of the line shows."
    ),
}

# --------------------------------------------------------------------------- #
# Why a job failed: JobError.message, by JobError.code (views.job_error)
# --------------------------------------------------------------------------- #

JOB_ERRORS = {
    # Used only when the flow gave no reason of its own; it normally does, and that
    # reason's message is the error's message.
    "photo_unreadable": "The photo cannot be read. Please take the photo again.",
    # Used only when the pages' verdicts are not known; see scan_unreadable() below.
    "scan_unreadable": "No page of the scan could be read.",
    "unreadable_file": "The file could not be opened for reading. It may be damaged or "
    "incomplete; export or scan it again.",
    "model_unavailable": "The model that reads invoices could not be reached: every request "
    "to it failed. Please try again later; if this keeps happening, tell the person who runs "
    "this service.",
    "no_line_items": "The model answered, but no line items were found in the document.",
    "no_header": "Line items were read, but the header and totals were not, so nothing could "
    "be checked.",
    "deadline_exceeded": "Reading took longer than the time allowed for one document and was "
    "stopped before the whole document was read.",
    "internal_error": "Something went wrong on the server while reading this document. The "
    "error has been logged.",
}


def _pages(pages: list[int]) -> tuple[str, bool]:
    """("page 2" | "pages 1–3" | "pages 1 and 4", more than one)."""
    if len(pages) == 1:
        return f"page {pages[0]}", False
    return f"pages {ranges(pages)}", True


def scan_unreadable(
    blurred: list[int], blank: list[int], too_large: list[int] | None = None
) -> str:
    """Why no page of a scan was sent: the pages too blurred, the pages without text and
    the pages holding an image too large to open."""
    too_large = too_large or []
    if not blurred and not too_large:
        return "The scan has no text on any page." if blank else JOB_ERRORS["scan_unreadable"]
    why = []
    if blurred:
        what, many = _pages(blurred)
        why.append(f"{what} {'are' if many else 'is'} too blurred")
    if too_large:
        what, many = _pages(too_large)
        why.append(f"{what} {'hold images' if many else 'holds an image'} too large to open")
    if blank:
        what, many = _pages(blank)
        why.append(f"{what} {'have' if many else 'has'} no text")
    listed = why[0] if len(why) == 1 else f"{', '.join(why[:-1])} and {why[-1]}"
    return f"No page of the scan could be read: {listed}."


# --------------------------------------------------------------------------- #
# Issues: IssueView.message, by the issue's code
# --------------------------------------------------------------------------- #
#
# The flow writes its issue messages for developers ("3 x 10.0 = 30.00, but amount is
# 31.00"). Here each issue gets a sentence or two for the person who checks the invoice:
# what was found, with the values involved, and what to check. The values come as data:
# the issue's page, fields and line, and the values the flow set where it computed them
# (Issue.values, a shape of recognition/issue_values.py). A sentence that names no value
# is chosen by the issue's code and form. An issue that has no wording, or whose wording
# fails, keeps the flow's text and logs a warning: a wording can never lose an issue or
# break a result. Nothing is read back from the flow's message.

type Wording = Callable[[Issue, Line | None], str]
"""(the issue, the line it is about or None) -> the text."""


# Numbers as a person reads them, as the flow's message prints the same number. An exact
# amount (Decimal) and a measured number (float) are printed as they are: "45.0", "4.00",
# "1E+2", "120.5".


def _limit(x: float) -> str:
    """A limit: of a measure ("140", "2.5") or of the time allowed ("240", "1e+06")."""
    return f"{x:g}"


def _pixels(n: int) -> str:
    return f"{n:,}"


def _share(x: float) -> str:
    """A share from 0 to 1 as a percentage, without the sign: 0.034 -> "3.4"."""
    return f"{x * 100:.1f}"


def _numbers(numbers: tuple[int, ...]) -> str:
    """(2, 3, 5) -> "2, 3 and 5"."""
    return _and([str(n) for n in numbers])


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def _and(items: list[str]) -> str:
    """["a"] -> "a"; ["a", "b", "c"] -> "a, b and c"."""
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _on_page(issue: Issue) -> str:
    return f"On page {issue.page}, " if issue.page is not None else ""


def _page(issue: Issue) -> str:
    return f"page {issue.page}" if issue.page is not None else "a page"


def _shown(value: object) -> str:
    """A value of a line as a person reads it: text as it is, a number in its JSON form."""
    if value is None:
        return "empty"
    return value if isinstance(value, str) else json.dumps(value)


# What the columns of a line are called on an invoice.
_COLUMNS = {
    "sku": "item code",
    "quantity": "quantity",
    "unit_price": "unit price",
    "tax_rate": "tax rate",
    "amount": "amount",
}

# What the fields of the header are called on an invoice.
_HEADER_FIELDS = {
    "subtotal": "subtotal",
    "tax_total": "tax total",
    "total": "total",
    "invoice_number": "invoice number",
    "supplier_tax_id": "supplier's tax number",
    "po_number": "purchase order number",
}


def _named(issue: Issue, prefix: str) -> list[str]:
    """The keys the issue names under "line." or "header." (its fields)."""
    return [f.removeprefix(prefix) for f in issue.fields if f.startswith(prefix)]


# The document's totals (doc_sums), one line (line_math), the tax, the numbering and the
# control totals printed among the lines (checkpoints).


def _totals_missing(issue: Issue, line: Line | None) -> str:
    named = [_HEADER_FIELDS.get(k, k) for k in _named(issue, "header.")]
    if not named:
        return (
            "The subtotal, the tax or the total was not read, so the line items cannot be "
            "checked against the totals. Check the totals on the invoice."
        )
    return (
        f"The {_and(named)} {_plural(len(named), 'was', 'were')} not read, so the line items "
        f"cannot be checked against the totals. Check {_plural(len(named), 'it', 'them')} on "
        "the invoice."
    )


def _subtotal_off(v: SubtotalOff) -> str:
    return (
        f"The line amounts add up to {v.lines}, but the invoice subtotal is "
        f"{v.subtotal} (a difference of {v.difference}). A line may be "
        "missing, read twice or have a wrong amount: check the line amounts against the "
        "invoice."
    )


def _total_off(v: TotalOff) -> str:
    return (
        f"The subtotal {v.subtotal} plus the tax {v.tax} makes "
        f"{v.subtotal + v.tax}, but the invoice total is {v.total}. Check "
        "these three figures on the invoice."
    )


def _line_product(v: LineProduct) -> str:
    text = (
        f"{v.quantity} × {v.unit_price} = {v.product}, but the "
        f"amount read is {v.amount}. One of the three values was misread: check the "
        "quantity, the unit price and the amount of this line."
    )
    if v.thousands:
        text += (
            " The quantity may be 1000 times too small: a thousands separator may have been "
            "read as a decimal point."
        )
    return text


def _tax_no_row(v: TaxNoRow) -> str:
    return (
        f"Lines worth {v.base} carry {v.rate}% tax, but the invoice's tax summary has "
        f"no {v.rate}% rate. Check the tax rates of these lines and the summary."
    )


def _tax_rate_off(v: TaxRateOff) -> str:
    return (
        f"The lines taxed at {v.rate}% add up to {v.lines}, but the "
        f"invoice's tax summary gives {v.printed} for that rate. Check the tax rates "
        "of the lines and the summary."
    )


def _tax_total_off(v: TaxTotalOff) -> str:
    return (
        f"The tax worked out from the lines is {v.computed}, but the invoice's tax "
        f"total is {v.printed}. Check the tax rates of the lines and the tax total."
    )


def _rate_unreadable(v: RateUnreadable) -> str:
    return (
        "The invoice's tax summary has a row whose rate could not be read as a number (base "
        f"{v.printed}). Check the tax rates in the summary."
    )


def _repeated(v: RepeatedNumbers) -> str:
    if len(v.numbers) == 1:
        return (
            f"Position number {v.numbers[0]} appears more than once. A line may have been read "
            "twice: check the lines with this number."
        )
    return (
        f"Position numbers {_numbers(v.numbers)} appear more than once. Lines may have been "
        "read twice: check the lines with these numbers."
    )


def _missing(v: MissingNumbers) -> str:
    if v.count == 1:
        return (
            f"Position number {v.numbers[0]} is missing from the sequence. A line may have "
            "been skipped: check the invoice around it."
        )
    # The flow lists the first ten.
    which = _numbers(v.numbers)
    if v.count > len(v.numbers):
        which = f"for example {which}"
    return (
        f"{v.count} position numbers are missing from the sequence: {which}. Lines may have "
        "been skipped: check the invoice around them."
    )


def _control_total_off(v: ControlTotalOff) -> str:
    return (
        f"Line items {v.first}–{v.last} add up to {v.lines}, but the printed "
        f'"{v.label}" is {v.printed}. A line in this range may be missing, read '
        "twice or misread: check them."
    )


# Whether what was read is printed in the PDF's own text. Which cells: the issue's fields.


def _not_printed(issue: Issue, line: Line | None) -> str:
    keys = _named(issue, "line.")
    if not keys:
        raise ValueError("the issue names no cell of the line")
    named = [
        f"{_COLUMNS.get(k, k)} ({_shown(line.get(k))})" if line is not None else _COLUMNS.get(k, k)
        for k in keys
    ]
    verb = _plural(len(keys), "is", "are")
    them = _plural(len(keys), "it", "them")
    return (
        f"This line's {_and(named)} {verb} not printed on the page. The model may have "
        f"misread {them}: check {them} on the invoice."
    )


def _header_not_printed(issue: Issue, v: HeaderNotPrinted) -> str:
    [key] = _named(issue, "header.")
    return (
        f"The {_HEADER_FIELDS.get(key, key)} read as {v.value} is not printed anywhere in the "
        "document. It may have been misread: check it on the invoice."
    )


def _line_items(printed: int, read: int) -> str:
    """ "seems to have 2 line items, but 1 was read" """
    items = _plural(printed, "line item", "line items")
    return f"seems to have {printed} {items}, but {read} {_plural(read, 'was', 'were')} read"


def _page_coverage(issue: Issue, v: PageCoverage) -> str:
    return (
        f"{_page(issue).capitalize()} {_line_items(v.printed, v.read)}. A line may be missing "
        "or read twice: check the page."
    )


def _document_coverage(v: DocumentCoverage) -> str:
    return (
        f"The document {_line_items(v.printed, v.read)}. A line may be missing or read "
        "twice: check the line items."
    )


# A page read by the model.


def _page_failed(issue: Issue, line: Line | None) -> str:
    # Whatever broke (a request that failed, an answer cut short or not valid JSON, an
    # error while reading): line items may still have been saved from a broken answer, or
    # found in the PDF's text by code, so the text does not claim they are missing.
    return (
        f"{_page(issue).capitalize()} could not be read in full: the model's answer was "
        "broken or every request failed. Line items from this page may be missing: check "
        "the page."
    )


def _not_read(issue: Issue, line: Line | None) -> str:
    return (
        f"{_page(issue).capitalize()} was not read: reading the document took longer than "
        "allowed. Its line items are missing; upload the document again to read it in full."
    )


def _page_empty(issue: Issue, line: Line | None) -> str:
    return (
        f"No line items were read from {_page(issue)}. If the page has line items, they are "
        "missing: check it."
    )


def _cut_by_deadline(issue: Issue, v: CutByDeadline) -> str:
    items = _plural(v.lines, "line item", "line items")
    return (
        f"{_page(issue).capitalize()} was not read to the end: reading the document took "
        f"longer than allowed. The {v.lines} {items} read from it may be incomplete: check "
        "the page."
    )


def _dropped(issue: Issue, v: RowsDropped) -> str:
    if v.rows == 1:
        return (
            f"1 row read from {_page(issue)} had neither an item code nor an amount and was "
            "left out. If it is a line item, it is missing: check the page."
        )
    return (
        f"{v.rows} rows read from {_page(issue)} had neither an item code nor an amount and "
        "were left out. If any of them is a line item, it is missing: check the page."
    )


def _disagree(issue: Issue, v: AttemptCounts) -> str:
    return (
        f"{_page(issue).capitalize()} was read {_times(len(v.counts))}, and the readings found "
        f"different numbers of line items ({_numbers(v.counts)}). A line may be missing or "
        "read twice: check the page."
    )


def _times(n: int) -> str:
    return "twice" if n == 2 else f"{n} times"


# A PDF whose rows were taken from its text by code.


def _layout_row(issue: Issue, v: LayoutRow) -> str:
    match v.kind:
        case "stray_words":
            what = (
                "the row of a line item also holds words outside the table's columns: "
                f'"{v.row}". Check that this line was read in full.'
            )
        case "no_tax_rate":
            what = (
                f'no tax rate was found in the row "{v.row}", though the table has a tax rate '
                "column. Check the tax rate of this line."
            )
        case "text_under_item":
            what = (
                f'the text "{v.row}" under a line item was not added to it. Check whether it '
                "belongs to the description of that line."
            )
        case _:
            assert_never(v.kind)
    said = _on_page(issue) + what
    return said[0].upper() + said[1:]


def _rows_compared(issue: Issue, v: RowsCompared) -> str:
    match issue.check:
        case "layout_swapped":
            return (
                f"On {_page(issue)}, the line items taken from the text of the PDF and those "
                "read by the model disagree: quantity and unit price look swapped "
                f"({v.crossed} of {v.compared} lines). The model's reading of every page was "
                "used instead: check the quantities and unit prices."
            )
        case "layout_mismatch":
            return (
                f"On {_page(issue)}, the line items taken from the text of the PDF and those "
                f"read by the model differ (only {v.same} of {v.compared} lines agree). The "
                "model's reading of every page was used instead: check the line items."
            )
    raise ValueError(f"rows compared for {issue.check}")


def _unconfirmed(why: str) -> Wording:
    def text(issue: Issue, line: Line | None) -> str:
        return (
            f"The line items were taken from the text of the PDF, but the control reading of "
            f"{_page(issue)} {why}, so nothing confirms which column is the quantity and "
            "which the unit price. Check that they are not swapped."
        )

    return text


# The document as a whole.


def _deadline(v: DeadlineStop) -> str:
    text = [
        (
            f"Reading was stopped after {_limit(v.seconds)} seconds, the time allowed for one "
            "document."
        )
    ]
    if len(v.unread) == 1:
        text.append(f"Page {v.unread[0]} was not read.")
    elif v.unread:
        text.append(f"Pages {ranges(v.unread)} were not read.")
    if not v.header_read:
        text.append("The header (supplier, invoice number, totals) was not read.")
    if not v.unread and v.header_read:
        text.append(
            "Every page was read; only a second reading of a page or of the header was cut short."
        )
    return " ".join(text)


# A photo, or a page of a scan, measured before it is read.


def _measured(issue: Issue, v: Measured) -> str:
    limit = _limit(v.limit)
    # Only sharpness stops a page or a photo; the other measures are warnings, and their
    # sentences say that values were read. Raised as an error, they have no sentence.
    error = issue.severity == "error"
    match issue.check:
        case "photo_blurred" if error:
            return (
                f"The photo is too blurred to read: its sharpness is {v.value}, and at least "
                f"{limit} is needed. Nothing was read; please take the photo again, holding "
                "the phone still."
            )
        case "photo_blurred":
            return (
                f"The photo is not quite sharp (sharpness {v.value}; {limit} or more is safe). "
                "Amounts are usually read correctly, but item codes may be misread: check them."
            )
        case "scan_blurred" if error:
            return (
                f"{_page(issue).capitalize()} is too blurred to read (sharpness {v.value}; at "
                f"least {limit} is needed), so it was not read. Scan it again at a higher "
                "resolution."
            )
        case "scan_blurred":
            return (
                f"{_page(issue).capitalize()} is not quite sharp (sharpness {v.value}; {limit} or "
                "more is safe). Values read from it may contain errors: check them."
            )
        case "photo_dark" | "scan_dark" if not error:
            what = "The photo" if issue.check == "photo_dark" else _page(issue).capitalize()
            return (
                f"{what} is dark (the paper's brightness is {v.value} of 255; below {limit} "
                "counts as dark). Values read from it may contain errors: check them."
            )
        case "photo_text_small" if not error:
            return (
                f"The text is small in the photo (its lines are {v.value} pixels tall; {limit} or "
                "more is safe). Values may be misread: check them, or take the photo again "
                "from closer."
            )
    raise ValueError(f"a measure for {issue.check} ({issue.severity})")


def _frame_side(issue: Issue, v: FrameSide) -> str:
    side = f"the {v.side} edge" if v.side else "an edge"
    match issue.check:
        case "photo_text_at_edge" if issue.severity == "error":
            return (
                f"Text runs into {side} of the photo: part of the invoice is cut off. Nothing "
                "was read; please take the photo again with the whole sheet in the frame."
            )
        case "photo_text_at_edge":
            return (
                f"Text comes close to {side} of the photo: part of the invoice may be cut off. "
                "Check that no line or total is missing."
            )
        case "photo_sheet_cut" if issue.severity != "error":
            return (
                f"The sheet runs out of the photo past {side}. No text is near that edge, but "
                "check that nothing of the invoice is missing."
            )
    raise ValueError(f"a side of the frame for {issue.check} ({issue.severity})")


def _glare(issue: Issue, v: Glare) -> str:
    if issue.severity == "error":  # a warning only: its sentence says values were read
        raise ValueError("glare raised as an error")
    return (
        f"Glare covers {_share(v.share)}% of the sheet, so some text may be washed out. Check "
        "the values in the bright part of the photo."
    )


def _too_large(issue: Issue, v: TooLarge) -> str:
    return (
        f"{_page(issue).capitalize()} holds an image too large to open ({_pixels(v.pixels)} "
        f"pixels; at most {_pixels(v.limit)} can be opened), so it was not read. Save the scan "
        "at a lower resolution and upload it again."
    )


# A page of a scan read a second time, by another model (second_reading.py).


def _difference(d: Difference) -> str:
    """One field the two readings differ on, each value as the flow quotes it."""
    label = _COLUMNS.get(d.field, d.field)
    if d.theirs is None:
        return f"no {label} where the result has {d.ours}"
    if d.ours is None:
        return f"{label} {d.theirs} where the result has none"
    return f"{label} {d.theirs} instead of {d.ours}"


def _second_row(issue: Issue, v: SecondRow) -> str:
    place = f"Position {issue.line_no}" if issue.line_no is not None else f"Line item {v.item}"
    found = _and([_difference(d) for d in v.differences])
    return (
        f"{place} on {_page(issue)} was read differently by a second model: {found}. One of "
        "the two readings is wrong: check this line on the invoice."
    )


def _second_column(issue: Issue, v: SecondColumn) -> str:
    label = _COLUMNS.get(v.difference.field, v.difference.field)
    where = f"position {v.line_no}" if v.line_no is not None else f"line item {v.item}"
    return (
        f"On {_page(issue)}, a second model reads the {label} differently on {v.differ} of "
        f"{v.compared} line items (for example at {where}: {_difference(v.difference)}). The "
        "two readings write this column differently, or one of them misread it: check the "
        f"{label} column on this page."
    )


def _second_page(issue: Issue, v: SecondPage) -> str:
    if v.found == v.result:
        counts = (
            f"On {_page(issue)}, a second model finds as many line items as the result "
            f"({v.result}), but not at the same positions."
        )
    else:
        items = _plural(v.found, "line item", "line items")
        counts = (
            f"On {_page(issue)}, a second model finds {v.found} {items} where the result has "
            f"{v.result}."
        )
    text = [counts]
    if not v.matched:
        text.append("The lines could not be matched one to one, so they were not compared.")
    if v.extra:
        are = _plural(len(v.extra), "is", "are")
        text.append(
            f"{_plural(len(v.extra), 'Position', 'Positions')} {ranges(v.extra)} "
            f"{are} not in the result."
        )
    if v.absent:
        were = _plural(len(v.absent), "was", "were")
        text.append(
            f"{_plural(len(v.absent), 'Position', 'Positions')} {ranges(v.absent)} "
            f"of the result {were} not found by the second reading."
        )
    text.append("Check this page on the invoice.")
    return " ".join(text)


# The sentences that name no value, by (code, form). A check with several sentences has a
# form for each of them and no (code, None) key: an issue of it that has neither values
# nor a form has no sentence and keeps the flow's text (and logs a warning). So does, on
# purpose, a reason of quality.py that has no sentence here (photo_<code>, scan_<code>;
# none is raised today).
_PLAIN: dict[tuple[str, IssueForm | None], Wording] = {
    ("doc_sums", "supplier"): lambda i, line: (
        "The supplier's name was not read. Check who issued the invoice."
    ),
    ("doc_sums", "no_lines"): lambda i, line: (
        "No line items were read, so the totals cannot be checked against them."
    ),
    ("doc_sums", "totals_missing"): _totals_missing,
    ("doc_sums", "amounts_missing"): lambda i, line: (
        "Some line items have no amount, so they cannot be checked against the subtotal. "
        "Check the amounts of the lines."
    ),
    ("line_math", "unusable"): lambda i, line: (
        "The quantity, the unit price or the amount of this line is missing or is not a "
        "number, so the line cannot be checked. Check it on the invoice."
    ),
    ("grounding_value", None): _not_printed,
    ("grounding_row", None): lambda i, line: (
        "Each value of this line is printed on the page, but not together in one row: the "
        "line may mix values from different rows. Check it on the invoice."
    ),
    ("page_read", "failed"): _page_failed,
    ("page_read", "not_read"): _not_read,
    ("page_empty", None): _page_empty,
    ("layout_unverified", "deadline"): _unconfirmed("was stopped by the time limit"),
    ("layout_unverified", "no_rows"): _unconfirmed("returned no line items"),
    ("unreadable_file", None): lambda i, line: JOB_ERRORS["unreadable_file"],
    ("photo_no_text", None): lambda i, line: (
        "No text was found in the photo, so nothing was read. Check that the photo shows the "
        "invoice and is close enough to read."
    ),
    ("photo_sheet_edge_unseen", None): lambda i, line: (
        "No edge of the sheet is visible in the photo, so nothing confirms that the whole "
        "invoice is in it. Check that no line or total is cut off."
    ),
    ("scan_no_text", None): lambda i, line: (
        "No page of the scan has any text, so nothing was read. Check that this is the right file."
    ),
}


def _worded(issue: Issue, line: Line | None) -> str:
    """The issue's text from its values, or from its code and form; raises when it has
    neither, values of a shape not made for its code, or both."""
    values = issue.values
    if values is not None and (issue.form is not None or issue.check not in type(values).codes):
        raise ValueError(f"{type(values).__name__} (form {issue.form!r}) on {issue.check}")
    match values:
        case None:
            return _PLAIN[issue.check, issue.form](issue, line)
        case SubtotalOff():
            return _subtotal_off(values)
        case TotalOff():
            return _total_off(values)
        case LineProduct():
            return _line_product(values)
        case TaxNoRow():
            return _tax_no_row(values)
        case TaxRateOff():
            return _tax_rate_off(values)
        case TaxTotalOff():
            return _tax_total_off(values)
        case RateUnreadable():
            return _rate_unreadable(values)
        case RepeatedNumbers():
            return _repeated(values)
        case MissingNumbers():
            return _missing(values)
        case ControlTotalOff():
            return _control_total_off(values)
        case HeaderNotPrinted():
            return _header_not_printed(issue, values)
        case PageCoverage():
            return _page_coverage(issue, values)
        case DocumentCoverage():
            return _document_coverage(values)
        case CutByDeadline():
            return _cut_by_deadline(issue, values)
        case RowsDropped():
            return _dropped(issue, values)
        case AttemptCounts():
            return _disagree(issue, values)
        case LayoutRow():
            return _layout_row(issue, values)
        case RowsCompared():
            return _rows_compared(issue, values)
        case DeadlineStop():
            return _deadline(values)
        case Measured():
            return _measured(issue, values)
        case FrameSide():
            return _frame_side(issue, values)
        case Glare():
            return _glare(issue, values)
        case TooLarge():
            return _too_large(issue, values)
        case SecondRow():
            return _second_row(issue, values)
        case SecondColumn():
            return _second_column(issue, values)
        case SecondPage():
            return _second_page(issue, values)
        case _:
            assert_never(values)


def issue_text(issue: Issue, line: Line | None = None) -> str:
    """The message of an issue for a person; line: the line the issue is about, if any."""
    try:
        return _worded(issue, line)
    except Exception:  # fault boundary: a wording must never cost the document its result
        logger.warning(
            "issue %s (form %r, values %s) could not be worded",
            issue.check,
            issue.form,
            type(issue.values).__name__,
            exc_info=True,
        )
        return issue.message
