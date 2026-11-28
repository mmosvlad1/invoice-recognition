"""Every text of the API that a person reads, in one place.

Written for the accountant who uploads an invoice: full sentences, no internal names.
Machine-readable codes stay where they are used; only the words live here.
"""

import json
import logging
import re
from collections.abc import Callable
from typing import assert_never

from invoice_recognition.recognition.checks import Issue, ranges
from invoice_recognition.recognition.issue_values import (
    ControlTotalOff,
    DocumentCoverage,
    HeaderNotPrinted,
    IssueForm,
    LineProduct,
    MissingNumbers,
    PageCoverage,
    RepeatedNumbers,
    SubtotalOff,
    TaxNoRow,
    TaxRateOff,
    TaxTotalOff,
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
# break a result.
#
# Transitional: the codes not yet given values are still worded from the flow's message,
# by the patterns of ISSUE_TEXTS below.

type Wording = Callable[[Issue, Line | None], str]
"""(the issue, the line it is about or None) -> the text."""

type Parsed = Callable[[Issue, Line | None], str | None]
"""Transitional: a wording read from the flow's message; None: keep the flow's text."""


# Numbers as a person reads them, as the flow's message prints the same number. An exact
# amount (Decimal) and a measured number (float) are printed as they are: "45.0", "4.00",
# "1E+2", "120.5".


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


def _rules(*rules: tuple[str, Callable[[re.Match[str], Issue, Line | None], str]]) -> Parsed:
    """A wording from (pattern of the flow's message, text for it) pairs; the first
    pattern that matches the whole message wins."""
    compiled = [(re.compile(pattern), text) for pattern, text in rules]

    def word(issue: Issue, line: Line | None) -> str | None:
        for pattern, text in compiled:
            if (m := pattern.fullmatch(issue.message)) is not None:
                return text(m, issue, line)
        return None

    return word


def _always(text: Callable[[Issue], str]) -> Parsed:
    """A wording that does not need the flow's message."""
    return lambda issue, line: text(issue)


_NUMBER = r"(-?[\d.]+(?:E[+-]?\d+)?)"


# What the columns of a line are called on an invoice.
_COLUMNS = {
    "sku": "item code",
    "quantity": "quantity",
    "unit_price": "unit price",
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


def _dropped(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    n = int(m.group(1))
    if n == 1:
        return (
            f"1 row read from {_page(issue)} had neither an item code nor an amount and was "
            "left out. If it is a line item, it is missing: check the page."
        )
    return (
        f"{n} rows read from {_page(issue)} had neither an item code nor an amount and were "
        "left out. If any of them is a line item, it is missing: check the page."
    )


def _disagree(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    counts = m.group(1).split(" and ")
    return (
        f"{_page(issue).capitalize()} was read {_times(len(counts))}, and the readings found "
        f"different numbers of line items ({_and(counts)}). A line may be missing or read "
        "twice: check the page."
    )


def _times(n: int) -> str:
    return "twice" if n == 2 else f"{n} times"


def _not_read(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    return (
        f"Page {m.group(1)} was not read: reading the document took longer than allowed. "
        "Its line items are missing; upload the document again to read it in full."
    )


def _not_read_to_the_end(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    page, n = m.group(1), int(m.group(2))
    items = _plural(n, "line item", "line items")
    return (
        f"Page {page} was not read to the end: reading the document took longer than allowed. "
        f"The {n} {items} read from it may be incomplete: check the page."
    )


def _page_failed(issue: Issue) -> str:
    # The flow's message here is an error (a request that failed, an answer cut short or
    # not valid JSON). Line items may still have been saved from a broken answer, or found
    # in the PDF's text by code, so the text does not claim they are missing.
    return (
        f"{_page(issue).capitalize()} could not be read in full: the model's answer was "
        "broken or every request failed. Line items from this page may be missing: check "
        "the page."
    )


def _row(kind: str) -> Callable[[re.Match[str], Issue, Line | None], str]:
    def text(m: re.Match[str], issue: Issue, line: Line | None) -> str:
        row = m.group(1)
        if kind == "stray_words":
            what = (
                "the row of a line item also holds words outside the table's columns: "
                f'"{row}". Check that this line was read in full.'
            )
        elif kind == "no_tax_rate":
            what = (
                f'no tax rate was found in the row "{row}", though the table has a tax rate '
                "column. Check the tax rate of this line."
            )
        else:
            what = (
                f'the text "{row}" under a line item was not added to it. Check whether it '
                "belongs to the description of that line."
            )
        said = _on_page(issue) + what
        return said[0].upper() + said[1:]

    return text


def _crossed(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    page, _, compared, crossed = m.groups()
    return (
        f"On page {page}, the line items taken from the text of the PDF and those read by the "
        f"model disagree: quantity and unit price look swapped ({crossed} of {compared} lines). "
        "The model's reading of every page was used instead: check the quantities and unit "
        "prices."
    )


def _differ(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    page, same, compared, _ = m.groups()
    return (
        f"On page {page}, the line items taken from the text of the PDF and those read by the "
        f"model differ (only {same} of {compared} lines agree). The model's reading of every "
        "page was used instead: check the line items."
    )


def _unconfirmed(why: str) -> Callable[[re.Match[str], Issue, Line | None], str]:
    def text(m: re.Match[str], issue: Issue, line: Line | None) -> str:
        return (
            f"The line items were taken from the text of the PDF, but the control reading of "
            f"page {m.group(1)} {why}, so nothing confirms which column is the quantity and "
            "which the unit price. Check that they are not swapped."
        )

    return text


def _deadline(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    seconds, rest = m.groups()
    rest = rest.replace(
        "Every page was read; a repeated read of a page or of the header was stopped.",
        "Every page was read; only a second reading of a page or of the header was cut short.",
    )
    return f"Reading was stopped after {seconds} seconds, the time allowed for one document. {rest}"


def _blurred_photo(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    value, limit = m.group(1), m.group(2)
    if issue.severity == "error":
        return (
            f"The photo is too blurred to read: its sharpness is {value}, and at least {limit} "
            "is needed. Nothing was read; please take the photo again, holding the phone still."
        )
    return (
        f"The photo is not quite sharp (sharpness {value}; {limit} or more is safe). Amounts "
        "are usually read correctly, but item codes may be misread: check them."
    )


def _edge(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    side = f"the {m.group(1)} edge" if m.group(1) else "an edge"
    if issue.severity == "error":
        return (
            f"Text runs into {side} of the photo: part of the invoice is cut off. Nothing was "
            "read; please take the photo again with the whole sheet in the frame."
        )
    return (
        f"Text comes close to {side} of the photo: part of the invoice may be cut off. Check "
        "that no line or total is missing."
    )


def _sheet_cut(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    side = f"the {m.group(1)} edge" if m.group(1) else "an edge"
    return (
        f"The sheet runs out of the photo past {side}. No text is near that edge, but check "
        "that nothing of the invoice is missing."
    )


def _dark(what: Callable[[Issue], str]) -> Callable[[re.Match[str], Issue, Line | None], str]:
    def text(m: re.Match[str], issue: Issue, line: Line | None) -> str:
        value, limit = m.groups()
        return (
            f"{what(issue)} is dark (the paper's brightness is {value} of 255; below {limit} "
            "counts as dark). Values read from it may contain errors: check them."
        )

    return text


def _glare(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    return (
        f"Glare covers {m.group(1)}% of the sheet, so some text may be washed out. Check the "
        "values in the bright part of the photo."
    )


def _small(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    value, limit = m.groups()
    return (
        f"The text is small in the photo (its lines are {value} pixels tall; {limit} or more is "
        "safe). Values may be misread: check them, or take the photo again from closer."
    )


def _blurred_page(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    value, limit = m.group(1), m.group(2)
    page = _page(issue).capitalize()
    if issue.severity == "error":
        return (
            f"{page} is too blurred to read (sharpness {value}; at least {limit} is needed), so "
            "it was not read. Scan it again at a higher resolution."
        )
    return (
        f"{page} is not quite sharp (sharpness {value}; {limit} or more is safe). Values read "
        "from it may contain errors: check them."
    )


def _item_codes(text: str) -> str:
    """The flow's words for a second reading name the item code "SKU" where it labels a
    value ("SKU A instead of B", "no SKU where ..."); a person reads "item code"."""
    return re.sub(r"(^|, | and |no )SKU(?= )", r"\1item code", text)


def _read_twice_row(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    place, page, found = m.groups()
    return (
        f"{place} on page {page} was read differently by a second model: "
        f"{_item_codes(found)}. One of the two readings is wrong: check this line on the "
        "invoice."
    )


def _read_twice_column(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    page, label, differ, compared, where, example = m.groups()
    label = "item code" if label == "SKU" else label
    return (
        f"On page {page}, a second model reads the {label} differently on {differ} of "
        f"{compared} line items (for example at {where}: {_item_codes(example)}). The two "
        f"readings write this column differently, or one of them misread it: check the {label} "
        "column on this page."
    )


def _read_twice_page(m: re.Match[str], issue: Issue, line: Line | None) -> str:
    page, found, result, rest = m.groups()
    return (
        f"On page {page}, a second model finds {found} where the result has {result}. "
        f"{rest}Check this page on the invoice."
    )


# Transitional: the codes not yet given values, worded from the flow's message. A code
# leaves this table in the commit that gives its issues values.
ISSUE_TEXTS: dict[str, Parsed] = {
    # A page.
    "page_read": _rules(
        (r"Page (\d+) was not read: the time limit for the document passed\.", _not_read),
        (
            (
                r"Page (\d+) was not read to the end: the time limit for the document passed\. "
                r"The (\d+) line items? read before that may be incomplete\."
            ),
            _not_read_to_the_end,
        ),
        (r"(?s:.*)", lambda m, i, line: _page_failed(i)),
    ),
    "page_empty": _always(
        lambda i: (
            f"No line items were read from {_page(i)}. If the page has line items, they "
            "are missing: check it."
        )
    ),
    "rows_dropped": _rules(
        (r"(\d+) row\(s\) of the answer have neither sku nor amount and were left out", _dropped)
    ),
    "attempts_disagree": _rules(
        (r"attempts returned different numbers of lines: (\d+(?: and \d+)+)", _disagree)
    ),
    # A PDF whose rows were taken from its text by code.
    "layout_row": _rules(
        (r"stray_words: (.*)", _row("stray_words")),
        (r"no_tax_rate: (.*)", _row("no_tax_rate")),
        (r"text_under_item: (.*)", _row("text_under_item")),
    ),
    "layout_swapped": _rules(
        (
            (
                r"rows found by code and rows read by the model disagree on page (\d+): quantity "
                r"and unit price are crossed \((\d+) of (\d+) agree, (\d+) crossed\); the document "
                r"was read page by page by the model"
            ),
            _crossed,
        ),
    ),
    "layout_mismatch": _rules(
        (
            (
                r"rows found by code and rows read by the model disagree on page (\d+): the rows "
                r"differ \((\d+) of (\d+) agree, (\d+) crossed\); the document was read page by "
                r"page by the model"
            ),
            _differ,
        ),
    ),
    "layout_unverified": _rules(
        (
            (
                r"the check read of page (\d+) was stopped by the time limit for the document, so "
                r"the rows found by code were not confirmed"
            ),
            _unconfirmed("was stopped by the time limit"),
        ),
        (
            (
                r"the model returned no rows for page (\d+), so nothing confirms which column is "
                r"quantity and which is unit price"
            ),
            _unconfirmed("returned no line items"),
        ),
    ),
    # A page of a scan read a second time, by another model (second_reading.py).
    "second_reading": _rules(
        (
            (
                r"(Position \d+|Line item \d+) on page (\d+): a second reading by another model "
                r"gives (.*)\. One of the two readings is wrong; please check this line\."
            ),
            _read_twice_row,
        ),
        (
            (
                r"On page (\d+) a second reading by another model gives a different (.+?) on "
                r"(\d+) of (\d+) line items, for example at (.+?): (.*)\. The two readings "
                r"write this column differently, or one of them misread it; please check the "
                r".+ column on this page\."
            ),
            _read_twice_column,
        ),
        (
            (
                r"Page (\d+): a second reading by another model finds (\d+ line items?); the "
                r"result has (\d+)\. (.*)Please check this page\."
            ),
            _read_twice_page,
        ),
        (
            (
                r"Page (\d+): a second reading by another model finds as many line items as "
                r"the result \((\d+)\), but not at the same positions\. (.*)Please check this "
                r"page\."
            ),
            lambda m, i, line: (
                f"On page {m[1]}, a second model finds as many line items as the result "
                f"({m[2]}), but not at the same positions. {m[3]}Check this page on the "
                "invoice."
            ),
        ),
    ),
    # The document as a whole.
    "unreadable_file": _always(lambda i: JOB_ERRORS["unreadable_file"]),
    "deadline_exceeded": _rules(
        (r"Reading stopped at the time limit of ([\d.]+) s for one document\. (.*)", _deadline)
    ),
    # A photo, measured before it is read.
    "photo_blurred": _rules((r"Text sharpness (\S+) is below (\S+)\..*", _blurred_photo)),
    "photo_no_text": _always(
        lambda i: (
            "No text was found in the photo, so nothing was read. Check that the photo "
            "shows the invoice and is close enough to read."
        )
    ),
    "photo_text_at_edge": _rules(
        (r"Text reaches the edge of the frame(?: \((\w+) side\))?, .*", _edge)
    ),
    "photo_sheet_cut": _rules(
        (
            r"The sheet runs out of the frame(?: \((\w+) side\))?; no text is near that edge\.",
            _sheet_cut,
        )
    ),
    "photo_sheet_edge_unseen": _always(
        lambda i: (
            "No edge of the sheet is visible in the photo, so nothing confirms that the "
            "whole invoice is in it. Check that no line or total is cut off."
        )
    ),
    "photo_dark": _rules(
        (r"Paper brightness (\S+) of 255 is below (\S+)\.", _dark(lambda i: "The photo"))
    ),
    "photo_glare": _rules((r"Glare covers ([\d.]+)% of the sheet\.", _glare)),
    "photo_text_small": _rules((r"Text lines are (\S+) px tall, below (\S+)\.", _small)),
    # A page of a scan, measured before it is read.
    "scan_blurred": _rules((r"Text sharpness (\S+) of the page is below (\S+)\..*", _blurred_page)),
    "scan_dark": _rules(
        (
            r"Paper brightness (\S+) of 255 is below (\S+)\.",
            _dark(lambda i: _page(i).capitalize()),
        )
    ),
    "scan_too_large": _rules(
        (
            (
                r"The images of the page have ([\d,]+) pixels, more than the ([\d,]+) that may be "
                r"opened\. The page was not sent to the model\."
            ),
            lambda m, i, line: (
                f"{_page(i).capitalize()} holds an image too large to open "
                f"({m[1]} pixels; at most {m[2]} can be opened), so it was not read. Save the scan "
                "at a lower resolution and upload it again."
            ),
        ),
    ),
    "scan_no_text": _always(
        lambda i: (
            "No page of the scan has any text, so nothing was read. Check that this is "
            "the right file."
        )
    ),
}


# The sentences that name no value, by (code, form). A check with several sentences has a
# form for each of them and no (code, None) key: an issue of it that has neither values
# nor a form has no sentence and keeps the flow's text.
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
        case _:
            assert_never(values)


def _new(issue: Issue) -> bool:
    """Transitional: the issue is worded from its values or form, never from its message."""
    return issue.values is not None or (issue.check, issue.form) in _PLAIN


def _parsed(issue: Issue, line: Line | None) -> str:
    """Transitional: the text from the flow's message, or the message itself."""
    word = ISSUE_TEXTS.get(issue.check)
    text = word(issue, line) if word is not None else None
    return text if text is not None else issue.message


def issue_text(issue: Issue, line: Line | None = None) -> str:
    """The message of an issue for a person; line: the line the issue is about, if any."""
    try:
        # Chosen before any wording runs: an issue with values or a form is never read
        # from its message, even when its wording fails.
        if _new(issue):
            return _worded(issue, line)
        return _parsed(issue, line)
    except Exception:  # fault boundary: a wording must never cost the document its result
        logger.warning(
            "issue %s (form %r, values %s) could not be worded",
            issue.check,
            issue.form,
            type(issue.values).__name__,
            exc_info=True,
        )
        return issue.message
