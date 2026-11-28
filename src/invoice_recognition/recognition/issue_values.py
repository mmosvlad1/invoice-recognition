"""The values an issue's text for a person names, as the flow computed them.

The flow writes each issue's `message` for a developer. The text a person reads is worded
in api/messages.py from data instead: the issue's page, fields and line, and one of the
shapes below (Issue.values), set where the check computes the values from the same
variables as its message. A check with several sentences that name no value tells them
apart by a form (Issue.form).

Exact amounts stay Decimal, the very numbers the message prints; measured numbers and
limits stay float; counts and position numbers are int. `codes` names the issue codes a
shape may be attached to.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar, Literal

# --------------------------------------------------------------------------- #
# The document's totals (checks.check_doc_sums)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SubtotalOff:
    """The line amounts against the invoice subtotal."""

    codes: ClassVar[frozenset[str]] = frozenset({"doc_sums"})
    lines: Decimal
    subtotal: Decimal
    difference: Decimal


@dataclass(frozen=True, slots=True)
class TotalOff:
    """Subtotal plus tax against the invoice total."""

    codes: ClassVar[frozenset[str]] = frozenset({"doc_sums"})
    subtotal: Decimal
    tax: Decimal
    total: Decimal


# --------------------------------------------------------------------------- #
# One line, the tax, the numbering, the control totals
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LineProduct:
    """Quantity x unit price against the amount of a line; thousands: the quantity looks
    1000 times too small."""

    codes: ClassVar[frozenset[str]] = frozenset({"line_math"})
    quantity: Decimal
    unit_price: Decimal
    product: Decimal
    amount: Decimal
    thousands: bool


@dataclass(frozen=True, slots=True)
class TaxNoRow:
    """Lines carry a rate the tax summary has no row for; base: what they add up to."""

    codes: ClassVar[frozenset[str]] = frozenset({"tax"})
    rate: Decimal
    base: Decimal


@dataclass(frozen=True, slots=True)
class TaxRateOff:
    """The lines at a rate against the base the tax summary prints for it."""

    codes: ClassVar[frozenset[str]] = frozenset({"tax"})
    rate: Decimal
    lines: Decimal
    printed: Decimal


@dataclass(frozen=True, slots=True)
class TaxTotalOff:
    """Without a tax summary: the tax computed from the lines against the tax total."""

    codes: ClassVar[frozenset[str]] = frozenset({"tax"})
    computed: Decimal
    printed: Decimal


@dataclass(frozen=True, slots=True)
class RateUnreadable:
    """A row of the tax summary whose rate is not a number; printed: its base."""

    codes: ClassVar[frozenset[str]] = frozenset({"tax"})
    printed: Decimal


@dataclass(frozen=True, slots=True)
class RepeatedNumbers:
    """Position numbers printed on more than one line: how many, and the first ten."""

    codes: ClassVar[frozenset[str]] = frozenset({"numbering"})
    count: int
    numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MissingNumbers:
    """Position numbers missing from the sequence: how many, and the first ten."""

    codes: ClassVar[frozenset[str]] = frozenset({"numbering"})
    count: int
    numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ControlTotalOff:
    """Line items first-last (position in the document, from 1) add up to `lines`; the
    control total printed after them, labelled `label`, says `printed`."""

    codes: ClassVar[frozenset[str]] = frozenset({"checkpoints"})
    first: int
    last: int
    lines: Decimal
    printed: Decimal
    label: str


# --------------------------------------------------------------------------- #
# What is printed in the PDF's own text
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HeaderNotPrinted:
    """A total or an identifier of the header not printed anywhere: the total as read, the
    identifier as compared (upper-cased). Which one: the issue's field."""

    codes: ClassVar[frozenset[str]] = frozenset({"grounding_header"})
    value: Decimal | str


@dataclass(frozen=True, slots=True)
class Coverage:
    """Rows that look like line items printed, and the lines read: on the issue's page, or
    in the whole document when the issue has no page."""

    codes: ClassVar[frozenset[str]] = frozenset({"coverage"})
    printed: int
    read: int


# --------------------------------------------------------------------------- #
# A page, and a PDF whose rows were found by code (pipeline.py)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CutByDeadline:
    """A page no attempt at which ran to its end; lines: the line items kept from a cut
    attempt (at least one: a page with none is the form "not_read")."""

    codes: ClassVar[frozenset[str]] = frozenset({"page_read"})
    lines: int


@dataclass(frozen=True, slots=True)
class RowsDropped:
    """Rows of a page's answer with neither an item code nor an amount, left out."""

    codes: ClassVar[frozenset[str]] = frozenset({"rows_dropped"})
    rows: int


@dataclass(frozen=True, slots=True)
class AttemptCounts:
    """The line items each clean reading of a page found, in the order read."""

    codes: ClassVar[frozenset[str]] = frozenset({"attempts_disagree"})
    counts: tuple[int, ...]


# What the parser of a PDF's text could not place, in a row it did place.
LayoutRowKind = Literal["stray_words", "no_tax_rate", "text_under_item"]


@dataclass(frozen=True, slots=True)
class LayoutRow:
    """A row the parser of the PDF's text could not place fully: what is wrong with it,
    and the row as printed."""

    codes: ClassVar[frozenset[str]] = frozenset({"layout_row"})
    kind: LayoutRowKind
    row: str


@dataclass(frozen=True, slots=True)
class RowsCompared:
    """The rows found by code against the same page read by the model: how many agree,
    how many were compared, and in how many quantity and unit price are crossed."""

    codes: ClassVar[frozenset[str]] = frozenset({"layout_swapped", "layout_mismatch"})
    same: int
    compared: int
    crossed: int


@dataclass(frozen=True, slots=True)
class DeadlineStop:
    """The document's time limit passed: the limit, the pages not read, and whether the
    header was read."""

    codes: ClassVar[frozenset[str]] = frozenset({"deadline_exceeded"})
    seconds: float
    unread: tuple[int, ...]
    header_read: bool


# --------------------------------------------------------------------------- #
# A photo or a page of a scan, measured before it is read (pipeline._photo_issue,
# _scan_issue, _too_large_issue)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Measured:
    """A number measured on the image, and the limit it was held against."""

    codes: ClassVar[frozenset[str]] = frozenset(
        {"photo_blurred", "photo_dark", "photo_text_small", "scan_blurred", "scan_dark"}
    )
    value: float
    limit: float


@dataclass(frozen=True, slots=True)
class FrameSide:
    """The side of the photo the sheet or its text runs into, when known."""

    codes: ClassVar[frozenset[str]] = frozenset({"photo_text_at_edge", "photo_sheet_cut"})
    side: str | None


@dataclass(frozen=True, slots=True)
class Glare:
    """The share of the sheet covered by glare, from 0 to 1."""

    codes: ClassVar[frozenset[str]] = frozenset({"photo_glare"})
    share: float


@dataclass(frozen=True, slots=True)
class TooLarge:
    """A page whose images have more pixels together than may be opened."""

    codes: ClassVar[frozenset[str]] = frozenset({"scan_too_large"})
    pixels: int
    limit: int


# --------------------------------------------------------------------------- #
# A page of a scan read a second time, by another model (second_reading.py)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Difference:
    """One field the two readings of a line disagree on: the key of the line ("sku",
    "quantity", ...) and each reading's value as the flow quotes it, None for none."""

    field: str
    theirs: str | None
    ours: str | None


@dataclass(frozen=True, slots=True)
class SecondRow:
    """A line item the second reading reads otherwise: its place on the page, counted
    from 1 (the issue's line_no when the page numbers its lines), and what differs."""

    codes: ClassVar[frozenset[str]] = frozenset({"second_reading"})
    item: int
    differences: tuple[Difference, ...]


@dataclass(frozen=True, slots=True)
class SecondColumn:
    """A field the second reading reads otherwise on most line items of a page: on how
    many of those compared, and the first of them (its place on the page, counted from 1,
    and its position number if printed) as an example."""

    codes: ClassVar[frozenset[str]] = frozenset({"second_reading"})
    differ: int
    compared: int
    item: int
    line_no: int | None
    difference: Difference


@dataclass(frozen=True, slots=True)
class SecondPage:
    """The line items of a page that have no counterpart in the other reading: how many
    each reading found, whether the lines could be matched at all, and the position
    numbers only the second reading has (extra) or only the result has (absent)."""

    codes: ClassVar[frozenset[str]] = frozenset({"second_reading"})
    found: int
    result: int
    matched: bool
    extra: tuple[int, ...]
    absent: tuple[int, ...]


type IssueValues = (
    SubtotalOff
    | TotalOff
    | LineProduct
    | TaxNoRow
    | TaxRateOff
    | TaxTotalOff
    | RateUnreadable
    | RepeatedNumbers
    | MissingNumbers
    | ControlTotalOff
    | HeaderNotPrinted
    | Coverage
    | CutByDeadline
    | RowsDropped
    | AttemptCounts
    | LayoutRow
    | RowsCompared
    | DeadlineStop
    | Measured
    | FrameSide
    | Glare
    | TooLarge
    | SecondRow
    | SecondColumn
    | SecondPage
)

# Which of a check's sentences that name no value an issue is; only for a check that has
# more than one sentence.
type IssueForm = Literal[
    "supplier",  # doc_sums: the supplier's name was not read
    "no_lines",  # doc_sums: no line items
    "totals_missing",  # doc_sums: a total of the header was not read (which: the fields)
    "amounts_missing",  # doc_sums: some lines have no amount
    "unusable",  # line_math: a value of the line is missing or not a number
    "failed",  # page_read: the answer was broken, every request failed, or reading broke
    "not_read",  # page_read: the time limit passed before anything of the page was read
    "deadline",  # layout_unverified: the control reading was stopped by the time limit
    "no_rows",  # layout_unverified: the control reading returned no rows
]
