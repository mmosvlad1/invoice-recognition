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
class RepeatedNumbers:
    """Position numbers printed on more than one line: the first ten, in order."""

    codes: ClassVar[frozenset[str]] = frozenset({"numbering"})
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
class PageCoverage:
    """Rows that look like line items printed on the issue's page, and the lines read."""

    codes: ClassVar[frozenset[str]] = frozenset({"coverage"})
    printed: int
    read: int


@dataclass(frozen=True, slots=True)
class DocumentCoverage:
    """Rows that look like line items printed in the document, and the lines read."""

    codes: ClassVar[frozenset[str]] = frozenset({"coverage"})
    printed: int
    read: int


type IssueValues = (
    SubtotalOff
    | TotalOff
    | LineProduct
    | TaxNoRow
    | TaxRateOff
    | TaxTotalOff
    | RepeatedNumbers
    | MissingNumbers
    | ControlTotalOff
    | HeaderNotPrinted
    | PageCoverage
    | DocumentCoverage
)

# Which of a check's sentences that name no value an issue is; only for a check that has
# more than one sentence.
type IssueForm = Literal[
    "supplier",  # doc_sums: the supplier's name was not read
    "no_lines",  # doc_sums: no line items
    "totals_missing",  # doc_sums: a total of the header was not read (which: the fields)
    "amounts_missing",  # doc_sums: some lines have no amount
    "unusable",  # line_math: a value of the line is missing or not a number
]
