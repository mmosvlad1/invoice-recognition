"""Two readings of one scanned page, by two different models, compared row by row.

On a scan nothing checks an item code (SKU): there is no text layer to hold it against
and no arithmetic involves it, so a misread code passes every check. A second reading of
the page by a model of another provider is a different reader: where the two disagree on
a line item, one of them misread it, and a person is asked to look. Measured on two
synthetic scans (15 pages, 468 line items): all five misreads of the first model were
shown by a disagreement, with one false alarm on the 463 items it read correctly. Two
models that misread a code the same way are not caught.

The first reading stays the result. A disagreement is an issue of severity "warning" that
names the cells and offers the second reader's values as a suggestion; no line is
changed.

What is compared: the SKU, quantity, unit price, tax rate and amount.
- SKUs ignore letter case, and a run of whitespace inside counts as one space - the rule
  that was measured. Every other character counts: "SG-54066" is not "SG54066", "AB-12" is
  not "A-B12" (a dropped dash is a misread that breaks a lookup). Added after the control
  run and not measured on new documents: characters that are not letters or digits
  around the code are ignored ("*987123*" = "987123"). A SKU is text or a whole number
  (12345.0 is 12345); anything else is no SKU.
- Numbers are compared by value, whichever way they are written: 20 = 20.0 = "20%",
  "1,234.50" = "1 234.50" = "1.234,50" = 1234.5, "10,5" = 10.5, a Unicode minus or an en
  dash as "-". A number too large or too precise for an invoice (checks.dec), a flag, a
  list or text that is no number counts as missing.
- Two tax rates also agree when one is a bare number strictly between 0 and 1 that is
  the other divided by 100 (0.2 and 20); a rate written with "%" is never read as a
  fraction ("0.5%" is 0.5).
- Two missing values agree; a missing value and a read one differ.
Not compared: the description (a description cut by a page break is joined in the first
reading only), the unit and control totals.

When one field differs on most of the rows of a page - on at least COLUMN_ROWS rows and
on more than half of the rows compared - the two readers read that column differently
(one left it empty, or writes it otherwise): one issue for the page and the column, not
one per row. For the SKU only rows whose codes differ in spelling alone (the same letters
and digits) count towards that; codes that differ in a letter or digit are misreads, and
each keeps its own issue with its suggestion.

Pure functions: no request, no I/O.
"""

import re
from dataclasses import dataclass
from decimal import Decimal

from invoice_recognition.recognition import checks
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.issue_values import (
    Difference,
    SecondColumn,
    SecondPage,
    SecondRow,
)
from invoice_recognition.shapes import Line

FIELDS = ("sku", "quantity", "unit_price", "tax_rate", "amount")
_LABELS = {
    "sku": "SKU",
    "quantity": "quantity",
    "unit_price": "unit price",
    "tax_rate": "tax rate",
    "amount": "amount",
}
# A field that differs on at least this many rows, and on more than half of the rows
# compared, is reported once for the column.
COLUMN_ROWS = 3

# Written for a minus before a number: the minus sign and the en dash.
_MINUS = ("\u2212", "\u2013")
_NUMBER = re.compile(r"-?\d[\d.,]*")
_SHOWN_SKU = 40  # characters of a SKU quoted in a message


# --------------------------------------------------------------------------- #
# One value
# --------------------------------------------------------------------------- #


def _sku_text(v: object) -> str | None:
    """An item code as written, inner whitespace collapsed; None when there is none. A
    whole number written as a float (12345.0) is that number."""
    if isinstance(v, bool):
        return None
    if isinstance(v, float):
        if not v.is_integer():
            return None
        v = int(v)
    if not isinstance(v, str | int):
        return None
    return " ".join(str(v).split()) or None


def _trimmed(text: str) -> str:
    start, end = 0, len(text)
    while start < end and not text[start].isalnum():
        start += 1
    while end > start and not text[end - 1].isalnum():
        end -= 1
    return text[start:end]


def sku(v: object) -> str | None:
    """An item code as it is compared, None when there is none."""
    text = _sku_text(v)
    return None if text is None else (_trimmed(text.upper()) or None)


def _spelled(v: object) -> str | None:
    """An item code's letters and digits alone: two codes that differ only in spelling
    (dashes, spaces, dots) have the same."""
    key = sku(v)
    return None if key is None else "".join(c for c in key if c.isalnum())


def _text_number(text: str) -> Decimal | None:
    """A number written as text, in either convention of separators."""
    t = "".join(c for c in text.strip().removesuffix("%") if not c.isspace() and c != "'")
    for minus in _MINUS:
        t = t.replace(minus, "-")
    if not _NUMBER.fullmatch(t):
        return None
    if "," in t and "." in t:  # the one that comes last separates the decimals
        thousands = "," if t.rindex(".") > t.rindex(",") else "."
        t = t.replace(thousands, "").replace(",", ".")
    elif "," in t:
        head, _, tail = t.partition(",")
        # One comma: decimals unless exactly three digits follow a number that does not
        # start with 0 ("1,234" is a thousand; "10,5" and "0,500" are not).
        decimal = t.count(",") == 1 and (len(tail) != 3 or head.lstrip("-") == "0")
        t = t.replace(",", ".") if decimal else t.replace(",", "")
    elif t.count(".") > 1:  # "1.234.567"
        t = t.replace(".", "")
    return checks.dec(t)


def number(field: str, v: object) -> Decimal | None:
    """A number of a line item as it is compared, None when there is none."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        d = checks.dec(v)
    elif isinstance(v, str):
        d = _text_number(v)
    else:
        return None
    return d


def _fraction(v: object) -> Decimal | None:
    """A tax rate written as a bare fraction (0.2 for 20 %): a number strictly between 0
    and 1 not written with "%". None for anything else."""
    if isinstance(v, str) and v.strip().endswith("%"):
        return None
    d = number("tax_rate", v)
    return d if d is not None and 0 < d < 1 else None


def same(field: str, first: object, second: object) -> bool:
    """Whether the two readers read the same value for one field of a line item."""
    if field == "sku":
        return sku(first) == sku(second)
    a, b = number(field, first), number(field, second)
    if a == b:
        return True
    if field != "tax_rate" or a is None or b is None:
        return False
    fa, fb = _fraction(first), _fraction(second)
    return (fa is not None and fa * 100 == b) or (fb is not None and fb * 100 == a)


def _shown(field: str, v: object) -> str | None:
    """A value as a message quotes it; None when there is none."""
    if field == "sku":
        text = _sku_text(v) if sku(v) is not None else None
        if text is None:
            return None
        return text if len(text) <= _SHOWN_SKU else text[: _SHOWN_SKU - 1] + "…"
    d = number(field, v)
    if d is None:
        return None
    text = f"{d.normalize():f}"
    # A rate written as a fraction is shown as it is, not as a percentage it is not.
    return f"{text}%" if field == "tax_rate" and _fraction(v) is None else text


def _suggested(field: str, v: object) -> str | int | float | None:
    """The second reader's value as a cell would take it: the SKU as text, a number as a
    JSON number; None when it read none."""
    if field == "sku":
        return _sku_text(v) if sku(v) is not None else None
    d = number(field, v)
    return None if d is None else checks.plain(d)


# --------------------------------------------------------------------------- #
# Which line is which
# --------------------------------------------------------------------------- #


def _numbers(lines: list[Line]) -> list[int] | None:
    """The position numbers of the lines, when every line has its own integer number."""
    nos = [item.get("line_no") for item in lines]
    ints = [n for n in nos if isinstance(n, int) and not isinstance(n, bool)]
    return ints if len(ints) == len(nos) and len(set(ints)) == len(ints) else None


@dataclass
class Alignment:
    """Which line of one reading is which line of the other.

    pairs: (index in the first reading, index in the second). only_first / only_second:
    indices of lines the other reading has no counterpart for. by: "line_no" - by the
    printed position number; "order" - top to bottom; "none" - the lines cannot be
    matched, nothing is paired."""

    pairs: list[tuple[int, int]]
    only_first: list[int]
    only_second: list[int]
    by: str


def align(first: list[Line], second: list[Line]) -> Alignment:
    """By position number when each reader gives every line of the page its own integer
    number and the two have at least one number in common. Numbers with nothing in common
    say nothing about each other - typically one reader counts the rows of the page 1, 2,
    ..., k while the other gives the printed positions 79, 80, ... - and are treated as no
    numbers.

    Otherwise by order when both read as many lines, otherwise not at all: no number on
    one side, the numbers of only one side, a number used twice, or one row without a
    number (even a single one) all mean order with equal counts and no pairing at all
    with different counts."""
    a, b = _numbers(first), _numbers(second)
    if a is not None and b is not None and not (a and b and set(a).isdisjoint(b)):
        at = {no: j for j, no in enumerate(b)}
        pairs = [(i, at[no]) for i, no in enumerate(a) if no in at]
        common = set(a) & set(b)
        return Alignment(
            pairs,
            [i for i, no in enumerate(a) if no not in common],
            [j for j, no in enumerate(b) if no not in common],
            "line_no",
        )
    if len(first) == len(second):
        return Alignment([(i, i) for i in range(len(first))], [], [], "order")
    return Alignment([], list(range(len(first))), list(range(len(second))), "none")


# --------------------------------------------------------------------------- #
# What a disagreement says
# --------------------------------------------------------------------------- #


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _difference(field: str, first: object, second: object) -> Difference:
    return Difference(field, _shown(field, second), _shown(field, first))


def _said(d: Difference) -> str:
    label = _LABELS[d.field]
    if d.theirs is None:
        return f"no {label} where the result has {d.ours}"
    if d.ours is None:
        return f"{label} {d.theirs} where the result has none"
    return f"{label} {d.theirs} instead of {d.ours}"


def _where(first: Line, index: int) -> tuple[str, int | None]:
    """ "position 12" (its printed number) or "line item 3" (from the top of the page)."""
    no = first.get("line_no")
    if isinstance(no, int) and not isinstance(no, bool):
        return f"position {no}", no
    return f"line item {index + 1}", None


def _row_issue(first: Line, second: Line, index: int, page: int, differ: list[str]) -> Issue:
    where, line_no = _where(first, index)
    place = f"{where.capitalize()} on page {page}"
    differences = tuple(_difference(f, first.get(f), second.get(f)) for f in differ)
    found = _and([_said(d) for d in differences])
    values = {f"line.{f}": _suggested(f, second.get(f)) for f in differ}
    # Only values the second reader did read can be written into a cell.
    suggested = {k: v for k, v in values.items() if v is not None}
    return Issue(
        "second_reading",
        f"{place}: a second reading by another model gives {found}. One of the two readings "
        "is wrong; please check this line.",
        line=index,
        page=page,
        severity="warning",
        line_no=line_no,
        fields=[f"line.{f}" for f in differ],
        suggestions=[{"set": suggested, "source": "second_reading"}] if suggested else [],
        values=SecondRow(index + 1, differences),
    )


def _column_issue(
    field: str, rows: list[tuple[Line, Line, int]], compared: int, page: int
) -> Issue:
    """rows: (first, second, index in the first reading) of every row that differs."""
    first, second, index = rows[0]
    where, line_no = _where(first, index)
    label = _LABELS[field]
    example = _difference(field, first.get(field), second.get(field))
    return Issue(
        "second_reading",
        f"On page {page} a second reading by another model gives a different {label} on "
        f"{len(rows)} of {compared} line items, for example at {where}: {_said(example)}. The "
        f"two readings write this column differently, or one of them misread it; please check "
        f"the {label} column on this page.",
        page=page,
        severity="warning",
        fields=[f"line.{field}"],
        values=SecondColumn(len(rows), compared, index + 1, line_no, example),
    )


def _items(n: int) -> str:
    return f"{n} line item" if n == 1 else f"{n} line items"


def _page_issue(first: list[Line], second: list[Line], found: Alignment, page: int) -> Issue:
    if len(first) == len(second):  # as many lines, at other positions
        counts = (
            f"Page {page}: a second reading by another model finds as many line items as the "
            f"result ({len(first)}), but not at the same positions."
        )
    else:
        counts = (
            f"Page {page}: a second reading by another model finds {_items(len(second))}; "
            f"the result has {len(first)}."
        )
    text = [counts]
    extra: list[int] = []  # position numbers only the second reading has
    absent: list[int] = []  # position numbers only the result has
    if found.by == "none":
        text.append("The lines could not be matched one to one, so they were not compared.")
    else:
        extra = [second[j]["line_no"] for j in found.only_second]
        absent = [first[i]["line_no"] for i in found.only_first]
        if extra:
            are = "is" if len(extra) == 1 else "are"
            text.append(
                f"Position{'s' * (len(extra) > 1)} {checks.ranges(extra)} {are} not in the result."
            )
        if absent:
            were = "was" if len(absent) == 1 else "were"
            text.append(
                f"Position{'s' * (len(absent) > 1)} {checks.ranges(absent)} of the result "
                f"{were} not found by the second reading."
            )
    text.append("Please check this page.")
    return Issue(
        "second_reading",
        " ".join(text),
        page=page,
        severity="warning",
        values=SecondPage(len(second), len(first), found.by != "none", tuple(extra), tuple(absent)),
    )


def compare_page(first: list[Line], second: list[Line], page: int) -> list[Issue]:
    """The issues of one page read twice, in this order: one per line item the readers
    disagree on, in the order of the first reading; one per column that differs on most
    rows (its rows then raise nothing for that field); at most one for the page when
    lines of one reading have no counterpart in the other. `line` indexes `first` (the
    page's own lines). Both readings are expected as the flow keeps a page: rows that are
    line items, with the quantity repair already applied."""
    found = align(first, second)
    differ = {
        (i, j): [f for f in FIELDS if not same(f, first[i].get(f), second[j].get(f))]
        for i, j in found.pairs
    }
    compared = len(found.pairs)
    columns: dict[str, list[tuple[int, int]]] = {}
    for f in FIELDS:
        pairs = [pair for pair, fields in differ.items() if f in fields]
        if f == "sku":  # a misread letter or digit is never folded into the column
            pairs = [
                (i, j)
                for i, j in pairs
                if _spelled(first[i].get(f)) is not None
                and _spelled(first[i].get(f)) == _spelled(second[j].get(f))
            ]
        if len(pairs) >= COLUMN_ROWS and 2 * len(pairs) > compared:
            columns[f] = pairs
    issues = []
    for (i, j), fields in differ.items():
        rest = [f for f in fields if (i, j) not in columns.get(f, [])]
        if rest:
            issues.append(_row_issue(first[i], second[j], i, page, rest))
    issues += [
        _column_issue(f, [(first[i], second[j], i) for i, j in pairs], compared, page)
        for f, pairs in columns.items()
    ]
    if found.only_first or found.only_second:
        issues.append(_page_issue(first, second, found, page))
    return issues
