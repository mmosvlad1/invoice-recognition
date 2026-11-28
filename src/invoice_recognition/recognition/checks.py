"""Deterministic checks on an extraction. No model is asked whether it is sure.

Each check answers a different question and, where it can, says WHERE the
problem is (line, page), so the flow can re-read one page instead of the
whole document and the user can be shown the doubtful rows - and which fields
of that row or of the header are in doubt (Issue.fields). Where arithmetic
determines it, an issue also carries the values that would settle it
(Issue.suggestions).

Expected shape of `doc`:
    header:      {supplier_name, subtotal, tax_total, total,
                  tax_breakdown?: [{rate, base, tax}], ...}
    lines:       [{sku, description, quantity, unit_price, tax_rate, amount, line_no?, page?}]
    checkpoints: [{amount, after_line, label?, page?}]   control totals printed in the
                 document (page carry-forwards, section subtotals); after_line = how many
                 line items precede it.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, cast

from invoice_recognition.recognition.evidence import CENT, Evidence, money, printed_forms
from invoice_recognition.recognition.issue_values import (
    ControlTotalOff,
    Coverage,
    HeaderNotPrinted,
    IssueForm,
    IssueValues,
    LineProduct,
    MissingNumbers,
    RateUnreadable,
    RepeatedNumbers,
    SubtotalOff,
    TaxNoRow,
    TaxRateOff,
    TaxTotalOff,
    TotalOff,
)
from invoice_recognition.shapes import Checkpoint, Doc, Header, JSONObject, Line


@dataclass
class Issue:
    check: str
    message: str
    line: int | None = None  # 0-based index into doc["lines"]
    page: int | None = None
    severity: str = "error"
    # Position number printed in front of that line, when the document numbers its lines.
    line_no: int | None = None
    # What is in doubt, as qualified names: "line.<key>" (a key of a line item) or
    # "header.<key>". With `line` set, "line.x" is that cell of that line; without it,
    # "line.x" is the whole column, never every row of it. [] - the page or the file as a
    # whole, not any cell.
    fields: list[str] = field(default_factory=list)
    # Values that would settle the issue, each an alternative to the others:
    # {"set": {<field>: <value>, ...}, "source": ...}. Usually none.
    suggestions: list[JSONObject] = field(default_factory=list)
    # Positions in Result.issues of the issues that explain this one: the same mistake,
    # seen by another check.
    follows_from: list[int] = field(default_factory=list)
    # What the text a person reads (api/messages.py) needs: which of the check's sentences
    # that name no value this is, or the values its sentence names. Set where the values
    # are computed. Not in to_json(): the flow's record keeps `message`, its words for a
    # developer.
    form: IssueForm | None = None
    values: IssueValues | None = field(default=None, repr=False)

    def to_json(self) -> JSONObject:
        # From the fields, not asdict(): that would copy form and values too.
        out = {
            f.name: deepcopy(getattr(self, f.name)) for f in fields(self) if f.name not in _WORDING
        }
        # Left out when unknown or empty: most issues are not about a numbered line and
        # have nothing to suggest, and their JSON then has only the keys every issue has.
        if out["line_no"] is None:
            del out["line_no"]
        for key in ("suggestions", "follows_from"):
            if not out[key]:
                del out[key]
        return out


# The fields of an Issue that only the text for a person reads.
_WORDING = ("form", "values")

# No figure on an invoice needs a decimal exponent beyond this, either way.
_MAX_EXPONENT = 100


def dec(v: object) -> Decimal | None:
    """A number from the model's JSON: a number or a string ("20%" included).
    None if it is not one.

    NaN and infinity are None too: Python's JSON parser accepts them, and a
    comparison with a NaN Decimal raises, which would crash every check.

    So is a number of absurd size, such as "1e999999" or "1e-999999999": a product
    of two of them overflows the decimal context (decimal.Overflow), and printing
    one in full, as the presence checks do, takes as many digits as its exponent.
    A value of 10^101 or more, or one with more than 100 digits after the point, is
    not an invoice figure."""
    if v is None or isinstance(v, bool):
        return None
    try:
        d = Decimal(str(v).strip().rstrip("%"))
    except InvalidOperation, ValueError:
        return None
    if not d.is_finite():
        return None
    # The exponent is an int for every finite number ("n", "N" or "F" otherwise).
    exponent = cast(int, d.as_tuple().exponent)
    if d.adjusted() > _MAX_EXPONENT or exponent < -_MAX_EXPONENT:
        return None
    return d


def plain(x: Decimal) -> int | float:
    """A number as JSON writes it: an int when it is whole, else a float."""
    return int(x) if x == x.to_integral_value() else float(x)


def ranges(numbers: Iterable[int]) -> str:
    """Numbers for a message, runs joined: [2, 5, 6, 7, 12] -> "2, 5–7 and 12"."""
    runs: list[list[int]] = []
    for n in sorted(numbers):
        if runs and n == runs[-1][-1] + 1:
            runs[-1].append(n)
        else:
            runs.append([n])
    parts = [f"{r[0]}–{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs]
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def _lines(doc: Doc) -> list[Line]:
    return [item for item in (doc.get("lines") or []) if isinstance(item, dict)]


# --------------------------------------------------------------------------- #
# 1. Document-level totals
# --------------------------------------------------------------------------- #


def check_doc_sums(doc: Doc) -> list[Issue]:
    return list(_doc_sums(doc).values())


def _lines_sum(lines: list[Line]) -> Decimal:
    """The sum of the amounts of the lines; a line without one counts as nothing."""
    amounts = (dec(item.get("amount")) for item in lines)
    return sum((a for a in amounts if a is not None), Decimal(0))


def _doc_sums(doc: Doc) -> dict[str, Issue]:
    """The issues of check_doc_sums, each under the name of what it verifies."""
    out: dict[str, Issue] = {}
    header: Header = doc.get("header") or {}
    lines = _lines(doc)
    supplier = header.get("supplier_name")
    # A number, list or object is not a supplier name.
    if not (isinstance(supplier, str) and supplier.strip()):
        out["supplier"] = Issue(
            "doc_sums", "supplier is missing", fields=["header.supplier_name"], form="supplier"
        )
    if not lines:
        return out | {"lines": Issue("doc_sums", "no line items", form="no_lines")}
    subtotal, tax, total = (dec(header.get(k)) for k in ("subtotal", "tax_total", "total"))
    if subtotal is None or tax is None or total is None:
        missing = [
            f"header.{k}"
            for k, v in (("subtotal", subtotal), ("tax_total", tax), ("total", total))
            if v is None
        ]
        return out | {
            "totals": Issue(
                "doc_sums",
                "totals are missing in the header",
                fields=missing,
                form="totals_missing",
            )
        }
    if any(dec(item.get("amount")) is None for item in lines):
        # One issue for every line without an amount: the column, not each row.
        out["amounts"] = Issue(
            "doc_sums",
            "some lines have no amount",
            fields=["line.amount"],
            form="amounts_missing",
        )
    lines_sum = _lines_sum(lines)
    difference = lines_sum - subtotal
    if abs(difference) > CENT:
        out["subtotal"] = Issue(
            "doc_sums",
            f"lines add up to {lines_sum}, invoice subtotal is {subtotal} "
            f"(difference {difference})",
            fields=["header.subtotal", "line.amount"],
            values=SubtotalOff(lines_sum, subtotal, difference),
        )
    if abs(subtotal + tax - total) > CENT:
        out["total"] = Issue(
            "doc_sums",
            f"subtotal {subtotal} + tax {tax} != total {total}",
            fields=["header.subtotal", "header.tax_total", "header.total"],
            values=TotalOff(subtotal, tax, total),
        )
    return out


# --------------------------------------------------------------------------- #
# 2. Arithmetic inside one line
# --------------------------------------------------------------------------- #


def check_line_math(doc: Doc) -> list[Issue]:
    out = []
    for i, item in enumerate(_lines(doc)):
        q, p, a = dec(item.get("quantity")), dec(item.get("unit_price")), dec(item.get("amount"))
        if q is None or p is None or a is None:
            missing = [
                f"line.{k}"
                for k, v in (("quantity", q), ("unit_price", p), ("amount", a))
                if v is None
            ]
            out.append(
                Issue(
                    "line_math",
                    "quantity, unit price or amount is missing or is not a usable number",
                    line=i,
                    page=item.get("page"),
                    fields=missing,
                    # Only the amount is missing: quantity x unit price is what it would be.
                    suggestions=(
                        [_suggest("line.amount", money(q * p))]
                        if a is None and q is not None and p is not None
                        else []
                    ),
                    form="unusable",
                )
            )
        elif abs((product := money(q * p)) - a) > CENT:
            hint = ""
            # Arithmetic cannot say which of the three numbers is wrong; the hint can.
            doubt = ["line.quantity", "line.unit_price", "line.amount"]
            thousands = abs(money(q * 1000 * p) - a) <= CENT
            if thousands:
                hint = (
                    " (quantity looks 1000x too small: thousands separator read as a decimal point)"
                )
                doubt = ["line.quantity"]
            out.append(
                Issue(
                    "line_math",
                    f"{q} x {p} = {product}, but amount is {a}{hint}",
                    line=i,
                    page=item.get("page"),
                    fields=doubt,
                    suggestions=_line_alternatives(q, p, a, doubt),
                    values=LineProduct(q, p, product, a, thousands),
                )
            )
    return out


# A quantity is not printed with more decimals than this (grams of a kilogram).
_QUANTITY_DECIMALS = 3


def _suggest(name: str, value: Decimal) -> JSONObject:
    return {"set": {name: plain(value)}, "source": "arithmetic"}


def _decimals(x: Decimal) -> int:
    """How many digits `x` has after the point."""
    return max(0, -cast(int, x.as_tuple().exponent))


def _rounded(x: Decimal, places: int) -> Decimal | None:
    """`x` rounded half up to `places` decimals; None when that takes more digits than the
    28-digit precision of decimal arithmetic holds (quantize raises)."""
    try:
        return x.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None


def _first(tried: Iterable[Decimal | None], exact: Callable[[Decimal], bool]) -> Decimal | None:
    """The first value tried that is not zero and is exact."""
    return next((v for v in tried if v is not None and v != 0 and exact(v)), None)


def _line_alternatives(q: Decimal, p: Decimal, a: Decimal, doubt: list[str]) -> list[JSONObject]:
    """For each number in doubt, the value that makes quantity x unit price give the amount
    when the other two are right: alternatives, never a combined fix. Only exact values,
    whose product, rounded to cents, is the amount itself, and only plausible ones: a
    quantity of at most three decimals (a whole one when the quantity was read whole), a
    unit price in cents or with as many decimals as the price was read with, never zero.

    The quantity is the amount over the price rounded to 0, 1, 2 and then 3 decimals (it
    also finds the quantity whose thousands separator was read as a decimal point); the
    unit price is the amount over the quantity rounded to 2 decimals and then more, up to
    those of the price as read. The first that is exact is offered."""
    found: dict[str, Decimal] = {}
    if "line.quantity" in doubt and p != 0:
        places = range(1 if q == q.to_integral_value() else _QUANTITY_DECIMALS + 1)
        v = _first((_rounded(a / p, n) for n in places), lambda x: money(x * p) == a)
        if v is not None:
            found["line.quantity"] = v
    if "line.unit_price" in doubt and q != 0:
        places = range(2, max(2, _decimals(p)) + 1)
        v = _first((_rounded(a / q, n) for n in places), lambda x: money(q * x) == a)
        if v is not None:
            found["line.unit_price"] = v
    if "line.amount" in doubt:
        found["line.amount"] = money(q * p)
    return [_suggest(k, v) for k, v in found.items()]


# --------------------------------------------------------------------------- #
# 3. Tax per rate
# --------------------------------------------------------------------------- #


def check_tax(doc: Doc) -> list[Issue]:
    return list(_tax(doc).values())


def _tax(doc: Doc) -> dict[object, Issue]:
    """The issues of check_tax, each under the rate it is about ("total" without a
    breakdown)."""
    out: dict[object, Issue] = {}
    header: Header = doc.get("header") or {}
    lines = _lines(doc)
    by_rate: dict[Decimal, Decimal] = {}
    for item in lines:
        r, a = dec(item.get("tax_rate")), dec(item.get("amount"))
        if r is not None and a is not None:
            by_rate[r] = by_rate.get(r, Decimal(0)) + a
    breakdown = header.get("tax_breakdown") or []
    if not isinstance(breakdown, list):  # a number, string or object is not a breakdown
        breakdown = []
    if breakdown:
        printed = {
            dec(t.get("rate")): (dec(t.get("base")), dec(t.get("tax")))
            for t in breakdown
            if isinstance(t, dict)
        }
        # None: a printed rate that is not a number. Union, not {*a, *b}: the order of
        # a set decides the order of rates that sort equal (None and 0), hence of issues.
        rates = set[Decimal | None](by_rate) | set(printed)
        for rate in sorted(rates, key=lambda x: x or 0):
            base_lines = by_rate.get(rate, Decimal(0)) if rate is not None else Decimal(0)
            base_printed = (printed.get(rate) or (None, None))[0]
            if base_printed is None:
                # No line is at a rate that is not a number (base_lines is 0).
                if base_lines and rate is not None:
                    out[rate] = Issue(
                        "tax",
                        f"lines carry {rate}% tax on {base_lines}, "
                        f"but the invoice has no {rate}% row",
                        fields=["line.tax_rate", "header.tax_breakdown"],
                        values=TaxNoRow(rate, base_lines),
                    )
            elif abs(base_lines - base_printed) > CENT:
                out[rate] = Issue(
                    "tax",
                    f"lines at {rate}% add up to {base_lines}, invoice says {base_printed}",
                    fields=["line.tax_rate", "line.amount", "header.tax_breakdown"],
                    values=(
                        TaxRateOff(rate, base_lines, base_printed)
                        if rate is not None
                        else RateUnreadable(base_printed)
                    ),
                )
    else:
        tax_total = dec(header.get("tax_total"))
        expected = sum(
            (money(base * rate / 100) for rate, base in by_rate.items()),
            Decimal(0),
        )
        if tax_total is not None and abs(expected - tax_total) > Decimal("0.05"):
            out["total"] = Issue(
                "tax",
                f"tax computed from lines is {expected}, invoice says {tax_total}",
                fields=["header.tax_total", "line.tax_rate"],
                values=TaxTotalOff(expected, tax_total),
            )
    return out


# --------------------------------------------------------------------------- #
# 4. Printed position numbers are consecutive
# --------------------------------------------------------------------------- #


def check_numbering(doc: Doc) -> list[Issue]:
    lines = _lines(doc)
    nos = [item.get("line_no") for item in lines]
    ints = [n for n in nos if isinstance(n, int) and not isinstance(n, bool)]
    if len(lines) < 2 or len(ints) < 0.8 * len(lines):
        return []  # document does not number its lines
    out = []
    dup = [n for n, c in Counter(ints).items() if c > 1]
    if dup:
        shown = sorted(dup)[:10]
        out.append(
            Issue(
                "numbering",
                f"position numbers repeated: {shown}",
                fields=["line.line_no"],
                values=RepeatedNumbers(len(dup), tuple(shown)),
            )
        )
    present = set(ints)
    missing = [n for n in range(min(ints), max(ints) + 1) if n not in present]
    if missing:
        first_after = {n: i for i, n in reversed(list(enumerate(nos))) if isinstance(n, int)}
        nxt = next((first_after[n] for n in sorted(present) if n > missing[0]), None)
        page = lines[nxt].get("page") if nxt is not None else None
        listed = missing[:10]
        out.append(
            Issue(
                "numbering",
                f"{len(missing)} position number(s) missing, e.g. {listed}",
                line=nxt,
                page=page,
                # A whole row is missing before this one; no cell of this one is in doubt.
                fields=[],
                values=MissingNumbers(len(missing), tuple(listed)),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# 5. Control totals printed in the document
# --------------------------------------------------------------------------- #


def check_checkpoints(doc: Doc, partial: bool = False) -> list[Issue]:
    """A checkpoint is either a running total (carried forward) or the subtotal
    of the lines since the previous checkpoint. Accept either reading; a
    failure pins the error to the segment between two checkpoints.

    partial=True: `doc` is a page range from the middle of a document, so the
    first checkpoint cannot be verified (its lines are on earlier pages); it
    only sets the base for the following ones."""
    return list(_checkpoints(doc, partial).values())


def _checkpoints(doc: Doc, partial: bool) -> dict[range, Issue]:
    """The issues of check_checkpoints, each under the lines it adds up (indexes into the
    line items): no two control totals that are verified add up the same lines."""
    lines = _lines(doc)
    amounts = [dec(item.get("amount")) or Decimal(0) for item in lines]
    out: dict[range, Issue] = {}
    prev_pos, prev_amount = 0, Decimal(0)
    cps: list[Checkpoint] = sorted(
        (c for c in (doc.get("checkpoints") or []) if isinstance(c, dict)),
        key=lambda c: c.get("after_line") or 0,
    )
    first = True
    for c in cps:
        amount, pos = dec(c.get("amount")), c.get("after_line")
        if amount is None or not isinstance(pos, int) or not (prev_pos <= pos <= len(lines)):
            continue
        segment = sum(amounts[prev_pos:pos], Decimal(0))
        # No lines since the previous checkpoint: a running total repeated at the
        # top of a page ("carried forward"). Nothing to verify, it is the new base.
        skip = (pos == prev_pos and not first) or (partial and first) or (pos == 0)
        first = False
        if skip:
            prev_pos, prev_amount = pos, amount
            continue
        if abs(segment - amount) > CENT and abs(prev_amount + segment - amount) > CENT:
            label = str(c.get("label") or "control total")
            out[range(prev_pos, pos)] = Issue(
                "checkpoints",
                f'lines {prev_pos + 1}-{pos} add up to {segment}, but "{label}" says {amount}',
                line=prev_pos,
                page=c.get("page"),
                # Not "line.amount": the line is only the first row of the segment, and
                # framing its amount would point at a cell that may well be right.
                fields=[],
                values=ControlTotalOff(prev_pos + 1, pos, segment, amount, label),
            )
        prev_pos, prev_amount = pos, amount
    return out


# --------------------------------------------------------------------------- #
# One wrong line amount behind the issues of the document's arithmetic
# --------------------------------------------------------------------------- #


def with_amounts(doc: Doc, amounts: dict[int, Decimal]) -> Doc:
    """A copy of `doc` in which the line at each index (of its line items) has the amount
    given. The lines of `doc` are not changed."""
    lines = [
        item | {"amount": amounts[i]} if i in amounts else item
        for i, item in enumerate(_lines(doc))
    ]
    return doc | {"lines": lines}


# What testing each line may cost, counted as the lines with a new amount times the size
# of the document (its lines, control totals and tax rows). Testing whether an issue that
# passes with the new amounts fails again without one line's runs its check over the
# whole document once more, so this bounds the rows all the tests visit together: about
# 0.05 s of the event loop. Above it, every one among an issue's own lines that has a new
# amount is taken as needed, untested: the document was misread throughout, or is long.
TEST_BUDGET = 25_000


def _size(doc: Doc) -> int:
    """The rows a check of `doc` goes through: lines, control totals and tax rows."""
    breakdown = (doc.get("header") or {}).get("tax_breakdown")
    checkpoints = doc.get("checkpoints")
    return sum(len(x) for x in (doc.get("lines"), checkpoints, breakdown) if isinstance(x, list))


def explained_by_lines(
    doc: Doc, amounts: dict[int, Decimal], partial: bool = False
) -> list[tuple[Issue, list[int]]]:
    """The issues of the document's arithmetic that pass once lines get other amounts.

    amounts: {line index: the amount it would have}. Looked at: the lines against the
    subtotal (check_doc_sums), the tax by rate (check_tax) and the control totals
    (check_checkpoints); with partial=True (a page range) the control totals only, as
    check_checkpoints(doc, partial=True) raises them. Returns each issue that fails on
    `doc` and passes with the new amounts, as raised on `doc`, with the indexes of the
    lines it needs, in order: those among its own lines (all for the subtotal, those of
    its rate for the tax, those it adds up for a control total) without whose new amount
    it fails again. When testing that would cost more than TEST_BUDGET, all of its own
    lines that have one, untested.

    Each check runs on `doc` and with the new amounts, and then once without each line
    tested: the time grows with the length of the document, not with its square."""
    lines = _lines(doc)
    of_rate: dict[Decimal, list[int]] = {}
    for i, item in enumerate(lines):
        rate = dec(item.get("tax_rate"))
        if rate is not None:
            of_rate.setdefault(rate, []).append(i)

    def subtotal(d: Doc) -> dict[str, Issue]:
        return {k: i for k, i in _doc_sums(d).items() if k == "subtotal"}

    def checkpoints(d: Doc) -> dict[range, Issue]:
        return _checkpoints(d, partial)

    # The lines an issue depends on, from the key it is raised under.
    def every_line(key: object) -> Iterable[int]:
        return range(len(lines))

    def same_rate(key: object) -> Iterable[int]:
        if key == "total":  # without a breakdown: every line with a rate
            return sorted(i for at in of_rate.values() for i in at)
        return of_rate.get(key, []) if isinstance(key, Decimal) else []

    def added_up(key: object) -> Iterable[int]:
        return key if isinstance(key, range) else []

    checks: list[tuple[Callable[[Doc], Mapping[Any, Issue]], Callable[[object], Iterable[int]]]] = (
        [(checkpoints, added_up)]
        if partial
        else [(subtotal, every_line), (_tax, same_rate), (checkpoints, added_up)]
    )
    changed = with_amounts(doc, amounts)
    out = []
    for check, own in checks:
        after = check(changed)
        explained = [
            (key, issue, [i for i in own(key) if i in amounts])
            for key, issue in check(doc).items()
            if key not in after
        ]
        if len(amounts) * _size(doc) <= TEST_BUDGET:
            # A line that is the only one of an issue's own lines with a new amount is
            # needed without a test: without it the issue is back where it was on `doc`.
            tested = sorted({i for *_, mine in explained if len(mine) > 1 for i in mine})
            without = {
                i: check(with_amounts(doc, {j: v for j, v in amounts.items() if j != i}))
                for i in tested
            }
            explained = [
                (key, issue, [i for i in mine if len(mine) == 1 or key in without[i]])
                for key, issue, mine in explained
            ]
        out += [(issue, needed) for _, issue, needed in explained if needed]
    return out


def agree_with_subtotal(doc: Doc, amounts: dict[int, Decimal]) -> list[int]:
    """The lines whose amount in `amounts` makes the lines add up to the subtotal of the
    header (within a cent) and whose amount as read does not, with the other lines in
    `amounts` either as read or each at its amount from there. Indexes into the line items
    of `doc`, in order; none without a subtotal. One pass over the lines."""
    lines = _lines(doc)
    subtotal = dec((doc.get("header") or {}).get("subtotal"))
    if subtotal is None or not lines:
        return []
    read = {i: dec(lines[i].get("amount")) or Decimal(0) for i in amounts}
    as_read = _lines_sum(lines)
    changed = as_read + sum((amounts[i] - read[i] for i in amounts), Decimal(0))

    def fits(total: Decimal) -> bool:
        return abs(total - subtotal) <= CENT

    return [
        i
        for i in sorted(amounts)
        # The other lines as read, then each at its new amount.
        if any(
            fits(rest + amounts[i]) and not fits(rest + read[i])
            for rest in (as_read - read[i], changed - amounts[i])
        )
    ]


# --------------------------------------------------------------------------- #
# 6-7. Is the claim actually printed on the page? (text layer)
# --------------------------------------------------------------------------- #


def check_grounding(doc: Doc, evidence: Evidence) -> list[Issue]:
    """Every extracted line must be found as ONE printed row carrying its sku,
    quantity, price and amount. Two failure kinds:
      grounding_value - a value is not on the page at all (misread or invented)
      grounding_row   - all values exist, but never together on one row
                        (values mixed up between lines)"""
    out = []
    used: set[tuple[int, int]] = set()
    for i, item in enumerate(_lines(doc)):
        sku = str(item.get("sku") or "").upper().strip()
        nums = {k: dec(item.get(k)) for k in ("quantity", "unit_price", "amount")}
        forms = {k: printed_forms(v) for k, v in nums.items() if v is not None}
        pages = evidence.page(item.get("page")) or evidence.pages
        found = False
        for p in pages:
            for ri, row in enumerate(p.rows):
                if (p.number, ri) in used:
                    continue
                toks = row.token_set
                if (not sku or sku in toks) and all(f & toks for f in forms.values()):
                    used.add((p.number, ri))
                    found = True
                    break
            if found:
                break
        if found:
            continue
        page_tokens = set[str]().union(*(p.token_set for p in pages)) if pages else set()
        absent = [k for k, f in forms.items() if not (f & page_tokens)]
        if sku and sku not in page_tokens:
            absent.insert(0, "sku")
        if absent:
            out.append(
                Issue(
                    "grounding_value",
                    f"not printed on the page: {', '.join(absent)}",
                    line=i,
                    page=item.get("page"),
                    fields=[f"line.{k}" for k in absent],
                )
            )
        else:
            out.append(
                Issue(
                    "grounding_row",
                    "values exist on the page but not together in one row",
                    line=i,
                    page=item.get("page"),
                    # The values that were looked for: a line without an sku is matched
                    # without one, and a value that is not a number is not looked for.
                    fields=[f"line.{k}" for k in (["sku"] if sku else []) + list(forms)],
                )
            )
    return out


def check_header_grounding(doc: Doc, evidence: Evidence) -> list[Issue]:
    """Totals and identifiers in the header must be printed somewhere in the
    document. A total that is not printed was computed by the model."""
    header: Header = doc.get("header") or {}
    tokens = set[str]().union(*(p.token_set for p in evidence.pages)) if evidence.pages else set()
    out = []
    # tax_total is often not printed as one figure (several rates, one row each),
    # so only the two figures that practically every invoice prints are required.
    for key in ("subtotal", "total"):
        v = dec(header.get(key))
        if v is not None and not (printed_forms(v) & tokens):
            out.append(
                Issue(
                    "grounding_header",
                    f"{key} {v} is not printed anywhere in the document",
                    fields=[f"header.{key}"],
                    values=HeaderNotPrinted(v),
                )
            )
    for key in ("invoice_number", "supplier_tax_id", "po_number"):
        s = str(header.get(key) or "").upper().strip()
        if s and " " not in s and s not in tokens:
            out.append(
                Issue(
                    "grounding_header",
                    f'{key} "{s}" is not printed anywhere in the document',
                    fields=[f"header.{key}"],
                    values=HeaderNotPrinted(s),
                )
            )
    return out


def check_coverage(doc: Doc, evidence: Evidence) -> list[Issue]:
    """Count rows that look like line items in the evidence and compare with
    how many lines were extracted - per page when the lines carry a page."""
    lines = _lines(doc)
    out = []
    if lines and all(isinstance(item.get("page"), int) for item in lines):
        got = Counter(item["page"] for item in lines)
        for p in evidence.pages:
            want, read = len(p.item_rows), got.get(p.number, 0)
            if want != read:
                out.append(
                    Issue(
                        "coverage",
                        f"page {p.number}: {want} item rows printed, {read} extracted",
                        page=p.number,
                        values=Coverage(want, read),
                    )
                )
    else:
        want = sum(len(p.item_rows) for p in evidence.pages)
        if want != len(lines):
            out.append(
                Issue(
                    "coverage",
                    f"{want} item rows printed, {len(lines)} extracted",
                    values=Coverage(want, len(lines)),
                )
            )
    return out
