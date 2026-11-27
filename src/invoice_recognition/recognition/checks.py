"""Deterministic checks on an extraction. No model is asked whether it is sure.

Each check answers a different question and, where it can, says WHERE the
problem is (line, page), so the flow can re-read one page instead of the
whole document and the user can be shown the doubtful rows.

Expected shape of `doc`:
    header:      {supplier_name, subtotal, tax_total, total,
                  tax_breakdown?: [{rate, base, tax}], ...}
    lines:       [{sku, description, quantity, unit_price, tax_rate, amount, line_no?, page?}]
    checkpoints: [{amount, after_line, label?, page?}]   control totals printed in the
                 document (page carry-forwards, section subtotals); after_line = how many
                 line items precede it.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation

from invoice_recognition.recognition.evidence import CENT, Evidence, money, printed_forms
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

    def to_json(self) -> JSONObject:
        out = asdict(self)
        # Left out when unknown: most issues are not about a numbered line, and their
        # JSON then has only the keys every issue has.
        if out["line_no"] is None:
            del out["line_no"]
        return out


def dec(v: object) -> Decimal | None:
    """A number from the model's JSON: a number or a string ("20%" included).
    None if it is not one."""
    if v is None or isinstance(v, bool):
        return None
    try:
        return Decimal(str(v).strip().rstrip("%"))
    except InvalidOperation, ValueError:
        return None


def _lines(doc: Doc) -> list[Line]:
    return [item for item in (doc.get("lines") or []) if isinstance(item, dict)]


# --------------------------------------------------------------------------- #
# 1. Document-level totals
# --------------------------------------------------------------------------- #


def check_doc_sums(doc: Doc) -> list[Issue]:
    out = []
    header: Header = doc.get("header") or {}
    lines = _lines(doc)
    supplier = header.get("supplier_name")
    # A number, list or object is not a supplier name.
    if not (isinstance(supplier, str) and supplier.strip()):
        out.append(Issue("doc_sums", "supplier is missing"))
    if not lines:
        return [*out, Issue("doc_sums", "no line items")]
    subtotal, tax, total = (dec(header.get(k)) for k in ("subtotal", "tax_total", "total"))
    amounts = [dec(item.get("amount")) for item in lines]
    if subtotal is None or tax is None or total is None:
        return [*out, Issue("doc_sums", "totals are missing in the header")]
    if None in amounts:
        out.append(Issue("doc_sums", "some lines have no amount"))
    lines_sum = sum((a for a in amounts if a is not None), Decimal(0))
    if abs(lines_sum - subtotal) > CENT:
        out.append(
            Issue(
                "doc_sums",
                f"lines add up to {lines_sum}, invoice subtotal is {subtotal} "
                f"(difference {lines_sum - subtotal})",
            )
        )
    if abs(subtotal + tax - total) > CENT:
        out.append(Issue("doc_sums", f"subtotal {subtotal} + tax {tax} != total {total}"))
    return out


# --------------------------------------------------------------------------- #
# 2. Arithmetic inside one line
# --------------------------------------------------------------------------- #


def check_line_math(doc: Doc) -> list[Issue]:
    out = []
    for i, item in enumerate(_lines(doc)):
        q, p, a = dec(item.get("quantity")), dec(item.get("unit_price")), dec(item.get("amount"))
        if q is None or p is None or a is None:
            out.append(
                Issue(
                    "line_math",
                    "quantity, unit price or amount is missing",
                    line=i,
                    page=item.get("page"),
                )
            )
        elif abs(money(q * p) - a) > CENT:
            hint = ""
            if abs(money(q * 1000 * p) - a) <= CENT:
                hint = (
                    " (quantity looks 1000x too small: thousands separator read as a decimal point)"
                )
            out.append(
                Issue(
                    "line_math",
                    f"{q} x {p} = {money(q * p)}, but amount is {a}{hint}",
                    line=i,
                    page=item.get("page"),
                )
            )
    return out


# --------------------------------------------------------------------------- #
# 3. Tax per rate
# --------------------------------------------------------------------------- #


def check_tax(doc: Doc) -> list[Issue]:
    out = []
    header: Header = doc.get("header") or {}
    lines = _lines(doc)
    by_rate: dict[Decimal, Decimal] = {}
    for item in lines:
        r, a = dec(item.get("tax_rate")), dec(item.get("amount"))
        if r is not None and a is not None:
            by_rate[r] = by_rate.get(r, Decimal(0)) + a
    breakdown = header.get("tax_breakdown") or []
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
                if base_lines:
                    out.append(
                        Issue(
                            "tax",
                            f"lines carry {rate}% tax on {base_lines}, "
                            f"but the invoice has no {rate}% row",
                        )
                    )
            elif abs(base_lines - base_printed) > CENT:
                out.append(
                    Issue(
                        "tax",
                        f"lines at {rate}% add up to {base_lines}, invoice says {base_printed}",
                    )
                )
    else:
        tax_total = dec(header.get("tax_total"))
        expected = sum(
            (money(base * rate / 100) for rate, base in by_rate.items()),
            Decimal(0),
        )
        if tax_total is not None and abs(expected - tax_total) > Decimal("0.05"):
            out.append(
                Issue("tax", f"tax computed from lines is {expected}, invoice says {tax_total}")
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
        out.append(Issue("numbering", f"position numbers repeated: {sorted(dup)[:10]}"))
    present = set(ints)
    missing = [n for n in range(min(ints), max(ints) + 1) if n not in present]
    if missing:
        first_after = {n: i for i, n in reversed(list(enumerate(nos))) if isinstance(n, int)}
        nxt = next((first_after[n] for n in sorted(present) if n > missing[0]), None)
        page = lines[nxt].get("page") if nxt is not None else None
        out.append(
            Issue(
                "numbering",
                f"{len(missing)} position number(s) missing, e.g. {missing[:10]}",
                line=nxt,
                page=page,
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
    lines = _lines(doc)
    amounts = [dec(item.get("amount")) or Decimal(0) for item in lines]
    out = []
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
            label = c.get("label") or "control total"
            out.append(
                Issue(
                    "checkpoints",
                    f'lines {prev_pos + 1}-{pos} add up to {segment}, but "{label}" says {amount}',
                    line=prev_pos,
                    page=c.get("page"),
                )
            )
        prev_pos, prev_amount = pos, amount
    return out


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
                )
            )
        else:
            out.append(
                Issue(
                    "grounding_row",
                    "values exist on the page but not together in one row",
                    line=i,
                    page=item.get("page"),
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
                Issue("grounding_header", f"{key} {v} is not printed anywhere in the document")
            )
    for key in ("invoice_number", "supplier_tax_id", "po_number"):
        s = str(header.get(key) or "").upper().strip()
        if s and " " not in s and s not in tokens:
            out.append(
                Issue("grounding_header", f'{key} "{s}" is not printed anywhere in the document')
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
            want = len(p.item_rows)
            if want != got.get(p.number, 0):
                out.append(
                    Issue(
                        "coverage",
                        f"page {p.number}: {want} item rows printed, "
                        f"{got.get(p.number, 0)} extracted",
                        page=p.number,
                    )
                )
    else:
        want = sum(len(p.item_rows) for p in evidence.pages)
        if want != len(lines):
            out.append(Issue("coverage", f"{want} item rows printed, {len(lines)} extracted"))
    return out
