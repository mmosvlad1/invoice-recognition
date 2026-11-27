"""Sample invoices with their ground truth, and model answers recorded on them.

tests/fixtures/invoices: generated invoices (`<name>.pdf` with a text layer, and one
scan, `s_classic.scan.pdf`) and their ground truth (`<name>.truth.json`: meta, header,
lines and, for a document whose page breaks cut rows, split_lines). Numbers in the
truth are strings, as printed.

tests/fixtures/answers: what the cheap model really answered. `<name>.layout.json` is
its description of the line-item table of a text-layer PDF; `m_split.scan.pages.json`
holds its answers to the five pages of the scan of m_split, asked with the "carry_in"
field, with the sha256 of the prompt each answer was given for.
"""

import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

# The JSON shapes of the package, spelled out: this module imports nothing from the
# package, so a reference run of other code can load the same samples.
type JSONObject = dict[str, Any]
type Header = dict[str, Any]
type Line = dict[str, Any]

FIXTURES = Path(__file__).parent / "fixtures"
INVOICES = FIXTURES / "invoices"
ANSWERS = FIXTURES / "answers"

TEXT_LAYER = ("s_classic", "s_dense", "m_grouped", "m_split", "l_classic", "l_dense", "xl_grouped")
"""The generated invoices with a text layer."""
WITH_LAYOUT = tuple(n for n in TEXT_LAYER if n != "m_split")
"""The ones with a recorded layout answer."""

LINE_KEYS = (
    "line_no",
    "sku",
    "description",
    "quantity",
    "unit",
    "unit_price",
    "tax_rate",
    "amount",
)
_NUMERIC = ("quantity", "unit_price", "tax_rate", "amount")


def _number(v: str) -> int | float:
    f = float(v)
    return int(f) if f == int(f) else f


def truth(name: str) -> JSONObject:
    data: JSONObject = json.loads((INVOICES / f"{name}.truth.json").read_text())
    return data


def truth_doc(name: str) -> tuple[bytes, JSONObject, Header, dict[int, list[Line]]]:
    """(pdf bytes, truth, the header the model would return, {page: the rows it would
    return}) for a generated document: the answers of a model that reads it right."""
    t = truth(name)
    pages: dict[int, list[Line]] = {p: [] for p in range(1, t["meta"]["pages"] + 1)}
    for item in t["lines"]:
        row = {k: (_number(item[k]) if k in _NUMERIC else item.get(k)) for k in LINE_KEYS}
        pages[item["page"]].append(row)
    h = t["header"]
    text = (
        "supplier_name",
        "supplier_tax_id",
        "invoice_number",
        "invoice_date",
        "due_date",
        "po_number",
        "currency",
    )
    header: Header = {k: h[k] for k in text}
    header |= {k: _number(h[k]) for k in ("subtotal", "tax_total", "total")}
    header["tax_breakdown"] = [{k: _number(v) for k, v in r.items()} for r in h["tax_breakdown"]]
    return (INVOICES / f"{name}.pdf").read_bytes(), t, header, pages


def saved_layout(name: str) -> JSONObject:
    """The layout of the document's line-item table, as the model described it."""
    data: JSONObject = json.loads((ANSWERS / f"{name}.layout.json").read_text())
    return data


def saved_scan_pages() -> JSONObject:
    """The model's answers to the pages of m_split.scan.pdf, asked with "carry_in"."""
    data: JSONObject = json.loads((ANSWERS / "m_split.scan.pages.json").read_text())
    return data


def _num(v: Any) -> Decimal | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return Decimal(str(v).strip().rstrip("%"))
    except InvalidOperation:
        return None


def _key(item: Line) -> tuple[str, Decimal | None, Decimal | None, Decimal | None, Decimal | None]:
    sku = " ".join(str(item.get("sku") or "").upper().split())
    q, p, r, a = (_num(item.get(k)) for k in _NUMERIC)
    return sku, q, p, r, a


def exact(lines: list[Line], truth_lines: list[Line]) -> bool:
    """Every line read is a line of the truth and every line of the truth is read: the
    same sku (case and spacing aside), quantity, unit price, tax rate and amount, each
    as many times as in the truth. Order and descriptions are not compared."""
    return Counter(_key(item) for item in lines if isinstance(item, dict)) == Counter(
        _key(item) for item in truth_lines
    )
