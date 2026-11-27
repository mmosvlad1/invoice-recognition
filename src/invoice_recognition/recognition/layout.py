"""A text-layer PDF: the model describes the layout of the line-item table once
per document, code extracts every line item from the text layer.

Division of labour:
  model - meaning: which column is what, the number format, how non-item rows
          look. It sees the text of ONE page and never copies a value.
  code  - geometry and reading: finds where each column sits (X anchors), walks
          every page, cuts each text row into cells, parses the numbers.

Nothing in this module knows a particular invoice: column labels, markers of
carry-forward / subtotal rows and the number format all come from the model's
answer. What the parser assumes about tables in general:
  - a table row is one text line of the PDF text layer (pdfium line breaks);
  - the column headings sit on one text line and each heading overlaps its
    column horizontally;
  - every column is left- or right-aligned, at the same X on every page of the
    document;
  - a cell of a right-aligned column is one word, unless the words before it
    follow a right-aligned cell;
  - a wrapped description continues on the next text line, starting at the
    description column; across a page break it continues in the first such line
    of the next page that comes before any item or declared row;
  - a line item prints quantity, unit price and amount (a declared discount row
    may print the amount only).
"""

import ctypes
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from invoice_recognition.recognition import checks
from invoice_recognition.shapes import Checkpoint, JSONObject, Line

FIELDS = ("line_no", "sku", "description", "quantity", "unit", "unit_price", "tax_rate", "amount")
TOTAL_KINDS = ("carry_forward", "section_total")  # non-item rows that carry a control total
X_TOL = 1.0  # points; loose glyph boxes of one column edge agree to ~0.1 pt in a generated PDF


# --------------------------------------------------------------------------- #
# Text layer with coordinates
# --------------------------------------------------------------------------- #


@dataclass
class Word:
    text: str
    x0: float
    x1: float


@dataclass
class TextLine:
    page: int  # 1-based
    index: int  # 1-based among the non-empty lines of the page; the number shown to the model
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def text_lines(data: bytes) -> list[list[TextLine]]:
    """Text lines of every page, each word with its left and right X.

    Lines are the same as in `evidence.pdf_text_layer` (pdfium's own line
    breaks), so row counts are comparable with the evidence module."""
    pdf = pdfium.PdfDocument(data)
    rect = pdfium_c.FS_RECTF()
    out: list[list[TextLine]] = []
    try:
        for pno, page in enumerate(pdf, start=1):
            tp = page.get_textpage()
            rows: list[list[Word]] = [[]]
            cur: Word | None = None
            for i in range(tp.count_chars()):
                u = pdfium_c.FPDFText_GetUnicode(tp, i)
                ch = chr(u) if not 0xD800 <= u <= 0xDFFF else "�"
                if ch.isspace() or u == 0:
                    if cur is not None:
                        rows[-1].append(cur)
                        cur = None
                    if ch == "\n":
                        rows.append([])
                    continue
                pdfium_c.FPDFText_GetLooseCharBox(tp, i, ctypes.byref(rect))
                if cur is None:
                    cur = Word(ch, rect.left, rect.right)
                else:
                    cur.text += ch
                    cur.x0, cur.x1 = min(cur.x0, rect.left), max(cur.x1, rect.right)
            if cur is not None:
                rows[-1].append(cur)
            tp.close()
            rows = [r for r in rows if r]
            out.append([TextLine(pno, n, r) for n, r in enumerate(rows, start=1)])
    finally:
        pdf.close()
    return out


def scanned_pages(data: bytes, min_cover: float = 0.85) -> list[int]:
    """Pages (1-based) where one image covers nearly the whole sheet, however the page
    is turned. Such a page is a scan, whatever text lies over it: the text layer was put
    there by somebody's OCR and is no better a witness than our own OCR would be."""
    pdf = pdfium.PdfDocument(data)
    out = []
    try:
        for pno, page in enumerate(pdf, start=1):
            w, h = page.get_size()
            # The size is that of the page as shown; the bounds of an image are in the
            # coordinates of the page as stored. A page shown turned by 90 or 270 degrees
            # (/Rotate) has its width and height swapped against its images.
            if page.get_rotation() % 180:
                w, h = h, w
            for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]):
                left, bottom, right, top = obj.get_bounds()
                seen = max(0.0, min(right, w) - max(left, 0.0)) * max(
                    0.0, min(top, h) - max(bottom, 0.0)
                )
                if seen >= min_cover * w * h:
                    out.append(pno)
                    break
    finally:
        pdf.close()
    return out


# --------------------------------------------------------------------------- #
# The layout request
# --------------------------------------------------------------------------- #

LAYOUT_PROMPT = """\
You are given the text layer of one page of a supplier invoice as numbered text lines:
the line number, "|", then the text exactly as printed. It is the first page that carries
line items. If the invoice is longer, the first lines of the following page are added below
it, so that you can see what the top of a continuation page looks like.

A program will read every line item of the whole document from the text layer. It only needs
you to describe the LAYOUT of the line-item table. Do NOT copy line items or their values.

Return one JSON object and nothing else, no explanations:

{
  "header_line": int,
  "columns": [{"label": str, "field": str}],
  "numbers": {"decimal": str, "thousands": str},
  "tax": {"in_row": bool, "document_rate": number or null},
  "description_wraps": bool,
  "non_item_rows": [{"kind": str, "starts_with": str}],
  "discount_rows": {"present": bool, "marker": str or null}
}

- header_line: the number of the line that holds the column headings of the line-item table.
- columns: EVERY column of that table, left to right, none skipped. label is the heading of
  the column copied exactly as printed on header_line (a heading of two words stays two
  words). field says what the column holds, one of: line_no (position number), sku (article
  / item code), description, quantity, unit (unit of measure), unit_price, tax_rate, amount
  (net line total), other (anything else). Each field except other is used at most once.
  Read the headings carefully: the price column may come before the quantity column.
- numbers: look at the values in the item rows. decimal is the character that separates the
  fraction ("." or ","), thousands the character that groups thousands (",", ".", " " or ""
  if none is used). "1.234,56" means decimal "," and thousands "."; then a quantity printed
  as "5.000" is five thousand.
- tax.in_row: true if every line item prints its tax rate in one of the columns. If not,
  false, and document_rate is the single tax rate printed elsewhere on the page (20 for
  20%), or null.
- description_wraps: true if a long description continues on the next text line (a line
  with only description text right under an item row).
- non_item_rows: every kind of line printed among the item rows, or between the table
  heading and the end of the items, that is NOT a line item. kind is one of:
  "carry_forward" (a running total carried to or from another page, printed at the bottom
  or at the top of a page), "section_total" (subtotal of a group of items),
  "section_header" (heading of a group of items), "other". starts_with is the fixed words
  such a line begins with, copied as printed, WITHOUT the part that changes from line to
  line (identifiers, dates, amounts). One entry per distinct beginning; a carry-forward
  line at the top of the following page counts too. Do not list the table heading, wrapped
  descriptions, the invoice header, the page footer or the totals block at the end of the
  invoice.
- discount_rows: present is true if discounts or rebates are printed as rows of their own in
  the table; marker is the fixed words by which the description of such a row is recognised,
  else null.
"""


def layout_request(pages: list[list[TextLine]], item_page: int, next_lines: int = 6) -> str:
    """Prompt + the numbered text of page `item_page` (1-based) + the top of the next page."""
    body = [f"--- page {item_page} of {len(pages)} ---"]
    body += [f"{tl.index}| {tl.text}" for tl in pages[item_page - 1]]
    if item_page < len(pages) and next_lines:
        body.append(f"--- first lines of page {item_page + 1} (not numbered) ---")
        body += [tl.text for tl in pages[item_page][:next_lines]]
    return LAYOUT_PROMPT + "\n" + "\n".join(body) + "\n"


# --------------------------------------------------------------------------- #
# From the model's answer to column geometry
# --------------------------------------------------------------------------- #


class LayoutError(ValueError):
    """The model's answer cannot be used: the whole document needs the fallback path."""


@dataclass
class Column:
    field: str  # one of FIELDS or "other"
    label: str
    align: str  # "l" | "r"
    anchor: float  # X of the left edge (align "l") or of the right edge (align "r")
    support: int  # how many words on the sample page sit on that anchor


@dataclass
class Layout:
    header_line: int
    columns: list[Column]
    decimal: str
    thousands: str
    tax_in_row: bool
    document_rate: Decimal | None
    description_wraps: bool
    non_item: list[tuple[str, str]]  # (kind, normalised beginning)
    discount_marker: str | None
    number_re: re.Pattern[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        d = re.escape(self.decimal)
        if self.thousands:
            t = re.escape(self.thousands)
            self.number_re = re.compile(rf"(?:\d{{1,3}}(?:{t}\d{{3}})+|\d+)(?:{d}\d+)?")
        else:
            self.number_re = re.compile(rf"\d+(?:{d}\d+)?")

    def index(self, name: str) -> int | None:
        return next((i for i, c in enumerate(self.columns) if c.field == name), None)

    def number(self, token: str | None) -> Decimal | None:
        """Parse a printed number strictly in the declared format; None if it is not one,
        or if it is out of the range checks.dec accepts."""
        if not token:
            return None
        s = token.strip().replace("−", "-").strip("€$£").strip()
        neg = False
        if s.startswith("(") and s.endswith(")"):
            neg, s = True, s[1:-1]
        if s.startswith("-"):
            neg, s = True, s[1:]
        elif s.endswith("-"):
            neg, s = True, s[:-1]
        elif s.startswith("+"):
            s = s[1:]
        if not self.number_re.fullmatch(s):
            return None
        if self.thousands:
            s = s.replace(self.thousands, "")
        try:
            v = Decimal(s.replace(self.decimal, "."))
        except InvalidOperation:
            return None
        # A printed number of absurd length is not a usable value either (checks.dec).
        return checks.dec(-v if neg else v)


def _norm(s: object) -> str:
    return " ".join(str(s).casefold().split())


def _match_labels(words: list[Word], labels: list[str]) -> list[tuple[float, float]] | None:
    """Find every label, in order, as consecutive words of one line; spaces inside
    a label do not matter ("VAT %" = "VAT%"). Returns the X extent of each label."""
    toks = [w.text.casefold() for w in words]
    pos, out = 0, []
    for lab in labels:
        found = None
        for i in range(pos, len(toks)):
            joined, k = "", i
            while k < len(toks) and len(joined) < len(lab):
                joined += toks[k]
                k += 1
            if joined == lab:
                found = (i, k)
                break
        if found is None:
            return None
        out.append((words[found[0]].x0, words[found[1] - 1].x1))
        pos = found[1]
    return out


def _cluster(values: list[float], tol: float) -> tuple[float, int]:
    """The value most others agree with, and how many agree."""
    best, n = 0.0, 0
    for v in values:
        c = sum(1 for x in values if abs(x - v) <= tol)
        if c > n:
            best, n = v, c
    return best, n


def _object(value: object) -> JSONObject:
    """A JSON object of the answer, or an empty one when it is anything else."""
    return value if isinstance(value, dict) else {}


def resolve(raw: Any, page_lines: list[TextLine], tol: float = X_TOL) -> Layout:
    """Check the model's answer and work out where each column sits.

    The model names the headings; where the cells of each column are anchored
    is measured here: among the words printed under a heading, the left edges
    or the right edges agree - that edge is the anchor of the column."""
    if not isinstance(raw, dict):
        raise LayoutError("layout answer is not a JSON object")
    cols_raw = [
        c
        for c in (raw.get("columns") or [])
        if isinstance(c, dict) and str(c.get("label") or "").strip()
    ]
    if not cols_raw:
        raise LayoutError("no columns in the layout answer")
    fields: list[str] = [c["field"] if c.get("field") in FIELDS else "other" for c in cols_raw]
    dup = sorted({f for f in fields if f != "other" and fields.count(f) > 1})
    if dup:
        raise LayoutError(f"field used for more than one column: {', '.join(dup)}")
    if "amount" not in fields:
        raise LayoutError("no amount column in the layout answer")

    nums = _object(raw.get("numbers"))
    decimal, thousands = nums.get("decimal"), nums.get("thousands") or ""
    if (
        decimal not in (".", ",")
        or thousands not in (",", ".", " ", "'", "")
        or decimal == thousands
    ):
        raise LayoutError(f"unusable number format: decimal {decimal!r}, thousands {thousands!r}")

    # Header line: the declared one if the labels are found on it, else any line that has them.
    labels = ["".join(_norm(c["label"]).split()) for c in cols_raw]
    declared = raw.get("header_line")
    order = sorted(page_lines, key=lambda tl: tl.index != declared)
    header, spans = None, None
    for tl in order:
        spans = _match_labels(sorted(tl.words, key=lambda w: w.x0), labels)
        if spans:
            header = tl
            break
    if header is None or spans is None:
        raise LayoutError("column labels are not found together on any line of the page")

    below = [tl for tl in page_lines if tl.index > header.index]
    columns = []
    for c, f, (lx0, lx1) in zip(cols_raw, fields, spans):
        under = [w for tl in below for w in tl.words if w.x1 > lx0 and w.x0 < lx1]
        if not under:
            raise LayoutError(f"nothing is printed under the heading {c['label']!r}")
        left, n_left = _cluster([w.x0 for w in under], tol)
        right, n_right = _cluster([w.x1 for w in under], tol)
        if n_left == n_right:  # fixed-width content: take the edge the heading itself sits on
            align = "l" if abs(left - lx0) <= abs(right - lx1) else "r"
        else:
            align = "l" if n_left > n_right else "r"
        columns.append(
            Column(
                f,
                str(c["label"]).strip(),
                align,
                left if align == "l" else right,
                n_left if align == "l" else n_right,
            )
        )

    tax = _object(raw.get("tax"))
    # Every row gets this rate: a value that is not a usable number (NaN, absurd size) is
    # no rate, as a value that is not a number at all.
    document_rate = checks.dec(tax.get("document_rate"))
    non_item = [
        (str(r.get("kind") or "other"), _norm(r.get("starts_with") or ""))
        for r in (raw.get("non_item_rows") or [])
        if isinstance(r, dict)
    ]
    disc = _object(raw.get("discount_rows"))
    marker = _norm(disc.get("marker") or "") if disc.get("present") else ""
    return Layout(
        header_line=header.index,
        columns=columns,
        decimal=decimal,
        thousands=thousands,
        tax_in_row="tax_rate" in fields and tax.get("in_row", True) is not False,
        document_rate=document_rate,
        description_wraps=bool(raw.get("description_wraps")),
        non_item=[(k, p) for k, p in non_item if p],
        discount_marker=marker or None,
    )


# --------------------------------------------------------------------------- #
# Reading the rows
# --------------------------------------------------------------------------- #


def assign(
    words: list[Word], columns: list[Column], tol: float = X_TOL
) -> tuple[dict[int, list[Word]], list[Word]]:
    """Cut one text line into cells. A word belongs to a column if it sits on
    the column's anchor; a word on no anchor continues the left-aligned cell
    before it, or opens the right-aligned cell after it. Columns are taken
    left to right only. Returns (cells by column index, words left over)."""
    cells: dict[int, list[Word]] = {}
    stray: list[Word] = []
    pending: list[Word] = []
    last = -1
    for w in sorted(words, key=lambda w: w.x0):
        hit = next(
            (
                i
                for i in range(last + 1, len(columns))
                if abs((w.x0 if columns[i].align == "l" else w.x1) - columns[i].anchor) <= tol
            ),
            None,
        )
        if hit is not None:
            if pending and columns[hit].align == "r":
                cells[hit] = pending
            else:
                stray += pending
            pending = []
            cells.setdefault(hit, []).append(w)
            last = hit
        elif last >= 0 and columns[last].align == "l" and not pending:
            cells[last].append(w)
        else:
            pending.append(w)
    return cells, stray + pending


@dataclass
class Parsed:
    lines: list[Line]  # numbers are Decimal
    checkpoints: list[Checkpoint]
    notes: list[JSONObject]  # rows the parser could not place: {page, row, kind, text}


def parse_document(pages: list[list[TextLine]], layout: Layout, tol: float = X_TOL) -> Parsed:
    """Walk every page and collect line items, control totals and doubts.

    A row is a line item if its quantity, unit price and amount cells hold
    numbers in the declared format (whichever of these columns the layout
    has). Anything else is a declared non-item row, a continuation of the
    description above, or is ignored - and noted, if it looks like it might
    have been an item."""
    cols = layout.columns
    ix = {f: layout.index(f) for f in FIELDS}
    required = [f for f in ("quantity", "unit_price", "amount") if ix[f] is not None]
    lines: list[Line] = []
    checkpoints: list[Checkpoint] = []
    notes: list[JSONObject] = []

    def cell(cells: dict[int, list[Word]], name: str) -> str:
        i = ix[name]
        return " ".join(w.text for w in cells.get(i, [])) if i is not None else ""

    open_item: Line | None = None  # the last item, while no declared non-item row has closed it
    for page in pages:
        prev: Line | None = None  # the item a description-only row would continue
        carried = open_item  # an item the previous page may have left unfinished
        for row in page:
            cells, stray = assign(row.words, cols, tol)
            text = _norm(row.text)
            num = {f: layout.number(cell(cells, f)) for f in ("quantity", "unit_price", "amount")}
            is_item = all(num[f] is not None for f in required)
            if (
                not is_item
                and layout.discount_marker
                and layout.discount_marker in text
                and num["amount"] is not None
            ):
                is_item = True
            if is_item:
                rate_text = cell(cells, "tax_rate").replace("%", "").strip()
                rate = layout.number(rate_text) if layout.tax_in_row else layout.document_rate
                no_text = cell(cells, "line_no").rstrip(".")
                item: Line = {
                    "line_no": int(no_text) if no_text.isdigit() else None,
                    "sku": cell(cells, "sku"),
                    "description": cell(cells, "description"),
                    "quantity": num["quantity"],
                    "unit": cell(cells, "unit"),
                    "unit_price": num["unit_price"],
                    "tax_rate": rate,
                    "amount": num["amount"],
                    "page": row.page,
                }
                lines.append(item)
                prev = open_item = item
                carried = None
                if stray:
                    notes.append(
                        {
                            "page": row.page,
                            "row": row.index,
                            "kind": "stray_words",
                            "text": row.text,
                        }
                    )
                if layout.tax_in_row and rate is None:
                    notes.append(
                        {
                            "page": row.page,
                            "row": row.index,
                            "kind": "no_tax_rate",
                            "text": row.text,
                        }
                    )
                continue

            kind = next((k for k, p in layout.non_item if text.startswith(p)), None)
            if kind is not None:
                if kind in TOTAL_KINDS:
                    amount = num["amount"]
                    if amount is None:  # not on the amount anchor: take the last number of the row
                        amount = next(
                            (
                                v
                                for w in reversed(row.words)
                                if (v := layout.number(w.text)) is not None
                            ),
                            None,
                        )
                    if amount is not None:
                        checkpoints.append(
                            {
                                "label": row.text,
                                "amount": amount,
                                "after_line": len(lines),
                                "page": row.page,
                                "kind": kind,
                            }
                        )
                    else:
                        notes.append(
                            {
                                "page": row.page,
                                "row": row.index,
                                "kind": "total_without_amount",
                                "text": row.text,
                            }
                        )
                prev = open_item = carried = None
                continue

            only_description = (
                not stray and ix["description"] is not None and set(cells) == {ix["description"]}
            )
            if prev is not None and only_description:
                if layout.description_wraps:
                    prev["description"] = f"{prev['description']} {row.text}".strip()
                    continue  # the next row may continue it as well
                notes.append(
                    {
                        "page": row.page,
                        "row": row.index,
                        "kind": "text_under_item",
                        "text": row.text,
                    }
                )
            elif carried is not None and only_description and layout.description_wraps:
                # Top of a page, before any item or declared row: the description of the last
                # item of the previous page goes on here. Page furniture in between is skipped.
                carried["description"] = f"{carried['description']} {row.text}".strip()
                notes.append(
                    {
                        "page": row.page,
                        "row": row.index,
                        "kind": "joined_across_pages",
                        "text": row.text,
                    }
                )
                continue
            elif num["amount"] is not None:
                # A number in the amount column, yet neither an item nor a declared total.
                notes.append(
                    {
                        "page": row.page,
                        "row": row.index,
                        "kind": "unclassified_amount",
                        "text": row.text,
                    }
                )
            prev = None
    return Parsed(lines, checkpoints, notes)
