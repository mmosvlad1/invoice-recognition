"""Independent evidence about what is actually printed on each page.

The model's answer is a claim. This module produces a second, deterministic
witness to check that claim against: the text layer of a digital PDF. No LLM is
involved.

The witness is deliberately dumb. It does not understand the invoice; it only
knows which tokens sit on which text row, and it can spot rows where two
numbers multiply to a third (quantity x price = amount) - which is what a
line item looks like in any layout or language.
"""

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import cast

import pypdfium2 as pdfium

CENT = Decimal("0.01")
_NUM = re.compile(r"^-?\d[\d.,]*$")
_TRIM = ".,;:()[]"


def money(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


def parse_number(token: str, locale: str) -> Decimal | None:
    """Parse a printed number. locale "us": 1,234.56   "eu": 1.234,56"""
    t = token.replace("−", "-").rstrip(_TRIM)
    if not _NUM.match(t):
        return None
    t = t.replace(".", "").replace(",", ".") if locale == "eu" else t.replace(",", "")
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def printed_forms(x: Decimal) -> set[str]:
    """Every way a value may be printed, in either locale, so a check does not
    depend on guessing the number format: 2739.07 -> 2,739.07 / 2.739,07 / ..."""
    sign = "-" if x < 0 else ""
    a = abs(x)
    # The exponent is an int for every finite number ("n", "N" or "F" otherwise).
    exp = max(0, -cast(int, a.as_tuple().exponent))
    places = {exp, max(exp, 2)}
    if a == a.to_integral_value():
        places |= {0, 1}
    out: set[str] = set()
    for p in places:
        us = f"{a:,.{p}f}"
        plain = us.replace(",", "")
        eu = us.replace(",", "\0").replace(".", ",").replace("\0", ".")
        out |= {sign + us, sign + plain, sign + eu, sign + plain.replace(".", ",")}
    return out


@dataclass
class Row:
    text: str
    tokens: list[str]
    triple: tuple[Decimal, Decimal, Decimal] | None = None  # a x b = c found on this row

    @property
    def token_set(self) -> set[str]:
        return {t.upper().rstrip(_TRIM).replace("−", "-") for t in self.tokens}


@dataclass
class Page:
    number: int  # 1-based
    rows: list[Row]

    @property
    def item_rows(self) -> list[Row]:
        return [r for r in self.rows if r.triple]

    @property
    def token_set(self) -> set[str]:
        return set[str]().union(*(r.token_set for r in self.rows)) if self.rows else set()


@dataclass
class Evidence:
    source: str  # where the text comes from; "text_layer" is the only source here
    locale: str  # number format the document appears to use: "us" or "eu"
    pages: list[Page]

    def page(self, number: int | None) -> list[Page]:
        """The page with this number (as a list: empty if there is none); all pages for None."""
        if number is None:
            return self.pages
        return [p for p in self.pages if p.number == number]


def _find_triple(tokens: list[str], locale: str) -> tuple[Decimal, Decimal, Decimal] | None:
    """A row is a line item if two of its numbers multiply to its last number."""
    nums = [n for t in tokens if not t.endswith("%") and (n := parse_number(t, locale)) is not None]
    if len(nums) < 3:
        return None
    c = nums[-1]
    head = nums[:-1]
    for i in range(len(head)):
        for j in range(i + 1, len(head)):
            if abs(money(head[i] * head[j]) - c) <= CENT:
                return head[i], head[j], c
    return None


def build(page_texts: list[str], source: str) -> Evidence:
    """Evidence from the text of each page (one string per page, rows split by newlines)."""
    raw = [[line.split() for line in text.splitlines() if line.strip()] for text in page_texts]
    # The right number format is the one under which more rows "multiply out".
    hits = {
        loc: sum(1 for page in raw for toks in page if _find_triple(toks, loc))
        for loc in ("us", "eu")
    }
    locale = "eu" if hits["eu"] > hits["us"] else "us"
    pages = [
        Page(n, [Row(" ".join(toks), toks, _find_triple(toks, locale)) for toks in page])
        for n, page in enumerate(raw, start=1)
    ]
    return Evidence(source=source, locale=locale, pages=pages)


# --------------------------------------------------------------------------- #
# Source
# --------------------------------------------------------------------------- #


def pdf_text_layer(data: bytes) -> list[str]:
    """Text of each page from the PDF itself. Empty strings for scanned PDFs."""
    pdf = pdfium.PdfDocument(data)
    try:
        out: list[str] = []
        for page in pdf:
            tp = page.get_textpage()
            out.append(tp.get_text_range())
            tp.close()
        return out
    finally:
        pdf.close()


def has_text_layer(page_texts: list[str], min_chars: int = 40) -> bool:
    """True when EVERY page has at least `min_chars` characters of text.

    A PDF in which only some pages have text counts as having no text layer,
    and is read like a scan. Known limitation, kept on purpose for now."""
    return bool(page_texts) and all(len(t.strip()) >= min_chars for t in page_texts)
