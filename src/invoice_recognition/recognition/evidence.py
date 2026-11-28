"""Independent evidence about what is actually printed on each page.

The model's answer is a claim. This module produces a second, deterministic
witness to check that claim against: the text layer of a digital PDF. No LLM is
involved.

The witness is deliberately dumb. It does not understand the invoice; it only
knows which tokens sit on which text row, and it can spot rows where two
numbers multiply to a third (quantity x price = amount) - which is what a
line item looks like in any layout or language.
"""

import ctypes
import re
from collections import Counter
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import cast

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

CENT = Decimal("0.01")
_NUM = re.compile(r"^-?\d[\d.,]*$")
_TRIM = ".,;:()[]"


def money(x: Decimal) -> Decimal:
    """`x` rounded half up to cents.

    A finite number with 27 or more digits before the point cannot carry cents
    within the 28-digit precision of decimal arithmetic (quantize raises), and
    its cents are lost to that precision anyway: it is returned as it is."""
    try:
        return x.quantize(CENT, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        if not x.is_finite():
            raise
        return x


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
    depend on guessing the number format: 2739.07 -> 2,739.07 / 2.739,07 / ...

    `x` must be finite: NaN and infinity have no printed form."""
    if not x.is_finite():
        raise ValueError(f"not a finite number: {x}")
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


def comparable(token: str) -> str:
    """A printed word as the presence checks compare it: upper case, trailing
    punctuation dropped, a typographic minus as "-"."""
    return token.upper().rstrip(_TRIM).replace("−", "-")


@dataclass
class Row:
    text: str
    tokens: list[str]
    triple: tuple[Decimal, Decimal, Decimal] | None = None  # a x b = c found on this row

    @property
    def token_set(self) -> set[str]:
        return {comparable(t) for t in self.tokens}


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


# The clockwise quarter turn of the page (as /Rotate gives it) under which text written
# along +x, +y, -x or -y of user space reads left to right.
_TURN_OF = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270}

# A page is turned only when at least this share of its characters is written in one
# direction. A sideways note in the margin or a stamp - on a light page it may hold more
# characters than the page's own text - must never turn an upright page; and a page with
# no clear direction keeps the rotation it has.
UPRIGHT_SHARE = 0.8


def _turn_votes(tp: pdfium.PdfTextPage) -> Counter[int]:
    """How many printed characters of the page read upright under each turn (_TURN_OF).

    From each character's text matrix (FPDFText_GetMatrix): the PDF's own numbers, which
    do not include the page's /Rotate, so the vote is the same whatever the page's
    rotation. The direction is the larger of the matrix's a and b, the test _char_box of
    the layout module uses. The spaces and line breaks pdfium inserts itself do not vote:
    they carry the identity matrix, whatever the direction of the text around them (on a
    page of short cells, a third of the votes)."""
    votes: Counter[int] = Counter()
    m = pdfium_c.FS_MATRIX()
    for i in range(tp.count_chars()):
        if pdfium_c.FPDFText_IsGenerated(tp, i) == 1:
            continue
        if not pdfium_c.FPDFText_GetMatrix(tp, i, ctypes.byref(m)):
            continue
        if m.a == m.b == 0:  # defence only: pdfium leaves such characters out of the page
            continue
        if abs(m.a) >= abs(m.b):
            direction = (1 if m.a > 0 else -1, 0)
        else:
            direction = (0, 1 if m.b > 0 else -1)
        votes[_TURN_OF[direction]] += 1
    return votes


def upright_textpage(page: pdfium.PdfPage) -> pdfium.PdfTextPage:
    """The text page of `page`, built with the page turned so that its text reads left to
    right, top to bottom.

    pdfium groups characters into lines and orders them in display space, which includes
    the page's /Rotate: on an upright page shown turned by /Rotate 90 the lines of a
    wrapped description come out in the wrong order and a row splits in two; on /Rotate
    180 each row reads right to left. So the direction most characters are written in is
    voted on (_turn_votes); when at least UPRIGHT_SHARE of them read upright under
    another rotation than the page's own, the page is given that rotation and the text
    page is built again. Otherwise - an upright page, a page drawn turned and shown
    upright by its /Rotate, a page without characters, a page with no clear direction -
    it is the page's text page as pdfium builds it.

    Character boxes, origins and matrices stay in user space whatever the rotation, so
    the coordinates read from the text page do not change; only the lines and their
    order do.

    It MUTATES `page`: its rotation may be changed. Give it only a page of a document
    opened privately for reading its text and closed right after, never one whose
    rotation is read or that is saved: pdf_text_layer and layout.text_lines each open
    their own copy of the PDF from its bytes (immutable) for this. The bytes are not
    changed, so the page frames (boxes.page_frames, from the page's own /Rotate) and the
    files sent to the model are not affected."""
    tp = page.get_textpage()
    votes = _turn_votes(tp)
    if not votes:
        return tp
    turn, n = votes.most_common(1)[0]
    if turn == page.get_rotation() or n < UPRIGHT_SHARE * votes.total():
        return tp
    tp.close()
    page.set_rotation(turn)
    return page.get_textpage()


def pdf_text_layer(data: bytes) -> list[str]:
    """Text of each page from the PDF itself, read upright (upright_textpage). Empty
    strings for scanned PDFs."""
    pdf = pdfium.PdfDocument(data)
    try:
        out: list[str] = []
        for page in pdf:
            tp = upright_textpage(page)
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
