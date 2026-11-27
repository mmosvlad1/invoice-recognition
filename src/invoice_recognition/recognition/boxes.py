"""Where on the page each read value of a line item is printed.

A rectangle is given on the page as it is displayed: the crop box (cut to the media
box), turned by /Rotate, origin at the top-left corner, y downwards, in PDF points.
That is the space of a PDF.js viewport at scale 1 (PDF.js also multiplies by
/UserUnit, which pdfium ignores; a page placing the rectangles in proportion to its
viewport is right either way).

pdfium gives the boxes of characters in PDF user space: not turned, not moved to the
crop box. PageFrame maps them onto the displayed page. Pure Python, apart from
page_frames, which opens the PDF.

Two roads, as the flow reads a text-layer PDF:
  from_layout - line items found by code: the parser knows the very words of each cell;
  locate      - line items the model read: the printed row and values are looked up in
                the text layer afterwards, along its text lines in user space.
"row" frames the whole printed item on the layout road (its row and its wrapped
description), and only the item's own printed row on the model road. Neither road
changes a value, an issue or a request; they only say where things are printed.
"""

import logging
import math
import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass

import pypdfium2 as pdfium

from invoice_recognition.recognition import checks
from invoice_recognition.recognition import evidence as ev
from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition.layout import TextLine, Word
from invoice_recognition.shapes import Line

logger = logging.getLogger(__name__)

type Rect = tuple[float, float, float, float]
"""[x0, y0, x1, y1]: left, top, right, bottom on the displayed page, rounded outwards
to 0.1 point and held to the page: 0 <= x0 < x1 <= width, 0 <= y0 < y1 <= height."""

type LineBoxes = dict[str, Rect | None]
"""Where one line item is printed on its own page: always the keys of BOX_KEYS. "row" is
the whole printed item (its row and its wrapped description), never None; every other
key is the line's own key, None when that value is not framed."""

BOX_KEYS = ("row", *lay.FIELDS)

EDGE_TOL = 0.5  # points: how far a value may stick out of the page and still be framed


def _down(v: float) -> float:
    # round() first: 0.7 * 10 is 7.000000000000001, which ceil() would take to 0.8.
    return math.floor(round(v * 10, 6)) / 10


def _up(v: float) -> float:
    return math.ceil(round(v * 10, 6)) / 10


@dataclass(frozen=True)
class PageFrame:
    """One PDF page as displayed. (left, bottom, right, top): the view, crop box cut to
    the media box, in user space; rotation: /Rotate, clockwise, 0 / 90 / 180 / 270;
    width, height: the displayed size, rounded to 0.1."""

    left: float
    bottom: float
    right: float
    top: float
    rotation: int
    width: float
    height: float

    @classmethod
    def of(cls, view: tuple[float, float, float, float], rotation: int) -> PageFrame | None:
        """The frame of a page with this view and rotation; None when the view has no
        area (a crop box that misses the media box) or is not made of finite numbers: no
        rectangle could be placed on it.

        pdfium reports only quarter turns; any other rotation also gives None, so that
        _point never maps a page by a turn it was not given."""
        left, bottom, right, top = (float(v) for v in view)
        if rotation not in (0, 90, 180, 270) or not all(map(math.isfinite, view)):
            return None
        w, h = round(right - left, 1), round(top - bottom, 1)
        if w <= 0 or h <= 0:
            return None
        if rotation in (90, 270):
            w, h = h, w
        return cls(left, bottom, right, top, rotation, w, h)

    def _point(self, x: float, y: float) -> tuple[float, float]:
        """A point of user space on the displayed page (PDF.js PageViewport at scale 1)."""
        match self.rotation:
            case 0:
                return x - self.left, self.top - y
            case 90:
                return y - self.bottom, x - self.left
            case 180:
                return self.right - x, y - self.bottom
            case _:  # 270
                return self.top - y, self.right - x

    def to_display(self, x0: float, bottom: float, x1: float, top: float) -> Rect:
        """A box of user space on the displayed page: (x0, y0, x1, y1), not rounded and
        not held to the page."""
        ax, ay = self._point(x0, bottom)
        bx, by = self._point(x1, top)
        return min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)

    def place(self, x0: float, bottom: float, x1: float, top: float, *, cell: bool) -> Rect | None:
        """A box of user space as a rectangle of the result, rounded outwards.

        cell=True, a value: None unless it lies on the page (EDGE_TOL), then held to it;
        a value cut by the edge of the page is not claimed. cell=False, a whole row: cut
        to the page, None when nothing of it is left. A box with a coordinate that is not
        a finite number (a broken font) is never placed."""
        if not all(map(math.isfinite, (x0, bottom, x1, top))):
            return None
        dx0, dy0, dx1, dy1 = self.to_display(x0, bottom, x1, top)
        if cell and (
            dx0 < -EDGE_TOL
            or dy0 < -EDGE_TOL
            or dx1 > self.width + EDGE_TOL
            or dy1 > self.height + EDGE_TOL
        ):
            return None
        rx0, ry0 = max(0.0, _down(dx0)), max(0.0, _down(dy0))
        rx1, ry1 = min(self.width, _up(dx1)), min(self.height, _up(dy1))
        if rx0 >= rx1 or ry0 >= ry1:
            return None
        return rx0, ry0, rx1, ry1


def page_frames(data: bytes) -> list[PageFrame | None]:
    """The frame of every page of a PDF, in order. A page without a usable frame gets
    None, and the pages that have none are logged once."""
    pdf = pdfium.PdfDocument(data)
    frames: list[PageFrame | None] = []
    try:
        for page in pdf:
            frames.append(PageFrame.of(page.get_bbox(), page.get_rotation()))
    finally:
        pdf.close()
    missing = [p for p, f in enumerate(frames, start=1) if f is None]
    if missing:
        logger.warning("pages without a usable page box, nothing is framed on them: %s", missing)
    return frames


def _union(words: Iterable[Word]) -> tuple[float, float, float, float] | None:
    """The box around the words, in user space; None for no words."""
    ws = list(words)
    if not ws:
        return None
    return (
        min(w.x0 for w in ws),
        min(w.bottom for w in ws),
        max(w.x1 for w in ws),
        max(w.top for w in ws),
    )


def _place(frame: PageFrame, words: Iterable[Word], *, cell: bool) -> Rect | None:
    box = _union(words)
    return None if box is None else frame.place(*box, cell=cell)


def _frame_of(page: object, frames: list[PageFrame | None]) -> PageFrame | None:
    if not isinstance(page, int) or isinstance(page, bool) or not 1 <= page <= len(frames):
        return None
    return frames[page - 1]


# --------------------------------------------------------------------------- #
# Line items found by code
# --------------------------------------------------------------------------- #


def from_layout(
    parsed: lay.Parsed,
    layout: lay.Layout,
    frames: list[PageFrame | None],
    tol: float = lay.X_TOL,
) -> list[LineBoxes | None]:
    """Where each line item of the layout road is printed, parallel to parsed.lines.

    Exact: each item row is cut into cells again, by the same columns and tolerance the
    parser used, so a value's rectangle holds the very words its value was read from. A
    value that is None or empty, or a tax rate the layout does not print in the row, is
    not framed. The description is its cell plus every wrapped row on the item's page.
    A line on a page without a frame, or whose rows are all off the displayed page,
    gets None."""
    ix = {f: layout.index(f) for f in lay.FIELDS}
    out: list[LineBoxes | None] = []
    for item, rows in zip(parsed.lines, parsed.rows, strict=True):
        frame = _frame_of(item.get("page"), frames)
        row = (
            None if frame is None else _place(frame, (w for r in rows for w in r.words), cell=False)
        )
        if frame is None or row is None:
            out.append(None)
            continue
        cells, _ = lay.assign(rows[0].words, layout.columns, tol)
        found: LineBoxes = {"row": row}
        for key in lay.FIELDS:
            i = ix[key]
            words = list(cells.get(i, [])) if i is not None else []
            if key == "description":
                words += [w for r in rows[1:] for w in r.words]
            value = item.get(key)
            if value is None or value == "" or (key == "tax_rate" and not layout.tax_in_row):
                words = []
            found[key] = _place(frame, words, cell=True)
        out.append(found)
    return out


# --------------------------------------------------------------------------- #
# Line items read by the model
# --------------------------------------------------------------------------- #

# The values looked up in a row, in the order they claim printed words.
LOCATED = ("sku", "amount", "unit_price", "quantity", "tax_rate", "line_no")
# The values a row is recognised by (check_grounding's, which needs all of them).
ROW_KEYS = ("sku", "quantity", "unit_price", "amount")
MIN_ROW_MATCHES = 2  # values that must be printed in a row to take it as the line's row
MIN_COLUMN_ROWS = 2  # rows whose exact value gives a column its extent
# A number printed with spaces between its digit groups ("1 234,56") is several words.
# Words of one number stand closer than this, in heights of the words (a space is about
# a quarter of the type size, a gap between columns about one); at most this many.
NARROW_GAP = 0.5
MAX_GROUPS = 4
SPACED = re.compile(r"[-−]?\d{1,3}(?: \d{3})+(?:[.,]\d+)?")  # "1 234,56" as its words
# Words of one printed row overlap across the line by at least this share of the
# smaller of their heights; words of rows one under another do not overlap at all.
SAME_ROW = 0.5


@dataclass
class _Shown:
    """A word as compared, and where it lies along its text line."""

    text: str  # evidence.comparable
    word: Word
    a0: float  # along the text line, in user space: x, or y on a page drawn turned
    a1: float
    height: float  # across the text line

    @property
    def centre(self) -> float:
        return (self.a0 + self.a1) / 2


def _along_y(text_lines: list[TextLine]) -> bool:
    """The text lines of the page run along user-space y: the page is drawn turned (and
    usually shown upright by /Rotate). Told by how far the words of each line spread."""
    spread_x = spread_y = 0.0
    for tl in text_lines:
        if len(tl.words) > 1:
            spread_x += max(w.x1 for w in tl.words) - min(w.x0 for w in tl.words)
            spread_y += max(w.top for w in tl.words) - min(w.bottom for w in tl.words)
    return spread_y > spread_x


def _shown(words: Iterable[Word], along_y: bool) -> list[_Shown]:
    """Words in the order they are written along their line."""
    out = [
        _Shown(ev.comparable(w.text), w, w.bottom, w.top, w.x1 - w.x0)
        if along_y
        else _Shown(ev.comparable(w.text), w, w.x0, w.x1, w.top - w.bottom)
        for w in words
    ]
    out.sort(key=lambda w: w.a0)
    return out


def _rows(text_lines: list[TextLine], along_y: bool) -> list[list[_Shown]]:
    """The printed rows of the page, from its words: words that lie in one band across
    the line belong to one row, whichever text line pdfium put them in (it starts a new
    text line at a wide gap between columns, sometimes runs a whole table into one, and
    on a turned page groups the pieces otherwise). A word joins a row when it overlaps
    the band of the row's first word by SAME_ROW of the smaller height. Rows come in the
    order pdfium gives their first word."""

    def band(w: Word) -> tuple[float, float]:
        return (w.x0, w.x1) if along_y else (w.bottom, w.top)

    words = [(n, w) for n, w in enumerate(w for tl in text_lines for w in tl.words)]
    groups: list[tuple[float, float, list[tuple[int, Word]]]] = []
    for n, w in sorted(words, key=lambda nw: sum(band(nw[1])) / 2):
        lo, hi = band(w)
        if groups:
            glo, ghi, members = groups[-1]
            if min(hi, ghi) - max(lo, glo) >= SAME_ROW * min(hi - lo, ghi - glo):
                members.append((n, w))
                continue
        groups.append((lo, hi, [(n, w)]))
    groups.sort(key=lambda g: min(n for n, _ in g[2]))
    return [_shown((w for _, w in g[2]), along_y) for g in groups]


def _targets(item: Line) -> dict[str, set[str]]:
    """The printed forms of each value of the line, compared as the text layer is; an
    empty set for a value that is missing or not usable."""
    out: dict[str, set[str]] = {}
    sku = ev.comparable(str(item.get("sku") or "").strip())
    out["sku"] = {sku} if sku else set()
    for key in LOCATED[1:]:
        v = checks.dec(item.get(key))
        forms = {ev.comparable(f) for f in ev.printed_forms(v)} if v is not None else set()
        if key == "tax_rate":
            forms |= {f + "%" for f in forms}
        out[key] = forms
    return out


@dataclass
class _Found:
    """One line item located in one text line: the words each value claimed."""

    row: int  # index of the text line on the page
    claimed: dict[str, list[int]]  # key -> indexes of its words in the text line


def _assign_rows(targets: list[dict[str, set[str]]], rows: list[list[_Shown]]) -> list[int | None]:
    """The text line of each line item: one where at least two of its sku, quantity, unit
    price and amount are printed (all of them when it has fewer). The best matches are
    taken first - every line item whose row prints four of its values before any that
    prints three, and so on - so a line the model invented, or one with values misread,
    cannot take the row of a line read right. Among equals, lines in order, each taking
    the first free row after the row of the line before it, else the first free row.
    A row is taken once. Rows printed alike (the same values twice on a page) are told
    apart by the line number, when the line has one and it is printed in only some of
    them; otherwise by that order."""
    index: dict[str, set[int]] = {}
    texts: list[set[str]] = []
    for r, words in enumerate(rows):
        texts.append({w.text for w in words})
        for w in words:
            index.setdefault(w.text, set()).add(r)
    levels: list[dict[int, list[int]]] = []  # per line item: score -> rows, in order
    needs: list[int] = []
    for t in targets:
        keys = [k for k in ROW_KEYS if t[k]]
        needs.append(min(MIN_ROW_MATCHES, len(keys)) if keys else len(ROW_KEYS) + 1)
        score: dict[int, int] = {}
        for k in keys:
            for r in set[int]().union(*(index.get(f, set()) for f in t[k])):
                score[r] = score.get(r, 0) + 1
        by_score: dict[int, list[int]] = {}
        for r in sorted(score):
            by_score.setdefault(score[r], []).append(r)
        levels.append(by_score)
    assigned: list[int | None] = [None] * len(targets)
    used: set[int] = set()
    for level in range(len(ROW_KEYS), 0, -1):
        after = -1  # the row of the last line item before this one that has a row
        for i, by_score in enumerate(levels):
            if assigned[i] is None and level >= needs[i]:
                free = [r for r in by_score.get(level, []) if r not in used]
                numbered = [r for r in free if not targets[i]["line_no"].isdisjoint(texts[r])]
                if numbered:
                    free = numbered
                if free:
                    pick = next((r for r in free if r > after), free[0])
                    assigned[i] = pick
                    used.add(pick)
            row = assigned[i]
            if row is not None:
                after = row
    return assigned


def _exact(targets: dict[str, set[str]], words: list[_Shown]) -> dict[str, list[int]]:
    """The values printed exactly once in the row (the amount: its last match along the
    line), each word claimed by one value only, in the order of LOCATED."""
    claimed: dict[str, list[int]] = {}
    taken: set[int] = set()
    for key in LOCATED:
        hits = [j for j, w in enumerate(words) if j not in taken and w.text in targets[key]]
        if key == "amount" and hits:
            hits = hits[-1:]  # as the line-item witness takes the last number of a row
        if len(hits) == 1:
            claimed[key] = hits
            taken.add(hits[0])
    return claimed


def _columns(
    found: list[_Found | None], rows: list[list[_Shown]]
) -> dict[str, tuple[float, float]]:
    """Where each value is printed along the rows of the page: the extent of its exact
    matches that overlap their median extent - a match that lies elsewhere (a misread
    value equal to another number of its row) does not widen the column. A column needs
    at least MIN_COLUMN_ROWS such matches and half of the located rows."""
    spans: dict[str, list[tuple[float, float]]] = {}
    for f in found:
        if f is None:
            continue
        for key, js in f.claimed.items():
            ws = [rows[f.row][j] for j in js]
            spans.setdefault(key, []).append((min(w.a0 for w in ws), max(w.a1 for w in ws)))
    need = max(MIN_COLUMN_ROWS, math.ceil(sum(f is not None for f in found) / 2))
    out: dict[str, tuple[float, float]] = {}
    for key, v in spans.items():
        m0, m1 = statistics.median(s[0] for s in v), statistics.median(s[1] for s in v)
        keep = [s for s in v if s[1] > m0 and s[0] < m1]
        if len(keep) >= need:
            out[key] = (min(s[0] for s in keep), max(s[1] for s in keep))
    return out


def _check_exact(
    claimed: dict[str, list[int]], words: list[_Shown], columns: dict[str, tuple[float, float]]
) -> None:
    """Drop an exact match that is not in its column: the value was misread and happens
    to equal another number printed in the row. A value without a column is not taken
    either: nothing says the match is its cell. The sku is kept: it is matched as a whole
    code, not as a number."""
    for key in list(claimed):
        if key == "sku":
            continue
        if key in columns:
            lo, hi = columns[key]
            if all(lo <= words[j].centre <= hi for j in claimed[key]):
                continue
        del claimed[key]


def _last_number_only(claimed: dict[str, list[int]], words: list[_Shown]) -> None:
    """With no column anywhere in the document to check a match against, keep only what
    stands on its own: the sku (a whole code) and the amount when it is the last number of
    the row (as the line-item witness reads a row). A quantity, a price, a tax rate or a
    line number could be any number of the row."""
    numbers = [j for j, w in enumerate(words) if _is_number(w.text)]
    for key in list(claimed):
        if key == "sku" or (key == "amount" and numbers and claimed[key] == numbers[-1:]):
            continue
        del claimed[key]


def _is_number(text: str) -> bool:
    return any(ev.parse_number(text, locale) is not None for locale in ("us", "eu"))


def _spaced_numbers(
    words: list[_Shown], free: set[int], others: list[tuple[float, float]]
) -> list[list[int]]:
    """Runs of free words that together are one number printed with spaces between its
    digit groups ("1 234,56", "5 000,00"): standing close to each other (NARROW_GAP),
    none inside the column of another value, at most MAX_GROUPS words."""

    def joins(k: int) -> bool:  # words k and k + 1 may be parts of one number
        a, b = words[k], words[k + 1]
        return (
            k in free
            and k + 1 in free
            and b.a0 - a.a1 < NARROW_GAP * max(a.height, b.height)
            and not any(lo <= w.centre <= hi for w in (a, b) for lo, hi in others)
        )

    out = []
    for start in range(len(words)):
        end = start
        while end + 1 < len(words) and end - start + 1 < MAX_GROUPS and joins(end):
            end += 1
            if SPACED.fullmatch(" ".join(words[k].word.text for k in range(start, end + 1))):
                out.append(list(range(start, end + 1)))
    return out


def _by_column(
    targets: dict[str, set[str]],
    words: list[_Shown],
    claimed: dict[str, list[int]],
    columns: dict[str, tuple[float, float]],
) -> None:
    """Values not found exactly, framed by their column on the page: a value printed in
    the row more than once, by the one match inside its column; a value printed nowhere
    in its column (misread), by the free words that reach into the column, when they
    stand together - the printed cell a person must compare with the read value. A
    number printed with spaces between its digit groups is several words: it is matched
    and framed whole (_spaced_numbers), never in part."""
    taken = {j for js in claimed.values() for j in js}
    for key in LOCATED:
        if key in claimed or not targets[key] or key not in columns:
            continue
        lo, hi = columns[key]
        free = [j for j in range(len(words)) if j not in taken]
        others = [span for k, span in columns.items() if k != key]
        spaced = [
            run
            for run in _spaced_numbers(words, set(free), others)
            if words[run[-1]].a1 > lo and words[run[0]].a0 < hi
        ]
        whole = [run for run in spaced if "".join(words[k].text for k in run) in targets[key]]
        pick = [j for j in free if words[j].text in targets[key] and lo <= words[j].centre <= hi]
        if len(pick) + len(whole) > 1:
            continue
        pick = whole[0] if whole else pick
        if not pick:  # every word that reaches into the column
            pick = [j for j in free if words[j].a1 > lo and words[j].a0 < hi]
            if not pick or pick != list(range(pick[0], pick[-1] + 1)):
                continue
            if len(pick) > 1 and pick not in spaced:  # more than one value: not one cell
                continue
            # a part of a number printed with spaces: the whole number
            pick = next((run for run in spaced if set(pick) <= set(run)), pick)
        claimed[key] = pick
        taken.update(pick)


@dataclass
class _Page:
    """One page being located: its rows, the line items on it and where each was found."""

    frame: PageFrame
    along_y: bool
    rows: list[list[_Shown]]
    targets: list[dict[str, set[str]]]
    found: list[_Found | None]

    @property
    def located(self) -> int:
        return sum(f is not None for f in self.found)


def _start(items: list[Line], text_lines: list[TextLine], frame: PageFrame) -> _Page:
    along_y = _along_y(text_lines)
    rows = _rows(text_lines, along_y)
    targets = [_targets(item) for item in items]
    found = [
        None if r is None else _Found(r, _exact(t, rows[r]))
        for t, r in zip(targets, _assign_rows(targets, rows), strict=True)
    ]
    return _Page(frame, along_y, rows, targets, found)


def _finish(page: _Page, columns: dict[str, tuple[float, float]] | None) -> list[LineBoxes | None]:
    """The boxes of the page's lines. columns: those of the page, or borrowed from another
    page of the document; None when the document has none."""
    out: list[LineBoxes | None] = []
    for t, f in zip(page.targets, page.found, strict=True):
        if f is None:
            out.append(None)
            continue
        words = page.rows[f.row]
        if columns is None:
            _last_number_only(f.claimed, words)
        else:
            _check_exact(f.claimed, words, columns)
            _by_column(t, words, f.claimed, columns)
        row = _place(page.frame, (w.word for w in words), cell=False)
        if row is None:
            out.append(None)
            continue
        boxes: LineBoxes = {"row": row}
        for key in lay.FIELDS:
            js = f.claimed.get(key, [])
            boxes[key] = _place(page.frame, (words[j].word for j in js), cell=True)
        out.append(boxes)
    return out


def locate(
    lines: list[Line], pages: list[list[TextLine]], frames: list[PageFrame | None]
) -> list[LineBoxes | None]:
    """Where each line item read by the model is printed, parallel to `lines`: found
    afterwards in the text layer of its own page (pages[page - 1]).

    Matching works along the text lines in user space, never on the displayed page: on a
    page turned by /Rotate the displayed X of the words of a row is the same, or runs
    against the reading order. Display coordinates only place the rectangles.

    Row: a printed row of the page (its words, whichever text lines pdfium put them in)
    where at least two of the line's sku, quantity, unit price and amount are printed
    (check_grounding's rule, relaxed from all of them, so a line with one misread value
    is still found); the best matches are taken first (see _assign_rows). "row" frames
    that printed row only: a wrapped description is not looked for. Values: sku, amount,
    unit price, quantity, tax rate and line number by their printed forms in that row,
    kept only inside their column; one not printed there exactly is framed by its column
    (see _by_column). Columns come from the page's own rows; a page with one located
    line borrows those of the nearest page that has them. The description and the unit
    are not looked for: None. A line that is not found, has no page, or is on a page
    without a frame gets None. Never raises for a value of any JSON type.

    Limits, all measured on generated invoices and their mutated readings:
      - rows printed alike are told apart by the line number when it is printed, else
        by order only; a line whose misread value matches a number of its twin row (a
        quantity read as the twin's line number) can take that row;
      - when no page of the document has two located lines (a one-line invoice), there
        is no column to check a match against: only "row", the sku and an amount printed
        last in its row are framed;
      - a number printed with spaces between its digit groups ("1 234,56") is matched
        and framed only through its column (up to MAX_GROUPS words), never on a page
        without columns;
      - a minus printed after the number ("12.00-") is no printed form of the value: such
        a cell is framed only through its column.
    """
    out: list[LineBoxes | None] = [None] * len(lines)
    by_page: dict[int, tuple[PageFrame, list[int]]] = {}
    for i, item in enumerate(lines):
        page = item.get("page") if isinstance(item, dict) else None
        frame = _frame_of(page, frames)
        if frame is not None and isinstance(page, int) and page <= len(pages):
            by_page.setdefault(page, (frame, []))[1].append(i)
    work = {
        page: _start([lines[i] for i in idxs], pages[page - 1], frame)
        for page, (frame, idxs) in by_page.items()
    }
    own = {
        page: _columns(w.found, w.rows) for page, w in work.items() if w.located >= MIN_COLUMN_ROWS
    }
    for page, (_, idxs) in by_page.items():
        columns = own.get(page)
        if columns is None:  # one located row: the columns of the nearest page that has them
            near = sorted(
                (abs(other - page), other)
                for other, cols in own.items()
                if cols and work[other].along_y == work[page].along_y
            )
            columns = own[near[0][1]] if near else None
        for i, b in zip(idxs, _finish(work[page], columns), strict=True):
            out[i] = b
    return out
