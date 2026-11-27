"""Where each line is printed, in the flow's result: from the cells found by code, found in
the text layer for pages the model read, nothing for scans and photos - and never a change
to anything else the flow returns or sends. Fake OpenRouter, no network."""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest

from invoice_recognition.recognition import boxes as bx
from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.results import PageResult, Result
from invoice_recognition.shapes import JSONObject, Line

from boxes_help import Oracle, layout_of, variant
from drawn import jpeg, photo, scanned
from fakes import make_pdf, run
from golden_cases import CASES, Model, Status, header_for, line, request_of, result_of
from samples import INVOICES, saved_layout, truth_doc

PDF = "application/pdf"


def truth_model(name: str, layout: Any = None) -> Model:
    """A model that reads every page of the sample right, and describes its layout so."""
    _, _, header, pages = truth_doc(name)
    return Model([header], {p: [rows] for p, rows in pages.items()}, layout)


def sample(name: str, kind: str = "orig") -> bytes:
    data = (INVOICES / f"{name}.pdf").read_bytes()
    return data if kind == "orig" else variant(data, kind)


def code_boxes(data: bytes, name: str) -> list[bx.LineBoxes | None]:
    """from_layout on the document, as the flow runs the layout road."""
    pages = lay.text_lines(data)
    layout = layout_of(data, pages, saved_layout(name))
    return bx.from_layout(lay.parse_document(pages, layout), layout, bx.page_frames(data))


def located(data: bytes, lines: list[Line]) -> list[bx.LineBoxes | None]:
    return bx.locate(lines, lay.text_lines(data), bx.page_frames(data))


def test_layout_road_frames_the_cells_found_by_code() -> None:
    data = sample("s_classic")
    res = run(data, PDF, truth_model("s_classic", saved_layout("s_classic")))
    assert res.layout["status"] == "confirmed" and res.method == "layout"
    assert res.boxes == code_boxes(data, "s_classic")
    assert all(b is not None for b in res.boxes) and len(res.boxes) == len(res.lines) == 14
    assert res.frames == bx.page_frames(data) and len(res.frames) == 1
    assert res.text_layer is True


def _same_values(model: list[bx.LineBoxes | None], code: list[bx.LineBoxes | None]) -> None:
    """The model road frames the same cells as the layout road, and its row (the item row
    alone) lies inside the layout road's row (which also holds the wrapped description)."""
    for m, c in zip(model, code, strict=True):
        assert m is not None and c is not None
        for key in bx.LOCATED:
            assert m[key] == c[key], key
        mr, cr = m["row"], c["row"]
        assert mr is not None and cr is not None
        assert cr[0] <= mr[0] and cr[1] == mr[1] and mr[2] <= cr[2] and mr[3] <= cr[3]


def test_a_refused_layout_answer_has_every_page_located() -> None:
    data = sample("s_classic")
    res = run(data, PDF, truth_model("s_classic", Status(500)))
    assert res.layout["status"] == "failed" and res.method == "model"
    assert res.boxes == located(data, res.lines)
    _same_values(res.boxes, code_boxes(data, "s_classic"))


def test_a_page_whose_model_re_read_wins_is_located_the_others_come_from_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The code parser misses a line of page 2: the page fails its coverage check, the
    model reads it again and its reading is kept."""
    real = lay.parse_document

    def lossy(pages: list[list[lay.TextLine]], layout: lay.Layout) -> lay.Parsed:
        parsed = real(pages, layout)
        victim = next(i for i, item in enumerate(parsed.lines) if item["page"] == 2)
        del parsed.lines[victim], parsed.rows[victim]
        return parsed

    data = sample("m_grouped")
    # before the parser is made to lose a line: the boxes of every line found by code
    pages = lay.text_lines(data)
    code_lines = real(pages, layout_of(data, pages, saved_layout("m_grouped"))).lines
    expected = code_boxes(data, "m_grouped")
    monkeypatch.setattr(lay, "parse_document", lossy)
    res = run(data, PDF, truth_model("m_grouped", saved_layout("m_grouped")))
    assert [p.method for p in res.pages] == ["layout", "model", "layout", "layout"]
    on_2 = [i for i, item in enumerate(res.lines) if item["page"] == 2]
    page_2 = [res.lines[i] for i in on_2]
    assert [res.boxes[i] for i in on_2] == located(data, page_2)
    assert all(res.boxes[i] is not None for i in on_2)
    others = [b for i, b in enumerate(res.boxes) if i not in on_2]
    assert others == [b for b, item in zip(expected, code_lines) if item["page"] != 2]
    assert len(others) > 50


def golden(name: str) -> Callable[[], tuple[tuple[bytes, str], Model, Config]]:
    case = next(c for c in CASES if c.name == name)
    return lambda: (case.data(), case.model(), Config(**case.config))


def test_layout_switched_off_has_every_line_located() -> None:
    (data, mime), model, cfg = golden("text_layer_carry")()
    res = run(data, mime, model, cfg)
    assert res.layout["status"] == "off" and res.layout["reason"] == "disabled"
    assert model.calls["layout"] == 0
    assert len(res.boxes) == len(res.lines) > 100
    assert all(b is not None for b in res.boxes)
    assert res.boxes == located(data, res.lines)


def test_a_landscape_page_shown_upright_is_located_as_the_upright_page() -> None:
    """The layout road cannot read a page drawn turned and shown upright (every word of a
    row has the same user-space X), as before: the model reads it, and its lines are
    framed where they are seen."""
    res = run(
        sample("s_classic", "landscape"), PDF, truth_model("s_classic", saved_layout("s_classic"))
    )
    assert res.layout["status"] == "failed" and "LayoutError" in res.layout["reason"]
    upright = run(sample("s_classic"), PDF, truth_model("s_classic", Status(500)))
    assert res.lines == upright.lines
    assert len(res.boxes) == len(upright.boxes) == 14
    for a, b in zip(res.boxes, upright.boxes, strict=True):
        assert a is not None and b is not None
        for key in bx.BOX_KEYS:
            ra, rb = a[key], b[key]
            assert (ra is None) == (rb is None)
            if ra is not None and rb is not None:
                assert ra == pytest.approx(rb, abs=0.11)


def test_a_one_line_invoice_frames_its_row_sku_and_amount() -> None:
    """One page, one line item, its unit price read as the amount: nothing in the
    document gives a column, so only what stands on its own is framed."""
    rows = [
        "ACME Ltd   Invoice INV-1   2026-11-01   Payment within 14 days",
        "No  SKU  Description  Qty  Price  Amount",
        "1  SKU-001  Hex bolt M8 zinc  2  10.00  20.00",
        "Total 20.00",
    ]
    item = line(1, qty=2, price=20.0, amount=20.0)
    data = make_pdf([rows])
    res = run(data, PDF, Model([header_for([item])], {1: [[item]]}), Config(use_layout=False))
    assert res.text_layer and len(res.lines) == 1
    [b] = res.boxes
    assert b is not None
    with Oracle(data) as oracle:
        texts = {k: oracle.text_under(1, r) for k, r in b.items() if r is not None}
    assert texts == {"row": rows[2].replace("  ", " "), "sku": "SKU-001", "amount": "20.00"}


def test_scans_have_pages_but_no_rectangles() -> None:
    _, _, header, pages = truth_doc("s_classic")
    data = (INVOICES / "s_classic.scan.pdf").read_bytes()
    res = run(data, PDF, Model([header], {p: [rows] for p, rows in pages.items()}))
    assert res.lines and res.boxes == [None] * len(res.lines)
    assert len(res.frames) == 1 and res.frames[0] is not None and res.text_layer is False

    rows = [line(1), line(2)]
    res = run(scanned("good", "blank"), PDF, Model([header_for(rows)], {1: [rows]}))
    assert [p.method for p in res.pages] == ["model", "skipped"]
    assert res.boxes == [None, None] and res.text_layer is False
    assert [f is not None for f in res.frames] == [True, True]


def test_a_photo_has_no_frames_and_no_rectangles() -> None:
    rows = [line(1), line(2), line(3)]
    res = run(jpeg(photo()), "image/jpeg", Model([header_for(rows)], {1: [rows]}))
    assert len(res.lines) == 3 and res.boxes == [None, None, None]
    assert res.frames == [] and res.text_layer is False


# --------------------------------------------------------------------------- #
# A failure while framing changes nothing else
# --------------------------------------------------------------------------- #


def _boom(*args: object, **kwargs: object) -> Any:
    raise RuntimeError("framing broke")


def _outcome(res: Result, model: Model) -> tuple[JSONObject, list[JSONObject]]:
    return result_of(res.to_json()), [request_of(b) for b in model.bodies]


@pytest.mark.parametrize(
    ("broken", "case", "messages"),
    [
        ("from_layout", "s_classic.layout", ["cells of the rows found by code could not be"]),
        (
            "locate",
            "layout_failed",
            ["lines could not be located together", "lines of page 1 could not be located"],
        ),
        ("page_frames", "s_classic.layout", ["page boxes could not be read"]),
        ("page_frames", "layout_failed", ["page boxes could not be read"]),
    ],
)
def test_a_failure_while_framing_changes_nothing_else(
    broken: str,
    case: str,
    messages: list[str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data = sample("s_classic")
    layout: Any = Status(500) if case == "layout_failed" else saved_layout("s_classic")
    clean_model = truth_model("s_classic", layout)
    clean = run(data, PDF, clean_model)
    assert all(b is not None for b in clean.boxes)

    monkeypatch.setattr(bx, broken, _boom)
    model = truth_model("s_classic", layout)
    with caplog.at_level(logging.WARNING):
        res = run(data, PDF, model)
    assert _outcome(res, model) == _outcome(clean, clean_model)
    assert (res.status, res.lines, res.notes) == (clean.status, clean.lines, clean.notes)
    assert [i.to_json() for i in res.issues] == [i.to_json() for i in clean.issues]
    assert res.boxes == [None] * len(res.lines)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == len(messages), warnings
    assert all(m in w for m, w in zip(messages, warnings, strict=True)), warnings
    if case != "layout_failed":
        assert res.method == "layout"  # the layout road is still taken
    if broken == "page_frames":
        assert res.frames == []


def test_a_page_that_cannot_be_located_costs_its_own_rectangles_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = bx.locate

    def broken_on_page_2(lines: list[Line], *args: Any) -> list[bx.LineBoxes | None]:
        if any(item["page"] == 2 for item in lines):
            raise RuntimeError("framing broke")
        return real(lines, *args)

    data = sample("m_grouped")
    monkeypatch.setattr(bx, "locate", broken_on_page_2)
    with caplog.at_level(logging.WARNING):
        res = run(data, PDF, truth_model("m_grouped", Status(500)))
    on_2 = [item["page"] == 2 for item in res.lines]
    assert any(on_2) and not all(on_2)
    assert [b is None for b in res.boxes] == on_2
    # every other page located on its own, as when located with the others
    others = [item for item in res.lines if item["page"] != 2]
    assert [b for b, two in zip(res.boxes, on_2) if not two] == real(others, *_text(data))
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [
        "lines could not be located together; page by page",
        "lines of page 2 could not be located",
    ]


def test_a_page_whose_boxes_do_not_fit_its_lines_costs_its_own_rectangles_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = bx.locate

    def short_on_page_2(lines: list[Line], *args: Any) -> list[bx.LineBoxes | None]:
        pages = {item["page"] for item in lines}
        if len(pages) > 1:
            raise RuntimeError("framing broke")  # together: tried again page by page
        return [] if pages == {2} else real(lines, *args)

    monkeypatch.setattr(bx, "locate", short_on_page_2)
    with caplog.at_level(logging.WARNING):
        res = run(sample("m_grouped"), PDF, truth_model("m_grouped", Status(500)))
    on_2 = [item["page"] == 2 for item in res.lines]
    assert [b is None for b in res.boxes] == on_2
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [
        "lines could not be located together; page by page",
        "lines of page 2 could not be framed",
    ]


def _text(data: bytes) -> tuple[list[list[lay.TextLine]], list[bx.PageFrame | None]]:
    return lay.text_lines(data), bx.page_frames(data)


def test_a_failure_outside_the_pages_costs_every_rectangle(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(pl, "_layout_boxes", _boom)
    with caplog.at_level(logging.WARNING):
        res = run(sample("s_classic"), PDF, truth_model("s_classic", saved_layout("s_classic")))
    assert res.method == "layout" and res.status == "ok"
    assert res.boxes == [None] * len(res.lines) == [None] * 14
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == ["lines could not be located on their pages"]


def _thread_of_locate(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    real = bx.locate

    def spy(*args: Any) -> list[bx.LineBoxes | None]:
        seen.append(threading.current_thread().name)
        return real(*args)

    monkeypatch.setattr(bx, "locate", spy)
    return seen


def test_lines_are_located_on_the_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _thread_of_locate(monkeypatch)
    data = sample("s_classic")

    async def go() -> Result:
        transport = httpx.MockTransport(truth_model("s_classic", Status(500)))
        with ThreadPoolExecutor(1, thread_name_prefix="lane") as lane:
            async with httpx.AsyncClient(transport=transport) as client:
                return await pl.process(data, PDF, client, Config(), lane=lane)

    res = asyncio.run(go())
    assert all(b is not None for b in res.boxes)
    assert len(seen) == 1 and seen[0].startswith("lane")


def _read_pages(data: bytes) -> tuple[list[PageResult], pl.Prepared]:
    prep = pl.prepare(data, PDF, Config())
    _, _, _, pages = truth_doc("s_classic")
    return [PageResult(1, [r | {"page": 1} for r in pages[1]])], prep


def test_no_lines_are_located_after_the_time_limit(caplog: pytest.LogCaptureFixture) -> None:
    pages, prep = _read_pages(sample("s_classic"))
    late = pl.Gate(deadline=time.monotonic() - 1)
    with caplog.at_level(logging.WARNING):
        found = asyncio.run(pl._frame_lines(pages, prep, None, None, late))
    assert found == [None] * 14
    assert "time limit passed before" in caplog.records[-1].getMessage()

    async def busy_lane() -> list[bx.LineBoxes | None]:
        with ThreadPoolExecutor(1) as lane:
            blocker = threading.Event()
            lane.submit(blocker.wait, 5)  # the lane is opening another document
            gate = pl.Gate(deadline=time.monotonic() + 0.2)
            try:
                return await pl._frame_lines(pages, prep, None, lane, gate)
            finally:
                blocker.set()

    with caplog.at_level(logging.WARNING):
        found = asyncio.run(busy_lane())
    assert found == [None] * 14
    assert "time limit passed while" in caplog.records[-1].getMessage()
    # with time left, the same lines are located
    in_time = asyncio.run(pl._frame_lines(pages, prep, None, None, pl.Gate()))
    assert all(b is not None for b in in_time)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_every_golden_case_has_one_entry_per_line(case: Any) -> None:
    (data, mime), model, cfg = golden(case.name)()
    res = run(data, mime, model, cfg)
    assert len(res.boxes) == len(res.lines)
