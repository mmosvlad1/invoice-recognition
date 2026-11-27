"""Where each line is printed, as the API shows it: lines[].boxes and the page sizes. pdfium
is the witness of the rectangles (tests/boxes_help.py). Fake OpenRouter, no network."""

import json
import logging
import time
from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest

from invoice_recognition.api.app import create_app
from invoice_recognition.api.views import result_view
from invoice_recognition.recognition import evidence as ev
from invoice_recognition.recognition.boxes import BOX_KEYS, LOCATED, PageFrame
from invoice_recognition.recognition.results import PageResult, Result
from invoice_recognition.shapes import JSONObject

import drawn
from boxes_help import Oracle, variant
from fakes import api_settings, header_for, line, run
from golden_cases import Model, Status
from samples import INVOICES, saved_layout, truth_doc

PDF = "application/pdf"


def truth_model(name: str, layout: Any = None) -> Model:
    _, _, header, pages = truth_doc(name)
    return Model([header], {p: [rows] for p, rows in pages.items()}, layout)


def dumped(res: Result) -> JSONObject:
    out: JSONObject = json.loads(result_view(res).model_dump_json())
    return out


def _printed(text: str, value: object) -> bool:
    """The text under a rectangle is a way the value is printed (or the sku itself)."""
    word = ev.comparable(text)
    if isinstance(value, str):
        return word == ev.comparable(value)
    forms = {ev.comparable(f) for f in ev.printed_forms(Decimal(str(value)))}
    return word in forms or word.rstrip("%") in forms


@pytest.mark.parametrize(
    ("kind", "layout", "road", "size"),
    [
        ("orig", saved_layout("s_classic"), "layout", [595.3, 841.9]),
        ("orig", Status(500), "model", [595.3, 841.9]),
        ("rot90", Status(500), "model", [841.9, 595.3]),
    ],
)
def test_a_text_layer_pdf_shows_where_each_value_is_printed(
    kind: str, layout: Any, road: str, size: list[float]
) -> None:
    raw, _, _, _ = truth_doc("s_classic")
    data = raw if kind == "orig" else variant(raw, kind)
    res = run(data, PDF, truth_model("s_classic", layout))
    assert res.method == road
    out = dumped(res)
    assert out["pages"] == [
        {
            "page": 1,
            "method": road,
            "line_count": 14,
            "repaired_count": 0,
            "width": size[0],
            "height": size[1],
            "text_layer": True,
        }
    ]
    framed = 0
    with Oracle(data) as oracle:
        for item, b in zip(out["lines"], res.boxes, strict=True):
            boxes = item["boxes"]
            assert list(boxes) == list(BOX_KEYS)
            assert boxes == {k: None if r is None else list(r) for k, r in (b or {}).items()}
            for key in LOCATED:
                rect = boxes[key]
                if rect is not None:
                    text = oracle.text_under(item["page"], tuple(rect))
                    assert _printed(text, item[key]), (key, text, item[key])
                    framed += 1
            if road == "model":
                assert boxes["description"] is None and boxes["unit"] is None
            else:
                assert boxes["description"] is not None and boxes["unit"] is not None
    assert framed >= 14 * 5


def test_a_scan_has_page_sizes_and_no_boxes() -> None:
    _, _, header, pages = truth_doc("s_classic")
    data = (INVOICES / "s_classic.scan.pdf").read_bytes()
    out = dumped(run(data, PDF, Model([header], {p: [r] for p, r in pages.items()})))
    assert out["lines"] and all(item["boxes"] is None for item in out["lines"])
    [page] = out["pages"]
    assert page["text_layer"] is False
    assert isinstance(page["width"], float) and isinstance(page["height"], float)


def test_a_photo_has_neither_page_sizes_nor_boxes() -> None:
    rows = [line(1), line(2)]
    res = run(drawn.jpeg(drawn.photo()), "image/jpeg", Model([header_for(rows)], {1: [rows]}))
    out = dumped(res)
    assert [item["boxes"] for item in out["lines"]] == [None, None]
    assert [(p["width"], p["height"], p["text_layer"]) for p in out["pages"]] == [
        (None, None, False)
    ]


def test_a_page_without_a_frame_has_no_size_and_a_failed_result_has_no_pages() -> None:
    res = run(INVOICES.joinpath("s_classic.pdf").read_bytes(), PDF, truth_model("s_classic"))
    out = dumped(replace(res, frames=[None]))
    assert (out["pages"][0]["width"], out["pages"][0]["height"]) == (None, None)
    assert out["pages"][0]["text_layer"] is True
    # frames of an odd kind, or not finite: no size either
    nan = PageFrame(0, 0, 1, 1, 0, float("nan"), -1.0)
    assert dumped(replace(res, frames=[nan]))["pages"][0]["width"] is None
    flat = PageFrame(0, 0, 0, 1, 0, 0.0, 1.0)  # a side of 0 is no page either
    page = dumped(replace(res, frames=[flat]))["pages"][0]
    assert (page["width"], page["height"]) == (None, 1.0)
    failed = dumped(run(b"%PDF-1.7 broken", PDF, Model([{}], {})))
    assert failed["pages"] == [] and failed["lines"] == []


# --------------------------------------------------------------------------- #
# Boxes that do not fit never cost the document
# --------------------------------------------------------------------------- #

GOOD = {k: None for k in BOX_KEYS} | {"row": (10.0, 20.0, 300.0, 30.0)}
HOSTILE: dict[str, Any] = {
    "nan": GOOD | {"amount": (1.0, 2.0, float("nan"), 4.0)},
    "infinity": GOOD | {"row": (0.0, 0.0, float("inf"), 4.0)},
    "three numbers": GOOD | {"sku": (1.0, 2.0, 3.0)},
    "a list, not a tuple": GOOD | {"sku": [1.0, 2.0, 3.0, 4.0]},
    "a string": GOOD | {"quantity": "1, 2, 3, 4"},
    "no row": GOOD | {"row": None},
    "right to left": GOOD | {"amount": (5.0, 2.0, 1.0, 4.0)},
    "negative": GOOD | {"amount": (-5.0, 2.0, 1.0, 4.0)},
    "a key too many": GOOD | {"colour": (1.0, 2.0, 3.0, 4.0)},
    "a key too few": {k: v for k, v in GOOD.items() if k != "unit"},
    "not an object": [1.0, 2.0, 3.0, 4.0],
    "text": "boxes",
}


def _three_lines(**kw: Any) -> Result:
    rows = [line(1) | {"page": 1}, line(2) | {"page": 1}, line(3) | {"page": 1}]
    page = PageResult(1, rows, method="model")
    return Result(
        status="ok",
        header=header_for(rows),
        lines=rows,
        checkpoints=[],
        issues=[],
        pages=[page],
        evidence_source="text_layer",
        duration_s=1.0,
        calls=1,
        cost_usd=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        frames=[PageFrame.of((0, 0, 595.3, 841.9), 0)],
        text_layer=True,
        **kw,
    )


@pytest.mark.parametrize("hostile", HOSTILE)
def test_boxes_that_do_not_fit_cost_their_line_only(
    hostile: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        out = dumped(_three_lines(boxes=[GOOD, HOSTILE[hostile], GOOD]))
    assert len(out["lines"]) == 3 and out["header"]["supplier_name"] == "ACME Ltd"
    good = {k: None if r is None else list(r) for k, r in GOOD.items()}
    assert [item["boxes"] for item in out["lines"]] == [good, None, good]
    assert [r.getMessage()[:31] for r in caplog.records] == ["the boxes of line 1 are dropped"]


@pytest.mark.parametrize("boxes", [[GOOD], [GOOD] * 4, "boxes", None])
def test_boxes_not_parallel_to_the_lines_are_not_shown(
    boxes: Any, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        out = dumped(_three_lines(boxes=boxes))
    assert [item["boxes"] for item in out["lines"]] == [None, None, None]
    assert [r.levelno for r in caplog.records] == ([] if boxes is None else [logging.ERROR])


def test_a_key_boxes_the_model_sent_does_not_stay() -> None:
    res = _three_lines(boxes=[None, None, None])
    res.lines[0]["boxes"] = {"row": [1, 2, 3, 4]}
    assert dumped(res)["lines"][0]["boxes"] is None


# --------------------------------------------------------------------------- #
# The schema
# --------------------------------------------------------------------------- #


def test_openapi_shows_boxes_and_page_sizes_as_always_sent() -> None:
    schemas = create_app(api_settings()).openapi()["components"]["schemas"]
    line_view = next(v for k, v in schemas.items() if k.startswith("LineView"))
    assert "boxes" in line_view["required"]
    boxes = schemas["BoxesView"]
    assert set(boxes["required"]) == set(BOX_KEYS)
    assert "wrapped description" in boxes["properties"]["row"]["description"]
    assert boxes["properties"]["row"]["$ref"] == "#/components/schemas/Rect"
    assert boxes["properties"]["sku"]["anyOf"] == [
        {"$ref": "#/components/schemas/Rect"},
        {"type": "null"},
    ]
    rect = schemas["Rect"]  # an array of four numbers
    assert rect["type"] == "array" and len(rect["prefixItems"]) == 4
    assert rect["prefixItems"] == [{"$ref": "#/components/schemas/Coordinate"}] * 4
    assert schemas["Coordinate"] == {"type": "number", "minimum": 0}
    assert (rect["minItems"], rect["maxItems"]) == (4, 4)
    page_view = schemas["PageView"]
    assert {"width", "height", "text_layer"} <= set(page_view["required"])


def test_the_largest_sample_answers_with_its_boxes_in_good_time() -> None:
    """xl_grouped: 1,250 lines over 34 pages, on the layout road."""
    data = (INVOICES / "xl_grouped.pdf").read_bytes()
    res = run(data, PDF, truth_model("xl_grouped", saved_layout("xl_grouped")))
    assert res.method == "layout" and all(b is not None for b in res.boxes)
    started = time.perf_counter()
    text = result_view(res).model_dump_json()
    took = time.perf_counter() - started
    without = result_view(replace(res, boxes=[None] * len(res.lines))).model_dump_json()
    added = len(text) - len(without)
    assert 0.2e6 < added < 0.8e6  # measured: about 0.37 MB
    assert took < 5.0  # measured: well under 0.1 s; the bound is for a slow machine
