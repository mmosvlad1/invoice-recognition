"""A PDF with a text layer in the flow: layout from the model, rows by code, the witness
page and the fallback to reading every page by the model. Fake OpenRouter, no network."""

import asyncio
import io
import json

import httpx
import pytest
from pypdf import PdfReader, PdfWriter

from invoice_recognition.recognition import evidence as ev
from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.results import PageResult, Result
from invoice_recognition.shapes import JSONObject, Line

from drawn import scan, scan_with_invisible_ocr, scan_with_ocr_text, sheet
from fakes import Answer, Server, codes, header_for, line, run, sse
from samples import WITH_LAYOUT, exact, saved_layout, truth_doc


def swapped(layout: JSONObject) -> JSONObject:
    flip = {"quantity": "unit_price", "unit_price": "quantity"}
    columns = [c | {"field": flip.get(c["field"], c["field"])} for c in layout["columns"]]
    return layout | {"columns": columns}


def answers(pages: dict[int, list[Line]]) -> dict[int, list[Answer]]:
    return {p: [rows] for p, rows in pages.items()}


def lose_a_line(monkeypatch: pytest.MonkeyPatch, page: int) -> None:
    """Make the code parser miss one line item of `page`, as an irregular row would."""
    real = lay.parse_document

    def lossy(pages: list[list[lay.TextLine]], layout: lay.Layout) -> lay.Parsed:
        parsed = real(pages, layout)
        victim = next(i for i, item in enumerate(parsed.lines) if item["page"] == page)
        del parsed.lines[victim]
        return parsed

    monkeypatch.setattr(lay, "parse_document", lossy)


def test_layout_switched_off_reads_every_page_by_the_model() -> None:
    data, truth, header, pages = truth_doc("s_classic")
    server = Server(header, answers(pages))
    res = run(data, "application/pdf", server, Config(use_layout=False))

    assert res.evidence_source == "text_layer"
    assert [i.to_json() for i in res.issues] == []
    assert res.status == "ok"
    assert [item["sku"] for item in res.lines] == [item["sku"] for item in truth["lines"]]
    assert all(len(p.attempts) == 1 for p in res.pages)
    assert res.calls == len(res.pages) + 1 and res.cost_unknown_calls == 0
    assert server.calls["header"] == 1 and server.calls["layout"] == 0
    assert res.method == "model" and res.layout["status"] == "off" and res.notes == []
    assert res.layout["reason"] == "disabled"


@pytest.mark.parametrize("name", WITH_LAYOUT)
def test_layout_road_happy_path(name: str) -> None:
    data, truth, header, pages = truth_doc(name)
    server = Server(header, answers(pages), saved_layout(name))
    res = run(data, "application/pdf", server)

    assert [i.to_json() for i in res.issues] == []
    assert res.status == "ok" and exact(res.lines, truth["lines"])
    descriptions = [item["description"] for item in truth["lines"]]
    assert [item["description"] for item in res.lines] == descriptions
    assert [item["page"] for item in res.lines] == [item["page"] for item in truth["lines"]]
    assert res.method == "layout" and all(p.method == "layout" for p in res.pages)
    assert res.layout["status"] == "confirmed" and res.layout["witness_page"] == 1
    assert res.layout["same"] == res.layout["compared"] == len(pages[1])
    assert res.layout["pages_layout"] == len(res.pages) and res.layout["pages_model"] == 0
    # header + layout + the witness page, whatever the length of the document
    assert res.calls == 3 and dict(server.calls) == {"header": 1, "layout": 1, 1: 1}
    assert res.cost_usd == pytest.approx(0.003) and len(res.layout_attempts) == 1
    assert len(res.pages[0].attempts) == 1 and all(p.attempts == [] for p in res.pages[1:])
    out = res.to_json()
    assert out["method"] == "layout" and {p["method"] for p in out["pages"]} == {"layout"}
    assert out["layout"]["status"] == "confirmed" and len(out["layout_attempts"]) == 1
    json.dumps(out)


def test_undeclared_subtotal_rows_are_notes_not_issues() -> None:
    data, _, header, pages = truth_doc("xl_grouped")
    res = run(data, "application/pdf", Server(header, answers(pages), saved_layout("xl_grouped")))
    assert res.status == "ok" and res.issues == []
    assert len(res.notes) == 57 and {n["kind"] for n in res.notes} == {"unclassified_amount"}
    assert all(n["text"].startswith("Subtotal DN-") for n in res.notes)
    assert res.to_json()["notes"] == res.notes


@pytest.mark.parametrize("name", ["l_classic", "m_grouped"])
def test_swapped_quantity_and_price_send_the_document_to_the_model(name: str) -> None:
    data, truth, header, pages = truth_doc(name)
    server = Server(header, answers(pages), swapped(saved_layout(name)))
    res = run(data, "application/pdf", server)

    flagged = [i for i in res.issues if i.check.startswith("layout_")]
    assert [i.check for i in flagged] == ["layout_swapped"] and flagged[0].page == 1
    assert res.layout["status"] == "swapped" and res.layout["same"] == 0
    assert res.layout["crossed"] * 2 > res.layout["compared"]
    assert exact(res.lines, truth["lines"])  # the final lines are the model's
    assert res.method == "model" and res.notes == []
    assert all(server.calls[p] == 1 for p in pages)  # the witness page is not read again
    assert res.calls == len(pages) + 2
    assert res.status == "needs_review"  # two readings disagree about what is quantity


@pytest.mark.parametrize("answer", ["not json", "http500", "no columns", "wrong label"])
def test_unusable_layout_answer_falls_back_to_the_model(answer: str) -> None:
    data, truth, header, pages = truth_doc("l_classic")
    good = saved_layout("l_classic")
    wrong_label = [
        c | {"label": c["label"].replace("Unit price", "Price per unit")} for c in good["columns"]
    ]
    options: dict[str, Answer] = {
        "not json": sse("Sure! Here is the layout: {"),
        "http500": httpx.Response(500, text="upstream error"),
        "no columns": {"header_line": 13, "numbers": {"decimal": ".", "thousands": ","}},
        "wrong label": good | {"columns": wrong_label},
    }
    server = Server(header, answers(pages), options[answer])
    res = run(data, "application/pdf", server)

    assert res.layout["status"] == "failed" and res.layout["reason"]
    assert res.status == "ok" and res.issues == [] and exact(res.lines, truth["lines"])
    assert res.method == "model" and all(p.method == "model" for p in res.pages)
    assert all(server.calls[p] == 1 for p in pages) and server.calls["layout"] == 1
    assert res.calls == len(pages) + 2
    assert res.cost_unknown_calls == (1 if answer == "http500" else 0)


def test_page_that_fails_its_checks_is_read_by_the_model_and_only_that_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, truth, header, pages = truth_doc("l_classic")
    lose_a_line(monkeypatch, 3)
    server = Server(header, answers(pages), saved_layout("l_classic"))
    seen: list[PageResult] = []

    async def go() -> Result:
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            return await pl.process(data, "application/pdf", client, None, on_page=seen.append)

    res = asyncio.run(go())
    assert res.status == "ok" and res.issues == [] and exact(res.lines, truth["lines"])
    assert [p.method for p in res.pages] == ["layout", "layout", "model"] + ["layout"] * 7
    assert res.method == "mixed" and res.layout["pages_model"] == 1
    assert res.layout["pages_layout"] == 9
    assert dict(server.calls) == {"header": 1, "layout": 1, 1: 1, 3: 1} and res.calls == 4
    # every page is announced once, in its final form
    assert sorted(r.page for r in seen) == list(range(1, 11))


def test_witness_page_that_fails_its_checks_is_not_read_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, truth, header, pages = truth_doc("l_classic")
    lose_a_line(monkeypatch, 1)
    server = Server(header, answers(pages), saved_layout("l_classic"))
    res = run(data, "application/pdf", server)
    assert res.layout["status"] == "confirmed"  # 24 of 25 rows still agree
    assert res.status == "ok" and exact(res.lines, truth["lines"])
    assert res.pages[0].method == "model" and res.method == "mixed"
    assert dict(server.calls) == {"header": 1, "layout": 1, 1: 1} and res.calls == 3


def test_model_reread_that_is_no_better_keeps_the_rows_found_by_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, _, header, pages = truth_doc("l_classic")
    lose_a_line(monkeypatch, 3)
    failing = answers(pages) | {3: [httpx.Response(500, text="boom")]}
    res = run(data, "application/pdf", Server(header, failing, saved_layout("l_classic")))
    # both model attempts are accounted for
    assert res.pages[2].method == "layout" and len(res.pages[2].attempts) == 2
    # the page stays flagged (the document check adds "numbering")
    assert codes(res.pages[2].issues) == ["coverage"]
    assert res.status == "needs_review" and res.calls == 5 and res.cost_unknown_calls == 2


def test_witness_that_disagrees_in_another_way_is_a_mismatch() -> None:
    data, _, header, pages = truth_doc("l_classic")
    wrong = [r | {"quantity": r["quantity"] + 1} for r in pages[1]]
    server = Server(header, answers(pages) | {1: [wrong]}, saved_layout("l_classic"))
    res = run(data, "application/pdf", server)
    assert res.layout["status"] == "mismatch" and "layout_mismatch" in codes(res.issues)
    assert res.method == "model" and all(server.calls[p] >= 1 for p in pages)
    assert res.status == "needs_review"


def test_witness_without_rows_leaves_the_layout_unverified() -> None:
    data, truth, header, pages = truth_doc("l_classic")
    failing = answers(pages) | {1: [httpx.Response(500, text="boom")]}
    res = run(data, "application/pdf", Server(header, failing, saved_layout("l_classic")))
    assert res.layout["status"] == "unverified" and codes(res.issues) == ["layout_unverified"]
    # the rows found by code are kept, the document is flagged
    assert res.method == "layout" and exact(res.lines, truth["lines"])
    # header, layout, two failed attempts at the witness page
    assert res.status == "needs_review" and res.calls == 4


def test_witness_can_be_switched_off() -> None:
    data, truth, header, pages = truth_doc("l_classic")
    server = Server(header, answers(pages), saved_layout("l_classic"))
    res = run(data, "application/pdf", server, Config(layout_witness=False))
    assert res.status == "ok" and exact(res.lines, truth["lines"]) and res.method == "layout"
    assert res.layout["status"] == "unchecked" and res.layout["witness_page"] is None
    assert res.calls == 2 and dict(server.calls) == {"header": 1, "layout": 1}


def test_bug_in_the_layout_parser_falls_back_to_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(pages: list[list[lay.TextLine]], layout: lay.Layout) -> lay.Parsed:
        raise KeyError("unexpected")

    monkeypatch.setattr(lay, "parse_document", broken)
    data, truth, header, pages = truth_doc("s_classic")
    res = run(data, "application/pdf", Server(header, answers(pages), saved_layout("s_classic")))
    assert res.layout["status"] == "failed" and res.layout["reason"] == "KeyError: 'unexpected'"
    assert res.status == "ok" and res.method == "model" and exact(res.lines, truth["lines"])


def test_compare_witness_verdicts() -> None:
    rows = [line(n, qty=n + 1, price=100.0) for n in range(1, 11)]
    cross = [r | {"quantity": r["unit_price"], "unit_price": r["quantity"]} for r in rows]
    assert pl.compare_witness(rows, rows)["status"] == "confirmed"
    assert pl.compare_witness(cross, rows) == {
        "status": "swapped",
        "compared": 10,
        "same": 0,
        "crossed": 10,
    }
    # the model misread four rows of ten and dropped none: still the same layout
    noisy = [r | {"quantity": 999} if i < 4 else r for i, r in enumerate(rows)]
    assert pl.compare_witness(rows, noisy)["status"] == "confirmed"
    # exactly half is not a majority
    half = [r | {"quantity": 999} if i < 5 else r for i, r in enumerate(rows)]
    assert pl.compare_witness(rows, half)["status"] == "mismatch"
    # the model missed three rows: the rest are paired by sku, not by position
    assert pl.compare_witness(rows, rows[:2] + rows[5:])["status"] == "confirmed"
    # a row whose quantity equals its price says nothing about a swap, but it does agree
    square = [line(1, qty=100, price=100.0)]
    assert pl.compare_witness(square, square)["status"] == "confirmed"
    assert pl.compare_witness(rows, [])["status"] == "unverified"
    assert pl.compare_witness([], rows)["status"] == "mismatch"


# --------------------------------------------------------------------------- #
# Scans never take the layout road
# --------------------------------------------------------------------------- #


def test_scan_does_not_ask_for_a_layout() -> None:
    p1, p2 = [line(1), line(2)], [line(3)]
    server = Server(header_for(p1 + p2), {1: [p1], 2: [p2]})
    res = run(scan(2), "application/pdf", server)
    assert res.status == "ok" and server.calls["layout"] == 0 and res.calls == 3
    assert res.layout["status"] == "off" and res.layout["reason"] == "no text layer"
    assert res.method == "model" and res.layout_attempts == [] and res.notes == []


def test_scan_with_an_ocr_text_layer_is_treated_as_a_scan() -> None:
    rows = [line(1), line(2), line(3)]
    data = scan_with_ocr_text(rows)
    assert ev.has_text_layer(ev.pdf_text_layer(data))  # the text layer is there ...
    assert lay.scanned_pages(data) == [1]
    server = Server(header_for(rows), {1: [rows]})
    res = run(data, "application/pdf", server)
    # ... but it is not trusted: no layout road, no hard checks against it
    assert server.calls["layout"] == 0
    assert res.evidence_source is None and res.layout["status"] == "off"
    assert "scan" in res.layout["reason"]
    assert res.status == "ok" and res.method == "model" and len(res.lines) == 3


@pytest.mark.parametrize("name", WITH_LAYOUT)
def test_digital_pdfs_are_not_mistaken_for_scans(name: str) -> None:
    data, *_ = truth_doc(name)
    assert lay.scanned_pages(data) == []


def rotated(data: bytes, degrees: int) -> bytes:
    """The PDF with /Rotate set on every page: the page is shown turned, nothing redrawn."""
    reader, writer = PdfReader(io.BytesIO(data)), PdfWriter()
    for page in reader.pages:
        writer.add_page(page.rotate(degrees))
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.mark.parametrize("degrees", [0, 90, 180, 270])
def test_scan_shown_turned_is_still_a_scan(degrees: int) -> None:
    # A scanner, or the person who fixed a sideways scan, sets /Rotate instead of turning
    # the pixels; pdfium then reports the page turned, the image placed as stored.
    data = rotated(scan_with_invisible_ocr(sheet()), degrees)
    assert ev.has_text_layer(ev.pdf_text_layer(data))
    assert lay.scanned_pages(data) == [1]
    rows = [line(1), line(2)]
    server = Server(header_for(rows), {1: [rows]})
    res = run(data, "application/pdf", server)
    assert server.calls["layout"] == 0 and res.evidence_source is None
    assert "treated as a scan" in res.layout["reason"]
    assert res.status == "ok" and res.quality["pages"][0]["verdict"] == "ok"


def test_digital_page_shown_turned_is_not_a_scan() -> None:
    data, *_ = truth_doc("s_classic")
    assert lay.scanned_pages(rotated(data, 90)) == []
