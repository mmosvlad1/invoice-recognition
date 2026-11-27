"""The sample invoices and recorded answers, read through the page-by-page flow."""

import json

import pytest

from invoice_recognition.recognition.evidence import has_text_layer, pdf_text_layer
from invoice_recognition.recognition.pdf import page_count

from fakes import Server, run
from samples import (
    INVOICES,
    TEXT_LAYER,
    WITH_LAYOUT,
    exact,
    saved_layout,
    saved_scan_pages,
    truth,
    truth_doc,
)


@pytest.mark.parametrize("name", TEXT_LAYER)
def test_sample_read_right_by_the_model_passes_every_check(name: str) -> None:
    data, t, header, pages = truth_doc(name)
    assert page_count(data) == t["meta"]["pages"] == len(pages)
    server = Server(header, {p: [rows] for p, rows in pages.items()})
    res = run(data, "application/pdf", server)
    assert res.evidence_source == "text_layer"
    assert [i.to_json() for i in res.issues] == []
    assert res.status == "ok" and exact(res.lines, t["lines"])
    assert [item["description"] for item in res.lines] == [
        item["description"] for item in t["lines"]
    ]
    assert all(len(p.attempts) == 1 for p in res.pages) and res.calls == len(pages) + 1


def test_scan_sample_has_no_text_layer_and_is_read_without_evidence() -> None:
    data = (INVOICES / "s_classic.scan.pdf").read_bytes()
    assert not has_text_layer(pdf_text_layer(data))
    _, t, header, pages = truth_doc("s_classic")
    res = run(data, "application/pdf", Server(header, {p: [rows] for p, rows in pages.items()}))
    assert res.evidence_source is None and res.status == "ok" and exact(res.lines, t["lines"])


def test_split_sample_names_the_rows_cut_by_a_page_break() -> None:
    t = truth("m_split")
    assert [(s["page"], s["split_to_page"]) for s in t["split_lines"]] == [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    ]
    for s in t["split_lines"]:
        assert t["lines"][s["index"]]["line_no"] == s["line_no"]


@pytest.mark.parametrize("name", WITH_LAYOUT)
def test_recorded_layout_answer_names_the_columns(name: str) -> None:
    layout = saved_layout(name)
    fields = [c["field"] for c in layout["columns"]]
    assert {"sku", "quantity", "unit_price", "amount"} <= set(fields)
    assert layout["numbers"]["decimal"] in (".", ",")


def test_recorded_scan_answers_are_one_per_page_with_the_field() -> None:
    saved = saved_scan_pages()
    assert [p["page"] for p in saved["pages"]] == [1, 2, 3, 4, 5]
    for p in saved["pages"]:
        answer = json.loads(p["text"])
        assert isinstance(answer["carry_in"], str) and answer["rows"]
        assert len(p["prompt_sha256"]) == 64


def test_exact_compares_lines_as_a_multiset() -> None:
    t = [
        {"sku": "A-1", "quantity": "2", "unit_price": "1.50", "tax_rate": "20", "amount": "3.00"},
        {"sku": "A-1", "quantity": "2", "unit_price": "1.50", "tax_rate": "20", "amount": "3.00"},
        {"sku": "B 2", "quantity": "1", "unit_price": "5", "tax_rate": "7", "amount": "5"},
    ]
    read = [
        {"sku": "b  2", "quantity": 1, "unit_price": 5.0, "tax_rate": "7%", "amount": 5},
        {"sku": "A-1", "quantity": 2, "unit_price": 1.5, "tax_rate": 20, "amount": 3.0},
        {"sku": "A-1", "quantity": 2, "unit_price": 1.5, "tax_rate": 20, "amount": 3.0},
    ]
    assert exact(read, t)
    assert not exact(read[:2], t)  # a line missed
    assert not exact([*read, read[0]], t)  # a line invented
    assert not exact([read[0] | {"amount": 6}, *read[1:]], t)  # a field wrong
