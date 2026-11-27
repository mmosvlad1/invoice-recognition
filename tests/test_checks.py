"""Deterministic checks of an extraction."""

from decimal import Decimal

import pytest

from invoice_recognition.llm import parse_json
from invoice_recognition.recognition.checks import (
    Issue,
    check_checkpoints,
    check_coverage,
    check_doc_sums,
    check_grounding,
    check_header_grounding,
    check_line_math,
    check_numbering,
    check_tax,
    dec,
)
from invoice_recognition.recognition.evidence import build
from invoice_recognition.shapes import Checkpoint, Line

from fakes import header_for, line


def messages(issues: list[Issue]) -> list[str]:
    return [i.message for i in issues]


def cp(amount: float, after_line: int, label: str | None = None) -> Checkpoint:
    return {"amount": amount, "after_line": after_line, "label": label, "page": 1}


# --------------------------------------------------------------------------- #
# Numbers from the model and the JSON of an issue
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12, Decimal(12)),
        (1.5, Decimal("1.5")),
        ("1234.50", Decimal("1234.50")),
        (" 7 ", Decimal(7)),
        ("20%", Decimal(20)),
        (None, None),
        (True, None),  # a flag is not a number
        ("", None),
        ("12,50", None),
        ("n/a", None),
        ([1], None),
    ],
)
def test_dec(value: object, expected: Decimal | None) -> None:
    assert dec(value) == expected


def test_issue_json_leaves_out_an_unknown_line_number() -> None:
    assert Issue("tax", "m").to_json() == {
        "check": "tax",
        "message": "m",
        "line": None,
        "page": None,
        "severity": "error",
    }
    assert Issue("line_math", "m", line=0, page=2, line_no=7).to_json()["line_no"] == 7


# --------------------------------------------------------------------------- #
# 1. Document sums
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("supplier", [123, ["ACME"], {"name": "ACME"}, None, "  "])
def test_doc_sums_survives_a_supplier_that_is_not_a_string(supplier: object) -> None:
    lines = [line(1)]
    issues = check_doc_sums(
        {"header": header_for(lines) | {"supplier_name": supplier}, "lines": lines}
    )
    assert [i.message for i in issues] == ["supplier is missing"]


def test_doc_sums_of_a_consistent_document() -> None:
    lines = [line(1), line(2, qty=3, price=1.5)]
    assert check_doc_sums({"header": header_for(lines), "lines": lines}) == []


def test_doc_sums_lines_against_subtotal_and_subtotal_plus_tax_against_total() -> None:
    lines = [line(1), line(2)]
    header = header_for(lines) | {"subtotal": 45.0, "total": 50.0}
    assert messages(check_doc_sums({"header": header, "lines": lines})) == [
        "lines add up to 40.0, invoice subtotal is 45.0 (difference -5.0)",
        "subtotal 45.0 + tax 8.0 != total 50.0",
    ]


def test_doc_sums_tolerates_one_cent() -> None:
    lines = [line(1)]
    header = header_for(lines) | {"subtotal": 20.01, "total": 24.02}
    assert check_doc_sums({"header": header, "lines": lines}) == []


def test_doc_sums_without_lines_or_totals() -> None:
    assert messages(check_doc_sums({"header": {"supplier_name": "ACME"}, "lines": []})) == [
        "no line items"
    ]
    lines = [line(1)]
    header = header_for(lines) | {"tax_total": None}
    assert messages(check_doc_sums({"header": header, "lines": lines})) == [
        "totals are missing in the header"
    ]
    assert messages(check_doc_sums({})) == ["supplier is missing", "no line items"]


def test_doc_sums_line_without_amount() -> None:
    lines = [line(1), line(2) | {"amount": None}, "not a line"]
    header = header_for([line(1)])
    assert messages(check_doc_sums({"header": header, "lines": lines})) == [
        "some lines have no amount"
    ]


# --------------------------------------------------------------------------- #
# 2. Line arithmetic
# --------------------------------------------------------------------------- #


def test_line_math() -> None:
    lines = [
        line(1),
        line(2, amount=25.0) | {"page": 3},
        line(3, price=1000.0, amount=4000.0) | {"quantity": 0.004},  # 4.000 read as 4.0 thousandths
        line(4) | {"unit_price": None},
        line(5, qty=3, price=3.33, amount=10.0),  # 9.99 vs 10.00: within a cent
    ]
    issues = check_line_math({"lines": lines})
    assert [(i.line, i.page) for i in issues] == [(1, 3), (2, None), (3, None)]
    assert messages(issues) == [
        "2 x 10.0 = 20.00, but amount is 25.0",
        (
            "0.004 x 1000.0 = 4.00, but amount is 4000.0"
            " (quantity looks 1000x too small: thousands separator read as a decimal point)"
        ),
        "quantity, unit price or amount is missing",
    ]


# --------------------------------------------------------------------------- #
# 3. Tax
# --------------------------------------------------------------------------- #


def test_tax_with_a_breakdown() -> None:
    lines = [line(1), line(2) | {"tax_rate": 7}, line(3) | {"tax_rate": "7%"}]
    ok = {
        "tax_breakdown": [{"rate": 20, "base": 20, "tax": 4}, {"rate": 7, "base": 40, "tax": 2.8}]
    }
    assert check_tax({"header": ok, "lines": lines}) == []
    wrong = {"tax_breakdown": [{"rate": 20, "base": 25, "tax": 5}, "junk"]}
    assert messages(check_tax({"header": wrong, "lines": lines})) == [
        "lines carry 7% tax on 40.0, but the invoice has no 7% row",
        "lines at 20% add up to 20.0, invoice says 25",
    ]


def test_tax_breakdown_row_without_lines_or_with_an_unreadable_rate() -> None:
    lines = [line(1)]
    rows = [
        {"rate": 20, "base": 20, "tax": 4},
        {"rate": 0, "base": 5, "tax": 0},
        {"rate": "exempt", "base": 3, "tax": 0},
    ]
    # Rates are reported in ascending order; an unreadable rate (None) sorts like 0,
    # so between the two the order of the printed rows decides.
    assert messages(check_tax({"header": {"tax_breakdown": rows}, "lines": lines})) == [
        "lines at 0% add up to 0, invoice says 5",
        "lines at None% add up to 0, invoice says 3",
    ]
    assert messages(check_tax({"header": {"tax_breakdown": rows[::-1]}, "lines": lines})) == [
        "lines at None% add up to 0, invoice says 3",
        "lines at 0% add up to 0, invoice says 5",
    ]


def test_tax_without_a_breakdown_is_computed_from_the_lines() -> None:
    lines = [line(1), line(2, qty=1, price=0.1)]  # 20.00 + 0.10 at 20 %: 4.00 + 0.02
    assert check_tax({"header": {"tax_total": 4.05}, "lines": lines}) == []  # within 0.05
    assert messages(check_tax({"header": {"tax_total": 4.1}, "lines": lines})) == [
        "tax computed from lines is 4.02, invoice says 4.1"
    ]
    assert check_tax({"header": {}, "lines": lines}) == []  # nothing to compare with


# --------------------------------------------------------------------------- #
# 4. Numbering
# --------------------------------------------------------------------------- #


def test_numbering_gap_points_to_the_line_after_it() -> None:
    lines = [line(1), line(2), line(5) | {"page": 2}, line(6)]
    (issue,) = check_numbering({"lines": lines})
    assert issue.message == "2 position number(s) missing, e.g. [3, 4]"
    assert (issue.line, issue.page) == (2, 2)


def test_numbering_repeated_numbers() -> None:
    lines = [line(1), line(2), line(2), line(3)]
    assert messages(check_numbering({"lines": lines})) == ["position numbers repeated: [2]"]


def test_numbering_is_skipped_when_the_document_does_not_number_its_lines() -> None:
    unnumbered = [line(n) | {"line_no": None} for n in (1, 2, 3, 4, 5)]
    unnumbered[0]["line_no"] = 1
    assert check_numbering({"lines": unnumbered}) == []
    assert check_numbering({"lines": [line(7)]}) == []
    flags = [line(1) | {"line_no": True}, line(2) | {"line_no": False}]  # not numbers
    assert check_numbering({"lines": flags}) == []
    consecutive = [line(n) for n in (4, 5, 6)]  # need not start at 1
    assert check_numbering({"lines": consecutive}) == []


# --------------------------------------------------------------------------- #
# 5. Control totals
# --------------------------------------------------------------------------- #

FOUR = [line(1), line(2), line(3), line(4)]  # 20.0 each


def test_checkpoint_as_section_subtotal_or_running_total() -> None:
    subtotals = [cp(40, 2, "Section A"), cp(40, 4, "Section B")]
    assert check_checkpoints({"lines": FOUR, "checkpoints": subtotals}) == []
    running = [cp(40, 2, "Carried forward"), cp(80, 4, "Total")]
    assert check_checkpoints({"lines": FOUR, "checkpoints": running}) == []


def test_checkpoint_failure_is_pinned_to_its_segment() -> None:
    checkpoints = [cp(40, 2), cp(45, 4, "Section B")]
    (issue,) = check_checkpoints({"lines": FOUR, "checkpoints": checkpoints})
    assert issue.message == 'lines 3-4 add up to 40.0, but "Section B" says 45'
    assert (issue.check, issue.line, issue.page) == ("checkpoints", 2, 1)
    (unlabelled,) = check_checkpoints({"lines": FOUR, "checkpoints": [cp(41, 2)]})
    assert unlabelled.message == 'lines 1-2 add up to 40.0, but "control total" says 41'


def test_carried_forward_total_repeated_at_the_top_of_the_next_page() -> None:
    checkpoints = [cp(40, 2, "Carried forward"), cp(40, 2, "Brought forward"), cp(80, 4)]
    assert check_checkpoints({"lines": FOUR, "checkpoints": checkpoints}) == []


def test_partial_document_cannot_verify_its_first_checkpoint() -> None:
    # Page 2 of a document: the brought-forward total covers lines on page 1.
    checkpoints = [cp(500, 1, "Brought forward"), cp(560, 4, "Carried forward")]
    assert check_checkpoints({"lines": FOUR, "checkpoints": checkpoints}, partial=True) == []
    # Read as a whole document, the first one is checked (and fails); the second
    # still passes as a running total on top of it.
    issues = check_checkpoints({"lines": FOUR, "checkpoints": checkpoints})
    assert messages(issues) == ['lines 1-1 add up to 20.0, but "Brought forward" says 500']


def test_checkpoints_that_cannot_be_placed_are_ignored() -> None:
    checkpoints: list[Checkpoint] = [
        cp(999, 0, "Brought forward"),  # before any line: only a base
        cp(1, 9),  # beyond the last line
        {"amount": None, "after_line": 2},
        {"amount": 5, "after_line": 2.0},
    ]
    assert check_checkpoints({"lines": FOUR, "checkpoints": checkpoints}) == []


# --------------------------------------------------------------------------- #
# 6-7. Presence on the page
# --------------------------------------------------------------------------- #

PAGE_1 = "SKU-001 Item 1 2 10.00 20.00\nSKU-002 Item 2 2 10.00 20.00"
PAGE_2 = "SKU-003 Item 3 2 10.00 20.00\nSubtotal 60.00 Total 72.00 INV-1"


def on_page(lines: list[Line], page: int) -> list[Line]:
    return [item | {"page": page} for item in lines]


def test_grounding_of_lines_printed_as_rows() -> None:
    ev = build([PAGE_1, PAGE_2], "text_layer")
    lines = on_page([line(1), line(2)], 1) + on_page([line(3)], 2)
    assert check_grounding({"lines": lines}, ev) == []


def test_grounding_value_not_printed() -> None:
    ev = build([PAGE_1], "text_layer")
    lines = [line(1), line(2, amount=21.0), line(9)]
    issues = check_grounding({"lines": lines}, ev)
    assert [(i.check, i.line, i.message) for i in issues] == [
        ("grounding_value", 1, "not printed on the page: amount"),
        ("grounding_value", 2, "not printed on the page: sku"),
    ]


def test_grounding_row_values_mixed_between_rows() -> None:
    ev = build(["SKU-001 Item 1 2 10.00 20.00\nSKU-002 Item 2 3 5.00 15.00"], "text_layer")
    mixed = line(1, qty=3, price=5.0, amount=15.0)  # SKU-001 with the numbers of SKU-002
    (issue,) = check_grounding({"lines": [mixed]}, ev)
    assert (issue.check, issue.message) == (
        "grounding_row",
        "values exist on the page but not together in one row",
    )


def test_grounding_uses_each_printed_row_once() -> None:
    ev = build(["SKU-001 Item 1 2 10.00 20.00"], "text_layer")
    issues = check_grounding({"lines": [line(1), line(1)]}, ev)
    assert [(i.check, i.line) for i in issues] == [("grounding_row", 1)]


def test_grounding_looks_on_the_page_of_the_line() -> None:
    ev = build([PAGE_1, PAGE_2], "text_layer")
    (issue,) = check_grounding({"lines": on_page([line(3)], 1)}, ev)
    assert (issue.check, issue.page) == ("grounding_value", 1)
    # an unknown page number falls back to the whole document
    assert check_grounding({"lines": on_page([line(3)], 7)}, ev) == []


def test_grounding_in_the_other_number_format() -> None:
    ev = build(["SKU-001 Item 1 1.500 1,25 1.875,00"], "text_layer")
    assert ev.locale == "eu"
    assert check_grounding({"lines": [line(1, qty=1500, price=1.25)]}, ev) == []


def test_header_grounding() -> None:
    ev = build([PAGE_1, PAGE_2], "text_layer")
    printed = {"subtotal": 60, "total": 72.0, "invoice_number": "inv-1", "po_number": "PO 77"}
    assert check_header_grounding({"header": printed}, ev) == []
    invented = {"subtotal": 60.5, "total": 72, "supplier_tax_id": "DE123", "tax_total": 12.34}
    assert messages(check_header_grounding({"header": invented}, ev)) == [
        "subtotal 60.5 is not printed anywhere in the document",
        'supplier_tax_id "DE123" is not printed anywhere in the document',
    ]


def test_coverage_per_page_and_for_the_document() -> None:
    ev = build([PAGE_1, PAGE_2], "text_layer")
    lines = on_page([line(1), line(2)], 1) + on_page([line(3)], 2)
    assert check_coverage({"lines": lines}, ev) == []
    (issue,) = check_coverage({"lines": lines[:1] + lines[2:]}, ev)
    assert (issue.message, issue.page) == ("page 1: 2 item rows printed, 1 extracted", 1)
    # without page numbers the whole document is compared
    assert check_coverage({"lines": [line(1), line(2), line(3)]}, ev) == []
    assert messages(check_coverage({"lines": [line(1)]}, ev)) == [
        "3 item rows printed, 1 extracted"
    ]


# --------------------------------------------------------------------------- #
# Numbers that are not finite
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), "NaN", "Infinity", "-Infinity", "sNaN"]
)
def test_dec_treats_a_non_finite_number_as_missing(value: object) -> None:
    assert dec(value) is None


def test_non_finite_numbers_from_the_model_do_not_crash_the_checks() -> None:
    # Python's JSON parser accepts NaN and Infinity, so a model can send them.
    doc = parse_json(
        '{"header": {"supplier_name": "ACME", "subtotal": NaN, "tax_total": 4, "total": 24,'
        ' "invoice_number": "INV-1", "tax_breakdown": [{"rate": 20, "base": Infinity, "tax": 4}]},'
        ' "lines": [{"line_no": 1, "sku": "SKU-001", "quantity": NaN, "unit_price": 10,'
        ' "amount": 20, "tax_rate": Infinity, "page": 1}],'
        ' "checkpoints": [{"amount": -Infinity, "after_line": 1, "page": 1}]}'
    )
    assert doc is not None
    ev = build(["SKU-001 Item 1 2 10.00 20.00\nTotal 24.00 INV-1"], "text_layer")
    assert messages(check_doc_sums(doc)) == ["totals are missing in the header"]
    assert messages(check_line_math(doc)) == ["quantity, unit price or amount is missing"]
    assert messages(check_tax(doc)) == []  # neither the line's rate nor the printed base is known
    assert check_checkpoints(doc) == []
    assert check_numbering(doc) == []
    assert check_grounding(doc, ev) == []  # the sku and the known numbers are on one row
    assert check_header_grounding(doc, ev) == []
    assert check_coverage(doc, ev) == []
