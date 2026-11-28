"""Deterministic checks of an extraction."""

import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal

import pytest

from invoice_recognition.llm import parse_json
from invoice_recognition.recognition.checks import (
    TEST_BUDGET,
    Issue,
    agree_with_subtotal,
    check_checkpoints,
    check_coverage,
    check_doc_sums,
    check_grounding,
    check_header_grounding,
    check_line_math,
    check_numbering,
    check_tax,
    dec,
    explained_by_lines,
    ranges,
)
from invoice_recognition.recognition.evidence import CENT, build, money
from invoice_recognition.recognition.issue_values import SubtotalOff
from invoice_recognition.shapes import Checkpoint, JSONObject, Line

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


def test_issue_json_leaves_out_an_unknown_line_number_and_what_is_empty() -> None:
    assert Issue("tax", "m").to_json() == {
        "check": "tax",
        "message": "m",
        "line": None,
        "page": None,
        "severity": "error",
        "fields": [],  # always there: the page gets one shape
    }
    assert Issue("line_math", "m", line=0, page=2, line_no=7).to_json()["line_no"] == 7


def test_issue_json_with_fields_suggestions_and_what_it_follows_from() -> None:
    suggestion = {"set": {"line.amount": 74.16}, "source": "arithmetic"}
    issue = Issue(
        "doc_sums",
        "m",
        fields=["header.subtotal", "line.amount"],
        suggestions=[suggestion],
        follows_from=[0, 3],
    )
    assert issue.to_json() == {
        "check": "doc_sums",
        "message": "m",
        "line": None,
        "page": None,
        "severity": "error",
        "fields": ["header.subtotal", "line.amount"],
        "suggestions": [{"set": {"line.amount": 74.16}, "source": "arithmetic"}],
        "follows_from": [0, 3],
    }
    # The JSON is a copy: changing it leaves the issue as it was.
    issue.to_json()["suggestions"][0]["set"]["line.amount"] = 1
    assert issue.suggestions == [suggestion]


def test_issue_json_is_the_same_with_values_or_a_form() -> None:
    plain = Issue(
        "doc_sums",
        "m",
        line=1,
        page=2,
        line_no=3,
        fields=["header.subtotal"],
        suggestions=[{"set": {"header.subtotal": 1.5}, "source": "arithmetic"}],
        follows_from=[0],
    )
    values = SubtotalOff(Decimal("1.5"), Decimal(3), Decimal("-1.5"))
    for worded in (replace(plain, values=values), replace(plain, form="totals_missing")):
        out = worded.to_json()
        # The same keys in the same order and the same values: the same bytes.
        assert json.dumps(out) == json.dumps(plain.to_json())
        out["suggestions"][0]["set"]["header.subtotal"] = 9
        out["follows_from"].append(5)
        assert worded.suggestions == plain.suggestions and worded.follows_from == [0]


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
        "quantity, unit price or amount is missing or is not a usable number",
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
    assert messages(check_line_math(doc)) == [
        "quantity, unit price or amount is missing or is not a usable number"
    ]
    assert messages(check_tax(doc)) == []  # neither the line's rate nor the printed base is known
    assert check_checkpoints(doc) == []
    assert check_numbering(doc) == []
    assert check_grounding(doc, ev) == []  # the sku and the known numbers are on one row
    assert check_header_grounding(doc, ev) == []
    assert check_coverage(doc, ev) == []


# --------------------------------------------------------------------------- #
# Numbers too long for cents
# --------------------------------------------------------------------------- #

HUGE = "1" + "0" * 28  # 1E+28: 29 digits before the point


def test_line_math_and_tax_with_a_number_too_long_for_cents() -> None:
    big = line(1) | {"quantity": 1, "unit_price": HUGE, "amount": HUGE}
    assert check_line_math({"lines": [big]}) == []
    wrong = big | {"amount": "2" + "0" * 28}
    (issue,) = check_line_math({"lines": [wrong]})
    assert issue.message == (
        "1 x 10000000000000000000000000000 = 1.000000000000000000000000000E+28,"
        " but amount is 20000000000000000000000000000"
    )
    assert messages(check_tax({"header": {"tax_total": 1}, "lines": [big]})) == [
        "tax computed from lines is 2000000000000000000000000000, invoice says 1"
    ]


@pytest.mark.parametrize("breakdown", [5, True, "20%: 4.00", {"rate": 20, "base": 20, "tax": 4}])
def test_tax_breakdown_that_is_not_a_list_is_no_breakdown(breakdown: object) -> None:
    lines = [line(1)]  # 20.00 at 20 %
    header = {"tax_breakdown": breakdown, "tax_total": 4.0}
    assert check_tax({"header": header, "lines": lines}) == []
    assert messages(check_tax({"header": header | {"tax_total": 9}, "lines": lines})) == [
        "tax computed from lines is 4.00, invoice says 9"
    ]


# --------------------------------------------------------------------------- #
# Numbers of absurd size
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1e100", Decimal("1e100")),
        ("9.99e100", Decimal("9.99e100")),
        ("1e101", None),
        ("1e999999", None),
        ("-1e999999999", None),
        ("1e-100", Decimal("1e-100")),
        ("1e-101", None),
        ("0E-999999999", None),
        ("0." + "0" * 99 + "1", Decimal("1e-100")),  # 100 digits after the point
        ("0." + "0" * 100 + "1", None),  # 101
        (12.5, Decimal("12.5")),
    ],
)
def test_dec_treats_a_number_of_absurd_size_as_missing(
    value: object, expected: Decimal | None
) -> None:
    assert dec(value) == expected


def test_numbers_of_absurd_size_from_the_model_do_not_crash_the_checks() -> None:
    # The product of two such numbers overflows the decimal context, and printing one in
    # full for the presence checks would take a billion digits.
    doc = parse_json(
        '{"header": {"supplier_name": "ACME", "subtotal": "9e999999", "tax_total": "9e999999",'
        ' "total": "1e999999999", "tax_breakdown": [{"rate": 20, "base": "1e999999", "tax": 4}]},'
        ' "lines": [{"line_no": 1, "sku": "SKU-001", "quantity": "1e999999", "unit_price": 10,'
        ' "amount": "1e-999999999", "tax_rate": "1e999999", "page": 1}],'
        ' "checkpoints": [{"amount": "1e999999", "after_line": 1, "page": 1}]}'
    )
    assert doc is not None
    ev = build(["SKU-001 Item 1 2 10.00 20.00\nTotal 24.00 INV-1"], "text_layer")
    assert messages(check_doc_sums(doc)) == ["totals are missing in the header"]
    assert messages(check_line_math(doc)) == [
        "quantity, unit price or amount is missing or is not a usable number"
    ]
    assert check_tax(doc) == []
    assert check_checkpoints(doc) == []
    assert check_header_grounding(doc, ev) == []
    assert check_coverage(doc, ev) == []
    assert check_grounding(doc, ev) == []  # the sku is on a row; no other value is known


# --------------------------------------------------------------------------- #
# The fields each issue is about
# --------------------------------------------------------------------------- #

FIELDS_EV = build([PAGE_1, PAGE_2], "text_layer")
TWO = [line(1), line(2)]
HEADER = header_for(TWO)  # subtotal 40, tax 8, total 48


@pytest.mark.parametrize(
    ("issues", "message", "fields"),
    [
        pytest.param(
            lambda: check_doc_sums({"header": HEADER | {"supplier_name": None}, "lines": TWO}),
            "supplier is missing",
            ["header.supplier_name"],
            id="doc_sums-supplier",
        ),
        pytest.param(
            lambda: check_doc_sums({"header": HEADER, "lines": []}),
            "no line items",
            [],
            id="doc_sums-no-lines",
        ),
        pytest.param(
            lambda: check_doc_sums(
                {"header": HEADER | {"subtotal": None, "total": "n/a"}, "lines": TWO}
            ),
            "totals are missing in the header",
            ["header.subtotal", "header.total"],  # only the missing ones
            id="doc_sums-totals-missing",
        ),
        pytest.param(
            lambda: check_doc_sums(
                {"header": header_for([line(1)]), "lines": [line(1), line(2) | {"amount": None}]}
            ),
            "some lines have no amount",
            ["line.amount"],  # the column: one issue for every such line
            id="doc_sums-amount-missing",
        ),
        pytest.param(
            lambda: check_doc_sums(
                {"header": HEADER | {"subtotal": 45.0, "total": 53.0}, "lines": TWO}
            ),
            "lines add up to 40.0, invoice subtotal is 45.0 (difference -5.0)",
            ["header.subtotal", "line.amount"],
            id="doc_sums-lines-against-subtotal",
        ),
        pytest.param(
            lambda: check_doc_sums({"header": HEADER | {"total": 50.0}, "lines": TWO}),
            "subtotal 40.0 + tax 8.0 != total 50.0",
            ["header.subtotal", "header.tax_total", "header.total"],
            id="doc_sums-header-arithmetic",
        ),
        pytest.param(
            lambda: check_line_math({"lines": [line(1) | {"unit_price": None, "amount": "?"}]}),
            "quantity, unit price or amount is missing or is not a usable number",
            ["line.unit_price", "line.amount"],  # only the missing ones
            id="line_math-missing",
        ),
        pytest.param(
            lambda: check_line_math({"lines": [line(1, amount=25.0)]}),
            "2 x 10.0 = 20.00, but amount is 25.0",
            ["line.quantity", "line.unit_price", "line.amount"],
            id="line_math-mismatch",
        ),
        pytest.param(
            lambda: check_line_math(
                {"lines": [line(1, price=1000.0, amount=4000.0) | {"quantity": 0.004}]}
            ),
            "0.004 x 1000.0 = 4.00, but amount is 4000.0"
            " (quantity looks 1000x too small: thousands separator read as a decimal point)",
            ["line.quantity"],
            id="line_math-x1000",
        ),
        pytest.param(
            lambda: check_tax(
                {
                    "header": {"tax_breakdown": [{"rate": 20, "base": 20, "tax": 4}]},
                    "lines": [line(1), line(2) | {"tax_rate": 7}],
                }
            ),
            "lines carry 7% tax on 20.0, but the invoice has no 7% row",
            ["line.tax_rate", "header.tax_breakdown"],
            id="tax-rate-not-printed",
        ),
        pytest.param(
            lambda: check_tax(
                {"header": {"tax_breakdown": [{"rate": 20, "base": 25, "tax": 5}]}, "lines": TWO}
            ),
            "lines at 20% add up to 40.0, invoice says 25",
            ["line.tax_rate", "line.amount", "header.tax_breakdown"],
            id="tax-base-of-a-rate",
        ),
        pytest.param(
            lambda: check_tax({"header": {"tax_total": 9.0}, "lines": TWO}),
            "tax computed from lines is 8.00, invoice says 9.0",
            ["header.tax_total", "line.tax_rate"],
            id="tax-without-breakdown",
        ),
        pytest.param(
            lambda: check_numbering({"lines": [line(1), line(2), line(2), line(3)]}),
            "position numbers repeated: [2]",
            ["line.line_no"],
            id="numbering-repeated",
        ),
        pytest.param(
            lambda: check_numbering({"lines": [line(1), line(2), line(5)]}),
            "2 position number(s) missing, e.g. [3, 4]",
            [],  # a whole row is missing, not a cell
            id="numbering-missing",
        ),
        pytest.param(
            lambda: check_checkpoints({"lines": FOUR, "checkpoints": [cp(40, 2), cp(45, 4)]}),
            'lines 3-4 add up to 40.0, but "control total" says 45',
            [],
            id="checkpoints",
        ),
        pytest.param(
            lambda: check_grounding(
                {"lines": on_page([line(1, qty=3, amount=30.0)], 1)}, FIELDS_EV
            ),
            "not printed on the page: quantity, amount",
            ["line.quantity", "line.amount"],  # exactly the values not found
            id="grounding_value",
        ),
        pytest.param(
            lambda: check_grounding(
                {"lines": on_page([line(1, price=7.5, amount=20.0)], 1)}, FIELDS_EV
            ),
            "not printed on the page: unit_price",  # 2 and 20.00 are printed, 7.50 is not
            ["line.unit_price"],
            id="grounding_value-unit-price",
        ),
        pytest.param(
            lambda: check_grounding({"lines": on_page([line(1), line(1)], 1)}, FIELDS_EV),
            "values exist on the page but not together in one row",
            ["line.sku", "line.quantity", "line.unit_price", "line.amount"],
            id="grounding_row",
        ),
        pytest.param(
            lambda: check_grounding(
                {"lines": on_page([line(1) | {"sku": None, "unit_price": "?"}] * 3, 1)},
                FIELDS_EV,
            )[-1:],
            "values exist on the page but not together in one row",
            ["line.quantity", "line.amount"],  # no sku and no price were looked for
            id="grounding_row-without-sku",
        ),
        pytest.param(
            lambda: check_header_grounding({"header": {"total": 99.5}}, FIELDS_EV),
            "total 99.5 is not printed anywhere in the document",
            ["header.total"],
            id="grounding_header-number",
        ),
        pytest.param(
            lambda: check_header_grounding({"header": {"po_number": "PO-9"}}, FIELDS_EV),
            'po_number "PO-9" is not printed anywhere in the document',
            ["header.po_number"],
            id="grounding_header-identifier",
        ),
        pytest.param(
            lambda: check_coverage(
                {"lines": on_page([line(1)], 1) + on_page([line(3)], 2)}, FIELDS_EV
            ),
            "page 1: 2 item rows printed, 1 extracted",
            [],
            id="coverage-page",
        ),
        pytest.param(
            lambda: check_coverage({"lines": [line(1)]}, FIELDS_EV),
            "3 item rows printed, 1 extracted",
            [],
            id="coverage-document",
        ),
    ],
)
def test_every_issue_names_the_fields_it_is_about(
    issues: Callable[[], list[Issue]], message: str, fields: list[str]
) -> None:
    (issue,) = issues()
    assert issue.message == message
    assert issue.fields == fields
    assert issue.to_json()["fields"] == fields


# --------------------------------------------------------------------------- #
# The values that would settle a line's arithmetic
# --------------------------------------------------------------------------- #


def suggestions_for(item: Line) -> list[JSONObject]:
    (issue,) = check_line_math({"lines": [item]})
    return issue.suggestions


def arithmetic(**values: float) -> list[JSONObject]:
    return [{"set": {f"line.{k}": v}, "source": "arithmetic"} for k, v in values.items()]


# 12 x 6.18 = 74.16, with one of the three numbers read wrong in turn.
@pytest.mark.parametrize(
    ("q", "p", "a", "expected"),
    [
        # The amount: the right one is offered, and so is the price that fits the wrong
        # amount (12 x 5.93 = 71.16); no whole quantity gives 71.16 at 6.18.
        (12, 6.18, 71.16, arithmetic(unit_price=5.93, amount=74.16)),
        # The quantity: 74.16 / 6.18 = 12; 74.16 / 13 = 5.7046... is no price in cents.
        (13, 6.18, 74.16, arithmetic(quantity=12, amount=80.34)),
        # The price: 74.16 / 12 = 6.18; 74.16 / 6.81 = 10.889... is not a whole quantity.
        (12, 6.81, 74.16, arithmetic(unit_price=6.18, amount=81.72)),
    ],
    ids=["amount", "quantity", "unit-price"],
)
def test_line_math_offers_one_value_per_number_that_could_be_wrong(
    q: float, p: float, a: float, expected: list[JSONObject]
) -> None:
    assert suggestions_for(line(1) | {"quantity": q, "unit_price": p, "amount": a}) == expected


def test_line_math_offers_only_exact_values() -> None:
    # 10 / 7 has no end; 10 / 3 = 3.33, but 3 x 3.33 = 9.99: only the amount is exact.
    assert suggestions_for(line(1, qty=3, price=7.0, amount=10.0)) == arithmetic(amount=21)
    # 1 / 16 = 0.0625 would give the amount, but no quantity has four decimals, and
    # 0.063 x 16 = 1.008 is 1.01; 2.5 x 0.40 = 1.00.
    item = line(1, price=16.0, amount=1.0) | {"quantity": 2.5}
    assert suggestions_for(item) == arithmetic(unit_price=0.4, amount=40)


def test_line_math_rounds_the_quantity_it_offers() -> None:
    # 12 x 0.0603 = 0.72, but the amount 7.24 is 120 x 0.0603 = 7.236 in cents: the
    # quotient 120.066... rounded to a whole quantity. The price fits with as many
    # decimals as it was read with: 12 x 0.603 = 7.236.
    item = line(1, qty=12, price=0.0603, amount=7.24)
    assert suggestions_for(item) == arithmetic(quantity=120, unit_price=0.603, amount=0.72)
    # 1 / 0.33 = 3.0303...: 3 and 3.0 give 0.99, 3.03 gives 0.9999, which is 1.00.
    item = line(1, price=0.33, amount=1.0) | {"quantity": 2.5}
    assert suggestions_for(item) == arithmetic(quantity=3.03, unit_price=0.4, amount=0.83)


def test_line_math_rounds_the_quantity_half_up_to_three_decimals() -> None:
    # 1.00 / 0.0667 = 14.99...: 15 x 0.0667 = 1.0005, which is 1.00; 14 would be 0.93.
    item = line(1, qty=1, price=0.0667, amount=1.0)
    assert suggestions_for(item) == arithmetic(quantity=15, unit_price=1, amount=0.07)
    # 2.50 / 1.99 = 1.2562...: only three decimals give 2.50 (1.256 x 1.99 = 2.49944).
    item = line(1, price=1.99, amount=2.5) | {"quantity": 1.5}
    assert suggestions_for(item) == arithmetic(quantity=1.256, amount=2.99)


def test_line_math_offers_a_price_with_as_many_decimals_as_it_was_read_with() -> None:
    # 2500 x 0.2186 = 546.50 read as 564.50: 564.50 / 2500 = 0.2258, four decimals as read.
    item = line(1, qty=2500, price=0.2186, amount=564.5)
    assert suggestions_for(item) == arithmetic(unit_price=0.2258, amount=546.5)
    # A price read in cents gets a price in cents: 74.19 / 12 = 6.1825 is not offered.
    assert suggestions_for(line(1, qty=12, price=6.18, amount=74.19)) == arithmetic(amount=74.16)


def test_line_math_offers_a_whole_quantity_for_one_read_whole() -> None:
    # 99 / 10 = 9.9 pieces is no quantity for a line that counts pieces; 2 x 49.50 is.
    item = line(1, amount=99.0)
    assert suggestions_for(item) == arithmetic(unit_price=49.5, amount=20)
    # A quantity read with decimals may get decimals.
    item = line(1, amount=99.0) | {"quantity": 2.5}
    assert suggestions_for(item) == arithmetic(quantity=9.9, unit_price=39.6, amount=25)


def test_line_math_never_offers_a_zero_quantity_or_price() -> None:
    # The amount 0 would be 0 x 10.0 or 2 x 0.
    assert suggestions_for(line(1, amount=0.0)) == arithmetic(amount=20)


def test_line_math_values_are_json_numbers() -> None:
    (s,) = suggestions_for(line(1, qty=3, price=7.0, amount=10.0))
    assert type(s["set"]["line.amount"]) is int  # 21, not "21.00" or 21.0
    s = suggestions_for(line(1, amount=25.0))[-1]
    assert s["set"] == {"line.amount": 20} and type(s["set"]["line.amount"]) is int
    s = suggestions_for(line(1, qty=12, price=6.18, amount=71.16))[-1]
    assert s["set"] == {"line.amount": 74.16} and type(s["set"]["line.amount"]) is float


def test_line_math_with_a_zero_does_not_divide_by_it() -> None:
    # 5 / 10 = 0.5 is no whole quantity; nothing is divided by the quantity 0.
    assert suggestions_for(line(1, qty=0, price=10.0, amount=5.0)) == arithmetic(amount=0)
    assert suggestions_for(line(1, qty=2, price=0.0, amount=5.0)) == arithmetic(
        unit_price=2.5, amount=0
    )


def test_line_math_with_the_thousands_hint_offers_the_quantity_only() -> None:
    # 4000.0 / 1000.0 = 4
    item = line(1, price=1000.0, amount=4000.0) | {"quantity": 0.004}
    assert suggestions_for(item) == arithmetic(quantity=4)
    # 1012.29 / 0.12345 = 8199.99..., 8200 x 0.12345 = 1012.29.
    item = line(1, amount=1012.29) | {"quantity": 8.2, "unit_price": 0.12345}
    assert suggestions_for(item) == arithmetic(quantity=8200)


def test_only_a_missing_amount_gets_a_suggestion() -> None:
    (issue,) = check_line_math({"lines": [line(1) | {"amount": None}]})
    assert issue.fields == ["line.amount"]
    assert issue.suggestions == arithmetic(amount=20)  # 2 x 10.0
    for missing in ({"quantity": None}, {"unit_price": "?", "amount": None}):
        (issue,) = check_line_math({"lines": [line(1) | missing]})
        assert issue.suggestions == []


# --------------------------------------------------------------------------- #
# Issues of the document's arithmetic that one wrong line amount explains
# --------------------------------------------------------------------------- #


def test_issues_that_pass_with_the_amount_of_the_line_are_explained() -> None:
    lines = [line(1), line(2, amount=99.0), line(3)]
    header = header_for([line(1), line(2), line(3)])  # subtotal 60, tax 12
    doc = {"header": header, "lines": lines, "checkpoints": [cp(40, 2, "Section A")]}
    explained = explained_by_lines(doc, {1: Decimal("20.00")})
    assert [(i.check, i.message, needed) for i, needed in explained] == [
        ("doc_sums", "lines add up to 139.0, invoice subtotal is 60.0 (difference 79.0)", [1]),
        ("tax", "tax computed from lines is 27.80, invoice says 12.0", [1]),
        ("checkpoints", 'lines 1-2 add up to 119.0, but "Section A" says 40', [1]),
    ]
    # The lines are not changed.
    assert lines[1]["amount"] == 99.0


def test_an_issue_wrong_for_another_reason_is_not_explained() -> None:
    lines = [line(1), line(2, amount=99.0)]
    header = header_for([line(1), line(2)]) | {"subtotal": 45.0}  # 40 would be right
    doc = {"header": header, "lines": lines}
    assert check_doc_sums(doc)[0].message.startswith("lines add up to 119.0")
    assert [i.check for i, _ in explained_by_lines(doc, {1: Decimal("20.00")})] == ["tax"]
    # The header's own arithmetic is never explained by a line.
    doc = {"header": header_for(lines) | {"total": 1.0}, "lines": lines}
    assert [i.message for i in check_doc_sums(doc)] == ["subtotal 119.0 + tax 23.8 != total 1.0"]
    assert explained_by_lines(doc, {1: Decimal("20.00")}) == []


def test_an_explained_issue_needs_only_the_lines_it_depends_on() -> None:
    lines = [line(1, amount=99.0), line(2, amount=55.0) | {"tax_rate": 7}, line(3)]
    breakdown = [{"rate": 20, "base": 40, "tax": 8}, {"rate": 7, "base": 20, "tax": 1.4}]
    checkpoints = [cp(20, 1, "Section A"), cp(40, 3, "Section B")]
    doc = {"header": {"tax_breakdown": breakdown}, "lines": lines, "checkpoints": checkpoints}
    explained = explained_by_lines(doc, {0: Decimal(20), 1: Decimal(20)})
    assert [(i.message, needed) for i, needed in explained] == [
        ("lines at 7% add up to 55.0, invoice says 20", [1]),
        ("lines at 20% add up to 119.0, invoice says 40", [0]),
        ('lines 1-1 add up to 99.0, but "Section A" says 20', [0]),
        ('lines 2-3 add up to 75.0, but "Section B" says 40', [1]),
    ]


def test_on_a_page_only_the_control_totals_it_can_verify_are_explained() -> None:
    lines = [line(1, amount=99.0), line(2)]
    doc = {"header": {"subtotal": 1, "tax_total": 1}, "lines": lines}
    doc["checkpoints"] = [cp(30, 0, "Carried forward"), cp(70, 2, "Carried forward")]
    explained = explained_by_lines(doc, {0: Decimal(20)}, partial=True)
    assert [(i.message, needed) for i, needed in explained] == [
        ('lines 1-2 add up to 119.0, but "Carried forward" says 70', [0])
    ]


def misread_after(near: Line, wrong: int) -> tuple[JSONObject, dict[int, Decimal]]:
    """`near`, then `wrong` lines read as 99.0 for 2 x 10.0, and a header printed one cent
    above the lines at quantity x unit price; with the amounts that would settle them."""
    lines = [near] + [line(n, amount=99.0) for n in range(2, wrong + 2)]
    subtotal = Decimal(20) * len(lines)
    tax = subtotal / 5
    header: JSONObject = {"subtotal": float(subtotal + CENT), "tax_total": float(tax)}
    header |= {"supplier_name": "ACME Ltd", "total": float(subtotal + CENT + tax)}
    return {"header": header, "lines": lines}, {i: Decimal(20) for i in range(len(lines))}


def test_a_line_whose_amount_is_not_needed_is_left_out() -> None:
    # 20.02 for 20.00: with the other line at 20.00, the lines add up to 40.00 or 40.02,
    # both within a cent of 40.01, and the tax is 8.00 either way.
    doc, amounts = misread_after(line(1, amount=20.02), wrong=1)
    explained = explained_by_lines(doc, amounts)
    assert [(i.check, needed) for i, needed in explained] == [("doc_sums", [1]), ("tax", [1])]


def test_above_the_budget_every_line_of_an_issue_is_taken_untested() -> None:
    # Lines with a new amount times rows of the document: (n + 1) ** 2 is over the budget.
    n = math.isqrt(TEST_BUDGET)
    doc, amounts = misread_after(line(1, amount=20.02), wrong=n)
    assert len(amounts) * len(doc["lines"]) > TEST_BUDGET
    explained = explained_by_lines(doc, amounts)
    everyone = sorted(amounts)
    assert [(i.check, needed) for i, needed in explained] == [
        ("doc_sums", everyone),
        ("tax", everyone),
    ]
    # The lines of the issue, not every line: a control total takes those it adds up
    # (Section A adds up the first line as read) and the tax of a rate those of the rate.
    section_b = cp(20.0 * n, len(amounts), "Section B")
    doc["checkpoints"] = [cp(20.02, 1, "Section A"), section_b]
    for item in doc["lines"][1::2]:
        item["tax_rate"] = 7
    of_rate = {
        r: [i for i, item in enumerate(doc["lines"]) if item["tax_rate"] == r] for r in (7, 20)
    }
    doc["header"]["tax_breakdown"] = [
        {"rate": r, "base": 20.0 * len(at), "tax": 0} for r, at in of_rate.items()
    ]
    explained = explained_by_lines(doc, amounts)
    assert [(i.message.split(" add")[0], needed) for i, needed in explained][1:] == [
        ("lines at 7%", of_rate[7]),
        ("lines at 20%", of_rate[20]),
        (f"lines 2-{n + 1}", everyone[1:]),
    ]
    # Within the budget, the line is tested.
    doc, amounts = misread_after(line(1, amount=20.02), wrong=n - 1)
    assert len(amounts) * len(doc["lines"]) <= TEST_BUDGET
    explained = explained_by_lines(doc, amounts)
    assert [needed for _, needed in explained] == [sorted(amounts)[1:]] * 2


def test_the_budget_counts_control_totals_and_tax_rows_too() -> None:
    doc, amounts = misread_after(line(1, amount=20.02), wrong=99)  # 100 x 100 rows

    def subtotal_needs() -> list[list[int]]:
        return [needed for i, needed in explained_by_lines(doc, amounts) if i.check == "doc_sums"]

    assert subtotal_needs() == [sorted(amounts)[1:]]
    # The rate of the lines, then rates no line carries, printed with no base: they raise
    # nothing, but each test of a line goes through them.
    rows = TEST_BUDGET // len(amounts) - len(doc["lines"]) + 1
    breakdown = [{"rate": 20, "base": 20.0 * len(amounts), "tax": 0}]
    breakdown += [{"rate": 100 + r, "base": 0, "tax": 0} for r in range(rows - 1)]
    doc["header"]["tax_breakdown"] = breakdown
    assert check_tax(doc | {"lines": [line(n) for n in range(1, 101)]}) == []
    assert subtotal_needs() == [sorted(amounts)]


# What explained_by_lines answers, worked out the slow way it is defined: each check on
# the document as read and with the new amounts, then once more without each line's new
# amount. An issue is told apart from the others of its check by what its message says it
# verifies, so this does not depend on how the checks key their issues.
_VERIFIES = re.compile(r"^lines add up to|^tax computed|lines (?:at|carry) [^ ]+%|^lines \d+-\d+")


def explained_the_slow_way(
    doc: JSONObject, amounts: dict[int, Decimal], partial: bool
) -> list[tuple[str, list[int]]]:
    def verifies(issues: list[Issue]) -> dict[str, Issue]:
        found = (_VERIFIES.search(i.message) for i in issues)
        return {m.group(): i for m, i in zip(found, issues, strict=True) if m}

    def doc_sums(d: JSONObject) -> list[Issue]:
        return check_doc_sums(d)

    def checkpoints(d: JSONObject) -> list[Issue]:
        return check_checkpoints(d, partial=partial)

    def changed(new: dict[int, Decimal]) -> JSONObject:
        lines = [
            item | {"amount": new[i]} if i in new else item for i, item in enumerate(doc["lines"])
        ]
        return doc | {"lines": lines}

    out = []
    for check in [checkpoints] if partial else [doc_sums, check_tax, checkpoints]:
        after = verifies(check(changed(amounts)))
        for key, issue in verifies(check(doc)).items():
            if key in after:
                continue
            needed = [
                i
                for i in sorted(amounts)
                if key in verifies(check(changed({j: v for j, v in amounts.items() if j != i})))
            ]
            if needed:
                out.append((issue.message, needed))
    return out


def random_document(rng: random.Random) -> tuple[JSONObject, dict[int, Decimal]]:
    """Up to nine lines, about half with a wrong or missing amount, a header and control
    totals printed for the right amounts or a little off; and the right amounts of the
    wrong lines."""
    lines, right = [], []
    for n in range(1, rng.randint(1, 9) + 1):
        item = line(n, qty=rng.choice([1, 2, 3]), price=rng.choice([2.5, 10.0, 7.0]))
        item["tax_rate"] = rng.choice([20, 20, 7, 0, None])
        right.append(Decimal(str(item["amount"])))
        if rng.random() < 0.5:
            item["amount"] = rng.choice([99.0, 0.0, None, item["amount"] + 0.02])
        lines.append(item)
    off = [Decimal(0), Decimal(0), CENT, Decimal(3)]
    base: dict[int, Decimal] = {}
    for a, item in zip(right, lines, strict=True):
        if item["tax_rate"] is not None:
            base[item["tax_rate"]] = base.get(item["tax_rate"], Decimal(0)) + a
    subtotal = sum(right, Decimal(0)) + rng.choice(off)
    tax = sum((money(b * r / 100) for r, b in base.items()), Decimal(0)) + rng.choice(off)
    header: JSONObject = {"supplier_name": "ACME Ltd", "subtotal": float(subtotal)}
    header |= {"tax_total": float(tax), "total": float(subtotal + tax)}
    if rng.random() < 0.5:
        header["tax_breakdown"] = [
            {"rate": r, "base": float(b + rng.choice(off))} for r, b in sorted(base.items())
        ]
    checkpoints: list[Checkpoint] = []
    for pos in sorted(rng.sample(range(len(lines) + 1), rng.randint(0, 2))):
        start = checkpoints[-1]["after_line"] if checkpoints else 0
        running, since = sum(right[:pos], Decimal(0)), sum(right[start:pos], Decimal(0))
        amount = rng.choice([running, since]) + rng.choice(off)
        checkpoints.append(cp(float(amount), pos, f"Total {pos}"))
    doc = {"header": header, "lines": lines, "checkpoints": checkpoints}
    wrong = {
        i: a for i, (a, item) in enumerate(zip(right, lines, strict=True)) if item["amount"] != a
    }
    return doc, wrong


def test_explained_by_lines_agrees_with_the_slow_way_on_random_documents() -> None:
    rng = random.Random(20261127)
    seen = Counter[tuple[str, bool]]()
    for _ in range(500):
        doc, amounts = random_document(rng)
        assert len(amounts) * 40 <= TEST_BUDGET  # 9 lines, 2 totals, 4 tax rows at most
        for partial in (False, True):
            found = explained_by_lines(doc, amounts, partial)
            assert [(i.message, needed) for i, needed in found] == explained_the_slow_way(
                doc, amounts, partial
            ), doc
            seen.update((i.check, len(needed) > 1) for i, needed in found)
            # The subtotal depends on every line; some lines were not needed for it.
            seen.update(
                ("left out", True)
                for i, needed in found
                if i.check == "doc_sums" and needed != sorted(amounts)
            )
    # Every kind of issue was explained, by one line and by several.
    assert len(seen) == 7 and min(seen.values()) >= 20, seen


def test_agree_with_subtotal() -> None:
    def agree(lines: list[Line], subtotal: float, amounts: dict[int, int]) -> list[int]:
        doc = {"header": {"subtotal": subtotal}, "lines": lines}
        return agree_with_subtotal(doc, {i: Decimal(v) for i, v in amounts.items()})

    wrong = [line(1), line(2, amount=99.0)]
    assert agree(wrong, 40.0, {1: 20}) == [1]
    assert agree(wrong, 40.01, {1: 20}) == [1]  # within a cent
    assert agree(wrong, 40.02, {1: 20}) == []
    assert agree_with_subtotal({"header": {}, "lines": wrong}, {1: Decimal(20)}) == []
    assert agree([], 0.0, {}) == []
    # Two wrong lines: each agrees with the other at quantity x unit price, neither alone.
    two = [line(1, amount=21.0), line(2, amount=99.0), line(3)]
    assert agree(two, 60.0, {0: 20, 1: 20}) == [0, 1]
    assert agree(two, 60.0, {1: 20}) == []
    # A missing amount counts as none.
    assert agree([line(1), line(2) | {"amount": None}], 40.0, {1: 20}) == [1]
    # 20.02 for 20.00: the subtotal 20.01 fits the amount as read too, and raises no
    # issue; the subtotal 20.00 does.
    assert agree([line(1, amount=20.02)], 20.01, {0: 20}) == []
    assert agree([line(1, amount=20.02)], 20.0, {0: 20}) == [0]
    # Three wrong lines: each agrees with the others at quantity x unit price.
    three = [line(1, amount=21.0), line(2, amount=99.0), line(3, amount=30.0)]
    assert agree(three, 60.0, {0: 20, 1: 20, 2: 20}) == [0, 1, 2]
    # Line 0 alone settles the subtotal with the others as read: 20 + 15 + 10 = 45.
    # Line 1 is in doubt for its quantity (4 x 5.0 = 20.00, but 15.0, the right
    # amount), and with it at 20.0 nothing adds up to 45.
    mixed = [
        line(1, qty=2, price=10.0, amount=21.0),
        line(2, qty=4, price=5.0, amount=15.0),
        line(3, qty=4, price=2.5),
    ]
    assert agree(mixed, 45.0, {0: 20, 1: 20}) == [0]
    # +5 and -5 cancel: the lines add up as read, so the subtotal raises no issue, but
    # with the other at quantity x unit price each amount makes them add up and its
    # amount as read does not.
    cancel = [line(1, amount=25.0), line(2, amount=15.0), line(3)]
    assert agree(cancel, 60.0, {0: 20, 1: 20}) == [0, 1]


def test_ranges_join_runs_of_numbers() -> None:
    assert ranges([12, 2, 6, 5, 7]) == "2, 5–7 and 12"
    assert (ranges([3]), ranges([1, 2])) == ("3", "1–2")
