"""Two readings of one scanned page compared row by row (second_reading.py). Pure functions:
no request, no flow."""

import hashlib
import json

import pytest

from invoice_recognition.llm import parse_json
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition import second_reading as sr
from invoice_recognition.recognition.prompts import EXTRACT_PAGE
from invoice_recognition.shapes import Line

from fakes import line
from samples import saved_scan_pages, saved_second_pages


def unnumbered(*lines: Line) -> list[Line]:
    return [item | {"line_no": None} for item in lines]


# --------------------------------------------------------------------------- #
# One value
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "first", "second"),
    [
        ("tax_rate", 20, 20.0),
        ("tax_rate", 20, "20%"),
        ("tax_rate", 20, "20 %"),
        ("tax_rate", 20, 0.2),  # a rate written as a fraction
        ("tax_rate", "7%", 0.07),
        ("tax_rate", "0.2", "20"),
        ("tax_rate", 0.5, 0.5),
        ("quantity", "6600.0", 6600),
        ("amount", " 12.50 ", 12.5),
        ("amount", "1,234.50", 1234.5),
        ("amount", "1 234.50", 1234.5),
        ("amount", "1\u00a0234,50", 1234.5),  # a no-break space, a decimal comma
        ("amount", "1.234,50", "1,234.50"),
        ("amount", "1.234.567", 1234567),
        ("amount", "1,234", 1234),  # three digits after one comma: thousands
        ("unit_price", "10,5", 10.5),
        ("unit_price", "0,500", 0.5),
        ("amount", "\u22125.00", -5),  # a Unicode minus
        ("amount", "\u20135.00", -5),  # an en dash written for a minus
        ("sku", "sg-54066", " SG-54066 "),
        ("sku", "SG  54066", "sg 54066"),
        ("sku", "*987123*", "987123"),  # punctuation around the code
        ("sku", "(AB-12).", "ab-12"),
        ("sku", 12345.0, "12345"),  # a whole number written as a float
        ("sku", 0, "0"),
        ("amount", None, None),
        ("sku", None, ""),
        ("sku", None, True),  # a flag is no SKU
        ("sku", None, ["A-1"]),
        ("amount", "1e999999", None),  # an absurd number is no number (checks.dec)
        ("amount", "9" * 400, None),
        ("amount", float("nan"), None),
        ("amount", True, None),
        ("amount", [20], None),
        ("amount", "n/a", None),
    ],
)
def test_values_that_agree(field: str, first: object, second: object) -> None:
    assert sr.same(field, first, second) and sr.same(field, second, first)


@pytest.mark.parametrize(
    ("field", "first", "second"),
    [
        ("sku", "SG-50406", "SG-54066"),
        ("sku", "AB-12", "AB.12"),
        ("sku", "SG-54066", "SG54066"),  # a dropped dash is a misread
        ("sku", "AB-12", "A-B12"),
        ("sku", "AB-12", "AB 12"),
        ("sku", "AB-12", "AB\u201312"),  # an en dash is not a hyphen
        ("sku", 12345.5, "12345.5"),  # not a whole number: no SKU
        ("tax_rate", "0.2%", 20),  # written with "%": never a fraction
        ("tax_rate", "0.5%", 50),
        ("sku", "SG-54066", None),
        ("sku", 0, None),
        ("quantity", 66, 6600),
        ("amount", None, 0),
        ("tax_rate", 20, "7%"),
        ("tax_rate", 1, 100),  # 1 is one percent, not a fraction
        ("amount", "1,234.50", "1.23450"),
    ],
)
def test_values_that_disagree(field: str, first: object, second: object) -> None:
    assert not sr.same(field, first, second) and not sr.same(field, second, first)


# --------------------------------------------------------------------------- #
# Which line is which
# --------------------------------------------------------------------------- #


def test_lines_are_matched_by_their_position_numbers() -> None:
    first = [line(1), line(2), line(3)]
    second = [line(3), line(1), line(2)]
    found = sr.align(first, second)
    assert (found.by, found.pairs) == ("line_no", [(0, 1), (1, 2), (2, 0)])
    assert sr.compare_page(first, second, 1) == []


def test_lines_without_numbers_are_matched_in_order_when_the_counts_agree() -> None:
    first = unnumbered(line(1), line(2))
    found = sr.align(first, unnumbered(line(1), line(2)))
    assert (found.by, found.pairs, found.only_first, found.only_second) == (
        "order",
        [(0, 0), (1, 1)],
        [],
        [],
    )


@pytest.mark.parametrize(
    "second",
    [
        [line(1), line(1, qty=3)],  # the same number twice
        [line(1), line(2) | {"line_no": None}],  # a line without a number
        [line(1), line(2) | {"line_no": "2"}],  # a number that is text
        [line(1), line(2) | {"line_no": True}],  # a flag is no number
        unnumbered(line(1), line(2)),  # numbers on one side only
    ],
    ids=["duplicate", "missing", "text", "bool", "one-side"],
)
def test_numbers_that_cannot_be_trusted_fall_back_to_order(second: list[Line]) -> None:
    assert sr.align([line(1), line(2)], second).by == "order"


def test_rows_counted_from_one_on_one_side_are_paired_in_order() -> None:
    """The first reader gives the printed positions 79-81, the second counts the rows of
    the page 1, 2, 3: the numbers say nothing about each other."""
    first = [line(n) for n in (79, 80, 81)]
    second = [line(n) | {"sku": f"SKU-{n:03d}"} for n in (1, 2, 3)]
    second = [row | {"sku": item["sku"]} for row, item in zip(second, first, strict=True)]
    found = sr.align(first, second)
    assert (found.by, found.pairs) == ("order", [(0, 0), (1, 1), (2, 2)])
    assert sr.compare_page(first, second, 3) == []
    # With different counts nothing is paired: one issue for the page.
    (issue,) = sr.compare_page(first, second[:2], 3)
    assert "could not be matched one to one" in issue.message


def test_one_row_without_a_number_makes_the_reading_unnumbered() -> None:
    first = [line(1), line(2), line(3) | {"line_no": None}]
    assert sr.align(first, [line(1), line(2), line(3)]).by == "order"
    assert sr.align(first, [line(1), line(2)]).by == "none"


def test_different_counts_without_numbers_give_one_page_issue_and_no_row_issue() -> None:
    first = unnumbered(line(1), line(2), line(3))
    second = unnumbered(line(1, qty=5), line(2))  # would disagree, if it were compared
    found = sr.align(first, second)
    assert (found.by, found.pairs) == ("none", [])
    (issue,) = sr.compare_page(first, second, 2)
    assert (issue.check, issue.severity, issue.page, issue.line, issue.fields) == (
        "second_reading",
        "warning",
        2,
        None,
        [],
    )
    assert issue.suggestions == []
    assert issue.message == (
        "Page 2: a second reading by another model finds 2 line items; the result has 3. "
        "The lines could not be matched one to one, so they were not compared. "
        "Please check this page."
    )


def test_positions_only_one_reading_has_are_named_in_one_page_issue() -> None:
    first = [line(n) for n in (1, 2, 3, 7)]
    second = [line(n) for n in (1, 2, 4, 5, 6)]
    found = sr.align(first, second)
    assert (found.pairs, found.only_first, found.only_second) == (
        [(0, 0), (1, 1)],
        [2, 3],
        [2, 3, 4],
    )
    (issue,) = sr.compare_page(first, second, 3)
    assert (issue.line, issue.fields) == (None, [])
    assert issue.message == (
        "Page 3: a second reading by another model finds 5 line items; the result has 4. "
        "Positions 4–6 are not in the result. Positions 3 and 7 of the result were not "
        "found by the second reading. Please check this page."
    )


def test_one_position_missing_on_one_side_reads_in_the_singular() -> None:
    (issue,) = sr.compare_page([line(1), line(2)], [line(1)], 1)
    assert issue.message == (
        "Page 1: a second reading by another model finds 1 line item; the result has 2. "
        "Position 2 of the result was not found by the second reading. Please check this page."
    )


def test_a_matched_row_that_differs_comes_before_the_page_issue() -> None:
    issues = sr.compare_page([line(1), line(2)], [line(1, qty=4), line(3)], 1)
    assert [(i.line, i.fields) for i in issues] == [
        (0, ["line.quantity", "line.amount"]),
        (None, []),
    ]


# --------------------------------------------------------------------------- #
# What a disagreement says
# --------------------------------------------------------------------------- #


def test_one_differing_sku_is_one_issue_with_the_second_readers_value() -> None:
    first = [line(126), line(127) | {"sku": "SG-50406"}]
    second = [line(126), line(127) | {"sku": "SG-54066"}]
    (issue,) = sr.compare_page(first, second, 4)
    assert issue.to_json() == {
        "check": "second_reading",
        "message": "Position 127 on page 4: a second reading by another model gives SKU "
        "SG-54066 instead of SG-50406. One of the two readings is wrong; please check this "
        "line.",
        "line": 1,
        "page": 4,
        "severity": "warning",
        "line_no": 127,
        "fields": ["line.sku"],
        "suggestions": [{"set": {"line.sku": "SG-54066"}, "source": "second_reading"}],
    }


def test_several_differing_fields_are_one_issue_in_a_fixed_order() -> None:
    first = [line(5, qty=2, price=10.0)]
    # In the answer the fields come in another order than they are compared in.
    second = [
        {
            "amount": "30.00",
            "tax_rate": "7%",
            "unit_price": 15,
            "quantity": 2,
            "sku": "SKU-005",
            "line_no": 5,
        }
    ]
    (issue,) = sr.compare_page(first, second, 1)
    assert issue.fields == ["line.unit_price", "line.tax_rate", "line.amount"]
    # Numbers as numbers, whatever the second reader wrote.
    assert issue.suggestions == [
        {
            "set": {"line.unit_price": 15, "line.tax_rate": 7, "line.amount": 30},
            "source": "second_reading",
        }
    ]
    assert issue.message == (
        "Position 5 on page 1: a second reading by another model gives unit price 15 instead "
        "of 10, tax rate 7% instead of 20% and amount 30 instead of 20. One of the two "
        "readings is wrong; please check this line."
    )


def test_a_value_the_second_reader_left_empty_is_in_doubt_but_not_suggested() -> None:
    first = [line(1), line(2)]
    second = [line(1), line(2) | {"quantity": None, "amount": 25.0}]
    (issue,) = sr.compare_page(first, second, 1)
    assert issue.fields == ["line.quantity", "line.amount"]
    assert issue.suggestions == [{"set": {"line.amount": 25}, "source": "second_reading"}]
    assert "gives no quantity where the result has 2 and amount 25 instead of 20" in (issue.message)
    # Nothing it read differs: no suggestion at all, only the doubt.
    (issue,) = sr.compare_page(first, [line(1), line(2) | {"sku": None}], 1)
    assert (issue.fields, issue.suggestions) == (["line.sku"], [])
    assert "gives no SKU where the result has SKU-002" in issue.message


@pytest.mark.parametrize(
    "value", [True, float("nan"), float("inf"), [1, 2], {"a": 1}, "9" * 400, "n/a"]
)
def test_a_value_that_is_no_number_is_never_quoted_or_suggested(value: object) -> None:
    (issue,) = sr.compare_page([line(1)], [line(1) | {"amount": value}], 1)
    assert issue.fields == ["line.amount"] and issue.suggestions == []
    assert "gives no amount where the result has 20" in issue.message
    json.dumps(issue.to_json(), allow_nan=False)  # strict JSON


def test_a_long_sku_is_shortened_in_the_message_and_kept_in_the_suggestion() -> None:
    code = "X" * 60
    (issue,) = sr.compare_page([line(1)], [line(1) | {"sku": code}], 1)
    assert f"SKU {'X' * 39}… instead of SKU-001" in issue.message
    assert issue.suggestions[0]["set"] == {"line.sku": code}


def test_a_value_only_the_second_reader_read_is_suggested() -> None:
    (issue,) = sr.compare_page([line(1) | {"tax_rate": None}], [line(1)], 1)
    assert issue.suggestions == [{"set": {"line.tax_rate": 20}, "source": "second_reading"}]
    assert "gives tax rate 20% where the result has none" in issue.message


def test_an_unnumbered_row_is_named_by_its_place_on_the_page() -> None:
    first = unnumbered(line(1), line(2))
    (issue,) = sr.compare_page(first, unnumbered(line(1), line(2, qty=3)), 6)
    assert (issue.line, issue.line_no) == (1, None)
    assert issue.message.startswith("Line item 2 on page 6: a second reading by another model")


def test_description_unit_and_other_keys_are_never_compared() -> None:
    first = [line(1) | {"description": "Bolt M8 zinc plated", "unit": "pcs", "page": 1}]
    second = [line(1) | {"description": "Bolt M8", "unit": "box", "repaired": "quantity_x1000"}]
    assert sr.compare_page(first, second, 1) == []


def test_nothing_read_on_either_side_agrees() -> None:
    assert sr.compare_page([], [], 1) == []
    (issue,) = sr.compare_page([], [line(1)], 1)
    assert issue.message.startswith("Page 1: a second reading by another model finds 1 line item")


# --------------------------------------------------------------------------- #
# A whole column read differently
# --------------------------------------------------------------------------- #


def page_of(n: int, **change: object) -> list[Line]:
    return [line(k) | change for k in range(1, n + 1)]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (page_of(5, tax_rate=20), page_of(5, tax_rate=0.2)),
        (page_of(5, amount=1234.5), page_of(5, amount="1,234.50")),
        (page_of(5, amount=1234.5), page_of(5, amount="1 234.50")),
        (page_of(5, unit_price=10.5), page_of(5, unit_price="10,5")),
        (page_of(5, amount=-5), page_of(5, amount="\u22125")),
        (page_of(5, sku="*AB-12*"), page_of(5, sku="AB-12")),
    ],
    ids=[
        "tax-fraction",
        "thousands-comma",
        "thousands-space",
        "decimal-comma",
        "minus",
        "sku-punctuation",
    ],
)
def test_a_column_written_another_way_raises_nothing(first: list[Line], second: list[Line]) -> None:
    assert sr.compare_page(first, second, 1) == []


def test_skus_spelled_differently_on_most_rows_are_one_issue_for_the_column() -> None:
    first = [line(n) | {"sku": f"AB-{n:02d}"} for n in range(1, 6)]
    second = [item | {"sku": item["sku"].replace("-", " ")} for item in first]
    second[3] = second[3] | {"sku": "AB 07"}  # a misread digit, besides the spelling
    issues = sr.compare_page(first, second, 1)
    assert [(i.line, i.fields) for i in issues] == [(3, ["line.sku"]), (None, ["line.sku"])]
    assert issues[0].suggestions == [{"set": {"line.sku": "AB 07"}, "source": "second_reading"}]
    assert issues[1].message.startswith(
        "On page 1 a second reading by another model gives a different SKU on 4 of 5 line "
        "items, for example at position 1: SKU AB 01 instead of AB-01."
    )


def test_skus_misread_on_most_rows_keep_an_issue_each() -> None:
    first = [line(n) | {"sku": f"AB-{n:02d}"} for n in range(1, 5)]
    second = [item | {"sku": f"AB-{item['line_no'] + 10:02d}"} for item in first]
    issues = sr.compare_page(first, second, 1)
    assert [(i.line, i.fields) for i in issues] == [(k, ["line.sku"]) for k in range(4)]
    assert all(i.suggestions for i in issues)


def test_a_rate_written_as_a_fraction_is_shown_as_it_was_read() -> None:
    (issue,) = sr.compare_page([line(1)], [line(1) | {"tax_rate": 0.07}], 1)
    assert "tax rate 0.07 instead of 20%" in issue.message
    assert issue.suggestions == [{"set": {"line.tax_rate": 0.07}, "source": "second_reading"}]


def test_a_float_sku_is_shown_and_suggested_as_a_whole_number() -> None:
    (issue,) = sr.compare_page([line(1)], [line(1) | {"sku": 12345.0}], 1)
    assert "SKU 12345 instead of SKU-001" in issue.message
    assert issue.suggestions == [{"set": {"line.sku": "12345"}, "source": "second_reading"}]


def test_a_column_one_reader_left_empty_is_one_issue_for_the_page() -> None:
    first = page_of(4, tax_rate=None)
    second = page_of(4)
    second[2] = second[2] | {"sku": "SKU-0O3"}  # a real misread on top
    issues = sr.compare_page(first, second, 2)
    assert [(i.line, i.line_no, i.fields) for i in issues] == [
        (2, 3, ["line.sku"]),  # the row keeps its own issue for its other field
        (None, None, ["line.tax_rate"]),
    ]
    column = issues[1]
    assert (column.page, column.severity, column.suggestions) == (2, "warning", [])
    assert column.message == (
        "On page 2 a second reading by another model gives a different tax rate on 4 of 4 "
        "line items, for example at position 1: tax rate 20% where the result has none. The "
        "two readings write this column differently, or one of them misread it; please check "
        "the tax rate column on this page."
    )


@pytest.mark.parametrize(
    ("differ", "rows", "column"),
    [(2, 2, False), (3, 6, False), (3, 5, True), (6, 6, True)],
    ids=["two-rows", "half", "three-of-five", "all"],
)
def test_a_column_needs_three_rows_and_more_than_half(differ: int, rows: int, column: bool) -> None:
    first = page_of(rows)
    second = [
        item | ({"quantity": None} if k < differ else {}) for k, item in enumerate(page_of(rows))
    ]
    issues = sr.compare_page(first, second, 1)
    if column:
        assert [(i.line, i.fields) for i in issues] == [(None, ["line.quantity"])]
        assert f"on {differ} of {rows} line items" in issues[0].message
    else:
        assert [(i.line, i.fields) for i in issues] == [
            (k, ["line.quantity"]) for k in range(differ)
        ]


# --------------------------------------------------------------------------- #
# What the two real models answered
# --------------------------------------------------------------------------- #


def kept(text: str, page: int, repaired: bool = True) -> list[Line]:
    """The lines of an answer as the flow keeps a page: line items only, repaired."""
    answer = parse_json(text)
    assert answer is not None
    lines, _ = pl.rows_to_page(answer["rows"], page)
    if repaired:
        pl.repair(lines)
    return lines


def test_recorded_answers_of_two_models_disagree_on_two_skus() -> None:
    """The five pages of the scan of m_split as the cheap model answered them (with the
    "carry_in" field) and as a model of another provider answered them (the plain page
    prompt). Position 79: the first model misread the SKU; position 127: the second did -
    a disagreement says that one of them is wrong, not which."""
    first, second = saved_scan_pages()["pages"], saved_second_pages()["pages"]
    for p, rec in enumerate(second, start=1):
        prompt = EXTRACT_PAGE.format(page=p, total=5)
        assert hashlib.sha256(prompt.encode()).hexdigest() == rec["prompt_sha256"]
    issues = [
        i
        for a, b in zip(first, second, strict=True)
        for i in sr.compare_page(kept(a["text"], a["page"]), kept(b["text"], b["page"]), a["page"])
    ]
    assert [(i.page, i.line_no, i.fields, i.suggestions[0]["set"]) for i in issues] == [
        (3, 79, ["line.sku"], {"line.sku": "FW-24947"}),
        (4, 127, ["line.sku"], {"line.sku": "SG-54066"}),
    ]
    assert len(kept(first[0]["text"], 1)) == len(kept(second[0]["text"], 1)) == 25
    # Without the quantity repair seven more rows would differ (6.6 against 6600: the
    # thousands separator read as a decimal point by one of them).
    unrepaired = [
        i
        for a, b in zip(first, second, strict=True)
        for i in sr.compare_page(
            kept(a["text"], a["page"], False), kept(b["text"], b["page"], False), a["page"]
        )
    ]
    assert len(unrepaired) == 9
