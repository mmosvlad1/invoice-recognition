"""The layout parser on its own: text lines with coordinates, the model's answer turned
into column anchors, and rows cut into cells."""

from dataclasses import replace
from decimal import Decimal
from typing import Any

import pytest

from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition.layout import Column, Layout, TextLine, Word, parse_document
from invoice_recognition.shapes import JSONObject

from fakes import make_pdf


def left(text: str, x0: float) -> Word:
    """A word that starts at x0 (5 points per character)."""
    return Word(text, x0, x0 + 5 * len(text))


def right(text: str, x1: float) -> Word:
    """A word that ends at x1."""
    return Word(text, x1 - 5 * len(text), x1)


LAYOUT = Layout(
    header_line=1,
    decimal=".",
    thousands=",",
    tax_in_row=False,
    document_rate=None,
    description_wraps=True,
    non_item=[("carry_forward", "carried forward")],
    discount_marker=None,
    columns=[
        Column("sku", "SKU", "l", 40.0, 1),
        Column("description", "Description", "l", 120.0, 1),
        Column("quantity", "Qty", "r", 400.0, 1),
        Column("unit_price", "Price", "r", 470.0, 1),
        Column("amount", "Amount", "r", 550.0, 1),
    ],
)


def item(page: int, index: int, sku: str, desc: str, qty: str, price: str, amount: str) -> TextLine:
    words = [left(sku, 40.0), left(desc, 120.0), right(qty, 400.0), right(price, 470.0)]
    return TextLine(page, index, [*words, right(amount, 550.0)])


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


def test_description_continues_across_a_page_break() -> None:
    footer = TextLine(1, 2, [left("Page", 30.0), left("1", 300.0)])
    top = TextLine(2, 1, [left("ACME", 30.0), left("Ltd", 300.0)])
    pages = [
        [item(1, 1, "A-1", "Bolt", "2", "10.00", "20.00"), footer],
        [
            top,
            TextLine(2, 2, [left("zinc", 120.0)]),
            item(2, 3, "A-2", "Nut", "3", "1.00", "3.00"),
            TextLine(2, 4, [left("plated", 120.0)]),
        ],
    ]
    parsed = parse_document(pages, LAYOUT)
    assert [d["description"] for d in parsed.lines] == ["Bolt zinc", "Nut plated"]
    assert [(n["kind"], n["page"]) for n in parsed.notes] == [("joined_across_pages", 2)]

    # a declared row between the two (a carry-forward at the top of the page) closes the item
    carried = [left("Carried", 300.0), left("forward", 345.0), right("20.00", 550.0)]
    pages[1].insert(1, TextLine(2, 9, carried))
    parsed = parse_document(pages, LAYOUT)
    assert [d["description"] for d in parsed.lines] == ["Bolt", "Nut plated"]
    assert parsed.notes == [] and len(parsed.checkpoints) == 1


def test_items_totals_and_doubtful_rows() -> None:
    page = [
        item(1, 1, "A-1", "Bolt", "1,000", "0.25", "250.00"),
        TextLine(1, 2, [left("Carried", 40.0), left("forward", 85.0), left("250.00", 300.0)]),
        TextLine(1, 3, [left("Note", 40.0), right("12.00", 550.0)]),
        item(1, 4, "A-2", "Nut", "x", "1.00", "3.00"),
    ]
    parsed = parse_document([page], LAYOUT)
    assert parsed.lines == [
        {
            "line_no": None,
            "sku": "A-1",
            "description": "Bolt",
            "quantity": Decimal(1000),
            "unit": "",
            "unit_price": Decimal("0.25"),
            "tax_rate": None,
            "amount": Decimal("250.00"),
            "page": 1,
        }
    ]
    # the total is not on the amount anchor: the last number of the row is taken
    assert parsed.checkpoints == [
        {
            "label": "Carried forward 250.00",
            "amount": Decimal("250.00"),
            "after_line": 1,
            "page": 1,
            "kind": "carry_forward",
        }
    ]
    assert [(n["row"], n["kind"]) for n in parsed.notes] == [
        (3, "unclassified_amount"),
        (4, "unclassified_amount"),
    ]


def test_discount_row_needs_only_an_amount() -> None:
    disc = replace(LAYOUT, discount_marker="rebate")
    row = TextLine(1, 1, [left("Volume", 120.0), left("rebate", 160.0), right("-5.00", 550.0)])
    (line,) = parse_document([[row]], disc).lines
    assert line["amount"] == Decimal("-5.00") and line["quantity"] is None
    assert parse_document([[row]], LAYOUT).lines == []


def test_assign_cuts_a_row_into_cells() -> None:
    words = [left("A-1", 40.0), left("Hex", 120.0), left("bolt", 140.0), right("2", 400.0)]
    words += [left("pcs", 405.0), right("10.00", 470.0), right("20.00", 550.0)]
    cells, stray = lay.assign(words, LAYOUT.columns)
    assert {i: [w.text for w in ws] for i, ws in cells.items()} == {
        0: ["A-1"],
        1: ["Hex", "bolt"],
        2: ["2"],
        3: ["pcs", "10.00"],  # a word on no anchor opens the right-aligned cell after it
        4: ["20.00"],
    }
    assert stray == []


@pytest.mark.parametrize(
    ("decimal", "thousands", "token", "value"),
    [
        (".", ",", "1,234.50", Decimal("1234.50")),
        (",", ".", "1.234,50", Decimal("1234.50")),
        (",", ".", "5.000", Decimal(5000)),
        (".", "", "1234.5", Decimal("1234.5")),
        (".", ",", "(12.00)", Decimal("-12.00")),
        (".", ",", "12.00-", Decimal("-12.00")),
        (".", ",", "−3", Decimal(-3)),
        (".", ",", "€7.10", Decimal("7.10")),
        (".", ",", "1,23.4", None),  # not grouped by thousands
        (".", ",", "12,50", None),
        (".", ",", "abc", None),
        (".", ",", "", None),
    ],
)
def test_number_is_parsed_strictly_in_the_declared_format(
    decimal: str, thousands: str, token: str, value: Decimal | None
) -> None:
    assert replace(LAYOUT, decimal=decimal, thousands=thousands).number(token) == value


# --------------------------------------------------------------------------- #
# The model's answer
# --------------------------------------------------------------------------- #

PAGE = [
    TextLine(1, 1, [left("Invoice", 40.0), left("INV-1", 120.0)]),
    TextLine(
        1,
        2,
        [left("SKU", 40.0), left("Description", 120.0), right("Qty", 400.0)]
        + [left("Unit", 420.0), left("price", 445.0), right("Amount", 550.0)],
    ),
    item(1, 3, "A-1", "Bolt", "2", "10.00", "20.00"),
    item(1, 4, "A-2", "Nut", "3", "1.00", "3.00"),
]
ANSWER: JSONObject = {
    "header_line": 2,
    "columns": [
        {"label": "SKU", "field": "sku"},
        {"label": "Description", "field": "description"},
        {"label": "Qty", "field": "quantity"},
        {"label": "Unit price", "field": "unit_price"},
        {"label": "Amount", "field": "amount"},
    ],
    "numbers": {"decimal": ".", "thousands": ","},
    "tax": {"in_row": False, "document_rate": 20},
    "description_wraps": True,
    "non_item_rows": [{"kind": "carry_forward", "starts_with": "Carried  Forward"}],
    "discount_rows": {"present": False, "marker": "ignored"},
}


def test_resolve_measures_where_each_column_sits() -> None:
    got = lay.resolve(ANSWER, PAGE)
    assert got.header_line == 2
    assert [(c.field, c.align, c.anchor) for c in got.columns] == [
        ("sku", "l", 40.0),
        ("description", "l", 120.0),
        ("quantity", "r", 400.0),
        ("unit_price", "r", 470.0),
        ("amount", "r", 550.0),
    ]
    assert got.document_rate == Decimal(20) and not got.tax_in_row
    assert got.non_item == [("carry_forward", "carried forward")]
    assert got.discount_marker is None
    rows = parse_document([PAGE], got).lines
    assert [(r["sku"], r["tax_rate"]) for r in rows] == [("A-1", Decimal(20)), ("A-2", Decimal(20))]


def test_resolve_finds_the_labels_on_another_line_than_declared() -> None:
    assert lay.resolve(ANSWER | {"header_line": 7}, PAGE).header_line == 2


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"columns": []}, "no columns"),
        ({"columns": ANSWER["columns"][:-1]}, "no amount column"),
        (
            {"columns": [*ANSWER["columns"], {"label": "Amount", "field": "amount"}]},
            "more than one column: amount",
        ),
        ({"numbers": {"decimal": ",", "thousands": ","}}, "unusable number format"),
        ({"numbers": "dot"}, "unusable number format"),
        (
            {
                "columns": [
                    c | {"label": "Cost"} if c["field"] == "unit_price" else c
                    for c in ANSWER["columns"]
                ]
            },
            "not found together",
        ),
    ],
)
def test_unusable_answer_raises_a_layout_error(change: JSONObject, error: str) -> None:
    with pytest.raises(lay.LayoutError, match=error):
        lay.resolve(ANSWER | change, PAGE)


@pytest.mark.parametrize("answer", [None, [], "columns"])
def test_answer_that_is_not_an_object_raises_a_layout_error(answer: Any) -> None:
    with pytest.raises(lay.LayoutError, match="not a JSON object"):
        lay.resolve(answer, PAGE)


def test_heading_with_nothing_under_it_raises_a_layout_error() -> None:
    page = [*PAGE[:2], TextLine(1, 3, [left("A-1", 40.0)])]
    with pytest.raises(lay.LayoutError, match="nothing is printed under the heading 'Description'"):
        lay.resolve(ANSWER, page)


# --------------------------------------------------------------------------- #
# The text layer and the request
# --------------------------------------------------------------------------- #


def test_text_lines_keep_the_words_and_their_x() -> None:
    pdf = make_pdf([["SKU-1   Bolt   20.00", "second line"], ["page two"]])
    pages = lay.text_lines(pdf)
    assert [[tl.text for tl in p] for p in pages] == [
        ["SKU-1 Bolt 20.00", "second line"],
        ["page two"],
    ]
    first = pages[0][0]
    assert (first.page, first.index) == (1, 1) and first.words[0].x0 == pytest.approx(40, abs=1)
    assert first.words[0].x1 < first.words[1].x0 < first.words[2].x0


def test_layout_request_shows_the_page_and_the_top_of_the_next() -> None:
    pages = [[TextLine(1, i, [left(f"row{i}", 40.0)]) for i in (1, 2)]]
    pages.append([TextLine(2, i, [left(f"next{i}", 40.0)]) for i in range(1, 9)])
    text = lay.layout_request(pages, 1)
    assert text.startswith(lay.LAYOUT_PROMPT + "\n--- page 1 of 2 ---\n1| row1\n2| row2\n")
    assert text.endswith(
        "--- first lines of page 2 (not numbered) ---\n"
        + "\n".join(f"next{i}" for i in range(1, 7))
        + "\n"
    )
    assert lay.layout_request(pages, 2).endswith("8| next8\n")  # the last page: nothing added


@pytest.mark.parametrize(
    ("rate", "expected"),
    [(20, Decimal(20)), ("7.5", Decimal("7.5")), (None, None), (True, None), ("n/a", None)]
    + [("NaN", None), ("1e999999", None), ("1e5000", None)],
)
def test_document_rate_must_be_a_usable_number(rate: object, expected: Decimal | None) -> None:
    got = lay.resolve(ANSWER | {"tax": {"in_row": False, "document_rate": rate}}, PAGE)
    assert got.document_rate == expected


def test_printed_number_of_absurd_length_is_not_a_number() -> None:
    assert LAYOUT.number("1" * 101) == Decimal("1" * 101)
    assert LAYOUT.number("1" * 102) is None  # 10^101 or more: beyond what checks.dec accepts
    assert replace(LAYOUT, thousands="").number("0." + "1" * 101) is None
    row = item(1, 1, "A-1", "Bolt", "2", "10.00", "1" * 5000)
    parsed = parse_document([[row]], LAYOUT)
    assert parsed.lines == [] and [n["kind"] for n in parsed.notes] == []
