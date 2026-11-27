"""The text-layer witness: printed numbers, item rows, number format."""

from decimal import Decimal

import pytest

from invoice_recognition.recognition.evidence import (
    build,
    has_text_layer,
    money,
    parse_number,
    pdf_text_layer,
    printed_forms,
)

from fakes import make_pdf

# --------------------------------------------------------------------------- #
# Printed numbers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("token", "locale", "value"),
    [
        ("1,234.56", "us", "1234.56"),
        ("1.234,56", "eu", "1234.56"),
        ("1234.56", "us", "1234.56"),
        ("1234,56", "eu", "1234.56"),
        ("12", "us", "12"),
        ("-7.50", "us", "-7.50"),
        ("−7,50", "eu", "-7.50"),  # typographic minus sign
        ("20.00,", "us", "20.00"),  # trailing punctuation of the sentence
        ("(15.00)", "us", None),  # leading bracket is not trimmed: not a number
        ("1.2.3", "eu", "123"),  # dots are thousands separators in "eu"
        ("1.2.3", "us", None),
        ("20%", "us", None),
        ("SKU-12", "us", None),
        ("", "us", None),
    ],
)
def test_parse_number(token: str, locale: str, value: str | None) -> None:
    assert parse_number(token, locale) == (None if value is None else Decimal(value))


def test_money_rounds_half_up_to_cents() -> None:
    assert money(Decimal("0.125")) == Decimal("0.13")
    assert money(Decimal("2.5")) == Decimal("2.50")
    assert money(Decimal("-0.125")) == Decimal("-0.13")


def test_printed_forms_cover_both_locales() -> None:
    forms = printed_forms(Decimal("2739.07"))
    assert {"2,739.07", "2.739,07", "2739.07", "2739,07"} <= forms
    assert "2739.1" not in forms  # never rounded


def test_printed_forms_of_a_whole_number() -> None:
    forms = printed_forms(Decimal(1200))
    assert {"1200", "1,200", "1.200", "1200.0", "1200.00", "1,200.00", "1.200,00"} <= forms


def test_printed_forms_keep_given_places_and_add_cents() -> None:
    assert {"0.5", "0,5", "0.50", "0,50"} <= printed_forms(Decimal("0.5"))
    assert {"1.125", "1,125"} <= printed_forms(Decimal("1.125"))
    assert "1.13" not in printed_forms(Decimal("1.125"))


def test_printed_forms_of_a_negative_number() -> None:
    assert {"-1,500.00", "-1.500,00", "-1500"} <= printed_forms(Decimal(-1500))


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "sNaN"])
def test_printed_forms_need_a_finite_number(value: str) -> None:
    with pytest.raises(ValueError, match="not a finite number"):
        printed_forms(Decimal(value))


# --------------------------------------------------------------------------- #
# Item rows and the number format
# --------------------------------------------------------------------------- #


def test_item_row_is_a_row_where_two_numbers_multiply_to_the_last() -> None:
    ev = build(["SKU-1 Bolt M8 4 2.50 10.00\nSubtotal 10.00\nTotal 12.00"], "text_layer")
    (page,) = ev.pages
    assert [r.text for r in page.item_rows] == ["SKU-1 Bolt M8 4 2.50 10.00"]
    assert page.item_rows[0].triple == (Decimal(4), Decimal("2.50"), Decimal("10.00"))


def test_item_row_tolerates_one_cent_and_ignores_percentages() -> None:
    rows = [
        "A-1 Cable 3 3.33 10.00",  # 9.99 printed as 10.00: within a cent
        "A-2 Nut 3 20% 2.00 6.00",  # 3 x 2.00; the tax rate is not a factor
        "A-3 Washer 3 2.00 6.02",  # two cents off: not an item row
        "A-4 Pin 2 4.00",  # only two numbers
    ]
    page = build(["\n".join(rows)], "text_layer").pages[0]
    assert [r.tokens[0] for r in page.item_rows] == ["A-1", "A-2"]
    assert page.item_rows[1].triple == (Decimal(3), Decimal("2.00"), Decimal("6.00"))


def test_locale_is_the_one_under_which_more_rows_multiply_out() -> None:
    # A row with a whole quantity multiplies out under either reading (both the
    # price and the amount scale by the same factor); a fractional quantity or a
    # thousands separator in it tells the formats apart.
    eu_rows = [
        "X-1 Panel 3 1.234,50 3.703,50",
        "X-2 Tape 2,5 4,00 10,00",
        "X-3 Clip 1.250 2,00 2.500,00",
    ]
    eu = build(["\n".join(eu_rows)], "text_layer")
    assert eu.locale == "eu"
    assert [r.triple for r in eu.pages[0].item_rows] == [
        (Decimal(3), Decimal("1234.50"), Decimal("3703.50")),
        (Decimal("2.5"), Decimal("4.00"), Decimal("10.00")),
        (Decimal(1250), Decimal("2.00"), Decimal("2500.00")),
    ]
    us_rows = ["X-1 Panel 3 1,234.50 3,703.50", "X-2 Tape 2.5 4.00 10.00"]
    us = build(["\n".join(us_rows)], "text_layer")
    assert us.locale == "us" and len(us.pages[0].item_rows) == 2


def test_locale_is_us_on_a_tie() -> None:
    assert build(["no numbers here"], "text_layer").locale == "us"


def test_pages_rows_and_tokens() -> None:
    ev = build(["First row\n\n  \nsku-9 Thing 1 5.00 5.00;", "Page two"], "text_layer")
    assert ev.source == "text_layer"
    assert [p.number for p in ev.pages] == [1, 2]
    assert [r.text for r in ev.pages[0].rows] == ["First row", "sku-9 Thing 1 5.00 5.00;"]
    assert ev.pages[0].rows[1].token_set == {"SKU-9", "THING", "1", "5.00"}
    assert ev.pages[1].token_set == {"PAGE", "TWO"}
    assert build([""], "text_layer").pages[0].token_set == set()


def test_page_lookup() -> None:
    ev = build(["one", "two", "three"], "text_layer")
    assert [p.number for p in ev.page(2)] == [2]
    assert ev.page(4) == []
    assert ev.page(None) == ev.pages


# --------------------------------------------------------------------------- #
# Text layer of a PDF
# --------------------------------------------------------------------------- #

ROWS = [
    "ACME Ltd Invoice INV-1",
    "SKU Description Qty Price Amount",
    "SKU-001 Hex bolt M8 4 2.50 10.00",
    "SKU-002 Washer M8 100 0.05 5.00",
    "Subtotal 15.00",
]


def test_text_layer_of_a_generated_pdf() -> None:
    texts = pdf_text_layer(make_pdf([ROWS, ["Page 2 of 2", "SKU-003 Nut M8 10 0.30 3.00"]]))
    assert [t.splitlines() for t in texts] == [ROWS, ["Page 2 of 2", "SKU-003 Nut M8 10 0.30 3.00"]]
    ev = build(texts, "text_layer")
    assert [[r.tokens[0] for r in p.item_rows] for p in ev.pages] == [
        ["SKU-001", "SKU-002"],
        ["SKU-003"],
    ]


def test_scanned_page_has_empty_text() -> None:
    assert pdf_text_layer(make_pdf([[], ROWS])) == ["", "\r\n".join(ROWS)]


def test_has_text_layer_needs_text_on_every_page() -> None:
    assert has_text_layer(pdf_text_layer(make_pdf([ROWS, ROWS])))
    assert not has_text_layer(pdf_text_layer(make_pdf([ROWS, []])))  # one page without text
    assert not has_text_layer([])


def test_has_text_layer_threshold() -> None:
    assert has_text_layer(["x" * 40])
    assert not has_text_layer(["  " + "x" * 39 + "\n"])  # whitespace does not count
    assert has_text_layer(["x" * 10], min_chars=10)
