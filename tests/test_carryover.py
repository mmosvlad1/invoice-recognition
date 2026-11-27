"""carryover.py on its own: the prompt with the "carry_in" field and the join by code."""

import json

from invoice_recognition.recognition import carryover as co
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition.prompts import EXTRACT_PAGE
from invoice_recognition.shapes import JSONObject, Line

from fakes import line

# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #


def test_prompt_is_the_page_prompt_plus_the_field() -> None:
    prompt = co.carry_prompt(2, 5)
    assert "page 2 of 5" in prompt and '"carry_in": str,\n  "rows": [' in prompt
    assert co.CARRY_RULE in prompt
    # the rules for line items are the flow's, to the letter
    assert prompt.endswith(EXTRACT_PAGE[EXTRACT_PAGE.index("Rules for line items:") :])
    # every placeholder is filled, braces are single
    assert "{page}" not in prompt and "{{" not in prompt and "}}" not in prompt
    # the old sentence would contradict the new shape
    assert 'return {"rows": []}' not in prompt and '"rows" is [].' in prompt
    # Nothing else is changed: undoing the three edits gives the page prompt back.
    back = co.EXTRACT_PAGE_CARRY
    for old, new in co._EDITS:
        assert back.count(new) == 1
        back = back.replace(new, old)
    assert back == EXTRACT_PAGE
    for said in (
        "Do NOT add it to the description of any line item on this page",
        "section headings",
        "table header",
        "previous page",
    ):
        assert said in co.CARRY_RULE


def test_carry_text_takes_text_only() -> None:
    assert co.carry_text("  compliant,\n country of origin  ") == "compliant, country of origin"
    assert co.carry_text(["first line", "second line"]) == "first line second line"
    for value in (None, 7, True, {"text": "x"}, [], ["a", 1], ""):
        assert co.carry_text(value) == ""


# --------------------------------------------------------------------------- #
# The rows of a page
# --------------------------------------------------------------------------- #


def item(no: int, desc: str | None = "Bolt") -> Line:
    return line(no) | {"description": desc}


TOTAL: JSONObject = {"total_label": "Page total", "amount": 6.0}


def test_line_item_rule_is_the_one_of_the_flow() -> None:
    rows = [
        item(1),
        TOTAL,
        {"description": "stray text"},
        {"sku": None, "amount": 5.0},
        {"sku": "X", "amount": None},
        "junk",
        None,
        {"total_label": "Subtotal", "sku": "X", "amount": 1.0},
    ]
    lines, totals = pl.rows_to_page(rows, 1)
    assert sum(co.is_line_item(r) for r in rows) == len(lines) == 3
    assert sum(co.is_total(r) for r in rows) == len(totals) == 2


def test_flow_rows_puts_control_totals_where_they_were_printed() -> None:
    a, b = line(1), line(2)
    totals = [
        {"label": "Carried forward", "amount": 5.0, "after_line": 0},
        {"label": "Subtotal", "amount": 45.0, "after_line": 2},
        {"label": "No position", "amount": 1.0, "after_line": None},
        {"label": "Far off", "amount": 2.0, "after_line": 99},
    ]
    rows = co.flow_rows([a, b], totals)
    assert rows == [
        {"total_label": "Carried forward", "amount": 5.0},
        a,
        b,
        {"total_label": "Subtotal", "amount": 45.0},
        {"total_label": "No position", "amount": 1.0},  # at the end of the page
        {"total_label": "Far off", "amount": 2.0},
    ]
    assert rows[1] is a and co.is_total(rows[-1]) and co.is_line_item(rows[2])
    assert co.flow_rows([], []) == [] and co.flow_rows([a], []) == [a]


# --------------------------------------------------------------------------- #
# Joining
# --------------------------------------------------------------------------- #


def test_join_appends_the_tail_to_the_last_row_of_the_previous_page() -> None:
    pages: list[JSONObject] = [
        {
            "page": 2,
            "rows": [item(3), item(4)],
            "carry_in": " compliant,  country of origin DE ",
        },
        {"page": 1, "rows": [item(1), item(2, "LED panel, RoHS ")], "carry_in": ""},
    ]
    before = json.dumps(pages)
    joined, notes = co.join_carry(pages)
    assert json.dumps(pages) == before  # the input is not changed
    assert [p["page"] for p in joined] == [1, 2]
    assert joined[0]["rows"][1]["description"] == "LED panel, RoHS compliant, country of origin DE"
    # nothing else is touched
    assert joined[0]["rows"][0] == item(1) and joined[1]["rows"] == [item(3), item(4)]
    assert notes == [
        {
            "page": 2,
            "kind": "joined_across_pages",
            "text": "compliant, country of origin DE",
            "reason": "",
            "to_page": 1,
            "row": 1,
            "line_no": 2,
            "sku": "SKU-002",
        }
    ]
    again, more = co.join_carry(joined)  # a joined page is not joined twice
    assert more == []
    assert again[0]["rows"][1]["description"] == "LED panel, RoHS compliant, country of origin DE"


def test_join_gives_a_row_without_a_description_the_tail_as_its_description() -> None:
    joined, notes = co.join_carry(
        [
            {"page": 1, "rows": [item(1, None)], "carry_in": ""},
            {"page": 2, "rows": [], "carry_in": "the whole text"},
        ]
    )
    assert joined[0]["rows"][0]["description"] == "the whole text" and notes[0]["kind"] == co.JOINED


def test_tail_is_left_alone_when_the_previous_page_ends_with_a_control_total() -> None:
    pages: list[JSONObject] = [
        {"page": 1, "rows": [item(1), item(2), TOTAL], "carry_in": ""},
        {"page": 2, "rows": [item(3)], "carry_in": "delivered on euro pallet"},
    ]
    joined, notes = co.join_carry(pages)
    assert [p["rows"] for p in joined] == [p["rows"] for p in pages]
    assert not joined[1].get("carry_joined")
    assert notes == [
        {
            "page": 2,
            "kind": "carry_without_owner",
            "text": "delivered on euro pallet",
            "reason": "Page 1 ends with a control total.",
        }
    ]


def test_tails_without_an_owner() -> None:
    def reasons(pages: list[JSONObject]) -> list[tuple[int, str, str]]:
        return [(n["page"], n["kind"], n["reason"]) for n in co.join_carry(pages)[1]]

    first: list[JSONObject] = [
        {"page": 1, "rows": [item(1)], "carry_in": "Section A"},
        {"page": 2, "rows": [item(2)], "carry_in": ""},
    ]
    assert reasons(first) == [(1, co.ORPHAN, "This is the first page of the document.")]
    gap: list[JSONObject] = [
        {"page": 1, "rows": [item(1)], "carry_in": ""},
        {"page": 3, "rows": [item(9)], "carry_in": "tail"},  # page 2 was not read
    ]
    assert reasons(gap) == [(3, co.ORPHAN, "Page 2 was not read.")]
    unread: list[JSONObject] = [
        {"page": 1, "rows": [item(1)], "carry_in": "", "unread": True},
        {"page": 2, "rows": [item(2)], "carry_in": "tail"},
    ]
    assert reasons(unread) == [(2, co.ORPHAN, "Page 1 could not be read in full.")]
    skipped: list[JSONObject] = [
        {"page": 1, "rows": [], "carry_in": "", "skipped": True},
        {"page": 2, "rows": [item(2)], "carry_in": "tail"},
    ]
    assert reasons(skipped) == [(2, co.ORPHAN, "Page 1 was not sent to the model.")]
    empty: list[JSONObject] = [
        {"page": 1, "rows": [], "carry_in": ""},
        {"page": 2, "rows": [item(2)], "carry_in": "tail"},
    ]
    assert reasons(empty) == [(2, co.ORPHAN, "Page 1 has no rows.")]
    # the flow drops a row with neither sku nor amount: it cannot own a tail
    stray: list[JSONObject] = [
        {"page": 1, "rows": [item(1), {"description": "no sku, no amount"}], "carry_in": ""},
        {"page": 2, "rows": [item(2)], "carry_in": "tail"},
    ]
    assert reasons(stray) == [(2, co.ORPHAN, "The last row of page 1 is not a line item.")]


def test_tail_already_joined_by_code_is_not_appended_twice() -> None:
    pages: list[JSONObject] = [
        {"page": 1, "rows": [item(1, "Bolt M8 Zinc  plated")], "joined_by_code": True},
        {"page": 2, "rows": [item(2)], "carry_in": "zinc plated"},
    ]
    joined, notes = co.join_carry(pages)
    assert joined[0]["rows"][0]["description"] == "Bolt M8 Zinc  plated"
    assert [(n["kind"], n["to_page"], n["row"]) for n in notes] == [(co.PRESENT, 1, 0)]


def test_empty_field_changes_nothing() -> None:
    for empty in ("", "   \n ", None):
        pages: list[JSONObject] = [
            {"page": 1, "rows": [item(1)], "carry_in": ""},
            {"page": 2, "rows": [item(2)], "carry_in": empty},
        ]
        joined, notes = co.join_carry(pages)
        assert notes == [] and [p["rows"] for p in joined] == [[item(1)], [item(2)]]
    # answers in the old shape
    joined, notes = co.join_carry([{"page": 1, "rows": [item(1)]}, {"page": 2, "rows": [item(2)]}])
    assert notes == [] and joined[0]["rows"] == [item(1)]


def test_every_boundary_is_joined_on_its_own() -> None:
    pages: list[JSONObject] = [
        {"page": 1, "rows": [item(1, "a")], "carry_in": ""},
        {"page": 2, "rows": [item(2, "b")], "carry_in": "A"},
        {"page": 3, "rows": [item(3, "c"), TOTAL], "carry_in": "B"},
        {"page": 4, "rows": [item(4, "d")], "carry_in": "C"},
    ]
    joined, notes = co.join_carry(pages)
    assert [p["rows"][0]["description"] for p in joined] == ["a A", "b B", "c", "d"]
    assert [(n["page"], n["kind"]) for n in notes] == [
        (2, co.JOINED),
        (3, co.JOINED),
        (4, co.ORPHAN),
    ]
