"""The result of a document as the API shows it."""

import json
import math
from dataclasses import asdict, dataclass, field, fields
from decimal import Decimal
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from invoice_recognition.api import messages
from invoice_recognition.api.views import MAX_DEPTH, LineView, ResultView, result_view
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.config import FAST, OTHER, Config
from invoice_recognition.recognition.pipeline import stronger_model_note
from invoice_recognition.recognition.results import Attempt, PageResult, Result
from invoice_recognition.shapes import Header, JSONObject, Line

import drawn
from fakes import Answer, Server, header_for, line, run
from samples import saved_layout, truth_doc


def attempt(lines: int = 1, issues: int = 0, **kw: object) -> Attempt:
    fields: JSONObject = {
        "model": "m",
        "duration_s": 1.5,
        "cost_usd": 0.001,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "lines": lines,
        "issues": issues,
    }
    return Attempt(**(fields | kw))


def result(
    pages: list[PageResult],
    *,
    header: Header | None = None,
    issues: list[Issue] | None = None,
    notes: list[JSONObject] | None = None,
    **kw: object,
) -> Result:
    lines = [item for p in pages for item in p.lines]
    fields: JSONObject = {
        "status": "needs_review",
        "header": header_for(lines) if header is None else header,
        "lines": lines,
        "checkpoints": [],
        "issues": issues or [],
        "pages": pages,
        "evidence_source": None,
        "duration_s": 3.25,
        "calls": 3,
        "cost_usd": 0.003,
        "prompt_tokens": 300,
        "completion_tokens": 150,
        "notes": notes or [],
    }
    return Result(**(fields | kw))


def on(page: int, *nos: int) -> list[Line]:
    return [line(n) | {"page": page} for n in nos]


def dumped(view: ResultView) -> JSONObject:
    """What a client gets: the view as JSON text, parsed."""
    out: JSONObject = json.loads(view.model_dump_json())
    return out


NO_FRAME: JSONObject = {"width": None, "height": None, "text_layer": False}

TWO_PAGES = [
    PageResult(1, on(1, 1, 2), attempts=[attempt(2)]),
    PageResult(2, on(2, 3, 4, 5), attempts=[attempt(3)]),
]


def test_lines_get_their_index_in_the_document_and_issues_point_at_it() -> None:
    issue = Issue("line_math", "2 x 10 is not 21", line=3, page=2, line_no=4)
    out = dumped(result_view(result(TWO_PAGES, issues=[issue])))
    assert [(item["index"], item["line_no"], item["page"]) for item in out["lines"]] == [
        (0, 1, 1),
        (1, 2, 1),
        (2, 3, 2),
        (3, 4, 2),
        (4, 5, 2),
    ]
    assert out["lines"][3] == {"index": 3} | line(4) | {"page": 2, "boxes": None}
    [view] = out["issues"]
    assert view == {
        "id": 0,
        "code": "line_math",
        "severity": "error",
        "message": "2 x 10 is not 21",
        "line_index": 3,
        "line_no": 4,
        "page": 2,
        "fields": [],
        "suggestions": [],
        "follows_from": [],
    }
    assert out["lines"][view["line_index"]]["line_no"] == view["line_no"]


def test_issue_ids_are_positions_and_unknowns_are_null() -> None:
    issues = [
        Issue("doc_sums", "the lines add up to 100, the subtotal is 90"),
        Issue("tax", "tax does not match", severity="warning"),
    ]
    out = dumped(result_view(result(TWO_PAGES, issues=issues)))["issues"]
    assert [i["id"] for i in out] == [0, 1]
    assert out[1]["severity"] == "warning"
    for i in out:
        assert (i["line_index"], i["line_no"], i["page"]) == (None, None, None)
        assert (i["fields"], i["suggestions"], i["follows_from"]) == ([], [], [])


@dataclass
class IssueWithFields(Issue):
    """An issue as the flow will carry it once issues name their fields: the view reads
    these keys from the issue's JSON, whether the flow has them yet or not."""

    fields: list[str] = field(default_factory=list)
    suggestions: list[JSONObject] = field(default_factory=list)
    follows_from: list[int] = field(default_factory=list)


def test_fields_suggestions_and_the_issues_followed_from_pass_through() -> None:
    first = IssueWithFields(
        "line_math",
        "2 x 10 is not 21",
        line=0,
        page=1,
        line_no=1,
        fields=["line.amount", "line.quantity"],
        suggestions=[
            {
                "set": {"line.amount": 20.0},
                "source": "arithmetic",
                "agrees_with": "header.subtotal",
            },
            {"set": {"line.quantity": 2.1}, "source": "arithmetic"},
        ],
    )
    second = IssueWithFields(
        "second_reading",
        "the second reader read SKU-002 as SKU-0O2",
        line=1,
        page=1,
        severity="warning",
        fields=["line.sku"],
        suggestions=[{"set": {"line.sku": "SKU-0O2"}, "source": "second_reading"}],
        follows_from=[0],
    )
    out = dumped(result_view(result(TWO_PAGES, issues=[first, second])))["issues"]
    for view, issue in zip(out, (first, second), strict=True):
        assert view["fields"] == issue.fields
        assert view["suggestions"] == issue.suggestions
        assert view["follows_from"] == issue.follows_from
    assert (out[1]["code"], out[1]["severity"]) == ("second_reading", "warning")


def test_an_issue_message_comes_from_the_wording_table_when_one_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        messages.ISSUE_TEXTS, "tax", lambda i, line: f"Tax on page {i.page} is off."
    )
    issues = [Issue("tax", "raw", page=2), Issue("doc_sums", "kept as the flow wrote it")]
    out = dumped(result_view(result(TWO_PAGES, issues=issues)))["issues"]
    assert [i["message"] for i in out] == ["Tax on page 2 is off.", "kept as the flow wrote it"]


def test_notes_pass_through_with_line_renamed_and_page_always_present() -> None:
    notes: list[JSONObject] = [
        {
            "page": 2,
            "kind": "joined_across_pages",
            "text": "Screws, M8 (continued)",
            "source": "carry_in",
            "to_page": 1,
            "line": 1,
            "line_no": 2,
            "sku": "SKU-002",
        },
        {"page": None, "kind": "carry_join_failed", "text": "KeyError: 3", "source": "carry_in"},
        {"kind": "stray_words", "text": "Thank you", "row": 7},
        {
            "page": None,
            "kind": "second_reading_missing",
            "text": "Pages 2 and 3 were not read a second time.",
            "pages": [2, 3],
        },
    ]
    out = dumped(result_view(result(TWO_PAGES, notes=notes)))["notes"]
    assert out[0] == {k: v for k, v in notes[0].items() if k != "line"} | {"line_index": 1}
    assert out[1] == notes[1]
    assert out[2] == {"page": None, "kind": "stray_words", "text": "Thank you", "row": 7}
    assert out[3] == notes[3]


def test_the_stronger_model_note_passes_through() -> None:
    page = PageResult(1, on(1, 1), attempts=[attempt(), attempt()], stronger_model=True)
    res = result([page], notes=[stronger_model_note(page)])
    [note] = dumped(result_view(res))["notes"]
    assert note["kind"] == "stronger_model_not_called" and note["page"] == 1
    assert note["text"].startswith("Page 1 still has problems after 2 reads.")


def test_a_repaired_quantity_gets_one_note() -> None:
    fixed = line(2, qty=8200, price=0.5) | {"page": 1, "repaired": "quantity_x1000"}
    pages = [PageResult(1, [*on(1, 1), fixed], repaired=1), PageResult(2, on(2, 3))]
    out = dumped(result_view(result(pages)))
    assert out["lines"][1]["repaired"] == "quantity_x1000"  # an extra key, passed through
    assert out["lines"][1]["quantity"] == 8200
    assert out["notes"] == [
        {
            "kind": "quantity_repaired",
            "page": 1,
            "text": "The quantity was read as 8.200 and corrected to 8200: the point in it is a "
            "thousands separator, as the amount of the line shows.",
            "line_index": 1,
            "line_no": 2,
        }
    ]
    assert [p["repaired_count"] for p in out["pages"]] == [1, 0]


def test_values_as_read_are_never_coerced() -> None:
    odd: Line = line(1) | {
        "quantity": "8.200",
        "unit_price": [1, 2],
        "amount": {"value": 10},
        "sku": 12345,
        "tax_rate": True,
        "unit": None,
        "colour": "red",
        "page": 1,
    }
    header: Header = {"total": "1.440,00", "currency": 978, "tax_breakdown": "none", "x": [1]}
    out = dumped(result_view(result([PageResult(1, [odd])], header=header)))
    assert out["lines"][0] == {"index": 0} | odd | {"boxes": None}
    for key, value in header.items():
        assert out["header"][key] == value
    assert out["header"]["supplier_name"] is None


def test_an_index_key_from_the_model_does_not_replace_the_position() -> None:
    lines = [line(1) | {"page": 1, "index": 99}]
    out = dumped(result_view(result([PageResult(1, lines)])))
    assert out["lines"][0]["index"] == 0


def test_nan_and_infinity_come_out_as_null() -> None:
    lines = [
        line(1) | {"page": 1, "quantity": math.nan, "amount": -math.inf, "x": [math.inf]},
    ]
    header: Header = {"total": math.inf, "tax_breakdown": [{"rate": math.nan}]}
    view = result_view(result([PageResult(1, lines)], header=header))
    for text in (view.model_dump_json(), json.dumps(view.model_dump(mode="json"))):
        out = json.loads(text)  # strict JSON: NaN would not parse back as null
        assert "NaN" not in text and "Infinity" not in text
        assert out["lines"][0]["quantity"] is None and out["lines"][0]["amount"] is None
        assert out["lines"][0]["x"] == [None]
        assert out["header"]["total"] is None
        assert out["header"]["tax_breakdown"] == [{"rate": None}]


def test_pages_checkpoints_and_details() -> None:
    pages = [
        PageResult(
            1,
            on(1, 1, 2),
            totals=[{"label": "Carried forward", "amount": "40,00", "after_line": 2, "page": 1}],
            attempts=[attempt(2), attempt(2, 1, cost_usd=0.0, cost_known=False, error="cut")],
            method="model",
        ),
        PageResult(2, method="skipped"),
    ]
    res = result(
        pages,
        checkpoints=[{"label": "Carried forward", "amount": "40,00", "after_line": 2, "page": 1}],
        header_attempts=[attempt(0)],
        layout={"status": "off", "reason": "no text layer"},
        cost_unknown_calls=1,
        quality={"verdict": "ok", "pages": [{"page": 2, "verdict": "empty"}]},
    )
    out = dumped(result_view(res))
    assert out["checkpoints"] == res.checkpoints
    assert out["pages"] == [
        {"page": 1, "method": "model", "line_count": 2, "repaired_count": 0} | NO_FRAME,
        {"page": 2, "method": "skipped", "line_count": 0, "repaired_count": 0} | NO_FRAME,
    ]
    details = out["details"]
    assert {k: v for k, v in details.items() if k != "attempts"} == {
        "method": "model",
        "layout": {"status": "off", "reason": "no text layer"},
        "evidence_source": None,
        "duration_s": 3.25,
        "cost_usd": 0.002,  # the attempts' own: the header's and the first page read's
        "cost_unknown_calls": 1,
        "calls": 3,
        "prompt_tokens": 300,
        "completion_tokens": 150,
        "quality": res.quality,
    }
    assert [(a["what"], a["page"]) for a in details["attempts"]] == [
        ("header", None),
        ("page", 1),
        ("page", 1),
    ]
    assert details["attempts"][2] == {
        "what": "page",
        "page": 1,
        "model": "m",
        "duration_s": 1.5,
        "cost_usd": None,  # not reported: not the 0.0 the flow keeps in its place
        "cost_known": False,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "line_count": 2,
        "issue_count": 1,
        "error": "cut",
    }


class ReadTwice(Result):
    """A result whose page 1 was also read by a second model, as the flow will say it in
    the page's JSON (the key is there only when such a request was made)."""

    def to_json(self) -> JSONObject:
        out = super().to_json()
        out["pages"][0]["second_reading"] = asdict(
            attempt(2, 1, model="other", cost_usd=0.004, prompt_tokens=7, completion_tokens=3)
        )
        return out


def test_a_second_reading_is_an_attempt_and_counts_in_the_totals() -> None:
    plain = result(TWO_PAGES, header_attempts=[attempt(0)])
    twice = ReadTwice(**{f.name: getattr(plain, f.name) for f in fields(Result)})
    out = dumped(result_view(twice))["details"]
    assert [(a["what"], a["page"], a["model"]) for a in out["attempts"]] == [
        ("header", None, "m"),
        ("page", 1, "m"),
        ("second_reading", 1, "other"),
        ("page", 2, "m"),
    ]
    assert out["calls"] == len(out["attempts"]) == 4
    assert out["cost_usd"] == pytest.approx(0.007)
    assert (out["prompt_tokens"], out["completion_tokens"]) == (307, 153)
    # Without the key, no such attempt: the plain result lists three.
    assert dumped(result_view(plain))["details"]["calls"] == 3


def nested(depth: int) -> object:
    value: object = "Hex bolt M8"
    for _ in range(depth):
        value = [value]
    return value


def test_odd_values_never_throw_the_document_away() -> None:
    odd: dict[Any, Any] = line(2) | {"page": 1, "quantity": object()}
    odd[3] = "a key that is a number"
    lines: list[Line] = [
        line(1) | {"page": 1, "description": nested(260)},
        odd,
        line(3) | {"page": 1, "unit_price": Decimal("6.18"), "amount": Decimal(12)},
    ]
    res = result(
        [PageResult(1, lines, attempts=[attempt(3, prompt_tokens=1234.0)])],
        quality={"verdict": "ok", "measures": {"boxes": np.int64(42), "glare": np.float32(0.5)}},
        notes=[{"page": 1, "kind": "stray_words", "text": "x", "row": np.int64(7)}],
    )
    out = dumped(result_view(res))
    assert [item["index"] for item in out["lines"]] == [0, 1, 2]
    deep, levels = out["lines"][0]["description"], 0
    while isinstance(deep, list):
        deep, levels = deep[0], levels + 1
    assert levels < MAX_DEPTH  # the rest is sent as its JSON text
    assert json.loads(deep) == nested(260 - levels)
    assert out["lines"][1]["quantity"].startswith("<object object at ")
    assert out["lines"][1]["3"] == "a key that is a number"
    assert (out["lines"][2]["unit_price"], out["lines"][2]["amount"]) == (6.18, 12)
    assert out["details"]["quality"]["measures"] == {"boxes": 42, "glare": 0.5}
    assert out["notes"][0]["row"] == 7
    assert out["details"]["attempts"][0]["prompt_tokens"] == 1234
    assert out["details"]["prompt_tokens"] == 1234


def test_a_token_count_sent_as_a_fraction_is_rounded() -> None:
    page = PageResult(1, on(1, 1), attempts=[attempt(1, completion_tokens=49.6)])
    out = dumped(result_view(result([page])))["details"]
    assert out["attempts"][0]["completion_tokens"] == 50 and out["completion_tokens"] == 50


def test_numbers_reported_as_text_are_read_and_anything_else_is_not_reported() -> None:
    sent_as_text = attempt(1, prompt_tokens="12", completion_tokens="49.6", cost_usd="0.001")
    odd = attempt(
        1, prompt_tokens={"n": 1}, completion_tokens=True, cost_usd="n/a", duration_s=None
    )
    page = PageResult(1, on(1, 1), attempts=[sent_as_text, odd])
    out = dumped(result_view(result([page])))
    assert len(out["lines"]) == 1
    first, second = out["details"]["attempts"]
    assert (first["prompt_tokens"], first["completion_tokens"]) == (12, 50)
    assert (first["cost_usd"], first["cost_known"]) == (0.001, True)
    assert (second["prompt_tokens"], second["completion_tokens"]) == (0, 0)
    assert (second["cost_usd"], second["cost_known"], second["duration_s"]) == (None, False, 0.0)
    details = out["details"]
    assert (details["prompt_tokens"], details["completion_tokens"]) == (12, 50)
    assert (details["cost_usd"], details["cost_unknown_calls"]) == (0.001, 1)


def test_text_that_cannot_be_utf8_is_made_safe_and_nothing_is_lost() -> None:
    lone = "\ud800"  # a lone surrogate: json.loads gives one for "\ud800" in an answer
    odd: dict[str, Any] = line(1) | {"page": 1, "description": f"Bolt {lone}"}
    odd[f"colour{lone}"] = f"red{lone}"
    res = result(
        [PageResult(1, [odd, line(2) | {"page": 1}], attempts=[attempt(2, error=lone)])],
        header={"supplier_name": f"ACME {lone}", "currency": "EUR"},
        issues=[Issue("tax", f"tax {lone} is off", page=1)],
        notes=[{"page": 1, "kind": "stray_words", "text": lone, lone: [lone]}],
        checkpoints=[{"label": lone, "amount": 1, "after_line": 1, "page": 1}],
        layout={"status": "off", "reason": lone},
        quality={"verdict": "ok", lone: lone},
    )
    text = result_view(res).model_dump_json()
    out = json.loads(text)
    assert [item["index"] for item in out["lines"]] == [0, 1]
    assert out["lines"][0]["description"] == "Bolt ?"
    assert out["lines"][0]["colour?"] == "red?"
    assert out["header"]["supplier_name"] == "ACME ?"
    assert out["issues"][0]["message"] == "tax ? is off"
    assert out["notes"][0]["text"] == "?" and out["notes"][0]["?"] == ["?"]
    assert out["checkpoints"][0]["label"] == "?"
    assert out["details"]["layout"]["reason"] == "?"
    assert out["details"]["quality"]["?"] == "?"
    assert out["details"]["attempts"][0]["error"] == "?"


def test_extra_keys_are_validated_as_json() -> None:
    with pytest.raises(ValidationError):
        LineView.model_validate({"index": 0, "colour": object()})
    assert LineView.model_validate({"index": 0, "colour": ["red", 1]}).model_extra == {
        "colour": ["red", 1]
    }


def test_the_schema_allows_extra_json_keys_where_they_pass_through() -> None:
    json_value = {"$ref": "#/$defs/JsonValue"}
    assert LineView.model_json_schema()["additionalProperties"] == json_value
    schema = ResultView.model_json_schema()
    assert schema["$defs"]["NoteView"]["additionalProperties"] == json_value
    assert schema["$defs"]["HeaderView"]["additionalProperties"] == json_value
    assert "additionalProperties" not in schema["$defs"]["PageView"]


def test_every_field_sent_is_required_in_the_serialisation_schema() -> None:
    defs = ResultView.model_json_schema(mode="serialization")["$defs"]
    assert set(defs["LineView"]["required"]) == {
        "index",
        "page",
        "line_no",
        "sku",
        "description",
        "quantity",
        "unit",
        "unit_price",
        "tax_rate",
        "amount",
        "boxes",
    }
    assert set(defs["IssueView"]["required"]) == {
        "id",
        "code",
        "severity",
        "message",
        "line_index",
        "line_no",
        "page",
        "fields",
        "suggestions",
        "follows_from",
    }
    assert set(defs["NoteView"]["required"]) == {"kind", "page", "text"}
    assert len(defs["HeaderView"]["required"]) == 11


# --------------------------------------------------------------------------- #
# Results the flow really returns
# --------------------------------------------------------------------------- #


def test_a_document_read_by_layout() -> None:
    data, _, header, pages = truth_doc("s_classic")
    answers: dict[int, list[Answer]] = {p: [rows] for p, rows in pages.items()}
    res = run(data, "application/pdf", Server(header, answers, saved_layout("s_classic")))
    out = dumped(result_view(res))
    assert out["details"]["method"] == "layout"
    assert out["details"]["layout"]["status"] == "confirmed"
    assert [a["what"] for a in out["details"]["attempts"]] == ["header", "layout", "page"]
    assert len(out["lines"]) == len(res.lines) > 0


def test_a_photo_with_its_quality_measures() -> None:
    rows = [line(1), line(2)]
    res = run(drawn.jpeg(drawn.photo()), "image/jpeg", Server(header_for(rows), {1: [rows]}))
    out = dumped(result_view(res))
    assert out["details"]["quality"]["verdict"] in ("ok", "warn")
    assert out["details"]["quality"]["measures"]


def test_a_scan_with_a_blank_page() -> None:
    rows = [line(1)]
    res = run(
        drawn.scanned("good", "blank"), "application/pdf", Server(header_for(rows), {1: [rows]})
    )
    out = dumped(result_view(res))
    assert [p["method"] for p in out["pages"]] == ["model", "skipped"]
    assert [n["kind"] for n in out["notes"]] == ["page_without_text"]
    assert out["details"]["quality"]["pages"][1]["verdict"] == "empty"


def test_a_file_refused_before_any_request() -> None:
    res = run(b"%PDF-1.7 broken", "application/pdf", Server({}, {}))
    out = dumped(result_view(res))
    assert out["details"]["layout"] is None and out["details"]["attempts"] == []
    assert [i["code"] for i in out["issues"]] == ["unreadable_file"]


def test_a_scan_read_twice_by_the_flow() -> None:
    """A real result: two pages, each read a second time by another model, which reads a
    SKU of page 2 differently."""
    first: dict[int, list[Answer]] = {1: [[line(1)]], 2: [[line(2)]]}
    second: dict[int, list[Answer]] = {1: [[line(1)]], 2: [[line(2) | {"sku": "SKU-0O2"}]]}
    server = Server(header_for([line(1), line(2)]), first, models={OTHER: Server({}, second)})
    res = run(drawn.scan(2), "application/pdf", server, Config(second_reading_model=OTHER))
    out = dumped(result_view(res))

    details = out["details"]
    assert [(a["what"], a["page"], a["model"]) for a in details["attempts"]] == [
        ("header", None, FAST),
        ("page", 1, FAST),
        ("second_reading", 1, OTHER),
        ("page", 2, FAST),
        ("second_reading", 2, OTHER),
    ]
    assert [a["issue_count"] for a in details["attempts"] if a["what"] == "second_reading"] == [
        0,
        1,
    ]
    # The view's totals are those of the attempts it lists, and the flow's agree with them.
    assert details["calls"] == res.calls == 5
    assert details["cost_usd"] == pytest.approx(res.cost_usd) and res.cost_usd == pytest.approx(
        0.005
    )
    assert (details["prompt_tokens"], details["completion_tokens"]) == (500, 250)
    (issue,) = out["issues"]
    assert (issue["code"], issue["severity"], issue["line_index"], issue["fields"]) == (
        "second_reading",
        "warning",
        1,
        ["line.sku"],
    )
    assert issue["suggestions"] == [{"set": {"line.sku": "SKU-0O2"}, "source": "second_reading"}]
