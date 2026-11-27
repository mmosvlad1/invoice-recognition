"""The "carry_in" field in the flow: which request a page gets, which attempt the tail is
taken from, and what the join does. Fake OpenRouter, no network, no key."""

import asyncio
import hashlib
import json
import re

import httpx
import pytest

from invoice_recognition.recognition import carryover as co
from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.prompts import EXTRACT_PAGE
from invoice_recognition.recognition.results import PageResult, Result
from invoice_recognition.shapes import JSONObject, Line

from drawn import jpeg, photo, scan
from fakes import Answer, Server, header_for, line, recording, run, sse
from samples import exact, saved_layout, saved_scan_pages, truth, truth_doc


def prompts(bodies: list[JSONObject]) -> dict[int, list[str]]:
    """Page number -> the texts of the page requests sent for it, in order."""
    out: dict[int, list[str]] = {}
    for b in bodies:
        text = b["messages"][0]["content"][0]["text"]
        if text.startswith("You read ONE page"):
            m = re.search(r"page (\d+) of \d+", text)
            assert m is not None
            out.setdefault(int(m.group(1)), []).append(text)
    return out


def page(rows: list[Line], carry: object = "") -> JSONObject:
    """An answer to the page prompt with the field."""
    return {"carry_in": carry, "rows": rows}


def descr(n: int, text: str) -> Line:
    return line(n) | {"description": text}


def lose_a_line(monkeypatch: pytest.MonkeyPatch, page: int) -> None:
    """Make the code parser miss one line item of `page`, as an irregular row would."""
    real = lay.parse_document

    def lossy(pages: list[list[lay.TextLine]], layout: lay.Layout) -> lay.Parsed:
        parsed = real(pages, layout)
        victim = next(i for i, item in enumerate(parsed.lines) if item["page"] == page)
        del parsed.lines[victim]
        return parsed

    monkeypatch.setattr(lay, "parse_document", lossy)


TAIL = "compliant, country of origin DE"


# --------------------------------------------------------------------------- #
# A scan of several pages
# --------------------------------------------------------------------------- #


def test_tail_is_joined_to_the_last_row_of_the_previous_page() -> None:
    p1 = [line(1), descr(2, "LED panel, box of 100, RoHS")]
    p2 = [line(3), line(4)]
    p3 = [line(5)]
    header = header_for(p1 + p2 + p3)
    handler, bodies = recording(Server(header, {1: [page(p1)], 2: [page(p2, TAIL)], 3: [page(p3)]}))
    res = run(scan(3), "application/pdf", handler)

    # every page is asked with the field
    assert prompts(bodies) == {p: [co.carry_prompt(p, 3)] for p in (1, 2, 3)}
    assert [item["description"] for item in res.lines] == [
        "Item 1",
        f"LED panel, box of 100, RoHS {TAIL}",
        "Item 3",
        "Item 4",
        "Item 5",
    ]
    assert res.notes == [
        {
            "page": 2,
            "kind": "joined_across_pages",
            "text": TAIL,
            "source": "carry_in",
            "to_page": 1,
            "line": 1,
            "line_no": 2,
            "sku": "SKU-002",
        }
    ]
    # a note is not an issue, a join is not a request
    assert res.status == "ok" and res.issues == [] and res.calls == 4
    assert res.pages[1].carry_in == TAIL and res.pages[0].carry_in == ""
    out = res.to_json()
    assert out["notes"] == res.notes and out["lines"][1]["description"].endswith(TAIL)
    json.dumps(out)

    # Without the field: the same rows, checks and status; only the description differs.
    old = run(
        scan(3),
        "application/pdf",
        Server(header, {1: [p1], 2: [p2], 3: [p3]}),
        Config(carry_in=False),
    )
    assert (old.status, old.issues, old.checkpoints, old.calls) == (
        res.status,
        res.issues,
        res.checkpoints,
        res.calls,
    )

    def strip(lines: list[Line]) -> list[Line]:
        return [{k: v for k, v in item.items() if k != "description"} for item in lines]

    assert strip(old.lines) == strip(res.lines)
    assert old.lines[1]["description"] == "LED panel, box of 100, RoHS" and old.notes == []


def test_note_points_at_the_line_in_the_whole_document() -> None:
    p1, p2, p3 = [line(1), line(2), line(3)], [line(4), descr(5, "Cable tie")], [line(6)]
    server = Server(
        header_for(p1 + p2 + p3), {1: [page(p1)], 2: [page(p2)], 3: [page(p3, "UV resistant")]}
    )
    res = run(scan(3), "application/pdf", server)
    (note,) = res.notes
    assert (note["page"], note["to_page"], note["line"], note["line_no"]) == (3, 2, 4, 5)
    assert res.lines[note["line"]]["description"] == "Cable tie UV resistant"
    assert res.status == "ok"


def test_previous_page_that_ends_with_a_control_total_is_not_touched() -> None:
    p1 = [line(1), line(2), {"total_label": "Carried forward", "amount": 40.0}]
    p2 = [line(3)]
    header = header_for([line(1), line(2), line(3)])
    res = run(scan(2), "application/pdf", Server(header, {1: [page(p1)], 2: [page(p2, TAIL)]}))
    assert [item["description"] for item in res.lines] == ["Item 1", "Item 2", "Item 3"]
    assert res.notes == [
        {
            "page": 2,
            "kind": "carry_without_owner",
            "text": TAIL,
            "source": "carry_in",
            "reason": "Page 1 ends with a control total.",
        }
    ]
    assert res.status == "ok" and res.issues == []
    assert res.checkpoints == [
        {"label": "Carried forward", "amount": 40.0, "after_line": 2, "page": 1}
    ]


def test_control_total_above_the_last_row_does_not_stop_the_join() -> None:
    p1 = [line(1), {"total_label": "Subtotal A", "amount": 20.0}, line(2)]
    p2 = [line(3)]
    header = header_for([line(1), line(2), line(3)])
    res = run(scan(2), "application/pdf", Server(header, {1: [page(p1)], 2: [page(p2, "zinc")]}))
    assert res.lines[1]["description"] == "Item 2 zinc"
    assert [n["kind"] for n in res.notes] == ["joined_across_pages"]
    assert res.status == "ok"


def test_tail_on_the_first_page_is_a_note_and_nothing_else() -> None:
    p1, p2 = [line(1), line(2)], [line(3)]
    server = Server(header_for(p1 + p2), {1: [page(p1, "Delivery note 4411")], 2: [page(p2)]})
    res = run(scan(2), "application/pdf", server)
    assert [item["description"] for item in res.lines] == ["Item 1", "Item 2", "Item 3"]
    assert res.notes == [
        {
            "page": 1,
            "kind": "carry_without_owner",
            "text": "Delivery note 4411",
            "source": "carry_in",
            "reason": "This is the first page of the document.",
        }
    ]
    assert res.status == "ok" and res.issues == []


def test_previous_page_without_rows_and_previous_page_that_was_not_read() -> None:
    p1, p3 = [line(1)], [line(2)]
    # page 2 has no rows at all (flagged as empty, as before); page 3 brings a tail
    server = Server(header_for(p1 + p3), {1: [page(p1)], 2: [page([])], 3: [page(p3, TAIL)]})
    res = run(scan(3), "application/pdf", server)
    assert [(n["kind"], n["reason"]) for n in res.notes] == [
        ("carry_without_owner", "Page 2 has no rows.")
    ]
    # the issue the flow had without the field; the note added none
    assert [i.check for i in res.issues] == ["page_empty"]

    # Page 1 came back cut off: three complete rows were salvaged, but nobody knows
    # whether the third is the last one printed.
    rows = ", ".join(json.dumps(line(n)) for n in (1, 2, 3))
    cut = sse('{"carry_in": "", "rows": [' + rows + ', {"line_no": 4, "sku": "SK')
    header = header_for([line(n) for n in (1, 2, 3, 4)])
    res = run(scan(2), "application/pdf", Server(header, {1: [cut], 2: [page([line(4)], TAIL)]}))
    assert len(res.lines) == 4 and res.lines[2]["description"] == "Item 3"
    assert [(n["kind"], n["reason"]) for n in res.notes] == [
        ("carry_without_owner", "Page 1 could not be read in full.")
    ]
    assert [i.check for i in res.issues] == ["page_read"]


@pytest.mark.parametrize(
    "value", [None, 7, True, ["compliant,", "country"], {"text": "x"}, "", "   "]
)
def test_carry_in_that_is_not_text_is_empty(value: object) -> None:
    p1, p2 = [line(1), line(2)], [line(3)]
    server = Server(header_for(p1 + p2), {1: [page(p1)], 2: [{"carry_in": value, "rows": p2}]})
    res = run(scan(2), "application/pdf", server)
    assert res.notes == [] and res.lines[1]["description"] == "Item 2" and res.status == "ok"


def test_answer_without_the_field_is_read_as_before() -> None:
    p1, p2 = [line(1), line(2)], [line(3)]
    # {"rows": [...]} only
    res = run(scan(2), "application/pdf", Server(header_for(p1 + p2), {1: [p1], 2: [p2]}))
    assert res.notes == [] and res.status == "ok" and len(res.lines) == 3
    assert res.pages[1].carry_in == ""


def test_carry_in_of_a_salvaged_answer_is_not_used() -> None:
    p1 = [line(1), line(2)]
    broken = sse(
        '{"carry_in": "'
        + TAIL
        + '", "rows": ['
        + json.dumps(line(3))
        + ', {"line_no": 4, "sku": "SK'
    )
    server = Server(header_for(p1 + [line(3)]), {1: [page(p1)], 2: [broken]})
    res = run(scan(2), "application/pdf", server)
    # the complete row is kept, as before
    assert [item["sku"] for item in res.lines] == ["SKU-001", "SKU-002", "SKU-003"]
    assert res.notes == [] and res.lines[1]["description"] == "Item 2"
    assert "page_read" in [i.check for i in res.issues]


# --------------------------------------------------------------------------- #
# Two attempts at a page
# --------------------------------------------------------------------------- #


def test_tail_comes_from_the_attempt_that_is_kept() -> None:
    p1 = [line(1), line(2)]
    bad, good = [line(3, amount=99.0), line(4)], [line(3), line(4)]
    header = header_for(p1 + good)

    # the second attempt is clean and wins: its tail is the one that is joined
    server = Server(header, {1: [page(p1)], 2: [page(bad, "WRONG TAIL"), page(good, TAIL)]})
    res = run(scan(2), "application/pdf", server)
    assert server.calls[2] == 2 and res.status == "ok"
    assert res.lines[1]["description"] == f"Item 2 {TAIL}"
    assert [n["text"] for n in res.notes] == [TAIL]

    # the second attempt fails: the first one stays, and so does its tail
    failing: list[Answer] = [page(bad, TAIL), httpx.Response(500, text="boom")]
    server = Server(header, {1: [page(p1)], 2: failing})
    res = run(scan(2), "application/pdf", server)
    assert server.calls[2] == 2 and res.lines[2]["amount"] == 99.0
    assert res.lines[1]["description"] == f"Item 2 {TAIL}"
    assert [n["text"] for n in res.notes] == [TAIL]

    # the attempt that is kept has no tail: the tail of the rejected attempt is not used
    server = Server(header, {1: [page(p1)], 2: [page(bad, "WRONG TAIL"), page(good)]})
    res = run(scan(2), "application/pdf", server)
    assert res.lines[1]["description"] == "Item 2" and res.notes == []


def test_both_attempts_are_asked_with_the_field() -> None:
    p1 = [line(1), line(2, amount=99.0)]
    server = Server(header_for(p1 + [line(3)]), {1: [page(p1)], 2: [page([line(3)])]})
    handler, bodies = recording(server)
    run(scan(2), "application/pdf", handler)
    assert prompts(bodies) == {1: [co.carry_prompt(1, 2)] * 2, 2: [co.carry_prompt(2, 2)]}


# --------------------------------------------------------------------------- #
# Where the request must not change
# --------------------------------------------------------------------------- #


def all_bodies(data: bytes, mime: str, cfg: Config, answer: Answer) -> list[str]:
    handler, bodies = recording(Server(header_for([line(1), line(2)]), {1: [answer]}))
    res = run(data, mime, handler, cfg)
    assert res.status == "ok" and res.notes == []
    assert [item["description"] for item in res.lines] == ["Item 1", "Item 2"]
    return sorted(json.dumps(b, sort_keys=True) for b in bodies)


@pytest.mark.parametrize("kind", ["one-page pdf", "photo"])
def test_one_page_pdf_and_photo_are_asked_exactly_as_before(kind: str) -> None:
    data, mime = (
        (scan(1), "application/pdf") if kind == "one-page pdf" else (jpeg(photo()), "image/jpeg")
    )
    rows = [line(1), line(2)]
    handler, bodies = recording(Server(header_for(rows), {1: [rows]}))
    res = run(data, mime, handler)
    assert res.status == "ok" and res.notes == []
    # the old prompt, to the byte
    assert prompts(bodies) == {1: [EXTRACT_PAGE.format(page=1, total=1)]}
    # the whole body of every request (prompt, attachment, model, limits) is the one sent
    # with the field switched off
    assert all_bodies(data, mime, Config(), rows) == all_bodies(
        data, mime, Config(carry_in=False), rows
    )
    # and a "carry_in" that comes back unasked is not looked at
    assert all_bodies(data, mime, Config(), page(rows, TAIL)) == all_bodies(
        data, mime, Config(carry_in=False), rows
    )


def test_switched_off_a_scan_of_several_pages_is_asked_and_read_as_before() -> None:
    p1, p2 = [line(1), line(2)], [line(3)]
    handler, bodies = recording(Server(header_for(p1 + p2), {1: [page(p1)], 2: [page(p2, TAIL)]}))
    res = run(scan(2), "application/pdf", handler, Config(carry_in=False))
    assert prompts(bodies) == {p: [EXTRACT_PAGE.format(page=p, total=2)] for p in (1, 2)}
    assert res.notes == []
    assert [item["description"] for item in res.lines] == ["Item 1", "Item 2", "Item 3"]
    assert res.status == "ok" and all(p.carry_in == "" for p in res.pages)


def test_read_page_without_the_argument_sends_the_old_prompt() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["messages"][0]["content"][0]["text"])
        return httpx.Response(200, text=sse(json.dumps(page([line(1)], TAIL))))

    async def go() -> PageResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            part = {"type": "text", "text": ""}
            return await pl.read_page(client, part, 3, 5, None, Config(), asyncio.Semaphore(1))

    pr = asyncio.run(go())
    assert seen == [EXTRACT_PAGE.format(page=3, total=5)]
    assert pr.carry_in == "" and len(pr.lines) == 1


# --------------------------------------------------------------------------- #
# A PDF with a text layer
# --------------------------------------------------------------------------- #


def split_doc() -> tuple[bytes, JSONObject, JSONObject, dict[int, list[Answer]], dict[int, str]]:
    """m_split.pdf: every page break cuts a row. Returns (data, truth, header, {page:
    answer with the field}, {page: tail}): the answers a model would give - the
    description of a cut row ends where the page ends, the rest is "carry_in" of the
    next page."""
    data, t, header, pages = truth_doc("m_split")
    tails = {}
    for s in t["split_lines"]:
        row = next(r for r in pages[s["page"]] if r["line_no"] == s["line_no"])
        words = row["description"].split()
        row["description"], tails[s["split_to_page"]] = " ".join(words[:-4]), " ".join(words[-4:])
    answers: dict[int, list[Answer]] = {
        p: [page(rows, tails.get(p, ""))] for p, rows in pages.items()
    }
    return data, t, header, answers, tails


def descriptions(res: Result) -> list[str]:
    return [" ".join(item["description"].split()) for item in res.lines]


def test_witness_request_is_not_changed() -> None:
    data, _, header, pages = truth_doc("l_classic")
    server = Server(header, {p: [rows] for p, rows in pages.items()}, saved_layout("l_classic"))
    handler, bodies = recording(server)
    res = run(data, "application/pdf", handler)
    assert res.method == "layout" and res.layout["status"] == "confirmed"
    assert res.layout["witness_page"] == 1
    # a document of ten pages, the old prompt
    assert prompts(bodies) == {1: [EXTRACT_PAGE.format(page=1, total=10)]}
    assert res.status == "ok" and res.notes == []


def test_page_read_by_the_model_on_the_layout_road_gets_the_field_and_the_witness_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, t, header, pages = truth_doc("l_classic")
    lose_a_line(monkeypatch, 3)
    server = Server(header, {p: [rows] for p, rows in pages.items()}, saved_layout("l_classic"))
    handler, bodies = recording(server)
    res = run(data, "application/pdf", handler)
    assert [p.method for p in res.pages] == ["layout", "layout", "model"] + ["layout"] * 7
    assert prompts(bodies) == {
        1: [EXTRACT_PAGE.format(page=1, total=10)],
        3: [co.carry_prompt(3, 10)],
    }
    assert res.status == "ok" and exact(res.lines, t["lines"])


def test_fallback_for_the_whole_text_layer_pdf_joins_and_grounding_stays_quiet() -> None:
    """Every page by the model, checked against the text layer; the joined tail raises nothing."""
    data, t, header, answers, _ = split_doc()
    handler, bodies = recording(Server(header, answers))
    res = run(data, "application/pdf", handler, Config(use_layout=False))
    n = len(answers)
    assert res.evidence_source == "text_layer" and res.method == "model"
    assert prompts(bodies) == {p: [co.carry_prompt(p, n)] for p in answers}
    assert [i.to_json() for i in res.issues] == [] and res.status == "ok"
    assert exact(res.lines, t["lines"])
    # all four rows are whole again
    assert descriptions(res) == [" ".join(item["description"].split()) for item in t["lines"]]
    assert [(note["kind"], note["page"], note["line_no"]) for note in res.notes] == [
        ("joined_across_pages", s["split_to_page"], s["line_no"]) for s in t["split_lines"]
    ]
    assert [res.lines[note["line"]]["line_no"] for note in res.notes] == [
        s["line_no"] for s in t["split_lines"]
    ]


def test_unusable_layout_answer_reads_the_rest_with_the_field() -> None:
    data, t, header, answers, _ = split_doc()
    handler, bodies = recording(Server(header, answers, sse("Sure! Here is the layout: {")))
    res = run(data, "application/pdf", handler)
    n = len(answers)
    assert res.layout["status"] == "failed" and res.method == "model"
    # the witness page was read before the layout failed: it keeps the old prompt and is
    # not read again
    assert prompts(bodies) == {1: [EXTRACT_PAGE.format(page=1, total=n)]} | {
        p: [co.carry_prompt(p, n)] for p in range(2, n + 1)
    }
    assert res.status == "ok" and res.issues == []
    assert descriptions(res) == [" ".join(item["description"].split()) for item in t["lines"]]
    assert [note["kind"] for note in res.notes] == ["joined_across_pages"] * len(t["split_lines"])


def test_tail_the_layout_parser_already_joined_is_not_joined_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Page 2 is read by the model, page 1 stays as found by code - and code has already
    taken the text at the top of page 2 into the last row of page 1."""
    data, t, header, answers, tails = split_doc()
    lose_a_line(monkeypatch, 2)
    # the same template, the same columns as l_classic
    handler, bodies = recording(Server(header, answers, saved_layout("l_classic")))
    res = run(data, "application/pdf", handler)
    assert [p.method for p in res.pages] == ["layout", "model", "layout", "layout", "layout"]
    assert prompts(bodies)[2] == [co.carry_prompt(2, 5)] and res.pages[1].carry_in == tails[2]
    cut = t["split_lines"][0]
    row = res.lines[cut["index"]]
    want = " ".join(t["lines"][cut["index"]]["description"].split())
    assert " ".join(row["description"].split()) == want  # once, not twice
    assert row["description"].count(tails[2]) == 1
    mine = [note for note in res.notes if note.get("source") == "carry_in"]
    assert mine == [
        {
            "page": 2,
            "kind": "carry_already_joined",
            "text": tails[2],
            "source": "carry_in",
            "to_page": 1,
            "line": cut["index"],
            "line_no": cut["line_no"],
            "sku": row["sku"],
        }
    ]
    assert res.issues == [] and res.status == "ok"


# --------------------------------------------------------------------------- #
# What the real model answered, through the flow
# --------------------------------------------------------------------------- #


def test_recorded_answers_to_a_scan_give_the_four_cut_rows_back() -> None:
    """No request: the five pages of the scan of m_split as the cheap model answered them
    with the field. The scan itself is not in the repository; a PDF of five blank pages
    stands in for it. This proves the join on recorded answers; it does not exercise any
    check of the real scan's pages (a blank page carries no image, so nothing measures it)."""
    saved = saved_scan_pages()
    recorded = {p["page"]: p for p in saved["pages"]}
    # the flow asks what the model was asked
    for p, rec in recorded.items():
        assert hashlib.sha256(co.carry_prompt(p, 5).encode()).hexdigest() == rec["prompt_sha256"]
    t = truth("m_split")
    h = t["header"]
    header: JSONObject = {k: h[k] for k in ("supplier_name", "invoice_number", "currency")}
    header |= {k: float(h[k]) for k in ("subtotal", "tax_total", "total")}
    header["tax_breakdown"] = [{k: float(v) for k, v in r.items()} for r in h["tax_breakdown"]]
    server = Server(header, {p: [sse(rec["text"])] for p, rec in recorded.items()})
    handler, bodies = recording(server)
    res = run(scan(5), "application/pdf", handler)
    assert prompts(bodies) == {p: [co.carry_prompt(p, 5)] for p in recorded}
    assert res.status == "ok" and res.issues == [] and len(res.lines) == 138
    cut = t["split_lines"]
    assert [(n["kind"], n["page"], n["line"], n["line_no"]) for n in res.notes] == [
        ("joined_across_pages", s["split_to_page"], s["index"], s["line_no"]) for s in cut
    ]
    for s in cut:
        got = " ".join(res.lines[s["index"]]["description"].split())
        assert got == " ".join(t["lines"][s["index"]]["description"].split())


# --------------------------------------------------------------------------- #
# join_carried on its own
# --------------------------------------------------------------------------- #


def test_join_carried_on_pages_found_by_code() -> None:
    def pages(prev_method: str, head: str) -> list[PageResult]:
        first = [line(1) | {"page": 1}, descr(2, head) | {"page": 1}]
        return [
            PageResult(1, first, method=prev_method),
            PageResult(2, [line(3) | {"page": 2}], carry_in="zinc  plated"),
        ]

    # found by code and the text is already there (whatever the spacing and the case):
    # nothing is appended
    prs = pages("layout", "Bolt M8 Zinc plated")
    assert [n["kind"] for n in pl.join_carried(prs)] == ["carry_already_joined"]
    assert prs[0].lines[1]["description"] == "Bolt M8 Zinc plated"
    # found by code and the text is not there (the parser did not join it): appended
    prs = pages("layout", "Bolt M8")
    assert [n["kind"] for n in pl.join_carried(prs)] == ["joined_across_pages"]
    assert prs[0].lines[1]["description"] == "Bolt M8 zinc plated"
    # read by the model: no look at what the description holds
    prs = pages("model", "Bolt M8 zinc plated")
    assert [n["kind"] for n in pl.join_carried(prs)] == ["joined_across_pages"]
    assert prs[0].lines[1]["description"] == "Bolt M8 zinc plated zinc plated"


def test_join_carried_skips_a_page_that_kept_a_read_error() -> None:
    prs = [
        PageResult(1, [line(1) | {"page": 1}], issues=[Issue("page_read", "timeout", page=1)]),
        PageResult(2, [line(2) | {"page": 2}], carry_in="zinc"),
    ]
    assert pl.join_carried(prs) == [
        {
            "page": 2,
            "kind": "carry_without_owner",
            "text": "zinc",
            "source": "carry_in",
            "reason": "Page 1 could not be read in full.",
        }
    ]
    assert prs[0].lines[0]["description"] == "Item 1"


def test_join_that_breaks_does_not_fail_the_document(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(pages: list[PageResult]) -> list[JSONObject]:
        raise ValueError("unexpected")

    monkeypatch.setattr(pl, "join_carried", broken)
    p1, p2 = [line(1), line(2)], [line(3)]
    server = Server(header_for(p1 + p2), {1: [page(p1)], 2: [page(p2, TAIL)]})
    res = run(scan(2), "application/pdf", server)
    assert res.status == "ok" and len(res.lines) == 3
    assert [(n["kind"], n["text"]) for n in res.notes] == [
        ("carry_join_failed", "ValueError: unexpected")
    ]
