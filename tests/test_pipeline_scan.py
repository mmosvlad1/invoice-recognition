"""Quality of the pages of a scanned PDF before any request, in the flow. Fake OpenRouter,
no network. The "scans" are drawn pages saved as the one image of a PDF page; the tests
check which page is sent, which is not and what is said about it."""

import asyncio
import base64
import io
import json
from collections import Counter
from dataclasses import asdict

import cv2
import httpx
import numpy as np
import pytest
from PIL import Image, ImageFilter

from invoice_recognition.llm import file_part
from invoice_recognition.recognition import carryover as co
from invoice_recognition.recognition import evidence as ev
from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition import quality as ql
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.pdf import extract_pages, page_count
from invoice_recognition.recognition.results import PageResult, Result
from invoice_recognition.shapes import JSONObject, Line

from fakes import (
    PHOTO_H,
    PHOTO_W,
    Answer,
    Server,
    header_for,
    jpeg,
    line,
    never,
    photo,
    recording,
    run,
    scan,
    scan_with_invisible_ocr,
    scan_with_ocr_text,
    scanned,
    sheet,
)
from samples import INVOICES, truth_doc


def rows(*pages: int, per_page: int = 2) -> dict[int, list[Line]]:
    """Line items of the pages that carry any, numbered through: rows(1, 3) = positions 1, 2
    on page 1 and 3, 4 on page 3."""
    return {p: [line(k * per_page + i + 1) for i in range(per_page)] for k, p in enumerate(pages)}


def server_for(by_page: dict[int, list[Line]]) -> Server:
    """Answers every page it has rows for; the header adds them up. A page it has no rows
    for must not be asked."""
    every = [item for p in sorted(by_page) for item in by_page[p]]
    answers: dict[int, list[Answer]] = {p: [r] for p, r in by_page.items()}
    return Server(header_for(every), answers)


def canon(bodies: list[JSONObject]) -> Counter[str]:
    """The requests as a bag: they go out concurrently."""
    return Counter(json.dumps(b, sort_keys=True) for b in bodies)


def attached(body: JSONObject) -> bytes:
    data: str = body["messages"][0]["content"][1]["file"]["file_data"]
    return base64.b64decode(data.split(",", 1)[1])


def filename(body: JSONObject) -> str:
    name: str = body["messages"][0]["content"][1]["file"]["filename"]
    return name


def header_body(bodies: list[JSONObject]) -> JSONObject:
    (body,) = [b for b in bodies if filename(b) == "header.pdf"]
    return body


def page_verdicts(res: Result) -> list[str]:
    return [p["verdict"] for p in res.quality["pages"]]


def same_reading(a: Result, b: Result) -> bool:
    """Everything the flow extracted and decided, the clock and the quality block aside."""
    drop = ("duration_s", "quality", "header_attempts", "pages")
    ja, jb = a.to_json(), b.to_json()

    def pages(j: JSONObject) -> list[JSONObject]:
        return [
            {k: v for k, v in p.items() if k != "attempts"} | {"attempts": len(p["attempts"])}
            for p in j["pages"]
        ]

    same = {k: v for k, v in ja.items() if k not in drop} == {
        k: v for k, v in jb.items() if k not in drop
    }
    return same and pages(ja) == pages(jb)


# --------------------------------------------------------------------------- #
# A good scan: nothing changes
# --------------------------------------------------------------------------- #


def test_good_scan_sends_the_same_requests_and_gives_the_same_result() -> None:
    data, by_page = scanned("good", "good", "good"), rows(1, 2, 3)
    handler, bodies = recording(server_for(by_page))
    res = run(data, "application/pdf", handler)
    before_handler, before_bodies = recording(server_for(by_page))
    before = run(data, "application/pdf", before_handler, Config(scan_quality=False))

    # to the byte: prompts, files, limits
    assert canon(bodies) == canon(before_bodies) and len(bodies) == 4
    # ... and they are the requests the flow always built: every page as its own PDF, the
    # header from the first and the last page
    files = {filename(b): b["messages"][0]["content"][1] for b in bodies}
    pdf = "application/pdf"
    assert files == {
        f"page{p}.pdf": file_part(extract_pages(data, [p]), pdf, f"page{p}.pdf") for p in (1, 2, 3)
    } | {"header.pdf": file_part(extract_pages(data, [1, 3]), pdf, "header.pdf")}
    texts = {b["messages"][0]["content"][0]["text"] for b in bodies if "page" in filename(b)}
    assert texts == {co.carry_prompt(p, 3) for p in (1, 2, 3)}

    assert res.status == "ok" and res.issues == [] and res.notes == []
    assert len(res.lines) == 6 and res.calls == 4
    assert same_reading(res, before)
    assert before.quality == {}
    assert res.quality["verdict"] == "ok" and res.quality["reasons"] == []
    assert page_verdicts(res) == ["ok"] * 3
    warn = res.quality["limits"]["sharpness_warn"]
    assert all(p["measures"]["text_sharpness"] > warn for p in res.quality["pages"])
    assert [p.method for p in res.pages] == ["model"] * 3 and res.method == "model"
    out = res.to_json()
    assert out["quality"] == res.quality
    json.dumps(out, allow_nan=False)  # plain numbers only: no numpy types, no infinity


def test_one_page_scan_keeps_its_request() -> None:
    data = scanned("good")
    handler, bodies = recording(server_for(rows(1)))
    res = run(data, "application/pdf", handler)
    assert res.status == "ok" and res.calls == 2 and page_verdicts(res) == ["ok"]
    # header and page: the same one page
    assert all(attached(b) == extract_pages(data, [1]) for b in bodies)


def test_the_sample_scan_passes_without_a_remark() -> None:
    """A scan the cheap model reads well: every page "ok", the requests untouched."""
    _, _, header, pages = truth_doc("s_classic")
    data = (INVOICES / "s_classic.scan.pdf").read_bytes()
    answers: dict[int, list[Answer]] = {p: [r] for p, r in pages.items()}
    handler, bodies = recording(Server(header, answers))
    res = run(data, "application/pdf", handler)
    before_handler, before_bodies = recording(Server(header, answers))
    before = run(data, "application/pdf", before_handler, Config(scan_quality=False))

    assert page_verdicts(res) == ["ok"] * len(pages) and res.quality["verdict"] == "ok"
    assert canon(bodies) == canon(before_bodies) and len(bodies) == len(pages) + 1
    assert same_reading(res, before)
    assert not [i for i in res.issues if i.check.startswith("scan_")]
    assert not [n for n in res.notes if n.get("source") == "scan_quality"]
    assert res.evidence_source is None  # it did go the road of a scan
    worst = min(p["measures"]["text_sharpness"] for p in res.quality["pages"])
    # the lowest of the 20 good pages the limits were set on is 734
    assert worst > 1.5 * res.quality["limits"]["sharpness_warn"]


# --------------------------------------------------------------------------- #
# A page that cannot be read
# --------------------------------------------------------------------------- #


def test_blurred_page_is_not_sent_and_is_named_with_its_number() -> None:
    data, by_page = scanned("good", "blurred", "good"), rows(1, 3)
    server = server_for(by_page)
    handler, bodies = recording(server)
    seen: list[PageResult] = []

    async def go() -> Result:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await pl.process(data, "application/pdf", client, None, on_page=seen.append)

    res = asyncio.run(go())
    assert dict(server.calls) == {"header": 1, 1: 1, 3: 1} and res.calls == 3  # not page 2
    assert [item["page"] for item in res.lines] == [1, 1, 3, 3]  # the others are read as usual
    (issue,) = res.issues
    assert (issue.check, issue.page, issue.severity) == ("scan_blurred", 2, "error")
    assert "below 100" in issue.message and "not sent" in issue.message
    assert res.status == "needs_review" and res.notes == []
    assert res.quality["verdict"] == "warn" and page_verdicts(res) == ["ok", "reject", "ok"]
    (why,) = res.quality["reasons"]
    assert (why["page"], why["code"], why["severity"]) == (2, "blurred", "reject")
    assert "blurred" in why["message"]
    # the page is in the result and was announced: a progress bar that counts pages still
    # reaches the end
    assert [p.page for p in res.pages] == [1, 2, 3]
    assert [p.method for p in res.pages] == ["model", "skipped", "model"]
    assert res.pages[1].lines == [] and res.pages[1].attempts == []
    assert [i.check for i in res.pages[1].issues] == ["scan_blurred"]
    assert sorted(p.page for p in seen) == [1, 2, 3]
    assert res.method == "model"  # a page that was not read does not make the reading "mixed"
    out = res.to_json()
    assert out["pages"][1] == {
        "page": 2,
        "lines": 0,
        "repaired": 0,
        "method": "skipped",
        "issues": [issue.to_json()],
        "attempts": [],
    }
    json.dumps(out, allow_nan=False)
    # the header is read from the first and the last page that are sent - here still 1 and 3
    assert attached(header_body(bodies)) == extract_pages(data, [1, 3])


def test_every_page_blurred_fails_without_a_request() -> None:
    res = run(scanned("blurred", "blurred"), "application/pdf", never)
    assert res.status == "failed" and res.calls == 0 and res.cost_usd == 0
    assert res.lines == [] and res.header == {} and res.pages == []
    assert [(i.check, i.page, i.severity) for i in res.issues] == [
        ("scan_blurred", 1, "error"),
        ("scan_blurred", 2, "error"),
    ]
    assert res.quality["verdict"] == "reject" and page_verdicts(res) == ["reject", "reject"]
    json.dumps(res.to_json(), allow_nan=False)


def test_blurred_pages_and_blank_pages_only_fail_without_a_request() -> None:
    """Every page that has text is unreadable: the blank ones do not save the document."""
    res = run(scanned("blank", "blurred", "blank"), "application/pdf", never)
    assert res.status == "failed" and res.calls == 0
    assert [(i.check, i.page) for i in res.issues] == [("scan_blurred", 2)]
    assert [(n["kind"], n["page"]) for n in res.notes] == [
        ("page_without_text", 1),
        ("page_without_text", 3),
    ]
    assert res.quality["verdict"] == "reject" and page_verdicts(res) == ["empty", "reject", "empty"]


def test_blurred_first_page_moves_the_header_to_the_pages_that_are_read() -> None:
    data, by_page = scanned("blurred", "good", "good"), rows(2, 3)
    handler, bodies = recording(server_for(by_page))
    res = run(data, "application/pdf", handler)
    # not from the page nobody can read
    assert attached(header_body(bodies)) == extract_pages(data, [2, 3])
    assert [item["page"] for item in res.lines] == [2, 2, 3, 3] and res.status == "needs_review"
    assert "scan_blurred" in [i.check for i in res.issues]


# --------------------------------------------------------------------------- #
# A page without text
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["blank", "back"])
def test_page_without_text_is_not_sent_and_does_not_spoil_the_status(kind: str) -> None:
    data, by_page = scanned("good", kind, "good"), rows(1, 3)
    server = server_for(by_page)
    res = run(data, "application/pdf", server)
    assert dict(server.calls) == {"header": 1, 1: 1, 3: 1}  # no request for the blank page
    assert res.status == "ok" and res.issues == [] and len(res.lines) == 4
    assert res.notes == [
        {
            "page": 2,
            "kind": "page_without_text",
            "text": "No text was found on page 2, so it was not read.",
            "source": "scan_quality",
        }
    ]
    assert res.quality["verdict"] == "ok" and page_verdicts(res) == ["ok", "empty", "ok"]
    assert [(r["page"], r["code"], r["severity"]) for r in res.quality["reasons"]] == [
        (2, "no_text", "skip")
    ]
    assert res.pages[1].method == "skipped" and res.pages[1].issues == []
    assert res.pages[1].attempts == []
    assert res.to_json()["notes"] == res.notes


def test_blank_last_page_is_left_out_of_the_header_request() -> None:
    """A duplex scan: the totals are on the last PRINTED page, not on the blank back of it."""
    data, by_page = scanned("good", "good", "blank"), rows(1, 2)
    handler, bodies = recording(server_for(by_page))
    res = run(data, "application/pdf", handler)
    assert res.status == "ok" and len(bodies) == 3
    head = attached(header_body(bodies))
    assert head == extract_pages(data, [1, 2]) and page_count(head) == 2
    # the pages that are read keep their numbers in the prompt
    texts = {b["messages"][0]["content"][0]["text"] for b in bodies if b is not header_body(bodies)}
    assert texts == {co.carry_prompt(1, 3), co.carry_prompt(2, 3)}


def test_document_of_blank_pages_fails_without_a_request() -> None:
    res = run(scanned("blank", "blank"), "application/pdf", never)
    assert res.status == "failed" and res.calls == 0
    (issue,) = res.issues
    assert (issue.check, issue.page, issue.severity) == ("scan_no_text", None, "error")
    assert issue.message == "No page of the document has text. No request was sent."
    assert [n["page"] for n in res.notes] == [1, 2] and res.quality["verdict"] == "reject"


def test_tail_carried_over_a_page_that_was_not_read_is_not_joined() -> None:
    """The row a tail belongs to is on the previous page. When that page was skipped, the
    tail has no owner."""
    data = scanned("good", "blank", "good")
    p1, p3 = [line(1), line(2)], [line(3)]
    answers: dict[int, list[Answer]] = {
        1: [{"carry_in": "", "rows": p1}],
        3: [{"carry_in": "country of origin DE", "rows": p3}],
    }
    res = run(data, "application/pdf", Server(header_for(p1 + p3), answers))
    assert [item["description"] for item in res.lines] == ["Item 1", "Item 2", "Item 3"]
    assert [(n["kind"], n["page"]) for n in res.notes] == [
        ("page_without_text", 2),
        ("carry_without_owner", 3),
    ]
    assert res.notes[1]["reason"] == "Page 2 was not sent to the model."
    assert res.status == "ok"


# --------------------------------------------------------------------------- #
# A doubtful page
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("kind", "code", "words"), [("soft", "blurred", "not sharp"), ("dark", "dark", "dark")]
)
def test_doubtful_page_is_read_as_before_and_flagged_with_its_number(
    kind: str, code: str, words: str
) -> None:
    data, by_page = scanned("good", kind), rows(1, 2)
    handler, bodies = recording(server_for(by_page))
    res = run(data, "application/pdf", handler)
    before_handler, before_bodies = recording(server_for(by_page))
    before = run(data, "application/pdf", before_handler, Config(scan_quality=False))

    assert canon(bodies) == canon(before_bodies) and len(bodies) == 3  # read exactly as before
    assert res.lines == before.lines and before.status == "ok"
    assert [(i.check, i.page, i.severity) for i in res.issues] == [(f"scan_{code}", 2, "warning")]
    assert "not sent" not in res.issues[0].message
    assert res.status == "needs_review"  # the rule of the flow: any issue
    assert res.quality["verdict"] == "warn" and page_verdicts(res) == ["ok", "warn"]
    assert [(r["page"], r["code"], r["severity"]) for r in res.quality["reasons"]] == [
        (2, code, "warn")
    ]
    assert words in res.quality["reasons"][0]["message"]
    assert [i.check for i in res.pages[1].issues] == [f"scan_{code}"]
    assert res.pages[1].method == "model"


# --------------------------------------------------------------------------- #
# Switched off, broken, not a scan
# --------------------------------------------------------------------------- #


def test_scan_quality_can_be_switched_off() -> None:
    data, by_page = scanned("good", "blurred", "blank"), rows(1, 2, 3)
    server = server_for(by_page)
    res = run(data, "application/pdf", server, Config(scan_quality=False))
    # every page goes to the model, as before
    assert dict(server.calls) == {"header": 1, 1: 1, 2: 1, 3: 1}
    assert res.status == "ok" and res.issues == [] and res.notes == [] and len(res.lines) == 6
    assert res.quality == {} and [p.method for p in res.pages] == ["model"] * 3


def test_measurement_that_breaks_on_a_page_leaves_it_unchecked_and_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = ql.measure_page
    calls = []

    def flaky(img: Image.Image, density: float = 0.0) -> ql.PageQuality:
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("unexpected")
        return real(img, density)

    monkeypatch.setattr(ql, "measure_page", flaky)
    data, by_page = scanned("good", "blurred", "good"), rows(1, 2, 3)
    server = server_for(by_page)
    res = run(data, "application/pdf", server)
    # page 2 is read the way it was before the check existed
    assert dict(server.calls) == {"header": 1, 1: 1, 2: 1, 3: 1}
    assert res.status == "ok" and res.issues == [] and len(res.lines) == 6
    assert page_verdicts(res) == ["ok", "unchecked", "ok"] and res.quality["verdict"] == "ok"
    assert "unexpected" in res.quality["pages"][1]["error"]
    json.dumps(res.to_json(), allow_nan=False)


@pytest.mark.parametrize("broken", ["measure_pdf_pages", "judge_page"])
def test_check_that_breaks_altogether_does_not_fail_the_document(
    monkeypatch: pytest.MonkeyPatch, broken: str
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no renderer")

    monkeypatch.setattr(ql, broken, boom)
    data, by_page = scanned("blurred", "blank"), rows(1, 2)
    server = server_for(by_page)
    res = run(data, "application/pdf", server)
    assert dict(server.calls) == {"header": 1, 1: 1, 2: 1} and res.status == "ok"
    assert res.issues == [] and res.quality["verdict"] == "unchecked"
    errors = [res.quality.get("error")] + [p.get("error") for p in res.quality["pages"]]
    assert any("no renderer" in (e or "") for e in errors)


def test_renderer_that_counts_other_pages_leaves_the_scan_unchecked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ql, "measure_pdf_pages", lambda data: [ql.NO_IMAGE])
    data, by_page = scanned("good", "good"), rows(1, 2)
    res = run(data, "application/pdf", server_for(by_page))
    assert res.status == "ok" and res.quality["verdict"] == "unchecked"
    assert res.quality["error"] == "ValueError: 1 pages measured, the document has 2"


def test_pdf_page_without_an_image_is_not_measured() -> None:
    """An empty PDF page is not a scan of anything: there is no image to measure."""
    p1 = [line(1), line(2)]
    server = Server(header_for(p1), {1: [p1]})
    res = run(scan(1), "application/pdf", server)
    assert res.quality == {} and res.status == "ok" and dict(server.calls) == {"header": 1, 1: 1}


def test_page_with_visible_content_over_its_image_is_not_judged_by_the_image() -> None:
    """A background picture with the real text printed over it: the picture alone looks
    blank, the page is not."""
    rows_ = [line(1), line(2), line(3)]
    data = scan_with_ocr_text(rows_)  # a plain gray image and visible text over it
    server = Server(header_for(rows_), {1: [rows_]})
    res = run(data, "application/pdf", server)
    assert dict(server.calls) == {"header": 1, 1: 1} and res.status == "ok"
    assert len(res.lines) == 3
    assert res.quality["verdict"] == "unchecked" and page_verdicts(res) == ["unchecked"]
    assert res.quality["pages"][0]["note"] == ql.NOT_ONLY_A_SCAN and res.notes == []


@pytest.mark.parametrize(("kind", "verdict"), [("good", "ok"), ("blurred", "reject")])
def test_scan_with_an_invisible_ocr_layer_is_measured_by_its_image(kind: str, verdict: str) -> None:
    data = scan_with_invisible_ocr(sheet(kind))
    # a scan for the flow, text layer or not
    assert ev.has_text_layer(ev.pdf_text_layer(data)) and lay.scanned_pages(data) == [1]
    rows_ = [line(1), line(2)]
    server = Server(header_for(rows_), {1: [rows_]}) if verdict == "ok" else never
    res = run(data, "application/pdf", server)
    assert page_verdicts(res) == [verdict]
    assert res.status == ("ok" if verdict == "ok" else "failed")


def test_page_with_ink_and_no_lines_of_text_is_left_to_the_model() -> None:
    img = np.full((PHOTO_H, PHOTO_W), 245, np.uint8)
    cv2.circle(img, (300, 400), 150, 40, cv2.FILLED)  # a picture, a stamp
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PDF", resolution=72, quality=95)
    p1 = [line(1)]
    server = Server(header_for(p1), {1: [p1]})
    res = run(buf.getvalue(), "application/pdf", server)
    assert dict(server.calls) == {"header": 1, 1: 1} and res.status == "ok"
    assert res.issues == [] and res.notes == []
    assert page_verdicts(res) == ["unchecked"] and res.quality["verdict"] == "unchecked"


def test_pdf_with_a_text_layer_is_not_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(data: bytes) -> list[ql.PageQuality | Exception | str]:
        raise AssertionError("a PDF with a text layer of its own is not a scan")

    monkeypatch.setattr(ql, "measure_pdf_pages", boom)
    data, _, header, pages = truth_doc("s_classic")
    answers: dict[int, list[Answer]] = {p: [r] for p, r in pages.items()}
    res = run(data, "application/pdf", Server(header, answers), Config(use_layout=False))
    assert res.status == "ok" and res.quality == {} and res.evidence_source == "text_layer"


def test_photo_is_not_touched_by_the_scan_limits() -> None:
    rows_ = [line(1)]
    soft = jpeg(photo().filter(ImageFilter.GaussianBlur(0.8)))
    a = run(soft, "image/jpeg", Server(header_for(rows_), {1: [rows_]}))
    b = run(
        soft,
        "image/jpeg",
        Server(header_for(rows_), {1: [rows_]}),
        Config(
            scan_quality=False,
            scan_sharpness_reject=5000,
            scan_sharpness_warn=5000,
            scan_paper_level=255,
        ),
    )
    assert a.quality == b.quality
    assert [i.to_json() for i in a.issues] == [i.to_json() for i in b.issues]
    assert "pages" not in a.quality and [i.check for i in a.issues] == ["photo_blurred"]


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #


def test_scan_limits_come_from_config() -> None:
    data, by_page = scanned("good", "soft"), rows(1, 2)

    def read(cfg: Config, server: Server | None = None) -> Result:
        return run(data, "application/pdf", server or server_for(by_page), cfg)

    assert read(Config(scan_sharpness_warn=0)).issues == []  # 0 switches a limit off
    strict = read(Config(scan_sharpness_reject=350), server_for(rows(1)))
    assert [(i.check, i.page, i.severity) for i in strict.issues] == [("scan_blurred", 2, "error")]
    assert "below 350" in strict.issues[0].message
    assert strict.quality["limits"]["sharpness_reject"] == 350 and strict.calls == 2
    lenient = run(
        scanned("blurred", "dark"),
        "application/pdf",
        server_for(by_page),
        Config(scan_sharpness_reject=0, scan_sharpness_warn=0, scan_paper_level=0),
    )
    assert lenient.status == "ok" and lenient.quality["verdict"] == "ok" and lenient.calls == 3


def test_config_scan_limits_are_the_limits_of_the_quality_module() -> None:
    cfg, lim = asdict(Config()), asdict(ql.ScanLimits())
    named = {k[5:]: v for k, v in cfg.items() if k.startswith("scan_") and k != "scan_quality"}
    assert named and set(named) <= set(lim)  # every scan_* limit of Config is in ScanLimits ...
    assert named == {k: lim[k] for k in named}  # ... with the same default
    assert Config().scan_quality is True
    # the reasons that are about a sheet lying on a desk have no limit here
    assert not {"glare", "sheet_touch", "text_margin_cut", "text_margin_unseen"} & set(lim)


def test_page_whose_measurement_broke_is_logged_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(img: Image.Image, density: float = 0.0) -> ql.PageQuality:
        raise ValueError("unexpected")

    monkeypatch.setattr(ql, "measure_page", broken)
    data, by_page = scanned("good"), rows(1)
    with caplog.at_level("WARNING"):
        res = run(data, "application/pdf", server_for(by_page))
    assert page_verdicts(res) == ["unchecked"] and res.status == "ok"
    tracebacks = [r.exc_info for r in caplog.records if r.exc_info]
    assert len(tracebacks) == 1
    ((_, error, _),) = tracebacks
    assert isinstance(error, ValueError) and str(error) == "unexpected"
