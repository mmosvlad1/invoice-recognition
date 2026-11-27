"""Photo quality in the flow: a photo is measured before any request. Fake OpenRouter."""

import base64
import io
import json
from dataclasses import asdict

import numpy as np
import pytest
from PIL import Image, ImageFilter, ImageOps

from invoice_recognition.llm import ContentPart
from invoice_recognition.recognition import photo as ph
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition import quality as ql
from invoice_recognition.recognition.config import Config

from drawn import PHOTO_H, SHEET, jpeg, scan
from drawn import photo as draw
from fakes import Server, header_for, line, never, recording, run


def drawn(kind: str = "good") -> bytes:
    """The drawn photo (a light sheet with rows of words on a dark desk), as a JPEG file."""
    img = draw()
    if kind == "blurred":
        img = img.filter(ImageFilter.GaussianBlur(3))
    elif kind == "soft":
        img = img.filter(ImageFilter.GaussianBlur(0.8))
    elif kind == "dark":
        img = Image.fromarray((np.asarray(img, dtype=np.float32) * 0.3).astype("uint8"))
    elif kind == "cut":  # the frame ends inside the last column of words
        img = img.crop((0, 0, SHEET[0] + 420, PHOTO_H))
    elif kind == "narrow_blurred":  # a blurred photo less than 1000 px wide
        big = img.resize((1500, 2000), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(3))
        img = big.resize((900, 1200), Image.Resampling.LANCZOS)
    elif kind == "blank":
        img = Image.new("L", img.size, 128)
    return jpeg(img)


def part_before_the_quality_check(data: bytes) -> ContentPart:
    """The photo part of a request, written out the way it was built before the check existed."""
    img = ph.rectify(ImageOps.exif_transpose(Image.open(io.BytesIO(data))))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def test_good_photo_is_read_as_before_and_the_request_is_the_same() -> None:
    rows = [line(1), line(2), line(3)]
    data = drawn()
    server = Server(header_for(rows), {1: [rows]})
    handler, bodies = recording(server)
    res = run(data, "image/jpeg", handler)
    assert res.status == "ok" and res.issues == [] and len(res.lines) == 3
    assert res.calls == 2 and dict(server.calls) == {"header": 1, 1: 1}
    # header and page: the same two requests with the same bytes of the photo as before
    parts = [b["messages"][0]["content"][1] for b in bodies]
    assert parts == [part_before_the_quality_check(data)] * 2
    assert all(len(b["messages"][0]["content"]) == 2 for b in bodies)
    assert res.quality["verdict"] == "ok" and res.quality["reasons"] == []
    assert res.quality["measures"]["text_sharpness"] > res.quality["limits"]["sharpness_warn"]
    out = res.to_json()
    assert out["quality"] == res.quality
    json.dumps(out, allow_nan=False)  # plain numbers only: no numpy types, no infinity


@pytest.mark.parametrize(
    ("kind", "code", "words"),
    [
        ("blurred", "blurred", "blurred"),
        ("cut", "text_at_edge", "cut off"),
        ("blank", "no_text", "No text was found"),
    ],
)
def test_unreadable_photo_is_rejected_before_any_request(kind: str, code: str, words: str) -> None:
    res = run(drawn(kind), "image/jpeg", never)
    assert res.status == "failed" and res.calls == 0 and res.cost_usd == 0
    assert res.lines == [] and res.header == {} and res.pages == []
    assert res.quality["verdict"] == "reject"
    (why,) = [r for r in res.quality["reasons"] if r["severity"] == "reject"]
    assert why["code"] == code and words in why["message"]  # the reason in words, for the user
    (issue,) = [i for i in res.issues if i.severity == "error"]
    assert issue.check == f"photo_{code}" and issue.page == 1
    assert issue.message.endswith(" No request was sent.")
    json.dumps(res.to_json(), allow_nan=False)


@pytest.mark.parametrize(("kind", "code"), [("dark", "dark"), ("soft", "blurred")])
def test_doubtful_photo_is_read_and_flagged(kind: str, code: str) -> None:
    rows = [line(1), line(2), line(3)]
    server = Server(header_for(rows), {1: [rows]})
    res = run(drawn(kind), "image/jpeg", server)
    assert dict(server.calls) == {"header": 1, 1: 1} and len(res.lines) == 3  # read as usual
    assert [(i.check, i.severity, i.page) for i in res.issues] == [(f"photo_{code}", "warning", 1)]
    assert res.status == "needs_review"  # the rule of the flow: any issue
    assert res.quality["verdict"] == "warn"
    assert [(r["code"], r["severity"]) for r in res.quality["reasons"]] == [(code, "warn")]
    assert res.to_json()["issues"][0]["severity"] == "warning"


def test_dark_but_sharp_photo_is_not_taken_for_a_blurred_one() -> None:
    rows = [line(1)]
    res = run(drawn("dark"), "image/jpeg", Server(header_for(rows), {1: [rows]}))
    assert res.quality["verdict"] == "warn"
    assert [r["code"] for r in res.quality["reasons"]] == ["dark"]
    good = run(drawn(), "image/jpeg", Server(header_for(rows), {1: [rows]}))
    sharp = res.quality["measures"]["text_sharpness"]
    assert sharp == pytest.approx(good.quality["measures"]["text_sharpness"], rel=0.1)


def test_narrow_blurred_photo_is_rejected() -> None:
    data = drawn("narrow_blurred")
    assert Image.open(io.BytesIO(data)).width == 900
    res = run(data, "image/jpeg", never)
    assert res.status == "failed" and [i.check for i in res.issues] == ["photo_blurred"]


def test_photo_quality_can_be_switched_off() -> None:
    rows = [line(1), line(2)]
    server = Server(header_for(rows), {1: [rows]})
    res = run(drawn("blurred"), "image/jpeg", server, Config(photo_quality=False))
    assert res.status == "ok" and res.issues == [] and res.calls == 2
    assert res.quality == {} and res.to_json()["quality"] == {}


def test_photo_limits_come_from_config() -> None:
    rows = [line(1)]
    soft = drawn("soft")  # a warning with the default limits

    def read(cfg: Config) -> list[str]:
        return [
            i.check
            for i in run(soft, "image/jpeg", Server(header_for(rows), {1: [rows]}), cfg).issues
        ]

    assert read(Config(photo_sharpness_warn=0)) == []  # 0 switches a limit off
    strict = run(soft, "image/jpeg", never, Config(photo_sharpness_reject=400))
    assert strict.status == "failed" and strict.calls == 0
    assert strict.quality["limits"]["sharpness_reject"] == 400
    assert "below 400" in strict.issues[0].message
    lenient = run(
        soft,
        "image/jpeg",
        Server(header_for(rows), {1: [rows]}),
        Config(photo_sharpness_reject=0, photo_sharpness_warn=0, photo_paper_level=0),
    )
    assert lenient.status == "ok" and lenient.quality["verdict"] == "ok"


def test_config_photo_limits_are_the_limits_of_the_quality_module() -> None:
    cfg, lim = asdict(Config()), asdict(ql.Limits())
    named = {k[6:]: v for k, v in cfg.items() if k.startswith("photo_") and k != "photo_quality"}
    assert named and set(named) <= set(
        lim
    )  # every photo_* limit of Config is in quality.Limits ...
    assert named == {k: lim[k] for k in named}  # ... with the same default


def test_quality_check_that_breaks_does_not_fail_the_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(
        img: Image.Image, lim: ql.Limits | None = None
    ) -> tuple[ql.Quality, list[ql.Reason]]:
        raise ValueError("unexpected")

    monkeypatch.setattr(ql, "check", broken)
    rows = [line(1), line(2)]
    res = run(drawn("blurred"), "image/jpeg", Server(header_for(rows), {1: [rows]}))
    # read the way every photo was before the check existed
    assert res.status == "ok" and res.calls == 2
    assert res.quality["verdict"] == "unchecked" and "unexpected" in res.quality["error"]


def test_damaged_image_is_not_measured() -> None:
    res = run(b"\xff\xd8\xff\xe0 this is not a picture", "image/jpeg", never)
    assert res.status == "failed" and res.quality == {}  # nothing could be measured


def test_pages_of_a_pdf_are_not_measured() -> None:
    p1 = [line(1), line(2)]
    res = run(scan(1), "application/pdf", Server(header_for(p1), {1: [p1]}))
    assert res.quality == {} and res.status == "ok"


@pytest.mark.parametrize(
    ("code", "value", "severity", "text"),
    [
        (
            "no_text",
            2.0,
            "reject",
            "No text was found in the photo (2 word-like regions, at least 5 needed).",
        ),
        (
            "text_at_edge",
            0.3,
            "reject",
            "Text reaches the edge of the frame (right side), 0.3 text heights from it:",
        ),
        ("sheet_cut", 0.8, "warn", "The sheet runs out of the frame (right side); no text is"),
        ("sheet_edge_unseen", None, "warn", "No edge of the sheet is visible, so nothing"),
        ("blurred", 120.0, "reject", "Text sharpness 120.0 is below 140. No request was sent."),
        (
            "blurred",
            300.0,
            "warn",
            "Text sharpness 300.0 is below 450. Numbers are usually still read; item codes (SKUs)",
        ),
        ("dark", 120.0, "warn", "Paper brightness 120.0 of 255 is below 150."),
        ("glare", 0.022, "warn", "Glare covers 2.2% of the sheet."),
        ("text_small", 9.5, "warn", "Text lines are 9.5 px tall, below 0."),
        ("unknown", None, "warn", "unknown"),
    ],
)
def test_photo_issue_names_the_number_and_the_limit(
    code: str, value: float | None, severity: str, text: str
) -> None:
    side = "right" if code in ("text_at_edge", "sheet_cut") else None
    issue = pl._photo_issue(ql.Reason(code, "for the user", severity, value, side), ql.Limits())
    assert issue.check == f"photo_{code}" and issue.page == 1
    assert issue.severity == ("error" if severity == "reject" else "warning")
    assert text in issue.message
    assert issue.message.endswith(" No request was sent.") == (severity == "reject")
    assert code == "unknown" or (issue.message[0].isupper() and issue.message.endswith("."))
