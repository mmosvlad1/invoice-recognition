"""A photo before the model sees it: upright, flattened, and read as one page."""

import base64
import io
import json

import pytest
from PIL import Image

from invoice_recognition.llm import ContentPart
from invoice_recognition.recognition import photo
from invoice_recognition.recognition.prompts import EXTRACT_PAGE

from fakes import SHEET, Server, codes, header_for, jpeg, line, never, recording, run
from fakes import photo as drawn


def sent_image(part: ContentPart) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1])))


def test_exif_orientation_is_applied_before_rectify(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stored 300x200 with a red mark in the stored top-left corner; EXIF orientation 6
    # says "rotate 90 degrees clockwise to display": upright it is 200x300, mark top-right.
    img = Image.new("RGB", (300, 200), "white")
    img.paste((255, 0, 0), (0, 0, 60, 60))
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95, exif=exif)

    seen = []

    def no_rectify(im: Image.Image) -> Image.Image:
        seen.append(im.size)
        return im

    monkeypatch.setattr(photo, "rectify", no_rectify)
    out = sent_image(photo.part(photo.load(buf.getvalue())))
    assert seen == [(200, 300)]  # rectify already gets the upright sheet
    assert out.size == (200, 300)
    mark, paper = out.convert("RGB").getpixel((170, 30)), out.convert("RGB").getpixel((30, 30))
    assert isinstance(mark, tuple) and isinstance(paper, tuple)
    assert mark[0] > 200 and mark[1] < 80 and mark[2] < 80  # the mark is top-right
    assert paper[1] > 200  # top-left is paper


def test_image_without_exif_is_sent_unchanged_in_size() -> None:
    data = jpeg(Image.new("RGB", (300, 200), "white"))
    assert sent_image(photo.part(photo.load(data))).size == (300, 200)


def test_sheet_on_a_desk_is_cut_out_and_flattened() -> None:
    x0, y0, x1, y1 = SHEET
    corners = photo.find_page(drawn())
    assert corners is not None
    want = [x0, y0, x1, y0, x1, y1, x0, y1]  # top-left, top-right, bottom-right, bottom-left
    assert corners.ravel().tolist() == pytest.approx(want, abs=3)
    flat = photo.rectify(drawn())
    assert flat.size == pytest.approx((x1 - x0, y1 - y0), abs=4)


def test_photo_without_a_desk_is_left_as_it_is() -> None:
    x0, y0, x1, y1 = SHEET
    close = drawn().crop((x0 + 10, y0 + 10, x1 - 10, y1 - 10))  # paper from edge to edge
    assert photo.find_page(close) is None
    assert photo.rectify(close) is close


def test_photo_that_cannot_be_flattened_is_sent_as_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(img: Image.Image) -> Image.Image:
        raise ValueError("unexpected")

    monkeypatch.setattr(photo, "rectify", broken)
    assert sent_image(photo.part(drawn())).size == drawn().size


def test_photo_is_read_as_one_page_with_the_page_prompt() -> None:
    rows = [line(1), line(2), line(3)]
    data = jpeg(drawn())
    server = Server(header_for(rows), {1: [rows]})
    handler, bodies = recording(server)
    res = run(data, "image/jpeg", handler)
    assert res.status == "ok" and res.issues == [] and len(res.lines) == 3
    assert res.calls == 2 and dict(server.calls) == {"header": 1, 1: 1}
    # header and page: the same flattened photo, the page asked without "carry_in"
    expected = photo.part(photo.load(data))
    assert [b["messages"][0]["content"][1] for b in bodies] == [expected] * 2
    assert bodies[1]["messages"][0]["content"][0]["text"] == EXTRACT_PAGE.format(page=1, total=1)
    assert res.evidence_source is None and res.layout["reason"] == "not a PDF"
    assert [p.method for p in res.pages] == ["model"] and res.notes == []
    json.dumps(res.to_json())


def test_damaged_image_gives_failed_result_not_exception() -> None:
    res = run(b"\xff\xd8\xff\xe0 this is not a picture", "image/jpeg", never)
    assert res.status == "failed"
    assert codes(res.issues) == ["unreadable_file"]
    assert res.calls == 0 and res.lines == [] and res.pages == []
    json.dumps(res.to_json())
