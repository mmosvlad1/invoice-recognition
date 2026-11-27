"""quality.py on the pages of a scanned PDF: drawn pages saved as the one image of a PDF
page, the way a scanner writes them. The tests check which verdict a page gets and how the
numbers move, not the limits themselves (those were measured on a set of scans)."""

import io
import json

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageFilter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from invoice_recognition.recognition import quality as ql

import drawn
from drawn import PHOTO_H as H
from drawn import PHOTO_W as W
from drawn import scan, scanned, sheet


def judged(
    img: Image.Image, lim: ql.ScanLimits | None = None
) -> tuple[str, list[str], ql.PageQuality]:
    q = ql.measure_page(img)
    verdict, why = ql.judge_page(q, lim)
    return verdict, [r.code for r in why], q


def test_verdicts_of_a_page() -> None:
    assert judged(sheet())[:2] == ("ok", [])
    assert judged(sheet("soft"))[:2] == ("warn", ["blurred"])
    assert judged(sheet("blurred"))[:2] == ("reject", ["blurred"])
    assert judged(sheet("dark"))[:2] == ("warn", ["dark"])
    assert judged(sheet("blank"))[:2] == ("empty", ["no_text"])
    dark_and_blurred = Image.fromarray(
        (np.asarray(sheet("blurred"), dtype=np.float32) * 0.4).astype("uint8")
    )
    assert judged(dark_and_blurred)[:2] == ("reject", ["blurred", "dark"])


def test_reasons_of_a_page_are_in_english() -> None:
    images = [sheet("soft"), sheet("blurred"), sheet("dark"), sheet("blank")]
    stamp = np.full((H, W), 245, np.uint8)
    cv2.circle(stamp, (300, 400), 150, 40, cv2.FILLED)
    images.append(Image.fromarray(stamp))
    found = {
        r.code + r.severity: r.message
        for img in images
        for r in ql.judge_page(ql.measure_page(img))[1]
    }
    assert len(found) == 5  # blurred warn and reject, dark, no_text, no_text_lines
    for message in found.values():
        assert message.isascii() and message[0].isupper() and message.endswith(".")


def test_no_reason_of_a_photo_on_a_desk_is_given_to_a_scan() -> None:
    """The same page as a photo: no desk in the frame, so the photo check cannot see the
    sheet edge. A scan is not asked."""
    assert "sheet_edge_unseen" in {r.code for r in ql.check(sheet())[1]}
    arr = np.asarray(sheet(), dtype=np.float32) * 0.8  # paper at 196: a burnt-out spot stands out
    yy, xx = np.mgrid[0:H, 0:W]
    spot = 255 * np.exp(-((xx - 300) ** 2 + (yy - 400) ** 2) / (2 * 40.0**2))
    glared = Image.fromarray((arr + spot).clip(0, 255).astype("uint8"))
    assert "glare" in {r.code for r in ql.check(glared)[1]}
    edge = sheet().crop((45, 0, W, H))  # the text runs into the border of the image
    assert "text_at_edge" in {r.code for r in ql.check(edge)[1]}
    for img in (sheet(), glared, edge):
        assert judged(img)[:2] == ("ok", [])


def test_show_through_is_not_text_but_faint_print_on_a_page_with_ink_is() -> None:
    verdict, _, q = judged(sheet("back"))
    # "words" are found, and still it is a blank back
    assert q.text.boxes >= 5 and q.contrast < 0.10 and q.ink == 0 and verdict == "empty"
    # The same faint print beside real ink (a stamp): the page is not blank, so it is
    # judged as text - and is unreadable.
    stamped = np.asarray(sheet("back")).copy()
    stamped[600:700, 100:300] = 30
    verdict, codes, q = judged(Image.fromarray(stamped))
    assert q.ink > 0.001 and (verdict, codes) == ("reject", ["blurred"])


def test_page_with_ink_and_no_lines_of_text_is_not_judged() -> None:
    img = np.full((H, W), 245, np.uint8)
    cv2.circle(img, (300, 400), 150, 40, cv2.FILLED)  # a picture, a stamp: not rows of words
    verdict, codes, q = judged(Image.fromarray(img))
    assert q.text.boxes < 5 and q.ink > 0.05
    assert (verdict, codes) == ("unchecked", ["no_text_lines"])


def test_blur_lowers_the_score_step_by_step_and_darkness_does_not() -> None:
    base = ql.measure_page(sheet()).text_sharpness
    steps = [
        ql.measure_page(sheet().filter(ImageFilter.GaussianBlur(r))).text_sharpness
        for r in (0.5, 1, 2, 3)
    ]
    assert all(b < a for a, b in zip([base, *steps], steps, strict=False))
    assert ql.measure_page(sheet("dark")).text_sharpness == pytest.approx(base, rel=0.1)
    q = ql.measure_page(sheet())
    # the number of the photo check
    photo_score = ql.Quality(W, H, ql.find_sheet(sheet()), q.text, q.light).text_sharpness
    assert q.text_sharpness == pytest.approx(photo_score)


def test_low_resolution_is_seen_only_at_the_resolution_of_the_scan() -> None:
    """A scan saved at a sixth of the resolution. Rendered to a fixed width it scores about
    as high as the original (the renderer repeats pixels); measured on its own pixels it is
    what it is: unreadable."""
    big = sheet().resize((W * 2, H * 2), Image.Resampling.BICUBIC)  # the "original": 1200 px wide
    # 200 px wide: a line of text is 3 px tall
    small = big.resize((W * 2 // 6, H * 2 // 6), Image.Resampling.LANCZOS)

    def as_page(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        # the same sheet of paper, whatever the pixels
        img.save(buf, "PDF", resolution=72 * img.width / 595, quality=95)
        return buf.getvalue()

    def fixed_width(data: bytes) -> float:
        pdf = pdfium.PdfDocument(data)
        rendered = pdf[0].render(scale=ql.NORM_W / pdf[0].get_size()[0]).to_pil().convert("L")
        return ql.measure_page(rendered).text_sharpness

    q_big, q_small = ql.measure_pdf_pages(as_page(big))[0], ql.measure_pdf_pages(as_page(small))[0]
    assert isinstance(q_big, ql.PageQuality) and isinstance(q_small, ql.PageQuality)
    assert (q_big.width, q_small.width) == (1200, 200)
    assert q_small.density == pytest.approx(q_big.density / 6, rel=0.01)
    assert q_small.text_sharpness < 0.2 * q_big.text_sharpness
    assert ql.judge_page(q_big)[0] == "ok" and ql.judge_page(q_small)[0] == "reject"
    assert fixed_width(as_page(small)) > 0.5 * fixed_width(
        as_page(big)
    )  # the flaw of a fixed render
    assert q_small.to_json()["dpi"] == pytest.approx(72 * 200 / 595, rel=0.01)


def test_page_made_of_several_images_is_rendered_at_their_resolution() -> None:
    """A scan written as two strips: no single image covers the sheet, so the page is
    rendered - at the density of the strips."""
    page = sheet()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(W, H), invariant=True)  # one point per pixel
    c.drawImage(ImageReader(page.crop((0, 0, W, H // 2))), 0, H // 2, width=W, height=H // 2)
    c.drawImage(ImageReader(page.crop((0, H // 2, W, H))), 0, 0, width=W, height=H // 2)
    c.save()
    (q,) = ql.measure_pdf_pages(buf.getvalue())
    assert isinstance(q, ql.PageQuality)
    assert (q.width, q.height) == (W, H) and q.density == pytest.approx(1.0)
    assert q.text.boxes == ql.measure_page(page).text.boxes and ql.judge_page(q)[0] == "ok"


def test_very_wide_scan_is_measured_at_a_bounded_width() -> None:
    wide = sheet().resize((ql.SCAN_MAX_W + 400, (ql.SCAN_MAX_W + 400) * 4 // 3))
    buf = io.BytesIO()
    wide.save(buf, "PDF", resolution=72 * wide.width / 595)
    (q,) = ql.measure_pdf_pages(buf.getvalue())
    assert isinstance(q, ql.PageQuality) and q.width == ql.SCAN_MAX_W


def test_page_with_visible_content_is_not_measured_as_a_scan() -> None:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595, 842), invariant=True)
    c.drawImage(ImageReader(sheet()), 0, 0, width=595, height=842)
    c.drawString(40, 800, "printed over the picture")
    c.showPage()
    c.drawString(40, 800, "no picture at all")
    c.save()
    assert ql.measure_pdf_pages(buf.getvalue()) == [ql.NOT_ONLY_A_SCAN, ql.NOT_ONLY_A_SCAN]


def test_page_quality_serialises() -> None:
    q = ql.measure_pdf_pages(scanned("good", "blank"))
    assert all(isinstance(x, ql.PageQuality) for x in q)
    out = json.loads(
        json.dumps([x.to_json() for x in q if isinstance(x, ql.PageQuality)], allow_nan=False)
    )
    assert out[0]["text_sharpness"] > 0 and out[0]["width"] == W
    assert out[0]["dpi"] == pytest.approx(72.0)
    assert out[1]["text"]["boxes"] == 0 and out[1]["ink"] == 0
    assert ql.measure_pdf_pages(scan(2)) == [ql.NO_IMAGE, ql.NO_IMAGE]


# --------------------------------------------------------------------------- #
# Pages too large to open
# --------------------------------------------------------------------------- #


def test_a_page_image_declaring_too_many_pixels_is_not_opened() -> None:
    # 30000 x 30000 gray would take 900 MB once decoded; its size is read, nothing else.
    data = drawn.declared_image_scan(30000, 30000)
    assert len(data) < 2000
    assert ql.measure_pdf_pages(data) == [ql.TooLarge(900_000_000, 50_000_000)]


def test_the_pixel_cap_is_a_parameter_and_counts_every_image() -> None:
    assert ql.measure_pdf_pages(scanned("good"), max_pixels=W * H - 1) == [
        ql.TooLarge(W * H, W * H - 1)
    ]
    [measured] = ql.measure_pdf_pages(scanned("good"), max_pixels=W * H)
    assert isinstance(measured, ql.PageQuality)


def test_a_render_of_the_page_is_capped_at_the_same_number_of_pixels() -> None:
    # A small image off to one side: the page is rendered at the image's density (2 px per
    # point), which for the whole sheet is about 2 million pixels.
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595, 842), invariant=True)
    c.drawImage(ImageReader(sheet().resize((100, 100))), 40, 700, width=50, height=50)
    c.save()
    pdf = pdfium.PdfDocument(buf.getvalue())
    try:
        free = ql.scan_page_image(pdf[0])
        capped = ql.scan_page_image(pdf[0], max_pixels=200_000)
    finally:
        pdf.close()
    assert not isinstance(free, str | ql.TooLarge) and not isinstance(capped, str | ql.TooLarge)
    assert free[0].width * free[0].height > 1_900_000
    assert capped[0].width * capped[0].height <= 200_000
    assert capped[1] == free[1]  # the density of the scan is reported as it is
