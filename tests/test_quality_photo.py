"""quality.py on small drawn "photos": a light sheet with rows of words on a dark desk.

No files, no network, no OCR engine. The drawings are crisp and clean, so the tests
check which reason fires and in which direction a number moves, not the limits
themselves (those were measured on a set of real and damaged photos).
"""

import itertools
import json

import cv2
import numpy as np
import pytest
from PIL import Image, ImageFilter

from invoice_recognition.recognition import photo as ph
from invoice_recognition.recognition import quality as ql

from drawn import PHOTO_H as H
from drawn import PHOTO_W as W
from drawn import SHEET, photo


def codes(img: Image.Image, lim: ql.Limits | None = None) -> dict[str, ql.Reason]:
    return {r.code: r for r in ql.check(img, lim)[1]}


def frame_score(img: Image.Image) -> float:
    """The score this module replaces: the variance of the Laplacian of the whole frame,
    on a denoised copy 1000 px wide (resized with INTER_AREA both ways)."""
    gray = np.asarray(img.convert("L"))
    f = 1000 / gray.shape[1]
    gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.medianBlur(gray, 5), cv2.CV_64F).var())


@pytest.fixture(scope="module")
def good() -> Image.Image:
    return photo()


# --------------------------------------------------------------------------- #
# Sheet in the frame
# --------------------------------------------------------------------------- #


def test_whole_sheet_in_frame_gives_no_reason(good: Image.Image) -> None:
    q, why = ql.check(good)
    assert q.sheet.found and q.sheet.free_corners == 4
    assert all(v == 0 for v in q.sheet.touch.values())
    assert q.text.boxes >= 50
    assert why == []


def test_sheet_cut_through_text_is_rejected_with_the_side(good: Image.Image) -> None:
    x0 = SHEET[0]
    cut = good.crop((0, 0, x0 + 420, H))  # the frame ends inside the last column of words
    q, why = ql.check(cut)
    assert q.sheet.found and q.sheet.touch["right"] > 0.5 and q.sheet.free_corners == 2
    r = {x.code: x for x in why}["text_at_edge"]
    assert (r.severity, r.side) == ("reject", "right")
    assert "on the right" in r.message


def test_sheet_cut_in_its_blank_margin_is_only_a_warning(good: Image.Image) -> None:
    cut = good.crop((SHEET[0] + 12, 0, W, H))  # the sheet runs out of the frame, the text does not
    q, why = ql.check(cut)
    assert q.sheet.touch["left"] > 0.5
    assert [(r.code, r.severity, r.side) for r in why] == [("sheet_cut", "warn", "left")]


def test_sheet_cut_between_rows_is_rejected(good: Image.Image) -> None:
    y0 = SHEET[1]
    cut = good.crop((0, 0, W, y0 + 60 + 8 * 30 + 12))  # the frame ends between two rows
    r = codes(cut)["text_at_edge"]
    assert (r.severity, r.side) == ("reject", "bottom")


def test_no_desk_visible_cannot_be_verified(good: Image.Image) -> None:
    x0, y0, x1, y1 = SHEET
    close = good.crop((x0 + 10, y0 + 10, x1 - 10, y1 - 10))
    q, why = ql.check(close)
    assert not q.sheet.found
    assert [(r.code, r.severity) for r in why] == [("sheet_edge_unseen", "warn")]


def test_no_desk_visible_and_text_runs_into_the_border(good: Image.Image) -> None:
    x0, y0, _, y1 = SHEET
    close = good.crop((x0 + 10, y0 + 10, x0 + 420, y1 - 10))
    r = codes(close)["text_at_edge"]
    assert (r.severity, r.side) == ("reject", "right")


def test_shadow_across_a_full_page_is_not_a_sheet_edge(good: Image.Image) -> None:
    x0, y0, x1, y1 = SHEET
    close = np.asarray(good.crop((x0 + 10, y0 + 10, x1 - 10, y1 - 10)), dtype=np.float32)
    # light falls off from left to right
    shade = np.linspace(1.0, 0.4, close.shape[1], dtype=np.float32)[None, :]
    shaded = Image.fromarray((close * shade).astype("uint8"))
    q, why = ql.check(shaded)
    assert not q.sheet.found
    assert "sheet_cut" not in {r.code for r in why} and "text_at_edge" not in {r.code for r in why}


def test_blank_image_has_no_text() -> None:
    q, why = ql.check(Image.fromarray(np.full((H, W), 128, np.uint8)))
    assert q.text.boxes == 0
    assert "no_text" in {r.code for r in why}


# --------------------------------------------------------------------------- #
# Sharpness on text
# --------------------------------------------------------------------------- #


def test_blur_lowers_both_text_scores_step_by_step(good: Image.Image) -> None:
    steps = [ql.measure(good.filter(ImageFilter.GaussianBlur(r))) for r in (0, 1, 2, 3)]
    for a, b in itertools.pairwise(steps):
        assert b.text_sharpness < a.text_sharpness
        assert b.text.edge_sharpness < a.text.edge_sharpness
    assert steps[-1].text.boxes >= 5  # the text is still found when blurred


def test_heavy_blur_is_named(good: Image.Image) -> None:
    r = codes(good.filter(ImageFilter.GaussianBlur(3)))["blurred"]
    assert r.severity == "reject"
    assert "blurred" not in codes(good)


def test_dark_photo_is_dark_not_blurred(good: Image.Image) -> None:
    dim = Image.fromarray((np.asarray(good, dtype=np.float32) * 0.4).astype("uint8"))
    q0, q1 = ql.measure(good), ql.measure(dim)
    # The plain Laplacian falls with the square of brightness; the score divided by paper
    # brightness must not.
    assert q1.text.text_laplacian < 0.3 * q0.text.text_laplacian
    assert q1.text_sharpness == pytest.approx(q0.text_sharpness, rel=0.2)
    assert q1.text.edge_sharpness == pytest.approx(q0.text.edge_sharpness, rel=0.2)
    why = codes(dim)
    assert "dark" in why and "blurred" not in why
    # darkness does not hide the sheet
    assert q1.sheet.found and all(v == 0 for v in q1.sheet.touch.values())


def test_narrow_photo_is_not_scored_sharper(good: Image.Image) -> None:
    """The same blurred picture at four widths. A score of the whole frame jumps on the ones
    narrower than 1000 px (it enlarges them by repeating pixels); the score the flow decides
    on stays where it was."""
    blurred = good.resize((1500, 2000), Image.Resampling.BICUBIC).filter(
        ImageFilter.GaussianBlur(3)
    )
    narrow = {
        w: blurred.resize((w, w * 4 // 3), Image.Resampling.LANCZOS) for w in (1200, 1000, 900, 600)
    }
    old = {w: frame_score(img) for w, img in narrow.items()}
    assert old[900] > 2 * old[1200] and old[600] > 2 * old[1200]  # the flaw
    scores = {w: ql.measure(img).text_sharpness for w, img in narrow.items()}
    assert all(s == pytest.approx(scores[1200], rel=0.15) for s in scores.values())
    assert all(codes(img)["blurred"].severity == "reject" for img in narrow.values())


def test_page_with_little_text_is_not_scored_as_blurred() -> None:
    """Two rows of words instead of eighteen, equally sharp: the frame score falls several
    times (less ink in the frame), the score on the text does not move."""
    full, sparse = photo(rows=18), photo(rows=2)
    assert frame_score(ph.rectify(sparse)) < 0.25 * frame_score(ph.rectify(full))
    assert ql.measure(sparse).text_sharpness == pytest.approx(
        ql.measure(full).text_sharpness, rel=0.05
    )
    assert ql.check(sparse)[1] == []


def test_edge_sharpness_does_not_depend_on_image_size(good: Image.Image) -> None:
    big = good.resize((W * 2, H * 2), Image.Resampling.BICUBIC)
    a, b = ql.measure(good), ql.measure(big)
    assert b.text.edge_sharpness == pytest.approx(a.text.edge_sharpness, rel=0.15)
    # reported in pixels of the image given
    assert b.text.height_px == pytest.approx(2 * a.text.height_px, rel=0.15)


def test_text_height_limit_is_off_by_default_and_works_when_set(good: Image.Image) -> None:
    tiny = good.resize((W // 2, H // 2), Image.Resampling.LANCZOS)
    h = ql.measure(tiny).text.height_px
    assert 0 < h < ql.measure(good).text.height_px
    assert "text_small" not in codes(tiny)
    assert "text_small" in codes(tiny, ql.Limits(text_height_px=h + 1))


# --------------------------------------------------------------------------- #
# Light
# --------------------------------------------------------------------------- #


def test_glare_spot_is_named_and_clean_paper_has_none(good: Image.Image) -> None:
    arr = np.asarray(good, dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    spot = 255 * np.exp(-((xx - 300) ** 2 + (yy - 400) ** 2) / (2 * 40.0**2))
    glared = Image.fromarray((arr + spot).clip(0, 255).astype("uint8"))
    assert ql.measure(good).light.glare == 0
    q, why = ql.check(glared)
    assert q.light.glare > 0.01
    assert [r.code for r in why] == ["glare"]


def test_white_paper_at_full_brightness_is_not_glare() -> None:
    assert ql.measure(photo(paper=255)).light.glare == 0


# --------------------------------------------------------------------------- #
# Shape of the result
# --------------------------------------------------------------------------- #


def test_result_serialises(good: Image.Image) -> None:
    q, why = ql.check(good.crop((0, 0, SHEET[0] + 420, H)))
    data = json.loads(json.dumps({"quality": q.to_json(), "reasons": [r.to_json() for r in why]}))
    assert "mask" not in data["quality"]["sheet"]
    assert data["quality"]["text_sharpness"] > 0
    assert data["reasons"][0]["code"] == "text_at_edge"
    # an edge with no text near it has no distance: null, not infinity
    assert ql.check(Image.new("L", (W, H), 128))[0].to_json()["text"][
        "edge_margin"
    ] == dict.fromkeys(ql.SIDES)


@pytest.mark.parametrize("lim", [ql.Limits(), ql.Limits(text_height_px=100)])
def test_every_reason_has_a_message_in_english(lim: ql.Limits) -> None:
    images = [
        photo(),
        photo().crop((0, 0, SHEET[0] + 420, H)),
        photo().crop((SHEET[0] + 12, 0, W, H)),
        photo().crop((70, 70, 530, 730)),
        photo().filter(ImageFilter.GaussianBlur(1)),
        photo().filter(ImageFilter.GaussianBlur(3)),
        Image.fromarray((np.asarray(photo(), dtype=np.float32) * 0.4).astype("uint8")),
        Image.new("L", (W, H), 128),
    ]
    found = {r.code: r.message for img in images for r in ql.check(img, lim)[1]}
    assert len(found) >= 6
    for message in found.values():
        assert message.isascii() and message[0].isupper() and message.endswith(".")
