"""The local half of the flow: what prepare() makes of a file before any request. No client,
no network."""

import base64
import io

from PIL import Image

from invoice_recognition.llm import ContentPart
from invoice_recognition.recognition import pipeline as pl
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.pdf import page_count

from drawn import jpeg, photo, scanned
from samples import INVOICES


def attached(part: ContentPart) -> tuple[str, str, bytes]:
    """(content type, file name or "", decoded bytes) of a part of a request."""
    if part["type"] == "file":
        name, url = part["file"]["filename"], part["file"]["file_data"]
    else:
        name, url = "", part["image_url"]["url"]
    head, data = url.split(",", 1)
    return head.removeprefix("data:").removesuffix(";base64"), name, base64.b64decode(data)


def test_text_layer_pdf_is_cut_into_one_pdf_per_page() -> None:
    data = (INVOICES / "m_grouped.pdf").read_bytes()
    n = page_count(data)
    prep = pl.prepare(data, "application/pdf", Config())

    assert prep.failure is None and prep.is_pdf and prep.n == n > 1
    assert prep.evidence is not None and prep.evidence.source == "text_layer"
    assert prep.text_pages is not None and len(prep.text_pages) == n  # the layout road
    assert prep.quality == {} and prep.skipped == {}
    assert len(prep.parts) == n
    for p, part in enumerate(prep.parts, start=1):
        mime, name, pdf = attached(part)
        assert (mime, name, page_count(pdf)) == ("application/pdf", f"page{p}.pdf", 1)
    assert prep.header_part is not None
    mime, name, pdf = attached(prep.header_part)
    assert (mime, name, page_count(pdf)) == ("application/pdf", "header.pdf", 2)  # first, last


def test_scan_keeps_a_part_for_every_page_and_heads_with_the_pages_sent() -> None:
    prep = pl.prepare(scanned("good", "soft", "blank"), "application/pdf", Config())

    assert prep.failure is None and prep.n == 3
    assert prep.evidence is None and prep.text_pages is None
    assert prep.skipped == {3: "empty"}  # not sent, but still cut: parts are by page number
    assert [i.check for i in prep.scan_issues[2]] == ["scan_blurred"]
    assert [note["kind"] for note in prep.scan_notes] == ["page_without_text"]
    assert prep.quality["verdict"] == "warn"
    assert [attached(part)[:2] for part in prep.parts] == [
        ("application/pdf", f"page{p}.pdf") for p in (1, 2, 3)
    ]
    assert prep.header_part is not None
    _, name, pdf = attached(prep.header_part)
    assert (name, page_count(pdf)) == ("header.pdf", 2)  # pages 1 and 2: the last one is blank


def test_photo_is_one_jpeg_that_is_also_the_header() -> None:
    prep = pl.prepare(jpeg(photo()), "image/jpeg", Config())

    assert prep.failure is None and not prep.is_pdf and prep.n == 1
    assert prep.quality["verdict"] == "ok" and prep.photo_issues == []
    assert len(prep.parts) == 1 and prep.header_part is prep.parts[0]
    mime, _, image = attached(prep.parts[0])
    assert mime == "image/jpeg" and Image.open(io.BytesIO(image)).format == "JPEG"


def test_damaged_file_gives_a_failure_and_nothing_to_send() -> None:
    for data, mime in ((b"%PDF-1.7\nnot really", "application/pdf"), (b"\xff\xd8", "image/jpeg")):
        prep = pl.prepare(data, mime, Config())

        assert prep.failure is not None
        issues, notes = prep.failure
        assert [i.check for i in issues] == ["unreadable_file"] and notes == []
        assert prep.parts == [] and prep.header_part is None
