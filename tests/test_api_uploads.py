"""The checks an uploaded file passes before a job is created: one test per refusal."""

import io
import zlib
from typing import Any, Literal

import pytest
from PIL import Image
from pypdf import PdfReader, PdfWriter

from invoice_recognition.api import messages
from invoice_recognition.api.problems import ApiError
from invoice_recognition.api.uploads import Upload, UploadLimits, inspect, sniff
from invoice_recognition.recognition.config import MAX_IMAGE_PIXELS, Config

import drawn
from fakes import make_pdf

PDF = make_pdf([["Invoice INV-1"], ["Page two"]])
LIMITS = UploadLimits()


def refused(data: bytes, limits: UploadLimits = LIMITS) -> ApiError:
    with pytest.raises(ApiError) as refusal:
        inspect(data, limits)
    return refusal.value


def encrypted(user_password: str) -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(PDF)))
    writer.encrypt(user_password=user_password, owner_password="owner", algorithm="AES-256")
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), 200).save(buf, "PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Accepted
# --------------------------------------------------------------------------- #


def test_a_pdf_is_counted() -> None:
    assert inspect(PDF, LIMITS) == Upload("pdf", "application/pdf", 2)


def test_images_are_one_page() -> None:
    assert inspect(drawn.jpeg(drawn.photo()), LIMITS) == Upload("image", "image/jpeg", 1)
    assert inspect(drawn.png(drawn.photo()), LIMITS) == Upload("image", "image/png", 1)


def test_a_pdf_anyone_may_open_is_accepted_though_encrypted() -> None:
    assert inspect(encrypted(""), LIMITS) == Upload("pdf", "application/pdf", 2)


def test_a_jpeg_cut_short_passes_the_checks() -> None:
    # Pillow reads the header only; decoding is the flow's, which then fails the job.
    data = drawn.jpeg(drawn.photo())
    assert inspect(data[: len(data) // 2], LIMITS) == Upload("image", "image/jpeg", 1)


def test_an_image_is_opened_only_as_the_format_its_first_bytes_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[list[str] | tuple[str, ...] | None] = []
    real = Image.open

    def spy(fp: io.BytesIO, mode: Literal["r"] = "r", formats: list[str] | None = None) -> Any:
        asked.append(formats)
        return real(fp, mode, formats)

    monkeypatch.setattr(Image, "open", spy)
    inspect(drawn.jpeg(drawn.photo()), LIMITS)
    inspect(drawn.png(drawn.photo()), LIMITS)
    assert asked == [["JPEG"], ["PNG"]]


def test_the_type_comes_from_the_content() -> None:
    assert sniff(PDF) == ("pdf", "application/pdf")
    assert sniff(b"\xff\xd8\xff\xe0") == ("image", "image/jpeg")
    assert sniff(b"\x89PNG\r\n\x1a\n") == ("image", "image/png")
    assert sniff(b" %PDF-1.7") is None


# --------------------------------------------------------------------------- #
# 413 file_too_large
# --------------------------------------------------------------------------- #


def test_a_file_exactly_at_the_limit_is_accepted_one_byte_more_is_not() -> None:
    assert inspect(PDF, UploadLimits(max_file_bytes=len(PDF))).pages_total == 2
    error = refused(PDF, UploadLimits(max_file_bytes=len(PDF) - 1))
    assert (error.status, error.code) == (413, "file_too_large")


def test_the_size_limit_is_said_in_megabytes() -> None:
    error = refused(b"%PDF-" + bytes(10 * 1024 * 1024))
    assert error.detail == "The file is larger than the limit of 10 MB."


# --------------------------------------------------------------------------- #
# 415 unsupported_file
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"Invoice INV-1, total 120.00 EUR\n",
        b"GIF89a\x01\x00\x01\x00\x00\x00\x00;",
        b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic",  # a HEIC photo's header
        b"PK\x03\x04",  # zip, docx, xlsx
    ],
    ids=["empty", "text", "gif", "heic", "zip"],
)
def test_anything_but_pdf_jpeg_png_is_unsupported(data: bytes) -> None:
    error = refused(data)
    assert (error.status, error.code, error.detail) == (
        415,
        "unsupported_file",
        messages.UNSUPPORTED_FILE,
    )


# --------------------------------------------------------------------------- #
# 422 unreadable_file and too_many_pages
# --------------------------------------------------------------------------- #


def unreadable(data: bytes, limits: UploadLimits = LIMITS) -> str:
    error = refused(data, limits)
    assert (error.status, error.code) == (422, "unreadable_file")
    return error.detail


def test_a_pdf_cut_short_is_unreadable() -> None:
    assert unreadable(PDF[: len(PDF) // 2]) == messages.PDF_DAMAGED
    assert unreadable(b"%PDF-1.7\nnothing else") == messages.PDF_DAMAGED


def test_a_pdf_that_needs_a_password_is_unreadable() -> None:
    assert unreadable(encrypted("user")) == messages.PDF_NEEDS_PASSWORD


def test_a_pdf_with_no_pages_is_unreadable() -> None:
    buf = io.BytesIO()
    PdfWriter().write(buf)
    assert unreadable(buf.getvalue()) == messages.PDF_NO_PAGES


def test_a_pdf_over_the_page_limit_is_refused() -> None:
    assert inspect(drawn.scan(50), LIMITS).pages_total == 50
    error = refused(drawn.scan(51))
    assert (error.status, error.code) == (422, "too_many_pages")
    assert (
        error.detail
        == "The PDF has 51 pages; the limit is 50. Split the file and upload the parts."
    )


def test_garbage_after_a_png_signature_is_unreadable() -> None:
    assert unreadable(b"\x89PNG\r\n\x1a\n" + b"not a chunk" * 10) == messages.IMAGE_DAMAGED


def test_a_png_with_a_broken_checksum_is_unreadable() -> None:
    data = bytearray(png(40, 40))
    at = data.index(b"IDAT") + 6  # inside the image data: its CRC no longer matches
    data[at] ^= 0xFF
    assert unreadable(bytes(data)) == messages.IMAGE_DAMAGED


def test_a_jpeg_signature_with_nothing_after_it_is_unreadable() -> None:
    assert unreadable(b"\xff\xd8\xff" + bytes(20)) == messages.IMAGE_DAMAGED


def test_an_image_over_the_pixel_limit_is_unreadable() -> None:
    assert inspect(png(100, 100), UploadLimits(max_image_pixels=10_000)).pages_total == 1
    detail = unreadable(png(100, 101), UploadLimits(max_image_pixels=10_000))
    assert detail == (
        "The image has more pixels than can be read; the limit is 10,000 pixels. "
        "Save the image at a lower resolution."
    )


def png_header_only(width: int, height: int) -> bytes:
    """A PNG that declares its size and carries no pixels at all."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return len(body).to_bytes(4) + kind + body + zlib.crc32(kind + body).to_bytes(4)

    ihdr = width.to_bytes(4) + height.to_bytes(4) + bytes([8, 0, 0, 0, 0])
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", b"") + chunk(b"IEND", b"")


@pytest.mark.parametrize(
    "size",
    [(8000, 7500), (20000, 20000)],  # 60 million: our limit; 400 million: Pillow's own
    ids=["over-the-limit", "decompression-bomb"],
)
def test_the_pixel_limit_is_checked_before_the_image_is_decoded(size: tuple[int, int]) -> None:
    detail = unreadable(png_header_only(*size))
    assert detail == (
        "The image has more pixels than can be read; the limit is 50 million pixels. "
        "Save the image at a lower resolution."
    )


def test_the_pixel_cap_of_an_image_is_the_flows_cap_for_a_page_image() -> None:
    assert UploadLimits().max_image_pixels == MAX_IMAGE_PIXELS == Config().max_image_pixels
