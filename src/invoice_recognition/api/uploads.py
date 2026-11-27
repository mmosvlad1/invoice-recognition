"""The cheap checks an uploaded file passes before a job is created.

The type of a file is decided by its first bytes, never by its name or the type the
client declared. A PDF is opened far enough to count its pages; an image far enough
to read its size and to check its structure (a PNG's checksums). Anything deeper, a
photo too blurred to read or a scan with nothing on it, is found by the flow and
ends the job as "failed" with a reason; it is not an upload error.

Synchronous on purpose: pypdf and Pillow work here, so the API runs `inspect` on the
same single thread as the flow's local work.
"""

import io
import logging
from dataclasses import dataclass
from typing import Literal

from PIL import Image
from pypdf.errors import FileNotDecryptedError

from invoice_recognition.api import messages
from invoice_recognition.api.problems import ApiError
from invoice_recognition.recognition.pdf import page_count

logger = logging.getLogger(__name__)

MB = 1024 * 1024

type Kind = Literal["pdf", "image"]

# First bytes -> (kind, media type, Pillow format).
_SIGNATURES: tuple[tuple[bytes, Kind, str, str], ...] = (
    (b"%PDF-", "pdf", "application/pdf", ""),
    (b"\xff\xd8\xff", "image", "image/jpeg", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "image", "image/png", "PNG"),
)


@dataclass(frozen=True)
class UploadLimits:
    max_file_bytes: int = 10 * MB
    max_pages: int = 50
    max_image_pixels: int = 50_000_000


@dataclass(frozen=True)
class Upload:
    """A file that passed the checks."""

    kind: Kind
    mime: str  # application/pdf | image/jpeg | image/png
    pages_total: int  # 1 for an image


def sniff(data: bytes) -> tuple[Kind, str] | None:
    """(kind, media type) by the first bytes; None for anything but a PDF, JPEG or PNG."""
    for magic, kind, mime, _ in _SIGNATURES:
        if data.startswith(magic):
            return kind, mime
    return None


def inspect(data: bytes, limits: UploadLimits) -> Upload:
    """The file as an Upload, or ApiError: 413 file_too_large, 415 unsupported_file,
    422 unreadable_file (does not open, needs a password, no pages, too many pixels) or
    422 too_many_pages."""
    if len(data) > limits.max_file_bytes:
        raise ApiError(
            413,
            "file_too_large",
            messages.FILE_TOO_LARGE.format(limit=messages.size(limits.max_file_bytes)),
        )
    found = sniff(data)
    if found is None:
        raise ApiError(415, "unsupported_file", messages.UNSUPPORTED_FILE)
    kind, mime = found
    if kind == "pdf":
        return Upload(kind, mime, _pdf_pages(data, limits))
    _check_image(data, mime, limits)
    return Upload(kind, mime, 1)


def _pdf_pages(data: bytes, limits: UploadLimits) -> int:
    try:
        n = page_count(data)
    except FileNotDecryptedError:
        # A PDF anyone may open (only the owner's password is set) never gets here.
        raise ApiError(422, "unreadable_file", messages.PDF_NEEDS_PASSWORD) from None
    except Exception:  # fault boundary: whatever a damaged file makes pypdf raise
        logger.warning("an uploaded PDF could not be opened", exc_info=True)
        raise ApiError(422, "unreadable_file", messages.PDF_DAMAGED) from None
    if n == 0:
        raise ApiError(422, "unreadable_file", messages.PDF_NO_PAGES)
    if n > limits.max_pages:
        text = messages.TOO_MANY_PAGES.format(pages=n, limit=limits.max_pages)
        raise ApiError(422, "too_many_pages", text)
    return n


def _check_image(data: bytes, mime: str, limits: UploadLimits) -> None:
    fmt = next(f for _, _, m, f in _SIGNATURES if m == mime)
    too_large = ApiError(
        422, "unreadable_file", messages.IMAGE_TOO_LARGE.format(limit=_pixels(limits))
    )
    try:
        # Reads the header only; only the format the first bytes named is tried.
        with Image.open(io.BytesIO(data), formats=[fmt]) as img:
            if img.width * img.height > limits.max_image_pixels:
                raise too_large
            # The structure of the file without decoding it: a PNG's chunk checksums.
            # A JPEG cut short passes; the flow finds it when it decodes the photo.
            img.verify()
    except ApiError:
        raise
    except Image.DecompressionBombError:  # far above any limit we set: Pillow's own
        raise too_large from None
    except Exception:  # fault boundary: whatever a damaged file makes Pillow raise
        logger.warning("an uploaded image could not be opened", exc_info=True)
        raise ApiError(422, "unreadable_file", messages.IMAGE_DAMAGED) from None


def _pixels(limits: UploadLimits) -> str:
    n = limits.max_image_pixels
    return f"{n / 1_000_000:g} million pixels" if n >= 1_000_000 else f"{n:,} pixels"
