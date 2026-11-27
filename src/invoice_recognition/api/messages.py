"""Every text of the API that a person reads, in one place.

Written for the accountant who uploads an invoice: full sentences, no internal names.
Machine-readable codes stay where they are used; only the words live here.
"""

from collections.abc import Callable

from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.pipeline import page_ranges

# --------------------------------------------------------------------------- #
# HTTP errors: Problem.detail (the code says the same for a program)
# --------------------------------------------------------------------------- #

BAD_REQUEST = "The request could not be read."
UNAUTHORIZED = "The access code is missing or wrong."
NOT_FOUND = "There is nothing at this address."
JOB_NOT_FOUND = (
    "There is no document with this id. The server keeps only the most recent documents, "
    "and none from before it was restarted."
)
METHOD_NOT_ALLOWED = "This address does not accept {method} requests."
# Also the upload's own check (uploads.py).
FILE_TOO_LARGE = "The file is larger than the limit of {limit}."
INVALID_REQUEST = (
    "The request is missing something or has a value that is not valid. "
    "The list of errors says which."
)
HTTP_ERROR = "The request could not be completed."
INTERNAL_ERROR = "Something went wrong on the server. The error has been logged."


def size(n: int) -> str:
    """A size in bytes as people read it: "10 MB" (1 MB = 1024 x 1024 bytes)."""
    mb = 1024 * 1024
    return f"{n / mb:g} MB" if n >= mb else f"{n:,} bytes"


# --------------------------------------------------------------------------- #
# Upload refusals (uploads.py; FILE_TOO_LARGE above)
# --------------------------------------------------------------------------- #

UNSUPPORTED_FILE = (
    "Only PDF, JPEG and PNG files can be read, and this file is none of them. "
    "Photos in HEIC format (iPhone) must be saved as JPEG first."
)
PDF_NEEDS_PASSWORD = "The PDF is protected by a password. Upload a copy that opens without one."
PDF_DAMAGED = "The PDF could not be opened. It may be damaged or incomplete."
PDF_NO_PAGES = "The PDF has no pages."
TOO_MANY_PAGES = (
    "The PDF has {pages} pages; the limit is {limit}. Split the file and upload the parts."
)
IMAGE_DAMAGED = "The image could not be opened. It may be damaged or incomplete."
TOO_MANY_DOCUMENTS = (
    "The server is busy with other documents: at most {limit} can wait or be read at once. "
    "Please try again shortly."
)
IMAGE_TOO_LARGE = (
    "The image has more pixels than can be read; the limit is {limit}. "
    "Save the image at a lower resolution."
)

# --------------------------------------------------------------------------- #
# The result (views.py)
# --------------------------------------------------------------------------- #

# Notes the API adds to the result, by kind. The flow's own notes keep their text.
NOTE_TEXTS = {
    # read: as printed, with three decimals ("8.200"); corrected: 8200
    "quantity_repaired": (
        "The quantity was read as {read} and corrected to {corrected}: the point in it is a "
        "thousands separator, as the amount of the line shows."
    ),
}

# The message of an issue, by its code, for a person. A code not listed keeps the
# message the flow wrote.
ISSUE_TEXTS: dict[str, Callable[[Issue], str]] = {}


def issue_text(issue: Issue) -> str:
    word = ISSUE_TEXTS.get(issue.check)
    return word(issue) if word is not None else issue.message


# --------------------------------------------------------------------------- #
# Why a job failed: JobError.message, by JobError.code (views.job_error)
# --------------------------------------------------------------------------- #

JOB_ERRORS = {
    # Used only when the flow gave no reason of its own; it normally does, and that
    # reason's message is the error's message.
    "photo_unreadable": "The photo cannot be read. Please take the photo again.",
    # Used only when the pages' verdicts are not known; see scan_unreadable() below.
    "scan_unreadable": "No page of the scan could be read.",
    "unreadable_file": "The file could not be opened for reading. It may be damaged or "
    "incomplete; export or scan it again.",
    "model_unavailable": "The model that reads invoices could not be reached: every request "
    "to it failed. Please try again later; if this keeps happening, tell the person who runs "
    "this service.",
    "no_line_items": "The model answered, but no line items were found in the document.",
    "no_header": "Line items were read, but the header and totals were not, so nothing could "
    "be checked.",
    "deadline_exceeded": "Reading took longer than the time allowed for one document and was "
    "stopped before the whole document was read.",
    "internal_error": "Something went wrong on the server while reading this document. The "
    "error has been logged.",
}


def _pages(pages: list[int]) -> tuple[str, bool]:
    """("page 2" | "pages 1–3" | "pages 1 and 4", more than one)."""
    if len(pages) == 1:
        return f"page {pages[0]}", False
    return f"pages {page_ranges(pages)}", True


def scan_unreadable(
    blurred: list[int], blank: list[int], too_large: list[int] | None = None
) -> str:
    """Why no page of a scan was sent: the pages too blurred, the pages without text and
    the pages holding an image too large to open."""
    too_large = too_large or []
    if not blurred and not too_large:
        return "The scan has no text on any page." if blank else JOB_ERRORS["scan_unreadable"]
    why = []
    if blurred:
        what, many = _pages(blurred)
        why.append(f"{what} {'are' if many else 'is'} too blurred")
    if too_large:
        what, many = _pages(too_large)
        why.append(f"{what} {'hold images' if many else 'holds an image'} too large to open")
    if blank:
        what, many = _pages(blank)
        why.append(f"{what} {'have' if many else 'has'} no text")
    listed = why[0] if len(why) == 1 else f"{', '.join(why[:-1])} and {why[-1]}"
    return f"No page of the scan could be read: {listed}."
