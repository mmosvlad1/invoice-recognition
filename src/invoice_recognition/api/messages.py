"""Every text of the API that a person reads, in one place.

Written for the accountant who uploads an invoice: full sentences, no internal names.
Machine-readable codes stay where they are used; only the words live here.
"""

from collections.abc import Callable

from invoice_recognition.recognition.checks import Issue

# --------------------------------------------------------------------------- #
# HTTP errors: Problem.detail (the code says the same for a program)
# --------------------------------------------------------------------------- #

BAD_REQUEST = "The request could not be read."
UNAUTHORIZED = "The access code is missing or wrong."
NOT_FOUND = "There is nothing at this address."
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
