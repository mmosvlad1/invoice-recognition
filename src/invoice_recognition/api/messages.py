"""Every text of the API that a person reads, in one place.

Written for the accountant who uploads an invoice: full sentences, no internal names.
Machine-readable codes stay where they are used; only the words live here.
"""

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
