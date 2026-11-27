"""The vendored PDF.js: the files are exactly those its README names, so that no hook,
editor or merge can change them unnoticed."""

import hashlib
import re

from invoice_recognition.api.page import VENDOR

PDFJS = VENDOR / "pdfjs-4.10.38"
SUMS = re.compile(r"^([0-9a-f]{64})  (\d+) +(\S+)$", re.MULTILINE)


def test_the_vendored_files_are_the_ones_named_in_the_readme() -> None:
    listed = SUMS.findall((PDFJS / "README.txt").read_text())
    assert [name for _, _, name in listed] == ["pdf.min.mjs", "pdf.worker.min.mjs", "LICENSE"]
    for digest, size, name in listed:
        data = (PDFJS / name).read_bytes()
        assert len(data) == int(size), name
        assert hashlib.sha256(data).hexdigest() == digest, name


def test_nothing_else_is_vendored() -> None:
    shipped = {p.relative_to(VENDOR).as_posix() for p in VENDOR.rglob("*") if p.is_file()}
    assert shipped == {
        f"pdfjs-4.10.38/{name}"
        for name in ("pdf.min.mjs", "pdf.worker.min.mjs", "LICENSE", "README.txt")
    }


def test_the_licence_is_apache_2() -> None:
    assert (
        "Apache License\n                           Version 2.0" in (PDFJS / "LICENSE").read_text()
    )
