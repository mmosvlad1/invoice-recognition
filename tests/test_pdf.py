"""Picking pages out of a PDF, encrypted ones too."""

import io

import pypdfium2
import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.errors import FileNotDecryptedError
from pypdf.generic import DictionaryObject

from invoice_recognition.recognition import evidence as ev
from invoice_recognition.recognition.pdf import extract_pages, page_count

from fakes import Answer, Server, make_pdf, run
from samples import INVOICES, saved_layout, truth_doc

THREE = [["Page one"], ["Page two"], ["Page three"]]


def texts(data: bytes) -> list[str]:
    return [page.extract_text() for page in PdfReader(io.BytesIO(data)).pages]


def test_page_count() -> None:
    assert page_count(make_pdf(THREE)) == 3
    assert page_count(make_pdf([["Only page"]])) == 1


def test_extract_one_page_by_its_1_based_number() -> None:
    data = make_pdf(THREE)
    for number, text in enumerate(("Page one", "Page two", "Page three"), start=1):
        page = extract_pages(data, [number])
        assert page_count(page) == 1
        assert texts(page) == [text]


def test_pages_come_out_in_the_order_asked() -> None:
    data = make_pdf(THREE)
    assert texts(extract_pages(data, [3, 1])) == ["Page three", "Page one"]
    assert texts(extract_pages(data, [1, 2, 3])) == ["Page one", "Page two", "Page three"]


def test_page_beyond_the_end_is_an_error() -> None:
    with pytest.raises(IndexError):
        extract_pages(make_pdf(THREE), [4])


@pytest.mark.parametrize("pages", [[0], [-1], [1, 0]])
def test_page_below_1_is_an_error_not_a_page_from_the_end(pages: list[int]) -> None:
    with pytest.raises(IndexError, match="page 0|page -1"):
        extract_pages(make_pdf(THREE), pages)


def test_extracted_bytes_are_deterministic() -> None:
    # The flow sends each page as its own PDF: the same document must give the
    # same request body every time.
    data = make_pdf(THREE)
    assert extract_pages(data, [2]) == extract_pages(data, [2])
    assert extract_pages(data, [1, 3]) == extract_pages(make_pdf(THREE), [1, 3])


# --------------------------------------------------------------------------- #
# Encrypted PDFs
# --------------------------------------------------------------------------- #


def encrypted(user_password: str) -> bytes:
    """s_classic.pdf encrypted with AES-256. user_password "": anyone may open it (only the
    owner's password, not needed to read, is set), as many generated invoices are."""
    writer = PdfWriter(clone_from=PdfReader(INVOICES / "s_classic.pdf"))
    writer.encrypt(user_password=user_password, owner_password="owner", algorithm="AES-256")
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_aes_pdf_without_a_password_to_open_is_read() -> None:
    data = encrypted("")
    encryption = PdfReader(io.BytesIO(data)).trailer["/Encrypt"].get_object()
    assert isinstance(encryption, DictionaryObject)
    assert (str(encryption["/V"]), str(encryption["/R"])) == ("5", "6")  # AES-256

    assert page_count(data) == 1
    assert texts(extract_pages(data, [1])) == texts((INVOICES / "s_classic.pdf").read_bytes())
    assert ev.has_text_layer(ev.pdf_text_layer(data))  # pdfium opens it too

    _, _, header, pages = truth_doc("s_classic")
    answers: dict[int, list[Answer]] = {p: [rows] for p, rows in pages.items()}
    res = run(data, "application/pdf", Server(header, answers, saved_layout("s_classic")))
    assert res.method == "layout" and res.status == "ok"


def test_pdf_with_a_password_to_open_is_refused() -> None:
    data = encrypted("user")
    with pytest.raises(FileNotDecryptedError):
        page_count(data)
    with pytest.raises(pypdfium2.PdfiumError, match="password"):
        ev.pdf_text_layer(data)
