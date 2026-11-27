"""Picking pages out of a PDF."""

import io

import pytest
from pypdf import PdfReader

from invoice_recognition.recognition.pdf import extract_pages, page_count

from fakes import make_pdf

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
