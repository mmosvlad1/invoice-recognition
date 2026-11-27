"""The page's samples: real files that pass the upload checks, and what the demo's fake
model makes of them through the real API. The small one has one amount misprinted on
purpose, so the review screen has something to show."""

import io
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from pypdf import PdfReader

from invoice_recognition.api import uploads
from invoice_recognition.api.page import STATIC_DIR
from invoice_recognition.recognition.config import MAX_IMAGE_PIXELS
from invoice_recognition.shapes import JSONObject

import demo_server
from fakes import api_client, wait_final
from samples import INVOICES

SAMPLES = STATIC_DIR / "samples"
SMALL = SAMPLES / "invoice-small-misprint.pdf"
LARGE = SAMPLES / "invoice-large.pdf"

LIMITS = uploads.UploadLimits(
    max_file_bytes=10 * 1024 * 1024, max_pages=50, max_image_pixels=MAX_IMAGE_PIXELS
)


def page_texts(data: bytes) -> list[str]:
    return [page.extract_text() for page in PdfReader(io.BytesIO(data)).pages]


@pytest.mark.parametrize(
    ("name", "kind", "pages"),
    [
        ("invoice-small-misprint.pdf", "pdf", 4),
        ("invoice-large.pdf", "pdf", 34),
        ("invoice-scan-simulated.pdf", "pdf", 4),
        ("invoice-photo-simulated.jpg", "image", 1),
    ],
)
def test_every_sample_passes_the_upload_checks(name: str, kind: str, pages: int) -> None:
    checked = uploads.inspect((SAMPLES / name).read_bytes(), LIMITS)
    assert (checked.kind, checked.pages_total) == (kind, pages)


def test_the_small_sample_differs_from_its_invoice_by_one_amount() -> None:
    printed = page_texts((INVOICES / "m_grouped.pdf").read_bytes())
    misprinted = page_texts(SMALL.read_bytes())
    assert misprinted[0] == printed[0] and misprinted[2:] == printed[2:]
    assert printed[1].count("688.68") == 1 and "344.34" not in printed[1]
    assert misprinted[1] == printed[1].replace("688.68", "344.34")


@pytest.fixture
async def demo() -> AsyncIterator[httpx.AsyncClient]:
    """A client of the demo app, started, without an access code."""
    app = demo_server.demo_app(delay=0, access_code="")
    async with app.router.lifespan_context(app), api_client(app) as client:
        yield client


async def read(client: httpx.AsyncClient, path: Path) -> JSONObject:
    answer = await client.post("/invoices", files={"file": (path.name, path.read_bytes())})
    assert answer.status_code == 202, answer.text
    return await wait_final(client, answer.json()["id"])


SUGGESTIONS = [
    {"set": {"line.quantity": 9}, "source": "arithmetic"},
    {"set": {"line.unit_price": 19.13}, "source": "arithmetic"},
    {"set": {"line.amount": 688.68}, "source": "arithmetic", "agrees_with": "header.subtotal"},
]


@pytest.mark.anyio
async def test_the_small_sample_has_one_line_to_check_and_what_follows_from_it(
    demo: httpx.AsyncClient,
) -> None:
    job = await read(demo, SMALL)
    result = job["result"]
    assert job["status"] == "needs_review"
    assert [p["method"] for p in result["pages"]] == ["layout"] * 4
    issues = result["issues"]
    assert [(i["id"], i["code"], i["page"], i["follows_from"]) for i in issues] == [
        (0, "line_math", 2, []),
        (1, "coverage", 2, []),
        (2, "doc_sums", None, [0]),
        (3, "tax", None, [0]),
        (4, "checkpoints", 3, [0]),
    ]
    misread = result["lines"][issues[0]["line_index"]]
    assert (misread["page"], misread["quantity"], misread["unit_price"]) == (2, 18, 38.26)
    assert misread["amount"] == 344.34
    assert issues[0]["suggestions"] == SUGGESTIONS
    assert "header.subtotal" in issues[2]["fields"]
    assert "7%" in issues[3]["message"]
    # Where the flow found each value on page 2: the cells the page frames.
    boxes = misread["boxes"]
    for key in ("row", "quantity", "unit_price", "amount"):
        x0, y0, x1, y1 = boxes[key]
        assert 0 <= x0 < x1 <= 595.3 and 0 <= y0 < y1 <= 841.9, key
    assert boxes["quantity"][0] > boxes["unit_price"][0]  # the columns: price, then quantity
    assert result["pages"][1] | {"line_count": 0} == {
        "page": 2,
        "method": "layout",
        "line_count": 0,
        "repaired_count": 0,
        "width": 595.3,
        "height": 841.9,
        "text_layer": True,
    }
    assert all(line["boxes"] is not None for line in result["lines"])


@pytest.mark.anyio
async def test_the_large_sample_adds_up(demo: httpx.AsyncClient) -> None:
    job = await read(demo, LARGE)
    assert job["status"] == "ok"
    assert len(job["result"]["lines"]) == 1250
    assert job["result"]["issues"] == []


@pytest.mark.anyio
async def test_the_scan_sample_is_framed_by_its_pages_only(demo: httpx.AsyncClient) -> None:
    job = await read(demo, SAMPLES / "invoice-scan-simulated.pdf")
    assert job["status"] == "ok"
    assert all(line["boxes"] is None for line in job["result"]["lines"])
    assert [p["text_layer"] for p in job["result"]["pages"]] == [False] * 4


@pytest.mark.anyio
async def test_a_file_the_demo_does_not_know_fails_as_if_the_provider_were_down(
    demo: httpx.AsyncClient,
) -> None:
    job = await read(demo, INVOICES / "s_classic.pdf")
    assert (job["status"], job["error"]["code"]) == ("failed", "model_unavailable")
