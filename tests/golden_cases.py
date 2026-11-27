"""The scenarios whose output is pinned in tests/fixtures/golden, and the fake model server
that answers them.

Each scenario is an input file, the answers a fake model gives to it and settings of the
flow. The golden files hold what `process()` returned and which requests it sent, in
order. This module imports nothing from the package (and runs on older Pythons too), so
the same scenarios can be run by another implementation of the flow and compared.
"""

import base64
import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageFilter

from drawn import PHOTO_H, SHEET, photo, png, scan, scan_with_ocr_text, scanned
from samples import INVOICES, saved_layout, truth_doc

GOLDEN = Path(__file__).parent / "fixtures" / "golden"


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #


@dataclass
class NoCost:
    """An answer whose usage reports no cost."""

    payload: Any


@dataclass
class Status:
    """An HTTP error instead of an answer."""

    code: int
    text: str = "upstream error"


def sse(text: str, cost: float | None = 0.001) -> str:
    usage = {"prompt_tokens": 100, "completion_tokens": 50} | (
        {"cost": cost} if cost is not None else {}
    )
    events = [
        {"choices": [{"delta": {"content": text}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage},
    ]
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def _response(answer: Any) -> httpx.Response:
    if isinstance(answer, BaseException):
        raise answer
    if isinstance(answer, Status):
        return httpx.Response(answer.code, text=answer.text)
    if isinstance(answer, str):
        return httpx.Response(200, text=answer)
    cost: float | None = 0.001
    if isinstance(answer, NoCost):
        answer, cost = answer.payload, None
    payload = {"rows": answer} if isinstance(answer, list) else answer
    return httpx.Response(200, text=sse(json.dumps(payload), cost))


_PAGE_OF = re.compile(r"page (\d+) of \d+")


class Model:
    """Answers the requests of the flow by their kind: header, layout, or a page by its
    number. Each list holds the answers to the attempts in turn; the last one repeats.

    models: {model: handler} - a request to one of these models goes to its handler, by
    the model named in the body, and is neither recorded in bodies nor counted in calls:
    the requests of the scenario's own model stay what they are without it."""

    def __init__(
        self,
        header: list[Any],
        pages: dict[int, list[Any]],
        layout: Any = None,
        *,
        models: dict[str, Callable[[httpx.Request], httpx.Response]] | None = None,
    ) -> None:
        self.header, self.pages, self.layout = header, pages, layout
        self.models = dict(models or {})
        self.calls: Counter[Any] = Counter()
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        other = self.models.get(body["model"])
        if other is not None:
            return other(request)
        self.bodies.append(body)
        prompt = body["messages"][0]["content"][0]["text"]
        if prompt.startswith("You read ONE page"):
            m = _PAGE_OF.search(prompt)
            assert m is not None
            key: Any = int(m.group(1))
            answers = self.pages[key]
        elif prompt.startswith("You are given the text layer"):
            key, answers = "layout", [self.layout]
        else:
            key, answers = "header", self.header
        answer = answers[min(self.calls[key], len(answers) - 1)]
        self.calls[key] += 1
        return _response(answer)


def line(no: int, qty: int = 2, price: float = 10.0, amount: float | None = None) -> dict[str, Any]:
    return {
        "line_no": no,
        "sku": f"SKU-{no:03d}",
        "description": f"Item {no}",
        "quantity": qty,
        "unit": "pcs",
        "unit_price": price,
        "tax_rate": 20,
        "amount": qty * price if amount is None else amount,
    }


def header_for(total_of: list[dict[str, Any]], **change: Any) -> dict[str, Any]:
    subtotal = round(sum(item["amount"] for item in total_of), 2)
    tax = round(subtotal * 0.2, 2)
    header = {
        "supplier_name": "ACME Ltd",
        "invoice_number": "INV-1",
        "currency": "EUR",
        "subtotal": subtotal,
        "tax_total": tax,
        "total": round(subtotal + tax, 2),
    }
    return header | change


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


@dataclass
class Case:
    name: str
    data: Callable[[], tuple[bytes, str]]
    model: Callable[[], Model]
    config: dict[str, Any] = field(default_factory=dict)


def _pdf(name: str) -> Callable[[], tuple[bytes, str]]:
    return lambda: ((INVOICES / name).read_bytes(), "application/pdf")


def _truth_model(name: str, layout: Any = None) -> Model:
    _, _, header, pages = truth_doc(name)
    return Model([header], {p: [rows] for p, rows in pages.items()}, layout)


def _swapped(layout: dict[str, Any]) -> dict[str, Any]:
    flip = {"quantity": "unit_price", "unit_price": "quantity"}
    return layout | {
        "columns": [c | {"field": flip.get(c["field"], c["field"])} for c in layout["columns"]]
    }


def _split_model() -> Model:
    """m_split read by a model that puts the end of every cut description into "carry_in"."""
    _, t, header, pages = truth_doc("m_split")
    tails = {}
    for s in t["split_lines"]:
        row = next(r for r in pages[s["page"]] if r["line_no"] == s["line_no"])
        words = row["description"].split()
        row["description"], tails[s["split_to_page"]] = " ".join(words[:-4]), " ".join(words[-4:])
    answers = {p: [{"carry_in": tails.get(p, ""), "rows": rows}] for p, rows in pages.items()}
    return Model([header], answers)


def _scan_retries() -> Model:
    p1 = [line(1), line(2, amount=99.0)]  # wrong arithmetic in both attempts
    p2 = [line(3), {"description": "Section B"}, line(4)]  # one row with neither sku nor amount
    p3 = [line(5), line(6, qty=3)]
    cut = sse('{"carry_in": "", "rows": [' + json.dumps(line(5)) + ', {"line_no": 6, "sku": "SK')
    every = [line(1), line(2), line(3), line(4), *p3]
    return Model(
        [header_for(every, total=None), header_for(every)],
        {
            1: [{"carry_in": "", "rows": p1}],
            2: [{"carry_in": "zinc plated", "rows": p2}],
            3: [cut, NoCost({"carry_in": "", "rows": p3})],
        },
    )


def _scan_failures() -> Model:
    rows = [line(1), line(2)]
    return Model(
        [header_for(rows)],
        {1: [RuntimeError("stream blew up")], 2: [Status(500), {"carry_in": "", "rows": rows}]},
    )


def _photo(kind: str) -> Callable[[], tuple[bytes, str]]:
    def make() -> tuple[bytes, str]:
        img = photo()
        if kind == "soft":
            img = img.filter(ImageFilter.GaussianBlur(0.8))
        elif kind == "cut":
            img = img.crop((0, 0, SHEET[0] + 420, PHOTO_H))
        return png(img), "image/png"

    return make


def _scan_quality() -> Model:
    p1, p4 = [line(1), line(2)], [line(3)]
    return Model(
        [header_for(p1 + p4)],
        {1: [{"carry_in": "", "rows": p1}], 4: [{"carry_in": "", "rows": p4}]},
    )


def _three_rows() -> Model:
    rows = [line(1), line(2), line(3)]
    return Model([header_for(rows)], {1: [rows]})


CASES = [
    Case("scan_retries", lambda: (scan(3), "application/pdf"), _scan_retries),
    Case("scan_failures", lambda: (scan(2), "application/pdf"), _scan_failures),
    Case(
        "layout_confirmed",
        _pdf("s_dense.pdf"),
        lambda: _truth_model("s_dense", saved_layout("s_dense")),
    ),
    Case(
        "layout_swapped",
        _pdf("m_grouped.pdf"),
        lambda: _truth_model("m_grouped", _swapped(saved_layout("m_grouped"))),
    ),
    Case("layout_failed", _pdf("s_classic.pdf"), lambda: _truth_model("s_classic", Status(500))),
    Case("text_layer_carry", _pdf("m_split.pdf"), _split_model, {"use_layout": False}),
    Case("photo_soft", _photo("soft"), _three_rows),
    Case("photo_cut", _photo("cut"), lambda: Model([{}], {})),
    Case(
        "scan_quality",
        lambda: (scanned("good", "blurred", "blank", "soft"), "application/pdf"),
        _scan_quality,
    ),
    Case(
        "scan_ocr_overlay",
        lambda: (scan_with_ocr_text([line(1), line(2), line(3)]), "application/pdf"),
        _three_rows,
    ),
]


# --------------------------------------------------------------------------- #
# What is pinned
# --------------------------------------------------------------------------- #


def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def request_of(body: dict[str, Any]) -> dict[str, Any]:
    """A request body with its prompt and attachment replaced by what identifies them: the
    sha256 of the prompt (and its first line), the sha256 of an attached PDF, the size of
    an attached image (its JPEG bytes depend on the platform's image libraries)."""
    out = {k: v for k, v in body.items() if k != "messages"}
    parts = []
    for part in body["messages"][0]["content"]:
        if part["type"] == "text":
            text = part["text"]
            parts.append(
                {"type": "text", "first_line": text.split("\n", 1)[0], "sha256": sha256(text)}
            )
        elif part["type"] == "file":
            f = part["file"]
            parts.append(
                {"type": "file", "filename": f["filename"], "sha256": sha256(f["file_data"])}
            )
        else:
            raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            parts.append({"type": part["type"], "size": list(Image.open(io.BytesIO(raw)).size)})
    out["messages"] = [{"role": body["messages"][0]["role"], "content": parts}]
    return out


def result_of(result: dict[str, Any]) -> dict[str, Any]:
    """Result.to_json() without the clock: duration_s of the document and of every attempt."""

    def attempts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in a.items() if k != "duration_s"} for a in items]

    out = {k: v for k, v in result.items() if k != "duration_s"}
    out["header_attempts"] = attempts(out["header_attempts"])
    out["layout_attempts"] = attempts(out["layout_attempts"])
    out["pages"] = [p | {"attempts": attempts(p["attempts"])} for p in out["pages"]]
    return out
