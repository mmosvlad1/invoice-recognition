"""The output of the flow on fixed scenarios, pinned against the research prototype.

Each file in tests/fixtures/golden holds what the prototype's process() returned on one
scenario of golden_cases.py (its Result JSON without the clock) and the requests it sent,
in order. The files are frozen reference output: they were produced once, by the earlier
research prototype, and are not regenerated from this repository. The new process() must
return the same and send the same, to the byte of every prompt and attached PDF. Each
scenario runs twice: plain, and served - with the arguments the API passes to process()
(the local work on a lane of its own, a limit on model requests shared with other
documents, an overall deadline far beyond the scenario).

The texts of the photo_ and scan_ issues were reworded when the flow was ported (full
sentences); the files carry the new wording, with the numbers the prototype produced. A
deliberate change of behaviour after the port updates the files by hand, in its own
commit.

Left out of the comparison, on purpose:
- the clock (duration_s of the document and of every attempt);
- text for the user that was translated when the flow was ported (the message of a
  quality reason, the reason of a carried tail without an owner, the text of a
  page_without_text note): it is not in the golden files; codes, pages and kinds are;
- numbers measured on an image (quality measures, the value of a quality reason and the
  numbers inside the message of a photo_ / scan_ issue) are compared within 5 %: opencv
  may compute them slightly differently on another platform;
- an attached image is compared by its size (within 2 px): its JPEG bytes depend on the
  platform's image libraries. When a drawn input itself comes out with other bytes than
  when the golden file was written, attached PDFs made from it are compared by name only.
"""

import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest

from invoice_recognition.recognition import pipeline
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.results import Result

from golden_cases import CASES, GOLDEN, Case, Model, request_of, result_of, sha256

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _close(actual: float, expected: float) -> bool:
    return actual == pytest.approx(expected, rel=0.05, abs=0.05)


def _same_text(actual: str, expected: str) -> bool:
    """The same words, and numbers within 5 %."""
    if _NUMBER.sub("#", actual) != _NUMBER.sub("#", expected):
        return False
    pairs = zip(_NUMBER.findall(actual), _NUMBER.findall(expected), strict=True)
    return all(_close(float(a), float(e)) for a, e in pairs)


def compare(actual: Any, expected: Any, path: str = "", loose: bool = False) -> list[str]:
    """Every difference between two JSON values, as "path: actual != expected"."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(actual) != set(expected):
            return [f"{path}: keys {sorted(actual)} != {sorted(expected)}"]
        measured_issue = str(expected.get("check", "")).startswith(("photo_", "scan_"))
        out = []
        for k in expected:
            inner = loose or k == "measures" or (k == "message" and measured_issue)
            out += compare(actual[k], expected[k], f"{path}.{k}", inner)
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            return [f"{path}: {len(actual)} items != {len(expected)}"]
        return [
            d
            for i, (a, e) in enumerate(zip(actual, expected, strict=True))
            for d in compare(a, e, f"{path}[{i}]", loose)
        ]
    numbers = (int, float)
    flags = isinstance(expected, bool) or isinstance(actual, bool)
    if loose and isinstance(expected, numbers) and isinstance(actual, numbers) and not flags:
        return [] if _close(actual, expected) else [f"{path}: {actual} != {expected}"]
    if loose and isinstance(expected, str) and isinstance(actual, str):
        return [] if _same_text(actual, expected) else [f"{path}: {actual!r} != {expected!r}"]
    if actual != expected or type(actual) is not type(expected):
        return [f"{path}: {actual!r} != {expected!r}"]
    return []


def comparable(result: dict[str, Any]) -> dict[str, Any]:
    """The result without the text for the user that was translated in the port, and with
    the value of every quality reason marked as measured."""
    out: dict[str, Any] = json.loads(json.dumps(result))
    quality = out.get("quality") or {}
    reasons = list(quality.get("reasons", []))
    for page in quality.get("pages", []):
        reasons += page.get("reasons", [])
    for r in reasons:
        r.pop("message", None)
        if "value" in r:
            r["measures"] = r.pop("value")  # compared within the tolerance of a measure
    for note in out.get("notes", []):
        note.pop("reason", None)
        if note.get("source") == "scan_quality":
            note.pop("text", None)
    return out


def comparable_requests(bodies: list[dict[str, Any]], exact_files: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = json.loads(json.dumps(bodies))
    for body in out:
        for part in body["messages"][0]["content"]:
            if part["type"] == "file" and not exact_files:
                part.pop("sha256")
            if part["type"] == "image_url":
                part["measures"] = {"width": part["size"][0], "height": part["size"][1]}
                del part["size"]
    return out


def run_case(case: Case, served: bool = False) -> tuple[bytes, Result, Model]:
    """process() on the scenario. served: called the way the API calls it, with the local
    work on a lane of its own, a limit on requests shared with other documents and a far
    deadline."""
    data, mime = case.data()
    model = case.model()

    async def go() -> Result:
        async with httpx.AsyncClient(transport=httpx.MockTransport(model)) as client:
            if not served:
                return await pipeline.process(data, mime, client, Config(**case.config))
            with ThreadPoolExecutor(1) as lane:
                return await pipeline.process(
                    data,
                    mime,
                    client,
                    # Not in case.config: the golden file pins that config.
                    Config(**case.config, deadline_s=240.0),
                    lane=lane,
                    request_slots=asyncio.Semaphore(24),
                )

    return data, asyncio.run(go()), model


@pytest.mark.parametrize("served", [False, True], ids=["plain", "served"])
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_flow_output_is_the_one_pinned(case: Case, served: bool) -> None:
    golden = json.loads((GOLDEN / f"{case.name}.json").read_text())
    assert golden["config"] == case.config
    data, res, model = run_case(case, served)
    exact_files = sha256(data) == golden["input_sha256"]

    differences = compare(
        comparable(result_of(res.to_json())), comparable(golden["result"]), "result"
    )
    sent = comparable_requests([request_of(b) for b in model.bodies], exact_files)
    differences += compare(sent, comparable_requests(golden["requests"], exact_files), "requests")
    assert differences == []


def test_every_golden_file_has_a_scenario() -> None:
    assert sorted(p.stem for p in GOLDEN.glob("*.json")) == sorted(c.name for c in CASES)


def test_comparison_sees_a_difference() -> None:
    golden = {"a": [1, {"b": "x"}], "measures": {"m": 100.0}, "check": "scan_dark"}
    assert compare(golden, golden) == []
    assert compare({**golden, "a": [1, {"b": "y"}]}, golden) == [".a[1].b: 'y' != 'x'"]
    assert compare({**golden, "measures": {"m": 103.0}}, golden) == []  # within 5 %
    assert compare({**golden, "measures": {"m": 110.0}}, golden) != []
    assert (
        compare(
            {"message": "sharpness 101.0 is below 100"}, {"message": "sharpness 100.0 is below 100"}
        )
        != []
    )
    issue = {"check": "scan_blurred", "message": "sharpness 101.0 is below 100"}
    assert compare(issue, issue | {"message": "sharpness 100.0 is below 100"}) == []
    assert compare(issue, issue | {"message": "sharpness 100.0 is above 100"}) != []
    assert compare(1, 1.0) != []  # an int is not a float in the JSON of a result
