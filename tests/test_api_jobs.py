"""Jobs in memory: the cap on unfinished ones, retention, and how a job is shown."""

import json
from datetime import UTC, datetime

import pytest

from invoice_recognition.api.jobs import MAX_FILENAME, Job, JobStore, Status, StoreFull
from invoice_recognition.api.review import decide
from invoice_recognition.api.views import (
    AcceptRequest,
    JobError,
    JobItem,
    JobView,
    job_item,
    job_view,
    result_view,
)
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.results import PageResult, Result

from fakes import header_for, line


def job(name: str = "invoice.pdf", status: Status = "queued") -> Job:
    j = Job(name, "pdf", "application/pdf", b"%PDF-1.7 ...", pages_total=3)
    j.status = status
    return j


def read(*nos: int) -> Result:
    lines = [line(n) | {"page": 1} for n in nos]
    return Result(
        status="ok",
        header=header_for(lines),
        lines=lines,
        checkpoints=[],
        issues=[],
        pages=[PageResult(1, lines)],
        evidence_source=None,
        duration_s=1.0,
        calls=2,
        cost_usd=0.002,
        prompt_tokens=200,
        completion_tokens=100,
    )


def test_a_new_job_is_queued_with_an_id_and_the_time_in_utc() -> None:
    j = job()
    assert len(j.id) == 32 and int(j.id, 16) >= 0
    assert j.id != job().id
    assert j.status == "queued" and not j.finished
    assert j.created_at.tzinfo is UTC and j.created_at.microsecond == 0
    assert (j.started_at, j.finished_at, j.result, j.view, j.error) == (None,) * 5
    assert j.decisions == {}
    assert j.size_bytes == len(b"%PDF-1.7 ...")


@pytest.mark.parametrize(
    ("sent", "kept"),
    [
        ("../../etc/\u202efdp.exe", "fdp.exe"),
        ("C:\\Users\\me\\invoice.pdf", "invoice.pdf"),
        ("a\u200bb\u2066c.pdf", "abc.pdf"),
        ("folder/", ""),
        ("Рахунок №7.pdf", "Рахунок №7.pdf"),
    ],
)
def test_a_file_name_keeps_its_last_part_without_format_characters(sent: str, kept: str) -> None:
    assert job(sent).filename == kept


def test_a_long_file_name_is_cut_to_255_characters() -> None:
    assert job("é" * 300).filename == "é" * MAX_FILENAME
    html = "<img src=x onerror=alert(1)>.pdf"
    assert job(html).filename == html  # text, kept as sent: whoever shows it escapes it


def test_a_file_name_that_cannot_be_utf8_is_made_safe_once() -> None:
    j = job("rechnung-\udcff.pdf")  # a lone surrogate, as a decoder may leave it
    assert j.filename == "rechnung-?.pdf"
    assert json.loads(job_view(j).model_dump_json())["filename"] == "rechnung-?.pdf"
    assert job("\udcff" * 300).filename == "?" * MAX_FILENAME


def test_the_eleventh_unfinished_job_is_refused_until_one_finishes() -> None:
    store = JobStore(max_unfinished=10, keep=50)
    jobs = [job() for _ in range(10)]
    for j in jobs:
        store.add(j)
    jobs[3].status = "processing"
    assert not store.has_room()
    with pytest.raises(StoreFull):
        store.add(job())
    jobs[7].status = "failed"
    assert store.has_room()
    store.add(job())
    assert len(store.unfinished()) == 10


def test_retention_drops_the_oldest_finished_never_an_unfinished_one() -> None:
    store = JobStore(max_unfinished=3, keep=3)
    waiting, done_1, done_2 = job(), job(status="ok"), job(status="failed")
    for j in (waiting, done_1, done_2):
        store.add(j)
    newest = job()
    store.add(newest)
    assert store.newest_first() == [newest, done_2, waiting]
    assert store.get(done_1.id) is None and store.get(waiting.id) is waiting
    later = job(status="needs_review")
    store.add(later)
    assert store.newest_first() == [later, newest, waiting]


def test_the_list_is_newest_first() -> None:
    store = JobStore(max_unfinished=5, keep=5)
    jobs = [job(f"{n}.pdf") for n in range(4)]
    for j in jobs:
        store.add(j)
    assert store.newest_first() == jobs[::-1]
    assert store.unfinished() == jobs
    assert store.get("0" * 32) is None


def test_the_store_needs_room_for_its_unfinished_jobs() -> None:
    with pytest.raises(ValueError, match="max_unfinished <= keep"):
        JobStore(max_unfinished=10, keep=9)


def test_a_job_is_shown_with_progress_and_times_ending_in_z() -> None:
    j = job("factura nº 7.pdf")
    j.created_at = datetime(2026, 11, 27, 14, 2, 11, tzinfo=UTC)
    j.started_at = datetime(2026, 11, 27, 14, 2, 12, tzinfo=UTC)
    j.status, j.pages_done = "processing", 1
    out = json.loads(job_view(j).model_dump_json())
    assert out == {
        "id": j.id,
        "filename": "factura nº 7.pdf",
        "kind": "pdf",
        "size_bytes": 12,
        "status": "processing",
        "created_at": "2026-11-27T14:02:11Z",
        "started_at": "2026-11-27T14:02:12Z",
        "finished_at": None,
        "progress": {"pages_total": 3, "pages_done": 1},
        "error": None,
        "result": None,
        "review": None,
    }
    item = json.loads(job_item(j).model_dump_json())
    assert "result" not in item and "review" not in item
    assert (item["line_count"], item["issue_count"], item["open_issue_count"]) == (None,) * 3
    assert item["created_at"] == "2026-11-27T14:02:11Z"


def test_a_finished_job_carries_its_result_and_its_counts() -> None:
    j = job(status="ok")
    j.result = read(1, 2, 3)
    j.view = result_view(j.result)
    j.finished_at = datetime(2026, 11, 27, 14, 3, 0, tzinfo=UTC)
    out = json.loads(job_view(j).model_dump_json())
    assert out["finished_at"] == "2026-11-27T14:03:00Z"
    assert [item["index"] for item in out["result"]["lines"]] == [0, 1, 2]
    assert out["review"] == {"open_count": 0, "issues": []}
    item = json.loads(job_item(j).model_dump_json())
    assert (item["line_count"], item["issue_count"], item["open_issue_count"]) == (3, 0, 0)


def test_a_document_with_issues_shows_where_their_review_stands() -> None:
    j = job(status="needs_review")
    res = read(1, 2, 3)
    res.status = "needs_review"
    res.issues = [
        Issue(
            "line_math",
            "2 x 10.0 = 20.00, but amount is 99.0",
            line=1,
            suggestions=[{"set": {"line.amount": 20}, "source": "arithmetic"}],
        ),
        Issue("doc_sums", "lines add up", follows_from=[0]),
        Issue("coverage", "page 1: 2 item rows printed, 3 extracted", page=1),
    ]
    j.result, j.view = res, result_view(res)
    decide(j, 0, AcceptRequest(action="accept", suggestion=0), datetime(2026, 11, 27, tzinfo=UTC))
    out = json.loads(job_view(j).model_dump_json())
    assert out["review"] == {
        "open_count": 1,
        "issues": [
            {
                "id": 0,
                "state": "accepted",
                "suggestion": 0,
                "set": {"line.amount": 20},
                "decided_at": "2026-11-27T00:00:00Z",
                "settled_by": [],
            },
            {
                "id": 1,
                "state": "settled",
                "suggestion": None,
                "set": None,
                "decided_at": None,
                "settled_by": [0],
            },
            {
                "id": 2,
                "state": "open",
                "suggestion": None,
                "set": None,
                "decided_at": None,
                "settled_by": [],
            },
        ],
    }
    # The reading's verdict and counts stay as they were.
    assert out["status"] == "needs_review" and len(out["result"]["issues"]) == 3
    item = json.loads(job_item(j).model_dump_json())
    assert (item["issue_count"], item["open_issue_count"]) == (3, 1)


def test_a_job_that_failed_with_internal_error_has_no_result_and_no_counts() -> None:
    j = job(status="failed")
    j.error = JobError(code="internal_error", message="Something went wrong.")
    view = json.loads(job_view(j).model_dump_json())
    assert view["result"] is None and view["review"] is None
    assert view["error"] == {"code": "internal_error", "message": "Something went wrong."}
    item = json.loads(job_item(j).model_dump_json())
    assert (item["line_count"], item["issue_count"], item["open_issue_count"]) == (None,) * 3


def test_a_failed_job_with_a_result_has_no_review() -> None:
    j = job(status="failed")
    j.result = read(1)
    j.result.status = "failed"
    j.view = result_view(j.result)
    assert job_view(j).result is not None and job_view(j).review is None
    assert job_item(j).open_issue_count is None


def test_no_result_and_no_error_are_shown_before_the_job_is_finished() -> None:
    j = job(status="processing")
    j.view = result_view(read(1))
    j.error = JobError(code="internal_error", message="Something went wrong.")
    assert job_view(j).result is None and job_view(j).error is None
    assert job_view(j).review is None and job_item(j).open_issue_count is None
    assert job_item(j).line_count is None and job_item(j).error is None


def test_every_field_sent_is_required_in_the_serialisation_schema() -> None:
    schema = JobView.model_json_schema(mode="serialization")
    defs = schema["$defs"]
    assert set(schema["required"]) == {
        "id",
        "filename",
        "kind",
        "size_bytes",
        "status",
        "created_at",
        "started_at",
        "finished_at",
        "progress",
        "error",
        "result",
        "review",
    }
    assert set(defs["ReviewView"]["required"]) == {"open_count", "issues"}
    assert set(defs["IssueReviewView"]["required"]) == {
        "id",
        "state",
        "suggestion",
        "set",
        "decided_at",
        "settled_by",
    }
    assert {"index", "line_no", "sku", "quantity", "amount"} <= set(defs["LineView"]["required"])
    assert len(defs["LineView"]["required"]) == 11  # with boxes
    assert {"fields", "suggestions", "follows_from", "line_index"} <= set(
        defs["IssueView"]["required"]
    )
    assert set(defs["NoteView"]["required"]) == {"kind", "page", "text"}
    item = JobItem.model_json_schema(mode="serialization")
    assert {"line_count", "issue_count", "open_issue_count"} <= set(item["required"])
    assert "result" not in item["properties"]
