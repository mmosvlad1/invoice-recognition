"""Jobs and the store that holds them: in the memory of the one process, nowhere else.

A restart loses every job. The store is used only from the event loop's thread (the
routes and the workers run there), so it needs no lock; the flow's local work on its
own thread never touches it.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from invoice_recognition.api.views import JobError, ResultView
from invoice_recognition.recognition.results import Result

type Kind = Literal["pdf", "image"]
type Status = Literal["queued", "processing", "ok", "needs_review", "failed"]

FINAL: tuple[Status, ...] = ("ok", "needs_review", "failed")

# A file name is what the client sent: untrusted text. It is kept to 255 characters;
# no real file name is longer (file systems allow 255 bytes or UTF-16 units).
MAX_FILENAME = 255


def safe_filename(name: str) -> str:
    """The name as text that can always be sent as UTF-8 (a lone surrogate, which a
    client can send percent-encoded, becomes "?"), cut to MAX_FILENAME characters. It
    is never interpreted: whoever shows it escapes it."""
    return name.encode("utf-8", "replace").decode("utf-8")[:MAX_FILENAME]


def now() -> datetime:
    """The time in UTC, to the second (the API shows no finer)."""
    return datetime.now(UTC).replace(microsecond=0)


@dataclass
class Job:
    filename: str
    kind: Kind
    mime: str
    data: bytes
    pages_total: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status: Status = "queued"
    pages_done: int = 0
    result: Result | None = None
    view: ResultView | None = None  # the result as the API shows it, built once
    error: JobError | None = None

    def __post_init__(self) -> None:
        self.filename = safe_filename(self.filename)

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def finished(self) -> bool:
        return self.status in FINAL


class StoreFull(Exception):
    """As many jobs as allowed are waiting or being read."""


class JobStore:
    """At most `max_unfinished` jobs queued or being read; of all jobs, the newest `keep`
    are kept. A job that is not finished is never dropped."""

    def __init__(self, max_unfinished: int, keep: int) -> None:
        if not 1 <= max_unfinished <= keep:
            raise ValueError("need 1 <= max_unfinished <= keep")
        self.max_unfinished, self.keep = max_unfinished, keep
        self._jobs: dict[str, Job] = {}  # in the order they were added

    def has_room(self) -> bool:
        return len(self.unfinished()) < self.max_unfinished

    def add(self, job: Job) -> None:
        """Raises StoreFull when no job more may wait. Then drops the oldest finished
        jobs until no more than `keep` are held."""
        if not self.has_room():
            raise StoreFull
        self._jobs[job.id] = job
        while len(self._jobs) > self.keep:
            oldest = next((j for j in self._jobs.values() if j.finished), None)
            if oldest is None:  # cannot happen while keep >= max_unfinished
                break
            del self._jobs[oldest.id]

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def newest_first(self) -> list[Job]:
        return list(reversed(self._jobs.values()))

    def unfinished(self) -> list[Job]:
        return [j for j in self._jobs.values() if not j.finished]
