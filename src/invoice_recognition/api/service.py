"""The reading service: a queue of jobs and the workers that read them in the background.

One service per app, started and stopped with it. It owns what every document shares:

- the store of jobs (at most `MAX_DOCUMENTS` unfinished, the newest `JOBS_KEPT` kept);
- the queue, read by `READING_AT_ONCE` workers; a job waits there as "queued";
- the lane, the one thread for the local work on files (opening, rendering, measuring
  pages: pdfium must never be called from two threads at once), shared by the flow and
  the upload checks, so the event loop stays free to answer status requests;
- the limit on model requests in flight, shared by every document being read.

A worker never lets a job end unfinished: whatever the flow or the building of the view
raises, the job ends as "failed" with the code internal_error, and the worker takes the
next one. Only a cancellation (the app stopping) ends a worker.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import httpx

from invoice_recognition.api import views
from invoice_recognition.api.jobs import FINAL, Job, JobStore, Status, now
from invoice_recognition.api.settings import Settings
from invoice_recognition.recognition import pipeline
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.results import PageResult

logger = logging.getLogger(__name__)

LANE = "lane"
"""The name of the lane's thread (with a number after it)."""


def config(settings: Settings) -> Config:
    """The flow's settings for every document. A page that fails its checks is read once
    more by the same model, as in every measured run."""
    return Config(
        page_models=(settings.page_model, settings.page_model),
        header_model=settings.header_model,
        layout_model=settings.layout_model,
        deadline_s=settings.deadline_s,
        # The cap the upload checks hold an image to holds the images of a scanned page too.
        max_image_pixels=settings.max_image_pixels,
    )


_FINAL: dict[str, Status] = {s: s for s in FINAL}


def _final(status: str) -> Status:
    """The status of a finished job, from the result's."""
    if status not in _FINAL:
        raise ValueError(f"the flow returned an unknown status {status!r}")
    return _FINAL[status]


class Service:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.client = client
        self.store = JobStore(settings.max_documents, settings.jobs_kept)
        self.config = config(settings)
        self.lane = ThreadPoolExecutor(max_workers=1, thread_name_prefix=LANE)
        self.request_slots = asyncio.Semaphore(settings.model_requests_in_flight)
        self._readers = settings.reading_at_once
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        # Strong references: the event loop keeps only weak ones to its tasks.
        self._workers: list[asyncio.Task[None]] = []

    def start(self) -> None:
        """Start the workers. Call from the running event loop."""
        self._workers = [
            asyncio.create_task(self._work(), name=f"reader-{i}") for i in range(self._readers)
        ]

    async def stop(self) -> None:
        """Cancel the workers, wait for them, then shut the lane down. A document that is
        being prepared on the lane is finished there first; the event loop is not blocked
        while that happens. Jobs not finished stay as they are: the process is ending."""
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        await asyncio.to_thread(self.lane.shutdown, wait=True, cancel_futures=True)

    def submit(self, job: Job) -> None:
        """Add the job to the store and the queue. Raises jobs.StoreFull when as many
        documents as allowed are already waiting or being read."""
        self.store.add(job)
        self._queue.put_nowait(job)

    async def _work(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self.read(job)
            finally:
                self._queue.task_done()

    async def read(self, job: Job) -> None:
        """Read one job and record how it ended. Raises only when cancelled."""
        job.status, job.started_at = "processing", now()
        done: set[int] = set()

        def on_page(page: PageResult) -> None:
            # Called on this thread (the flow's async part runs on the loop). A page is
            # counted once, however often it is announced, so the count only grows.
            done.add(page.page)
            job.pages_done = min(len(done), job.pages_total)

        try:
            result = await pipeline.process(
                job.data,
                job.mime,
                self.client,
                self.config,
                on_page,
                lane=self.lane,
                request_slots=self.request_slots,
            )
            # Inside the same boundary: a result the view cannot show fails this job now,
            # never a later status request.
            view = views.result_view(result)
            error = views.job_error(job.kind, result)
            status = _final(result.status)
        except Exception:  # fault boundary: a bug must not lose the job or stop the worker
            logger.exception("reading job %s failed", job.id)
            # No result: what the flow returned, if anything, cannot be trusted or shown.
            job.result, job.view, job.error = None, None, views.internal_error()
            job.finished_at, job.status = now(), "failed"
        else:
            job.result, job.view, job.error = result, view, error
            job.finished_at, job.status = now(), status
