# Invoice Recognition

A prototype written for a test task: a FastAPI app with a web page and an API that
turns an invoice (a digital PDF, a scan or a photo) into a header and line items. An
LLM reads the document through [OpenRouter](https://openrouter.ai/); code then
**verifies** what it read (line arithmetic, subtotals, tax, totals, presence of each
value on the page) instead of trusting it, and every doubt becomes an issue that names
the fields and suggests values for a person to accept or keep. An upload is answered
at once and read in the background. **Every uploaded document is sent to a third
party**: OpenRouter and the provider of the model behind it.

![The review screen of the misprinted sample: the document on the left with three
values framed in amber, the issues and their suggestions on the right](docs/screenshot.png)

*The review screen: one amount on page 2 is printed wrong on purpose; the three values
that disagree are framed on the page, and the issues that follow from it sit under the
line.*

## Quick start with Docker

Needs Docker with Compose 2.24 or newer and an OpenRouter API key.

```bash
cp .env.example .env          # fill in OPENROUTER_API_KEY and ACCESS_CODE
docker compose up -d --build --wait
open http://127.0.0.1:8000    # or paste the address into a browser
```

Make an access code with `python3 -c "import secrets; print(secrets.token_urlsafe(24))"`.
Write values in `.env` bare: no quotes and no `$` (Compose's parser strips quotes and
expands `${...}`). The page asks for the code once per tab; "Try the samples" on the
home screen uploads one of four test invoices. Another port: `HOST_PORT=8080` in
`.env`.

If `up` ends with the container **unhealthy**, the app refused to start; it says why,
without showing any value:

```bash
docker compose logs app       # e.g. "The API cannot start. OPENROUTER_API_KEY is not valid: ..."
docker compose logs -f app    # follow the log
docker compose down           # stop and remove it: every job is lost
```

The container runs as a non-root user on a read-only file system, with no
capabilities, 1.5 GiB of memory (swap included), two CPUs, and its port published on
`127.0.0.1` only (see "Running it in public").

## Without Docker

Needs [uv](https://docs.astral.sh/uv/) 0.13.10 or newer (it installs Python 3.14 and
the locked dependencies).

```bash
cp .env.example .env                    # fill it in, as above
uv sync
uv run fastapi run --host 127.0.0.1     # http://127.0.0.1:8000; without --host it listens on every interface
uv run fastapi dev                      # development: 127.0.0.1:8000, reloads on code changes
```

Settings come from the environment, then from `.env` in the working directory. Run one
process only: jobs live in its memory (`--workers 2` would split them), and every
reload of `fastapi dev` drops them all. The model requests honour `HTTPS_PROXY` (and,
on macOS, the system proxy settings): a forgotten proxy looks like a provider that
never answers.

**No key?** `uv run python tests/demo_server.py` serves the real app on
http://127.0.0.1:8012/ (access code `try-the-demo-page-0001`) with a **fake model** that
replays recorded readings of the four samples; nothing leaves the machine and any other
file ends as failed. `--port`, `--delay` and `--access-code` change it.

## What you will see

The four samples are generated test invoices; the scan and the photo are simulated
from them. Each was read once through the page, in the container, with the real model:

| Sample | Read | Time | Cost |
|---|---|---|---|
| Small PDF, 4 pages, one amount misprinted on purpose | `needs_review`: 5 issues, values framed on the page | 25.7 s | $0.0166 |
| Large PDF, 34 pages | 1,250 of 1,250 rows | 5.4 s | $0.0058 |
| Scan (simulated), 4 pages | 110 of 110 rows | 13.4 s | $0.0179 |
| Photo (simulated) | 14 of 14 rows | 4.0 s | $0.0031 |

In the small PDF, line 58 says 18 × 38.26 = 344.34; it should be 688.68. The app does
not know which of the three numbers is wrong, so it frames all three and suggests each
fix (quantity 9, unit price 19.13, or amount 688.68). The subtotal, the 7 % tax and a
delivery-note total disagree as a result; accepting the amount settles them too. The
misprinted sample takes longer because the page that fails its checks is read again.

## How reading works

- **A PDF with a text layer:** the model describes the table once (which column holds
  what); code takes every row from the text layer, with its position on the page.
- **Scans and photos:** the header is read once and each page separately; a page that
  fails its checks is read a second time.
- **Checks, not trust:** line arithmetic, subtotals and section totals, tax, the
  document totals, numbering, and whether each value is printed on the page.
- **Issues** name the fields they concern and suggest values; nothing is changed
  silently. A person accepts a suggestion or keeps the value as read.
- **A second reading of scans** by another provider's model is optional and off by
  default (below).

Why this design (what was measured and rejected) is in the analysis, part 1 of the
task, delivered separately.

### A second reading of scans (off by default)

On a scan nothing in the document can confirm an item code: there is no text layer to
hold it against, and no arithmetic involves it. With
`SECOND_READING_MODEL=openai/gpt-6-luna`, every page of a scan that is sent to the
model (not one flagged as blurred or dark) is read a second time by that model of
another provider. Where the two readings of a line item differ, the line gets an issue
`second_reading` (a warning) that names the cells and offers the other model's values
as a suggestion; the result keeps the first reading. A second reading that fails or
does not answer in time only leaves a note and never fails a document; second readings
yield the shared request slots to first readings and never hold more than half of
them.

It is off because of what it was measured to do. In a control run on 7 scanned
documents (15 pages, 174 compared rows; three synthetic scans and four real invoices
rendered as scans) it flagged the one error the first reader made and raised 2 false
alarms (1.15 %), just above the 1 % set as the limit before the run. Earlier, on two
synthetic scans, it showed 5 of 5 errors with 1 false alarm on 463 rows. It adds about
40 % to the cost of reading a scan and a few seconds. With so few rows these rates are
indicative only. Two models that misread a code the same way are not caught.

## Costs and times

Each measured once, on generated test invoices, with the default model
(`google/gemini-3.1-flash-lite`):

| Document | Rows read | Time | Cost |
|---|---|---|---|
| digital PDF, 1 page (through the API) | 14 of 14 | 4.9 s | $0.0040 |
| digital PDF, 4 pages (through the API) | 110 of 110 | 15.1 s | $0.0056 |
| digital PDF, 34 pages (through the API) | 1,250 of 1,250 | 7.8 s | $0.0066 |
| scan, 1 page (through the API) | 14 of 14 | 4.7 s | $0.0032 |
| the four samples (through the page, in the container) | see "What you will see" | 4.0–25.7 s | $0.043 together |

With the second reading switched on, a scan costs about 40 % more.

## Configuration

Every setting is described in `.env.example`; a variable set in the environment wins
over the file.

| Setting | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | (required) | the key every model request is sent with; without it the app refuses to start |
| `ACCESS_CODE` | empty | the code every `/invoices` request carries (`Authorization: Bearer <code>`); at least 16 printable ASCII characters. Empty switches the check off (local development only; a warning is logged) |
| `MAX_FILE_BYTES` | 10485760 (10 MB) | the largest upload |
| `MAX_PAGES` | 50 | the most pages of a PDF |
| `MAX_IMAGE_PIXELS` | 50000000 | the pixels of an uploaded image, and of all the images on one page of a scan together |
| `MAX_DOCUMENTS` | 10 | documents waiting or being read; one more is refused (503) |
| `READING_AT_ONCE` | 3 | documents read at the same time; the rest wait as `queued` |
| `MODEL_REQUESTS_IN_FLIGHT` | 24 | model requests at once, shared by every document |
| `DEADLINE_S` | 240 | seconds one document may take once its reading starts; what was read by then is kept |
| `JOBS_KEPT` | 50 | documents kept in memory with their files; older finished ones are dropped (use fewer for a public instance, e.g. 20) |
| `PAGE_MODEL`, `HEADER_MODEL`, `LAYOUT_MODEL` | `google/gemini-3.1-flash-lite` | the models, as OpenRouter names them |
| `SECOND_READING_MODEL` | empty (off) | a second reader for scans, e.g. `openai/gpt-6-luna` |
| `HOST_PORT` | 8000 | read by Docker Compose, not by the app: the port on `127.0.0.1` the container is published on |

## The API

Every `/invoices` request carries the access code. The examples assume

```bash
API=http://127.0.0.1:8000
AUTH="Authorization: Bearer $ACCESS_CODE"
```

The interactive documentation (Swagger UI) is at `/docs`; its "Authorize" button takes
the code.

### `POST /invoices`: upload a document

One multipart part, `file`: a PDF, a JPEG or a PNG, decided by the content, not the
name.

```bash
curl -i -H "$AUTH" -F "file=@invoice.pdf" $API/invoices
```

Only cheap checks run here. The answer is `202 Accepted` with `Location:
/invoices/{id}` and the job (`"status": "queued"`). Refusals:

| Status | `code` | When |
|---|---|---|
| 413 | `file_too_large` | the file is larger than 10 MB |
| 415 | `unsupported_file` | not a PDF, JPEG or PNG |
| 422 | `unreadable_file` | does not open, needs a password, has no pages, or an image of more than 50 million pixels |
| 422 | `too_many_pages` | a PDF of more than 50 pages |
| 503 | `too_many_documents` | 10 documents are already waiting or being read; `Retry-After: 30` |
| 400 / 422 | `invalid_request` | not exactly one file part |

A body far over the limit whose length is declared is refused by the framework before
it is read; that `413` comes as plain text. Every other error is JSON.

### `GET /invoices/{id}`: status, progress, result

```bash
curl -H "$AUTH" $API/invoices/5f0c2a9e4b7d4c1e9a3f6b8d2e1c7a90
```

`status` goes `queued` → `processing` → `ok` (every check passed), `needs_review` (an
issue needs a person) or `failed` (`error` says why). `progress` (`pages_done` of
`pages_total`) is a hint for a progress bar. `result` is `null` until then; it holds
the header, the line items, the issues (each names its fields, such as `line.amount`
or `header.total`, and may carry `suggestions`), notes, control totals, how each page
was read, and the cost and time of every model request. A failed job keeps what was
read. An unknown id is `404 not_found`.

In a PDF with its own text layer every line has `boxes`: a rectangle `[x0, y0, x1, y1]`
for the whole item (`row`) and for each value it frames (`sku`, `quantity`, `amount`,
...; `null` when not framed), on the line's own `page`, in points of the page as
displayed (origin top-left, y down; `pages[].width` / `height` is the page's size).
Scans and photos carry the page only: `boxes` is `null`.

### `GET /invoices`: every job

```bash
curl -H "$AUTH" $API/invoices
```

`{"items": [...]}`, newest first, without results; finished ones carry `line_count`,
`issue_count` and `open_issue_count`. No paging.

### `GET /invoices/{id}/file`: the uploaded file

```bash
curl -H "$AUTH" -o invoice.pdf $API/invoices/5f0c2a9e4b7d4c1e9a3f6b8d2e1c7a90/file
```

### `PUT /invoices/{id}/issues/{issue_id}/resolution`: decide on an issue

```bash
curl -X PUT -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"action": "accept", "suggestion": 2}' \
  $API/invoices/5f0c2a9e4b7d4c1e9a3f6b8d2e1c7a90/issues/0/resolution
```

`{"action": "accept", "suggestion": k}` takes item `k` of the issue's `suggestions`;
`{"action": "keep"}` keeps the values as read (for an issue without suggestions:
"checked"). `DELETE` on the same path takes the decision back. Both answer the
document's `review`: each issue's `state` (`open`, `accepted`, `kept`, or `settled` when
every issue it `follows_from` is accepted with the arithmetic amount of its line) and
`open_count`. Decisions never change the result; a cell's final value is the `set` of
its accepted decision, else as read.

| Status | `code` | When |
|---|---|---|
| 404 | `not_found` / `issue_not_found` | no such document / no such issue |
| 409 | `not_reviewable` | the document is not finished, or failed |
| 409 | `resolution_conflict` | the item would set a cell another accepted decision set to a different value |
| 422 | `invalid_request` / `invalid_resolution` | a malformed body / `accept` without suggestions or no item `k` |
| 413 | `body_too_large` | a body over 1 KiB |

### `GET /health`

```bash
curl $API/health      # {"status":"ok"}; no access code needed
```

### Errors

Every error is RFC 9457 problem details (`application/problem+json`) with a stable
`code` and a `detail` for a person:

```json
{"status": 422, "title": "Unprocessable Content", "code": "too_many_pages",
 "detail": "The PDF has 62 pages; the limit is 50. Split the file and upload the parts."}
```

## Limitations

- **Test data only.** Every figure here comes from generated test invoices (plus four
  real invoices rendered as scans, in one control run of the second reading); each was
  measured once. "Scan" and "photo" samples are simulations.
- **Memory only, one process.** Jobs, uploaded files and review decisions live in the
  process; a restart or an out-of-memory kill loses all of them, and the app comes
  back empty. Kept jobs hold their files: in the container (1.5 GiB limit, swap
  included) peaks of up to about 1.3 GiB were measured with `JOBS_KEPT=50` and about
  1.0 GiB with `JOBS_KEPT=20`; above the limit the container is killed and restarted.
- **A page read again is slow:** the misprinted sample takes about 25–30 s, because
  its failing page is read a second time.
- **No positions on scans and photos:** there is no OCR, so their values are not framed;
  the page is outlined instead.
- **Rows printed twice exactly alike** in a PDF: the frame of one may sit on the
  other.
- **Turned pages:** a PDF page turned by `/Rotate` is read in its own reading order,
  but a page whose content is drawn turned and shown upright by `/Rotate` (landscape
  content) is not read by code: the model reads it. Only generated turned pages were
  tested; no real rotated invoice was available.
- **About a dozen large uploads at the same moment** fill the container's 128 MB
  `/tmp` (it counts against the memory limit); the surplus gets `400 invalid_request`
  instead of `503`.
- **Browsers:** built and checked in Chrome; Firefox and Safari not tried; Safari
  before 17.4 is not supported.
- **Not built:** a fallback model when the provider fails (planned); persistence;
  users or any authentication beyond one shared code; a rate limit.

## Running it in public

The published port is on `127.0.0.1` only: processes on this machine reach it, and
so do containers on it (through `host.docker.internal` with Docker Desktop); other
machines do not. A tunnel or reverse proxy in front of it, with TLS, is the operator's
choice and responsibility. Before that:

- **Set `ACCESS_CODE`**, generated (16 characters at least; `token_urlsafe(24)` gives
  32). Without the code a request to `/invoices...` gets `401` before its body is
  read. Open without it: the page `/` and its files under `/static/`, `/health`,
  `/docs` and `/openapi.json`.
- **Everyone with the code sees every document**, and can download it; uploaded files
  stay in memory until the restart or until `JOBS_KEPT` newer ones push them out.
  The code keeps strangers out; it does not tell users apart.
- **Files go to OpenRouter and the model's provider**, and to a second provider only
  if `SECOND_READING_MODEL` is set.
- **Use a dedicated provider key with a small credit limit.** Whoever holds the code
  can spend: roughly $20–60 an hour for someone uploading as fast as the app allows
  (computed, not measured). When the credit runs out, uploads should end as
  `failed` / `model_unavailable` (expected, not tested).
- **`docker compose config` and `docker inspect` print the key and the code in
  clear**; keep them off screens and out of logs. The access log holds visitors'
  addresses and paths (never the code); it is rotated at 3 × 10 MB.
- Set `JOBS_KEPT` lower (e.g. 20) to bound memory.

## Development

```bash
uv sync                          # create the environment
uv run pre-commit install        # run the hooks on every commit
uv run pytest                    # tests
uv run ruff check                # lint
uv run ruff format --check       # formatting
uv run mypy                      # type check
uv run pre-commit run --all-files
uv lock --check                  # uv.lock matches pyproject.toml
```

The tests need no network and no key: the model provider is replaced by a fake server
(`tests/fakes.py`). They run on macOS, on Linux (CI) and in the image's Debian:

```bash
docker build --target test -t invoice-recognition-test .
docker run --rm --network none invoice-recognition-test
scripts/smoke.sh                 # the real compose.yaml, a dummy key, nothing leaves the machine
```

Node 20+ runs the page's script tests (`tests/js/`, started by `tests/test_web_js.py`);
without it they are skipped locally, and they fail where `CI` is set. The image has no
Node, so CI runs them in its other job. For a font a PDF does not embed, pdfium
measures characters on a substitute font that differs between macOS and Linux, so a
word's horizontal edges can differ by up to about 0.9 point; the tests allow for that.

The page is plain HTML, CSS and JavaScript modules in `src/invoice_recognition/web/`,
with no build step, served with its own Content-Security-Policy.

## Licence

Shared for review; no licence is granted.

PDF.js 4.10.38 is vendored unmodified in
`src/invoice_recognition/web/static/vendor/pdfjs-4.10.38/` under the Apache License 2.0
(its `LICENSE` is beside it).
