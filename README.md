# Invoice Recognition

A small FastAPI service that turns an invoice file (a digital PDF, a scan or a
photo) into structured data. The upload is acknowledged immediately and the
invoice is recognised in the background by an LLM-based flow; the result is
then verified by deterministic code checks (totals, line arithmetic, tax,
presence on the page) instead of being trusted as the model returned it.
Clients poll for the status and fetch the result once processing finishes.

The models are called through [OpenRouter](https://openrouter.ai/); by default
every request goes to `google/gemini-3.1-flash-lite`. **Every uploaded document
is sent to a third party**: OpenRouter, and through it the provider of the
model.

Cost and time per document, each measured once on synthetic invoices: the
first four through this API, the last in the research prototype that preceded
this repository (the same flow):

| Document | Cost | Time |
|---|---|---|
| digital PDF (text layer), 1 page, 14 line items | $0.004 | 5 s |
| digital PDF, 4 pages, 110 line items | $0.006 | 15 s |
| digital PDF, 34 pages, 1,250 line items | $0.007 | 8 s |
| scan, 1 page, 14 line items | $0.003 | 5 s |
| scan, 10 pages, 330 line items | ≈ $0.07 | ≈ 35 s |

## Run

Requires [uv](https://docs.astral.sh/uv/) 0.13.10 or newer (it installs
Python 3.14 and the dependencies from `uv.lock`) and an OpenRouter API key.

```bash
cp .env.example .env      # then fill it in: OPENROUTER_API_KEY, ACCESS_CODE
uv sync                   # create the environment
uv run fastapi dev        # development: http://127.0.0.1:8000, reloads on code changes
uv run fastapi run        # serving: listens on 0.0.0.0:8000, no reload
```

`fastapi run` listens on every network interface: with an empty `ACCESS_CODE`
the invoice routes are open to whoever can reach the port.

Without `uv run`, activate the environment first:

```bash
source .venv/bin/activate
fastapi run --port 8000
# or, the same app with uvicorn directly:
uvicorn invoice_recognition.api.main:app --host 0.0.0.0 --port 8000
```

The interactive documentation (Swagger UI) is at `/docs`; its "Authorize"
button takes the access code.

Settings are read from the environment, then from `.env` in the working
directory; a variable in the environment wins. Every setting is described in
`.env.example`. Without `OPENROUTER_API_KEY` the server refuses to start and
says which setting is missing.

Things to know before running it:

- **One process only.** Jobs and uploaded files live in the memory of the
  process that received them. Never start more than one worker
  (`--workers 2` would give each request a different, partial set of jobs).
- **A restart loses everything**, and so does every reload of `fastapi dev`:
  saving a source file restarts the process and drops every job.
- **Set `ACCESS_CODE` for anything reachable from outside.** Everyone who has
  the code sees every uploaded document and can download it. Make one with
  `python -c "import secrets; print(secrets.token_urlsafe(24))"`; a code shorter
  than 16 characters stops the start.
- **Files are sent to a model provider** (OpenRouter and the model behind it).
- **A proxy in the environment is used.** The model requests honour
  `HTTPS_PROXY` (and, on macOS, the system proxy settings). A forgotten proxy
  looks like a provider that never answers.

## The API

Every `/invoices` request carries the access code:
`Authorization: Bearer <code>`. The examples assume

```bash
API=http://127.0.0.1:8000
AUTH="Authorization: Bearer $ACCESS_CODE"
```

### `POST /invoices`: upload a document

One multipart part, `file`: a PDF, a JPEG or a PNG. The type is decided by
the file's content, not by its name or the declared type.

```bash
curl -i -H "$AUTH" -F "file=@invoice.pdf" $API/invoices
```

Only cheap checks run here. The answer is `202 Accepted` with
`Location: /invoices/{id}` and the job as `GET /invoices/{id}` shows it
(`"status": "queued"`). Refusals:

| Status | `code` | When |
|---|---|---|
| 413 | `file_too_large` | the file is larger than 10 MB |
| 415 | `unsupported_file` | not a PDF, JPEG or PNG |
| 422 | `unreadable_file` | does not open, needs a password, has no pages, or an image of more than 50 million pixels |
| 422 | `too_many_pages` | a PDF of more than 50 pages |
| 503 | `too_many_documents` | 10 documents are already waiting or being read; `Retry-After: 30` |
| 400 / 422 | `invalid_request` | not exactly one file part |

A body far over the limit whose length is declared (as curl and browsers do)
is refused by the framework before it is read; that `413` comes as plain text
("Content Too Large"). Every other error is JSON.

### `GET /invoices/{id}`: status, progress, result

```bash
curl -H "$AUTH" $API/invoices/5f0c2a9e4b7d4c1e9a3f6b8d2e1c7a90
```

`status` goes `queued` → `processing` → one of the final statuses: `ok` (every
check passed), `needs_review` (an issue needs a person) or `failed` (`error`
says why). `progress` (`pages_done` of `pages_total`) is a hint for a progress
bar; only a final status means the job is finished. `result` is `null` until
then; it holds the header, the line items, the issues found by the checks
(each names the fields it concerns, such as `line.amount` or `header.total`,
and may carry suggested values that would settle it), notes, control totals,
how each page was read and the cost and time of every model request. A failed job keeps what was read, except when reading
raised (`internal_error`). An unknown id is `404 not_found`.

### `GET /invoices`: every job

```bash
curl -H "$AUTH" $API/invoices
```

`{"items": [...]}`, newest first: each job without its result, with
`line_count` and `issue_count` once it is finished. No paging.

### `GET /invoices/{id}/file`: the uploaded file

```bash
curl -H "$AUTH" -o invoice.pdf $API/invoices/5f0c2a9e4b7d4c1e9a3f6b8d2e1c7a90/file
```

The original bytes, with the type decided at upload and the file name in
`Content-Disposition: inline`. The whole file, no range requests.

### `GET /health`

```bash
curl $API/health      # {"status":"ok"}; no access code needed
```

### Errors

Every error is RFC 9457 problem details (`application/problem+json`) with a
stable `code` and a `detail` for a person:

```json
{"status": 422, "title": "Unprocessable Content", "code": "too_many_pages",
 "detail": "The PDF has 62 pages; the limit is 50. Split the file and upload the parts."}
```

A request that does not match the schema adds `errors`, one entry per field.

## Limits

Each limit is a setting with this default (see `.env.example`):

| Setting | Default | What it limits |
|---|---|---|
| `MAX_FILE_BYTES` | 10485760 (10 MB) | the size of an uploaded file |
| `MAX_PAGES` | 50 | the pages of a PDF |
| `MAX_IMAGE_PIXELS` | 50000000 | the pixels of an uploaded image, and of all the images on one page of a scan together (an A4 page scanned at 600 dpi has about 35 million) |
| `MAX_DOCUMENTS` | 10 | documents waiting or being read; one more is refused (503) |
| `READING_AT_ONCE` | 3 | documents read at the same time; the rest wait as `queued` |
| `MODEL_REQUESTS_IN_FLIGHT` | 24 | model requests at once, shared by every document |
| `DEADLINE_S` | 240 | seconds one document may take once its reading starts; what was read by then is kept |
| `JOBS_KEPT` | 50 | documents kept in memory; older finished ones are dropped |

`PAGE_MODEL`, `HEADER_MODEL` and `LAYOUT_MODEL` name the models (OpenRouter
names).

## Access code

`ACCESS_CODE` is one shared secret, compared in constant time. A request to
`/invoices...` without it (or with a wrong one) gets `401 unauthorized` with
`WWW-Authenticate: Bearer`, before the server reads the request's body.
Open without the code: `/health`, the documentation `/docs` (with
`/docs/oauth2-redirect`, which its "Authorize" button uses) and the schema
`/openapi.json`; there is no `/redoc`. The code must be at least 16 printable
ASCII characters without spaces:

```bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

An empty `ACCESS_CODE` switches the check off, for local development only; the
server then logs a warning at start-up.

The code keeps strangers out; it does not tell users apart.

## Not built yet

- the web page for uploading and reviewing invoices (the API is all there is);
- a container image and the deployment;
- a second reading of scans by another model, and a fallback model when a
  request fails;
- where each value is printed on the page (`lines[].boxes`);
- storing a reviewer's decision on an issue.

## Development

```bash
uv sync                    # create the environment
uv run pre-commit install  # run the hooks on every commit
uv run pytest              # tests
uv run ruff check          # lint
uv run ruff format         # format
uv run mypy                # type check
```

The tests need no network and no key: the model provider is replaced by a fake
server (`tests/fakes.py`). The API tests are async (the anyio pytest plugin,
requests sent to the app through `httpx.ASGITransport`); the tests of the
recognition flow are plain functions that call `asyncio.run`.
