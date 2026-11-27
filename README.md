# Invoice Recognition

A small FastAPI service that turns an invoice file (a digital PDF, a scan or a
photo) into structured data. The upload is acknowledged immediately and the
invoice is recognised in the background by an LLM-based flow; the result is
then verified by deterministic code checks (totals, line arithmetic, tax,
presence on the page) instead of being trusted as the model returned it.
Clients poll for the status and fetch the result once processing finishes.

The models are called through [OpenRouter](https://openrouter.ai/); by default
every request goes to `google/gemini-3.1-flash-lite`. **Every uploaded
document is sent to a third party**: OpenRouter, and through it the provider
of the model (with the second reading below switched on, a scan goes to two:
Google's and OpenAI's models).

### A second reading of scans (off by default)

On a scan nothing in the document can confirm an item code: there is no text
layer to hold it against, and no arithmetic involves it. With
`SECOND_READING_MODEL=openai/gpt-6-luna`, every page of a scan that is sent to
the model (not one flagged as blurred or dark) is read a second time by that
model of another provider. Where the two readings of a line item differ, the
line gets an issue `second_reading` (a warning) that names the cells and
offers the other model's values as a suggestion; the result keeps the first
reading. A second reading that fails or does not answer in time only leaves a
note and never fails a document; second readings yield the shared request
slots to first readings and never hold more than half of them.

It is off because of what it was measured to do. In a control run on 7
scanned documents (15 pages, 174 compared rows; three synthetic scans and four
real invoices rendered as scans) it flagged the one error the first reader
made and raised 2 false alarms (1.15 %), just above the 1 % set as the limit
before the run. Earlier, on two synthetic scans, it showed 5 of 5 errors with
1 false alarm on 463 rows. It adds about 40 % to the cost of reading a scan and
a few seconds. With so few rows these rates are indicative only. Two models
that misread a code the same way are not caught.

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

With the second reading switched on, a scan costs about 40 % more (see
above).

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

The page is at `/` (below); the interactive documentation (Swagger UI) is at
`/docs`, and its "Authorize" button takes the access code.

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
- **Files are sent to a model provider** (OpenRouter and the model behind it;
  with the second reading on, a scan goes to two models of two providers).
- **A proxy in the environment is used.** The model requests honour
  `HTTPS_PROXY` (and, on macOS, the system proxy settings). A forgotten proxy
  looks like a provider that never answers.

## The page

Open `/` in a browser. The page asks for the access code once per tab (it keeps
it in the tab's session storage, not local storage: `/docs` runs a third-party
script on the same origin) and sends it with every request, the download of
the file included.

- **Home:** drop a PDF, JPEG or PNG (the 10 MB limit is checked before
  sending), or try one of four samples: generated test invoices; the scan and
  the photo are simulated from them, and the small PDF has one amount
  misprinted on purpose (line 58 printed 344.34 instead of 688.68), so its
  review has something to show. The list below polls once a second while a
  document is still being read.
- **Review:** the document on the left, drawn by PDF.js (or the photo), the
  header and the rows on the right. Only the rows with an issue are shown
  unless "Show all" is on; selecting a row scrolls the document to it, and with
  every row shown the two panes scroll together. Each issue offers what its
  `suggestions` allow (accept a value, keep the value as read, or mark it as
  checked), with "Undo"; the issues that follow from a line's amount are shown
  under that line and settle when the amount is accepted. On a PDF with a text
  layer each value in doubt is framed where it is printed (amber while open,
  green once decided; a row without a framed value gets a band over it); on a
  scan or a photo, which carry no positions, the page is outlined instead.
- Decisions are kept in memory with the job, like everything else, and lost on
  a restart. A cell's final value is the `set` of its accepted decision, else
  as read; cells never collide.

The page is plain HTML, CSS and JavaScript modules served by the app from
`src/invoice_recognition/web/`, with no build step. Its responses carry their
own Content-Security-Policy (only the app's own files run; markup is made by
one escaping template), and `/docs` keeps its own rules. PDF.js 4.10.38 is
vendored unmodified under `static/vendor/` (Apache License 2.0; source and
checksums in its `README.txt`): served by the app because the page's policy
allows no script from elsewhere, and pinned to the version the page was
designed and checked with.

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

Where each value is printed: in a PDF with its own text layer, every line has
`boxes` - a rectangle `[x0, y0, x1, y1]` for the whole item (`row`) and for each
value it frames (`sku`, `quantity`, `amount`, ...; `null` when not framed), on the
line's own `page`. They are in points of the page as displayed (crop box, `/Rotate`),
origin top-left, y down, and `pages[].width` / `height` is that page's size; to draw
on a page rendered `W` × `H` pixels, take `x · W / width` and `y · H / height`.
Scans and photos carry the page only: `boxes` is `null` and `pages[].text_layer` is
`false` (a photo has no `width` / `height` either).

### `GET /invoices`: every job

```bash
curl -H "$AUTH" $API/invoices
```

`{"items": [...]}`, newest first: each job without its result, with
`line_count`, `issue_count` and `open_issue_count` once it is finished. No
paging.

### `GET /invoices/{id}/file`: the uploaded file

```bash
curl -H "$AUTH" -o invoice.pdf $API/invoices/5f0c2a9e4b7d4c1e9a3f6b8d2e1c7a90/file
```

The original bytes, with the type decided at upload and the file name in
`Content-Disposition: inline`. The whole file, no range requests.

### `PUT /invoices/{id}/issues/{issue_id}/resolution`: decide on an issue

```bash
curl -X PUT -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"action": "accept", "suggestion": 2}' \
  $API/invoices/5f0c2a9e4b7d4c1e9a3f6b8d2e1c7a90/issues/0/resolution
```

A person reviewing a read document takes item `k` of an issue's
`suggestions` (`{"action": "accept", "suggestion": k}`, every cell of its
`set`) or keeps the values as read (`{"action": "keep"}`; for an issue
without suggestions it means "checked"). The same request again changes
nothing; a different one replaces the decision. `DELETE` on the same path takes
it back. Both answer the document's `review`, which `GET /invoices/{id}` also
returns: each issue's `state` (`open`, `accepted`, `kept`, or `settled` when
every issue it `follows_from` is accepted with the arithmetic amount of its
line) and `open_count`; list items carry `open_issue_count`.

Decisions are kept in memory with the job and lost with it. They never change
the result: values stay as read, and `status` and `issue_count` describe the
reading. A cell's final value is the `set` of its accepted decision, else as
read; cells never collide (409).

| Status | `code` | When |
|---|---|---|
| 404 | `not_found` / `issue_not_found` | no such document / no such issue |
| 409 | `not_reviewable` | the document is not finished, or failed |
| 409 | `resolution_conflict` | the item would set a cell that another accepted decision set to a different value (quantity, unit price, tax rate, amount, subtotal, tax and total compared as numbers, every other cell exactly) |
| 422 | `invalid_request` | the body is not one of the two shapes above, or `issue_id` is not an integer |
| 422 | `invalid_resolution` | `accept` on an issue without suggestions, or no item `k` |
| 413 | `body_too_large` | a body over 1 KiB (plain text when its length is declared) |

A malformed body is refused (422) before the document is looked up.

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

`PAGE_MODEL`, `HEADER_MODEL`, `LAYOUT_MODEL` and `SECOND_READING_MODEL` name
the models (OpenRouter names); `SECOND_READING_MODEL` is empty by default, and
scans are then read once.

## Access code

`ACCESS_CODE` is one shared secret, compared in constant time. A request to
`/invoices...` without it (or with a wrong one) gets `401 unauthorized` with
`WWW-Authenticate: Bearer`, before the server reads the request's body.
Open without the code: the page `/` and its files `/static/...`, `/health`,
the documentation `/docs` (with
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

- a container image and the deployment;
- a fallback model when a request fails;
- where each value is printed on a scan or a photo (no OCR: only the page).

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

The tests run on macOS and on Linux (CI). One thing differs between them: for a
font a PDF does not embed (the standard Helvetica of the sample invoices), pdfium
measures characters on a substitute font, so where a word starts and ends along
its line can differ by up to about 0.9 point; the tests allow for that, and the
height of a word is computed from its baseline and font size, the same on both.

Node 20+ runs the page's script tests (`tests/js/`, Node's own test runner, started by
`tests/test_web_js.py`); they are skipped without it, and required in CI.

To try the page without a key, `uv run python tests/demo_server.py` serves the app on
http://127.0.0.1:8012/ with a fake model that knows the page's four samples (access code
`try-the-demo-page-0001`; `--port`, `--delay` and `--access-code` change them). Nothing
leaves the machine.
