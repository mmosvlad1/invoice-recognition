"""Page-by-page invoice extraction.

Instead of one request for the whole document:
  - the header and totals are read from the first and last page,
  - every page is read by its own request, all in parallel,
  - every page is checked on its own (arithmetic; text layer when there is one)
    and re-read if it fails - only that page,
  - pages are concatenated in order and the document is checked as a whole.

The unit of work, of failure and of retry is a page, not a document.

A PDF with a text layer takes a shorter road (layout.py): the model describes
the layout of the table once, code reads the rows of every page from the text
layer, and the model reads one page the usual way as a witness. Pages are
checked exactly as above; a page that fails is read by the model, and so is
the whole document if the layout cannot be used or the witness disagrees.

A line item cut by a page break (carryover.py): in a PDF of more than one page the
answer of a page read by the model has one more field, "carry_in", for the end of a
description printed at the top of the page. When all pages are read, code appends it
to the last line item of the previous page. What was joined, and what found no row to
go to, is in Result.notes; it raises no issue.

A photo (any file that is not a PDF) is measured before the first request
(quality.py): a cut sheet, blur, darkness, glare. A photo that cannot be read ends as
"failed" with the reason and no request; a doubtful one is read and carries the
reason as an issue.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import fields as dataclass_fields
from decimal import Decimal
from typing import Any, cast

import httpx
from PIL import Image

from invoice_recognition.llm import (
    ContentPart,
    LLMResult,
    complete,
    file_part,
    parse_json,
    salvage_lines,
)
from invoice_recognition.recognition import carryover as co
from invoice_recognition.recognition import checks, photo
from invoice_recognition.recognition import evidence as ev
from invoice_recognition.recognition import layout as lay
from invoice_recognition.recognition import quality as ql
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.pdf import extract_pages, page_count
from invoice_recognition.recognition.prompts import EXTRACT_HEADER, EXTRACT_PAGE
from invoice_recognition.recognition.results import Attempt, PageResult, Result
from invoice_recognition.shapes import Checkpoint, Doc, Header, JSONObject, Line

logger = logging.getLogger(__name__)

LINE_KEYS = (
    "line_no",
    "sku",
    "description",
    "quantity",
    "unit",
    "unit_price",
    "tax_rate",
    "amount",
)


# --------------------------------------------------------------------------- #
# One page
# --------------------------------------------------------------------------- #


def _plain(x: Decimal) -> int | float:
    return int(x) if x == x.to_integral_value() else float(x)


def repair(lines: list[Line]) -> int:
    """Fix the one error that can be fixed without a model: a quantity whose
    thousands separator was read as a decimal point ("8.200" -> 8.2). The
    printed amount proves which reading is right."""
    fixed = 0
    for item in lines:
        q = checks.dec(item.get("quantity"))
        p = checks.dec(item.get("unit_price"))
        a = checks.dec(item.get("amount"))
        if q is None or p is None or a is None or abs(ev.money(q * p) - a) <= ev.CENT:
            continue
        if abs(ev.money(q * 1000 * p) - a) <= ev.CENT:
            item["quantity"] = _plain(q * 1000)
            item["repaired"] = "quantity_x1000"
            fixed += 1
    return fixed


def rows_to_page(rows: list[Any], page: int) -> tuple[list[Line], list[Checkpoint]]:
    """Split the model's ordered rows into line items and control totals. The
    position of a total is derived here, from the order - never asked from the model."""
    lines: list[Line] = []
    totals: list[Checkpoint] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "total_label" in row:
            totals.append(
                {
                    "label": row.get("total_label"),
                    "amount": row.get("amount"),
                    "after_line": len(lines),
                    "page": page,
                }
            )
        elif row.get("sku") is not None or row.get("amount") is not None:
            lines.append({k: row.get(k) for k in LINE_KEYS} | {"page": page})
    return lines, totals


def check_page(
    lines: list[Line], totals: list[Checkpoint], page: int, evidence: ev.Evidence | None
) -> list[Issue]:
    doc: Doc = {"lines": lines, "checkpoints": totals}
    issues = checks.check_line_math(doc) + checks.check_checkpoints(doc, partial=True)
    if evidence is not None:
        only = ev.Evidence(evidence.source, evidence.locale, evidence.page(page))
        issues += checks.check_grounding(doc, only) + checks.check_coverage(doc, only)
    for i in issues:
        i.page = page
        if i.line is not None and 0 <= i.line < len(lines):
            no = lines[i.line].get("line_no")
            if isinstance(no, int) and not isinstance(no, bool):
                i.line_no = no
    return issues


async def _call(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    part: ContentPart | None,
    max_tokens: int,
    timeout_s: float,
) -> LLMResult:
    """complete() that never raises: whatever goes wrong inside one request is
    that request's error, not the document's. part=None: a text-only request."""
    try:
        return await complete(
            client,
            model,
            prompt,
            [part] if part else [],
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    except Exception as e:  # fault boundary: one request
        logger.warning("request to %s failed", model, exc_info=True)
        return LLMResult(model=model, error=f"{type(e).__name__}: {e}")


def _attempt(res: LLMResult, lines: int, issues: int) -> Attempt:
    return Attempt(
        res.model,
        res.duration_s,
        res.cost_usd or 0.0,
        res.prompt_tokens,
        res.completion_tokens,
        lines,
        issues,
        res.error,
        cost_known=res.cost_usd is not None,
    )


async def read_page(
    client: httpx.AsyncClient,
    part: ContentPart,
    page: int,
    total: int,
    evidence: ev.Evidence | None,
    cfg: Config,
    sem: asyncio.Semaphore,
    carry: bool = False,
) -> PageResult:
    """Read one page, once per entry of cfg.page_models until an attempt passes its
    checks, and keep the best attempt.

    Issues of the returned page index its own lines (0-based within the page);
    process() shifts them to document positions.

    carry=True: the page is asked with the "carry_in" field (carryover.py);
    PageResult.carry_in is the field of the attempt that is kept. carry=False:
    EXTRACT_PAGE, to the byte."""
    best: PageResult | None = None
    best_key: tuple[bool, int, int] = (True, 0, 0)
    attempts: list[Attempt] = []
    counts: list[int] = []  # lines per attempt, for the attempts that were read cleanly
    prompt = co.carry_prompt(page, total) if carry else EXTRACT_PAGE.format(page=page, total=total)
    for model in cfg.page_models:
        async with sem:
            res = await _call(client, model, prompt, part, cfg.page_max_tokens, cfg.page_timeout_s)
        parsed = parse_json(res.text)
        rows = (parsed or {}).get("rows")
        tail = ""
        if rows is None:  # broken or cut-off answer: keep whatever complete rows it has
            rows = salvage_lines(res.text)
        elif carry and parsed is not None and isinstance(parsed.get("carry_in"), str):
            # Not taken from a salvaged answer; anything but a string is no tail.
            tail = co.carry_text(parsed["carry_in"])
        rows = rows if isinstance(rows, list) else []
        lines, totals = rows_to_page(rows, page)
        fixed = repair(lines)
        issues = check_page(lines, totals, page, evidence)
        failed = bool(res.error) or parsed is None
        if failed:
            why = res.error or f"answer is not valid JSON (finish={res.finish_reason})"
            issues.append(Issue("page_read", why, page=page))
        else:
            counts.append(len(lines))
            # Without a text layer nothing else notices a page that came back empty
            # (with one, check_coverage does). A page with only control totals is not empty.
            if not lines and not totals and evidence is None:
                issues.append(
                    Issue(
                        "page_empty",
                        "the answer has no line items and no control totals",
                        page=page,
                    )
                )
        dropped = len(rows) - len(lines) - len(totals)
        if dropped:
            issues.append(
                Issue(
                    "rows_dropped",
                    f"{dropped} row(s) of the answer have neither sku nor amount and were left out",
                    page=page,
                )
            )
        attempts.append(_attempt(res, len(lines), len(issues)))
        cand = PageResult(page, lines, totals, issues, [], fixed, carry_in=tail)
        # A clean read beats a failed one, then fewer issues, then more lines; on a tie the
        # earlier attempt stays. Counting issues alone let an emptier answer win.
        key = (failed, len(issues), -len(lines))
        if best is None or key < best_key:
            best, best_key = cand, key
        if not best.issues:
            break
    if best is None:
        raise ValueError("no attempt was made: Config.page_models is empty")
    best.attempts = attempts
    if evidence is None and len(set(counts)) > 1:
        # No text layer to say which attempt is complete: keep the page flagged.
        best.issues.append(
            Issue(
                "attempts_disagree",
                "attempts returned different numbers of lines: " + " and ".join(map(str, counts)),
                page=page,
            )
        )
    return best


# --------------------------------------------------------------------------- #
# Text-layer PDF: layout from the model, rows by code
# --------------------------------------------------------------------------- #

# Parser notes that do not touch any extracted value: reported in Result.notes, never as issues.
HARMLESS_NOTES = ("unclassified_amount", "total_without_amount", "joined_across_pages")


async def ask_layout(
    client: httpx.AsyncClient, prompt: str, cfg: Config
) -> tuple[JSONObject | None, list[Attempt], str | None]:
    """One text-only request. Returns (answer, attempts, why it cannot be used)."""
    res = await _call(client, cfg.layout_model, prompt, None, cfg.layout_max_tokens, 90)
    raw = None if res.error else parse_json(res.text)
    error = res.error or (
        None if raw is not None else f"answer is not valid JSON (finish={res.finish_reason})"
    )
    return raw, [_attempt(res, 0, 1 if error else 0)], error


def layout_pages(
    parsed: lay.Parsed, n: int, evidence: ev.Evidence
) -> tuple[dict[int, PageResult], list[JSONObject]]:
    """Rows found by code -> one PageResult per page, checked like a page read by
    the model. Also returns the harmless notes of the parser."""
    by_page: dict[int, list[Line]] = {p: [] for p in range(1, n + 1)}
    totals: dict[int, list[Checkpoint]] = {p: [] for p in range(1, n + 1)}
    first: dict[int, int] = {}  # index of the first line of each page within the document
    for idx, item in enumerate(parsed.lines):
        first.setdefault(item["page"], idx)
        by_page[item["page"]].append(
            {k: (_plain(v) if isinstance(v, Decimal) else v) for k, v in item.items()}
        )
    done = 0
    for p in range(1, n + 1):  # a page without lines starts where the previous one ended
        first.setdefault(p, done)
        done = first[p] + len(by_page[p])
    for c in parsed.checkpoints:
        totals[c["page"]].append(
            {
                "label": c["label"],
                "amount": _plain(c["amount"]),
                "after_line": c["after_line"] - first[c["page"]],
                "page": c["page"],
            }
        )
    pages = {}
    for p in range(1, n + 1):
        try:
            issues = check_page(by_page[p], totals[p], p, evidence)
        # Fault boundary: a bug here is this page's problem; the model reads it.
        except Exception as e:
            logger.warning("checks of page %d found by code failed", p, exc_info=True)
            issues = [Issue("page_read", f"{type(e).__name__}: {e}", page=p)]
        issues += [
            Issue("layout_row", f"{note['kind']}: {note['text']}", page=p)
            for note in parsed.notes
            if note["page"] == p and note["kind"] not in HARMLESS_NOTES
        ]
        pages[p] = PageResult(p, by_page[p], totals[p], issues, method="layout")
    return pages, [note for note in parsed.notes if note["kind"] in HARMLESS_NOTES]


def compare_witness(code: list[Line], model: list[Line]) -> JSONObject:
    """Rows of one page found by code against the same page read by the model.

    Arithmetic cannot tell quantity from unit price (q x p = p x q), so a layout
    answer with the two columns swapped passes every other check. The model
    reading the page is a second opinion on exactly that. Verdict:
      confirmed  - more than half of the rows agree in sku, quantity, price, amount
      swapped    - in more than half, quantity and price are crossed
      mismatch   - neither: the two readings disagree in some other way
      unverified - the model returned no rows to compare with
    Half is the threshold because the outcomes are two far-apart clusters: a
    right layout disagrees only where the model misread a row (a few percent in
    the measurements), a wrong one disagrees on every row whose quantity differs
    from its price. The middle keeps the widest margin from both."""
    free = list(range(len(model)))
    same = crossed = 0
    for c in code:
        sku = " ".join(str(c.get("sku") or "").upper().split())
        j = next(
            (j for j in free if " ".join(str(model[j].get("sku") or "").upper().split()) == sku),
            None,
        )
        if j is None:
            continue
        free.remove(j)
        cq, cp, ca = (checks.dec(c.get(k)) for k in ("quantity", "unit_price", "amount"))
        mq, mp, ma = (checks.dec(model[j].get(k)) for k in ("quantity", "unit_price", "amount"))
        if cq is None or cp is None or ca is None or ca != ma:
            continue
        if (cq, cp) == (mq, mp):
            same += 1
        elif (cq, cp) == (mp, mq):
            crossed += 1
    n = max(len(code), len(model))
    if not model:
        verdict = "unverified"
    elif same * 2 > n:
        verdict = "confirmed"
    elif crossed * 2 > n:
        verdict = "swapped"
    else:
        verdict = "mismatch"
    return {"status": verdict, "compared": n, "same": same, "crossed": crossed}


def _page_key(pr: PageResult) -> tuple[bool, int, int]:
    """The order of read_page between attempts: a clean read, then fewer issues, then more lines."""
    return (any(i.check == "page_read" for i in pr.issues), len(pr.issues), -len(pr.lines))


# --------------------------------------------------------------------------- #
# A description that goes on at the top of the next page
# --------------------------------------------------------------------------- #


def join_carried(pages: list[PageResult]) -> list[JSONObject]:
    """Append every PageResult.carry_in to the description of the last line item of the
    previous page, in place, by the rules of carryover.join_carry: nothing is touched
    when that page ends with a control total, has no rows, was not read, or there is no
    previous page. Returns the notes for Result.notes ("line" indexes Result.lines).

    A page that kept a "page_read" issue owns nothing: its answer failed or was cut off,
    so its last row is not known to be the last row printed. A page found by code already
    has the text when the layout parser joined it across the break: it is not appended
    twice."""
    entries: list[JSONObject] = [
        {
            "page": pr.page,
            "rows": co.flow_rows(pr.lines, pr.totals),
            "carry_in": pr.carry_in,
            "unread": any(i.check == "page_read" for i in pr.issues),
            "joined_by_code": pr.method == "layout",
        }
        for pr in pages
    ]
    joined, found = co.join_carry(entries)
    rows = {p["page"]: p["rows"] for p in joined}
    by_page = {pr.page: pr for pr in pages}
    first, done = {}, 0  # index of the first line of each page within the document
    for pr in pages:
        first[pr.page] = done
        done += len(pr.lines)
    notes: list[JSONObject] = []
    for f in found:
        note: JSONObject = {
            "page": f["page"],
            "kind": f["kind"],
            "text": f["text"],
            "source": "carry_in",
        }
        if "to_page" in f:
            # The owner is the last row of that page and a line item: owner.lines[-1].
            owner = by_page[f["to_page"]]
            if f["kind"] == co.JOINED:
                owner.lines[-1]["description"] = rows[f["to_page"]][f["row"]]["description"]
            note |= {
                "to_page": f["to_page"],
                "line": first[f["to_page"]] + len(owner.lines) - 1,
                "line_no": f["line_no"],
                "sku": f["sku"],
            }
        else:
            note["reason"] = f["reason"]
        notes.append(note)
    return notes


# --------------------------------------------------------------------------- #
# Photo quality
# --------------------------------------------------------------------------- #


def _photo_issue(r: ql.Reason, lim: ql.Limits) -> Issue:
    """A reason of quality.py as an issue. The text for the user is Reason.message
    (Result.quality); this one follows the other issues: with the number and the limit it
    was held against."""
    side = f" ({r.side} side)" if r.side else ""
    reject = r.severity == "reject"
    limit = lim.sharpness_reject if reject else lim.sharpness_warn
    blurred = f"Text sharpness {r.value} is below {limit:g}." + (
        "" if reject else " Numbers are usually still read; item codes (SKUs) may be misread."
    )
    no_text = (
        f"No text was found in the photo ({r.value:.0f} word-like regions, "
        f"at least {lim.min_boxes} needed)."
        if r.value is not None
        else ""
    )
    text = {
        "no_text": no_text,
        "text_at_edge": f"Text reaches the edge of the frame{side}, {r.value} text heights "
        "from it: part of the document is outside the photo.",
        "sheet_cut": f"The sheet runs out of the frame{side}; no text is near that edge.",
        "sheet_edge_unseen": "No edge of the sheet is visible, so nothing confirms that the "
        "whole document is in the photo.",
        "blurred": blurred,
        "dark": f"Paper brightness {r.value} of 255 is below {lim.paper_level:g}.",
        "glare": f"Glare covers {(r.value or 0) * 100:.1f}% of the sheet.",
        "text_small": f"Text lines are {r.value} px tall, below {lim.text_height_px:g}.",
    }.get(r.code) or r.code
    if reject:
        text += " No request was sent."
    return Issue(f"photo_{r.code}", text, page=1, severity="error" if reject else "warning")


def _photo_quality(img: Image.Image, cfg: Config) -> tuple[JSONObject, list[Issue]]:
    """Measure the photo as taken. Returns Result.quality and the issues for its reasons.
    Never raises: a photo the check cannot measure is read the way every photo was read
    before the check existed."""
    try:
        names = {f.name for f in dataclass_fields(ql.Limits)}
        lim = ql.Limits(
            **{
                k[6:]: v
                for k, v in asdict(cfg).items()
                if k.startswith("photo_") and k[6:] in names
            }
        )
        q, why = ql.check(img, lim)
        verdict = (
            "reject" if any(r.severity == "reject" for r in why) else ("warn" if why else "ok")
        )
        info = {
            "verdict": verdict,
            "reasons": [r.to_json() for r in why],
            "measures": q.to_json(),
            "limits": asdict(lim),
        }
        return info, [_photo_issue(r, lim) for r in why]
    except Exception as e:  # fault boundary: the measurement itself broke
        logger.warning("the photo could not be measured", exc_info=True)
        return {"verdict": "unchecked", "error": f"{type(e).__name__}: {e}", "reasons": []}, []


# --------------------------------------------------------------------------- #
# Whole document
# --------------------------------------------------------------------------- #


async def read_header(
    client: httpx.AsyncClient, part: ContentPart, cfg: Config
) -> tuple[Header, list[Attempt]]:
    attempts: list[Attempt] = []
    header: Header = {}
    for _ in range(2):
        res = await _call(client, cfg.header_model, EXTRACT_HEADER, part, 4_000, 90)
        header = parse_json(res.text) or {}
        attempts.append(_attempt(res, 0, 0 if header else 1))
        if header.get("total") is not None and header.get("supplier_name"):
            break
    return header, attempts


async def process(
    data: bytes,
    mime: str,
    client: httpx.AsyncClient,
    cfg: Config | None = None,
    on_page: Callable[[PageResult], None] | None = None,
) -> Result:
    """Read one invoice file. Never raises for a bad file or a bad answer: those end
    up in Result.issues and Result.status.

    on_page is called once per page, in its final form, as soon as it is settled."""
    cfg = cfg or Config()
    started = time.monotonic()
    sem = asyncio.Semaphore(cfg.concurrency)
    evidence: ev.Evidence | None = None
    text_pages: list[list[lay.TextLine]] | None = None  # set when the layout road is taken
    layout_info: JSONObject = {"status": "off", "reason": "not a PDF"}
    quality: JSONObject = {}
    photo_issues: list[Issue] = []

    def failed(issues: list[Issue]) -> Result:
        """Nothing was sent to a model: a result the caller can show, not a crash."""
        return Result(
            status="failed",
            header={},
            lines=[],
            checkpoints=[],
            issues=issues,
            pages=[],
            evidence_source=None,
            duration_s=time.monotonic() - started,
            calls=0,
            cost_usd=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            quality=quality,
        )

    try:
        if mime != "application/pdf":
            # Any other file is a photo: one page, read as an image (photo.py).
            n = 1
            img = photo.load(data)
            if cfg.photo_quality:
                # Before any request: on an unreadable photo the model does not answer
                # "nothing", it invents rows.
                quality, photo_issues = _photo_quality(img, cfg)
            if quality.get("verdict") != "reject":
                parts = [photo.part(img)]
                header_part = parts[0]
        else:
            n = page_count(data)
            texts = ev.pdf_text_layer(data)
            layout_info["reason"] = "no text layer"
            if ev.has_text_layer(texts):  # hard checks only against a text layer
                try:
                    scanned = lay.scanned_pages(data)
                # Fault boundary: cannot tell, so read it as if the check did not exist.
                except Exception:
                    logger.warning("could not look for page images", exc_info=True)
                    scanned = []
                if scanned:  # a scan with somebody's OCR text over it is still a scan
                    layout_info["reason"] = (
                        f"page image with text over it (pages {scanned[:5]}): treated as a scan"
                    )
                else:
                    evidence = ev.build(texts, "text_layer")
                    layout_info["reason"] = "disabled"
                    if cfg.use_layout:
                        try:
                            text_pages = lay.text_lines(data)
                        # Fault boundary: the layout road is an optimisation; without it
                        # the model reads every page.
                        except Exception as e:
                            logger.warning("no text layer with coordinates", exc_info=True)
                            layout_info["reason"] = (
                                "text layer with coordinates is not available: "
                                f"{type(e).__name__}: {e}"
                            )
            parts = [
                file_part(extract_pages(data, [p]), mime, f"page{p}.pdf") for p in range(1, n + 1)
            ]
            # The header is read from the first and the last page.
            if n:
                header_pages = [1, n] if n > 1 else [1]
                header_part = file_part(extract_pages(data, header_pages), mime, "header.pdf")
    except Exception as e:  # fault boundary: a damaged or unsupported file
        logger.warning("file cannot be read", exc_info=True)
        return failed([Issue("unreadable_file", f"{type(e).__name__}: {e}")])
    if quality.get("verdict") == "reject":
        return failed(photo_issues)

    # The "carry_in" field is asked for only where a previous page can exist. A photo and
    # a one-page PDF keep the request they always had, to the byte.
    carry = cfg.carry_in and mime == "application/pdf" and n > 1

    async def one(p: int, announce: bool = True, with_carry: bool | None = None) -> PageResult:
        try:
            ask = carry if with_carry is None else with_carry
            r = await read_page(client, parts[p - 1], p, n, evidence, cfg, sem, ask)
        except Exception as e:  # fault boundary: whatever breaks while reading one page
            logger.warning("page %d could not be read", p, exc_info=True)
            r = PageResult(p, issues=[Issue("page_read", f"{type(e).__name__}: {e}", page=p)])
        if on_page and announce:
            on_page(r)
        return r

    layout_attempts: list[Attempt] = []
    layout_issues: list[Issue] = []
    notes: list[JSONObject] = []
    if text_pages is None or evidence is None:
        first, *rest = await asyncio.gather(
            read_header(client, header_part, cfg), *(one(p) for p in range(1, n + 1))
        )
        header, header_attempts = cast(tuple[Header, list[Attempt]], first)
        pages = cast(list[PageResult], rest)
    else:
        # The page shown to the layout model and read by the witness: the first one with
        # rows that look like items.
        wp = next((p.number for p in evidence.pages if p.item_rows), 1)

        async def witness() -> PageResult | None:
            # The witness request is not changed: EXTRACT_PAGE. It is compared with the rows
            # found by code, nothing more.
            return await one(wp, announce=False, with_carry=False) if cfg.layout_witness else None

        (header, header_attempts), (raw, layout_attempts, error), seen = await asyncio.gather(
            read_header(client, header_part, cfg),
            ask_layout(client, lay.layout_request(text_pages, wp), cfg),
            witness(),
        )
        by_code: dict[int, PageResult] | None = None
        if error is None:
            try:
                parsed = lay.parse_document(text_pages, lay.resolve(raw, text_pages[wp - 1]))
                by_code, notes = layout_pages(parsed, n, evidence)
            except lay.LayoutError as e:  # the answer does not fit the page
                error = f"{type(e).__name__}: {e}"
            except Exception as e:  # fault boundary: a bug takes the same way out
                logger.warning("rows could not be read by the layout", exc_info=True)
                error = f"{type(e).__name__}: {e}"
        if by_code is None:
            layout_info = {"status": "failed", "reason": error}
        elif seen is None:
            layout_info = {"status": "unchecked", "reason": "witness page is switched off"}
        else:
            layout_info = compare_witness(by_code[wp].lines, seen.lines)
        layout_info["witness_page"] = wp if cfg.layout_witness else None

        if by_code is None or layout_info["status"] in ("failed", "swapped", "mismatch"):
            # Fallback for the whole document: every page by the model; the witness page is
            # already read.
            notes = []
            others = await asyncio.gather(
                *(one(p) for p in range(1, n + 1) if seen is None or p != wp)
            )
            pages = sorted([*others, *([seen] if seen else [])], key=lambda r: r.page)
            if seen and on_page:
                on_page(seen)
            if layout_info["status"] != "failed":
                what = (
                    "quantity and unit price are crossed"
                    if layout_info["status"] == "swapped"
                    else "the rows differ"
                )
                layout_issues.append(
                    Issue(
                        f"layout_{layout_info['status']}",
                        page=wp,
                        message=(
                            f"rows found by code and rows read by the model disagree on page "
                            f"{wp}: {what} ({layout_info['same']} of {layout_info['compared']} "
                            f"agree, {layout_info['crossed']} crossed); the document was read "
                            "page by page by the model"
                        ),
                    )
                )
        else:
            if layout_info["status"] == "unverified":
                layout_issues.append(
                    Issue(
                        "layout_unverified",
                        page=wp,
                        message=(
                            f"the model returned no rows for page {wp}, so nothing confirms "
                            "which column is quantity and which is unit price"
                        ),
                    )
                )
            found = by_code

            async def settle(p: int) -> PageResult:
                """A page found by code stays unless it fails its checks; then the model
                reads it and the better one is kept."""
                mine = found[p]
                theirs = seen if seen is not None and p == wp else None
                if mine.issues and theirs is None:
                    theirs = await one(p, announce=False)
                if theirs is not None and mine.issues and _page_key(theirs) < _page_key(mine):
                    chosen = theirs
                else:
                    chosen = mine
                    chosen.attempts = theirs.attempts if theirs is not None else []
                if on_page:
                    on_page(chosen)
                return chosen

            pages = list(await asyncio.gather(*(settle(p) for p in range(1, n + 1))))
            kept = {p.page for p in pages if p.method == "layout"}
            notes = [note for note in notes if note["page"] in kept]
    layout_info |= {
        "pages_layout": sum(1 for p in pages if p.method == "layout"),
        "pages_model": sum(1 for p in pages if p.method == "model"),
    }
    if carry:
        # Every page is read and its best attempt chosen; the lines are not concatenated
        # or checked as a document yet.
        try:
            notes = notes + join_carried(pages)
        except Exception as e:  # fault boundary: a bug here costs the tails, not the document
            logger.warning("descriptions carried over a page break were not joined", exc_info=True)
            failure = f"{type(e).__name__}: {e}"
            notes = notes + [
                {"page": None, "kind": "carry_join_failed", "text": failure, "source": "carry_in"}
            ]

    lines: list[Line] = []
    checkpoints: list[Checkpoint] = []
    issues: list[Issue] = []
    for pr in pages:
        offset = len(lines)
        lines += pr.lines
        checkpoints += [t | {"after_line": t["after_line"] + offset} for t in pr.totals]
        # Page issues index the page's own lines; in the Result every issue indexes Result.lines.
        for i in pr.issues:
            if i.line is not None:
                i.line += offset
        issues += pr.issues
    doc: Doc = {"header": header, "lines": lines, "checkpoints": checkpoints}
    issues += (
        checks.check_doc_sums(doc)
        + checks.check_tax(doc)
        + checks.check_numbering(doc)
        + checks.check_checkpoints(doc)
    )
    if evidence is not None:
        issues += checks.check_header_grounding(doc, evidence)

    issues += layout_issues + photo_issues

    all_attempts = header_attempts + layout_attempts + [a for p in pages for a in p.attempts]
    status = "failed" if not lines or not header else ("needs_review" if issues else "ok")
    return Result(
        status=status,
        header=header,
        lines=lines,
        checkpoints=checkpoints,
        issues=issues,
        pages=list(pages),
        evidence_source=evidence.source if evidence else None,
        duration_s=time.monotonic() - started,
        calls=len(all_attempts),
        cost_usd=sum(a.cost_usd for a in all_attempts),
        prompt_tokens=sum(a.prompt_tokens for a in all_attempts),
        completion_tokens=sum(a.completion_tokens for a in all_attempts),
        header_attempts=header_attempts,
        cost_unknown_calls=sum(1 for a in all_attempts if not a.cost_known),
        layout=layout_info,
        layout_attempts=layout_attempts,
        notes=notes,
        quality=quality,
    )
