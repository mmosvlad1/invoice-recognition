"""Page-by-page invoice extraction.

Instead of one request for the whole document:
  - the header and totals are read from the first and last page,
  - every page is read by its own request, all in parallel,
  - every page is checked on its own (arithmetic; text layer when there is one)
    and re-read if it fails - only that page,
  - pages are concatenated in order and the document is checked as a whole.

The unit of work, of failure and of retry is a page, not a document.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any, cast

import httpx

from invoice_recognition.llm import (
    ContentPart,
    LLMResult,
    complete,
    file_part,
    parse_json,
    salvage_lines,
)
from invoice_recognition.recognition import checks
from invoice_recognition.recognition import evidence as ev
from invoice_recognition.recognition.checks import Issue
from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.pdf import extract_pages, page_count
from invoice_recognition.recognition.prompts import EXTRACT_HEADER, EXTRACT_PAGE
from invoice_recognition.recognition.results import Attempt, PageResult, Result
from invoice_recognition.shapes import Checkpoint, Doc, Header, Line

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
) -> PageResult:
    """Read one page, once per entry of cfg.page_models until an attempt passes its
    checks, and keep the best attempt.

    Issues of the returned page index its own lines (0-based within the page);
    process() shifts them to document positions."""
    best: PageResult | None = None
    best_key: tuple[bool, int, int] = (True, 0, 0)
    attempts: list[Attempt] = []
    counts: list[int] = []  # lines per attempt, for the attempts that were read cleanly
    prompt = EXTRACT_PAGE.format(page=page, total=total)
    for model in cfg.page_models:
        async with sem:
            res = await _call(client, model, prompt, part, cfg.page_max_tokens, cfg.page_timeout_s)
        parsed = parse_json(res.text)
        rows = (parsed or {}).get("rows")
        if rows is None:  # broken or cut-off answer: keep whatever complete rows it has
            rows = salvage_lines(res.text)
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
        cand = PageResult(page, lines, totals, issues, [], fixed)
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
        )

    if mime != "application/pdf":
        return failed([Issue("unreadable_file", f"unsupported file type: {mime}")])
    try:
        n = page_count(data)
        texts = ev.pdf_text_layer(data)
        if ev.has_text_layer(texts):  # hard checks only against a text layer
            evidence = ev.build(texts, "text_layer")
        parts = [file_part(extract_pages(data, [p]), mime, f"page{p}.pdf") for p in range(1, n + 1)]
        # The header is read from the first and the last page.
        if n:
            header_pages = [1, n] if n > 1 else [1]
            header_part = file_part(extract_pages(data, header_pages), mime, "header.pdf")
    except Exception as e:  # fault boundary: a damaged or unsupported file
        logger.warning("file cannot be read", exc_info=True)
        return failed([Issue("unreadable_file", f"{type(e).__name__}: {e}")])

    async def one(p: int) -> PageResult:
        try:
            r = await read_page(client, parts[p - 1], p, n, evidence, cfg, sem)
        except Exception as e:  # fault boundary: whatever breaks while reading one page
            logger.warning("page %d could not be read", p, exc_info=True)
            r = PageResult(p, issues=[Issue("page_read", f"{type(e).__name__}: {e}", page=p)])
        if on_page:
            on_page(r)
        return r

    first, *rest = await asyncio.gather(
        read_header(client, header_part, cfg), *(one(p) for p in range(1, n + 1))
    )
    header, header_attempts = cast(tuple[Header, list[Attempt]], first)
    pages = cast(list[PageResult], rest)

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

    all_attempts = header_attempts + [a for p in pages for a in p.attempts]
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
    )
