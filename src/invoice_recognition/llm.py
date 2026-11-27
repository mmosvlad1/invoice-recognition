"""Thin async client for OpenRouter chat completions (streaming).

Streams so that time-to-first-token can be measured, and returns usage and cost
for every call: speed and price are first-class outputs here, not afterthoughts.

The caller owns the HTTP client. `openrouter_client` builds one that carries the
API key, so a missing key fails when the service starts, not on every request.
"""

import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from invoice_recognition.shapes import JSONObject, Line

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# httpx timeout of one request (connect, read, write, pool - each on its own), set
# by `complete` on every request. A stream that keeps sending is bounded by the
# total deadline of `complete` instead.
REQUEST_TIMEOUT_S = 120.0

type ContentPart = dict[str, Any]
"""One element of a chat message's content list (a file or an image)."""


@dataclass
class LLMResult:
    model: str
    text: str = ""
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None  # None: the provider did not report a cost
    ttft_s: float | None = None  # first visible token (after any reasoning)
    duration_s: float = 0.0
    provider: str | None = None
    error: str | None = None


def openrouter_client(
    api_key: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    """An HTTP client whose every request is authenticated with `api_key`.

    Surrounding whitespace is stripped (a key read from a file often ends with a
    newline). A key that is then empty, or still contains whitespace or anything
    but printable ASCII, is rejected here: as a header value it would make every
    request fail, with the key quoted in the error. The message never shows it.

    Timeouts are not set here: `complete` sets them on every request.
    `transport` replaces the network (tests pass an `httpx.MockTransport`).
    The caller closes the client (`async with` or `aclose()`).
    """
    key = api_key.strip()
    if not key or not all("!" <= c <= "~" for c in key):
        raise ValueError(
            "OpenRouter API key is empty or contains whitespace or non-ASCII characters"
        )
    return httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"}, transport=transport)


def file_part(data: bytes, mime: str, filename: str = "document") -> ContentPart:
    """Content part for a PDF or an image, as base64 data URL."""
    url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    if mime == "application/pdf":
        return {"type": "file", "file": {"filename": filename, "file_data": url}}
    return {"type": "image_url", "image_url": {"url": url}}


async def complete(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    parts: list[ContentPart],
    *,
    max_tokens: int,
    timeout_s: float = 300.0,  # total deadline for the call, not per chunk
    json_mode: bool = True,
) -> LLMResult:
    """One streamed chat completion: the prompt as text, then `parts`.

    HTTP errors, an error event in the stream, a stream with nothing readable
    and timeouts do not raise: they end up in `LLMResult.error`, next to
    whatever text, usage and timing were received before.

    A stream event that is valid JSON but has the wrong shape (a choice or
    `usage` that is not an object, content that is not a string) raises. The
    caller is expected to treat any exception as a failed request.
    """
    body: dict[str, Any] = {
        "model": model,
        "stream": True,
        "usage": {"include": True},
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, *parts]}],
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    res = LLMResult(model=model)
    chunks: list[str] = []
    malformed = 0  # stream chunks that were not a JSON object
    started = time.monotonic()
    try:
        async with (
            asyncio.timeout(timeout_s),
            client.stream("POST", OPENROUTER_URL, json=body, timeout=REQUEST_TIMEOUT_S) as r,
        ):
            if r.status_code != 200:
                text = (await r.aread()).decode(errors="replace")
                res.error = f"HTTP {r.status_code}: {text[:400]}"
                return res
            async for raw in r.aiter_lines():
                if not raw.startswith("data: "):
                    continue  # keep-alive comments
                payload = raw[6:]
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except _REFUSED:
                    event = None
                if not isinstance(event, dict):  # one broken chunk must not lose the rest
                    malformed += 1
                    continue
                if event.get("error"):
                    res.error = json.dumps(event["error"])[:400]
                    break
                res.provider = event.get("provider") or res.provider
                for choice in event.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        if res.ttft_s is None:
                            res.ttft_s = time.monotonic() - started
                        chunks.append(piece)
                    res.finish_reason = choice.get("finish_reason") or res.finish_reason
                if usage := event.get("usage"):
                    res.prompt_tokens = usage.get("prompt_tokens") or 0
                    res.completion_tokens = usage.get("completion_tokens") or 0
                    details = usage.get("completion_tokens_details") or {}
                    res.reasoning_tokens = details.get("reasoning_tokens") or 0
                    res.cost_usd = usage.get("cost")
    except TimeoutError, httpx.TimeoutException:
        res.error = f"timeout after {time.monotonic() - started:.0f}s"
    except httpx.HTTPError as e:
        res.error = f"{type(e).__name__}: {e}"
    finally:
        res.duration_s = time.monotonic() - started
        res.text = "".join(chunks)
    if malformed and not chunks and not res.error:
        res.error = f"malformed stream: {malformed} chunk(s) could not be parsed, no text received"
    return res


# What json.loads raises for text it refuses: broken JSON (JSONDecodeError, a ValueError),
# an integer longer than Python converts (ValueError: more than 4,300 digits) and nesting
# deeper than the parser's recursion allows (RecursionError). Each is a broken answer.
_REFUSED = (ValueError, RecursionError)


def parse_json(text: str) -> JSONObject | None:
    """Parse model output as a JSON object; tolerate a ```json fence. None if broken,
    including JSON the parser refuses: an integer of thousands of digits, or nesting
    deeper than it can follow."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        out = json.loads(s)
    except _REFUSED:
        return None
    return out if isinstance(out, dict) else None


_LINE_OBJ = re.compile(r'\{[^{}]*"sku"[^{}]*\}')


def salvage_lines(text: str) -> list[Line]:
    """Recover every complete line object from a truncated or broken answer.

    A cut-off response is not worthless: the lines before the cut are real work
    already paid for. A line object is a flat JSON object with an "sku" key.
    """
    lines: list[Line] = []
    for m in _LINE_OBJ.finditer(text):
        try:
            lines.append(json.loads(m.group(0)))
        except _REFUSED:
            pass
    return lines
