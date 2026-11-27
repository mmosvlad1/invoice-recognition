"""The OpenRouter client against a fake server (httpx.MockTransport). No network, no key."""

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from invoice_recognition.llm import (
    OPENROUTER_URL,
    PDF_NATIVE,
    ContentPart,
    LLMResult,
    complete,
    file_part,
    openrouter_client,
    parse_json,
    salvage_lines,
)
from invoice_recognition.shapes import JSONObject

from fakes import sse

type Handler = Callable[[httpx.Request], httpx.Response]


def call(
    handler: Handler,
    parts: list[ContentPart] | None = None,
    *,
    timeout_s: float = 300.0,
    json_mode: bool = True,
    plugins: list[JSONObject] | None = None,
) -> LLMResult:
    async def go() -> LLMResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await complete(
                client,
                "vendor/model",
                "Read this.",
                parts or [],
                max_tokens=100,
                timeout_s=timeout_s,
                json_mode=json_mode,
                plugins=plugins,
            )

    return asyncio.run(go())


def answer(body: str, status: int = 200) -> Handler:
    return lambda request: httpx.Response(status, text=body)


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


def test_request_body_is_pinned_to_the_byte() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=sse("{}"))

    call(handler, [file_part(b"%PDF", "application/pdf", "page1.pdf")])
    (request,) = seen
    assert (request.method, str(request.url)) == ("POST", OPENROUTER_URL)
    assert request.content == (
        b'{"model":"vendor/model","stream":true,"usage":{"include":true},"max_tokens":100,'
        b'"messages":[{"role":"user","content":[{"type":"text","text":"Read this."},'
        b'{"type":"file","file":{"filename":"page1.pdf",'
        b'"file_data":"data:application/pdf;base64,JVBERg=="}}]}],'
        b'"response_format":{"type":"json_object"}}'
    )


def test_plugins_are_one_more_key_and_change_nothing_else() -> None:
    bodies: list[JSONObject] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, text=sse("{}"))

    pdf = [file_part(b"%PDF", "application/pdf", "page1.pdf")]
    call(handler, pdf)
    call(handler, pdf, plugins=PDF_NATIVE)
    plain, with_plugins = bodies
    assert "plugins" not in plain
    assert with_plugins.pop("plugins") == [{"id": "file-parser", "pdf": {"engine": "native"}}]
    assert with_plugins == plain


def test_text_only_request_without_json_mode() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, text=sse("plain text"))

    res = call(handler, json_mode=False)
    assert "response_format" not in bodies[0]
    assert bodies[0]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Read this."}]}
    ]
    assert res.text == "plain text"


def test_each_request_has_its_own_httpx_timeout() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=sse("{}"))

    call(handler)
    assert seen[0].extensions["timeout"] == dict.fromkeys(
        ("connect", "read", "write", "pool"), 120.0
    )


def test_file_part_for_a_pdf_and_for_an_image() -> None:
    assert file_part(b"%PDF", "application/pdf") == {
        "type": "file",
        "file": {"filename": "document", "file_data": "data:application/pdf;base64,JVBERg=="},
    }
    assert file_part(b"\xff\xd8", "image/jpeg", "ignored.jpg") == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,/9g="},
    }


def test_client_factory_sends_the_key_with_every_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # complete() must not depend on the environment: the key comes with the client.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=sse("{}"))

    async def go() -> LLMResult:
        client = openrouter_client(" test-key\n", transport=httpx.MockTransport(handler))
        async with client:
            return await complete(client, "m", "p", [], max_tokens=10)

    res = asyncio.run(go())
    assert res.error is None
    assert seen[0].headers["Authorization"] == "Bearer test-key"  # the newline is stripped
    assert seen[0].extensions["timeout"]["read"] == 120.0  # set by complete(), not the client


@pytest.mark.parametrize(
    "key", ["", "  \n", "test key", "test\tkey", "test\x00key", "tést-key", "ключ"]
)
def test_client_factory_rejects_a_key_that_cannot_be_a_header_value(key: str) -> None:
    with pytest.raises(ValueError, match="key is empty or contains") as raised:
        openrouter_client(key)
    if key.strip():
        assert key.strip() not in str(raised.value)  # the key is never shown


# --------------------------------------------------------------------------- #
# Answer
# --------------------------------------------------------------------------- #


def test_content_usage_and_cost() -> None:
    events = [
        {"provider": "Google", "choices": [{"delta": {"content": '{"rows": '}}]},
        {"choices": [{"delta": {"content": "[]}"}}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 7},
                "cost": 0.0042,
            },
        },
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    res = call(answer(body))
    assert res.error is None
    assert res.text == '{"rows": []}'
    assert (res.model, res.provider, res.finish_reason) == ("vendor/model", "Google", "stop")
    assert (res.prompt_tokens, res.completion_tokens, res.reasoning_tokens) == (1200, 30, 7)
    assert res.cost_usd == 0.0042
    assert res.ttft_s is not None and 0 <= res.ttft_s <= res.duration_s


def test_missing_cost_is_none_not_zero() -> None:
    res = call(answer(sse("{}", cost=None)))
    assert res.error is None and res.cost_usd is None
    assert (res.prompt_tokens, res.completion_tokens) == (100, 50)


def test_http_error_keeps_the_start_of_the_body() -> None:
    res = call(answer("x" * 1000, status=500))
    assert res.error == "HTTP 500: " + "x" * 400
    assert res.text == "" and res.cost_usd is None


def test_error_event_in_the_stream() -> None:
    before = {"choices": [{"delta": {"content": '{"rows": ['}}]}
    error = {"error": {"code": 502, "message": "provider went away"}}
    after = {"choices": [{"delta": {"content": "never read"}}]}
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in (before, error, after))
    res = call(answer(body))
    assert res.error == '{"code": 502, "message": "provider went away"}'
    assert res.text == '{"rows": ['  # what came before the error is kept


def test_malformed_only_stream_is_an_error() -> None:
    body = 'data: {"choices": [{"delta": {"content": "{\\"ro\n\ndata: [1, 2]\n\ndata: [DONE]\n\n'
    res = call(answer(body))
    assert res.error == "malformed stream: 2 chunk(s) could not be parsed, no text received"
    assert res.text == ""


def test_one_broken_chunk_does_not_lose_the_answer() -> None:
    res = call(answer("data: {broken\n\n" + sse('{"rows": []}')))
    assert res.error is None and res.text == '{"rows": []}' and res.cost_usd == 0.001


def test_keep_alive_lines_are_skipped() -> None:
    body = ": OPENROUTER PROCESSING\n\n" + sse('{"rows": []}').replace(
        "\n\ndata: [DONE]", "\n\n: OPENROUTER PROCESSING\n\ndata: [DONE]"
    )
    res = call(answer(body))
    assert res.error is None and res.text == '{"rows": []}'


def test_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    res = call(handler)
    assert res.error == "ConnectError: connection refused"
    assert res.text == ""


def test_httpx_timeout_is_reported_as_a_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    res = call(handler)
    assert res.error is not None and re.fullmatch(r"timeout after \d+s", res.error)
    assert res.text == ""


def test_data_after_done_is_ignored() -> None:
    res = call(answer(sse('{"rows": []}') + sse("more", cost=0.5)))
    assert res.error is None and res.text == '{"rows": []}' and res.cost_usd == 0.001


def test_chunk_with_only_a_finish_reason() -> None:
    body = 'data: {"choices": [{"delta": {}, "finish_reason": "length"}]}\n\ndata: [DONE]\n\n'
    res = call(answer(body))
    assert (res.error, res.text, res.finish_reason, res.ttft_s) == (None, "", "length", None)


class _SlowStream(httpx.AsyncByteStream):
    """Sends the start of an answer, then nothing."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"choices": [{"delta": {"content": "{\\"rows\\": ["}}]}\n\n'
        await asyncio.sleep(30)
        yield b"data: [DONE]\n\n"


def test_total_deadline_stops_a_slow_stream() -> None:
    res = call(lambda request: httpx.Response(200, stream=_SlowStream()), timeout_s=0.2)
    assert res.error is not None and re.fullmatch(r"timeout after \d+s", res.error)
    assert res.text == '{"rows": ['  # what arrived before the deadline is kept
    assert 0.2 <= res.duration_s < 5


# --------------------------------------------------------------------------- #
# Reading the answer
# --------------------------------------------------------------------------- #


def test_parse_json_plain_and_fenced() -> None:
    assert parse_json('  {"rows": []}  ') == {"rows": []}
    assert parse_json('```json\n{"rows": [1]}\n```') == {"rows": [1]}
    assert parse_json('```\n{"a": 1}\n```\n') == {"a": 1}


@pytest.mark.parametrize("text", ["[1, 2]", '"text"', "42", "null", '{"rows": [', ""])
def test_parse_json_rejects_anything_but_an_object(text: str) -> None:
    assert parse_json(text) is None


HUGE = "9" * 5_000  # an integer Python refuses to convert (more than 4,300 digits)
DEEP = "[" * 1_000_000 + "]" * 1_000_000  # nesting deeper than the parser can follow


@pytest.mark.parametrize(
    "text",
    ['{"rows": [{"sku": "A-1", "amount": ' + HUGE + "}]}", '{"rows": ' + DEEP + "}"],
    ids=["huge-integer", "deep-nesting"],
)
def test_json_the_parser_refuses_is_a_broken_answer(text: str) -> None:
    assert parse_json(text) is None
    # Not salvaged from either, and nothing raised.
    assert salvage_lines(text) == []


def test_salvage_lines_skips_an_object_with_a_number_too_long_to_read() -> None:
    text = '[{"sku": "A-1", "amount": 1}, {"sku": "A-2", "amount": ' + HUGE + "}"
    assert salvage_lines(text) == [{"sku": "A-1", "amount": 1}]


def test_a_stream_event_the_parser_refuses_is_a_broken_chunk() -> None:
    usage = 'data: {"usage": {"prompt_tokens": ' + HUGE + "}}\n\n"
    res = call(answer(sse('{"rows": []}').replace("data: [DONE]", usage + "data: [DONE]")))
    assert (res.error, res.text) == (None, '{"rows": []}')


def test_salvage_lines_of_a_cut_off_answer() -> None:
    text = (
        '{"header": {"supplier_name": "ACME"}, "rows": ['
        '{"line_no": 1, "sku": "A-1", "amount": 10.0}, '
        '{"line_no": 2, "sku": "A-2", "amount": 20.5}, '
        '{"line_no": 3, "sku": "A-'
    )
    assert salvage_lines(text) == [
        {"line_no": 1, "sku": "A-1", "amount": 10.0},
        {"line_no": 2, "sku": "A-2", "amount": 20.5},
    ]


def test_salvage_lines_skips_objects_that_are_not_valid_json() -> None:
    text = '[{"sku": "A-1", "amount": 1}, {"sku": "A-2", "amount": 2,}, {"sku": "A-3"}'
    assert salvage_lines(text) == [{"sku": "A-1", "amount": 1}, {"sku": "A-3"}]
    assert salvage_lines("no objects here") == []
