"""Which outcomes of a request are failures of the model's provider: one case per outcome
the flow can see. Pure: no request is sent."""

import pytest

from invoice_recognition.llm import Failure, LLMResult
from invoice_recognition.recognition.fallback import provider_failed

WHOLE = '{"rows": []}'
BROKEN = '{"rows": [{"sku": "A'


def failed(failure: Failure, status: int | None = None, text: str = "") -> LLMResult:
    return LLMResult(
        "vendor/model", text=text, error="what went wrong", failure=failure, status=status
    )


def answered(text: str, finish: str | None) -> LLMResult:
    return LLMResult("vendor/model", text=text, finish_reason=finish)


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 599])
def test_a_provider_that_refuses_for_its_own_reasons(status: int) -> None:
    assert provider_failed(failed("http", status))


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 409, 412, 413, 422, 499, 302])
def test_a_request_refused_for_our_reasons(status: int) -> None:
    assert not provider_failed(failed("http", status))


def test_a_refusal_without_a_status_is_not_known_to_be_the_providers() -> None:
    assert not provider_failed(failed("http"))


@pytest.mark.parametrize("status", [None, 408, 429, 500, 502, 503])
def test_an_error_event_of_the_provider(status: int | None) -> None:
    assert provider_failed(failed("stream", status, text=BROKEN))


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_an_error_event_about_our_request(status: int) -> None:
    assert not provider_failed(failed("stream", status))


@pytest.mark.parametrize("failure", ["timeout", "network", "malformed", "exception"])
def test_a_request_that_did_not_complete(failure: Failure) -> None:
    assert provider_failed(failed(failure))
    assert provider_failed(failed(failure, text=WHOLE))  # even after a whole answer


def test_an_error_that_says_nothing_of_how_is_not_theirs() -> None:
    assert not provider_failed(LLMResult("vendor/model", error="something"))


def test_the_streams_own_marker_of_a_failure() -> None:
    assert provider_failed(answered("", "error"))
    assert provider_failed(answered(WHOLE, "error"))


@pytest.mark.parametrize("text", ["", " ", "\n\t "])
@pytest.mark.parametrize("finish", [None, "stop", "tool_calls"])
def test_an_empty_answer(text: str, finish: str | None) -> None:
    assert provider_failed(answered(text, finish))


@pytest.mark.parametrize("text", ["", "  "])
@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_an_empty_answer_the_model_ended_on_purpose(text: str, finish: str) -> None:
    assert not provider_failed(answered(text, finish))


def test_an_answer_cut_short_on_the_way() -> None:
    assert provider_failed(answered(BROKEN, None))


def test_a_whole_answer_without_a_finish_reason() -> None:
    assert not provider_failed(answered(WHOLE, None))
    assert not provider_failed(answered("```json\n" + WHOLE + "\n```", None))


@pytest.mark.parametrize("finish", ["stop", "length", "content_filter"])
@pytest.mark.parametrize("text", [WHOLE, BROKEN, "not json at all"])
def test_an_answer_the_model_ended_is_its_own(text: str, finish: str) -> None:
    # Broken or cut at the token limit: salvage and the next attempt handle it.
    assert not provider_failed(answered(text, finish))
