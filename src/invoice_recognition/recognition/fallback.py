"""Whether a request failed at the model's provider, so that a backup model of another
provider may be asked the same thing (Config.fallback_model).

Only a failure of the provider: it timed out, was rate limited or down (HTTP 408, 429,
5xx, or the same code in an error event of the stream), the connection broke, nothing
readable came, or nothing at all came where a JSON answer must. Never what the backup
cannot change: a request refused for our own reasons (any other 4xx: the request, the key
and the credits are shared by both models, and a wrong model name must stay visible), an
answer the model itself got wrong (broken JSON that ended normally, the token limit
spent), or a page that fails its checks - those are retried by the next attempt as
always. Pure: nothing here sends or reads anything but the result given.
"""

from typing import assert_never

from invoice_recognition.llm import LLMResult, parse_json

# The statuses a provider answers with when it, not the request, is the problem: it timed
# out or is rate limited. Every status from 500 up counts too.
_PROVIDER_STATUSES = frozenset({408, 429})

# Why a model may legitimately stop with nothing written: it spent every output token (a
# reasoning model can spend them all before the answer), or the answer was filtered.
# Asking again would end the same way.
_EMPTY_BY_DESIGN = frozenset({"length", "content_filter"})


def _provider_status(status: int) -> bool:
    return status in _PROVIDER_STATUSES or status >= 500


def provider_failed(res: LLMResult) -> bool:
    """The request failed at the provider of its model. A request cut short by the
    document's deadline is not asked here: the caller leaves it out first."""
    match res.failure:
        case "http":
            return res.status is not None and _provider_status(res.status)
        case "stream":  # a code that is not an integer says nothing against the provider
            return res.status is None or not 400 <= res.status < 500 or _provider_status(res.status)
        case "timeout" | "network" | "malformed" | "exception":
            return True
        case None:
            pass
        case _:
            assert_never(res.failure)
    if res.error is not None:  # an error that says nothing of how: not known to be theirs
        return False
    if res.finish_reason == "error":  # the stream's own marker of a failure mid-answer
        return True
    if not res.text.strip():  # JSON mode never answers nothing on purpose
        return res.finish_reason not in _EMPTY_BY_DESIGN
    # Every complete stream ends with a finish reason: without one, an answer that does
    # not parse was cut short on the way. One that parses is whole.
    return res.finish_reason is None and parse_json(res.text) is None
