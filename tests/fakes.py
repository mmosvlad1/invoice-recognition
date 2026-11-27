"""Shared test helpers: fake model answers and generated documents. No network."""

import json


def sse(text: str, cost: float | None = 0.001) -> str:
    """A streamed answer the way OpenRouter sends it: content, then usage, then [DONE].

    cost=None leaves the cost out of the usage, as some providers do.
    """
    usage = {"prompt_tokens": 100, "completion_tokens": 50} | (
        {"cost": cost} if cost is not None else {}
    )
    events = [
        {"choices": [{"delta": {"content": text}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage},
    ]
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
