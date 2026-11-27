"""Reading one invoice file and checking what was read.

PDF pages, evidence from the text layer, deterministic checks and the flow that
ties them together live here. Requests to the model go through the top-level
`invoice_recognition.llm`.

The entry point is `process(data, mime, client, cfg)`; it returns a `Result`.
"""

from invoice_recognition.recognition.config import Config
from invoice_recognition.recognition.pipeline import process
from invoice_recognition.recognition.results import Result

__all__ = ["Config", "Result", "process"]
