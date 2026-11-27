# Invoice Recognition

A small FastAPI service that turns an invoice file (a digital PDF, a scan or a
photo) into structured data. The upload is acknowledged immediately and the
invoice is recognised in the background by an LLM-based flow; the result is
then verified by deterministic code checks (totals, line arithmetic, tax,
presence on the page) instead of being trusted as the model returned it.
Clients poll for the status and fetch the result once processing finishes.

> **Status:** project scaffold only. The recognition flow and the API are not
> implemented yet.
