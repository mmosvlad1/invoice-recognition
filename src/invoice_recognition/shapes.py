"""Names for the JSON shapes that flow through recognition.

The data model is deliberately not a set of classes: line items and the rest of
an extraction stay the dicts parsed from the model's JSON answer, because every
check has to cope with whatever the model returned (a missing key, a number sent
as a string, a list where an object was expected). These aliases only give the
shapes a name for readers and for the type checker.

This module imports nothing from the package, so any module can import it
without creating an import cycle.
"""

from typing import Any

type JSONObject = dict[str, Any]
"""Any JSON object, as parsed from a model's answer."""

type Line = dict[str, Any]
"""One line item: sku, description, quantity, unit, unit_price, tax_rate, amount,
and optionally line_no (the position number printed in front of it) and page."""
