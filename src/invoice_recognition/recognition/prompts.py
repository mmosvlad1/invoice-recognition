"""Prompts of the page-by-page flow: line items of one page, and the header."""

LINE_RULES = """\
Rules for line items:
- Return EVERY line item, in document order. Never merge, summarise or skip lines.
- A line item is a product, service, charge or discount. These are NOT line items and
  must be left out: page subtotals, "carried forward" / "Übertrag" rows, section or
  delivery-note subtotals, section headings, the tax summary and the grand totals.
- Two identical rows are two line items. Keep both.
- Discounts are line items with a negative amount.
- Copy sku and description exactly as printed. Join a wrapped description into one string.
- Numbers are JSON numbers with a dot as decimal separator and no thousands separators.
  Work out the document's number format first: "1.234,56" means 1234.56 and "5.000" pcs
  means 5000 when the document uses a comma as the decimal separator.
- tax_rate is the percentage as a number (20 for 20%). amount is the net line total.
- If a value is unreadable, use null. Never guess and never calculate a value that is
  not printed."""


EXTRACT_PAGE = """\
You read ONE page of a supplier invoice: page {page} of {total}. The other pages are read
separately. Return only what is printed on THIS page, top to bottom.

Return one JSON object and nothing else:

{{
  "rows": [
    {{"line_no": int or null, "sku": str, "description": str, "quantity": number, "unit": str,
      "unit_price": number, "tax_rate": number, "amount": number}},
    {{"total_label": str, "amount": number}}
  ]
}}

"rows" lists, in the order printed, two kinds of entries:
- line items (first shape). line_no is the position number printed in front of the line;
  null if lines are not numbered.
- control totals printed among the line items (second shape): page subtotals, "carried
  forward" / "Übertrag" rows, section or delivery-note subtotals. Copy the label and the
  amount as printed, at the place where they are printed.

Leave out the document header, the tax summary and the grand totals at the end of the
invoice. If the page has no line items, return {{"rows": []}}.

""" + LINE_RULES.replace(
    "must be left out:",
    "must NOT be returned as line items (report them as control totals instead):",
)

EXTRACT_HEADER = """\
You read the header and the totals of a supplier invoice. You are given its first page and,
if the invoice is longer, its last page. Do NOT return line items. Return one JSON object
and nothing else:

{
  "supplier_name": str, "supplier_tax_id": str, "invoice_number": str,
  "invoice_date": "YYYY-MM-DD", "due_date": "YYYY-MM-DD", "po_number": str,
  "currency": "ISO 4217 code",
  "subtotal": number, "tax_total": number, "total": number,
  "tax_breakdown": [{"rate": number, "base": number, "tax": number}]
}

subtotal is the net amount before tax, tax_total the sum of all taxes, total the amount due.
tax_breakdown has one entry per tax rate printed in the tax summary: rate is the percentage
(20 for 20%), base the net amount taxed at that rate, tax the tax amount.
Numbers are JSON numbers with a dot as decimal separator and no thousands separators; work
out the document's number format first ("1.234,56" means 1234.56 when the document uses a
comma as the decimal separator). If a value is not printed, use null. Never guess."""
