// Numbers, sizes, times and labels as a person reads them. Pure: no DOM, no requests.

const numbers = (min, max) =>
  new Intl.NumberFormat("en-GB", { minimumFractionDigits: min, maximumFractionDigits: max });
const COUNT = numbers(0, 0);
const MONEY = numbers(2, 2);
const PRICE = numbers(2, 4);
const QUANTITY = numbers(0, 4);

/** The page's name, in the header and in every tab title. */
export const APP_NAME = "Invoice reader";

export const STATUS_LABEL = {
  queued: "Waiting",
  processing: "Reading",
  ok: "Ready",
  needs_review: "Needs review",
  failed: "Could not read",
};

/** "1 page", "4 pages"; counts above 999 with a separator: "1,250 rows". */
export const plural = (n, one, many = `${one}s`) => `${COUNT.format(n)} ${n === 1 ? one : many}`;

/** A file size: "15 KB", "1.4 MB". */
export const size = (bytes) =>
  bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1048576).toFixed(1)} MB`;

/** The time of day of an ISO timestamp, "14:05"; "" for none. */
export const clock = (iso) =>
  iso ? new Date(iso).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }) : "";

/** Seconds with one decimal: "21.1 s". */
export const seconds = (s) => `${Number(s).toFixed(1)} s`;

/** Dollars with four decimals, as providers bill: "$0.0055". */
export const dollars = (usd) => `$${Number(usd).toFixed(4)}`;

/** A sum of money: "103,495.24". */
export const money = (n) => MONEY.format(n);

/** A count with a thousands separator: "1,250". */
export const count = (n) => COUNT.format(n);

// The names a person reads for the fields of a line and of the header.
const FIELD_NAMES = {
  sku: "item code",
  description: "description",
  quantity: "quantity",
  unit: "unit",
  unit_price: "price",
  tax_rate: "tax rate",
  amount: "amount",
  line_no: "position",
  supplier_name: "supplier",
  supplier_tax_id: "supplier tax ID",
  invoice_number: "invoice number",
  invoice_date: "date",
  due_date: "due date",
  po_number: "PO number",
  currency: "currency",
  subtotal: "subtotal",
  tax_total: "tax",
  total: "total",
  tax_breakdown: "tax table",
};

const bare = (qualified) => String(qualified).replace(/^(line|header)[.]/, "");

/** "line.unit_price" -> "price"; an unknown key with its underscores as spaces. */
export const fieldName = (qualified) => FIELD_NAMES[bare(qualified)] ?? bare(qualified).replaceAll("_", " ");

const BY_FIELD = {
  amount: money,
  subtotal: money,
  tax_total: money,
  total: money,
  unit_price: (n) => PRICE.format(n),
  quantity: (n) => QUANTITY.format(n),
  tax_rate: (n) => `${QUANTITY.format(n)}%`,
};

/**
 * A value read from the document, as text. Numbers are formatted for their field; text
 * is shown exactly as read ("8.200" stays "8.200") unless `numeric` (a suggested value,
 * computed by the code) and it holds a number; null is "—"; anything else is its JSON.
 * Never markup: the caller puts it into the template, which escapes it.
 */
export function value(qualified, v, { numeric = false } = {}) {
  if (v == null) return "—";
  const format = BY_FIELD[bare(qualified)];
  if (typeof v === "number") return format && Number.isFinite(v) ? format(v) : String(v);
  if (typeof v === "string") {
    const n = Number(v);
    return numeric && format && v.trim() !== "" && Number.isFinite(n) ? format(n) : v;
  }
  return JSON.stringify(v);
}
