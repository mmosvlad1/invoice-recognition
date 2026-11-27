// Markup is made one way only: the `html` tagged template, which escapes every value put
// into it unless that value is itself markup made by `html`. `put` is the one place that
// writes markup into the document. File names and everything read from an invoice are
// untrusted text; this is what keeps them text.

import { STATUS_LABEL, clock, count, dollars, plural, seconds, size, value } from "./format.mjs";
import { listResult, passedNote, placeIssues, progressShare, stateOf, summaryTitle } from "./model.mjs";

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const MADE_HERE = Symbol("html");

/** A piece of markup that is safe to put into the page. Only `html` makes one. */
export class Markup {
  #text;
  constructor(text, token) {
    if (token !== MADE_HERE) throw new TypeError("Markup is made only by html``");
    this.#text = text;
    Object.freeze(this);
  }
  toString() {
    return this.#text;
  }
}

function piece(value) {
  if (value == null || value === false) return "";
  if (value instanceof Markup) return value.toString();
  if (Array.isArray(value)) return value.map(piece).join("");
  return String(value).replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

/** The tagged template: html`<p>${text}</p>`. Attribute values must be quoted. */
export function html(strings, ...values) {
  if (!Array.isArray(strings?.raw)) throw new TypeError("html is used as a tag: html`...`");
  let out = strings[0];
  values.forEach((value, i) => {
    out += piece(value) + strings[i + 1];
  });
  return new Markup(out, MADE_HERE);
}

/** Replace the content of `element` with `markup`. Bars marked `data-fill="0.4"` get
 * their width here, through CSSOM: the page's CSP blocks style attributes in markup. */
export function put(element, markup) {
  if (!(markup instanceof Markup)) throw new TypeError("put() takes markup made by html``");
  element.innerHTML = markup.toString();
  for (const bar of element.querySelectorAll("[data-fill]")) {
    bar.style.width = `${Math.round(Number(bar.dataset.fill) * 1000) / 10}%`;
  }
}

// ------------------------------------------------------------------ shared pieces

export const ICON_WARN = html`<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.8 15 14H1z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M8 6.2v3.6M8 11.6v.1" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`;
export const ICON_OK = html`<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8.5" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="m6.2 10.3 2.5 2.5 5.1-5.3" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

export const badge = (status) => html`<span class="badge ${status}">${STATUS_LABEL[status] ?? status}</span>`;

/** A progress bar; `share` from 0 to 1. */
export const bar = (share, label) => html`<div class="bar" role="progressbar" aria-label="${label}"
  aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(share * 100)}"><span data-fill="${share}"></span></div>`;

/** "The access code is needed to continue." with the button that asks for it again. */
export const needsCode = () => html`<div class="empty needs-code">
  <p>The access code is needed to continue.</p>
  <button class="btn btn-primary" type="button" data-action="enter-code">Enter the access code</button></div>`;

// ------------------------------------------------------------------ the list of documents

const ID = /^[0-9a-f]{32}$/;

/** The address of a document's review screen; "#/" for an id that is not one. */
export const docHref = (id) => (ID.test(id) ? `#/doc/${id}` : "#/");

function docStatus(item) {
  if (item.status !== "processing") return badge(item.status);
  const p = item.progress;
  return html`<div class="progress">${badge(item.status)}
    <span class="progress-text">${p.pages_done} of ${plural(p.pages_total, "page")}</span>
    ${bar(progressShare(p), "Pages read")}</div>`;
}

function docResult(item) {
  if (item.status === "failed") return html`<span>${item.error?.message ?? "Could not be read."}</span>`;
  const result = listResult(item);
  if (!result) return html`<span class="muted">—</span>`;
  return html`${result.text}${result.open && html` · <strong>${result.open}</strong>`}`;
}

function docRow(item) {
  return html`<tr data-id="${item.id}">
    <td class="fname"><a href="${docHref(item.id)}">${item.filename || "(no name)"}</a><span class="muted col-size">${item.kind === "pdf" ? "PDF" : "Image"} · ${size(item.size_bytes)}</span></td>
    <td class="num col-pages">${item.progress.pages_total}</td>
    <td class="col-time muted">${clock(item.created_at)}</td>
    <td>${docStatus(item)}</td>
    <td>${docResult(item)}</td>
  </tr>`;
}

/** The documents table, or the line that says there is none. */
export function docList(items) {
  if (!items.length) {
    return html`<div class="empty">Nothing uploaded yet. Drop an invoice above or try one of the samples.</div>`;
  }
  return html`<table class="doctable">
    <thead><tr>
      <th>File</th><th class="num col-pages">Pages</th><th class="col-time">Uploaded</th><th>Status</th><th>Result</th>
    </tr></thead>
    <tbody>${items.map(docRow)}</tbody>
  </table>`;
}

// ------------------------------------------------------------------ the review screen

/** The badge of a document, with "all checked" once a person has decided every issue. */
export function reviewStatus(job) {
  const done = job.status === "needs_review" && job.review?.open_count === 0;
  return html`${badge(job.status)}${done && html` <span class="muted">· all checked</span>`}`;
}

/** A review screen with nothing but `content` and the way back. */
export const withBack = (content) => html`<section class="review">
  <a class="back" href="#/"><span aria-hidden="true">←</span> Documents</a>${content}</section>`;

/** The 404 view: the server's sentence and the way back. */
export const notFound = (detail) => withBack(html`<div class="empty">${detail}</div>`);

const headerDoubt = (result, review) =>
  new Set(
    result.issues
      .filter((i) => stateOf(review, i.id) === "open")
      .flatMap((i) => i.fields ?? [])
      .filter((f) => f.startsWith("header.")),
  );

/** The header strip: who, which invoice, the totals; amber where an open issue doubts it. */
export function headerFields(result, review) {
  const h = result.header ?? {};
  const doubt = headerDoubt(result, review);
  const rates = Array.isArray(h.tax_breakdown)
    ? h.tax_breakdown.filter((t) => t && typeof t === "object").map((t) => value("line.tax_rate", t.rate))
    : [];
  if (doubt.has("header.tax_breakdown")) doubt.add("header.tax_total"); // the table is shown with the tax
  const cell = (key, label, sub = null, cls = "") => {
    const field = `header.${key}`;
    return html`<div class="${cls}${doubt.has(field) ? " doubt" : ""}"><dt>${label}</dt><dd>${value(field, h[key])}${
      sub && html`<span class="sub">${sub}</span>`
    }</dd></div>`;
  };
  return html`${cell("supplier_name", "Supplier", h.supplier_tax_id != null && `Tax ID ${value("header.supplier_tax_id", h.supplier_tax_id)}`)}
    ${cell("invoice_number", "Invoice no.")}${cell("invoice_date", "Date")}${cell("currency", "Currency")}
    ${cell("subtotal", "Subtotal", null, "money")}${cell("tax_total", "Tax", rates.join(" · "), "money")}${cell("total", "Total", null, "money total")}`;
}

/** The right pane while the document waits or is being read. */
export function progressPanel(job) {
  if (job.status === "queued") {
    return html`<div class="state">
      <h2>Waiting to be read</h2>
      <p>The server reads a few documents at the same time (three unless set otherwise); this one starts as soon as a place is free.</p>
      ${bar(0, "Pages read")}
      <p class="fine">You can leave this page. The document stays in the list.</p></div>`;
  }
  const p = job.progress;
  return html`<div class="state">
    <h2>Reading the document…</h2>
    <p>${p.pages_done} of ${plural(p.pages_total, "page")} done.</p>
    ${bar(progressShare(p), "Pages read")}
    <p class="fine">A PDF with a text layer is read in one pass, so its counter may jump to the end at once. You can leave this page; the document stays in the list.</p></div>`;
}

/** The right pane of a document that could not be read. */
export function failedPanel(job) {
  const what = job.kind === "image" ? "photo" : "document";
  const result = job.result;
  const calls = result?.details?.calls;
  let cost = null;
  if (calls === 0) cost = "Nothing was sent to the model, so reading it cost nothing.";
  else if (typeof calls === "number") cost = `Reading it took ${plural(calls, "model request")} and cost ${dollars(result.details.cost_usd)}.`;
  const lines = Array.isArray(result?.lines) ? result.lines : [];
  return html`<div class="state failed">
      <h2>This ${what} can't be read</h2>
      <p class="reason">${job.error?.message ?? "It could not be read."}</p>
      <div class="actions"><a class="btn btn-primary" href="#/" data-action="upload-another">Upload another file</a></div>
      ${cost && html`<p class="fine">${cost}</p>`}
    </div>
    ${lines.length > 0 && html`<details class="read-before"><summary>Rows read before it stopped (${count(lines.length)})</summary>
      <div class="rows">${rowsHead}${rowList(lines, { rows: new Map(), children: new Map() }, null, null)}</div></details>`}
    ${result && details(result)}`;
}

const rowsHead = html`<div class="rows-head" aria-hidden="true">
  <span class="h-sku">Item code</span><span>Description</span><span class="num">Qty</span><span class="h-unit">Unit</span>
  <span class="num h-price">Price</span><span class="num h-tax">Tax</span><span class="num">Amount</span></div>`;

const COLUMNS = [
  ["sku", "sku"],
  ["description", "desc"],
  ["quantity", "num qty"],
  ["unit", "unit"],
  ["unit_price", "num price"],
  ["tax_rate", "num tax"],
  ["amount", "num amount"],
];

function cells(line, issues, review) {
  const doubt = new Set(
    issues.filter((i) => stateOf(review, i.id) === "open").flatMap((i) => i.fields ?? []),
  );
  return COLUMNS.map(([key, cls]) => {
    const text = value(`line.${key}`, line[key]);
    if (key === "description") return html`<span class="cell desc" title="${text}">${text}</span>`;
    if (doubt.has(`line.${key}`)) return html`<span class="cell ${cls} doubt" title="In doubt"><span>${text}</span></span>`;
    return html`<span class="cell ${cls}">${text}</span>`;
  });
}

function follows(child, review) {
  if (stateOf(review, child.id) === "settled") {
    return html`<p class="follows is-settled">${ICON_OK}<span>Settled by the amount accepted above: ${child.message}</span></p>`;
  }
  return html`<p class="follows"><span><span class="follows-label">This also explains:</span> ${child.message}</span></p>`;
}

function why(issue, placed, review) {
  const children = placed.children.get(issue.id) ?? [];
  return html`<div class="why">
    <p class="why-text">${ICON_WARN}<span>${issue.message}</span></p>
    ${children.map((child) => follows(child, review))}
  </div>`;
}

function row(line, issues, placed, review, selected) {
  const problem = issues.length > 0;
  const label = `Row on page ${line.page ?? "unknown"}: ${value("line.sku", line.sku)} ${value("line.description", line.description)}. Show on the page`;
  return html`<div class="row${problem ? " problem" : ""}${line.index === selected ? " is-selected" : ""}" data-line="${line.index}">
    <div class="row-main" tabindex="0" role="button" data-focus="row-${line.index}" aria-label="${label}">${cells(line, issues, review)}</div>
    ${issues.map((issue) => why(issue, placed, review))}
  </div>`;
}

function rowList(lines, placed, review, selected) {
  let page;
  return lines.map((line) => {
    const separator = line.page !== page && line.page != null && html`<p class="page-sep">Page ${line.page}</p>`;
    page = line.page;
    return html`${separator}${row(line, placed.rows.get(line.index) ?? [], placed, review, selected)}`;
  });
}

function card(issue, placed, review) {
  const where = issue.page != null ? `Page ${issue.page}` : "Document";
  return html`<div class="docissue" data-issue="${issue.id}">
    <button class="docissue-title" type="button" data-page="${issue.page ?? ""}" data-focus="card-${issue.id}">${ICON_WARN} ${where}</button>
    <p>${issue.message}</p>
    ${(placed.children.get(issue.id) ?? []).map((child) => follows(child, review))}
  </div>`;
}

/** The right pane of a read document: summary, cards, rows, details. */
export function resultPane(job, { showAll, selected }) {
  const result = job.result;
  const review = job.review;
  const placed = placeIssues(result, review);
  const total = result.lines.length;
  const shown = showAll ? result.lines : result.lines.filter((l) => placed.rows.has(l.index));
  const hidden = total - placed.rows.size;
  const ok = result.issues.length === 0;
  const note = ok
    ? `${plural(total, "row")} read. Every row, the tax table and the totals match, and nothing needs a look.`
    : passedNote(total, hidden, showAll);
  return html`<div class="summary${ok ? " ok" : ""}">
      <h2>${ok && ICON_OK}${summaryTitle(result, review)}</h2>
      ${note && html`<p>${note}</p>`}
      <div><button class="switch" type="button" role="switch" aria-checked="${String(showAll)}" id="toggle-all" data-focus="toggle-all">
        <span class="track"></span><span>Show all ${plural(total, "row")}</span></button></div>
    </div>
    ${placed.cards.map((issue) => card(issue, placed, review))}
    ${shown.length > 0 && html`<div class="rows" id="rows">${rowsHead}${rowList(shown, placed, review, selected)}</div>`}
    ${details(result)}`;
}

// ------------------------------------------------------------------ details (diagnostics)

const READING_PATH = {
  layout: "Table read by code from the text layer",
  model: "Page by page by the model",
  mixed: "Partly by code, partly by the model",
};
const PAGE_METHOD = { layout: "code (text layer)", model: "model", skipped: "skipped" };
const ATTEMPT = { header: "header", layout: "table layout", page: "page", second_reading: "second reading" };

const kv = (label, text) => html`<div><dt>${label}</dt><dd>${text}</dd></div>`;
const isNumber = (n) => typeof n === "number" && Number.isFinite(n);
/** The entries of a list that are objects; anything else is left out. */
const objects = (list) => (Array.isArray(list) ? list.filter((x) => x && typeof x === "object") : []);
const table = (head, rows) =>
  html`<table class="mini"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;

function layoutCheck(layout) {
  if (!layout || typeof layout !== "object" || layout.status == null) return null;
  const witness = isNumber(layout.witness_page) ? ` · witness page ${layout.witness_page}` : "";
  return `${value("", layout.status)}${witness}`;
}

function quality(q) {
  if (!q || typeof q !== "object") return null;
  if (Array.isArray(q.pages)) {
    const pages = objects(q.pages);
    return html`<div><h3>Page quality</h3>${table(
      html`<th>Page</th><th>Verdict</th><th class="num">Sharpness</th>`,
      pages.map((p) => html`<tr><td>${value("", p.page)}</td><td>${value("", p.verdict)}</td>
        <td class="num">${isNumber(p.measures?.text_sharpness) ? Math.round(p.measures.text_sharpness) : "—"}</td></tr>`),
    )}</div>`;
  }
  if (q.verdict == null) return null;
  const sharpness = q.measures?.text_sharpness;
  const limit = q.limits?.sharpness_reject;
  return html`<div><h3>Photo quality</h3><dl class="kv">
    ${kv("Verdict", value("", q.verdict))}
    ${isNumber(sharpness) && kv("Sharpness", `${Math.round(sharpness)}${isNumber(limit) ? ` (refused below ${limit})` : ""}`)}
  </dl></div>`;
}

/** "Details": how the document was read, what it cost, each request, notes, quality. */
export function details(result) {
  const d = result.details && typeof result.details === "object" ? result.details : {};
  const unknown = isNumber(d.cost_unknown_calls) && d.cost_unknown_calls > 0;
  const cost = `${dollars(d.cost_usd ?? 0)}${unknown ? ` + ${plural(d.cost_unknown_calls, "request")} without a reported cost` : ""}`;
  const pages = objects(result.pages);
  const attempts = objects(d.attempts);
  const notes = objects(result.notes);
  const check = layoutCheck(d.layout);
  return html`<details class="details"><summary>Details: time, cost, how each page was read</summary><div class="details-body">
    <dl class="kv">
      ${kv("Reading path", READING_PATH[d.method] ?? value("", d.method))}
      ${check && kv("Layout check", check)}
      ${kv("Time", seconds(d.duration_s ?? 0))}
      ${kv("Cost", cost)}
      ${kv("Requests", count(d.calls ?? 0))}
      ${kv("Tokens", `${count(d.prompt_tokens ?? 0)} in · ${count(d.completion_tokens ?? 0)} out`)}
    </dl>
    ${pages.length > 0 && html`<div><h3>Pages</h3>${table(
      html`<th>Page</th><th>Read by</th><th class="num">Rows</th><th class="num">Quantities repaired</th>`,
      pages.map((p) => html`<tr><td>${p.page}</td><td>${PAGE_METHOD[p.method] ?? p.method}</td>
        <td class="num">${count(p.line_count)}</td><td class="num">${count(p.repaired_count)}</td></tr>`),
    )}</div>`}
    ${attempts.length > 0 && html`<div><h3>Model requests</h3>${table(
      html`<th>Step</th><th>Page</th><th>Model</th><th class="num">Time</th><th class="num">Cost</th><th class="num">Rows</th><th class="num">Issues</th><th>Error</th>`,
      attempts.map((a) => html`<tr><td>${ATTEMPT[a.what] ?? a.what}</td><td>${a.page ?? "—"}</td><td class="model">${a.model}</td>
        <td class="num">${seconds(a.duration_s)}</td><td class="num">${a.cost_usd == null ? "—" : dollars(a.cost_usd)}</td>
        <td class="num">${a.line_count}</td><td class="num">${a.issue_count}</td><td>${a.error ?? ""}</td></tr>`),
    )}</div>`}
    ${notes.length > 0 && html`<div><h3>Notes</h3>${notes.map((n) => html`<p>${n.page != null && `Page ${n.page}: `}${value("", n.text)}</p>`)}</div>`}
    ${quality(d.quality)}
  </div></details>`;
}
