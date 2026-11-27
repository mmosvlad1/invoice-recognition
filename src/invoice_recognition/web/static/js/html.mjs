// Markup is made one way only: the `html` tagged template, which escapes every value put
// into it unless that value is itself markup made by `html`. `put` is the one place that
// writes markup into the document. File names and everything read from an invoice are
// untrusted text; this is what keeps them text.

import { STATUS_LABEL, clock, plural, size } from "./format.mjs";
import { listResult, progressShare } from "./model.mjs";

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
