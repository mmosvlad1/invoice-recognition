// Markup: the tagged template escapes, `put` takes only its markup, the builders keep
// hostile text as text. Run by tests/test_web_js.py (node --test), which writes real API
// answers to API_SAMPLES_DIR first.
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import * as html from "../../src/invoice_recognition/web/static/js/html.mjs";

const { Markup, details, docHref, docList, failedPanel, headerFields, notFound, progressPanel, put, resultPane, reviewStatus } = html;
const tag = html.html;

test("every interpolated value is escaped", () => {
  const out = String(tag`<p title="${`"'<>&`}">${"<b>&</b>"}</p>`);
  assert.equal(out, '<p title="&quot;&#39;&lt;&gt;&amp;">&lt;b&gt;&amp;&lt;/b&gt;</p>');
});

test("markup passes, arrays join, nothing and false are empty, numbers are text", () => {
  const inner = tag`<i>${"x"}</i>`;
  assert.equal(String(tag`${inner}|${[inner, "<"]}|${null}${undefined}${false}|${0}`), "<i>x</i>|<i>x</i>&lt;||0");
});

test("only the tag makes markup, and only as a tag", () => {
  assert.throws(() => new Markup("<b>", Symbol("html")), TypeError);
  assert.throws(() => html.html(["<b>"]), TypeError);
  assert.ok(Object.isFrozen(tag`x`));
});

test("put() refuses a string and sets the width of bars through CSSOM", () => {
  const bar = { dataset: { fill: "0.25" }, style: {} };
  const element = { innerHTML: "", querySelectorAll: () => [bar] };
  assert.throws(() => put(element, "<b>hi</b>"), TypeError);
  put(element, tag`<span data-fill="${0.25}"></span>`);
  assert.equal(element.innerHTML, '<span data-fill="0.25"></span>');
  assert.equal(bar.style.width, "25%");
});

test("a link is built only for an id of 32 hex digits", () => {
  assert.equal(docHref("0123456789abcdef0123456789abcdef"), "#/doc/0123456789abcdef0123456789abcdef");
  assert.equal(docHref('javascript:alert(1)//"'), "#/");
});

// ------------------------------------------------------------------ real answers, made hostile

// One payload closes an attribute and opens tags; the other only adds an attribute, which
// works where a value is put into an attribute without quotes.
const PAYLOAD = '"><img src=x onerror=alert(1)><svg onload=alert(2)>';
const PAYLOADS = [PAYLOAD, "x onmouseover=alert(1)"];
const DIR = process.env.API_SAMPLES_DIR;
const answers = (suffix) =>
  DIR ? readdirSync(DIR).filter((f) => f.endsWith(suffix)).map((f) => JSON.parse(readFileSync(join(DIR, f), "utf8"))) : [];

// Keys whose value is any JSON read from the document: replaced whole, numbers included;
// of an object of such values (the header, a suggestion's cells), each value in turn.
const ANY_JSON = new Set(["quality", "layout", "tax_breakdown"]);
const VALUES = new Set(["header", "set"]);
const LINE_VALUES = new Set(["line_no", "sku", "description", "quantity", "unit", "unit_price", "tax_rate", "amount", "label"]);

/** A copy with every string, and every value read from the document, set to `payload`. */
function hostile(value, payload = PAYLOAD, key = null) {
  if (key && (ANY_JSON.has(key) || LINE_VALUES.has(key))) return payload;
  if (key && VALUES.has(key) && value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).map((k) => [k, payload]));
  }
  if (typeof value === "string") return payload;
  if (Array.isArray(value)) return value.map((v) => hostile(v, payload));
  if (value && typeof value === "object") {
    const out = { extra_key: payload };
    for (const [k, v] of Object.entries(value)) out[k] = hostile(v, payload, k);
    return out;
  }
  return value;
}

const escaped = (text) => text.replace(/[<>"]/g, (c) => ({ "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function assertSafe(markup, what) {
  const out = String(markup);
  assert.ok(!out.includes(PAYLOAD), `${what}: the payload came through`);
  assert.ok(!out.includes("<img") && !out.includes("<svg onload"), `${what}: a tag came through`);
  for (const tag of out.match(/<[^>]*>/g) ?? []) {
    // Inside a tag, with every quoted value taken out, no event handler may be left.
    assert.doesNotMatch(tag.replace(/"[^"]*"/g, '""'), /\son\w+=/, `${what}: an attribute came through`);
  }
}

test("the API answers were written for these tests", { skip: !DIR && "run through pytest" }, () => {
  assert.equal(answers(".item.json").length, 3);
  assert.equal(answers(".job.json").length, 3);
});

test("the list keeps hostile values as text", { skip: !DIR && "run through pytest" }, () => {
  const items = answers(".item.json");
  assertSafe(docList(items), "the list as answered");
  for (const payload of PAYLOADS) {
    const bad = items.map((item) => hostile(item, payload));
    assertSafe(docList(bad), "the list made hostile");
    assert.ok(String(docList(bad)).includes(escaped(payload)), "the payload was shown, as text");
    for (const item of items) {
      for (const status of ["queued", "processing", "failed"]) {
        assertSafe(docList([{ ...hostile(item, payload), status }]), `a ${status} item made hostile`);
      }
    }
  }
});

/** Every builder of the review screen, on one job. */
function reviewMarkup(job) {
  const out = [reviewStatus(job), progressPanel({ ...job, status: "queued" }), progressPanel({ ...job, status: "processing" })];
  if (job.result) {
    out.push(headerFields(job.result, job.review), details(job.result), failedPanel({ ...job, status: "failed" }));
    for (const showAll of [false, true]) out.push(resultPane(job, { showAll, selected: 0 }));
  } else {
    out.push(failedPanel(job));
  }
  return out;
}

test("the review screen keeps hostile values as text", { skip: !DIR && "run through pytest" }, () => {
  for (const job of answers(".job.json")) {
    for (const markup of reviewMarkup(job)) assertSafe(markup, `${job.filename} as answered`);
    for (const payload of PAYLOADS) {
      const bad = hostile(job, payload);
      // The structure the screen relies on stays; every value in it is hostile.
      bad.status = job.status;
      if (bad.result) {
        bad.result.lines = job.result.lines.map((l) => ({ ...hostile(l, payload), index: l.index, page: l.page }));
        bad.result.issues = job.result.issues.map((i) => ({
          ...hostile(i, payload), id: i.id, line_index: i.line_index, follows_from: i.follows_from,
        }));
      }
      const markups = reviewMarkup(bad);
      for (const markup of markups) assertSafe(markup, `${job.filename} made hostile`);
      assert.ok(markups.join("").includes(escaped(payload)), "the payload was shown, as text");
    }
    assertSafe(notFound(PAYLOAD), "the 404 view");
  }
});

// ------------------------------------------------------------------ odd but possible answers

const LINE = { index: 0, page: null, line_no: null, sku: "A-1", description: "Bolts", quantity: "8.200", unit: "pcs", unit_price: 1.5, tax_rate: 20, amount: 12.3 };
const DETAILS = {
  method: "mixed", layout: null, evidence_source: null, duration_s: 3.25, cost_usd: 0.002, cost_unknown_calls: 1, calls: 2,
  prompt_tokens: 1200, completion_tokens: 300, quality: {},
  attempts: [{ what: "page", page: 1, model: "m", duration_s: 1, cost_usd: null, cost_known: false, prompt_tokens: 1, completion_tokens: 1, line_count: 1, issue_count: 0, error: "timed out" }],
};
const RESULT = {
  header: { supplier_name: "ACME", tax_breakdown: "20% on everything", subtotal: "12.30", total: null },
  lines: [LINE], issues: [], notes: [{ kind: "x", page: null, text: "A note." }], checkpoints: [],
  pages: [{ page: 1, method: "model", line_count: 1, repaired_count: 0 }], details: DETAILS,
};

test("a failed reading with rows shows them read-only, and its cost", () => {
  const job = { id: "x", filename: "f.pdf", kind: "pdf", status: "failed", error: { code: "deadline_exceeded", message: "Stopped." }, result: RESULT, review: null };
  const out = String(failedPanel(job));
  assert.match(out, /Rows read before it stopped \(1\)/);
  assert.match(out, /cost \$0\.0020/);
  assert.match(out, /8\.200/); // as read
  assert.doesNotMatch(out, /Page null|undefined/);
  const nothingSent = String(failedPanel({ ...job, kind: "image", result: { ...RESULT, lines: [], details: { ...DETAILS, calls: 0 } } }));
  assert.match(nothingSent, /This photo can't be read/);
  assert.match(nothingSent, /Nothing was sent to the model, so reading it cost nothing/);
  assert.match(String(failedPanel({ ...job, result: null })), /Stopped\./);
});

test("details show the path, unknown costs and odd values without throwing", () => {
  const out = String(details(RESULT));
  assert.match(out, /Partly by code, partly by the model/);
  assert.match(out, /\+ 1 request without a reported cost/);
  assert.match(out, /<td class="num">—<\/td>/); // a request without a cost
  assert.match(String(details({ ...RESULT, details: { ...DETAILS, method: "model" } })), /Page by page by the model/);
  assert.match(String(details({ ...RESULT, details: { ...DETAILS, method: "new_path" } })), /new_path/);
  assert.doesNotThrow(() => details({ ...RESULT, details: { ...DETAILS, quality: { pages: "none", verdict: 3 } } }));
});

test("the header strip takes a tax table that is not a list, and text for numbers", () => {
  const out = String(headerFields(RESULT, null));
  assert.match(out, /<dd>12\.30<\/dd>/);
  assert.match(out, /<dd>—<\/dd>/);
  assert.doesNotMatch(out, /20% on everything/);
});

test("the summary of a document where everything adds up", () => {
  const out = String(resultPane({ result: RESULT, review: { open_count: 0, issues: [] } }, { showAll: false, selected: null }));
  assert.match(out, /Everything adds up/);
  assert.match(out, /aria-checked="false"/);
  assert.doesNotMatch(out, /class="row/);
  assert.match(String(resultPane({ result: RESULT, review: null }, { showAll: true, selected: 0 })), /row is-selected/);
});

test("an entry of a list that is not an object is left out of the details", () => {
  const odd = { ...RESULT, notes: [null, 3, ...RESULT.notes], pages: [null, ...RESULT.pages], details: { ...DETAILS, attempts: [null, "x", ...DETAILS.attempts] } };
  const out = String(details(odd));
  assert.equal((out.match(/<tr><td>/g) ?? []).length, 2); // one page, one request
  assert.match(out, /<dt>Requests<\/dt>/);
  assert.match(out, /<h3>Model requests<\/h3>/);
});

test("a doubtful cell says so on hover, a row says what selecting it does", { skip: !DIR && "run through pytest" }, () => {
  const job = answers(".job.json").find((j) => j.status === "needs_review");
  const out = String(resultPane(job, { showAll: false, selected: null }));
  assert.match(out, /class="cell num amount doubt" title="In doubt"/);
  assert.match(out, /aria-label="Row on page 2: [^"]*\. Show on the page"/);
});
