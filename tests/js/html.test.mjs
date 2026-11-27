// Markup: the tagged template escapes, `put` takes only its markup, the builders keep
// hostile text as text. Run by tests/test_web_js.py (node --test), which writes real API
// answers to API_SAMPLES_DIR first.
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import * as html from "../../src/invoice_recognition/web/static/js/html.mjs";

const { Markup, docHref, docList, put } = html;
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
