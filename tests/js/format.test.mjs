// Formatting for people. Run by tests/test_web_js.py (node --test).
import assert from "node:assert/strict";
import { test } from "node:test";

import { clock, plural, size } from "../../src/invoice_recognition/web/static/js/format.mjs";

test("plural counts with one form or the other and a thousands separator", () => {
  assert.equal(plural(1, "row"), "1 row");
  assert.equal(plural(0, "row"), "0 rows");
  assert.equal(plural(1250, "row"), "1,250 rows");
  assert.equal(plural(2, "entry", "entries"), "2 entries");
});

test("sizes are in KB below a megabyte, then in MB", () => {
  assert.equal(size(100), "1 KB");
  assert.equal(size(15_516), "15 KB");
  assert.equal(size(1_400_000), "1.3 MB");
});

test("a time of day, or nothing", () => {
  assert.match(clock("2026-11-27T15:02:11Z"), /^\d\d:\d\d$/);
  assert.equal(clock(null), "");
});
