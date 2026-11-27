// Formatting for people. Run by tests/test_web_js.py (node --test).
import assert from "node:assert/strict";
import { test } from "node:test";

import { clock, count, dollars, fieldName, plural, seconds, size, value } from "../../src/invoice_recognition/web/static/js/format.mjs";

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

test("fields are named as a person says them", () => {
  assert.equal(fieldName("line.unit_price"), "price");
  assert.equal(fieldName("header.supplier_tax_id"), "supplier tax ID");
  assert.equal(fieldName("line.colour_code"), "colour code");
});

test("values: numbers by field, text as read, null as a dash, other JSON as JSON", () => {
  assert.equal(value("line.amount", 688.68), "688.68");
  assert.equal(value("header.total", 121686.57), "121,686.57");
  assert.equal(value("line.unit_price", 0.2477), "0.2477");
  assert.equal(value("line.quantity", 9), "9");
  assert.equal(value("line.tax_rate", 7), "7%");
  assert.equal(value("line.quantity", "8.200"), "8.200");
  assert.equal(value("line.quantity", "8.200", { numeric: true }), "8.2");
  assert.equal(value("line.sku", "SG-54066", { numeric: true }), "SG-54066");
  assert.equal(value("line.amount", null), "—");
  assert.equal(value("header.tax_breakdown", [{ rate: 7 }]), '[{"rate":7}]');
  assert.equal(value("line.amount", true), "true");
});

test("money, seconds, dollars and counts", () => {
  assert.equal(dollars(0.0055), "$0.0055");
  assert.equal(seconds(21.14), "21.1 s");
  assert.equal(count(1250), "1,250");
});
