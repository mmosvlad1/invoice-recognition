// The page's rules, without a browser. Run by tests/test_web_js.py (node --test).
import assert from "node:assert/strict";
import { test } from "node:test";

import * as model from "../../src/invoice_recognition/web/static/js/model.mjs";

test("an access code is printable ASCII without spaces, at least 16 characters", () => {
  assert.equal(model.checkCode("try-the-demo-page-0001"), null);
  assert.equal(model.checkCode("short-code"), "The access code has at least 16 characters.");
  assert.match(model.checkCode("has a space in it, long"), /without spaces/);
  assert.match(model.checkCode("ключ-доступу-довгий-1"), /without spaces/);
});

test("polling backs off 2, 4, 8, then every 15 seconds", () => {
  assert.deepEqual([1, 2, 3, 4, 5, 9].map(model.backoff), [2, 4, 8, 15, 15, 15]);
});

test("progress is a share of the pages, never above one", () => {
  assert.equal(model.progressShare({ pages_total: 4, pages_done: 1 }), 0.25);
  assert.equal(model.progressShare({ pages_total: 0, pages_done: 0 }), 0);
  assert.equal(model.progressShare({ pages_total: 2, pages_done: 3 }), 1);
});

test("the list's result column: rows, issues, and how many are still open", () => {
  const item = (line_count, issue_count, open_issue_count) => ({ line_count, issue_count, open_issue_count });
  assert.equal(model.listResult(item(null, null, null)), null);
  assert.deepEqual(model.listResult(item(110, 0, 0)), { text: "110 rows · no issues", open: null });
  assert.deepEqual(model.listResult(item(110, 5, 5)), { text: "110 rows · 5 issues", open: "5 open" });
  assert.deepEqual(model.listResult(item(110, 5, 0)), { text: "110 rows · 5 issues", open: "all checked" });
  assert.deepEqual(model.listResult(item(1, 1, undefined)), { text: "1 row · 1 issue", open: null });
});

test("an upload's refusal shows the server's sentence, a malformed request a fixed one", () => {
  const refusal = { status: 415, code: "unsupported_file", detail: "This is not a PDF, JPEG or PNG file." };
  assert.equal(model.uploadMessage(refusal), refusal.detail);
  const malformed = { status: 422, code: "invalid_request", detail: "body.file: Field required" };
  assert.equal(model.uploadMessage(malformed), "The upload could not be read. Try again.");
});

test("an error that is not a problem document gets a sentence by its status", () => {
  assert.equal(model.fallbackDetail(0), model.UNREACHABLE);
  assert.equal(model.fallbackDetail(413), "The file is larger than the limit of 10 MB.");
  assert.match(model.fallbackDetail(502), /\(502\)/);
});

test("an answer that arrives after its screen was left is not handed on", async () => {
  const life = model.screenLife();
  assert.equal(await life.guard(Promise.resolve(1)), 1);
  await assert.rejects(life.guard(Promise.reject(new Error("refused"))), /refused/);
  let answer;
  const late = life.guard(new Promise((resolve) => {
    answer = resolve;
  }));
  life.leave();
  answer({ status: "ok" });
  await assert.rejects(late, (e) => e === model.LEFT);
  await assert.rejects(life.guard(Promise.reject(new Error("refused"))), (e) => e === model.LEFT);
});

// ------------------------------------------------------------------ the review screen

// Shaped like the small sample: a line with three items, a page card, three issues that
// follow from the line.
const issue = (id, line_index, page, follows_from = [], suggestions = []) => ({
  id, code: "x", message: `issue ${id}`, line_index, page, fields: [], suggestions, follows_from,
});
const ITEMS = [
  { set: { "line.quantity": 9 }, source: "arithmetic" },
  { set: { "line.unit_price": 19.13 }, source: "arithmetic" },
  { set: { "line.amount": 688.68 }, source: "arithmetic", agrees_with: "header.subtotal" },
];
const RESULT = {
  lines: Array.from({ length: 60 }, (_, index) => ({ index, page: index < 30 ? 1 : 2 })),
  issues: [issue(0, 58, 2, [], ITEMS), issue(1, null, 2), issue(2, null, null, [0]), issue(3, null, null, [0]), issue(4, 52, 3, [0])],
};
const decided = (id, state, suggestion = null) => ({
  id, state, suggestion, set: suggestion == null ? null : ITEMS[suggestion].set, decided_at: null, settled_by: state === "settled" ? [0] : [],
});
const review = (...states) => ({
  open_count: states.filter((s) => s.state === "open").length,
  issues: RESULT.issues.map((i) => states.find((s) => s.id === i.id) ?? decided(i.id, "open")),
});
const ids = (issues) => issues.map((i) => i.id);

test("before any decision, what follows from a line sits under it", () => {
  for (const r of [null, review()]) {
    const placed = model.placeIssues(RESULT, r);
    assert.deepEqual([...placed.rows.keys()], [58]);
    assert.deepEqual(ids(placed.cards), [1]);
    assert.deepEqual(ids(placed.children.get(0)), [2, 3, 4]);
  }
});

test("accepting the amount keeps them under it, settled by the server", () => {
  const r = review(decided(0, "accepted", 2), decided(2, "settled"), decided(3, "settled"), decided(4, "settled"));
  const placed = model.placeIssues(RESULT, r);
  assert.deepEqual(ids(placed.children.get(0)), [2, 3, 4]);
  assert.deepEqual(ids(placed.cards), [1]);
});

test("accepting the quantity, or keeping the line, sends them out as their own issues", () => {
  for (const parent of [decided(0, "accepted", 0), decided(0, "kept")]) {
    const placed = model.placeIssues(RESULT, review(parent));
    assert.equal(placed.children.size, 0);
    assert.deepEqual(ids(placed.cards), [1, 2, 3]);
    assert.deepEqual(ids(placed.rows.get(52)), [4]);
  }
});

test("an issue decided on its own stays its own, whatever its parent", () => {
  const placed = model.placeIssues(RESULT, review(decided(0, "accepted", 2), decided(2, "kept")));
  assert.deepEqual(ids(placed.children.get(0)), [3, 4]);
  assert.deepEqual(ids(placed.cards), [1, 2]);
});

test("a second reading's item does not count as the amount's arithmetic", () => {
  const result = structuredClone(RESULT);
  result.issues[0].suggestions = [{ set: { "line.amount": 688.68 }, source: "second_reading" }];
  const r = review({ ...decided(0, "accepted"), suggestion: 0, set: { "line.amount": 688.68 } });
  assert.equal(model.placeIssues(result, r).children.size, 0);
});

test("the summary counts what is open", () => {
  assert.equal(model.summaryTitle(RESULT, null), "5 issues open");
  assert.equal(model.summaryTitle(RESULT, { open_count: 1 }), "1 issue open");
  assert.equal(model.summaryTitle(RESULT, { open_count: 0 }), "All 5 issues checked");
  assert.equal(model.summaryTitle({ issues: [] }, null), "Everything adds up");
});

test("the first open line is selected first", () => {
  assert.equal(model.firstOpenLine(RESULT, null, model.placeIssues(RESULT, null)), 58);
  const r = review(decided(0, "kept"));
  assert.equal(model.firstOpenLine(RESULT, r, model.placeIssues(RESULT, r)), 52);
  const none = { lines: [], issues: [] };
  assert.equal(model.firstOpenLine(none, null, model.placeIssues(none, null)), null);
});

test("the end of a reading is read out", () => {
  assert.equal(model.finishedText({ status: "needs_review", result: RESULT }), "Reading finished: 5 issues.");
  assert.equal(model.finishedText({ status: "ok", result: { issues: [] } }), "Reading finished: everything adds up.");
  assert.equal(model.finishedText({ status: "failed", error: { message: "Too blurred." } }), "Reading failed: Too blurred.");
});

test("the rows that passed: one, several, every one, none", () => {
  assert.equal(model.passedNote(110, 109, false), "The other 109 rows passed every check and are hidden.");
  assert.equal(model.passedNote(2, 1, false), "The other row passed every check and is hidden.");
  assert.equal(model.passedNote(5, 5, false), "Every row passed every check and is hidden.");
  assert.equal(model.passedNote(110, 109, true), "The other 109 rows passed every check.");
  assert.equal(model.passedNote(3, 0, false), "");
});

// ------------------------------------------------------------------ where an issue points

// An A4 page (595 x 842 points) with two located lines and one that was not located.
const PAGE = { page: 1, method: "layout", line_count: 3, repaired_count: 0, width: 595, height: 842, text_layer: true };
const boxes = (top) => ({
  row: [40, top, 555, top + 14], line_no: null, sku: [40, top, 90, top + 10], description: [100, top, 300, top + 10],
  quantity: [310, top, 340, top + 10], unit: null, unit_price: [400, top, 440, top + 10], tax_rate: null, amount: [500, top, 555, top + 10],
});
const PLACED = {
  pages: [PAGE, { page: 2, width: null, height: null }],
  lines: [
    { index: 0, page: 1, boxes: boxes(100) },
    { index: 1, page: 1, boxes: boxes(421) },
    { index: 2, page: 1, boxes: null },
    { index: 3, page: 2, boxes: boxes(100) },
  ],
};
const A4 = [[595, 842], [595, 842]];
const lineIssue = (line_index, fields, page = 1) => ({ id: 9, line_index, page, fields });
const close = (box, expected) => {
  for (const [key, v] of Object.entries(expected)) assert.ok(Math.abs(box[key] - v) < 1e-9, `${key}: ${box[key]} != ${v}`);
};

test("a value in doubt is framed in percent of its page", () => {
  const t = model.target(lineIssue(1, ["line.amount"]), PLACED, A4);
  assert.equal(t.page, 1);
  assert.equal(t.frames.length, 1);
  close(t.frames[0], { left: (500 / 595) * 100, top: 50, width: (55 / 595) * 100, height: (10 / 842) * 100 });
  assert.equal(t.band, null);
  assert.equal(model.target(lineIssue(1, ["line.quantity", "line.unit_price", "line.amount"]), PLACED, A4).frames.length, 3);
});

test("a value without a rectangle falls back to the row, a line without boxes to the page", () => {
  const unit = model.target(lineIssue(1, ["line.unit"]), PLACED, A4);
  assert.deepEqual(unit.frames, []);
  close(unit.band, { top: 50, left: (40 / 595) * 100 });
  assert.deepEqual(model.target(lineIssue(2, ["line.amount"]), PLACED, A4), { page: 1, frames: [], band: null });
});

test("no frame where the page's shape differs or its size is unknown", () => {
  const letter = [[612, 792]];
  assert.deepEqual(model.target(lineIssue(1, ["line.amount"]), PLACED, letter), { page: 1, frames: [], band: null });
  assert.equal(model.framesAllowed(PLACED, 1, [[595 * 1.004, 842]]), true); // within 0.5 %
  assert.equal(model.framesAllowed(PLACED, 1, [[595 * 1.006, 842]]), false);
  assert.equal(model.framesAllowed(PLACED, 1, [[595 * 2, 842 * 2]]), true); // /UserUnit: same shape
  assert.deepEqual(model.target(lineIssue(3, ["line.amount"], 2), PLACED, A4), { page: 2, frames: [], band: null });
  assert.equal(model.framesAllowed({ pages: [{ page: 1 }] }, 1, A4), false); // before page sizes reach the API
});

test("issues of a page, of the header or of nothing point at their page or nowhere", () => {
  assert.deepEqual(model.target({ id: 1, line_index: null, page: 2, fields: [] }, PLACED, A4), { page: 2, frames: [], band: null });
  assert.deepEqual(model.target({ id: 2, line_index: null, page: null, fields: ["header.subtotal"] }, PLACED, A4), null);
  const odd = structuredClone(PLACED);
  odd.lines[1].boxes.amount = [500, 421, "x", 431];
  assert.equal(model.target(lineIssue(1, ["line.amount"]), odd, A4).frames.length, 0);
});

test("the row under a point of the document", () => {
  assert.equal(model.lineAt(PLACED, 1, 10, A4).index, 0); // above every row: the page's first
  assert.equal(model.lineAt(PLACED, 1, 30, A4).index, 0);
  assert.equal(model.lineAt(PLACED, 1, 60, A4).index, 1);
  assert.equal(model.lineAt(PLACED, 2, 60, A4).index, 3); // no sizes for page 2: its first line
  assert.equal(model.lineAt(PLACED, 3, 60, A4), null);
  assert.equal(model.rowBox(PLACED.lines[2], PLACED, A4), null);
});
