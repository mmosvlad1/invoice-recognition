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
