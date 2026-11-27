// The poller: when it polls again, backs off and stops. Run by tests/test_web_js.py.
// `document` is the one browser object it uses: a stand-in here.
import assert from "node:assert/strict";
import { test } from "node:test";

globalThis.document = { hidden: false, addEventListener() {}, removeEventListener() {} };
const api = await import("../../src/invoice_recognition/web/static/js/api.mjs");

const tick = () => new Promise((resolve) => setImmediate(resolve));

function setup(t, step) {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const banner = [];
  api.onConnection((lost) => banner.push(lost));
  const calls = { n: 0 };
  const poller = api.poller(async () => {
    calls.n += 1;
    return step(calls.n);
  });
  return { poller, banner, calls };
}

test("polls every second while the step asks for more, and stops when it does not", async (t) => {
  const { calls } = setup(t, (n) => n < 3);
  await tick();
  for (let i = 0; i < 5; i++) {
    t.mock.timers.tick(1000);
    await tick();
  }
  assert.equal(calls.n, 3);
});

test("a failed step backs off 2, 4, 8 s with the banner, and a success hides it", async (t) => {
  const { calls, banner } = setup(t, (n) => {
    if (n <= 3) throw new api.Problem(0, null, "unreachable");
    return false;
  });
  await tick();
  assert.equal(calls.n, 1);
  for (const [wait, expected] of [[1999, 1], [1, 2], [4000, 3], [7999, 3], [1, 4]]) {
    t.mock.timers.tick(wait);
    await tick();
    assert.equal(calls.n, expected, `after ${wait} ms`);
  }
  assert.deepEqual(banner, [true, true, true, false]);
});

test("a missing code stops polling until kicked", async (t) => {
  const { poller, calls, banner } = setup(t, (n) => {
    if (n === 1) throw new api.Problem(401, "no_code", "needed");
    return false;
  });
  await tick();
  t.mock.timers.tick(60_000);
  await tick();
  assert.equal(calls.n, 1);
  assert.deepEqual(banner, []);
  poller.kick();
  await tick();
  assert.equal(calls.n, 2);
});

test("an answer that arrives after stop() shows nothing and polls no more", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const banner = [];
  api.onConnection((lost) => banner.push(lost));
  let fail;
  let calls = 0;
  const poller = api.poller(() => {
    calls += 1;
    return new Promise((_, reject) => {
      fail = reject;
    });
  });
  await tick();
  poller.stop();
  fail(new api.Problem(0, null, "unreachable"));
  await tick();
  t.mock.timers.tick(60_000);
  await tick();
  assert.equal(calls, 1);
  assert.deepEqual(banner, [false]); // stop() hides the banner; the late failure does not show it
});

test("a kick while a step runs runs one more step after it", async (t) => {
  let finish;
  const { poller, calls } = setup(t, () => new Promise((resolve) => {
    finish = resolve;
  }));
  await tick();
  poller.kick();
  poller.kick();
  finish(false);
  await tick();
  assert.equal(calls.n, 2);
});
