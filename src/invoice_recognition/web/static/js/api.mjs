// The client of the API: every request carries the access code; an error answer becomes
// a Problem {status, code, detail}. Polling with back-off lives here too.

import { MAX_UPLOAD_BYTES, TOO_LARGE, backoff, fallbackDetail } from "./model.mjs";

// ------------------------------------------------------------------ the access code
// Kept for this tab only (session storage), never in local storage: /docs runs a
// third-party script on the same origin.

const CODE_KEY = "invoice-reader.access-code";
let codeInMemory = null; // when the browser refuses storage

function storedCode() {
  try {
    return sessionStorage.getItem(CODE_KEY);
  } catch {
    return codeInMemory;
  }
}

function storeCode(code) {
  codeInMemory = code;
  try {
    if (code == null) sessionStorage.removeItem(CODE_KEY);
    else sessionStorage.setItem(CODE_KEY, code);
  } catch {
    // storage refused (a private window): the code lives in memory until the tab closes
  }
}

/** @type {(wrong: boolean) => Promise<string>} asks a person for the code; set by the app */
let askForCode = () => Promise.reject(new Problem(401, "no_code", "The access code is needed to continue."));
let asking = null; // one dialog for every request refused at the same time

export function onCodeNeeded(ask) {
  askForCode = ask;
}

async function newCode(wrong) {
  asking ??= askForCode(wrong).finally(() => {
    asking = null;
  });
  storeCode(await asking);
}

// ------------------------------------------------------------------ requests

export class Problem extends Error {
  constructor(status, code, detail) {
    super(detail);
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

async function problemOf(response) {
  if (response.headers.get("content-type")?.startsWith("application/problem+json")) {
    try {
      const body = await response.json();
      if (typeof body.detail === "string") return new Problem(response.status, body.code ?? null, body.detail);
    } catch {
      // not the JSON it claims to be: the fallback below
    }
  }
  return new Problem(response.status, null, fallbackDetail(response.status));
}

async function send(path, init, code) {
  const headers = new Headers(init.headers);
  if (code) headers.set("Authorization", `Bearer ${code}`);
  try {
    return await fetch(path, { ...init, headers });
  } catch {
    throw new Problem(0, null, fallbackDetail(0));
  }
}

/** fetch() with the code. A 401 drops the code, asks for a new one and sends the request
 * again, until the code is right or the person cancels (Problem 401 "no_code"). */
async function request(path, init = {}) {
  for (;;) {
    const code = storedCode();
    const response = await send(path, init, code);
    if (response.status !== 401) {
      if (!response.ok) throw await problemOf(response);
      return response;
    }
    // Only the code this request was refused with: another may have been entered since.
    if (code && storedCode() === code) storeCode(null);
    await newCode(Boolean(code));
  }
}

const json = async (path, init) => (await request(path, init)).json();

export const list = () => json("/invoices");
export const get = (id) => json(`/invoices/${id}`);

/** The uploaded file, as a Blob: fetched with the code, never by URL. */
export const file = async (id) => (await request(`/invoices/${id}/file`)).blob();

/** Upload a file; the accepted job. Too large: refused here, nothing is sent. */
export async function upload(file) {
  if (file.size > MAX_UPLOAD_BYTES) throw new Problem(413, "file_too_large", TOO_LARGE);
  const form = new FormData();
  form.append("file", file);
  return json("/invoices", { method: "POST", body: form });
}

/** Upload one of the page's sample files, exactly as a person's own file. */
export async function uploadSample(sample) {
  let response;
  try {
    response = await fetch(`/static/samples/${sample.file}`);
  } catch {
    throw new Problem(0, null, fallbackDetail(0));
  }
  if (!response.ok) throw new Problem(response.status, null, fallbackDetail(response.status));
  const blob = await response.blob();
  return upload(new File([blob], sample.file, { type: sample.type }));
}

// ------------------------------------------------------------------ polling

/** @type {(lost: boolean) => void} shows or hides "Lost the connection"; set by the app */
let connectionShown = () => {};

export function onConnection(show) {
  connectionShown = show;
}

/**
 * Call `step` now and then again every second while it returns true. A failed step is
 * tried again after 2, 4, 8, 15, 15… s while the connection banner shows. Nothing runs
 * while the tab is hidden; becoming visible runs a step at once. A missing access code
 * stops the polling until `kick()`.
 * @param {() => Promise<boolean>} step
 */
export function poller(step) {
  let timer = null;
  let failures = 0;
  let running = false;
  let again = false; // asked to run while a step was running: run once more after it
  let stopped = false;

  async function run() {
    clearTimeout(timer);
    timer = null;
    if (stopped || document.hidden) return;
    if (running) {
      again = true;
      return;
    }
    running = true;
    again = false;
    let wait = null; // seconds until the next step; null: none
    let noCode = false;
    try {
      const more = await step();
      if (stopped) return; // its screen was left meanwhile: nothing more to show
      if (more) wait = 1;
      failures = 0;
      connectionShown(false);
    } catch (error) {
      if (stopped) return;
      noCode = error.code === "no_code";
      if (!noCode) {
        failures += 1;
        wait = backoff(failures);
        connectionShown(true);
      }
    } finally {
      running = false;
    }
    if (stopped || noCode) return;
    if (again) run();
    else if (wait != null) timer = setTimeout(run, wait * 1000);
  }

  const onVisible = () => {
    if (!document.hidden) run();
  };
  document.addEventListener("visibilitychange", onVisible);
  run();
  return {
    kick: run,
    stop() {
      stopped = true;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
      connectionShown(false);
    },
  };
}
