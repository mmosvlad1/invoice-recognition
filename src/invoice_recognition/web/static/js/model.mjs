// What the page decides from the API's answers: texts, counts, rules. Pure: no DOM, no
// requests, so every rule here is tested without a browser.

import { plural } from "./format.mjs";

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
export const TOO_LARGE = "The file is larger than the limit of 10 MB.";
export const UNREACHABLE = "The server could not be reached. Check the connection and try again.";
export const MIN_CODE = 16;

/** Why a typed access code cannot be right, or null when it may be: the server's rule. */
export function checkCode(text) {
  if (!/^[!-~]*$/.test(text)) {
    return "The access code has only letters, digits and punctuation, without spaces.";
  }
  if (text.length < MIN_CODE) return `The access code has at least ${MIN_CODE} characters.`;
  return null;
}

/** What an awaited request rejects with once its screen was left. */
export const LEFT = Object.freeze({ status: 0, code: "left", detail: "The screen was left." });

/**
 * The life of a screen: `guard(promise)` settles as `promise` does while the screen is
 * shown, and rejects with LEFT once `leave()` was called, so that nothing after an await
 * renders into a screen that is gone. Every await of a screen goes through it.
 */
export function screenLife() {
  let left = false;
  return {
    leave() {
      left = true;
    },
    async guard(promise) {
      let result;
      try {
        result = await promise;
      } catch (error) {
        throw left ? LEFT : error;
      }
      if (left) throw LEFT;
      return result;
    },
  };
}

/** Seconds to wait before polling again after `failures` failed polls in a row (1, 2, …). */
export const backoff = (failures) => Math.min(2 ** failures, 15);

/** The share of pages read, from 0 to 1. */
export const progressShare = (p) => (p.pages_total > 0 ? Math.min(1, p.pages_done / p.pages_total) : 0);

export const unfinished = (status) => status === "queued" || status === "processing";

/** A finished job's result column: {text: "110 rows · 5 issues", open: "5 open" | null}, or
 * null while it has none. `open_issue_count` is the review's progress (null without one). */
export function listResult(item) {
  if (item.line_count == null) return null;
  const rows = plural(item.line_count, "row");
  if (!item.issue_count) return { text: `${rows} · no issues`, open: null };
  const text = `${rows} · ${plural(item.issue_count, "issue")}`;
  const open = item.open_issue_count;
  if (open == null) return { text, open: null };
  return { text, open: open === 0 ? "all checked" : `${open} open` };
}

/** The sentence an upload's refusal shows. `problem`: {status, code, detail}. */
export function uploadMessage(problem) {
  if (problem.code === "invalid_request") return "The upload could not be read. Try again.";
  return problem.detail;
}

/** The sentence for an error answer that is not a problem document. */
export function fallbackDetail(status) {
  if (status === 0) return UNREACHABLE;
  if (status === 413) return TOO_LARGE;
  return `The server answered with an error (${status}). Try again.`;
}
