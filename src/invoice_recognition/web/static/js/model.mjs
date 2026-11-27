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

// ------------------------------------------------------------------ the review screen

/** The review's state of an issue: open | accepted | kept | settled ("open" without one). */
export const stateOf = (review, id) => review?.issues?.[id]?.state ?? "open";

/** How many issues are still open: the review's count, or every issue before a review. */
export const openCount = (result, review) => review?.open_count ?? result.issues.length;

/**
 * Whether issue `issue` is shown under the issues it follows from: while the server says
 * it is settled, or while it is open and could still be settled — every parent is open or
 * accepted with an arithmetic item that sets the line's amount. Reads the parents' state
 * only; the server alone decides that an issue is settled.
 */
export function nestsUnderParents(issue, result, review) {
  const parents = issue.follows_from ?? [];
  if (!parents.length || !parents.every((p) => result.issues[p] && !result.issues[p].follows_from?.length)) {
    return false;
  }
  const own = stateOf(review, issue.id);
  if (own === "settled") return true;
  if (own !== "open") return false;
  return parents.every((p) => {
    const state = stateOf(review, p);
    if (state === "open") return true;
    const decided = review.issues[p];
    const item = result.issues[p].suggestions?.[decided.suggestion];
    return state === "accepted" && item?.source === "arithmetic" && Object.hasOwn(decided.set ?? {}, "line.amount");
  });
}

const push = (map, key, value) => (map.get(key) ?? map.set(key, []).get(key)).push(value);

/**
 * Where each issue is shown: under its line's row (`rows`: line index -> issues), as a
 * card for its page or the document (`cards`), or under the issues it follows from
 * (`children`: parent id -> issues).
 */
export function placeIssues(result, review) {
  const rows = new Map();
  const cards = [];
  const children = new Map();
  for (const issue of result.issues) {
    if (nestsUnderParents(issue, result, review)) {
      for (const parent of issue.follows_from) push(children, parent, issue);
    } else if (issue.line_index != null && result.lines[issue.line_index]) {
      push(rows, issue.line_index, issue);
    } else {
      cards.push(issue);
    }
  }
  return { rows, cards, children };
}

/** The title of the summary above the rows. */
export function summaryTitle(result, review) {
  const total = result.issues.length;
  if (!total) return "Everything adds up";
  const open = openCount(result, review);
  return open ? `${plural(open, "issue")} open` : `All ${plural(total, "issue")} checked`;
}

/** What the summary says of the rows without an issue; "" when every row has one. */
export function passedNote(total, hidden, showAll) {
  const tail = (verb) => (showAll ? "." : ` and ${verb} hidden.`);
  if (hidden === 0) return "";
  if (hidden === total) return `Every row passed every check${tail("is")}`;
  if (hidden === 1) return `The other row passed every check${tail("is")}`;
  return `The other ${plural(hidden, "row")} passed every check${tail("are")}`;
}

/** The row selected when a result opens: the first open issue's line, or null. */
export function firstOpenLine(result, review, placed) {
  const lines = [...placed.rows.keys()].sort((a, b) => a - b);
  const open = lines.find((index) => placed.rows.get(index).some((i) => stateOf(review, i.id) === "open"));
  return open ?? lines[0] ?? null;
}

/** What the live region says when a reading ends. */
export function finishedText(job) {
  if (job.status === "failed") return `Reading failed: ${job.error?.message ?? "the document could not be read."}`;
  const n = job.result?.issues.length ?? 0;
  return `Reading finished: ${n ? plural(n, "issue") : "everything adds up"}.`;
}
