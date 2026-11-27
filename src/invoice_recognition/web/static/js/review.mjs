// The review screen of one document: polled until it is read, then its header, the
// issues and the rows; re-drawn from state after every change.

import * as api from "./api.mjs";
import { APP_NAME, clock, plural, size } from "./format.mjs";
import {
  failedPanel,
  headerFields,
  html,
  needsCode,
  notFound,
  progressPanel,
  put,
  resultPane,
  reviewStatus,
  withBack,
} from "./html.mjs";
import { finishedText, firstOpenLine, placeIssues, screenLife, unfinished } from "./model.mjs";

/**
 * Show the review screen of document `id` in `view`; returns its teardown.
 * @param {HTMLElement} view
 * @param {string} id
 * @param {{announce: (text: string) => void, onUploadAnother: () => void}} options
 */
export function showReview(view, id, { announce, onUploadAnother }) {
  const S = { job: null, version: null, built: false, sawReading: false, showAll: false, selected: null, autoselected: false };
  const $ = (elementId) => document.getElementById(elementId);
  const life = screenLife(); // every await goes through life.guard()
  put(view, withBack(html`<p class="empty">Opening…</p>`));

  async function step() {
    let job;
    try {
      job = await life.guard(api.get(id));
    } catch (problem) {
      if (problem.status === 404) {
        put(view, notFound(problem.detail));
        return false;
      }
      if (problem.code === "no_code") askAgain();
      throw problem;
    }
    update(job);
    return unfinished(job.status);
  }

  /** "The access code is needed" with its button, in the pane if there is one. */
  function askAgain() {
    if (S.built) return put($("r-body"), needsCode()); // its clicks: onClick
    put(view, withBack(needsCode()));
    view.querySelector("[data-action=enter-code]").addEventListener("click", () => polling.kick());
  }

  function update(job) {
    const version = JSON.stringify([job.status, job.progress, job.review]);
    if (version === S.version) return;
    const first = S.version === null;
    S.version = version;
    S.job = job;
    if (unfinished(job.status)) S.sawReading = true;
    else if (S.sawReading && !first) {
      S.sawReading = false;
      announce(finishedText(job));
    }
    build();
    render();
  }

  function build() {
    if (S.built) return;
    S.built = true;
    view.replaceChildren($("tpl-review").content.cloneNode(true));
    document.title = `${S.job.filename || "(no name)"} · ${APP_NAME}`;
    $("r-filename").textContent = S.job.filename || "(no name)";
    const body = $("r-body");
    body.addEventListener("click", onClick);
    body.addEventListener("keydown", onKey);
  }

  function render() {
    const job = S.job;
    put($("r-status"), reviewStatus(job));
    $("r-meta").textContent = `${job.kind === "pdf" ? "PDF" : "Image"} · ${plural(job.progress.pages_total, "page")} · ${size(job.size_bytes)} · uploaded ${clock(job.created_at)}`;
    const fields = $("r-fields");
    fields.hidden = !job.result?.header || job.status === "failed";
    if (!fields.hidden) put(fields, headerFields(job.result, job.review));
    if (unfinished(job.status)) return redraw(progressPanel(job));
    if (job.status === "failed") return redraw(failedPanel(job));
    if (!S.autoselected) {
      S.autoselected = true;
      S.selected = firstOpenLine(job.result, job.review, placeIssues(job.result, job.review));
    }
    redraw(resultPane(job, S));
  }

  /** Replace the right pane, keeping its scroll position and the focused control. */
  function redraw(markup) {
    const pane = $("datapane");
    const top = pane.scrollTop;
    const focused = document.activeElement?.closest?.("#r-body [data-focus]")?.dataset.focus;
    put($("r-body"), markup);
    pane.scrollTop = top;
    if (focused) $("r-body").querySelector(`[data-focus="${CSS.escape(focused)}"]`)?.focus({ preventScroll: true });
  }

  /** Scroll the rows pane (only it) so that the row of line `index` is in the middle. */
  function scrollToRow(index) {
    const row = $("r-body").querySelector(`.row[data-line="${index}"]`);
    const pane = $("datapane");
    if (row) pane.scrollTop = row.offsetTop - (pane.clientHeight - row.offsetHeight) / 2;
  }

  function select(index) {
    S.selected = index;
    for (const row of $("r-body").querySelectorAll(".row.is-selected")) row.classList.remove("is-selected");
    $("r-body").querySelector(`.row[data-line="${index}"]`)?.classList.add("is-selected");
  }

  function onClick(e) {
    const target = e.target;
    if (target.closest("#toggle-all")) {
      S.showAll = !S.showAll;
      render();
      scrollToRow(S.selected);
    } else if (target.closest("[data-action=enter-code]")) {
      polling.kick();
    } else if (target.closest("[data-action=upload-another]")) {
      onUploadAnother();
    } else if (target.closest(".row-main")) {
      select(Number(target.closest(".row").dataset.line));
    }
  }

  function onKey(e) {
    const main = e.target.closest?.(".row-main");
    if (!main || e.target !== main) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      select(Number(main.parentElement.dataset.line));
    } else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const all = [...$("r-body").querySelectorAll(".row-main")];
      all[all.indexOf(main) + (e.key === "ArrowDown" ? 1 : -1)]?.focus();
    }
  }

  const polling = api.poller(step);
  return () => {
    life.leave();
    polling.stop();
  };
}
