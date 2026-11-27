// The review screen of one document: the document on the left, polled until it is read,
// then its header, the issues and the rows on the right, re-drawn from state after every
// change. Selecting a row shows it on the page; in "show all" the two panes scroll
// together, led by the one a person last touched.

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
import {
  decidedText,
  finishedText,
  firstOpenLine,
  framedFields,
  lineAt,
  placeIssues,
  rowBox,
  screenLife,
  stateOf,
  target,
  unfinished,
} from "./model.mjs";
import { createViewer } from "./viewer.mjs";

/**
 * Show the review screen of document `id` in `view`; returns its teardown.
 * @param {HTMLElement} view
 * @param {string} id
 * @param {{announce: (text: string) => void, onUploadAnother: () => void}} options
 */
export function showReview(view, id, { announce, onUploadAnother }) {
  const S = {
    job: null,
    version: null,
    built: false,
    sawReading: false,
    showAll: false,
    selected: null,
    autoselected: false,
    viewer: null,
    shown: false, // the viewer has shown the first selection
    pending: new Set(), // issues whose decision is on its way
    errors: new Map(), // issue id -> why the server refused a decision
    driver: "none", // the pane a person last touched: "rows" | "doc"
  };
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
    syncScrolling();
    openDocument();
  }

  async function openDocument() {
    try {
      const file = await life.guard(api.file(id));
      S.viewer = createViewer($("docpane"), $("pages"), file, S.job.kind);
      await life.guard(S.viewer.ready);
      showOnDocument();
    } catch (problem) {
      // A refusal of the file is shown; the viewer shows its own failure; a left screen nothing.
      if (problem?.status != null && problem.code !== "left") put($("pages"), html`<div class="empty">${problem.detail}</div>`);
    }
  }

  const sizes = () => S.viewer.sizes();
  const hasResult = () => S.job?.result && !unfinished(S.job.status) && S.job.status !== "failed";

  /** The marks of every issue on the document, then the selection. */
  function showOnDocument() {
    if (!S.viewer || !hasResult()) return;
    const { result, review } = S.job;
    const marks = [];
    for (const issue of S.viewer.isPhoto ? [] : result.issues) {
      const t = target({ ...issue, fields: framedFields(issue, review) }, result, sizes());
      const done = stateOf(review, issue.id) !== "open";
      for (const box of t?.frames ?? []) marks.push({ page: t.page, box, kind: "value", done, line: issue.line_index });
      if (t?.band) marks.push({ page: t.page, box: t.band, kind: "band", done, line: issue.line_index });
    }
    S.viewer.setMarks(marks);
    highlight(!S.shown);
    S.shown = true;
  }

  /** Show the selected row on the document; `scroll`: bring it into view. */
  function highlight(scroll) {
    const v = S.viewer;
    if (!v || !hasResult()) return;
    const { result, review } = S.job;
    v.setActive(S.selected);
    v.focusBand(null);
    if (scroll) v.outline(null); // a new selection; a re-draw keeps the page outlined
    const placed = placeIssues(result, review);
    const line = result.lines[S.selected];
    if (!line) {
      const card = placed.cards.find((i) => i.page != null);
      if (scroll && card) showPage(card.page);
      return;
    }
    const issues = placed.rows.get(S.selected) ?? [];
    const pointed = issues.map((i) => target(i, result, sizes())).find((t) => t?.frames.length || t?.band);
    const row = v.isPhoto ? null : rowBox(line, result, sizes());
    if (!issues.length && row) v.focusBand(line.page, row);
    if (!scroll || line.page == null) return;
    S.driver = "none";
    const box = pointed?.frames[0] ?? pointed?.band ?? row;
    if (box) v.scrollTo(line.page, box.top);
    else showPage(line.page);
  }

  function showPage(page) {
    S.driver = "none";
    S.viewer?.outline(page);
    S.viewer?.scrollTo(page, 0, { at: 0.04 });
  }

  /** In "show all", the pane a person scrolls leads and the other follows. */
  function syncScrolling() {
    const rows = $("datapane");
    const doc = $("docpane");
    for (const [el, who] of [[rows, "rows"], [doc, "doc"]]) {
      for (const type of ["wheel", "touchstart", "pointerdown", "keydown"]) {
        el.addEventListener(type, () => {
          S.driver = who;
        }, { passive: true });
      }
    }
    let pending = false;
    const follow = (who) => () => {
      if (!S.showAll || S.driver !== who || pending || !S.viewer || !hasResult()) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        if (who === "rows") rowsLead();
        else documentLeads();
      });
    };
    rows.addEventListener("scroll", follow("rows"), { passive: true });
    doc.addEventListener("scroll", follow("doc"), { passive: true });
  }

  function rowsLead() {
    const pane = $("datapane");
    const top = pane.getBoundingClientRect().top + 40;
    const row = [...pane.querySelectorAll(".row")].find((r) => r.getBoundingClientRect().bottom > top);
    const line = row && S.job.result.lines[Number(row.dataset.line)];
    if (line?.page == null) return;
    const box = S.viewer.isPhoto ? null : rowBox(line, S.job.result, sizes());
    S.viewer.scrollTo(line.page, box?.top ?? 0, { at: 0.2, instant: true });
  }

  function documentLeads() {
    const at = S.viewer.anchor(0.2);
    const line = at && lineAt(S.job.result, at.page, at.top, sizes());
    const row = line && $("r-body").querySelector(`.row[data-line="${line.index}"]`);
    if (row) $("datapane").scrollTop = row.offsetTop - 40;
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
    showOnDocument();
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
    highlight(true);
  }

  function onClick(e) {
    const clicked = e.target;
    if (clicked.closest("#toggle-all")) {
      S.showAll = !S.showAll;
      render();
      scrollToRow(S.selected);
    } else if (clicked.closest("[data-action=enter-code]")) {
      polling.kick();
    } else if (clicked.closest("[data-action=upload-another]")) {
      onUploadAnother();
    } else if (clicked.closest("[data-body]")) {
      const button = clicked.closest("[data-body]");
      decide(Number(button.dataset.issue), JSON.parse(button.dataset.body));
    } else if (clicked.closest("[data-undo]")) {
      decide(Number(clicked.closest("[data-undo]").dataset.undo), null);
    } else if (clicked.closest(".row-main")) {
      select(Number(clicked.closest(".row").dataset.line));
    } else if (clicked.closest(".docissue-title")) {
      const page = clicked.closest(".docissue-title").dataset.page;
      select(null);
      if (page) showPage(Number(page));
    }
  }

  /** Send a decision (`body` null: take it back); the answer is the new review. */
  async function decide(issueId, body) {
    if (S.pending.has(issueId)) return;
    const issue = S.job.result.issues[issueId];
    S.pending.add(issueId);
    S.errors.delete(issueId);
    if (issue.line_index != null) S.selected = issue.line_index;
    render();
    let review;
    try {
      review = await life.guard(body ? api.decide(id, issueId, body) : api.undo(id, issueId));
    } catch (problem) {
      if (problem.code === "left") return;
      S.pending.delete(issueId);
      if (problem.status === 404 && problem.code === "not_found") {
        put(view, notFound(problem.detail));
        polling.stop();
        return;
      }
      S.errors.set(issueId, problem.detail);
      render();
      return;
    }
    S.pending.delete(issueId);
    S.job = { ...S.job, review };
    S.version = JSON.stringify([S.job.status, S.job.progress, review]);
    render();
    const entry = review.issues[issueId];
    announce(entry.state === "open" ? "Decision taken back." : `${decidedText(issue, entry)}.`);
    const focus = entry.state === "open" ? `choice-${issueId}-0` : `undo-${issueId}`;
    reveal($("r-body").querySelector(`[data-focus="${focus}"]`));
  }

  /** Focus `element` and scroll the rows pane (only it) as far as needed to show it. */
  function reveal(element) {
    if (!element) return;
    element.focus({ preventScroll: true });
    const pane = $("datapane");
    const box = element.getBoundingClientRect();
    const frame = pane.getBoundingClientRect();
    if (box.top < frame.top + 40) pane.scrollTop -= frame.top + 40 - box.top; // under the sticky head
    else if (box.bottom > frame.bottom) pane.scrollTop += box.bottom - frame.bottom + 8;
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
    S.viewer?.destroy();
  };
}
