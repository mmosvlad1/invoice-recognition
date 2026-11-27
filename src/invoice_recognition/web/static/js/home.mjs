// The home screen: upload a file or a sample, and the list of documents, polled while
// one of them is still being read.

import * as api from "./api.mjs";
import { plural } from "./format.mjs";
import { docHref, docList, needsCode, put } from "./html.mjs";
import { screenLife, unfinished, uploadMessage } from "./model.mjs";

const SAMPLES = {
  small: { file: "invoice-small-misprint.pdf", type: "application/pdf" },
  large: { file: "invoice-large.pdf", type: "application/pdf" },
  scan: { file: "invoice-scan-simulated.pdf", type: "application/pdf" },
  photo: { file: "invoice-photo-simulated.jpg", type: "image/jpeg" },
};

/**
 * Show the home screen in `view`; returns its teardown.
 * @param {HTMLElement} view
 * @param {{announce: (text: string) => void}} options
 */
export function showHome(view, { announce }) {
  view.replaceChildren(document.getElementById("tpl-home").content.cloneNode(true));
  const $ = (id) => document.getElementById(id);
  const dropzone = $("dropzone");
  const input = $("file-input");
  const errorBox = $("upload-error");
  const listBody = $("doclist-body");
  let justAdded = null;
  const life = screenLife();

  const showError = (text) => {
    errorBox.textContent = text;
    errorBox.hidden = !text;
  };

  async function send(uploading) {
    showError("");
    try {
      const job = await life.guard(uploading);
      justAdded = job.id;
      announce(`Upload accepted: ${job.filename}. Reading starts shortly.`);
      polling.kick();
    } catch (problem) {
      if (problem.code === "left") return;
      showError(problem.code === "no_code" ? problem.detail : uploadMessage(problem));
    }
  }

  async function refresh() {
    let items;
    try {
      ({ items } = await life.guard(api.list()));
    } catch (problem) {
      if (problem.code === "no_code") put(listBody, needsCode());
      throw problem;
    }
    $("doclist-count").textContent = items.length ? plural(items.length, "document") : "";
    put(listBody, docList(items));
    if (justAdded) {
      listBody.querySelector(`tr[data-id="${CSS.escape(justAdded)}"] a`)?.focus({ preventScroll: true });
      justAdded = null;
    }
    return items.some((item) => unfinished(item.status));
  }

  input.addEventListener("change", () => {
    if (input.files[0]) send(api.upload(input.files[0]));
    input.value = "";
  });
  for (const type of ["dragenter", "dragover"]) {
    dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-over");
    });
  }
  dropzone.addEventListener("dragleave", (e) => {
    if (!dropzone.contains(e.relatedTarget)) dropzone.classList.remove("is-over");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-over");
    const file = e.dataTransfer.files[0];
    if (file) send(api.upload(file));
  });
  view.querySelector(".sample-buttons").addEventListener("click", (e) => {
    const button = e.target.closest("[data-sample]");
    if (button) send(api.uploadSample(SAMPLES[button.dataset.sample]));
  });
  listBody.addEventListener("click", (e) => {
    if (e.target.closest("[data-action=enter-code]")) polling.kick();
    else if (!e.target.closest("a")) {
      const row = e.target.closest("tr[data-id]");
      if (row) location.hash = docHref(row.dataset.id);
    }
  });

  const polling = api.poller(refresh);
  return () => {
    life.leave();
    polling.stop();
  };
}
