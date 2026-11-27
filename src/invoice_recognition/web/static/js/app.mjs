// The page: one screen at a time, chosen by the address (#/ or #/doc/<id>); the access
// code dialog; the connection banner; the live region that reads changes out.

import * as api from "./api.mjs";
import { APP_NAME } from "./format.mjs";
import { showHome } from "./home.mjs";
import { checkCode } from "./model.mjs";
import { showReview } from "./review.mjs";

const view = document.getElementById("view");
const live = document.getElementById("live");

/** Read a short change out to screen-reader users. */
function announce(text) {
  live.textContent = "";
  requestAnimationFrame(() => {
    live.textContent = text;
  });
}

// ------------------------------------------------------------------ the access code

const dialog = document.getElementById("access");
const form = document.getElementById("access-form");
const field = document.getElementById("access-code");
const fieldError = document.getElementById("access-error");
const cancel = document.getElementById("access-cancel");

function showFieldError(text) {
  fieldError.textContent = text ?? "";
  fieldError.hidden = !text;
}

/** Ask for the code; resolves with it, rejects (no_code) when the person cancels. */
function askForCode(wrong) {
  return new Promise((resolve, reject) => {
    field.value = "";
    showFieldError(wrong ? "That code is not right." : null);
    const finish = (code) => {
      form.removeEventListener("submit", onSubmit);
      dialog.removeEventListener("cancel", onCancel);
      cancel.removeEventListener("click", onCancel);
      dialog.close();
      if (code) resolve(code);
      else reject(new api.Problem(401, "no_code", "The access code is needed to continue."));
    };
    const onSubmit = (e) => {
      e.preventDefault(); // handled here: the page's CSP allows no form submission
      const code = field.value.trim();
      const error = checkCode(code);
      if (error) showFieldError(error);
      else finish(code);
    };
    const onCancel = (e) => {
      e.preventDefault(); // Esc: closed by finish(), like the Cancel button
      finish(null);
    };
    form.addEventListener("submit", onSubmit);
    dialog.addEventListener("cancel", onCancel);
    cancel.addEventListener("click", onCancel);
    dialog.showModal();
  });
}

api.onCodeNeeded(askForCode);

const banner = document.getElementById("connection");
api.onConnection((lost) => {
  banner.hidden = !lost;
});

// ------------------------------------------------------------------ the router

let teardown = null;
let focusUpload = false; // "Upload another file" on a failed document: focus the upload

function route() {
  teardown?.();
  document.title = APP_NAME;
  const doc = location.hash.match(/^#\/doc\/([0-9a-f]{32})$/);
  if (doc) {
    teardown = showReview(view, doc[1], {
      announce,
      onUploadAnother: () => {
        focusUpload = true;
      },
    });
  } else {
    teardown = showHome(view, { announce, focusUpload });
  }
  focusUpload = false;
  view.focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

window.addEventListener("hashchange", route);
route();
