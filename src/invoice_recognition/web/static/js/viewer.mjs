// The document beside its rows: a PDF drawn by PDF.js, or the photo. Every page gets
// its box at once (the page's shape); canvases are drawn only for pages near the view,
// one at a time, and at most MAX_CANVASES are kept. Marks are placed in percent of the
// page box, so they stay on their values at any width.

import { html, put } from "./html.mjs";

const PDFJS = "/static/vendor/pdfjs-4.10.38/";
const MAX_CANVASES = 10;

let pdfjsLoaded = null;
function pdfjs() {
  pdfjsLoaded ??= import(`${PDFJS}pdf.min.mjs`).then((lib) => {
    lib.GlobalWorkerOptions.workerSrc = `${PDFJS}pdf.worker.min.mjs`;
    return lib;
  });
  return pdfjsLoaded;
}

/** Place `element` over `box` ({left, top, width, height} in percent). */
function place(element, box) {
  for (const key of ["left", "top", "width", "height"]) element.style[key] = `${box[key]}%`;
}

/**
 * Show `file` (a Blob: the uploaded PDF or photo) in `container`, inside the scrolling
 * `pane`. `ready` resolves when every page has its box.
 * @param {HTMLElement} pane
 * @param {HTMLElement} container
 * @param {Blob} file
 * @param {"pdf" | "image"} kind
 */
export function createViewer(pane, container, file, kind) {
  /** @type {{el: HTMLElement, size: number[], canvas: HTMLCanvasElement | null, width: number, task: any}[]} */
  const pages = [];
  const near = new Set(); // page numbers within a viewport of the view
  let pdf = null;
  let loading = null; // PDF.js's loading task, until the document is open
  let photoUrl = null;
  let rendering = false;
  let destroyed = false;
  let observer = null;

  function addPage(n, width, height) {
    const el = document.createElement("div");
    el.className = "page";
    el.dataset.page = String(n);
    el.style.aspectRatio = `${width} / ${height}`;
    el.setAttribute("role", "img");
    el.setAttribute("aria-label", `Page ${n}`);
    put(el, html`<span class="page-loading" aria-hidden="true">Loading…</span><span class="page-label" aria-hidden="true">Page ${n}</span>`);
    container.append(el);
    pages.push({ el, size: [width, height], canvas: null, width: 0, task: null });
  }

  const center = () => {
    const middle = pane.scrollTop + pane.clientHeight / 2;
    const index = pages.findIndex((p) => p.el.offsetTop + p.el.offsetHeight >= middle);
    return index < 0 ? pages.length : index + 1;
  };

  /** The next page to draw: near the view, without a canvas of the current width. */
  function next() {
    const width = container.clientWidth;
    const stale = (p) => !p.canvas || Math.abs(p.width - width) / width > 0.25;
    const c = center();
    return [...near].filter((n) => stale(pages[n - 1])).sort((a, b) => Math.abs(a - c) - Math.abs(b - c))[0];
  }

  async function draw(n) {
    const p = pages[n - 1];
    const width = container.clientWidth;
    const page = await pdf.getPage(n);
    if (destroyed) return;
    const scale = Math.min(3, (width * Math.min(window.devicePixelRatio || 1, 2)) / p.size[0]);
    const viewport = page.getViewport({ scale });
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(viewport.width);
    canvas.height = Math.round(viewport.height);
    p.task = page.render({ canvasContext: canvas.getContext("2d"), viewport });
    try {
      await p.task.promise;
    } finally {
      p.task = null;
    }
    p.canvas?.remove();
    p.el.prepend(canvas);
    Object.assign(p, { canvas, width });
    release();
  }

  /** Keep at most MAX_CANVASES canvases: drop those of the pages farthest from the view
   * (never one near it: it would only be drawn again). */
  function release() {
    const c = center();
    const drawn = pages.filter((p) => p.canvas);
    const away = drawn
      .filter((p) => !near.has(Number(p.el.dataset.page)))
      .sort((a, b) => Math.abs(b.el.dataset.page - c) - Math.abs(a.el.dataset.page - c));
    for (const p of away.slice(0, Math.max(0, drawn.length - MAX_CANVASES))) {
      p.canvas.remove();
      p.canvas = null;
    }
  }

  async function pump() {
    if (rendering || destroyed || !pdf) return;
    const n = next();
    if (n == null) return;
    rendering = true;
    try {
      await draw(n);
    } catch (error) {
      if (error?.name !== "RenderingCancelledException") console.error(`Page ${n} could not be drawn`, error);
    } finally {
      rendering = false;
    }
    pump();
  }

  function watch() {
    observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const n = Number(entry.target.dataset.page);
          if (entry.isIntersecting) near.add(n);
          else {
            near.delete(n);
            pages[n - 1].task?.cancel(); // scrolled away while it was drawn
          }
        }
        pump();
      },
      { root: pane, rootMargin: "100% 0px" },
    );
    for (const p of pages) observer.observe(p.el);
  }

  // A change of width keeps the same spot of the same page in view; a large one draws
  // the pages near the view again, sharp at the new size.
  let spot = null;
  let restoring = false;
  pane.addEventListener("scroll", () => {
    if (!restoring) spot = anchorAt(0);
  }, { passive: true });
  let resizeTimer = null;
  const resized = new ResizeObserver(() => {
    const el = spot && pages[spot.page - 1]?.el;
    if (el) {
      restoring = true;
      pane.scrollTop = el.offsetTop + (spot.top / 100) * el.offsetHeight;
      requestAnimationFrame(() => {
        restoring = false;
      });
    }
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(pump, 250);
  });

  /** What is `at` of the way down the pane: {page, top (percent of that page)}, or null. */
  function anchorAt(at) {
    const y = pane.scrollTop + pane.clientHeight * at;
    const p = pages.find((q) => y < q.el.offsetTop + q.el.offsetHeight + 8);
    if (!p) return null;
    return { page: Number(p.el.dataset.page), top: (Math.max(0, y - p.el.offsetTop) / p.el.offsetHeight) * 100 };
  }

  const ready = (async () => {
    if (kind === "image") {
      photoUrl = URL.createObjectURL(file);
      const img = new Image();
      img.src = photoUrl;
      img.alt = "";
      await img.decode();
      if (destroyed) return;
      addPage(1, img.naturalWidth, img.naturalHeight);
      pages[0].el.querySelector(".page-loading").replaceWith(img);
      return;
    }
    const lib = await pdfjs();
    const data = await file.arrayBuffer();
    if (destroyed) return;
    loading = lib.getDocument({ data, isEvalSupported: false });
    pdf = await loading.promise;
    for (let n = 1; n <= pdf.numPages; n++) {
      const viewport = (await pdf.getPage(n)).getViewport({ scale: 1 });
      if (destroyed) return;
      addPage(n, viewport.width, viewport.height);
    }
    watch();
  })();
  resized.observe(container);
  ready.catch((error) => {
    if (destroyed) return; // left while loading: nothing to show
    console.error("The document could not be shown", error);
    put(container, html`<div class="empty">The document could not be shown (${error?.message ?? String(error)}).</div>`);
  });

  const pageEl = (n) => pages[n - 1]?.el;

  return {
    ready,
    /** [width, height] of each page as shown: points of a PDF page, pixels of a photo. */
    sizes: () => pages.map((p) => p.size),
    isPhoto: kind === "image",

    /** Draw marks: [{page, box, kind: "value" | "band", done, line}]; the old ones go. */
    setMarks(marks) {
      for (const old of container.querySelectorAll(".mark:not(.focus-band)")) old.remove();
      for (const m of marks) {
        const el = document.createElement("div");
        el.className = `mark ${m.kind}${m.done ? " resolved" : ""}`;
        el.dataset.line = String(m.line);
        el.setAttribute("aria-hidden", "true");
        place(el, m.box);
        pageEl(m.page)?.append(el);
      }
    },
    /** Emphasise the marks of one line (null: none). */
    setActive(line) {
      for (const el of container.querySelectorAll(".mark")) el.classList.toggle("is-active", el.dataset.line === String(line));
    },
    /** The blue band over a row without issues; null removes it. */
    focusBand(page, box) {
      for (const el of container.querySelectorAll(".mark.focus-band")) el.remove();
      if (!box) return;
      const el = document.createElement("div");
      el.className = "mark focus-band";
      el.setAttribute("aria-hidden", "true");
      place(el, box);
      pageEl(page)?.append(el);
    },
    /** Outline a page (null: none). */
    outline(page) {
      for (const p of pages) p.el.classList.toggle("is-target", p.el === pageEl(page));
    },
    /** Scroll so that `top` (percent of the page's height) is `at` of the way down the pane. */
    scrollTo(page, top = 0, { at = 0.3, instant = false } = {}) {
      const el = pageEl(page);
      if (!el) return;
      const y = el.offsetTop + (top / 100) * el.offsetHeight;
      pane.scrollTo({ top: Math.max(0, y - pane.clientHeight * at), behavior: instant ? "auto" : "smooth" });
    },
    anchor: anchorAt,
    destroy() {
      destroyed = true;
      observer?.disconnect();
      resized.disconnect();
      clearTimeout(resizeTimer);
      for (const p of pages) p.task?.cancel();
      loading?.destroy(); // ends the worker, also while the document is still loading
      if (photoUrl) URL.revokeObjectURL(photoUrl);
    },
  };
}
