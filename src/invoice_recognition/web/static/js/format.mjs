// Numbers, sizes, times and labels as a person reads them. Pure: no DOM, no requests.

const COUNT = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });

/** The page's name, in the header and in every tab title. */
export const APP_NAME = "Invoice reader";

export const STATUS_LABEL = {
  queued: "Waiting",
  processing: "Reading",
  ok: "Ready",
  needs_review: "Needs review",
  failed: "Could not read",
};

/** "1 page", "4 pages"; counts above 999 with a separator: "1,250 rows". */
export const plural = (n, one, many = `${one}s`) => `${COUNT.format(n)} ${n === 1 ? one : many}`;

/** A file size: "15 KB", "1.4 MB". */
export const size = (bytes) =>
  bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1048576).toFixed(1)} MB`;

/** The time of day of an ISO timestamp, "14:05"; "" for none. */
export const clock = (iso) =>
  iso ? new Date(iso).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }) : "";
