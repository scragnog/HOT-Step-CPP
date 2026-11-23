/**
 * Saving a generated file from the browser: one owner for the anchor +
 * object-URL dance, so each caller only says what to save and what to call it.
 *
 * Every caller here used to build the same five lines by hand, and they had
 * drifted: some revoke the URL at once (racing the click in Safari), some never
 * append the anchor (ignored by Firefox), some leak the URL entirely.
 */

/** How long a saved file's object URL stays alive. Revoking in the same tick as
 *  `click()` loses the download in Safari, where the navigation is async. */
const REVOKE_AFTER_MS = 60_000;

/** Hand `blob` to the browser's downloader as `filename`. */
export function saveBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  // Firefox only follows a click on an anchor that is in the document. Guarded
  // because tests (and a headless parse) may run this with a minimal document.
  document.body?.appendChild(anchor);
  anchor.click();
  anchor.remove?.();
  setTimeout(() => URL.revokeObjectURL(url), REVOKE_AFTER_MS);
}

/** Save `text` as a file of `mime` type. */
export function saveText(text: string, filename: string, mime = 'text/plain;charset=utf-8'): void {
  saveBlob(new Blob([text], { type: mime }), filename);
}

/** Save `value` as pretty-printed JSON — the export format the app's imports read. */
export function saveJson(value: unknown, filename: string): void {
  saveBlob(new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' }), filename);
}

/** Hand an existing URL to the browser's downloader. An empty filename lets the
 *  response's Content-Disposition name the file. */
export function saveUrl(url: string, filename: string): void {
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body?.appendChild(anchor);
  anchor.click();
  anchor.remove?.();
}

/** A filename-safe stem: anything that is not a letter, number, underscore,
 *  space, dot or dash becomes one underscore. Keeps accented titles readable.
 *  Falls back when the name held nothing usable. */
export function safeFileStem(name: string | undefined | null, fallback: string): string {
  const safe = (name ?? '').replace(/[^\p{L}\p{N}_ .-]+/gu, '_').trim();
  return safe || fallback;
}
