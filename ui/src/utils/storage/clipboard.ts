/**
 * clipboard.ts — the one place the app writes to the system clipboard.
 *
 * Six call sites had their own `navigator.clipboard?.writeText(...).then(...)`
 * chain, each deciding differently what a blocked clipboard means (insecure
 * origin, denied permission, a browser without the API). Here it is one answer:
 * resolve true when the text was handed over, false when it was not, never a
 * rejection to leak into a React event handler.
 */

/** Copy `text`. Resolves true only if the browser accepted it. */
export async function copyText(text: string): Promise<boolean> {
  const clip = typeof navigator === 'undefined' ? undefined : navigator.clipboard;
  if (!clip?.writeText) return false;
  try {
    await clip.writeText(text);
    return true;
  } catch {
    return false;
  }
}

/**
 * Copy `text` and run `onCopied` only when the browser accepted it. Five
 * surfaces wrote this same `.then(ok => { if (ok) markCopied(); })` chain.
 */
export function copyTextThen(text: string, onCopied: () => void): void {
  void copyText(text).then((ok) => { if (ok) onCopied(); });
}
