import { useEffect, useRef } from 'react';

/**
 * Run `onEscape` for an Escape keypress while `enabled`. The listener lives on
 * document, so it closes a portalled panel the same as an inline one, and the
 * callback is read through a ref so a re-render never re-subscribes mid-press.
 *
 * `capture` is for a surface layered over another one that also handles
 * Escape: it runs before the inner handlers and lets the caller stop the event.
 */
export function useEscapeKey(
  onEscape: (e: KeyboardEvent) => void,
  enabled = true,
  capture = false,
): void {
  const handler = useRef(onEscape);
  handler.current = onEscape;

  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handler.current(e);
    };
    document.addEventListener('keydown', onKey, capture);
    return () => document.removeEventListener('keydown', onKey, capture);
  }, [enabled, capture]);
}
