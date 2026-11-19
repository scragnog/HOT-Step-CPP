import { useEffect, useRef } from 'react';

export interface GlobalEventOptions {
  /** Subscribe only while this is true. Handy for popovers: open/closed. */
  enabled?: boolean;
  /** Capture phase, so ancestor scrolling is followed too, not just window. */
  capture?: boolean;
  /** `document` for events that never reach window, e.g. visibilitychange. */
  target?: 'window' | 'document';
}

/**
 * Subscribe to a window or document event while a component is mounted.
 *
 * The handler lives in a ref, so inline closures and per-render callbacks do not
 * tear the listener down and re-add it on every paint — the hand-written pairs
 * this replaced did exactly that, re-measuring a popover on every keystroke.
 */
type GlobalEventMap = WindowEventMap & DocumentEventMap;

export function useGlobalEvent<K extends keyof GlobalEventMap>(
  type: K,
  handler: (event: GlobalEventMap[K]) => void,
  options: GlobalEventOptions = {},
): void {
  const { enabled = true, capture = false, target = 'window' } = options;
  const handlerRef = useRef(handler);
  useEffect(() => { handlerRef.current = handler; });
  useEffect(() => {
    if (!enabled) return;
    const el: Window | Document = target === 'document' ? document : window;
    const listener = (event: Event) => handlerRef.current(event as GlobalEventMap[K]);
    (el as EventTarget).addEventListener(type, listener, capture);
    return () => (el as EventTarget).removeEventListener(type, listener, capture);
  }, [type, enabled, capture, target]);
}
