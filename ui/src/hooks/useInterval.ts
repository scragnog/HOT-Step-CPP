import { useEffect, useRef } from 'react';

/**
 * Run `callback` every `ms` while `enabled` holds.
 *
 * The three call sites this replaced each rebuilt the same
 * `const id = setInterval(...); return () => clearInterval(id)` pair, and two
 * of them captured a stale callback because the effect did not depend on it.
 * The callback lives in a ref here, so a fresh closure is always the one that
 * runs and the interval is not torn down on every render.
 */
export function useInterval(callback: () => void, ms: number, enabled = true): void {
  const callbackRef = useRef(callback);
  useEffect(() => { callbackRef.current = callback; });
  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(() => callbackRef.current(), ms);
    return () => clearInterval(id);
  }, [ms, enabled]);
}
