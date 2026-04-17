// usePolling.ts — Hook for polling an async function at an interval
//
// Automatically starts/stops based on a condition.

import { useEffect, useRef } from 'react';

export function usePolling(
  fn: () => Promise<void>,
  interval: number,
  enabled: boolean,
) {
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    if (!enabled) return;

    let active = true;
    const poll = async () => {
      while (active) {
        await fnRef.current();
        await new Promise(resolve => setTimeout(resolve, interval));
      }
    };
    poll();
    return () => { active = false; };
  }, [interval, enabled]);
}
