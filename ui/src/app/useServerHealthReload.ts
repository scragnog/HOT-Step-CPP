// useServerHealthReload.ts — wait for the server to answer, then reload.
// The only component-owned interval left in the UI (a liveness probe that
// must not pollute the query cache).

import { useEffect } from 'react';

/** Poll after either kind of disconnect, then reload once the server answers. */
export function useServerHealthReload(startDelayMs: number, onWaiting?: () => void, onReconnected?: () => void): void {
  useEffect(() => {
    let cancelled = false;
    let checking = false;
    let poll: number | undefined;
    let reload: number | undefined;
    const start = window.setTimeout(() => {
      onWaiting?.();
      poll = window.setInterval(async () => {
        if (checking) return;
        checking = true;
        try {
          const res = await fetch('/api/health', { signal: AbortSignal.timeout(2000) });
          if (res.ok && !cancelled) {
            window.clearInterval(poll);
            onReconnected?.();
            reload = window.setTimeout(() => window.location.reload(), 500);
          }
        } catch { /* The server is still offline. */ }
        finally { checking = false; }
      }, 2000);
    }, startDelayMs);
    return () => {
      cancelled = true;
      window.clearTimeout(start);
      window.clearInterval(poll);
      window.clearTimeout(reload);
    };
  }, [startDelayMs, onWaiting, onReconnected]);
}

