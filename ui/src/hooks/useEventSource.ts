// useEventSource.ts — React hook for SSE log streaming
//
// Subscribes to the server's SSE endpoint through the tab's shared connection
// (App's presence beacon reads the same stream), buffers lines, and reopens
// the stream after a disconnect.
//
// Performance: batches incoming messages and flushes at ~100ms intervals
// to avoid per-message React re-renders during heavy logging.

import { useState, useEffect, useRef, useCallback } from 'react';
import { subscribeEventSource } from '../services/sharedEventSource';

export interface LogLine {
  id: number;
  ts: number;
  text: string;
  source: 'engine' | 'server';
}

const MAX_CLIENT_LINES = 500;
const BATCH_INTERVAL_MS = 100;

export function useEventSource(url: string, enabled: boolean) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [connected, setConnected] = useState(false);
  const batchRef = useRef<LogLine[]>([]);
  const flushTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  // Flush batched lines into React state
  const flush = useCallback(() => {
    flushTimer.current = undefined;
    if (batchRef.current.length === 0) return;
    const batch = batchRef.current;
    batchRef.current = [];
    setLines(prev => {
      const next = [...prev, ...batch];
      return next.length > MAX_CLIENT_LINES
        ? next.slice(next.length - MAX_CLIENT_LINES)
        : next;
    });
  }, []);

  useEffect(() => {
    if (!enabled) {
      setConnected(false);
      return;
    }
    const unsubscribe = subscribeEventSource(url, {
      open: () => setConnected(true),
      message: (data) => {
        try {
          const line: LogLine = JSON.parse(data);
          batchRef.current.push(line);
          // Schedule a flush if one isn't already pending
          if (!flushTimer.current) {
            flushTimer.current = setTimeout(flush, BATCH_INTERVAL_MS);
          }
        } catch {
          // Ignore malformed data
        }
      },
      error: () => setConnected(false),
    }, { reconnect: true });

    return () => {
      unsubscribe();
      if (flushTimer.current) {
        clearTimeout(flushTimer.current);
        flushTimer.current = undefined;
      }
    };
  }, [url, enabled, flush]);

  const clear = useCallback(() => {
    batchRef.current = [];
    setLines([]);
  }, []);

  return { lines, connected, clear };
}
