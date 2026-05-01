// useEventSource.ts — React hook for SSE log streaming
//
// Connects to the server's SSE endpoint, buffers lines,
// and auto-reconnects on disconnect.
//
// Performance: batches incoming messages and flushes at ~100ms intervals
// to avoid per-message React re-renders during heavy logging.

import { useState, useEffect, useRef, useCallback } from 'react';

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
  const esRef = useRef<EventSource | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
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

  const connect = useCallback(() => {
    if (!enabled) return;
    if (esRef.current) {
      esRef.current.close();
    }

    const es = new EventSource(url);
    esRef.current = es;

    es.onopen = () => {
      setConnected(true);
    };

    es.onmessage = (event) => {
      try {
        const line: LogLine = JSON.parse(event.data);
        batchRef.current.push(line);
        // Schedule a flush if one isn't already pending
        if (!flushTimer.current) {
          flushTimer.current = setTimeout(flush, BATCH_INTERVAL_MS);
        }
      } catch {
        // Ignore malformed data
      }
    };

    es.onerror = () => {
      setConnected(false);
      es.close();
      esRef.current = null;
      // Auto-reconnect after 3 seconds
      reconnectTimer.current = setTimeout(connect, 3000);
    };
  }, [url, enabled, flush]);

  useEffect(() => {
    if (enabled) {
      connect();
    } else {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
      setConnected(false);
    }

    return () => {
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
      }
      if (flushTimer.current) {
        clearTimeout(flushTimer.current);
      }
    };
  }, [enabled, connect]);

  const clear = useCallback(() => {
    batchRef.current = [];
    setLines([]);
  }, []);

  return { lines, connected, clear };
}
