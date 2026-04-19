// useEventSource.ts — React hook for SSE log streaming
//
// Connects to the server's SSE endpoint, buffers lines,
// and auto-reconnects on disconnect.

import { useState, useEffect, useRef, useCallback } from 'react';

export interface LogLine {
  id: number;
  ts: number;
  text: string;
  source: 'engine' | 'server';
}

const MAX_CLIENT_LINES = 2000;

export function useEventSource(url: string, enabled: boolean) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [connected, setConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

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
        setLines(prev => {
          const next = [...prev, line];
          return next.length > MAX_CLIENT_LINES
            ? next.slice(next.length - MAX_CLIENT_LINES)
            : next;
        });
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
  }, [url, enabled]);

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
    };
  }, [enabled, connect]);

  const clear = useCallback(() => {
    setLines([]);
  }, []);

  return { lines, connected, clear };
}
