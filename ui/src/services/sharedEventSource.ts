// sharedEventSource.ts — one EventSource per URL per tab
//
// Chrome allows six HTTP/1.1 connections per host, shared by every tab, and
// each open stream holds one for its whole life. Two components reading the
// same stream must not cost two of them, or a few open tabs starve ordinary
// requests and the app looks frozen.
//
// The servers replay their buffer on connect, so a subscriber joining a stream
// that is already open is handed the messages received since it (re)opened.

interface Subscriber {
  message?: (data: string) => void;
  error?: () => void;
  open?: () => void;
}

interface Entry {
  es: EventSource;
  subs: Set<Subscriber>;
  seen: string[];
  reconnect: boolean;
  retry?: number;
  terminal?: boolean;
  isTerminal?: (data: string) => boolean;
}

// ponytail: drops oldest-first, unlike the training buffer, which evicts logs
// before samples; ten times its size so a late joiner on a long run still
// sees the frames it needs. Evict by type here if that ever falls short.
const REPLAY_CAP = 20000;
const entries = new Map<string, Entry>();

function connect(url: string, entry: Entry) {
  const es = new EventSource(url);
  entry.es = es;
  es.onopen = () => {
    entry.seen = [];  // the server replays from the start on every connect
    entry.subs.forEach(s => s.open?.());
  };
  es.onmessage = e => {
    if (entry.terminal) return;
    entry.seen.push(e.data);
    if (entry.seen.length > REPLAY_CAP) entry.seen.shift();
    if (entry.isTerminal?.(e.data)) {
      entry.terminal = true;
      window.clearTimeout(entry.retry);
      es.close();
    }
    entry.subs.forEach(s => s.message?.(e.data));
  };
  es.onerror = () => {
    if (entry.terminal) return;
    entry.seen = [];  // whatever reconnects next replays from the start
    entry.subs.forEach(s => s.error?.());
    // EventSource retries drops and clean server closes alike. A terminal
    // frame must close it explicitly above to prevent a full replay.
    // CLOSED HTTP failures may also be retried for long-lived streams.
    if (es.readyState === EventSource.CLOSED && entry.reconnect && entries.get(url) === entry) {
      entry.retry = window.setTimeout(() => { entry.retry = undefined; connect(url, entry); }, 3000);
    }
  };
}

/** Subscribe to a server-sent event stream, sharing the tab's connection.
 *  `reconnect` reopens the stream after an HTTP error, not only after a drop. */
export function subscribeEventSource(url: string, sub: Subscriber, opts: { reconnect?: boolean; isTerminal?: (data: string) => boolean; replayOnly?: boolean } = {}): () => void {
  // Finished jobs need a single replay. Native EventSource hides HTTP status
  // and reports missing history as a console error; fetch lets us handle it.
  if (opts.replayOnly) {
    const controller = new AbortController();
    void (async () => {
      try {
        const response = await fetch(url, { signal: controller.signal });
        if (!response.ok || !response.body) throw new Error('History unavailable');
        if (controller.signal.aborted) return;
        sub.open?.();
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let pending = '';
        try {
          while (!controller.signal.aborted) {
            const { value, done } = await reader.read();
            pending += decoder.decode(value, { stream: !done });
            pending = pending.replace(/\r\n/g, '\n');
            let boundary: number;
            while ((boundary = pending.indexOf('\n\n')) >= 0) {
              const frame = pending.slice(0, boundary);
              pending = pending.slice(boundary + 2);
              const data = frame.split('\n').filter(line => line.startsWith('data:'))
                .map(line => line.slice(5).replace(/^ /, '')).join('\n');
              if (!data || controller.signal.aborted) continue;
              const terminal = opts.isTerminal?.(data);
              sub.message?.(data);
              if (terminal) return;
            }
            if (done) throw new Error('Incomplete history');
          }
        } finally {
          await reader.cancel().catch(() => {});
          reader.releaseLock();
        }
      } catch {
        if (!controller.signal.aborted) sub.error?.();
      }
    })();
    return () => controller.abort();
  }
  let entry = entries.get(url);
  if (!entry) {
    entry = { es: null as unknown as EventSource, subs: new Set(), seen: [], reconnect: !!opts.reconnect, isTerminal: opts.isTerminal };
    entries.set(url, entry);
    connect(url, entry);
  } else {
    entry.reconnect ||= !!opts.reconnect;
    entry.isTerminal ??= opts.isTerminal;
    if (!entry.terminal && entry.isTerminal && entry.seen.some(entry.isTerminal)) {
      entry.terminal = true;
      window.clearTimeout(entry.retry);
      entry.es.close();
    }
    // A stream that gave up (an HTTP error, no retry pending) reopens for a newcomer.
    if (!entry.terminal && entry.es.readyState === EventSource.CLOSED && entry.retry === undefined) connect(url, entry);
    // After returning, so a handler can already call its own unsubscribe.
    const joined = entry, backlog = [...entry.seen];
    queueMicrotask(() => {
      if (joined.es.readyState === EventSource.OPEN && joined.subs.has(sub)) sub.open?.();
      for (const data of backlog) { if (!joined.subs.has(sub)) break; sub.message?.(data); }
    });
  }
  entry.subs.add(sub);
  const mine = entry;
  return () => {
    if (!mine.subs.delete(sub) || mine.subs.size) return;
    window.clearTimeout(mine.retry);
    mine.es.close();
    if (entries.get(url) === mine) entries.delete(url);
  };
}
