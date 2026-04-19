// logs.ts — Live log streaming and VRAM proxy
//
// Ring buffer captures logs from both the Node server and ace-server child.
// SSE endpoint streams them to the UI in real time.
// VRAM endpoint proxies to ace-server's GET /vram.

import { Router, Request, Response } from 'express';
import { config } from '../config.js';

const router = Router();

// ── Ring buffer for log lines ─────────────────────────────────────────

export interface LogLine {
  id: number;
  ts: number;     // epoch ms
  text: string;
  source: 'engine' | 'server';
}

const MAX_LINES = 2000;
const lines: LogLine[] = [];
let nextId = 0;
const subscribers: Set<(line: LogLine) => void> = new Set();

/** Push a log line into the buffer and notify SSE subscribers */
export function pushLog(text: string, source: 'engine' | 'server' = 'server'): void {
  const line: LogLine = { id: nextId++, ts: Date.now(), text, source };
  lines.push(line);
  if (lines.length > MAX_LINES) {
    lines.splice(0, lines.length - MAX_LINES);
  }
  for (const cb of subscribers) {
    try { cb(line); } catch { /* subscriber dead, will be cleaned up */ }
  }
}

/** Subscribe to new log lines. Returns unsubscribe function. */
export function subscribeLines(cb: (line: LogLine) => void): () => void {
  subscribers.add(cb);
  return () => { subscribers.delete(cb); };
}

// ── SSE endpoint: GET /api/logs ───────────────────────────────────────

router.get('/', (req: Request, res: Response) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  // Send backlog
  const afterId = req.query.after ? parseInt(req.query.after as string, 10) : -1;
  for (const line of lines) {
    if (line.id > afterId) {
      res.write(`data: ${JSON.stringify(line)}\n\n`);
    }
  }

  // Stream new lines
  const onLine = (line: LogLine) => {
    try {
      res.write(`data: ${JSON.stringify(line)}\n\n`);
    } catch {
      subscribers.delete(onLine);
    }
  };
  subscribers.add(onLine);

  // Keepalive ping every 15s
  const keepalive = setInterval(() => {
    try {
      res.write(': keepalive\n\n');
    } catch {
      clearInterval(keepalive);
      subscribers.delete(onLine);
    }
  }, 15000);

  req.on('close', () => {
    clearInterval(keepalive);
    subscribers.delete(onLine);
  });
});

// ── VRAM proxy: GET /api/logs/vram ────────────────────────────────────

router.get('/vram', async (_req: Request, res: Response) => {
  try {
    const resp = await fetch(`${config.aceServer.url}/vram`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!resp.ok) {
      res.json({ used_mb: 0, total_mb: 0, free_mb: 0 });
      return;
    }
    const data = await resp.json();
    res.json(data);
  } catch {
    // ace-server not reachable or no CUDA
    res.json({ used_mb: 0, total_mb: 0, free_mb: 0 });
  }
});

export default router;
