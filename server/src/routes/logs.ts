// logs.ts — Live log streaming and VRAM proxy
//
// Ring buffer captures logs from both the Node server and ace-server child.
// SSE endpoint streams them to the UI in real time.
// VRAM endpoint proxies to ace-server's GET /vram.

import { Router, Request, Response } from 'express';
import { config } from '../config.js';
import { countPresence } from './health.js';

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

/** Noisy GGML/CUDA patterns that flood logs with no actionable info. */
const ENGINE_NOISE = [
  'CUDA graph warmup',
  'CUDA Graph id',
  'ggml_backend_cuda_graph_compute',
];

/** Push a log line into the buffer and notify SSE subscribers */
export function pushLog(text: string, source: 'engine' | 'server' = 'server'): void {
  // Suppress repetitive engine noise
  if (source === 'engine' && ENGINE_NOISE.some(p => text.includes(p))) return;

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

  res.write('retry: 2000\n\n');
  // Each tab's one stream doubles as its presence beacon (see health.ts).
  countPresence(req);

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

// ── Loaded models proxy: GET /api/logs/models-loaded ──────────────────
// Lists the GPU modules currently resident in the engine (for the manual
// unload dropdown on the VRAM indicator).
router.get('/models-loaded', async (_req: Request, res: Response) => {
  try {
    const resp = await fetch(`${config.aceServer.url}/models/loaded`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!resp.ok) { res.json({ loaded: [] }); return; }
    res.json(await resp.json());
  } catch {
    res.json({ loaded: [] });
  }
});

// ── Manual unload proxy: POST /api/logs/models-unload { label } ───────
router.post('/models-unload', async (req: Request, res: Response) => {
  try {
    const resp = await fetch(`${config.aceServer.url}/models/unload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: req.body?.label }),
      signal: AbortSignal.timeout(5000),
    });
    res.status(resp.status).json(await resp.json().catch(() => ({})));
  } catch {
    res.status(502).json({ error: 'ace-server unreachable' });
  }
});

export default router;
