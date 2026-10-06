// health.ts — Health check and diagnostics route

import { Router } from 'express';
import fs from 'node:fs';
import { aceClient } from '../services/aceClient.js';
import { APP_VERSION, config } from '../config.js';
import { engineReady, engineBootStatus } from '../engineState.js';
import { startupCommit, dirtyCheckout } from '../services/training/workerUpdate.js';

const router = Router();

// Open browser tabs hold an SSE stream: /api/logs since a tab keeps one
// connection for both (App.tsx), /presence from tabs loaded before that.
// open-browser-if-needed.ps1 reads `clients` from /api/health to decide
// whether to open a new tab.
let presence = 0;
export function countPresence(req: { on(event: 'close', cb: () => void): unknown }): void {
  presence++;
  req.on('close', () => { presence--; });
}
router.get('/presence', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.write('retry: 2000\n\n');
  countPresence(req);
});

// GET /api/health — overall system health
router.get('/', async (_req, res) => {
  let aceStatus = 'disconnected';
  let aceVersion = '';

  try {
    const health = await aceClient.health();
    aceStatus = health.status || 'ok';

    // Try to get version info from props
    try {
      const props = await aceClient.props();
      aceVersion = (props as any).version || '';
    } catch {
      // Props not critical for health
    }
  } catch {
    aceStatus = 'disconnected';
  }

  res.json({
    status: 'ok',
    version: APP_VERSION,
    commit: startupCommit,
    dirty: dirtyCheckout(),
    engineBuiltAt: (() => { try { return fs.statSync(config.aceServer.exe).mtime.toISOString(); } catch { return null; } })(),
    aceServer: {
      status: aceStatus,
      url: config.aceServer.url,
      version: aceVersion,
    },
    server: {
      port: config.server.port,
      uptime: process.uptime(),
    },
    clients: presence,
    engine: {
      ready: engineReady,
      bootStatus: engineBootStatus,
    },
  });
});

export default router;
