// health.ts — Health check and diagnostics route

import { Router } from 'express';
import fs from 'node:fs';
import { aceClient } from '../services/aceClient.js';
import { APP_VERSION, BUILD_INFO, config } from '../config.js';
import { engineReady, engineBootStatus, engineNoModels, engineDevice, backendFamily } from '../engineState.js';
import { startupCommit, cachedDirtyCheckout } from '../services/training/workerUpdate.js';
import type { HealthResponse } from '../contracts/health.js';

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

  // Both probes at once, capped under the UI's 2 s abort on this route: the
  // Restart overlay and the offline check must hear Node answer even while the
  // engine is slow to reply or still booting (aceFetch retries a refused
  // connect for ~1.5 s).
  const [health, props] = await Promise.allSettled([aceClient.health(1500), aceClient.props(1500)]);
  if (health.status === 'fulfilled') {
    aceStatus = health.value.status || 'ok';
    // Props not critical for health
    if (props.status === 'fulfilled') aceVersion = (props.value as any).version || '';
  }

  res.json({
    status: 'ok',
    version: APP_VERSION,
    commit: startupCommit,
    dirty: cachedDirtyCheckout(),
    build: {
      tag: BUILD_INFO?.tag ?? null,
      sha: BUILD_INFO?.sha ?? (startupCommit || null),
      variant: BUILD_INFO?.variant ?? null,
      channel: BUILD_INFO?.channel ?? null,
    },
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
      noModels: engineNoModels,
      backend: backendFamily(engineDevice),
      device: engineDevice,
    },
  } satisfies HealthResponse);
});

export default router;
