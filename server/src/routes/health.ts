// health.ts — Health check and diagnostics route

import { Router } from 'express';
import { aceClient } from '../services/aceClient.js';
import { config } from '../config.js';
import { engineReady, engineBootStatus } from '../engineState.js';

const router = Router();

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
    aceServer: {
      status: aceStatus,
      url: config.aceServer.url,
      version: aceVersion,
    },
    server: {
      port: config.server.port,
      uptime: process.uptime(),
    },
    engine: {
      ready: engineReady,
      bootStatus: engineBootStatus,
    },
  });
});

export default router;
