// models.ts — Model listing route
//
// Proxies /props from ace-server and returns available models + adapters.
// Also detects PP-VAE availability from the models directory.

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { aceClient } from '../services/aceClient.js';
import { config } from '../config.js';

const router = Router();

// GET /api/models — list available models from ace-server
router.get('/', async (_req, res) => {
  try {
    const props = await aceClient.props();
    res.json({
      models: props.models,
      adapters: props.adapters,
      config: props.cli,
      defaults: props.default,
    });
  } catch (err: any) {
    // ace-server is down (e.g. no models installed yet).
    // Return a valid empty response so the UI stays alive and can show
    // the model manager / download UI instead of crashing to a blank page.
    res.json({
      models: { dit: [], lm: [], vae: [], understand: [] },
      adapters: [],
      config: {},
      defaults: {},
      aceServerDown: true,
      error: err.message,
    });
  }
});

// GET /api/models/health — check ace-server connectivity
router.get('/health', async (_req, res) => {
  const reachable = await aceClient.isReachable();
  res.json({
    aceServer: reachable ? 'connected' : 'disconnected',
  });
});

// GET /api/models/pp-vae — check PP-VAE model availability
// Scans the models directory for pp-vae-*.gguf files.
// Returns { available: true, models: ["pp-vae-F32.gguf", ...] } or { available: false, models: [] }
router.get('/pp-vae', (_req, res) => {
  try {
    const modelsDir = config.aceServer.models;
    let ppVaeModels: string[] = [];
    if (fs.existsSync(modelsDir)) {
      ppVaeModels = fs.readdirSync(modelsDir)
        .filter(f => f.startsWith('pp-vae') && f.endsWith('.gguf'));
    }
    res.json({
      available: ppVaeModels.length > 0,
      models: ppVaeModels,
    });
  } catch (err: any) {
    res.json({ available: false, models: [], error: err.message });
  }
});

export default router;
