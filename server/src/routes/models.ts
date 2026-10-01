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
      lmAdapters: props.lm_adapters ?? [],
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
      lmAdapters: [],
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

// GET /api/models/stablestep — check StableStep (SA3) model availability
// GGML is the only engine backend: 4 GGUF files at the models dir root, plus
// tokenizer.json in onnx/sa3/ (Node tokenizes).
// Returns { available, backends: { gguf }, files } — files lists what
// is actually present in the sa3 directory.
const SA3_GGUF_FILES = [
  'sa3-dit-BF16.gguf',
  'sa3-same-enc-F16.gguf',
  'sa3-same-dec-F16.gguf',
  'sa3-text-enc-BF16.gguf',
];
router.get('/stablestep', (_req, res) => {
  try {
    const modelsDir = config.aceServer.models;
    const sa3Dir = path.join(modelsDir, 'onnx', 'sa3');
    let sa3Files: string[] = [];
    if (fs.existsSync(sa3Dir)) {
      sa3Files = fs.readdirSync(sa3Dir).filter(f => !f.endsWith('.part'));
    }
    const tokenizerOk = fs.existsSync(path.join(sa3Dir, 'tokenizer.json'));
    const ggufWeights = SA3_GGUF_FILES.every(f => fs.existsSync(path.join(modelsDir, f)));
    const gguf = tokenizerOk && ggufWeights;
    res.json({
      available: gguf,
      backends: { gguf },
      // Which half is missing. The tokenizer is 34 MB and lives under onnx/sa3
      // even though the GGML backend needs it, so deleting that folder takes
      // the backend down — and the old answer to that was a bare "not
      // installed" next to 5.8 GB of present GGUFs.
      tokenizer: tokenizerOk,
      ggufWeights,
      files: sa3Files,
    });
  } catch (err: any) {
    res.json({
      available: false,
      backends: { gguf: false },
      tokenizer: false,
      ggufWeights: false,
      files: [],
      error: err.message,
    });
  }
});

// GET /api/models/stablestep/adapters — list StableStep DoRA adapter GGUFs
// (<modelsDir>/sa3-adapters/*.gguf). These are merged into the SA3 DiT at
// refine time with per-adapter strength; GGUF backend only.
router.get('/stablestep/adapters', (_req, res) => {
  try {
    const dir = path.join(config.aceServer.models, 'sa3-adapters');
    let adapters: Array<{ name: string; sizeMb: number }> = [];
    if (fs.existsSync(dir)) {
      adapters = fs.readdirSync(dir)
        .filter(f => f.endsWith('.gguf') && !f.endsWith('.part'))
        .map(f => ({
          name: f.slice(0, -'.gguf'.length),
          sizeMb: Math.round(fs.statSync(path.join(dir, f)).size / 1e6),
        }))
        .sort((a, b) => a.name.localeCompare(b.name));
    }
    res.json({ adapters });
  } catch (err: any) {
    res.json({ adapters: [], error: err.message });
  }
});

export default router;
