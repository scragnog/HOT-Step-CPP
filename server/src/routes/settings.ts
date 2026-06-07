// settings.ts — Environment settings API
//
// Exposes whitelisted .env keys to the Settings UI with bidirectional sync.
// GET  /api/settings/env — returns current values for exposed keys
// PUT  /api/settings/env — updates .env file and hot-reloads config

import { Router } from 'express';
import fs from 'fs';
import { execFile } from 'child_process';
import { promisify } from 'util';
import {
  ENV_FILE_PATH,
  EXPOSED_ENV_KEYS,
  RESTART_REQUIRED_KEYS,
  reloadEnvConfig,
  config,
} from '../config.js';

const execFileAsync = promisify(execFile);
const router = Router();

/** Set of exposed keys for fast lookup */
const exposedSet = new Set<string>(EXPOSED_ENV_KEYS);

/** Map of env keys to their resolved defaults from config.
 *  When .env doesn't set a value, the Settings UI should still show
 *  what the server is actually using (e.g. <PROJECT_ROOT>/models). */
function getResolvedDefaults(): Record<string, string> {
  return {
    ACESTEPCPP_MODELS: config.aceServer.models,
    ACESTEPCPP_ADAPTERS: config.aceServer.adapters,
    ACESTEPCPP_PORT: String(config.aceServer.port),
    ACESTEPCPP_HOST: config.aceServer.host,
    CUDA_VISIBLE_DEVICES: config.aceServer.cudaVisibleDevices,
    SERVER_PORT: String(config.server.port),
    DATA_DIR: config.data.dir,
  };
}

/**
 * Parse .env content into an ordered array of { key, value, raw } entries.
 * Preserves comments, blank lines, and original formatting.
 */
function parseEnvLines(content: string): Array<{ key?: string; value?: string; raw: string }> {
  return content.split(/\r?\n/).map((raw) => {
    const trimmed = raw.trim();
    // blank or comment line
    if (!trimmed || trimmed.startsWith('#')) {
      return { raw };
    }
    // KEY=VALUE (capture first = only)
    const eqIdx = trimmed.indexOf('=');
    if (eqIdx > 0) {
      const key = trimmed.slice(0, eqIdx).trim();
      const value = trimmed.slice(eqIdx + 1).trim();
      return { key, value, raw };
    }
    return { raw };
  });
}

/**
 * GET /api/settings/env
 *
 * Returns the current .env values for all exposed keys.
 * Keys not set in .env are backfilled with their resolved defaults
 * so the UI always shows the actual path/value the server is using.
 * Also returns the restart-required key list so the UI can badge them.
 */
router.get('/env', (_req, res) => {
  try {
    const content = fs.existsSync(ENV_FILE_PATH)
      ? fs.readFileSync(ENV_FILE_PATH, 'utf-8')
      : '';

    const lines = parseEnvLines(content);
    const defaults = getResolvedDefaults();
    const values: Record<string, string> = {};

    // Seed all exposed keys with resolved defaults (not empty strings)
    for (const key of EXPOSED_ENV_KEYS) {
      values[key] = defaults[key] ?? '';
    }

    // Override with explicit .env values
    for (const line of lines) {
      if (line.key && exposedSet.has(line.key)) {
        values[line.key] = line.value ?? '';
      }
    }

    res.json({
      values,
      restartKeys: [...RESTART_REQUIRED_KEYS],
    });
  } catch (err: any) {
    console.error('[Settings] Failed to read .env:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/**
 * PUT /api/settings/env
 *
 * Receives a partial map of key-value pairs.
 * Updates the .env file preserving structure, then hot-reloads config.
 */
router.post('/env', (req, res) => {
  try {
    const updates: Record<string, string> = req.body?.values;
    if (!updates || typeof updates !== 'object') {
      res.status(400).json({ error: 'Missing "values" object in request body' });
      return;
    }

    // Filter to only exposed keys
    const safeUpdates: Record<string, string> = {};
    for (const [key, value] of Object.entries(updates)) {
      if (exposedSet.has(key) && typeof value === 'string') {
        safeUpdates[key] = value;
      }
    }

    if (Object.keys(safeUpdates).length === 0) {
      res.json({ updated: [], restartRequired: false });
      return;
    }

    // Read current .env
    const content = fs.existsSync(ENV_FILE_PATH)
      ? fs.readFileSync(ENV_FILE_PATH, 'utf-8')
      : '';

    const lines = parseEnvLines(content);
    const updatedKeys = new Set<string>();

    // Update existing lines in-place
    const newLines = lines.map((line) => {
      if (line.key && line.key in safeUpdates) {
        updatedKeys.add(line.key);
        return { ...line, raw: `${line.key}=${safeUpdates[line.key]}` };
      }
      return line;
    });

    // Append any keys that weren't already in the file
    for (const [key, value] of Object.entries(safeUpdates)) {
      if (!updatedKeys.has(key)) {
        newLines.push({ key, value, raw: `${key}=${value}` });
        updatedKeys.add(key);
      }
    }

    // Write back, preserving original line endings
    const eol = content.includes('\r\n') ? '\r\n' : '\n';
    const newContent = newLines.map((l) => l.raw).join(eol);
    fs.writeFileSync(ENV_FILE_PATH, newContent, 'utf-8');

    // Hot-reload into live config
    const changed = reloadEnvConfig();
    const restartRequired = changed.some((k) => RESTART_REQUIRED_KEYS.has(k));

    console.log(`[Settings] Updated .env: ${[...updatedKeys].join(', ')}${restartRequired ? ' (restart required)' : ''}`);

    res.json({
      updated: [...updatedKeys],
      restartRequired,
    });
  } catch (err: any) {
    console.error('[Settings] Failed to update .env:', err.message);
    res.status(500).json({ error: err.message });
  }
});

/**
 * GET /api/settings/gpus
 *
 * Detect available NVIDIA GPUs via nvidia-smi.
 * Returns an array of { index, name, memoryMB } objects.
 * Returns empty array if nvidia-smi is unavailable (AMD, Intel, CPU-only).
 */
router.get('/gpus', async (_req, res) => {
  try {
    const { stdout } = await execFileAsync('nvidia-smi', [
      '--query-gpu=index,name,memory.total',
      '--format=csv,noheader,nounits',
    ], { timeout: 5000 });

    const gpus = stdout.trim().split('\n')
      .filter(line => line.trim())
      .map(line => {
        const [index, name, memoryMB] = line.split(',').map(s => s.trim());
        return {
          index: parseInt(index, 10),
          name,
          memoryMB: parseInt(memoryMB, 10),
        };
      });

    res.json({ gpus });
  } catch {
    // nvidia-smi not found or failed — not an NVIDIA system
    res.json({ gpus: [] });
  }
});

export default router;
