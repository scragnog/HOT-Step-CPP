// settings.ts — Environment settings API
//
// Exposes whitelisted .env keys to the Settings UI with bidirectional sync.
// GET  /api/settings/env — returns current values for exposed keys
// PUT  /api/settings/env — updates .env file and hot-reloads config

import { Router } from 'express';
import fs from 'fs';
import {
  ENV_FILE_PATH,
  EXPOSED_ENV_KEYS,
  RESTART_REQUIRED_KEYS,
  reloadEnvConfig,
} from '../config.js';

const router = Router();

/** Set of exposed keys for fast lookup */
const exposedSet = new Set<string>(EXPOSED_ENV_KEYS);

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
 * Also returns the restart-required key list so the UI can badge them.
 */
router.get('/env', (_req, res) => {
  try {
    const content = fs.existsSync(ENV_FILE_PATH)
      ? fs.readFileSync(ENV_FILE_PATH, 'utf-8')
      : '';

    const lines = parseEnvLines(content);
    const values: Record<string, string> = {};

    // Seed all exposed keys with empty strings so UI always gets full list
    for (const key of EXPOSED_ENV_KEYS) {
      values[key] = '';
    }

    // Fill from .env file
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

export default router;
