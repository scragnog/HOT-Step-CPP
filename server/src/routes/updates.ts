// updates.ts — update notices and verified staging, without activation.
//   GET  /api/updates/status  cached state, never calls GitHub
//   POST /api/updates/check   refresh now (once a minute), optional { channel }
//   POST /api/updates/prepare { candidateId }, returns 202; poll status
//   POST /api/updates/prepare/cancel { candidateId }, cancels staging only
// Types: contracts/updates.ts.

import { Router } from 'express';
import path from 'node:path';
import { DATA_ROOT, PROJECT_ROOT } from '../config.js';
import { trustedOrigin } from '../services/updates/origin.js';
import { updates } from '../services/updates/index.js';
import { createDownloader, PrepareError } from '../services/updates/download.js';
import type { UpdatesCheckRequest, UpdatesCheckResponse, UpdatesStatusResponse } from '../contracts/updates.js';

const router = Router();
// The protocol-1 launcher supplies its stable install root. No fallback into
// the selected version or data folder is safe for updater state.
const stableRoot = process.env.HOT_STEP_INSTALL_ROOT;
export const updateDownloads = createDownloader({
  fetch: (url, init) => fetch(url, init), install: () => updates.status().installed,
  platform: process.platform, arch: process.arch, now: Date.now,
  payloadRoot: PROJECT_ROOT, dataRoot: DATA_ROOT,
  stateRoot: stableRoot && path.isAbsolute(stableRoot) ? path.join(stableRoot, 'updater') : null,
});

router.get('/status', (_req, res) => {
  res.json({ ...updates.status(), prepare: updateDownloads.status() } satisfies UpdatesStatusResponse);
});

router.post('/check', async (req, res) => {
  if (!trustedOrigin(req.get('origin'), req.get('host'))) { res.status(403).json({ error: 'Cross-origin request refused' }); return; }
  const { channel } = (req.body ?? {}) as UpdatesCheckRequest;
  if (channel !== undefined && channel !== 'release' && channel !== 'master') { res.status(400).json({ error: 'channel must be "release" or "master"' }); return; }
  const checked = await updates.check({ manual: true, channel });
  if (!checked.throttled && checked.channel === 'release' && !checked.stale) await updateDownloads.discover();
  res.json({ ...checked, prepare: updateDownloads.status() } satisfies UpdatesCheckResponse);
});

function candidateBody(body: unknown): string | null {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return null;
  const b = body as Record<string, unknown>;
  return Object.keys(b).length === 1 && typeof b.candidateId === 'string' && /^[0-9a-f-]{36}$/.test(b.candidateId) ? b.candidateId : null;
}

router.post('/prepare', (req, res) => {
  if (!trustedOrigin(req.get('origin'), req.get('host'))) { res.status(403).json({ error: 'Cross-origin request refused' }); return; }
  const id = candidateBody(req.body);
  if (!id) { res.status(400).json({ error: 'Only a server-issued candidateId is accepted' }); return; }
  try { res.status(202).json(updateDownloads.prepare(id)); }
  catch (e) { res.status(e instanceof PrepareError ? e.status : 500).json({ error: (e as Error).message }); }
});

router.post('/prepare/cancel', (req, res) => {
  if (!trustedOrigin(req.get('origin'), req.get('host'))) { res.status(403).json({ error: 'Cross-origin request refused' }); return; }
  const id = candidateBody(req.body);
  if (!id) { res.status(400).json({ error: 'Only a server-issued candidateId is accepted' }); return; }
  try { res.status(202).json(updateDownloads.cancel(id)); }
  catch (e) { res.status(e instanceof PrepareError ? e.status : 500).json({ error: (e as Error).message }); }
});

export default router;
