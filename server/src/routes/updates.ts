// updates.ts — update notices and verified staging, without activation.
//   GET  /api/updates/status  cached state, never calls GitHub
//   POST /api/updates/check   refresh now (once a minute), optional { channel }
//   POST /api/updates/prepare { candidateId }, returns 202; poll status
//   POST /api/updates/prepare/cancel { candidateId }, cancels staging only
//   POST /api/updates/apply   { candidateId }, 202; the server stops and the
//                             stable launcher installs (services/updates/apply.ts)
// Types: contracts/updates.ts.

import { Router } from 'express';
import path from 'node:path';
import { BUILD_INFO, DATA_ROOT, LAUNCHER_PROTOCOL, PROJECT_ROOT, config } from '../config.js';
import { RUNTIME } from '../dataroot/bootstrap.js';
import { getBusyBlockers, updatePreflight } from '../dataroot/pointer.js';
import { maintenance } from '../services/updates/maintenance.js';
import { askLauncher, createApplier, installTreeProblems } from '../services/updates/apply.js';
import { ENV_PATH_KEYS } from '../dataroot/paths.js';
import { exitForLauncher } from './shutdown.js';
import { trustedOrigin } from '../services/updates/origin.js';
import { updates } from '../services/updates/index.js';
import { createDownloader, PrepareError } from '../services/updates/download.js';
import type { UpdatesApplyResponse, UpdatesCheckRequest, UpdatesCheckResponse, UpdatesStatusResponse } from '../contracts/updates.js';

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

const updaterDir = stableRoot && path.isAbsolute(stableRoot) ? path.join(stableRoot, 'updater') : '';
export const updateApplier = createApplier({
  unsupported: () => {
    if (!BUILD_INFO || BUILD_INFO.updaterProtocol < 1) return 'This build installs updates manually (updater protocol 0)';
    if (LAUNCHER_PROTOCOL !== BUILD_INFO.updaterProtocol || !updaterDir) return 'HOT-Step was not started from HOT-Step.bat in its install folder';
    if (process.platform !== 'win32') return 'Updates install automatically on Windows portable installs only';
    if (!DATA_ROOT) return 'This install has no data folder';
    return null;
  },
  revalidate: () => updateDownloads.revalidate(),
  payloadDir: PROJECT_ROOT, installRoot: stableRoot ?? '', updaterDir, dataRoot: DATA_ROOT ?? '',
  // updatePreflight guards the running app/<id>; any other payload, and the
  // launcher's own files, are covered by checking the whole install folder.
  preflight: () => [...updatePreflight(), ...installTreeProblems(stableRoot!, DATA_ROOT!, Object.fromEntries(ENV_PATH_KEYS.map(k => [k, process.env[k]])))],
  blockers: getBusyBlockers, gate: maintenance,
  snapshotSet: () => ({
    dbs: [config.data.dbPath, config.lireek.dbPath],
    files: [RUNTIME.envPath, RUNTIME.pointerPath, path.join(config.data.dir, 'lireek.db.migrated')].filter((f): f is string => !!f),
  }),
  ask: askLauncher,
  stop: () => void exitForLauncher(),
});

router.get('/status', (_req, res) => {
  res.json({ ...updates.status(), prepare: updateDownloads.status(), install: updateApplier.status() } satisfies UpdatesStatusResponse);
});

router.post('/check', async (req, res) => {
  if (!trustedOrigin(req.get('origin'), req.get('host'))) { res.status(403).json({ error: 'Cross-origin request refused' }); return; }
  const { channel } = (req.body ?? {}) as UpdatesCheckRequest;
  if (channel !== undefined && channel !== 'release' && channel !== 'master') { res.status(400).json({ error: 'channel must be "release" or "master"' }); return; }
  const checked = await updates.check({ manual: true, channel });
  if (!checked.throttled && checked.channel === 'release' && !checked.stale) await updateDownloads.discover();
  res.json({ ...checked, prepare: updateDownloads.status(), install: updateApplier.status() } satisfies UpdatesCheckResponse);
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

// 409: unsupported, not staged, launcher refused, or busy ({ blockers }: the
// gate reopened and nothing was cancelled). 400: preflight ({ errors }).
router.post('/apply', async (req, res) => {
  if (!trustedOrigin(req.get('origin'), req.get('host'))) { res.status(403).json({ error: 'Cross-origin request refused' }); return; }
  const id = candidateBody(req.body);
  if (!id) { res.status(400).json({ error: 'Only a server-issued candidateId is accepted' }); return; }
  try { res.status(202).json(await updateApplier.apply(id) satisfies UpdatesApplyResponse); }
  catch (e: any) {
    const status = e instanceof PrepareError ? e.status : Number.isInteger(e?.status) ? e.status : 500;
    res.status(status).json({ error: e?.message ?? String(e), ...(e?.blockers ? { blockers: e.blockers } : {}), ...(e?.errors ? { errors: e.errors } : {}) });
  }
});

export default router;
