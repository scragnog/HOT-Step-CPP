// routes/resolve.ts — POST /api/resolve/preview
//
// Resolve a Create or written-song intent into the exact body /api/generate
// accepts, with provenance, and return it without queuing anything. The
// client submits that body unchanged; `version` (sha256 of the canonical body)
// identifies what was previewed, and verifyResolvedRequest checks a submitted
// body against it. Nothing here writes a selection, a setting or a file.

import { Router } from 'express';
import { getUserId } from './auth.js';
import { getActiveBackendId, getBackend } from '../services/backends/registry.js';
import { resolveIntentSchema, type ResolvePreviewResponse } from '../contracts/resolution.js';
import { requestVersion, resolveCreateIntent, resolveWrittenSongIntent } from '../services/generation/resolve/resolveIntent.js';
import { IntentDataError, loadCreateData, loadWrittenSongData } from '../services/generation/resolve/loadIntentData.js';

const router = Router();

router.post('/preview', async (req, res) => {
  if (!getUserId(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
  const parsed = resolveIntentSchema.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: 'Invalid intent', issues: parsed.error.issues.map(i => ({ path: i.path.join('.'), message: i.message })) });
    return;
  }
  const intent = parsed.data;
  const engine = intent.engine ?? getActiveBackendId();
  if (!getBackend(engine)) { res.status(400).json({ error: `Unknown engine '${engine}'` }); return; }
  try {
    const resolved = intent.kind === 'create'
      ? resolveCreateIntent(intent, engine, await loadCreateData(intent, engine))
      : resolveWrittenSongIntent(intent, engine, await loadWrittenSongData(intent, engine));
    const body: ResolvePreviewResponse = { ...resolved, version: requestVersion(resolved.request) };
    res.json(body);
  } catch (err) {
    if (err instanceof IntentDataError) { res.status(err.status).json({ error: err.message }); return; }
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  }
});

export default router;
