// routes/resolve.ts — POST /api/resolve/preview
//
// Resolve a Create or written-song intent into the exact body /api/generate
// accepts, with provenance, and return it without queuing anything. The
// client submits that body unchanged. It carries `expectedBackend` set to the
// engine it was resolved for, so /api/generate refuses it (409) if the active
// engine has changed since; `version` (sha256 of the canonical body)
// identifies what was previewed. Nothing here writes a selection, a setting or
// a file.

import { Router } from 'express';
import type { Request } from 'express';
import { getUserId } from './auth.js';
import { getActiveBackendId, getBackend } from '../services/backends/registry.js';
import { resolveIntentSchema, type ResolvePreviewResponse } from '../contracts/resolution.js';
import {
  requestVersion, resolveCreateIntent, resolveWrittenSongIntent, ResolveConflictError,
} from '../services/generation/resolve/resolveIntent.js';
import {
  defaultIntentDataSources, IntentDataError, loadCreateData, loadWrittenSongData, type IntentDataSources,
} from '../services/generation/resolve/loadIntentData.js';

export interface ResolveRouterDeps {
  userId: (req: Request) => string | null;
  activeEngine: () => string;
  isEngine: (id: string) => boolean;
  sources: IntentDataSources;
}

export function createResolveRouter(deps: ResolveRouterDeps): Router {
  const router = Router();
  router.get('/path', (req, res) => {
    if (!deps.userId(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
    // 'old' is the only explicit rollback; unset or any other value selects resolved.
    res.json({ path: process.env.GENERATION_INTENT_PATH === 'old' ? 'old' : 'resolved' });
  });
  router.post('/preview', async (req, res) => {
    if (!deps.userId(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
    const parsed = resolveIntentSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: 'Invalid intent', issues: parsed.error.issues.map(i => ({ path: i.path.join('.'), message: i.message })) });
      return;
    }
    const intent = parsed.data;
    const engine = intent.engine ?? deps.activeEngine();
    if (!deps.isEngine(engine)) { res.status(400).json({ error: `Unknown engine '${engine}'` }); return; }
    try {
      const resolved = intent.kind === 'create'
        ? resolveCreateIntent(intent, engine, await loadCreateData(intent, engine, deps.sources))
        : resolveWrittenSongIntent(intent, engine, await loadWrittenSongData(intent, engine, deps.sources));
      const body: ResolvePreviewResponse = { ...resolved, version: requestVersion(resolved.request) };
      res.json(body);
    } catch (err) {
      if (err instanceof IntentDataError) { res.status(err.status).json({ error: err.message }); return; }
      if (err instanceof ResolveConflictError) { res.status(400).json({ error: err.message }); return; }
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });
  return router;
}

export default createResolveRouter({
  userId: getUserId,
  activeEngine: getActiveBackendId,
  isEngine: id => !!getBackend(id),
  sources: defaultIntentDataSources,
});
