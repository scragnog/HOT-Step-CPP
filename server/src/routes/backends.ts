// routes/backends.ts — backend registry + capabilities API
//
// Phase 1 scaffolding (docs/plans/multi-backend-architecture.md §4.1/§4.2).
// GET  /api/backends            — list registered backends + which is active
// GET  /api/capabilities        — active (or ?backend=<id>) backend's manifest
// POST /api/backends/active     — switch the active backend { id }
//
// Style follows routes/plugins.ts (minimal route, short TTL cache so the UI
// can poll cheaply) and generalises the capabilities-endpoint pattern in
// routes/training.ts (~176-265): every probe degrades independently, the
// route itself never throws upward.
//
// NOTE on mounting: /api/capabilities is a top-level path per the plan
// (§4.2), sitting alongside /api/backends rather than nested under it — so
// this router is mounted at '/api' (not '/api/backends') in index.ts, and
// every route below spells its own full sub-path.

import { Router } from 'express';
import {
  getBackend,
  listBackends,
  getActiveBackendId,
  setActiveBackendId,
} from '../services/backends/registry.js';
import { listMm3Planks, readMm3PlankMeta } from '../services/backends/minimax/plank.js';
import {
  listMm3Hiddens, readMm3HiddensMeta, deleteMm3Hiddens, mm3HiddensDir,
} from '../services/backends/minimax/hiddens.js';
import {
  listMm3LmAdapters,
  mm3LmAdapterDir,
  MM3_LM_ADAPTER_DEFAULT_SCALES,
} from '../services/backends/minimax/lmAdapter.js';
import type { BackendCapabilities } from '../services/backends/types.js';

const router = Router();

// GET /api/backends — { id, displayName, active }[]
router.get('/backends', (_req, res) => {
  const activeId = getActiveBackendId();
  res.json({
    backends: listBackends().map(b => ({
      id: b.id,
      displayName: b.displayName,
      resourcePool: b.resourcePool,
      active: b.id === activeId,
    })),
    activeId,
  });
});

// POST /api/backends/active { id } — switch the active backend
//
// VRAM arbitration (plan §4.4): both families live in the one ace-server
// process, so switching is model RESIDENCY, not process switching. The
// outgoing backend's weights are released fire-and-forget — the switch itself
// is a settings write and must answer instantly, and a slow/hung engine must
// never be able to wedge the toggle. Nothing here branches on backend id: it
// calls the optional `releaseVram()` the outgoing backend declares.
router.post('/backends/active', (req, res) => {
  const id = req.body?.id;
  if (typeof id !== 'string' || !id.trim()) {
    res.status(400).json({ error: 'Missing "id" in request body' });
    return;
  }
  const previousId = getActiveBackendId();
  const ok = setActiveBackendId(id);
  if (!ok) {
    res.status(404).json({ error: `Unknown backend: ${id}` });
    return;
  }

  if (previousId !== id) {
    const outgoing = getBackend(previousId);
    if (outgoing?.releaseVram) {
      void outgoing.releaseVram().catch(err => {
        // Log and move on — the user asked to switch backends, not to
        // guarantee an eviction.
        console.warn(`[backends] releaseVram failed for outgoing '${previousId}':`, err?.message || err);
      });
    }
    console.log(`[backends] active backend: ${previousId} → ${id}`);
  }

  res.json({ activeId: getActiveBackendId() });
});

// GET /api/backends/models?backend=<id> — the backend's model catalogue.
//
// Backend-shaped by design: ACE reports {lm,dit,vae,embedding} name lists,
// MiniMax-Music3 reports {lm,synth} quant ladders. The UI renders whatever
// buckets it is given rather than branching on backend id (plan §2 principle 2).
router.get('/backends/models', async (req, res) => {
  const id = (req.query.backend as string) || getActiveBackendId();
  const backend = getBackend(id);
  if (!backend) {
    res.status(404).json({ error: `Unknown backend: ${id}` });
    return;
  }
  try {
    const models = await backend.models();
    res.json({ backend: id, selectable: typeof backend.selectModel === 'function', ...models });
  } catch (err: any) {
    console.error(`[backends] models probe failed for '${id}':`, err?.message || err);
    // Same never-throw-upward contract as /capabilities: an empty catalogue is
    // an honest degrade, a 500 would blank the UI's whole Models cluster.
    res.json({ backend: id, selectable: false, buckets: {}, adapters: [], lmAdapters: [], defaults: {} });
  }
});

// POST /api/backends/models { backend?, selection: { bucket: value } }
//
// Only meaningful for backends that hold model choice as engine STATE (MM3
// picks a quant and loads it). ACE passes model names per generate call and
// declares no selectModel, so it answers 501 rather than pretending.
router.post('/backends/models', async (req, res) => {
  const id = (req.body?.backend as string) || getActiveBackendId();
  const backend = getBackend(id);
  if (!backend) {
    res.status(404).json({ error: `Unknown backend: ${id}` });
    return;
  }
  if (typeof backend.selectModel !== 'function') {
    res.status(501).json({ error: `Backend '${id}' does not support model selection` });
    return;
  }
  const selection = req.body?.selection;
  if (!selection || typeof selection !== 'object' || Array.isArray(selection)) {
    res.status(400).json({ error: 'Missing "selection" object in request body' });
    return;
  }
  try {
    const result = await backend.selectModel(selection as Record<string, string>);
    // The catalogue's `defaults` (what is actually in force) just changed.
    cache.delete(id);
    res.json(result);
  } catch (err: any) {
    // Unlike the probes above this DOES surface as an error — silently keeping
    // the old weights while the UI shows the new pick is the worst outcome.
    res.status(400).json({ error: err?.message || String(err) });
  }
});

// GET /api/capabilities?backend=<id> — defaults to the active backend.
// Cached ~10s per backend id (mirrors the 60s /api/plugins cache, shorter
// because `up` should track engine state fairly closely).
const CACHE_TTL = 10_000;
const cache = new Map<string, { data: BackendCapabilities; ts: number }>();

router.get('/capabilities', async (req, res) => {
  const id = (req.query.backend as string) || getActiveBackendId();
  const backend = getBackend(id);
  if (!backend) {
    res.status(404).json({ error: `Unknown backend: ${id}` });
    return;
  }

  const cached = cache.get(id);
  if (cached && Date.now() - cached.ts < CACHE_TTL) {
    res.json(cached.data);
    return;
  }

  try {
    const data = await backend.capabilities();
    cache.set(id, { data, ts: Date.now() });
    res.json(data);
  } catch (err: any) {
    console.error(`[backends] capabilities probe failed for '${id}':`, err.message);
    // Degrade to an honest "down" manifest rather than a 500 — mirrors the
    // never-throw-upward contract of routes/training.ts's /capabilities.
    res.json({
      backend: id,
      up: false,
      core: { duration: { max: 0, auto: false }, bpm: false, keyscale: false, negativePrompt: false, batch: { max: 1 }, seed: false },
      features: {
        models: false, lm: false, plugins: false, samplerPlugins: false, adapters: false,
        lmAdapters: false, postProcess: false,
        stableStep: false, whisper: false, lyricTimestamps: false, cover: false, repaint: false,
        lego: false, extract: false, streaming: false, training: false, midi: false,
        stems: false, understand: false, conceptSteering: false,
      },
      extensions: [],
    } satisfies BackendCapabilities);
  }
});

// ── MM3 Plank ────────────────────────────────────────────────────────────────
//
// The planks themselves are read engine-side via the generation request; these
// two routes exist only so the picker can list them and preview one before the
// user commits to a replay.

/** GET /api/mm3/planks — the saved planks, newest first. */
router.get('/mm3/planks', (_req, res) => {
  res.json({ planks: listMm3Planks() });
});

/** GET /api/mm3/plank-meta?file=<name> — one plank's sidecar metadata.
 *  `file` arrives from the browser, so it is resolved through the same
 *  containment check the replay path uses; anything outside the plank
 *  directory reads nothing. */
router.get('/mm3/plank-meta', (req, res) => {
  const ref = String(req.query.file ?? '').trim();
  if (!ref) {
    res.status(400).json({ error: 'missing ?file= parameter' });
    return;
  }
  const meta = readMm3PlankMeta(ref);
  if (!meta) {
    res.status(404).json({ error: 'plank sidecar not found or unreadable' });
    return;
  }
  res.json(meta);
});

// ── MM3 saved plans ──────────────────────────────────────────────────────────
//
// The plans themselves never cross this boundary — at ~600 MB for a 200 s song
// the engine reads and writes them itself and the server only ever hands it a
// path. These routes exist so the picker can list them, preview one, and delete
// the ones that are eating the disk.

/** GET /api/mm3/plans — the saved plans, newest first, with sizes. */
router.get('/mm3/plans', (_req, res) => {
  res.json({ plans: listMm3Hiddens(), dir: mm3HiddensDir() });
});

/** GET /api/mm3/plan-meta?file=<name> — one plan's sidecar metadata. Same
 *  containment rule as the plank route above: `file` comes from the browser. */
router.get('/mm3/plan-meta', (req, res) => {
  const ref = String(req.query.file ?? '').trim();
  if (!ref) {
    res.status(400).json({ error: 'missing ?file= parameter' });
    return;
  }
  const meta = readMm3HiddensMeta(ref);
  if (!meta) {
    res.status(404).json({ error: 'plan sidecar not found or unreadable' });
    return;
  }
  res.json(meta);
});

/** DELETE /api/mm3/plans/:file — drop a saved plan and its sidecar.
 *  These are by far the largest artefacts the app writes, so reclaiming the
 *  space has to be possible without going to the filesystem. */
router.delete('/mm3/plans/:file', (req, res) => {
  if (!deleteMm3Hiddens(String(req.params.file ?? ''))) {
    res.status(404).json({ error: 'plan not found, or the reference left the plan directory' });
    return;
  }
  res.json({ ok: true });
});

// ── MM3 runtime LM adapters ──────────────────────────────────────────────────
//
// The picker's catalogue. Adapters are applied ENGINE-side per generation
// (mm3-lm-adapter.h) from the path `params.mm3LmAdapter` resolves to, so this
// route is read-only metadata: what is installed, what each one was trained
// with, and the scale dials the UI should prefill.
//
// MM3-namespaced like the plank routes above rather than living on
// /api/backends/*: the generic catalogue (`models().lmAdapters`) carries the
// bare file list for any backend, and this adds the sidecar detail that only
// this backend has a shape for. `features.lmAdapters` in the manifest — not a
// backend id — is what gates the UI that calls it.

/** GET /api/mm3/lm-adapters — installed MM3 LM LoRAs + their sidecar metadata. */
router.get('/mm3/lm-adapters', (_req, res) => {
  // listMm3LmAdapters never throws (a missing adapters directory is an empty
  // list, which renders as "None" — the honest degrade).
  res.json({
    adapters: listMm3LmAdapters(),
    defaultScales: MM3_LM_ADAPTER_DEFAULT_SCALES,
    dir: mm3LmAdapterDir(),
  });
});

export default router;
