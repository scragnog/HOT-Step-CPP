// routes/preferences.ts — /api/preferences: Batch 4 slice 7a.
//
// Named preset families (one collection each):
//   GET    /presets/:family                         list, newest first
//   POST   /presets/:family              { body }   create
//   PUT    /presets/:family/:id          { expectedRevision, body }
//   DELETE /presets/:family/:id?expectedRevision=N
//   POST   /presets/:family/import       { items }  batch import, see contracts/preferences.ts
// family: vst-chain | scale-override | ai-continue-style | ai-continue-lyric | yue2-joint
//
// Singleton settings families (one current-value document per installation):
//   GET    /settings/:family                         the document, or { document: null }
//   PUT    /settings/:family             { expectedRevision?, body }   create (first call) or update
//   POST   /settings/:family/import      { items: [one item] }
// family: ai-continue-template | storm-tuning
//
// No auth: every kind here is installation-scoped (no user id in any source
// browser key), the same reasoning routes/vst.ts already applies to the VST
// active chain. A caller still needs *some* local identity to satisfy
// TypedDocuments' signatures; services/preferences/presets.ts supplies the
// fixed LOCAL_OWNER, which installation scope discards anyway.

import { Router } from 'express';
import type { Request, Response } from 'express';
import { WorkflowError } from '../services/workflows/workflowJobs.js';
import { typedDocuments } from './workflows.js';
import { PRESET_FAMILIES, SINGLETON_FAMILIES, type PresetFamily, type SingletonFamily } from '../services/preferences/kinds.js';
import { getSingleton, importPreset, importSingleton, upsertSingleton, LOCAL_OWNER } from '../services/preferences/presets.js';
import {
  createPreferenceSchema, importPreferenceBatchSchema, updatePreferenceSchema,
} from '../contracts/preferences.js';

const router = Router();

const fail = (res: Response, err: unknown) => {
  if (err instanceof WorkflowError) { res.status(err.status).json({ error: err.message, ...err.extra }); return; }
  res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
};
const handle = (fn: (req: Request, res: Response) => unknown) => (req: Request, res: Response) => {
  try { res.json(fn(req, res)); } catch (err) { fail(res, err); }
};
const invalid = (res: Response, what: string, issues: Array<{ path: PropertyKey[]; message: string }>) => {
  res.status(400).json({ error: `Invalid ${what}`, issues: issues.map(i => ({ path: i.path.map(String).join('.'), message: i.message })) });
};

const presetFamily = (req: Request) => {
  const entry = PRESET_FAMILIES[req.params.family as PresetFamily];
  if (!entry) throw new WorkflowError(404, `Unknown preset family '${req.params.family}'`);
  return entry;
};
const singletonFamily = (req: Request) => {
  const family = req.params.family as SingletonFamily;
  const def = SINGLETON_FAMILIES[family];
  if (!def) throw new WorkflowError(404, `Unknown settings family '${req.params.family}'`);
  return def;
};

// ── Named presets ───────────────────────────────────────────────────────────

router.get('/presets/:family', handle(req => {
  const { def } = presetFamily(req);
  return { documents: typedDocuments(def).list(LOCAL_OWNER) };
}));

router.post('/presets/:family', (req, res) => {
  const parsed = createPreferenceSchema.safeParse(req.body);
  if (!parsed.success) { invalid(res, 'preset', parsed.error.issues); return; }
  handle(() => {
    const { def } = presetFamily(req);
    res.status(201);
    return { document: typedDocuments(def).create(LOCAL_OWNER, parsed.data.body, { origin: 'client' }) };
  })(req, res);
});

router.put('/presets/:family/:id', (req, res) => {
  const parsed = updatePreferenceSchema.safeParse(req.body);
  if (!parsed.success) { invalid(res, 'preset update', parsed.error.issues); return; }
  handle(() => {
    const { def } = presetFamily(req);
    return { document: typedDocuments(def).update(String(req.params.id), LOCAL_OWNER, parsed.data.expectedRevision, parsed.data.body, { origin: 'client' }) };
  })(req, res);
});

router.delete('/presets/:family/:id', handle(req => {
  const { def } = presetFamily(req);
  const expected = Number(req.query.expectedRevision);
  if (!Number.isInteger(expected) || expected < 1) throw new WorkflowError(400, 'expectedRevision is required');
  typedDocuments(def).remove(String(req.params.id), LOCAL_OWNER, expected);
  return { removed: true };
}));

router.post('/presets/:family/import', (req, res) => {
  const parsed = importPreferenceBatchSchema.safeParse(req.body);
  if (!parsed.success) { invalid(res, 'preset import', parsed.error.issues); return; }
  handle(() => {
    const { def, nameOf } = presetFamily(req);
    const td = typedDocuments(def);
    return { results: parsed.data.items.map(item => importPreset(td, nameOf, item)) };
  })(req, res);
});

// ── Singleton settings ───────────────────────────────────────────────────────

router.get('/settings/:family', handle(req => ({ document: getSingleton(typedDocuments(singletonFamily(req))) })));

router.put('/settings/:family', (req, res) => {
  const body = req.body as { expectedRevision?: number; body?: unknown };
  if (!('body' in body)) { invalid(res, 'settings update', [{ path: ['body'], message: 'Required' }]); return; }
  handle(() => ({ document: upsertSingleton(typedDocuments(singletonFamily(req)), body.expectedRevision, body.body) }))(req, res);
});

router.post('/settings/:family/import', (req, res) => {
  const parsed = importPreferenceBatchSchema.safeParse(req.body);
  if (!parsed.success) { invalid(res, 'settings import', parsed.error.issues); return; }
  handle(() => {
    const td = typedDocuments(singletonFamily(req));
    return { results: parsed.data.items.map(item => importSingleton(td, item)) };
  })(req, res);
});

export default router;
