// /api/studio-drafts: ordered playlist and revisioned editing documents.
import { Router, type Request, type Response } from 'express';
import { z } from 'zod/v4';
import { getDb } from '../db/database.js';
import { config } from '../config.js';
import { getUserId } from './auth.js';
import { saveDraftSchema, type StudioDraftBody } from '../contracts/studioDrafts.js';
import { StudioDrafts, validateDraft, withoutStaleSourceResults } from '../services/studioDrafts/index.js';
import { WorkflowError } from '../services/workflows/workflowJobs.js';

const commandSchema = z.object({ expectedRevision: z.number().int().nonnegative(), command: z.unknown() });

export function createStudioDraftsRouter(store: StudioDrafts, userIdOf: (req: Request) => string | null): Router {
  const router = Router();
  const user = (req: Request) => userIdOf(req)!;
  const fail = (res: Response, err: unknown) => {
    if (err instanceof WorkflowError) { res.status(err.status).json({ error: err.message, ...err.extra }); return; }
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  };
  const handle = (fn: (req: Request) => unknown, status = 200) => (req: Request, res: Response) => {
    try { res.status(status).json(fn(req)); } catch (err) { fail(res, err); }
  };
  router.use((req, res, next) => {
    if (!userIdOf(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
    next();
  });
  router.get('/playlist', handle(req => ({ document: store.playlist(user(req)) })));
  router.post('/playlist/commands', handle(req => {
    const input = commandSchema.safeParse(req.body);
    if (!input.success) throw new WorkflowError(400, 'Invalid playlist command request');
    return { document: store.command(user(req), input.data.expectedRevision, input.data.command) };
  }));
  router.get('/handoffs/:id', handle(req => ({ document: store.handoff(user(req), String(req.params.id)) })));
  router.get('/drafts', handle(req => ({ documents: store.drafts.list(user(req)).filter(d =>
    !req.query.studio || d.body.studio === req.query.studio) })));
  router.post('/drafts', handle(req => {
    const input = saveDraftSchema.safeParse(req.body);
    if (!input.success || input.data.expectedRevision !== undefined) throw new WorkflowError(400, 'Invalid new draft');
    validateDraft(input.data.body);
    return { document: store.drafts.create(user(req), input.data.body, { origin: 'client' }) };
  }, 201));
  router.get('/drafts/:id', handle(req => {
    const document = store.drafts.get(String(req.params.id), user(req));
    return { document, sourceError: store.sourceError(user(req), document.body, config.data.dir) };
  }));
  router.put('/drafts/:id', handle(req => {
    const input = saveDraftSchema.safeParse(req.body);
    if (!input.success || input.data.expectedRevision === undefined) throw new WorkflowError(400, 'Expected revision is required');
    validateDraft(input.data.body);
    const id = String(req.params.id);
    if (store.drafts.get(id, user(req)).body.studio !== input.data.body.studio) throw new WorkflowError(400, 'Cannot change draft studio');
    return { document: store.drafts.update(id, user(req), input.data.expectedRevision,
      (current: StudioDraftBody) => withoutStaleSourceResults(current, input.data.body), { origin: 'client' }) };
  }));
  router.delete('/drafts/:id', handle(req => {
    const expected = Number(req.query.expectedRevision);
    if (!Number.isInteger(expected) || expected < 1) throw new WorkflowError(400, 'Expected revision is required');
    store.drafts.remove(String(req.params.id), user(req), expected);
    return { removed: true };
  }));
  router.post('/import', handle(req => store.importValue(user(req), req.body)));
  return router;
}

const lazy = Router();
let inner: Router | null = null;
lazy.use((req, res, next) => { (inner ??= createStudioDraftsRouter(new StudioDrafts(getDb()), getUserId))(req, res, next); });
export default lazy;
