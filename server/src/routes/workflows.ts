// routes/workflows.ts — /api/workflows: durable workflow jobs
// (services/workflows/workflowJobs.ts) and revisioned documents
// (services/workflows/revisions.ts).
//
//   POST   /jobs                    submit { kind, idempotencyKey, input }
//   GET    /jobs[?kind=&status=]    list, newest first
//   GET    /jobs/:id[?after=N]      { job, events after N, gap }
//   GET    /jobs/:id/events[?after=N]  SSE: a snapshot frame, then events
//                                      (id: seq; Last-Event-ID resumes)
//   POST   /jobs/:id/cancel
//   POST   /jobs/:id/retry          failed, cancelled or interrupted only
//   POST   /documents               create { kind, data } at revision 1
//   GET    /documents?kind=
//   GET    /documents/:id
//   PUT    /documents/:id           { expectedRevision, data }; 409 if stale
//   DELETE /documents/:id?expectedRevision=N
//
// Studios add job kinds with registerWorkflowKind() from their own modules.

import { Router } from 'express';
import type { Request, Response } from 'express';
import { getUserId } from './auth.js';
import { getDb } from '../db/database.js';
import { audioIntentQueue } from './audioQueue.js';
import { WorkflowError, WorkflowJobs, type WorkflowKind } from '../services/workflows/workflowJobs.js';
import { WorkflowDocuments } from '../services/workflows/revisions.js';
import {
  createWorkflowDocumentSchema, submitWorkflowJobSchema, updateWorkflowDocumentSchema,
  type WorkflowEvent, type WorkflowJobStatus,
} from '../contracts/workflow.js';

const STATUSES = new Set(['pending', 'running', 'succeeded', 'failed', 'cancelled', 'interrupted']);
const FINAL = new Set(['succeeded', 'failed', 'cancelled', 'interrupted']);
const HEARTBEAT_MS = 25_000;

const cursor = (v: unknown): number => {
  const n = typeof v === 'string' ? Number(v) : NaN;
  return Number.isInteger(n) && n >= 0 ? n : 0;
};

export function createWorkflowRouter(jobs: WorkflowJobs, docs: WorkflowDocuments, userIdOf: (req: Request) => string | null): Router {
  const router = Router();
  router.use((req, res, next) => {
    if (!userIdOf(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
    next();
  });
  const fail = (res: Response, err: unknown) => {
    if (err instanceof WorkflowError) { res.status(err.status).json({ error: err.message, ...err.extra }); return; }
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  };
  const handle = (fn: (req: Request, res: Response) => unknown) => (req: Request, res: Response) => {
    try { res.json(fn(req, res)); } catch (err) { fail(res, err); }
  };
  const invalid = (res: Response, what: string, issues: Array<{ path: PropertyKey[]; message: string }>) => {
    res.status(400).json({ error: `Invalid ${what}`, issues: issues.map(i => ({ path: i.path.join('.'), message: i.message })) });
  };
  const user = (req: Request) => userIdOf(req)!;
  const id = (req: Request) => String(req.params.id);

  // ── Jobs ──────────────────────────────────────────────────────────────────

  router.post('/jobs', (req, res) => {
    const parsed = submitWorkflowJobSchema.safeParse(req.body);
    if (!parsed.success) { invalid(res, 'workflow job', parsed.error.issues); return; }
    handle(() => {
      const out = jobs.submit(parsed.data, user(req));
      res.status(out.created ? 201 : 200);
      return out;
    })(req, res);
  });
  router.get('/jobs', handle(req => {
    const status = typeof req.query.status === 'string' ? req.query.status : undefined;
    if (status && !STATUSES.has(status)) throw new WorkflowError(400, `Unknown status '${status}'`);
    const kind = typeof req.query.kind === 'string' ? req.query.kind : undefined;
    return { jobs: jobs.list(user(req), { kind, status: status as WorkflowJobStatus | undefined }) };
  }));
  router.get('/jobs/:id', handle(req => jobs.replay(id(req), user(req), cursor(req.query.after))));
  router.post('/jobs/:id/cancel', handle(req => ({ job: jobs.cancel(id(req), user(req)) })));
  router.post('/jobs/:id/retry', handle(req => ({ job: jobs.retry(id(req), user(req)) })));

  router.get('/jobs/:id/events', (req, res) => {
    const jobId = id(req);
    const after = Math.max(cursor(req.query.after), cursor(req.headers['last-event-id']));
    // Subscribe before reading the replay, so nothing lands between the two.
    const live: WorkflowEvent[] = [];
    let send: ((e: WorkflowEvent) => void) | null = null;
    const unsubscribe = jobs.subscribe(jobId, e => { if (send) send(e); else live.push(e); });
    let replay;
    try { replay = jobs.replay(jobId, user(req), after); }
    catch (err) { unsubscribe(); fail(res, err); return; }
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.flushHeaders();
    let last = after;
    const heartbeat = setInterval(() => res.write(': ping\n\n'), HEARTBEAT_MS);
    const close = () => { clearInterval(heartbeat); unsubscribe(); if (!res.writableEnded) res.end(); };
    send = e => {
      if (e.seq <= last || res.writableEnded) return;
      last = e.seq;
      res.write(`id: ${e.seq}\ndata: ${JSON.stringify({ type: 'event', event: e })}\n\n`);
      // Close only at the job's current end: a retried job's stream also
      // carries the earlier attempt's final status.
      if (e.type === 'status' && FINAL.has((e.data as { status?: string } | null)?.status ?? '')) {
        const now = jobs.get(jobId);
        if (FINAL.has(now.status) && now.lastSeq <= e.seq) close();
      }
    };
    res.write(`data: ${JSON.stringify({ type: 'snapshot', job: replay.job, gap: replay.gap })}\n\n`);
    for (const e of [...replay.events, ...live]) send(e);
    // A finished job whose final event is behind the cursor has nothing more to say.
    if (FINAL.has(replay.job.status) && last >= replay.job.lastSeq) close();
    res.on('close', close);
  });

  // ── Documents ─────────────────────────────────────────────────────────────

  router.post('/documents', (req, res) => {
    const parsed = createWorkflowDocumentSchema.safeParse(req.body);
    if (!parsed.success) { invalid(res, 'document', parsed.error.issues); return; }
    handle(() => { res.status(201); return { document: docs.create(user(req), parsed.data.kind, parsed.data.data) }; })(req, res);
  });
  router.get('/documents', handle(req => {
    if (typeof req.query.kind !== 'string' || !req.query.kind) throw new WorkflowError(400, 'kind is required');
    return { documents: docs.list(user(req), req.query.kind) };
  }));
  router.get('/documents/:id', handle(req => ({ document: docs.get(id(req), user(req)) })));
  router.put('/documents/:id', (req, res) => {
    const parsed = updateWorkflowDocumentSchema.safeParse(req.body);
    if (!parsed.success) { invalid(res, 'document update', parsed.error.issues); return; }
    handle(() => ({ document: docs.update(id(req), user(req), parsed.data.expectedRevision, parsed.data.data) }))(req, res);
  });
  router.delete('/documents/:id', handle(req => {
    const expected = Number(req.query.expectedRevision);
    if (!Number.isInteger(expected) || expected < 1) throw new WorkflowError(400, 'expectedRevision is required');
    docs.remove(id(req), user(req), expected);
    return { removed: true };
  }));
  return router;
}

// ── Process singletons ──────────────────────────────────────────────────────

let jobsInstance: WorkflowJobs | null = null;
let docsInstance: WorkflowDocuments | null = null;
const earlyKinds: WorkflowKind<any>[] = [];

/** Add a job kind. Studio modules call this at import, before initDb(), so
 *  kinds wait here until the service exists. */
export function registerWorkflowKind(def: WorkflowKind<any>): void {
  if (jobsInstance) jobsInstance.register(def);
  else earlyKinds.push(def);
}

/** The process's workflow service, on the app database. On first use it
 *  marks jobs a previous process left running as interrupted, before any
 *  kind is registered, so nothing can start ahead of that. */
export function workflowJobs(): WorkflowJobs {
  if (!jobsInstance) {
    const audio = audioIntentQueue();
    jobsInstance = new WorkflowJobs({
      db: getDb(),
      audio: {
        enqueue: (input, userId) => {
          const out = audio.enqueue(input, userId);
          void audio.tick();
          return out;
        },
        get: intentId => audio.get(intentId),
        cancel: intentId => audio.cancel(intentId),
      },
    });
    jobsInstance.reconcileAfterRestart();
    for (const def of earlyKinds.splice(0)) jobsInstance.register(def);
  }
  return jobsInstance;
}

export function workflowDocuments(): WorkflowDocuments {
  return docsInstance ??= new WorkflowDocuments(getDb());
}

/** Called once the server listens: reconciles, then starts pending jobs. */
export function startWorkflowJobs(): void {
  workflowJobs().pump();
}

/** Mounted at /api/workflows. Built on its first request, after initDb(). */
const lazy = Router();
let inner: Router | null = null;
lazy.use((req, res, next) => { (inner ??= createWorkflowRouter(workflowJobs(), workflowDocuments(), getUserId))(req, res, next); });
export default lazy;
