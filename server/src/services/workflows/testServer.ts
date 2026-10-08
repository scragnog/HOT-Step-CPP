// workflows/testServer.ts — an in-memory /api/workflows for tests (server and
// ui/src/services/workflowApi.test.ts, whose packages cannot see express).
// Not imported by the app.

import express from 'express';
import Database from 'better-sqlite3';
import type { Server } from 'node:http';
import type { Request } from 'express';
import { z } from 'zod/v4';
import { WorkflowJobs, type WorkflowDeps } from './workflowJobs.js';
import { WorkflowDocuments } from './revisions.js';
import { createWorkflowRouter } from '../../routes/workflows.js';

export { z };

const noAudio: WorkflowDeps['audio'] = {
  enqueue: () => { throw new Error('No audio queue in this test'); },
  get: () => { throw new Error('No audio queue in this test'); },
  cancel: () => { throw new Error('No audio queue in this test'); },
};

export interface WorkflowTestServer {
  db: Database.Database; jobs: WorkflowJobs; docs: WorkflowDocuments; origin: string; close(): Promise<void>;
}

/** Serve the workflow router on a free port. Default user: bearer token 't' is user 'u'. */
export async function startWorkflowTestServer(opts: {
  db?: Database.Database; jobs?: WorkflowJobs; userIdOf?: (req: Request) => string | null;
} = {}): Promise<WorkflowTestServer> {
  const db = opts.db ?? new Database(':memory:');
  const jobs = opts.jobs ?? new WorkflowJobs({ db, audio: noAudio });
  const docs = new WorkflowDocuments(db);
  const app = express();
  app.use(express.json());
  app.use('/api/workflows', createWorkflowRouter(jobs, docs, opts.userIdOf ?? (req => req.headers.authorization === 'Bearer t' ? 'u' : null)));
  const server: Server = await new Promise(r => { const s = app.listen(0, () => r(s)); });
  const port = (server.address() as { port: number }).port;
  return {
    db, jobs, docs,
    origin: `http://127.0.0.1:${port}`,
    close: () => new Promise<void>(r => server.close(() => r())),
  };
}
