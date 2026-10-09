// fakeServer.ts — mounts the REAL production route factories against an
// isolated temp DB/files tree and a fake ace-server engine (fakeEngine.ts),
// so the reference client exercises actual Node logic end to end without
// ever spawning a real engine, downloading a model or calling a remote
// worker. Deliberately NOT index.ts: it skips the ace-server process spawn,
// CUDA runtime download, model-selection restore and warm-on-startup — none
// of that is route-mounting, and all of it either spawns a real process or
// reaches a real network dependency, which this harness must never do.
//
// Order matters. config.ts and db/database.ts are singletons that read
// process.env ONCE, at their first import — ACESTEPCPP_HOST/PORT (read by
// aceClient.ts:12's module-level BASE snapshot) and DATA_DIR must already be
// set before ANY route module is imported, since route modules transitively
// import config.ts. Every app/route import below is therefore a dynamic
// `await import()` performed AFTER the env is set, not a static import —
// the same trick contracts/stemSeparation.test.ts already uses for DATA_DIR
// alone; this does it for the engine URL too.

import express, { type Express } from 'express';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import type { Server } from 'node:http';
import { startFakeAceEngine, type FakeAceEngine } from './fakeEngine.js';
import { allowedOrigins, installNetworkGuard, installSubprocessGuard, violations } from './safetyGuards.js';

export interface FakeServer {
  app: Express;
  origin: string;
  dataDir: string;
  engine: FakeAceEngine;
  /** Safety-guard violations recorded during this server's lifetime (see
   *  safetyGuards.ts) — a real subprocess exec or off-origin fetch attempt,
   *  even one production code caught and degraded gracefully. A test asserts
   *  this stays empty; it must never be populated and ignored. */
  violations: string[];
  close(): Promise<void>;
}

/** Starts the fake engine, points a fresh Node process environment at it and
 *  an isolated temp data dir, then mounts every production route this batch
 *  needs. Call once per test FILE (module-level singletons like generate.ts's
 *  in-memory job map and the audio/workflow job pumps are shared across every
 *  request this server receives, exactly as they are in the real app) — not
 *  once per test case. */
export async function startFakeServer(): Promise<FakeServer> {
  const engine = await startFakeAceEngine();
  const engineUrl = new URL(engine.origin);

  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'frontend-reference-'));
  process.env.DATA_DIR = dataDir;
  process.env.ACESTEPCPP_HOST = engineUrl.hostname;
  process.env.ACESTEPCPP_PORT = engineUrl.port;

  // PROJECT_ROOT (config.ts:14) governs everything else that points at the
  // real checkout: the essentia/whisper binary paths, the models/adapters/
  // noise-sample dirs, the engine dir, and ENV_FILE_PATH (settings.ts writes
  // .env there). Pointing it at a fresh, otherwise-empty temp dir makes every
  // one of those a path that does not exist, so the real checkout's .env is
  // never touched and essentiaAvailable()'s fs.existsSync check — the thing
  // Cover's open step calls through to — is always false, with no change to
  // coverWorkflow.ts or essentiaClient.ts.
  process.env.HOT_STEP_ROOT = fs.mkdtempSync(path.join(os.tmpdir(), 'frontend-reference-root-'));

  // Delete every binary-path/model-path override the parent shell or an
  // inherited .env might have exported — each of these overrides PROJECT_ROOT
  // for its own setting (config.ts:66-69,124-125,306,324), so HOT_STEP_ROOT
  // alone does not isolate them. Explicitly set fixture paths rather than
  // deleting the model/adapter/training ones: config.ts's own fallback for
  // TRAINING_DIR joins DATA_DIR, but ACESTEPCPP_MODELS/ADAPTERS fall back to
  // PROJECT_ROOT, and WHISPER_MODELS_DIR falls back to ACESTEPCPP_MODELS — an
  // inherited value for any of the four would override those fallbacks and
  // escape the isolated root.
  delete process.env.ESSENTIA_BIN;
  delete process.env.WHISPER_EXE;
  process.env.ACESTEPCPP_MODELS = path.join(dataDir, 'fixture-models');
  process.env.ACESTEPCPP_ADAPTERS = path.join(dataDir, 'fixture-adapters');
  process.env.TRAINING_DIR = path.join(dataDir, 'fixture-training');
  process.env.WHISPER_MODELS_DIR = path.join(dataDir, 'fixture-whisper-models');

  // Fail closed before any production module is imported: no route mounted
  // below may spawn a real process or reach a real network endpoint, and
  // production catching that error gracefully must not hide it from the
  // suite (see safetyGuards.ts). Only this harness's own two servers —
  // the fake engine, already listening, and the fake server about to start —
  // may ever be fetch's target; nothing else on 127.0.0.1, any other port
  // included, is a fixture.
  installSubprocessGuard();
  installNetworkGuard();
  allowedOrigins.add(engine.origin);

  const { initDb, closeDb } = await import('../../src/db/database.js');
  initDb();

  // index.ts only flips this to true after the real ace-server child reports
  // healthy (setEngineReady, called post-spawn). There's no child here, so
  // this harness has to make the same declaration itself, or every /api/generate
  // submission 503s with "Engine not ready" before ever reaching our fake engine.
  const { setEngineReady } = await import('../../src/engineState.js');
  setEngineReady(true, 'Ready (fake engine)');

  // index.ts starts both queues' executors after its own listen() callback
  // (startAudioIntentQueue, startWorkflowJobs). Without this, items enqueued
  // by a workflow step's ctx.audio.enqueue() (e.g. Insta-Gen/Repaint/Lego's
  // render step) sit at 'pending' forever — nothing but the executor's own
  // interval ever calls queue.tick() for them, unlike /api/generate's direct
  // submit path or the audio-queue route's own POST /items handler, which
  // call tick() inline.
  const { startAudioIntentQueue, audioIntentQueue } = await import('../../src/routes/audioQueue.js');
  const { startWorkflowJobs } = await import('../../src/routes/workflows.js');
  startAudioIntentQueue();
  startWorkflowJobs();

  const authRoutes = (await import('../../src/routes/auth.js')).default;
  const songRoutes = (await import('../../src/routes/songs.js')).default;
  const generateRoutes = (await import('../../src/routes/generate.js')).default;
  const stemStudioRoutes = (await import('../../src/routes/stemStudio.js')).default;
  const supersepRoutes = (await import('../../src/routes/supersep.js')).default;
  const songBuilderRoutes = (await import('../../src/routes/songBuilder.js')).default;
  const trainingRoutes = (await import('../../src/routes/training.js')).default;
  const backendsRoutes = (await import('../../src/routes/backends.js')).default;
  const workflowRoutes = (await import('../../src/routes/workflows.js')).default;
  const { registerRepaintLayerWorkflows } = await import('../../src/services/workflows/repaintLayerWorkflow.js');
  const preferencesRoutes = (await import('../../src/routes/preferences.js')).default;
  const studioDraftsRoutes = (await import('../../src/routes/studioDrafts.js')).default;
  const streamSessionsRoutes = (await import('../../src/routes/streamSessions.js')).default;
  const exportImportRoutes = (await import('../../src/routes/exportImport.js')).default;
  const audioQueueRoutes = (await import('../../src/routes/audioQueue.js')).default;
  const settingsRoutes = (await import('../../src/routes/settings.js')).default;
  const uploadRoutes = (await import('../../src/routes/upload.js')).default;
  const resolveRoutes = (await import('../../src/routes/resolve.js')).default;
  // Insta-Gen's and Cover's WorkflowKinds register as an IMPORT SIDE EFFECT
  // of these two route modules (inspire.ts:531, yue2Cover.ts:182) — there is
  // no separate registerXWorkflows() call anywhere else to make instead, so
  // both routes must be mounted even though the test sequences below call
  // /api/workflows/jobs, never /api/inspire or /api/yue2-cover directly.
  const inspireRoutes = (await import('../../src/routes/inspire.js')).default;
  const yue2CoverRoutes = (await import('../../src/routes/yue2Cover.js')).default;

  const app = express();
  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ extended: true, limit: '50mb' }));

  app.use('/api/auth', authRoutes);
  app.use('/api/songs', songRoutes);
  app.use('/api/generate', generateRoutes);
  app.use('/api/stem-studio', stemStudioRoutes);
  app.use('/api/supersep', supersepRoutes);
  app.use('/api/builder', songBuilderRoutes);
  app.use('/api/training', trainingRoutes);
  app.use('/api/audio-queue', audioQueueRoutes);
  app.use('/api/workflows', workflowRoutes);
  app.use('/api/preferences', preferencesRoutes);
  app.use('/api/studio-drafts', studioDraftsRoutes);
  app.use('/api/stream-sessions', streamSessionsRoutes);
  app.use('/api/export-import', exportImportRoutes);
  app.use('/api/settings', settingsRoutes);
  app.use('/api/upload', uploadRoutes);
  app.use('/api/resolve', resolveRoutes);
  app.use('/api/inspire', inspireRoutes);
  app.use('/api/yue2-cover', yue2CoverRoutes);
  // Mounted at '/api', not '/api/backends' — this router spells its own full
  // sub-paths (/backends, /capabilities, ...), same as index.ts:142.
  app.use('/api', backendsRoutes);

  // Repaint and Lego (Stem Builder) have no route of their own to register
  // their workflow kinds from — index.ts:138 calls this for the same reason.
  registerRepaintLayerWorkflows();

  const server: Server = await new Promise(resolve => {
    const s = app.listen(0, '127.0.0.1', () => resolve(s));
  });
  const port = (server.address() as { port: number }).port;
  const origin = `http://127.0.0.1:${port}`;
  // The exact origin client.ts's fetch calls target — added only now, once
  // it is actually known, so there is no window where a guess could be wrong.
  allowedOrigins.add(origin);

  return {
    app,
    engine,
    dataDir,
    violations,
    origin,
    close: async () => {
      // The queue's own setInterval is unref'd so it never blocks process
      // exit on its own, but it keeps firing on this module-level singleton
      // after closeDb() below — and ticking a closed db throws, which the
      // test runner's own uncaughtException handling turns into a hang
      // rather than a clean exit. stop() before closeDb(), same ordering
      // a real shutdown would need, which index.ts never does today.
      audioIntentQueue().stop();
      await new Promise<void>(resolve => server.close(() => resolve()));
      await engine.close();
      closeDb();
      fs.rmSync(dataDir, { recursive: true, force: true });
      fs.rmSync(process.env.HOT_STEP_ROOT!, { recursive: true, force: true });
    },
  };
}
