// generate.ts — Generation orchestration route
//
// Orchestrates the two-step generation flow:
//   1. POST /lm → poll → get enriched JSON with audio_codes
//   2. POST /synth → poll → get audio
//   3. Save audio + metadata to SQLite
//
// Maintains an in-memory job map for frontend polling.
// LM results are cached by seed+params to skip the LM phase on repeats.

import { Router } from 'express';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { aceClient } from '../services/aceClient.js';
import { config } from '../config.js';
import { getUserId } from './auth.js';
import { logGeneration, failGenerationLog } from '../services/logger.js';
import { engineReady, engineBootStatus } from '../engineState.js';
import { isEngineSuspended, engineCrashSeq } from '../services/aceEngineProcess.js';
import { pushLog } from './logs.js';
import { getBackend, getActiveBackendId } from '../services/backends/registry.js';
import { mm3StreamUrl } from '../services/backends/minimax/client.js';
import {
  runOnGpuLane, runOnWorkerLane, gpuLaneBusy, gpuLaneDepth, gpuLaneOwner, gpuLaneNextFamily, releaseGpuLane, resetGpuLane,
  type LaneLease,
} from '../services/generation/gpuLane.js';
import { inWorkerScope, workerForJob } from '../services/engineWorkers.js';
import { isActiveJob, type GenerationJob } from '../services/generation/jobTypes.js';
import { pollUntilDone } from '../services/generation/pollUntilDone.js';
import { dropYue2BatchMember } from '../services/backends/yue2/generate.js';
import { yue2Synth, splitMultipartMixed } from '../services/backends/yue2/client.js';
import { tokenizeForSa3 } from '../services/sa3Tokenizer.js';
import { buildSlotRequest, renderSlotOnLane, waitForSlotJob, type SlotJobEnd } from '../services/storm/slot.js';
import { initDrift, updateDriftPressure, resolveStartDrift, defaultDriftPressure, snapshot as driftSnapshot } from '../services/storm/drift.js';
import type { DriftRuntime, DriftPressure } from '../services/storm/drift.js';
import { translateParams } from '../services/generation/translateParams.js';
import { timbreReferenceMissing } from '../services/generation/sourceAudio.js';
import { buildEnvelope, GenerationEnvelopeError } from '../services/generation/envelope.js';
import { isGenerationIntent, readBackendOverride } from '../contracts/generation.js';
import { preparedParams } from '../contracts/yue2Prepare.js';
import { GenerationIntentError, resolveGenerationIntent } from '../services/generation/intent.js';
import { evictOtherFamilies, noteEnqueued, noteFinished } from '../services/generation/residency.js';
import { runYue2PlanPreview } from '../services/backends/yue2/generate.js';
import {
  checkoutState, createGenerateCapture, generateCaptureDir, generateCaptureMode,
} from '../services/generation/requestCapture.js';
import type {
  GenerationAttempt,
  GenerationEndReason,
  GenerationOutcome,
} from '../services/backends/types.js';
import { openStreamSession, type StreamSession } from '../services/streamSessions/index.js';
import { getDb } from '../db/database.js';
import { GenerationJobLedger, ledgerStatusBody, type LedgerEntry } from '../services/generation/jobLedger.js';
import { registerBusyCheck, registerJobsBusyCheck, trackBusy } from '../dataroot/pointer.js';

export type { GenerationJob, StageTiming } from '../services/generation/jobTypes.js';

const router = Router();

const jobs = new Map<string, GenerationJob>();
registerJobsBusyCheck('generation', jobs);

// Each job's outcome, kept in the database so /status can still answer after a
// restart empties `jobs` (services/generation/jobLedger.ts). Best-effort: a
// ledger failure is logged and never blocks or fails a generation.
let ledgerInstance: GenerationJobLedger | null = null;
function withLedger(what: string, fn: (ledger: GenerationJobLedger) => void): void {
  try { fn(ledgerInstance ??= new GenerationJobLedger(getDb())); } catch (err) {
    console.warn(`[Generate] job ledger ${what} failed: ${(err as Error).message}`);
  }
}

/** Tests only: run the ledger on a test database instead of the app's. */
export function setGenerationJobLedgerForTests(ledger: GenerationJobLedger | null): void {
  ledgerInstance = ledger;
}

/** Mark a job cancelled, in memory and in the ledger. A pending job may never
 *  reach its lane (a restart first), so the lane's own ledger write is not enough. */
export function markGenerationJobCancelled(job: GenerationJob): void {
  job.status = 'cancelled';
  withLedger('cancel', ledger => ledger.finished(job));
}

// TTL cleanup: prune terminal jobs older than 1 hour every 10 minutes.
// Prevents unbounded memory growth during long batch sessions.
const JOB_TTL_MS = 60 * 60 * 1000; // 1 hour
setInterval(() => {
  const now = Date.now();
  let pruned = 0;
  for (const [id, job] of jobs) {
    if (['succeeded', 'failed', 'cancelled'].includes(job.status) && now - job.createdAt > JOB_TTL_MS) {
      jobs.delete(id);
      pruned++;
    }
  }
  if (pruned > 0) console.log(`[Generate] Pruned ${pruned} terminal job(s) from memory`);
}, 10 * 60 * 1000).unref();

// translateParams is now imported from ../services/generation/translateParams.ts

/** Pure guard, kept exportable for a server-side test: a caller (the MCP
 *  server submits one on every generation) that names which backend it
 *  planned the request for, when that no longer matches the active one. Reads
 *  `expectedBackend` only — `body.backend` stays the existing log-only field
 *  envelope.ts already reads for `submittedBackendMismatch`. */
export function expectedBackendMismatch(
  body: unknown, activeBackendId: string,
): { expectedBackend: string; activeBackend: string } | null {
  const expected = (body as Record<string, unknown> | null | undefined)?.expectedBackend;
  if (typeof expected !== 'string' || expected === activeBackendId) return null;
  return { expectedBackend: expected, activeBackend: activeBackendId };
}

function emptyOutcome(job: GenerationJob, endReason: GenerationEndReason): GenerationOutcome {
  return {
    endReason,
    stages: [],
    artifacts: [],
    songIds: [],
    result: job.result,
    error: job.error,
  };
}

function effectiveSeed(job: GenerationJob, attempt: GenerationAttempt): number | string | undefined {
  // MM3's exact uint64 seed string is authoritative when it is available;
  // never round it through Number. ACE and ordinary MM3 renders retain the
  // existing numeric job.params value.
  const exactMm3Seed = job.mm3TakeSeeds?.[0]
    ?? (typeof attempt.effective.seed === 'string' ? attempt.effective.seed : undefined);
  if (typeof exactMm3Seed === 'string' && exactMm3Seed.length > 0) return exactMm3Seed;
  const seed = job.params?.seed;
  return typeof seed === 'number' || typeof seed === 'string' ? seed : undefined;
}

function finalizeAttempt(
  attempt: GenerationAttempt,
  job: GenerationJob,
  outcome?: GenerationOutcome,
  thrownError?: string,
): void {
  const seed = effectiveSeed(job, attempt);
  if (seed !== undefined) attempt.effective.seed = seed;
  const lmSeed = job.params?.lmSeed;
  if (typeof lmSeed === 'number' || typeof lmSeed === 'string') {
    attempt.effective.lmSeed = lmSeed;
  }
  attempt.endedAt = Date.now();
  attempt.error = thrownError ?? outcome?.error ?? job.error;
  attempt.endReason = outcome?.endReason
    ?? (job.status === 'cancelled' || thrownError?.includes('Cancelled') ? 'cancelled' : 'failed');
}

/** Run the full generation pipeline */
async function runGeneration(
  job: GenerationJob,
  signal: AbortSignal,
  attempt: GenerationAttempt,
  lease: LaneLease,
): Promise<GenerationOutcome> {
  if (job.status === 'cancelled') return emptyOutcome(job, 'cancelled');
  const backendId = job.envelope?.backendId;
  const backend = backendId ? getBackend(backendId) : undefined;
  if (!backend) throw new Error(`Captured generation backend '${backendId ?? '(missing)'}' is not registered`);
  if (!lease.isCurrent()) return emptyOutcome(job, 'reset');
  // A backend switch evicts the outgoing family in the background; no family
  // loads weights until that eviction, and every other family's, is confirmed.
  // A batch follower's own turn does no engine work: leave its worker's VRAM be.
  if (!job.coalescedInto) await evictOtherFamilies(backendId);
  return backend.generate(job, {
    envelope: job.envelope,
    attempt,
    lease,
    signal,
    pollUntilDone,
    hooks: { onEngineJob() {}, onStage() {}, onArtifact() {} },
    // Only jobs no worker has taken yet: one already dispatched is preparing
    // in its own lane turn, maybe on another worker.
    pendingJobs: () => Array.from(jobs.values())
      .filter(j => j.id !== job.id && j.status === 'pending' && !j.workerId && j.envelope?.backendId === backendId)
      .sort((a, b) => a.createdAt - b.createdAt),
    releaseLane: () => releaseGpuLane(lease),
    nextLaneFamily: gpuLaneNextFamily,
    runOnLane: fn => runOnGpuLane(fn, { label: `${job.id} finish`, family: backendId }),
    assignFollower(follower, lead) {
      follower.workerId = lead?.workerId;
      withLedger('assign', ledger => ledger.assigned(follower));
    },
  });
}
// ── Async generation queue ────────────────────────────────────────────
// Each generation holds one engine worker's lease on the shared GPU lane
// (services/generation/gpuLane.ts) for its whole run, every retry included,
// and every engine call, log line and crash epoch inside follows that worker.
// One engine runs one job at a time; with worker concurrency off (the
// default) the primary is the only worker, so generations run one at a time
// as they always have. The lane is shared rather than private because
// post-processing re-runs are just as GPU-hungry and must not race a render:
// those stay exclusive and wait for every worker.
/** Each queued job's whole lane run, so a batch follower can wait for its lead. */
const laneRuns = new Map<string, Promise<void>>();

function enqueueGeneration(job: GenerationJob): void {
  const family = job.envelope.backendId;
  noteEnqueued(family);
  if (gpuLaneBusy()) {
    console.log(`[Generate] Job ${job.id} queued (${gpuLaneDepth() + 1} waiting)`);
  }

  const run = runOnWorkerLane(async (lease) => {
    // A YuE2 batch follower already runs inside its lead's engine job.
    if (!job.coalescedInto) job.workerId = lease.worker!.id;
    withLedger('assign', ledger => ledger.assigned(job));
    const retryPolicy = job.envelope.policy.retry;
    let attemptNumber = 0;
    let reseededForAttempt = false;

    while (attemptNumber < retryPolicy.maxAttempts) {
      attemptNumber++;
      const attempt: GenerationAttempt = {
        attempt: attemptNumber,
        startedAt: Date.now(),
        effective: { models: structuredClone(job.envelope.models) },
        reseeded: reseededForAttempt,
        engineJobIds: [],
        ...(job.envelope.submittedBackendMismatch === undefined ? {} : {
          submittedBackendMismatch: job.envelope.submittedBackendMismatch,
        }),
      };
      job.attempts.push(attempt);
      const abortController = new AbortController();
      (job as any)._abort = abortController;
      try {
        const outcome = await runGeneration(job, abortController.signal, attempt, lease);
        finalizeAttempt(attempt, job, outcome);
        break; // A returned outcome, including a consumed failure, ends this retry scope.
      } catch (err: any) {
        const msg = err.message || '';
        const isRetryable = !msg.includes('Cancelled')
          && !msg.includes('Unauthorized')
          && job.status !== 'cancelled';
        finalizeAttempt(attempt, job, undefined, msg);

        if (isRetryable && attemptNumber < retryPolicy.maxAttempts) {
          console.log(`[Generate] Job ${job.id} failed (attempt ${attemptNumber}), retrying: ${msg}`);
          logGeneration(job.id, 'WARNING', `[Retry] Attempt ${attemptNumber} failed: ${msg} — retrying with new seed...`);

          // Reset job state for retry
          job.status = 'pending';
          job.stage = `Retrying (attempt ${attemptNumber + 1})...`;
          job.progress = 0;
          job.error = undefined;
          job.aceJobId = undefined;

          // Randomize seed on retry — bad LM output (same seed) may have caused the stall
          if (retryPolicy.reseedOnRetry) {
            job.params.seed = Math.floor(Math.random() * 2_147_483_647);
            job.params.randomSeed = true;
            // Finalize the current attempt before changing job.params. The
            // next attempt owns the reseeded flag and captures this new seed.
            reseededForAttempt = true;
          }

          // Brief pause before retry
          await new Promise(r => setTimeout(r, 2000));
        } else {
          // Final failure — no more retries
          job.status = 'failed';
          job.error = msg;
          job.stage = 'Failed';
          console.error(`[Generate] Job ${job.id} failed permanently${attemptNumber > 1 ? ` after ${attemptNumber} attempt(s)` : ''}: ${msg}`);
          failGenerationLog(job.id, msg, 'unknown');
          break;
        }
      } finally {
        if ((job as any)._abort === abortController) delete (job as any)._abort;
      }
    }

  }, { label: `generate:${job.id}`, family }).catch((err: any) => {
    // The retry loop above swallows every generation failure, so reaching here
    // means the lane never ran the job (no engine worker can take it) or broke.
    // End the job with the error unless a cancel or reset already ended it.
    const msg = err?.message || String(err);
    console.error(`[Generate] Job ${job.id} lane error:`, msg);
    if (isActiveJob(job)) {
      job.status = 'failed';
      job.error = msg;
      job.stage = 'Failed';
      failGenerationLog(job.id, msg, 'unknown');
    }
  }).finally(() => {
    noteFinished(family);
  }).then(async () => {
    // A batch follower's own turn returns at once; how it ends (rendered, or
    // handed back because the batch failed or its lead was cancelled) is only
    // known once the lead's run ends. Record it then, and queue a handed-back
    // job again, so it is never stranded.
    const lead = job.coalescedInto ? laneRuns.get(job.coalescedInto) : undefined;
    if (lead) await lead;
    withLedger('finish', ledger => ledger.finished(job));
    if (job.status === 'pending' && !job.coalescedInto) enqueueGeneration(job);
  }).finally(() => { if (laneRuns.get(job.id) === run) laneRuns.delete(job.id); });
  laneRuns.set(job.id, run);
  void trackBusy('generation', run);
}

// POST /api/generate — start a generation job
// ── YuE2 score preview ────────────────────────────────────────────────────
// Plan the lead sheet only and hand it back (seconds), so the user can look
// at it before committing to a render (minutes). Runs on the GPU lane like a
// generation so it never overlaps one; the approved score comes back in as
// `yue2Abc` on an ordinary POST / with the seed pinned.
router.post('/yue2/plan', async (req, res) => {
  if (isEngineSuspended()) {
    res.status(503).json({ error: 'Engine is paused for training preprocessing — try again when the job finishes' });
    return;
  }
  if (!engineReady) {
    res.status(503).json({ error: `Engine not ready: ${engineBootStatus}` });
    return;
  }
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }
  if (getActiveBackendId() !== 'yue2') {
    res.status(400).json({ error: 'Score preview is a YuE2 feature — switch the active backend to YuE2' });
    return;
  }
  const params = { ...(req.body ?? {}), backend: 'yue2' };
  const abort = new AbortController();
  // res 'close' fires on client disconnect; req 'close' fires as soon as the
  // body is consumed on Node 16+, which cancelled every preview immediately.
  res.on('close', () => { if (!res.writableFinished) abort.abort(); });
  try {
    noteEnqueued('yue2');
    const preview = await runOnGpuLane(async () => {
      await evictOtherFamilies('yue2');
      return runYue2PlanPreview(params, abort.signal);
    }, { label: 'yue2 score preview', family: 'yue2' });
    res.json(preview);
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  } finally {
    noteFinished('yue2');
  }
});

// Dev-only fixture capture (HOTSTEP_GENERATE_CAPTURE; off by default). It runs
// ahead of the handler, so the body is recorded exactly as the caller sent it,
// and it needs the same token: an unauthenticated caller writes nothing.
const generateCapture = createGenerateCapture({
  mode: () => generateCaptureMode(),
  userId: getUserId,
  activeBackendId: getActiveBackendId,
  dir: generateCaptureDir,
  checkout: () => checkoutState(),
});

export type SubmitGenerationResult =
  | { ok: true; jobId: string; status: GenerationJob['status'] }
  | { ok: false; status: number; body: Record<string, unknown> };

/** A submission carrying a YuE2 Prepare document (`yue2Prepare`) passes the
 *  shared check (contracts/yue2Prepare.ts) and is queued without it; any
 *  other submission is returned as it came. */
export function withoutPrepareDocument(submission: any): { submission: any } | { error: string } {
  if (!submission || typeof submission !== 'object' || submission.yue2Prepare === undefined) return { submission };
  const pair = preparedParams(submission);
  if ('error' in pair) return pair;
  const { yue2Prepare: _, ...rest } = submission;
  return { submission: rest };
}

/** Validate and queue one generation, exactly as POST /api/generate does.
 *  Exported so the durable audio intent queue (services/audioQueue) submits
 *  through the same checks, job map and GPU lane instead of a second path. */
export async function submitGeneration(raw: any, userId: string | null): Promise<SubmitGenerationResult> {
  // The engine is deliberately stopped while a training preprocess job owns the
  // GPU — say so instead of the generic "not ready" boot message.
  if (isEngineSuspended()) {
    return { ok: false, status: 503, body: { error: 'Engine is paused for training preprocessing — try again when the job finishes' } };
  }

  // Reject requests while engine is still bootstrapping (downloading DLLs, etc.)
  if (!engineReady) {
    return { ok: false, status: 503, body: {
      error: `Engine not ready: ${engineBootStatus}`,
      detail: 'The CUDA runtime is still being set up. Please wait a moment and try again.',
    } };
  }

  if (!userId) return { ok: false, status: 401, body: { error: 'Unauthorized' } };

  // An MCP (or other headless) caller asserts which backend it planned the
  // request for; the UI never sends this. `backend` stays log-only — only
  // `expectedBackend` can block a submit, so a caller that races a backend
  // switch fails loudly instead of training/rendering on the wrong one.
  // With a backendOverride, `expectedBackend` is checked against the override
  // (the backend the job will run on); without one, against the active
  // backend. A malformed override is a 400 here; an unknown one skips this and
  // gets its 400 below.
  let checkedOverride: string | undefined;
  try { checkedOverride = readBackendOverride(raw); } catch (err) {
    return { ok: false, status: 400, body: { error: err instanceof Error ? err.message : 'Invalid backendOverride' } };
  }
  if (checkedOverride === undefined || getBackend(checkedOverride)) {
    const mismatch = expectedBackendMismatch(
      isGenerationIntent(raw) ? raw.input : raw,
      checkedOverride ?? getActiveBackendId(),
    );
    if (mismatch) {
      return { ok: false, status: 409, body: checkedOverride === undefined ? {
        error: `Expected backend '${mismatch.expectedBackend}' but the active backend is '${mismatch.activeBackend}'`,
        ...mismatch,
      } : {
        error: `Expected backend '${mismatch.expectedBackend}' but backendOverride selects '${checkedOverride}'`,
        expectedBackend: mismatch.expectedBackend,
        backendOverride: checkedOverride,
        activeBackend: getActiveBackendId(),
      } };
    }
  }

  let submission = raw;
  if (isGenerationIntent(raw)) {
    try {
      // An opt-in `backendOverride` resolves the intent for that backend;
      // otherwise the active one, as before.
      let override: string | undefined;
      try { override = readBackendOverride(raw); } catch (err) {
        throw new GenerationIntentError(err instanceof Error ? err.message : 'Invalid backendOverride');
      }
      if (override !== undefined && !getBackend(override)) {
        throw new GenerationIntentError(`Unknown backendOverride '${override}'`);
      }
      const backendId = override ?? getActiveBackendId();
      const backend = getBackend(backendId);
      if (!backend) throw new Error(`Active generation backend '${backendId}' is not registered`);
      const capabilities = await backend.capabilities();
      if (override === undefined && getActiveBackendId() !== backendId) {
        return { ok: false, status: 409, body: { error: `Active backend changed while resolving generation intent`, activeBackend: getActiveBackendId() } };
      }
      submission = resolveGenerationIntent(raw, backendId, capabilities.extensions);
    } catch (err) {
      return { ok: false, status: err instanceof GenerationIntentError ? 400 : 500, body: {
        error: err instanceof Error ? err.message : 'Cannot resolve generation intent',
      } };
    }
  }

  // A profile can carry a timbre reference whose upload was deleted; the
  // synth phase would find that out minutes in. Refuse it at the form (#162).
  if (!submission?.backend || submission.backend === 'ace') {
    const missing = timbreReferenceMissing(submission);
    if (missing) {
      return { ok: false, status: 400, body: { error: `Timbre reference file is missing (${missing}). The upload it referred to is gone; re-upload it or clear the reference.` } };
    }
  }

  const prepared = withoutPrepareDocument(submission);
  if ('error' in prepared) return { ok: false, status: 409, body: { error: prepared.error } };
  submission = prepared.submission;

  const jobId = uuidv4();
  const enqueuedAt = Date.now();
  let envelope: ReturnType<typeof buildEnvelope>;
  try {
    envelope = buildEnvelope(submission, userId, jobId, enqueuedAt);
  } catch (err) {
    const clientError = err instanceof GenerationEnvelopeError && err.code !== 'unknown_backend';
    return { ok: false, status: clientError ? 400 : 500, body: { error: err instanceof Error ? err.message : 'Cannot resolve generation request' } };
  }
  const job: GenerationJob = {
    id: jobId,
    userId,
    envelope,
    status: 'pending',
    stage: 'Queued',
    progress: 0,
    params: structuredClone(submission),
    attempts: [],
    createdAt: enqueuedAt,
  };

  if (envelope.submittedBackendMismatch !== undefined) {
    console.log(`[Generate] Job ${job.id} submitted for '${envelope.submittedBackendMismatch}' but runs on '${envelope.backendId}'`);
  }

  jobs.set(job.id, job);
  withLedger('open', ledger => ledger.opened(job));

  // Enqueue — runs immediately if nothing else is generating,
  // otherwise waits until the current job finishes.
  enqueueGeneration(job);

  return { ok: true, jobId: job.id, status: job.status };
}

/** True while a queued or running job will render build session `id` (Save song reads its files when it runs). */
export function buildSessionHasActiveJob(id: string): boolean {
  return [...jobs.values()].some(j => isActiveJob(j) && (j.params?.yue2Build as { session?: unknown } | undefined)?.session === id);
}

/** The in-memory job, while it is still held (terminal jobs go after an hour). */
export function getGenerationJob(id: string): GenerationJob | undefined {
  return jobs.get(id);
}

router.post('/', generateCapture, async (req, res) => {
  const result = await submitGeneration(req.body, getUserId(req));
  if (!result.ok) { res.status(result.status).json(result.body); return; }
  res.json({ jobId: result.jobId, status: result.status });
});

// GET /api/generate/status/:id — poll job status
router.get('/status/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) {
    // Gone from memory: a restart, or the one-hour prune. The ledger says how it ended.
    let entry: LedgerEntry | null = null;
    withLedger('read', ledger => { entry = ledger.get(req.params.id); });
    if (entry) { res.json(ledgerStatusBody(entry)); return; }
    res.status(404).json({ error: 'Job not found' });
    return;
  }

  res.json({
    jobId: job.id,
    status: job.status,
    stage: job.stage,
    progress: job.progress,
    result: job.result,
    error: job.error,
    attempts: job.attempts,
    // Engine worker the job runs on (multi-GPU); null until one takes it.
    workerId: job.workerId ?? null,
    ace_job_id: job.aceJobId ?? null,
    ace_phase: job.acePhase ?? null,
    ace_phase_progress: job.acePhaseProgress ?? null,
    // YuE2 queue coalescing: the jobs rendered together in one engine batch,
    // so the queue can box them and show one time for the lot.
    batch: job.coalescedMembers
      ? { lead: job.id, members: job.coalescedMembers }
      : job.coalescedInto
        ? { lead: job.coalescedInto, members: jobs.get(job.coalescedInto)?.coalescedMembers ?? [job.coalescedInto, job.id] }
        : null,
    // MM3 live-audio stream: GET /api/generate/mm3/stream/:id is worth opening
    // only once this is true. Absent/false on every other backend and on every
    // render that did not ask for it.
    mm3_streaming: job.mm3Streaming === true,
    mm3_interleaved: job.mm3Interleaved ?? null,
    mm3_duration: job.mm3Duration ?? null,
    // How many songs this render is producing, and each one's seed. Sent from
    // the moment the engine accepts the job so the grid can stand up one card
    // per take immediately — waiting for audio would put them all on screen at
    // the end, which is the one thing streaming exists to avoid.
    mm3_takes: job.mm3Takes ?? 1,
    mm3_take_seeds: job.mm3TakeSeeds ?? null,
    // Natural-ending arbitration outcome. Null until the render is done (the
    // count is not knowable before then — see the note in
    // backends/minimax/generate.ts), and null entirely when the toggle was off.
    mm3_ending: job.mm3Ending ?? null,
  });
});

// GET /api/generate/mm3/stream/:id — the live audio of a running MM3 render.
//
// A PIPE, not a handler: the engine already produces exactly the byte stream
// the browser wants (concatenated self-contained WAVs, chunked), so this reads
// the response body and writes it straight out. Buffering it — even into a
// single Buffer before sending — would undo the whole feature, so nothing here
// may await the end of the body.
//
// Keyed on the NODE job id, not the engine's, so the browser never has to learn
// engine ids and a cancel through the existing /cancel/:id keeps working.
router.get('/mm3/stream/:id', async (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) { res.status(404).json({ error: 'Job not found' }); return; }
  if (!job.aceJobId) {
    // Submitted but not yet handed to the engine. A 409 rather than a wait:
    // the client polls status and reopens, and holding the socket here would
    // pin a connection for an unbounded queue wait.
    res.status(409).json({ error: 'This job has not reached the engine yet' });
    return;
  }
  if (job.mm3Streaming !== true) {
    res.status(409).json({ error: 'This job is not streaming' });
    return;
  }

  const upstream = new AbortController();
  // Browser tab closed / player stopped — drop the engine connection too, so
  // the engine stops holding chunks for a reader that has gone away. Watch res
  // (not req): req emits 'close' as soon as its body is consumed.
  res.on('close', () => { if (!res.writableEnded) upstream.abort(); });

  // `?take=N` selects which song of an ensemble render to stream. The takes
  // decode in lockstep and have their own queues in the engine, so N of these
  // can be open at once and all advance together. Omitted means take 0, which
  // is what a one-take render has always served.
  const take = Math.max(0, Number(req.query.take) || 0);
  let session: StreamSession | null = null;

  try {
    // This request runs outside the job's lease: address the job's own worker.
    const streamUrl = inWorkerScope(workerForJob(job), () => mm3StreamUrl(job.aceJobId!));
    const engUrl = take > 0
      ? `${streamUrl}${streamUrl.includes('?') ? '&' : '?'}take=${take}`
      : streamUrl;
    const eng = await fetch(engUrl, { signal: upstream.signal });
    if (!eng.ok || !eng.body) {
      const msg = await eng.text().catch(() => '');
      res.status(eng.status === 409 ? 409 : 502).type('application/json').send(msg || JSON.stringify({ error: 'stream unavailable' }));
      return;
    }
    res.setHeader('Content-Type', 'audio/wav');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('X-Accel-Buffering', 'no');
    // Node's copy for analysis and explicit recording (services/streamSessions).
    // It sees only what this pipe forwards and can never hold the pipe up.
    session = openStreamSession('mm3', { jobId: job.id, take });
    if (session) res.setHeader('X-Stream-Session', session.id);

    const reader = eng.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      const forwarded = res.write(Buffer.from(value));
      session?.feed(value);
      if (!forwarded) {
        // Respect backpressure: a browser that is not draining fast enough
        // must slow the pipe rather than grow an unbounded Node-side buffer.
        await new Promise<void>(resolve => res.once('drain', () => resolve()));
      }
    }
    res.end();
  } catch (err: any) {
    if (err?.name === 'AbortError') { try { res.end(); } catch {} return; }
    console.warn(`[MM3 Stream] job ${job.id}: ${err?.message ?? err}`);
    if (!res.headersSent) res.status(502).json({ error: String(err?.message ?? err) });
    else { try { res.end(); } catch {} }
  } finally {
    session?.end();
  }
});

// POST /api/generate/cancel/:id — cancel a running job
/** Cancel one job; false when the id is unknown. Shared with the audio queue. */
export function cancelGenerationJob(id: string): boolean {
  const job = jobs.get(id);
  if (!job) return false;
  markGenerationJobCancelled(job);
  // A cancel arrives outside the job's lease: send it to the job's own worker.
  inWorkerScope(workerForJob(job), () => {
    // One song of a running YuE2 batch: the engine drops it, the rest carry on.
    if (dropYue2BatchMember(job)) return;
    if (job.aceJobId) {
      aceClient.cancelJob(job.aceJobId).catch(() => {});
    }
    if ((job as any)._abort) {
      (job as any)._abort.abort();
    }
  });
  return true;
}

router.post('/cancel/:id', (req, res) => {
  if (!cancelGenerationJob(req.params.id)) { res.status(404).json({ error: 'Job not found' }); return; }
  res.json({ success: true, jobId: req.params.id });
});

// POST /api/generate/cancel-all — cancel all running jobs
router.post('/cancel-all', (req, res) => {
  let cancelled = 0;
  for (const [, job] of jobs) {
    if (isActiveJob(job)) {
      markGenerationJobCancelled(job);
      if (job.aceJobId) {
        const aceJobId = job.aceJobId;
        inWorkerScope(workerForJob(job), () => aceClient.cancelJob(aceJobId)).catch(() => {});
      }
      if ((job as any)._abort) {
        (job as any)._abort.abort();
      }
      cancelled++;
    }
  }
  res.json({ success: true, cancelled });
});

// GET /api/generate/queue — queue health / status inspection
router.get('/queue', (_req, res) => {
  const activeJob = Array.from(jobs.values()).find(j => isActiveJob(j) && j.status !== 'pending');

  // Count all non-terminal jobs in the jobs Map (includes pending jobs waiting in queue)
  const depth = Array.from(jobs.values()).filter(isActiveJob).length;

  res.json({
    depth,
    running: gpuLaneBusy(),
    owner: gpuLaneOwner()?.label ?? null,
    draining: gpuLaneOwner()?.draining ?? false,
    current: activeJob ? {
      id: activeJob.id,
      status: activeJob.status,
      stage: activeJob.stage,
      progress: activeJob.progress,
      age: Math.round((Date.now() - activeJob.createdAt) / 1000),
      aceJobId: activeJob.aceJobId,
    } : null,
    pending: gpuLaneDepth(),
  });
});

// POST /api/generate/reset-queue — force-reset: cancel everything, drain queue
router.post('/reset-queue', (_req, res) => {
  let cancelled = 0;

  // Cancel all non-terminal jobs in the jobs Map
  for (const [, job] of jobs) {
    if (isActiveJob(job)) {
      job.status = 'failed';
      job.error = 'Queue reset by user';
      job.stage = 'Reset';
      withLedger('finish', ledger => ledger.finished(job));
      if (job.aceJobId) {
        const aceJobId = job.aceJobId;
        inWorkerScope(workerForJob(job), () => aceClient.cancelJob(aceJobId)).catch(() => {});
      }
      if ((job as any)._abort) {
        (job as any)._abort.abort();
      }
      cancelled++;
    }
  }

  // Drain the pending execution queue
  const drained = resetGpuLane();

  console.log(`[Generate] Queue reset: ${cancelled} job(s) cancelled, ${drained} pending drained`);

  res.json({
    success: true,
    cancelled,
    drained,
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// STORM streaming (MDMAchine / A&E Concepts) — continuous slot generation.
//
// POST /stream opens one long chunked audio/wav response and generates synth
// slots back-to-back (complete WAV files concatenated; the client splits on
// RIFF headers). Slots go straight to the engine via submitSynth — no LM
// phase, no SQLite row, no entry in the normal job queue. Live parameter
// changes arrive via POST /control keyed by streamId (DJ mode runs two
// parallel streams, one per deck).
// ═══════════════════════════════════════════════════════════════════════════

interface StreamControlState {
  guidanceScale: number;
  lssStrength: number;
  inferenceSteps: number;
  duration: number;
  bpm: number;
  seedLock: boolean;
  nextSeed: number | null;
  nextPrompt: string | null;
  nextLyrics: string | null;
  nextBpm: number | null;
  nextDuration: number | null;
  baseCaption: string | null;
  baseLyrics: string | null;
  baseInstrumental: boolean | null;
  streamPaused: boolean;
  extraPluginParams: Record<string, number | string>;
  // Live overrides — null = keep using the stream-start snapshot value
  inferMethod: string | null;
  scheduler: string | null;
  guidanceMode: string | null;
  /** Active drift runtime, set at stream start when baseParams.drift is present. */
  drift?: DriftRuntime;
  /** baseSeed resolved at stream start; used by /storm/control pre-stream drift seeding. */
  baseSeed?: number;
  /** True when ctrl.drift was set by /storm/control while no stream was live. */
  driftPending?: boolean;
}

const streamRunning = new Map<string, boolean>();
registerBusyCheck('storm', () => [...streamRunning.values()].some(Boolean) ? ['a Storm stream is running'] : []);
const streamControl = new Map<string, StreamControlState>();
/** In-flight engine job per stream so stop/disconnect can cancel promptly. */
const streamAceJob = new Map<string, string>();

function getStreamControl(streamId: string): StreamControlState {
  let ctrl = streamControl.get(streamId);
  if (!ctrl) {
    ctrl = {
      guidanceScale: 7.0, lssStrength: 0, inferenceSteps: 0, duration: 0, bpm: 0,
      seedLock: false, nextSeed: null, nextPrompt: null, nextLyrics: null,
      nextBpm: null, nextDuration: null, baseCaption: null, baseLyrics: null,
      baseInstrumental: null, streamPaused: false, extraPluginParams: {},
      inferMethod: null, scheduler: null, guidanceMode: null,
    };
    streamControl.set(streamId, ctrl);
  }
  return ctrl;
}

function stopStream(streamId: string): void {
  const wasRunning = streamRunning.get(streamId) === true;
  streamRunning.set(streamId, false);
  // Drop the live drift runtime so it cannot leak into the next stream.
  // A pending (idle-time) drift is left alone.
  const c = streamControl.get(streamId);
  if (c && wasRunning && !c.driftPending) c.drift = undefined;
  const aceJobId = streamAceJob.get(streamId);
  if (aceJobId) {
    aceClient.cancelJob(aceJobId).catch(() => {});
    streamAceJob.delete(streamId);
  }
}

/** Per-slot render watchdog — a slot stuck past this is cancelled. */
const STREAM_SLOT_TIMEOUT_MS = 30 * 60 * 1000;

// POST /api/generate/storm/stream — start a continuous stream
router.post('/storm/stream', async (req, res) => {
  if (!engineReady) {
    res.status(503).json({ error: `Engine not ready: ${engineBootStatus}` });
    return;
  }
  // The stream drives the engine directly; refuse while queued jobs are active
  // so it doesn't interleave with (and stall) normal library generations.
  const activeJobs = [...jobs.values()].filter(isActiveJob);
  if (activeJobs.length > 0) {
    res.status(409).json({ error: `${activeJobs.length} generation job(s) active — wait for the queue to drain before streaming` });
    return;
  }

  const baseParams = req.body as Record<string, unknown>;
  const streamId = typeof baseParams.streamId === 'string' ? baseParams.streamId : 'default';

  streamRunning.set(streamId, true);
  const ctrl = getStreamControl(streamId);
  // Reset one-shot fields at stream start (sticky base fields persist across restarts)
  ctrl.nextPrompt = null;
  ctrl.nextLyrics = null;
  ctrl.streamPaused = false;

  res.setHeader('Content-Type', 'audio/wav');
  res.setHeader('Transfer-Encoding', 'chunked');
  // Node's copy of each slot for analysis and explicit recording (services/streamSessions).
  const session = openStreamSession('storm', { streamId });
  if (session) res.setHeader('X-Stream-Session', session.id);
  // Ends the session on every exit, including a throw before the slot loop's finally.
  res.on('close', () => session?.end());

  // Browser tab closed / fetch aborted without POST /stop — stop generating.
  // Must watch res, not req: on Node >=16 the request stream emits 'close' as
  // soon as its body is consumed (express.json already did), which would kill
  // every stream instantly. res emits 'close' when the connection terminates.
  res.on('close', () => {
    if (streamRunning.get(streamId) && !res.writableFinished) {
      console.log(`[STORM ${streamId}] client disconnected — stopping stream`);
      stopStream(streamId);
    }
  });

  const baseSeed = typeof baseParams.seed === 'number'
    ? baseParams.seed
    : Math.floor(Math.random() * 2147483647);
  // "Keep DiT & VAE loaded" setting — same per-request ?keep_loaded=1 the
  // normal synth phase sends; without it the engine evicts between slots.
  const coResident = baseParams.coResident === true;
  // Snapshot the active backend at stream start. Switching backends mid-stream
  // is not supported; a new stream must be started after a backend switch.
  const streamBackend = getActiveBackendId();
  let slotIdx = 0;

  // Record the resolved baseSeed so pre-stream /storm/control calls can use it.
  ctrl.baseSeed = baseSeed;

  // Resolve the drift runtime. A pre-stream /storm/control drift is kept only
  // when it is flagged pending; leftovers from a finished stream are dropped.
  ctrl.drift = resolveStartDrift(ctrl.drift, ctrl.driftPending === true, baseParams.drift, baseSeed);
  ctrl.driftPending = false;

  console.log(`[STORM ${streamId}] stream started (baseSeed=${baseSeed}, backend=${streamBackend})`);
  pushLog(`[STORM ${streamId}] stream started`);

  try {
    while (streamRunning.get(streamId)) {
      // Max-buffer pause gate: the client sends stream_pause=true when its
      // playback buffer is full; we idle here (GPU free) until it drains.
      while (ctrl.streamPaused && streamRunning.get(streamId)) {
        await new Promise(r => setTimeout(r, 500));
      }
      if (!streamRunning.get(streamId)) break;

      // Log one-shot fields before buildSlotRequest reads them (they are
      // cleared below after the helper returns).
      if (ctrl.nextPrompt !== null) {
        console.log(`[STORM ${streamId}] slot ${slotIdx}: applying queued prompt`);
      }
      if (ctrl.nextLyrics !== null) {
        console.log(`[STORM ${streamId}] slot ${slotIdx}: lyrics applied (one-shot), instrumental off`);
      }

      // Build the per-backend wire request via the shared slot helper.
      // The helper reads ctrl but does not mutate it.
      const slotResult = buildSlotRequest(baseParams, ctrl, slotIdx, baseSeed, streamBackend);

      // Consume one-shot fields (same clearing the inline code did before).
      if (ctrl.nextPrompt !== null) ctrl.nextPrompt = null;
      if (ctrl.nextLyrics !== null) {
        ctrl.baseInstrumental = false;
        ctrl.nextLyrics = null;
      }
      if (ctrl.nextSeed !== null) ctrl.nextSeed = null;
      if (ctrl.nextBpm !== null) ctrl.nextBpm = null;
      if (ctrl.nextDuration !== null) ctrl.nextDuration = null;

      // Resolves only once the engine is done with the slot, so the lane is
      // held through it. Stop/disconnect cancels an ACE/YuE2 job and waits for
      // its terminal state; an SA3 slot (one synchronous fetch, no mid-render
      // cancel) finishes. Either way the output is discarded.
      const renderSlot = async (): Promise<Buffer | null> => {
        if (slotResult.kind === 'sa3') {
          const { ids, nTokens } = await tokenizeForSa3(slotResult.caption);
          return aceClient.submitSa3Synth({
            tokens: ids,
            nTokens,
            durationSec: slotResult.durationSec,
            steps: slotResult.steps,
            sampler: slotResult.sampler,
            seed: slotResult.seed,
            ...(slotResult.seedB !== undefined ? { seedB: slotResult.seedB, seedBlend: slotResult.seedBlend } : {}),
            adapters: slotResult.adapters,
            outFmt: 'f32',
            ...(slotResult.solver        ? { solver:        slotResult.solver }        : {}),
            ...(slotResult.scheduler     ? { scheduler:     slotResult.scheduler }     : {}),
            ...(slotResult.guidanceMode  ? { guidanceMode:  slotResult.guidanceMode }  : {}),
            ...(slotResult.guidanceScale !== undefined ? { guidanceScale: slotResult.guidanceScale } : {}),
            ...(slotResult.pluginParams  ? { pluginParams:  slotResult.pluginParams }  : {}),
            ...(slotResult.postprocessPlugin ? { postprocessPlugin: slotResult.postprocessPlugin } : {}),
          });
        }

        // Submit to the right backend and get a shared-job id.
        const synthJobId = slotResult.kind === 'yue2'
          ? (await yue2Synth(slotResult.yue2Req)).job_id
          : await aceClient.submitSynth(slotResult.aceReq, 'wav32', coResident);

        // Stop/disconnect cancels mid-render; the wait still lasts until the
        // engine settles (poll errors are retried), so the lane is held.
        streamAceJob.set(streamId, synthJobId);
        let end: SlotJobEnd;
        try {
          end = await waitForSlotJob({
            poll: () => aceClient.pollJob(synthJobId),
            cancel: () => aceClient.cancelJob(synthJobId),
            isRunning: () => streamRunning.get(streamId) === true,
            onTimeout: () => streamRunning.set(streamId, false),
            engineCrashSeq,
            timeoutMs: STREAM_SLOT_TIMEOUT_MS,
            log: msg => console.warn(`[STORM ${streamId}] slot ${slotIdx}: ${msg}`),
          });
        } finally {
          streamAceJob.delete(streamId);
        }
        if (end !== 'done') {
          console.warn(`[STORM ${streamId}] slot ${slotIdx}: ${end}`);
          return null;
        }
        if (!streamRunning.get(streamId)) return null;

        const audioRes = await aceClient.getJobResult(synthJobId);
        if (!audioRes.ok) {
          console.warn(`[STORM ${streamId}] slot ${slotIdx}: result fetch failed (${audioRes.status})`);
          return null;
        }
        const audioBody = Buffer.from(await audioRes.arrayBuffer());
        // YuE2 may return multipart/mixed (one WAV per track); take the first
        // part. ACE always returns a single WAV so the same path works for both.
        const contentType = audioRes.headers.get('content-type') || 'audio/wav';
        return contentType.startsWith('multipart/mixed')
          ? (splitMultipartMixed(audioBody, contentType)[0] ?? audioBody)
          : audioBody;
      };

      const wavBuf = await renderSlotOnLane({
        label: `storm ${streamId} slot ${slotIdx}`,
        family: streamBackend,
        isRunning: () => streamRunning.get(streamId) === true,
        evict: () => evictOtherFamilies(streamBackend),
        render: renderSlot,
      });
      if (!wavBuf) break;
      res.write(wavBuf);
      session?.chunk(wavBuf);
      slotIdx++;
    }
  } catch (err) {
    console.error(`[STORM stream ${streamId}] error:`, err);
  } finally {
    stopStream(streamId);
    // Keep a drift that a POST /storm/control set as pending after the stream
    // ended; everything else (including live drift) is discarded.
    const endCtrl = streamControl.get(streamId);
    const pendingDrift = endCtrl?.driftPending ? endCtrl.drift : undefined;
    streamControl.delete(streamId);
    session?.end();
    if (pendingDrift) {
      const fresh = getStreamControl(streamId);
      fresh.drift = pendingDrift;
      fresh.driftPending = true;
    }
    console.log(`[STORM ${streamId}] stream ended after ${slotIdx} slot(s)`);
    pushLog(`[STORM ${streamId}] stream ended after ${slotIdx} slot(s)`);
    res.end();
  }
});

// POST /api/generate/storm/stop — stop a stream. No further slot starts. An
// in-flight ACE/YuE2 slot is cancelled; an in-flight SA3 slot runs to the end
// (no mid-render cancel). Its output is discarded either way, and the GPU lane
// stays held until the engine has finished.
router.post('/storm/stop', (req, res) => {
  const b = req.body as Record<string, unknown>;
  const streamId = typeof b.streamId === 'string' ? b.streamId : 'default';
  stopStream(streamId);
  res.json({ ok: true });
});

// POST /api/generate/storm/control — live-mutate a stream's next-slot params
router.post('/storm/control', (req, res) => {
  const b = req.body as Record<string, unknown>;
  const streamId = typeof b.streamId === 'string' ? b.streamId : 'default';
  const ctrl = getStreamControl(streamId);

  if (typeof b.guidance_scale === 'number') ctrl.guidanceScale = b.guidance_scale;
  if (typeof b.lss_strength === 'number') ctrl.lssStrength = b.lss_strength;
  if (typeof b.inference_steps === 'number') ctrl.inferenceSteps = b.inference_steps;
  if (typeof b.duration === 'number') ctrl.duration = b.duration;
  if (typeof b.bpm === 'number') ctrl.bpm = b.bpm;
  if (typeof b.next_bpm === 'number') ctrl.nextBpm = b.next_bpm;
  if (typeof b.next_duration === 'number') ctrl.nextDuration = b.next_duration;
  if (typeof b.seed_lock === 'boolean') ctrl.seedLock = b.seed_lock;
  if (typeof b.seed === 'number') ctrl.nextSeed = b.seed;
  if (typeof b.prompt === 'string') ctrl.nextPrompt = b.prompt;
  if (typeof b.lyrics === 'string') ctrl.nextLyrics = b.lyrics;
  if (typeof b.stick_prompt === 'string') ctrl.baseCaption = b.stick_prompt;
  if (b.stick_prompt === null) ctrl.baseCaption = null;
  if (typeof b.stick_lyrics === 'string') { ctrl.baseLyrics = b.stick_lyrics; ctrl.baseInstrumental = false; }
  if (b.stick_lyrics === null) ctrl.baseLyrics = null;
  if (typeof b.stream_pause === 'boolean') ctrl.streamPaused = b.stream_pause;
  if (typeof b.infer_method === 'string') ctrl.inferMethod = b.infer_method;
  if (typeof b.scheduler === 'string') ctrl.scheduler = b.scheduler;
  if (typeof b.guidance_mode === 'string') ctrl.guidanceMode = b.guidance_mode;
  // Generic plugin params — numeric and string values (e.g. rk_order: 'auto')
  if (b.plugin_params && typeof b.plugin_params === 'object') {
    for (const [k, v] of Object.entries(b.plugin_params as Record<string, unknown>)) {
      if (typeof v === 'number' || typeof v === 'string') ctrl.extraPluginParams[k] = v;
    }
  }
  // Drift pressure patch -- ignore bad fields, never throw.
  if (b.drift && typeof b.drift === 'object' && !Array.isArray(b.drift)) {
    const live = streamRunning.get(streamId) === true;
    if (live && ctrl.drift) {
      // Stream is live: merge into existing runtime.
      updateDriftPressure(ctrl.drift, b.drift as Partial<DriftPressure>);
    } else if (!live && ctrl.drift && ctrl.driftPending) {
      // Already pending: merge further idle-time patches.
      updateDriftPressure(ctrl.drift, b.drift as Partial<DriftPressure>);
    } else {
      // Fresh runtime. When idle it is stored as pending; state is re-seeded
      // with the real baseSeed at stream start (resolveStartDrift).
      const seed = ctrl.baseSeed ?? Math.floor(Math.random() * 2147483647);
      const driftRuntime: DriftRuntime = { state: initDrift(seed), pressure: defaultDriftPressure() };
      updateDriftPressure(driftRuntime, b.drift as Partial<DriftPressure>);
      ctrl.drift = driftRuntime;
      ctrl.driftPending = !live;
    }
  }

  res.json({ ok: true, streamId });
});

// GET /api/generate/storm/control — read back a stream's live state
router.get('/storm/control', (req, res) => {
  const streamId = typeof req.query.streamId === 'string' ? req.query.streamId : 'default';
  const ctrl = getStreamControl(streamId);
  const { drift: _drift, ...ctrlWithoutDrift } = ctrl;
  res.json({
    running: streamRunning.get(streamId) ?? false,
    ...ctrlWithoutDrift,
    ...(ctrl.drift ? { drift: driftSnapshot(ctrl.drift) } : {}),
  });
});

export default router;
