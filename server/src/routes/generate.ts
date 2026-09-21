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
import { isEngineSuspended } from '../services/aceEngineProcess.js';
import { pushLog } from './logs.js';
import { getBackend, getActiveBackendId } from '../services/backends/registry.js';
import { mm3StreamUrl } from '../services/backends/minimax/client.js';
import { runOnGpuLane, gpuLaneBusy, gpuLaneDepth, gpuLaneOwner, resetGpuLane, type LaneLease } from '../services/generation/gpuLane.js';
import { isActiveJob, type GenerationJob } from '../services/generation/jobTypes.js';
import { pollUntilDone } from '../services/generation/pollUntilDone.js';
import { translateParams } from '../services/generation/translateParams.js';
import { buildEnvelope, GenerationEnvelopeError } from '../services/generation/envelope.js';
import { noteEnqueued, noteFinished } from '../services/generation/residency.js';
import { runYue2PlanPreview } from '../services/backends/yue2/generate.js';
import type {
  GenerationAttempt,
  GenerationEndReason,
  GenerationOutcome,
} from '../services/backends/types.js';

export type { GenerationJob, StageTiming } from '../services/generation/jobTypes.js';

const router = Router();

const jobs = new Map<string, GenerationJob>();

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
  return backend.generate(job, {
    envelope: job.envelope,
    attempt,
    lease,
    signal,
    pollUntilDone,
    hooks: { onEngineJob() {}, onStage() {}, onArtifact() {} },
  });
}
// ── Async generation queue ────────────────────────────────────────────
// Generations run one at a time on the shared GPU lane (services/generation/
// gpuLane.ts). The C++ engine is single-GPU, and concurrent runGeneration
// calls also leak progress between jobs because subscribeLines() is a global
// pub/sub with no job tagging. The lane is shared rather than private because
// post-processing re-runs are just as GPU-hungry and must not race a render.
function enqueueGeneration(job: GenerationJob): void {
  const family = job.envelope.backendId;
  noteEnqueued(family);
  if (gpuLaneBusy()) {
    console.log(`[Generate] Job ${job.id} queued (${gpuLaneDepth() + 1} waiting)`);
  }

  void runOnGpuLane(async (lease) => {
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
    // means the lane itself broke. Never leave that silent.
    console.error(`[Generate] Job ${job.id} lane error:`, err?.message || err);
  }).finally(() => noteFinished(family));
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
    const preview = await runOnGpuLane(() => runYue2PlanPreview(params, abort.signal), { label: 'yue2 score preview', family: 'yue2' });
    res.json(preview);
  } catch (err) {
    res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
  } finally {
    noteFinished('yue2');
  }
});

router.post('/', (req, res) => {
  // The engine is deliberately stopped while a training preprocess job owns the
  // GPU — say so instead of the generic "not ready" boot message.
  if (isEngineSuspended()) {
    res.status(503).json({ error: 'Engine is paused for training preprocessing — try again when the job finishes' });
    return;
  }

  // Reject requests while engine is still bootstrapping (downloading DLLs, etc.)
  if (!engineReady) {
    res.status(503).json({
      error: `Engine not ready: ${engineBootStatus}`,
      detail: 'The CUDA runtime is still being set up. Please wait a moment and try again.',
    });
    return;
  }

  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const jobId = uuidv4();
  const enqueuedAt = Date.now();
  let envelope: ReturnType<typeof buildEnvelope>;
  try {
    envelope = buildEnvelope(req.body, userId, jobId, enqueuedAt);
  } catch (err) {
    const clientError = err instanceof GenerationEnvelopeError && err.code !== 'unknown_backend';
    res.status(clientError ? 400 : 500).json({ error: err instanceof Error ? err.message : 'Cannot resolve generation request' });
    return;
  }
  const job: GenerationJob = {
    id: jobId,
    userId,
    envelope,
    status: 'pending',
    stage: 'Queued',
    progress: 0,
    params: structuredClone(req.body),
    attempts: [],
    createdAt: enqueuedAt,
  };

  if (envelope.submittedBackendMismatch !== undefined) {
    console.log(`[Generate] Job ${job.id} submitted for '${envelope.submittedBackendMismatch}' but runs on '${envelope.backendId}'`);
  }

  jobs.set(job.id, job);

  // Enqueue — runs immediately if nothing else is generating,
  // otherwise waits until the current job finishes.
  enqueueGeneration(job);

  res.json({
    jobId: job.id,
    status: job.status,
  });
});

// GET /api/generate/status/:id — poll job status
router.get('/status/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) { res.status(404).json({ error: 'Job not found' }); return; }

  res.json({
    jobId: job.id,
    status: job.status,
    stage: job.stage,
    progress: job.progress,
    result: job.result,
    error: job.error,
    attempts: job.attempts,
    ace_job_id: job.aceJobId ?? null,
    ace_phase: job.acePhase ?? null,
    ace_phase_progress: job.acePhaseProgress ?? null,
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

  try {
    const engUrl = take > 0
      ? `${mm3StreamUrl(job.aceJobId)}${mm3StreamUrl(job.aceJobId).includes('?') ? '&' : '?'}take=${take}`
      : mm3StreamUrl(job.aceJobId);
    const eng = await fetch(engUrl, { signal: upstream.signal });
    if (!eng.ok || !eng.body) {
      const msg = await eng.text().catch(() => '');
      res.status(eng.status === 409 ? 409 : 502).type('application/json').send(msg || JSON.stringify({ error: 'stream unavailable' }));
      return;
    }
    res.setHeader('Content-Type', 'audio/wav');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('X-Accel-Buffering', 'no');

    const reader = eng.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!res.write(Buffer.from(value))) {
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
  }
});

// POST /api/generate/cancel/:id — cancel a running job
router.post('/cancel/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) { res.status(404).json({ error: 'Job not found' }); return; }

  job.status = 'cancelled';
  if (job.aceJobId) {
    aceClient.cancelJob(job.aceJobId).catch(() => {});
  }
  if ((job as any)._abort) {
    (job as any)._abort.abort();
  }

  res.json({ success: true, jobId: job.id });
});

// POST /api/generate/cancel-all — cancel all running jobs
router.post('/cancel-all', (req, res) => {
  let cancelled = 0;
  for (const [, job] of jobs) {
    if (isActiveJob(job)) {
      job.status = 'cancelled';
      if (job.aceJobId) {
        aceClient.cancelJob(job.aceJobId).catch(() => {});
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
      if (job.aceJobId) {
        aceClient.cancelJob(job.aceJobId).catch(() => {});
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

// GET /api/generate/stream/:id — SSE endpoint for streaming preview audio
// Frontend connects via EventSource. Receives:
//   event: status   — job status/stage/progress updates
//   event: preview  — new preview WAV file available for playback
//   event: done     — generation complete, final audio URL
//   event: error    — generation failed
router.get('/stream/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) { res.status(404).json({ error: 'Job not found' }); return; }

  // SSE headers
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');  // nginx compatibility
  res.flushHeaders();

  let lastPreviewIdx = 0;
  let lastStage = '';
  let lastProgress = -1;
  let closed = false;

  const sendSSE = (event: string, data: unknown) => {
    if (closed) return;
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  // Poll interval — check for new previews and status changes
  const interval = setInterval(() => {
    if (closed) return;

    // Send status updates when changed
    if (job.stage !== lastStage || job.progress !== lastProgress) {
      lastStage = job.stage || '';
      lastProgress = job.progress || 0;
      sendSSE('status', {
        status: job.status,
        stage: job.stage,
        progress: job.progress,
      });
    }

    // Send new preview events
    const previews = job.streamPreviews;
    if (previews && previews.length > lastPreviewIdx) {
      for (let i = lastPreviewIdx; i < previews.length; i++) {
        // Convert filesystem path to a URL relative to /audio/
        // Preview WAVs live in data/audio/stream/ → served at /audio/stream/
        const p = previews[i];
        const audioDir = config.data.audioDir.replace(/\\/g, '/');
        const filePath = (p.path || '').replace(/\\/g, '/');
        let url = p.path;
        if (filePath.startsWith(audioDir)) {
          url = '/audio' + filePath.substring(audioDir.length);
        } else {
          // Fallback: extract filename only
          const filename = path.basename(p.path);
          url = `/audio/stream/${filename}`;
        }
        sendSSE('preview', {
          url,
          step: p.step,
          totalSteps: p.totalSteps,
          slot: p.slot,
        });
      }
      lastPreviewIdx = previews.length;
    }

    // Terminal states — send final event and close
    if (job.status === 'succeeded') {
      sendSSE('done', {
        result: job.result,
      });
      cleanup();
    } else if (job.status === 'failed' || job.status === 'cancelled') {
      sendSSE('error', {
        status: job.status,
        error: job.error,
      });
      cleanup();
    }
  }, 250);  // 4Hz polling — fast enough for audio preview updates

  const cleanup = () => {
    closed = true;
    clearInterval(interval);
    res.end();
  };

  // Client disconnect
  req.on('close', cleanup);
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
}

const streamRunning = new Map<string, boolean>();
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
  streamRunning.set(streamId, false);
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
  let slotIdx = 0;

  console.log(`[STORM ${streamId}] stream started (baseSeed=${baseSeed})`);
  pushLog(`[STORM ${streamId}] stream started`);

  try {
    while (streamRunning.get(streamId)) {
      // Max-buffer pause gate: the client sends stream_pause=true when its
      // playback buffer is full; we idle here (GPU free) until it drains.
      while (ctrl.streamPaused && streamRunning.get(streamId)) {
        await new Promise(r => setTimeout(r, 500));
      }
      if (!streamRunning.get(streamId)) break;

      const mergedPluginParams = {
        ...(typeof baseParams.pluginParams === 'object' && baseParams.pluginParams !== null
          ? baseParams.pluginParams as Record<string, unknown> : {}),
        ...ctrl.extraPluginParams,
      };

      // NOTE: keys here must match what translateParams reads (camelCase for
      // UI-facing params). An explicit per-slot seed also requires
      // randomSeed=false or translateParams re-randomizes it.
      const slotReq: Record<string, unknown> = {
        ...baseParams,
        randomSeed: false,
        seed: ctrl.nextSeed !== null
          ? ctrl.nextSeed
          : (ctrl.seedLock ? baseSeed : baseSeed + slotIdx),
        guidanceScale: ctrl.guidanceScale,
        ...(ctrl.lssStrength > 0 ? { lssStrength: ctrl.lssStrength } : {}),
        ...(ctrl.inferenceSteps > 0 ? { inferenceSteps: ctrl.inferenceSteps } : {}),
        ...(ctrl.nextBpm !== null ? { bpm: ctrl.nextBpm } : ctrl.bpm > 0 ? { bpm: ctrl.bpm } : {}),
        ...(ctrl.nextDuration !== null ? { duration: ctrl.nextDuration } : ctrl.duration > 0 ? { duration: ctrl.duration } : {}),
        // Solver / scheduler / guidance live overrides — null keeps the snapshot value
        ...(ctrl.inferMethod ? { inferMethod: ctrl.inferMethod } : {}),
        ...(ctrl.scheduler ? { scheduler: ctrl.scheduler } : {}),
        ...(ctrl.guidanceMode ? { guidanceMode: ctrl.guidanceMode } : {}),
        ...(Object.keys(mergedPluginParams).length > 0 ? { pluginParams: mergedPluginParams } : {}),
      };

      // One-shot overrides (consumed after applying); Stick makes them base
      if (ctrl.nextPrompt !== null) {
        slotReq.caption = ctrl.nextPrompt;
        slotReq.prompt = ctrl.nextPrompt;
        ctrl.nextPrompt = null;
        console.log(`[STORM ${streamId}] slot ${slotIdx}: applying queued prompt`);
      } else if (ctrl.baseCaption !== null) {
        slotReq.caption = ctrl.baseCaption;
        slotReq.prompt = ctrl.baseCaption;
      }
      if (ctrl.nextLyrics !== null) {
        slotReq.lyrics = ctrl.nextLyrics;
        slotReq.instrumental = false;
        ctrl.baseInstrumental = false;
        ctrl.nextLyrics = null;
        console.log(`[STORM ${streamId}] slot ${slotIdx}: lyrics applied (one-shot), instrumental off`);
      } else if (ctrl.baseLyrics !== null) {
        slotReq.lyrics = ctrl.baseLyrics;
        slotReq.instrumental = false;
      } else if (ctrl.baseInstrumental === false) {
        slotReq.instrumental = false;
      }
      if (ctrl.nextSeed !== null) ctrl.nextSeed = null;
      if (ctrl.nextBpm !== null) ctrl.nextBpm = null;
      if (ctrl.nextDuration !== null) ctrl.nextDuration = null;

      const aceReq = translateParams(slotReq);

      // Inline poll — checks streamRunning every 300 ms so stop/disconnect
      // cancels mid-render instead of waiting out the slot.
      const synthJobId = await aceClient.submitSynth(aceReq, 'wav32', coResident);
      streamAceJob.set(streamId, synthJobId);
      const slotStart = Date.now();
      let jobDone = false;
      while (!jobDone) {
        if (!streamRunning.get(streamId)) {
          await aceClient.cancelJob(synthJobId).catch(() => {});
          console.log(`[STORM ${streamId}] slot ${slotIdx}: cancelled mid-render`);
          break;
        }
        if (Date.now() - slotStart > STREAM_SLOT_TIMEOUT_MS) {
          await aceClient.cancelJob(synthJobId).catch(() => {});
          console.warn(`[STORM ${streamId}] slot ${slotIdx}: render timeout — stopping stream`);
          streamRunning.set(streamId, false);
          break;
        }
        const status = await aceClient.pollJob(synthJobId);
        if (status.status === 'done') { jobDone = true; break; }
        if (status.status === 'failed' || status.status === 'cancelled') {
          console.warn(`[STORM ${streamId}] slot ${slotIdx}: ${status.status}`);
          break;
        }
        await new Promise(r => setTimeout(r, 300));
      }
      streamAceJob.delete(streamId);
      if (!jobDone || !streamRunning.get(streamId)) break;

      const audioRes = await aceClient.getJobResult(synthJobId);
      if (!audioRes.ok) {
        console.warn(`[STORM ${streamId}] slot ${slotIdx}: result fetch failed (${audioRes.status})`);
        break;
      }
      const wavBuf = Buffer.from(await audioRes.arrayBuffer());
      res.write(wavBuf);
      slotIdx++;
    }
  } catch (err) {
    console.error(`[STORM stream ${streamId}] error:`, err);
  } finally {
    stopStream(streamId);
    streamControl.delete(streamId);
    console.log(`[STORM ${streamId}] stream ended after ${slotIdx} slot(s)`);
    pushLog(`[STORM ${streamId}] stream ended after ${slotIdx} slot(s)`);
    res.end();
  }
});

// POST /api/generate/storm/stop — stop a stream (cancels the in-flight slot)
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

  res.json({ ok: true, streamId });
});

// GET /api/generate/storm/control — read back a stream's live state
router.get('/storm/control', (req, res) => {
  const streamId = typeof req.query.streamId === 'string' ? req.query.streamId : 'default';
  res.json({
    running: streamRunning.get(streamId) ?? false,
    ...getStreamControl(streamId),
  });
});

export default router;
