// inspire.ts — Inspire API endpoint
//
// Wraps the engine's LM inspire mode (POST /lm?mode=inspire) to generate
// lyrics + metadata without audio codes.  Used by Insta-Gen's "Preview
// Lyrics" flow so the user can review/edit before committing to full
// generation.
//
// Async job pattern mirrors generate.ts: submit → poll → result.

import { Router } from 'express';
import { v4 as uuidv4 } from 'uuid';
import { aceClient, type AceRequest } from '../services/aceClient.js';
import { getUserId } from './auth.js';
import { engineReady, engineBootStatus } from '../engineState.js';
import { subscribeLines } from './logs.js';
import { translateParams } from '../services/generation/translateParams.js';

const router = Router();

/** Inspire job state */
interface InspireJob {
  id: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';
  stage?: string;
  progress?: number;
  aceJobId?: string;
  result?: {
    caption: string;
    lyrics: string;
    bpm: number;
    duration: number;
    keyScale: string;
    timeSignature: string;
    vocalLanguage: string;
  };
  error?: string;
  createdAt: number;
}

const inspireJobs = new Map<string, InspireJob>();

// Cleanup old jobs after 10 minutes
setInterval(() => {
  const cutoff = Date.now() - 10 * 60 * 1000;
  for (const [id, job] of inspireJobs) {
    if (job.createdAt < cutoff && job.status !== 'running') {
      inspireJobs.delete(id);
    }
  }
}, 60_000);

/** Poll ace-server job until completion */
async function pollUntilDone(aceJobId: string, job: InspireJob, signal: AbortSignal): Promise<void> {
  const POLL_INTERVAL = 500;
  const MAX_POLLS = 600; // 5 minutes max (inspire is fast)

  for (let i = 0; i < MAX_POLLS; i++) {
    if (signal.aborted || job.status === 'cancelled') {
      await aceClient.cancelJob(aceJobId);
      throw new Error('Cancelled');
    }

    const status = await aceClient.pollJob(aceJobId);
    if (status.status === 'done') return;
    if (status.status === 'failed') throw new Error('Inspire failed on ace-server');
    if (status.status === 'cancelled') throw new Error('Cancelled by ace-server');

    await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
  }

  throw new Error('Inspire timed out');
}

/** Run inspire pipeline */
async function runInspire(job: InspireJob, params: any): Promise<void> {
  if (job.status === 'cancelled') return;

  const aceReq = translateParams(params);
  const abortController = new AbortController();
  (job as any)._abort = abortController;

  try {
    job.status = 'running';
    job.stage = 'Generating lyrics & metadata...';
    job.progress = 10;

    // Diagnostic logging — trace exactly what reaches the engine
    console.log(`[Inspire] Job ${job.id} — input params:`, JSON.stringify({
      caption: params.caption,
      subject: params.subject,
      vocalLanguage: params.vocalLanguage,
      useCotCaption: params.useCotCaption,
    }));
    console.log(`[Inspire] Job ${job.id} — aceReq:`, JSON.stringify({
      caption: aceReq.caption,
      vocal_language: aceReq.vocal_language,
      use_cot_caption: aceReq.use_cot_caption,
      lyrics: aceReq.lyrics || '(empty)',
      lm_temperature: aceReq.lm_temperature,
      lm_cfg_scale: aceReq.lm_cfg_scale,
      lm_top_p: aceReq.lm_top_p,
    }));

    // Subscribe to engine logs for LM Phase 1 progress
    const unsub = subscribeLines((line) => {
      if (line.source !== 'engine') return;
      const lm1 = line.text.match(/\[LM-Phase1\] Step (\d+).*?([\d.]+) tok\/s/);
      if (lm1) {
        job.stage = `Composing lyrics: Step ${lm1[1]} (${lm1[2]} tok/s)`;
        job.progress = 30;
        return;
      }
      if (line.text.includes('[LM-Phase1] Prefill')) {
        job.stage = 'Preparing language model...';
        job.progress = 15;
      } else if (line.text.includes('[Adapter]') && line.text.includes('Merge')) {
        job.stage = 'Loading adapter...';
      }
    });

    console.log(`[Inspire] Job ${job.id} — submitting LM inspire request`);

    const lmJobId = await aceClient.submitLm(aceReq, 'inspire');
    job.aceJobId = lmJobId;

    await pollUntilDone(lmJobId, job, abortController.signal);

    // Fetch inspire results
    const resultRes = await aceClient.getJobResult(lmJobId);
    const lmResults = await resultRes.json() as AceRequest[];

    unsub();

    if (!lmResults || lmResults.length === 0) {
      throw new Error('No results from inspire mode');
    }

    const first = lmResults[0];

    job.status = 'succeeded';
    job.progress = 100;
    job.stage = 'Done!';
    job.result = {
      caption: first.caption || aceReq.caption || '',
      lyrics: first.lyrics || '',
      bpm: first.bpm || 120,
      duration: first.duration || 120,
      keyScale: first.keyscale || 'C major',
      timeSignature: first.timesignature || '4',
      vocalLanguage: first.vocal_language || params.vocalLanguage || 'en',
    };

    console.log(`[Inspire] Job ${job.id} — complete. BPM=${job.result.bpm}, lang=${job.result.vocalLanguage}, caption=${job.result.caption.substring(0, 100)}, lyrics=${job.result.lyrics.substring(0, 200)}`);

  } catch (err: any) {
    if (err.message === 'Cancelled') {
      job.status = 'cancelled';
      job.stage = 'Cancelled';
    } else {
      job.status = 'failed';
      job.error = err.message || 'Unknown error';
      job.stage = 'Failed';
      console.error(`[Inspire] Job ${job.id} failed:`, err.message);
    }
  }
}

// ── Serialization queue (shares the engine with generate) ──
// Inspire jobs go through the same single-GPU bottleneck.
// For now we run them independently — they're fast and only use the LM.
// If contention becomes an issue, we can merge with the generate queue.

// POST /api/inspire — start an inspire job
router.post('/', (req, res) => {
  if (!engineReady) {
    res.status(503).json({
      error: `Engine not ready: ${engineBootStatus}`,
      detail: 'Please wait for the engine to finish starting up.',
    });
    return;
  }

  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const job: InspireJob = {
    id: uuidv4(),
    status: 'pending',
    stage: 'Starting...',
    progress: 0,
    createdAt: Date.now(),
  };

  inspireJobs.set(job.id, job);
  runInspire(job, req.body);

  res.json({
    jobId: job.id,
    status: job.status,
  });
});

// GET /api/inspire/status/:id — poll inspire job status
router.get('/status/:id', (req, res) => {
  const job = inspireJobs.get(req.params.id);
  if (!job) { res.status(404).json({ error: 'Job not found' }); return; }

  res.json({
    jobId: job.id,
    status: job.status,
    stage: job.stage,
    progress: job.progress,
    result: job.result,
    error: job.error,
  });
});

// POST /api/inspire/cancel/:id — cancel an inspire job
router.post('/cancel/:id', (req, res) => {
  const job = inspireJobs.get(req.params.id);
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

export default router;
