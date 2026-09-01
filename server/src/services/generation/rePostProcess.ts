// generation/rePostProcess.ts — run the PP chain on an already-rendered song
//
// The normal path post-processes a render as the generation finishes. When the
// global post-processing toggle was off at generation time the raw WAV is all
// that exists, and there was no way to change your mind afterwards. This runs
// the same chain over that raw WAV on demand.
//
// Two things make it safe to do after the fact:
//   - runPostProcessingChain() works on a COPY (<base>_mastered.wav) and never
//     touches the raw render, so a re-run cannot damage the original.
//   - a song only carries a mastered_audio_url when a stage actually ran, so
//     an empty column is a reliable "this has never been processed" signal.
//
// The second point is also the safety rule: PP is not idempotent — running a
// mastered file through mastering again overcooks it — so an already-processed
// song is refused rather than stacked.

import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { config } from '../../config.js';
import { getDb } from '../../db/database.js';
import { vstChainActive } from '../../routes/vst.js';
import {
  startGenerationLog, logGeneration, logGenerationParams,
  finishGenerationLog, failGenerationLog,
} from '../logger.js';
import { runOnGpuLane, gpuLaneBusy, gpuLaneDepth } from './gpuLane.js';
import { runPostProcessingChain, normalizePpParams, type PostProcessParams } from './postProcessing.js';

export interface RePostProcessJob {
  id: string;
  songId: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed';
  stage: string;
  error?: string;
  /** Set on success — the URL now stored in songs.mastered_audio_url. */
  masteredAudioUrl?: string;
  createdAt: number;
}

const jobs = new Map<string, RePostProcessJob>();

/** Songs with a re-run in flight, so a double-click cannot start two. */
const inFlight = new Set<string>();

// Terminal jobs are only kept long enough for the browser to poll the result.
const JOB_TTL_MS = 30 * 60 * 1000;
setInterval(() => {
  const now = Date.now();
  for (const [id, job] of jobs) {
    if ((job.status === 'succeeded' || job.status === 'failed') && now - job.createdAt > JOB_TTL_MS) {
      jobs.delete(id);
    }
  }
}, 5 * 60 * 1000).unref();

export function getRePostProcessJob(jobId: string): RePostProcessJob | undefined {
  return jobs.get(jobId);
}

/** The in-flight job for a song, if any. Lets the UI re-attach after a reload. */
export function findRePostProcessJobBySong(songId: string): RePostProcessJob | undefined {
  for (const job of jobs.values()) {
    if (job.songId === songId && (job.status === 'pending' || job.status === 'running')) return job;
  }
  return undefined;
}

/**
 * Names the stages this params object would actually run, in chain order.
 * Empty means the chain would copy the WAV, change nothing, delete the copy
 * and report success — which looks exactly like a bug from the outside, so
 * callers refuse the request instead of running it.
 *
 * Mirrors the gates in runPostProcessingChain(); keep the two in step.
 */
export function requestedPpStages(params: PostProcessParams): string[] {
  const stages: string[] = [];
  if (params.stableStepOn ?? params.stableStep) stages.push('StableStep');
  if (params.ppVaeReencode) stages.push('PP-VAE re-encode');
  if (params.spectralLifterEnabled) stages.push('Spectral Lifter');
  if (params.vocalNaturalizerEnabled && !params.instrumental) stages.push('Vocal Naturalizer');
  if ((params.gainOffsetDb ?? 0) !== 0) stages.push('Gain Offset');
  if (vstChainActive()) stages.push('VST chain');
  if (params.masteringEnabled && params.masteringReference) stages.push('Mastering');
  if (params.lufsEnabled && params.lufsTarget !== undefined) stages.push('Final Normalizer');
  return stages;
}

/** A song row's generation_params, or {} if it is missing or malformed. */
function safeParseGenerationParams(song: any): Record<string, any> {
  try {
    return song.generation_params ? JSON.parse(song.generation_params) : {};
  } catch {
    return {};
  }
}

export type PpEligibility =
  | { ok: true; rawWavPath: string; audioUrl: string }
  | { ok: false; status: number; error: string };

/**
 * Everything that must hold before a song can be post-processed after the fact.
 * Kept next to the runner so the route and the runner cannot disagree.
 */
export function checkPpEligibility(song: any): PpEligibility {
  if (!song) return { ok: false, status: 404, error: 'Song not found' };
  if (song.mastered_audio_url) {
    return {
      ok: false,
      status: 409,
      error: 'This track has already been post-processed. Running the chain again would overcook it.',
    };
  }
  if (inFlight.has(song.id)) {
    return { ok: false, status: 409, error: 'Post-processing is already running for this track' };
  }

  const audioUrl: string = song.audio_url || '';
  if (!audioUrl) return { ok: false, status: 400, error: 'Song has no audio file' };
  if (!audioUrl.toLowerCase().endsWith('.wav')) {
    return { ok: false, status: 400, error: 'Post-processing needs the raw WAV render, and this song is not a WAV' };
  }

  const rawWavPath = path.join(config.data.audioDir, path.basename(audioUrl));
  if (!fs.existsSync(rawWavPath)) {
    return { ok: false, status: 404, error: 'Audio file not found on disk' };
  }
  return { ok: true, rawWavPath, audioUrl };
}

/**
 * Queue a post-processing pass for an already-rendered song and return its job
 * id immediately. The pass itself waits for the GPU lane, so it never competes
 * with a running generation.
 *
 * `params` arrives from the browser holding the live PP settings. The master
 * switch is forced on here: the request itself is the user overriding the
 * global toggle.
 */
export function startRePostProcess(song: any, params: PostProcessParams): RePostProcessJob {
  const eligibility = checkPpEligibility(song);
  if (!eligibility.ok) {
    throw Object.assign(new Error(eligibility.error), { status: eligibility.status });
  }

  const { audioUrl } = eligibility;
  const jobId = uuidv4();
  const job: RePostProcessJob = {
    id: jobId,
    songId: song.id,
    status: 'pending',
    stage: gpuLaneBusy() ? `Queued (${gpuLaneDepth() + 1} ahead)` : 'Queued',
    createdAt: Date.now(),
  };
  jobs.set(jobId, job);
  inFlight.add(song.id);

  // Three fields belong to the SONG, not to the current form state, and taking
  // them from the browser would post-process this render as if it were whatever
  // is loaded in the panel right now:
  //   instrumental — decides whether the Vocal Naturalizer runs at all
  //   caption      — the StableStep/SA3 refine prompt
  //   seed         — what stableStepSeedFollowsDit follows
  const genParams = safeParseGenerationParams(song);
  const songCaption: string = genParams.caption || song.style || '';
  const ppParams: PostProcessParams = normalizePpParams(
    {
      ...params,
      // Forced on regardless of what the browser sent — the request itself is
      // the user overriding the global toggle.
      postProcessingEnabled: true,
      instrumental: !!genParams.instrumental,
      seed: typeof genParams.seed === 'number' ? genParams.seed : undefined,
    },
    [songCaption],
  );
  const stages = requestedPpStages(ppParams);

  startGenerationLog(jobId, 'postprocess');
  logGeneration(jobId, 'INFO', `[Post-Processing] Re-run for song ${song.id} (${song.title || 'untitled'})`);
  logGeneration(jobId, 'INFO', `[Post-Processing] Source: ${audioUrl}`);
  logGeneration(jobId, 'INFO', `[Post-Processing] Stages: ${stages.join(' -> ')}`);
  logGenerationParams(jobId, ppParams as Record<string, any>);

  void runOnGpuLane(async () => {
    job.status = 'running';
    job.stage = 'Starting...';

    const result = await runPostProcessingChain(
      [audioUrl],
      ppParams,
      1,
      jobId,
      (level, msg) => logGeneration(jobId, level, msg),
      (stage) => { job.stage = stage; },
    );

    const masteredUrl = result.masteredUrls[0] || '';
    if (!masteredUrl) {
      // Every requested stage declined to run — a missing model, or a VST that
      // failed to load. Reporting success here would leave the user wondering
      // why nothing changed.
      throw new Error('The chain ran but no stage produced audio — check the engine log for a missing model or a failed VST');
    }

    const quality = result.qualityScores[0];
    const qualityJson = (quality?.unmastered || quality?.mastered) ? JSON.stringify(quality) : '';
    if (qualityJson) {
      getDb().prepare('UPDATE songs SET mastered_audio_url = ?, quality_scores = ? WHERE id = ?')
        .run(masteredUrl, qualityJson, song.id);
    } else {
      getDb().prepare('UPDATE songs SET mastered_audio_url = ? WHERE id = ?')
        .run(masteredUrl, song.id);
    }

    job.masteredAudioUrl = masteredUrl;
    job.status = 'succeeded';
    job.stage = 'Complete';
    logGeneration(jobId, 'INFO', `[Post-Processing] Wrote ${masteredUrl}`);
    finishGenerationLog(jobId, 'postprocess');
    console.log(`[PostProcess] Song ${song.id}: ${masteredUrl}`);
  }).catch((err: any) => {
    const msg = err?.message || String(err);
    job.status = 'failed';
    job.error = msg;
    job.stage = 'Failed';
    logGeneration(jobId, 'ERROR', `[Post-Processing] ${msg}`);
    failGenerationLog(jobId, msg, 'postprocess');
    console.error(`[PostProcess] Song ${song.id} failed: ${msg}`);
  }).finally(() => {
    inFlight.delete(song.id);
  });

  return job;
}
