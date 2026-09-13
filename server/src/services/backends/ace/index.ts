// backends/ace/index.ts — ACE-Step 1.5 backend
//
// Backend manifest plus the thin step 4 wrapper around the existing ACE
// generation path. translateParams and the LM-echo body remain in the
// generation module; this file does not move or rewrite that mapping.

import { aceClient } from '../../aceClient.js';
import { config } from '../../../config.js';
import { isEngineSuspended, restartAceServer, stopAceServer } from '../../aceEngineProcess.js';
import { engineReady } from '../../../engineState.js';
import { runAceGeneration } from './generate.js';
import type {
  EngineBackend,
  BackendCapabilities,
  BackendModels,
  BackendLifecycleStatus,
  GenerationArtifact,
  GenerationContext,
  GenerationOperation,
  GenerationOutcome,
  ResolvedRequest,
  StageProfile,
} from '../types.js';
import type { GenerationJob } from '../../generation/jobTypes.js';

function status(): BackendLifecycleStatus {
  if (isEngineSuspended()) return 'suspended';
  if (engineReady) return 'ready';
  return 'down';
}

async function capabilities(): Promise<BackendCapabilities> {
  const up = engineReady && !isEngineSuspended();
  return {
    backend: 'ace',
    up,
    core: {
      // The LM's constrained-decoding FSM builds a duration prefix tree over
      // 10..600 s (metadata-fsm.h) and clamps anything above 600, so 600 is
      // the engine's real ceiling. 240 was a UI number with nothing behind it
      // (issue #101). The slider max is capability-driven, so this value IS
      // the user-visible limit. Quality past a few minutes is the user's call.
      // No auto: the LM is TOLD a length and the FSM aims for it — there is
      // no stop token that ends the song where it wants to end.
      duration: { max: 600, auto: false },
      bpm: true,
      keyscale: true,
      negativePrompt: true,
      batch: { max: 8 },
      seed: true,
    },
    features: {
      models: true,
      lm: true,
      plugins: true,
      // ACE is where the Lua sampler plugins came from; its own Generation
      // dropdown already renders the controls, so this flag changes nothing
      // here. It exists so MM3 can claim the capability without also claiming
      // ACE's dropdown (see the field docs in ../types.ts).
      samplerPlugins: true,
      adapters: true,
      // ACE's planner-LM LoRAs live inside its own LM/Thinking cluster
      // (lmAdapter/lmAdapterScale globals), not the generic picker this flag
      // gates — so false here means "not that UI", not "no LM adapters".
      lmAdapters: false,
      lmAdapterSelectable: false,
      postProcess: true,
      stableStep: true,
      whisper: true,
      lyricTimestamps: true,
      cover: true,
      repaint: true,
      lego: true,
      extract: true,
      streaming: true,
      training: true,
      midi: true,
      stems: true,
      understand: true,
      conceptSteering: true,
    },
    // ACE's solver/scheduler/guidance/adapter knobs are still surfaced via
    // the existing /api/plugins registry, not through this manifest yet —
    // Phase 1 leaves this empty rather than half-duplicating that route.
    extensions: [],
  };
}

async function models(): Promise<BackendModels> {
  try {
    const props = await aceClient.props();
    return {
      buckets: {
        lm: props.models.lm,
        dit: props.models.dit,
        vae: props.models.vae,
        embedding: props.models.embedding,
      },
      adapters: props.adapters,
      lmAdapters: props.lm_adapters ?? [],
      defaults: props.default,
    };
  } catch {
    // ace-server unreachable — same degrade-empty contract as routes/models.ts
    return {
      buckets: { lm: [], dit: [], vae: [], embedding: [] },
      adapters: [],
      lmAdapters: [],
      defaults: {},
    };
  }
}

const ACE_OPERATIONS: readonly GenerationOperation[] = [
  'text2music', 'cover', 'repaint', 'lego', 'extract', 'complete', 'cover-nofsq',
];

function firstText(submission: Readonly<Record<string, unknown>>, keys: string[]): string {
  for (const key of keys) {
    const value = submission[key];
    if (typeof value === 'string' && value) return value;
  }
  return '';
}

function optionalNumber(value: unknown, positive = false): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
  return positive && value <= 0 ? undefined : value;
}

/** Pure descriptive snapshot. The live ACE mapper remains translateParams(). */
function resolveRequest(submission: Readonly<Record<string, unknown>>): ResolvedRequest {
  const instrumental = typeof submission.instrumental === 'boolean'
    ? submission.instrumental : undefined;
  const duration = optionalNumber(submission.duration, true);
  const common = {
    caption: firstText(submission, ['prompt', 'songDescription', 'caption', 'style']),
    lyrics: instrumental === true
      ? '[Instrumental]'
      : typeof submission.lyrics === 'string' ? submission.lyrics : '',
    ...(instrumental === undefined ? {} : { instrumental }),
    ...(duration === undefined ? {} : { duration }),
    ...(typeof submission.seed === 'number' && Number.isFinite(submission.seed)
      ? { seed: submission.seed } : {}),
    ...(typeof submission.randomSeed === 'boolean' ? { randomSeed: submission.randomSeed } : {}),
    ...(typeof submission.batchSize === 'number' && Number.isFinite(submission.batchSize) && submission.batchSize > 0
      ? { batchSize: submission.batchSize } : {}),
    ...(typeof submission.title === 'string' ? { title: submission.title } : {}),
  };
  const taskType = typeof submission.taskType === 'string' && submission.taskType
    ? submission.taskType : 'text2music';
  const model = (key: string): string => {
    const value = submission[key];
    return typeof value === 'string' ? value : '';
  };
  return {
    operation: taskType,
    common,
    models: {
      dit: model('ditModel'),
      lm: model('lmModel'),
      vae: model('vaeModel'),
      embedding: model('embeddingModel'),
    },
    options: {},
    policy: { retry: { maxAttempts: 2, reseedOnRetry: true } },
  };
}

function stageProfile(stage: string | undefined): StageProfile {
  const text = stage ?? '';
  return { stallMs: text.startsWith('Decoding audio (VAE)') || !/: Step \d+/.test(text) ? 900_000 : 120_000 };
}

function outcomeFromJob(job: GenerationJob): GenerationOutcome {
  const result = job.result;
  const artifacts: GenerationArtifact[] = (result?.audioUrls ?? []).map((url, trackIndex) => ({
    kind: 'audio', trackIndex, url,
  }));
  // Per-track where the backend reported it; the scalars are take 0's only.
  const mastered = result?.masteredAudioUrls
    ?? (result?.masteredAudioUrl ? [result.masteredAudioUrl] : []);
  mastered.forEach((url, trackIndex) => {
    if (url) artifacts.push({ kind: 'mastered', trackIndex, url });
  });
  const noAdapter = result?.noAdapterAudioUrls
    ?? (result?.noAdapterAudioUrl ? [result.noAdapterAudioUrl] : []);
  noAdapter.forEach((url, trackIndex) => {
    if (url) artifacts.push({ kind: 'noadapter', trackIndex, url });
  });
  return {
    endReason: job.status === 'succeeded' ? 'completed' : job.status === 'cancelled' ? 'cancelled' : 'failed',
    stages: result?.timing ?? [],
    artifacts,
    songIds: result?.songIds ?? [],
    result,
    error: job.error,
  };
}

export const aceBackend: EngineBackend = {
  id: 'ace',
  displayName: 'ACE-Step 1.5',
  resourcePool: 'gpu',
  lifecycle: {
    // Phase 1: the engine process is bootstrapped and managed by
    // index.ts/aceEngineProcess.ts directly (respawn, crash budget,
    // suspension). Routing start/stop through here is a later-phase move —
    // for now these delegate to the same restart/stop primitives so the
    // interface shape is exercised without duplicating lifecycle ownership.
    async start() {
      await restartAceServer();
    },
    async stop() {
      await stopAceServer('Stopped via backend abstraction');
    },
    status,
  },
  capabilities,
  models,
  operations: ACE_OPERATIONS,
  resolveRequest,
  async generate(job: GenerationJob, ctx: GenerationContext): Promise<GenerationOutcome> {
    await runAceGeneration(job, { pollUntilDone: ctx.pollUntilDone, signal: ctx.signal });
    return outcomeFromJob(job);
  },
  stageProfile,
  arbitratesResidencyInEngine: false,
  /** Model-residency arbitration (plan §4.4). Evicts every resident, not
   *  in-use ACE module so the other family isn't fighting it for VRAM. Uses
   *  the same GET /models/loaded + POST /models/unload pair the VRAM
   *  indicator's manual unload uses (routes/logs.ts:120-145) — one label per
   *  call is all the engine's endpoint accepts. Best-effort throughout: this
   *  runs fire-and-forget behind a backend switch and must never throw. */
  async releaseVram() {
    let loaded: Array<{ label: string; in_use?: boolean }> = [];
    try {
      const res = await fetch(`${config.aceServer.url}/models/loaded`, {
        signal: AbortSignal.timeout(3000),
      });
      if (!res.ok) return;
      const body = await res.json() as { loaded?: Array<{ label: string; in_use?: boolean }> };
      loaded = Array.isArray(body.loaded) ? body.loaded : [];
    } catch {
      return;   // engine unreachable — nothing resident that we can free anyway
    }

    const freed: string[] = [];
    for (const m of loaded) {
      if (!m?.label || m.in_use) continue;   // in-use modules are skipped engine-side too
      if (await aceClient.unloadLabel(m.label)) freed.push(m.label);
    }
    if (freed.length) {
      console.log(`[Backends] ACE-Step residency released: ${freed.join(', ')}`);
    }
  },
};
