// backends/sa3/index.ts — Stable Audio 3 as a PRIMARY generation backend
//
// HOT-Step (MDMAchine / A&E Concepts). SA3 already ships in HOT-Step-CPP as
// the StableStep post-processing refine (SDEdit: audio in → partial noise →
// denoise → audio out). This module exposes the model's PRIMARY use —
// text-to-audio from pure noise — as a third entry in the backend picker
// alongside ACE-Step and MiniMax-Music3.
//
// Shape mirrors backends/minimax/index.ts: same EngineBackend contract, same
// lifecycle story (no process of its own — a model family inside ace-server),
// same capabilities → generic-dropdown rendering. Deliberately smaller: SA3
// has no planner LM, no LRC, no streaming, no multi-take.
//
// Engine side: hot-step-server.cpp /sa3/props + /sa3-synth, which is
// /sa3-refine minus the input (sa3-refine.h sa3_refine_run_backend with
// noise_only=true). Node side: backends/sa3/generate.ts.
//
// The StableStep refine is untouched. It stays the post-processing stage it
// always was, on every backend that declares `stableStep: true`.

import { engineReady } from '../../../engineState.js';
import { isEngineSuspended } from '../../aceEngineProcess.js';
import { aceClient } from '../../aceClient.js';
import type { Sa3Props } from '../../aceClient.js';
import { runSa3Generation } from './generate.js';
import { sa3ModelCatalogue, sa3SelectedDit, selectSa3Model } from './models.js';
import type {
  EngineBackend,
  BackendCapabilities,
  BackendModels,
  BackendLifecycleStatus,
  BackendExtensionParam,
  GenerationArtifact,
  GenerationContext,
  GenerationOperation,
  GenerationOutcome,
  ResolvedRequest,
} from '../types.js';
import type { GenerationJob } from '../../generation/jobTypes.js';

/** Soft ceiling for the duration slider. The engine's tiled encode/decode
 *  handles any length (sa3_chunk_starts in sa3-refine.h); this bounds the UI
 *  and VRAM, not the model. Matches the engine's own clamp in /sa3-synth. */
export const SA3_MAX_DURATION_SEC = 600;

// ── Props (last-known-good cache, same discipline as mm3Props) ─────────────

let lastGoodProps: Sa3Props | null = null;
let lastGoodAt = 0;

async function sa3PropsCached(): Promise<{ props: Sa3Props | null; stale: boolean }> {
  const p = await aceClient.sa3Props();
  if (p) {
    lastGoodProps = p;
    lastGoodAt = Date.now();
    return { props: p, stale: false };
  }
  return { props: lastGoodProps, stale: true };
}

function status(): BackendLifecycleStatus {
  if (isEngineSuspended()) return 'suspended';
  if (!engineReady) return 'down';
  return 'ready';
}

// ── Capabilities ────────────────────────────────────────────────────────────

/** Backend-declared knobs, rendered by the generic dropdown. Keys are the
 *  `params.sa3*` fields backends/sa3/generate.ts reads. Same schema the Lua
 *  plugins use so PluginControls renders them. */
const SA3_EXTENSIONS: BackendExtensionParam[] = [
  {
    key: 'sa3Steps',
    label: 'Steps',
    type: 'slider',
    default: 8,
    min: 1,
    max: 64,
    step: 1,
    hint: 'Denoise steps from pure noise. 8 is the default and ear-tested best '
        + 'with MD Pingpong Blend (it beat 12 and 24); more steps tend to sound worse.',
    group: 'generation',
  },
  {
    key: 'sa3Sampler',
    label: 'Sampler',
    type: 'select',
    default: 'md_pingpong_blend',
    options: [
      { value: 'md_pingpong_blend', label: 'MD Pingpong Blend (default)' },
      { value: 'pingpong', label: 'PingPong (native)' },
      { value: 'euler',    label: 'Euler' },
    ],
    hint: 'SA3 sampler. MD Pingpong Blend runs the Lua plugin of that name and '
        + 'needs no Sampler Plugins toggle. Ignored when a Lua solver plugin is selected — '
        + 'a solver REPLACES the native step, they are alternatives not a stack.',
    group: 'generation',
  },
  // The opt-in gate for the shared Lua pickers. Same key MM3 declares — the
  // generic dropdown renders SamplerPluginControls when this is on, and
  // backends/sa3/generate.ts only forwards solver/scheduler/guidance when
  // it is (mapSa3SamplerPlugins). Off = native pingpong, parity-tested.
  {
    key: 'samplerPluginsEnabled',
    type: 'toggle',
    label: 'Sampler Plugins',
    hint: 'Drive the SA3 denoise with the shared Lua solver/scheduler/guidance '
        + 'plugins instead of the native pingpong loop. Full-loop (owns_loop) '
        + 'solvers are not offered: SA3 is distilled for pingpong sampling and '
        + 'they skip the per-step re-noise it needs. '
        + 'Guidance is mostly decorative: SA3 has no CFG uncond branch, so '
        + 'APG-family modes pass through unchanged.',
    default: false,
    group: 'generation',
  },
  // Optional output loudness normalization. Off by default — the output level
  // swings roughly 18 dB across captions and this is the escape valve. Uses
  // BS.1770-4 integrated LUFS measurement (audioLevel.integratedLufs) followed
  // by a look-ahead peak limiter at -1 dBFS (audioLevel.limitPeaks). Applied
  // after the engine WAV is written, before post-processing.
  {
    key: 'sa3LufsEnabled',
    type: 'toggle',
    label: 'Loudness Normalize',
    hint: 'Normalize the SA3 output to a target integrated LUFS before '
        + 'post-processing. The output level can swing ~18 dB depending on the '
        + 'caption; this brings it to a consistent level. Peak ceiling is always '
        + '-1 dBFS.',
    default: false,
    group: 'generation',
  },
  {
    key: 'sa3LufsTarget',
    type: 'slider',
    label: 'LUFS Target',
    default: -14,
    min: -30,
    max: -6,
    step: 0.5,
    hint: 'Integrated LUFS target after normalization. -14 LUFS is the streaming '
        + 'standard; -18 to -23 suits material that will go through a mastering '
        + 'chain. Peak ceiling is always -1 dBFS regardless.',
    group: 'generation',
    visible_when: { key: 'sa3LufsEnabled', equals: 'true' },
  },
];

async function capabilities(): Promise<BackendCapabilities> {
  const { props, stale } = await sa3PropsCached();
  const synthReady = props?.synth_ready === true;
  const up = engineReady && !isEngineSuspended() && synthReady;

  // Honest "point the user at the model manager" signal: engine reachable,
  // props answered, weights genuinely absent. Fail open on a stale manifest.
  const modelsMissing = !stale && props != null && !synthReady;

  return {
    backend: 'sa3',
    up,
    core: {
      // auto: false — SA3 has no stop token. The requested duration IS the
      // output length (the engine trims to it exactly).
      duration: { max: SA3_MAX_DURATION_SEC, auto: false },
      bpm: false,
      keyscale: false,
      // No CFG uncond branch (trained at cfg=1) — see sa3-refine.h. A negative
      // prompt has nowhere to go.
      negativePrompt: false,
      batch: { max: 1 },
      seed: true,
      sampleRate: 44100,
      promptTokenLimit: props?.prompt_token_limit ?? 256,
      propsStale: stale,
      modelsMissing,
    },
    features: {
      models: true,
      lm: false,
      plugins: false,
      // The SA3 sampler loop runs the same Lua solvers/schedulers/guidance as
      // ACE and MM3 (sa3-refine.h Sa3PluginParams). Same gating pattern as
      // MM3: `plugins: false` picks the generic dropdown, `samplerPlugins:
      // true` renders the plugin controls beside the extensions.
      samplerPlugins: true,
      // SA3 medium is distilled for pingpong sampling and owns_loop solvers
      // bypass the per-step re-noise that distillation needs, so they are
      // filtered out of the dropdown like on MM3 (finding 2026-09-12). The
      // engine mirrors this at the C++ level (sa3-refine.h resolves owns_loop
      // solvers to native pingpong instead of dispatching them).
      samplerPluginsOwnsLoop: false,
      // DoRA adapters in models/sa3-adapters/*.gguf. These are the StableStep
      // adapters — the same files, the same merge-at-load path. `adapters`
      // rather than `lmAdapters` because there is no LM; the catalogue comes
      // back from models().
      adapters: props != null && props.adapters.length > 0,
      lmAdapters: false,
      lmAdapterSelectable: false,
      forcedAlignment: false,
      // VST chain + mastering read the rate from the WAV; SA3 is 44.1k native.
      postProcess: true,
      // Refining SA3 output with SA3 is legal but a no-op in spirit. Off.
      stableStep: false,
      whisper: true,
      lyricTimestamps: false,
      cover: true,
      coverAudioTransform: true,
      repaint: false,
      lego: false,
      extract: false,
      streaming: false,
      training: false,
      midi: false,
      stems: false,
      understand: false,
      conceptSteering: false,
    },
    extensions: SA3_EXTENSIONS,
  };
}

// ── Models ──────────────────────────────────────────────────────────────────

async function models(): Promise<BackendModels> {
  const { props } = await sa3PropsCached();
  return {
    ...sa3ModelCatalogue(),
    adapters: props?.adapters ?? [],
  };
}

const SA3_OPERATIONS: readonly GenerationOperation[] = ['text2music', 'cover'];

function firstText(submission: Readonly<Record<string, unknown>>, keys: string[]): string {
  for (const key of keys) {
    const value = submission[key];
    if (typeof value === 'string' && value) return value;
  }
  return '';
}

/** Pure descriptive snapshot, same contract as minimax/yue2's resolveRequest:
 *  a frozen record of what was asked for at enqueue time. The live mapping
 *  (defaults, clamping, plugin whitelist) stays in generate.ts's own readers
 *  of job.params — this function never invokes them. */
function resolveRequest(submission: Readonly<Record<string, unknown>>): ResolvedRequest {
  const duration = typeof submission.duration === 'number' && Number.isFinite(submission.duration) && submission.duration > 0
    ? submission.duration : undefined;
  const options: Record<string, unknown> = {};
  for (const key of Object.keys(submission)) {
    if (key.startsWith('sa3')) options[key] = submission[key];
  }
  const common = {
    // SA3 has no CFG-uncond negative-prompt slot and no vocals — the caption
    // is the whole prompt (see generate.ts's header note on tokenisation).
    caption: firstText(submission, ['caption', 'prompt', 'songDescription', 'style']),
    lyrics: '',
    ...(duration === undefined ? {} : { duration }),
    ...(typeof submission.seed === 'number' && Number.isFinite(submission.seed) ? { seed: submission.seed } : {}),
    ...(typeof submission.randomSeed === 'boolean' ? { randomSeed: submission.randomSeed } : {}),
    ...(typeof submission.title === 'string' ? { title: submission.title } : {}),
  };
  return {
    operation: submission.taskType === 'cover' ? 'cover' : 'text2music',
    common,
    models: { dit: typeof submission.sa3DitModel === 'string'
      ? submission.sa3DitModel : sa3SelectedDit() },
    options,
    policy: { retry: { maxAttempts: 2, reseedOnRetry: true } },
  };
}

function outcomeFromJob(job: GenerationJob): GenerationOutcome {
  const result = job.result;
  const artifacts: GenerationArtifact[] = (result?.audioUrls ?? []).map((url, trackIndex) => ({
    kind: 'audio', trackIndex, url,
  }));
  const mastered = result?.masteredAudioUrls
    ?? (result?.masteredAudioUrl ? [result.masteredAudioUrl] : []);
  mastered.forEach((url, trackIndex) => {
    if (url) artifacts.push({ kind: 'mastered', trackIndex, url });
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

// ── Backend ─────────────────────────────────────────────────────────────────

export const sa3Backend: EngineBackend = {
  id: 'sa3',
  displayName: 'Stable Audio 3',
  resourcePool: 'gpu',
  lifecycle: {
    async start() { /* no separate process — ace-server owns the lifecycle */ },
    // The SA3 model store entry is small (~2 GB) and the ModelHandle in the
    // engine handler already releases it per request. Nothing to unload.
    async stop() { /* nothing resident between requests */ },
    status,
  },
  capabilities,
  models,
  selectModel: selectSa3Model,
  operations: SA3_OPERATIONS,
  resolveRequest,
  /** /sa3-synth is one synchronous call — no engine job id to poll, so
   *  ctx.pollUntilDone is unused here (same shape as MM3/YuE2's generate, minus
   *  the poll). The shared abort signal from the lane replaces the runner's
   *  own AbortController, so a cancel/reset from the lane reaches the fetch
   *  the same way it does on every other backend. */
  async generate(job: GenerationJob, ctx: GenerationContext): Promise<GenerationOutcome> {
    await runSa3Generation(job, { signal: ctx.signal });
    return outcomeFromJob(job);
  },
  // No releaseVram: nothing stays resident between /sa3-synth calls. Cross-
  // family VRAM arbitration is handled generically by routes/backends.ts,
  // which calls the OUTGOING backend's own releaseVram() (e.g. MM3's) when
  // the active backend changes — SA3 needs no defensive call of its own.
};
