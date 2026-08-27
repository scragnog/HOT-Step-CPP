// backends/minimax/client.ts — typed HTTP client for the engine's /mm3/* family
//
// MiniMax-Music3 lives INSIDE the same ace-server process as the ACE pipeline
// (plan §4.3: "same ace-server process, separate source tree"), so this client
// talks to the same base URL aceClient does — there is no second engine port.
// Only the family-scoped endpoints differ:
//
//   GET  /mm3/props           discovery + residency (see the BLOCKING warning)
//   POST /mm3/synth           production generation → { job_id, id, seed, ... }
//   GET  /mm3/job?id=         MM3-native stage detail (never takes the MM3 mutex)
//   POST /mm3/unload          free MM3 VRAM (idempotent)
//   POST /mm3/tokenize-check  cold-capable prompt-token pre-flight
//
// Progress / result / cancel for an MM3 job go through the STANDARD job
// endpoints (GET /job?id=, GET /job?id=&result=1, POST /job?id=&cancel=1) —
// i.e. aceClient.pollJob / getJobResult / cancelJob work unchanged. That is
// deliberate: the MM3 job rides the engine's one GPU worker queue like any ACE
// job, so the whole polling/cancel machinery is shared, not duplicated.
//
// NOTE on the base URL: aceClient.ts:12 snapshots `config.aceServer.url` at
// import time (a known latent staleness bug, plan §3.1). This module reads it
// per call instead, so a hot-reloaded host/port is picked up.

import { config } from '../../../config.js';

const base = () => config.aceServer.url;

/** GET /mm3/props takes the engine-side MM3 mutex, which an in-flight MM3
 *  generation holds for its ENTIRE run. A props call during a generation
 *  therefore blocks until the song finishes. Never wait on it — 2.5 s, then
 *  fall back to the last known-good manifest. */
const TIMEOUT_PROPS    = 2_500;
const TIMEOUT_QUICK    = 15_000;   // synth submit, unload — all cheap/synchronous
const TIMEOUT_TOKENIZE = 30_000;   // cold-capable: may load the tokenizer from the GGUF header
const TIMEOUT_JOB      = 10_000;   // /mm3/job never takes the MM3 mutex

// ── Wire shapes (mirrors engine/src/minimax/mm3-server.h + mm3-job.h) ──

export interface Mm3PropsFile {
  found?: boolean;
  path?: string;
  probe_ok?: boolean;
  [k: string]: unknown;
}

/** One mm3-{lm,synth}-<quant>.gguf found on disk. `quant` is the filename
 *  token verbatim ('f16' | 'q8_0' | 'Q4_K_M' | ...) and is the authoritative
 *  label — general.file_type is display-only and unassigned for some quants. */
export interface Mm3VariantFile {
  quant: string;
  filename: string;
  bytes: number;
}

export interface Mm3RoleVariants {
  available: Mm3VariantFile[];
  /** The quant actually in force. Differs from `requested` when the requested
   *  file has gone missing and the engine fell back to best-first. */
  selected: string;
  /** What was last asked for; '' means auto. */
  requested: string;
}

export interface Mm3Props {
  backend: string;                 // 'minimax-m3'
  model: string;                   // 'MiniMax-Music3'
  available: boolean;              // both GGUFs discovered
  loaded: boolean;                 // weights currently resident in VRAM
  /** Both GGUFs found AND their headers parsed clean — i.e. POST /mm3/synth
   *  will get as far as loading weights. NOT a VRAM/residency claim. */
  synth_ready: boolean;
  prompt_token_limit: number;      // 5000
  max_audio_frames_limit: number;  // 9000 (= 360 s @ 25 fps)
  models_dir?: string;
  search_dirs?: string[];
  /** Since the 5-way split: per-role entries (depth/cond/dit/voc). `synth` is
   *  a legacy alias the engine still emits — it mirrors the DiT role. */
  files?: {
    lm?: Mm3PropsFile; depth?: Mm3PropsFile; cond?: Mm3PropsFile;
    dit?: Mm3PropsFile; voc?: Mm3PropsFile; synth?: Mm3PropsFile;
  };
  variants?: {
    lm?: Mm3RoleVariants; depth?: Mm3RoleVariants; cond?: Mm3RoleVariants;
    dit?: Mm3RoleVariants; voc?: Mm3RoleVariants; synth?: Mm3RoleVariants;
  };
  vram?: Record<string, number>;
  errors?: string[];
  [k: string]: unknown;
}

export interface Mm3SelectModelResult {
  changed: boolean;
  /** True when the switch evicted resident weights (a re-warm will follow).
   *  Since the split, only the affected parts are evicted — a DiT swap keeps
   *  the 17 GB LM warm. */
  unloaded: boolean;
  lm: string;                 // resolved filenames
  depth?: string;
  cond?: string;
  dit?: string;
  voc?: string;
  /** Legacy alias — mirrors `dit`. */
  synth: string;
  available: boolean;
  error?: string;
}

/** Per-role quant selection. '' (or omitted) = auto/best-first. */
export interface Mm3Selection {
  lm?: string;
  depth?: string;
  cond?: string;
  dit?: string;
  voc?: string;
  /** Legacy: one token for the whole non-LM stack (pre-split contract). Only
   *  sent when `dit` is absent; the engine maps it onto all four roles. */
  synth?: string;
}

export interface Mm3PropsResult {
  /** null only if /mm3/props has never answered since server start. */
  props: Mm3Props | null;
  /** true when `props` came from the last-known-good cache (probe timed out or
   *  failed), so callers can degrade honestly instead of reporting "down". */
  stale: boolean;
  error?: string;
  /** Epoch ms of the fetch that produced `props` (0 when never fetched). */
  fetchedAt: number;
}

export interface Mm3SynthRequest {
  /** REQUIRED, non-blank. The Structured Caption (see .claude/skills/mm3-captioning). */
  caption: string;
  /** "" (or omitted) → the engine substitutes its instrumental lyric. */
  lyrics?: string;
  /** Seconds. Required unless max_frames is given; engine maps to frames @25 fps. */
  duration?: number;
  /** Escape hatch — wins over duration. */
  max_frames?: number;
  /** Ensemble takes: render N DIFFERENT songs from this one prompt in a single
   *  batched autoregressive pass. Take t is drawn from seed + t. Clamped by the
   *  engine to the checkpoint's row budget (4 with a CFG pair), so read the
   *  response's `takes` for what will actually be produced. Omit for 1. */
  takes?: number;
  /** -1 = engine draws one; the resolved value comes back in the response. */
  seed?: number;
  /** Default = the checkpoint's flow.cfg_scale (1.7). */
  cfg_flow?: number;
  /** Default = the checkpoint's flow.steps (30). */
  steps?: number;
  /** 16 | 24 | 32, default 16. */
  get_wav_bits?: number;
  /** Emit LRC lyric timestamps from the LM's alignment heads. Since the
   *  post-hoc replay pass (2026-08-21) this costs a fraction of a second
   *  after the AR stage — the decode itself is untouched. Ignored for
   *  instrumentals. */
  get_lrc?: boolean;

  // ── MM3 Plank (engine: minimax/mm3-job.h) ─────────────────────────────────
  /** Capture the AR stage's output codes so a later render can replay them.
   *  When set, GET /mm3/job?id=<id>&ar=1 serves the blob once the job is done.
   *  Zero cost when false (the default) — the codes already exist in memory. */
  get_ar_codes?: boolean;
  /** Replay previously-captured codes instead of sampling them. Both must be
   *  sent together, with forced_acoustic.length === forced_semantic.length * 7.
   *  This does NOT make the render faster: the AR loop still runs every
   *  per-frame forward pass. It pins WHICH codes come out, which is what makes
   *  a flow-stage A/B comparison valid. */
  forced_semantic?: number[];  // [I], entry 0 is the un-emitted iteration
  forced_acoustic?: number[];  // [I * 7], flat, iteration-major

  // ── AR cache (engine: minimax/mm3-ar-cache.h) ─────────────────────────────
  /** Reuse the previous render's AR (planner) output when every AR-affecting
   *  input is unchanged, skipping stage 1 entirely. UNLIKE the plank above this
   *  IS a large speedup — AR is roughly half a render — because the flow DiT's
   *  real input is the frame-hidden block, not the codes. Costs one block of
   *  ENGINE host RAM (~128 KB per frame, so ~600 MB for a 200 s song), which is
   *  why it is opt-in. The engine keys the lookup itself; the server never has
   *  to decide what "unchanged" means. */
  reuse_ar?: boolean;

  // ── Streaming (engine: minimax/mm3-job.h, GET /mm3/stream) ────────────────
  /** Emit each 200-frame window's audio as soon as it is vocoded and cropped,
   *  so playback can start before the render finishes. The finished WAV is
   *  still produced and fetched exactly as it is without this — streaming is an
   *  ADDITIONAL output, never a replacement, so every downstream feature (song
   *  row, library, mastering, stems) is unaffected.
   *
   *  Opt-in because it is not free: the vocoder moves inside the flow loop and
   *  the emitted chunks are buffered in engine host RAM until a reader drains
   *  them. The engine echoes `streaming` back to say whether it will serve. */
  stream?: boolean;
  /** Seed for the AR stage only. Omitted = tied to `seed`, which is MM3's
   *  native behaviour (one seed drives both the plan and the flow noise).
   *  Splitting them lets the flow noise be rerolled while the plan — and
   *  therefore the AR cache entry — stays put. */
  ar_seed?: number;

  // ── Saved plans (engine: minimax/mm3-hiddens-file.h) ──────────────────────
  // The AR cache above, on disk. The blob never crosses this wire in either
  // direction — the engine reads and writes it itself, because at ~600 MB for a
  // 200 s song a round trip through Node would cost more than the render it
  // saves. These fields carry PATHS only.

  /** Replay a saved plan: absolute path to a `.mm3hiddens` file. Primes the
   *  engine's in-memory AR slot, so the run then behaves exactly like a cache
   *  hit — no LM load, no staged handover, codes and LRC handed back. The
   *  engine REFUSES a plan saved under a different LM, depth model or LM
   *  adapter and plans fresh instead, so a stale file cannot render silently
   *  wrong. Mutually exclusive with `forced_semantic`. */
  forced_frame_hiddens_file?: string;
  /** Write this render's plan to disk after it succeeds. Needs a path. */
  save_frame_hiddens?: boolean;
  /** Absolute path the engine writes the plan to. Written atomically (tmp +
   *  rename), so an interrupted save leaves any previous file intact. */
  frame_hiddens_save_path?: string;

  // ── Sampler plugins (engine: minimax/mm3-plugins.h) ───────────────────────
  // The SAME Lua solver/scheduler/guidance plugins the ACE DiT uses, driving
  // MM3's flow DiT through a convention adapter. Field names are ACE's, so one
  // UI control set feeds both backends.
  //
  // Every one of these is optional and every one defaults to MM3's native,
  // parity-proven flow loop. Omit them and nothing changes — which is why the
  // server only sends them when the user has explicitly opted in.

  /** Lua solver plugin name. Full-loop (owns_loop) solvers are not supported on
   *  MM3 yet — the engine warns and falls back to native Euler. */
  infer_method?: string;
  /** Lua scheduler plugin name. */
  scheduler?: string;
  /** Lua guidance plugin name, or "apg" for the native APG path. */
  guidance_mode?: string;
  /** Timestep warp handed to a scheduler plugin, (0, 20]. Default 1.0, which is
   *  what MM3's own hardcoded schedule uses. Ignored without a scheduler. */
  flow_shift?: number;
  /** APG per-channel norm clip, [0, 100]. Default 2.5. */
  apg_norm_threshold?: number;
  /** Declared plugin params, {"pluginName:key": value} — the same map and the
   *  same coercion rules as the ACE path's `plugin_params`. */
  plugin_params?: Record<string, string>;
}

export interface Mm3SynthResponse {
  job_id: string;
  id: string;                 // the ACE /synth spelling — same value
  seed: number;               // RESOLVED seed (never -1)
  /** The same seed as a DECIMAL STRING, and the only spelling that survives.
   *  A uint64 does not fit a float64: 18226392072674864222 and its two
   *  successors all parse to the same JS number, which is what made an
   *  ensemble's takes report one seed and become individually
   *  unreproducible. Prefer this everywhere a seed is stored or displayed. */
  seed_str?: string;
  /** Ensemble takes the engine ACCEPTED — clamped to the checkpoint's row
   *  budget, so it is how many songs and how many streams will exist, not
   *  what was asked for. 1 for an ordinary render. */
  takes?: number;
  max_frames: number;
  duration: number;
  prompt_tokens: number;
  prompt_token_limit: number;
  instrumental: boolean;
  steps: number;
  cfg_flow: number;
  wav_bits: number;
  /** Whether GET /mm3/stream will actually serve this job. Today it mirrors the
   *  `stream` field that was sent; it is a separate answer because the engine
   *  reserves the right to decline (the co-residency path has a VRAM gate) and
   *  a decline must never fail the render — the caller just does not stream. */
  streaming?: boolean;
  /** Present only when a sampler plugin was actually selected. Absent means the
   *  native flow loop ran — which is what makes this the honest answer to "did
   *  my picks reach the engine?". */
  sampler_plugins?: {
    solver: string;
    scheduler: string;
    guidance: string;
    shift: number;
    n_params: number;
  };
}

/** MM3-native stage detail. `stage` is the real MM3 vocabulary
 *  (queued|arbitrating|warming|warm|ar|cond|flow|vocode|stitch|encoding|done|
 *  failed|cancelled), unlike GET /job's ACE-phase mapping. */
export interface Mm3JobDetail {
  id: string;
  status: string;             // running | done | failed | cancelled | unknown
  phase: string;              // the ACE-phase mapping
  stage: string;
  window: number;
  n_windows: number;
  step: number;
  n_steps: number;
  seed: number;
  /** Lossless decimal-string seed — see Mm3SynthResponse.seed_str. */
  seed_str?: string;
  max_frames: number;
  prompt_tokens: number;
  instrumental: boolean;
  /** AR cache: true when stage 1 was skipped entirely for this job. Set as soon
   *  as the job starts, not only on the finished result. */
  ar_cached?: boolean;
  /** True when that cache was primed from a saved plan rather than by the
   *  previous render. Reported apart from `ar_cached` because the two are
   *  indistinguishable from the outside and mean different things: one is the
   *  cache working, the other is a plan the user picked. */
  ar_from_file?: boolean;
  /** Streaming: true when this job was submitted with `stream: true`. */
  streaming?: boolean;
  /** True when windows were dispatched WHILE the planner was still running —
   *  i.e. audio starts seconds in rather than after the whole plan. Decided on
   *  the worker thread by a VRAM check, so it is only knowable from here, never
   *  from the submit response. False on a streamed run is not a failure; it is
   *  the serial fallback. */
  stream_interleaved?: boolean;
  /** Ensemble takes this job actually rendered — always present, and 1 for an
   *  ordinary render. It is the CLAMPED count (the engine caps it at the
   *  checkpoint's row budget), so it is what exists to fetch, never what was
   *  asked for. */
  takes?: number;
  /** Per-take summary, present only when there is more than one take. Take t's
   *  audio is at GET /mm3/take?id=<id>&take=<t> and its live stream at
   *  GET /mm3/stream?id=<id>&take=<t>. */
  take_detail?: Array<{
    take: number;
    seed: number;
    /** Lossless decimal-string seed — the only one safe to store or show. */
    seed_str?: string;
    frames: number;
    duration_s: number;
    eos: boolean;
    rms: number;
    peak: number;
    /** Take 0 is served by the shared /job?id=&result=1 as well; the rest only
     *  by /mm3/take. Reported so a client need not special-case index 0. */
    audio_ready: boolean;
    streaming: boolean;
  }>;
  /** Chunks pushed so far — 0 while the AR stage is still planning. */
  stream_chunks?: number;
  stream_mb?: number;
  /** Whether a reader is currently attached to GET /mm3/stream. */
  stream_reader?: boolean;
  evicted_modules?: number;
  evicted_mb?: number;
  arbitrate_ms?: number;
  warm_ms?: number;
  error?: string;
  result?: {
    frames: number;
    n_samples: number;
    sample_rate: number;
    channels: number;
    duration_sec: number;
    rms: number;
    peak: number;
    eos: boolean;
    has_nan: boolean;
    /** MM3 Plank: true when get_ar_codes was set and the blob is ready for
     *  GET /mm3/job?id=<id>&ar=1. */
    ar_codes_available?: boolean;
    ms?: { ar: number; cond: number; flow: number; voc: number; total: number };
  };
}

export interface Mm3TokenizeCheck {
  tokens: number;
  limit: number;
  /** false is a 200, not an error — only POST /mm3/synth turns a "no" into a 400. */
  ok: boolean;
  instrumental: boolean;
  caption_clean?: string;
  lyrics_normalized?: string;
  prompt?: string;
}

export interface Mm3UnloadResult {
  /** engine kept its host-side AR slot across this unload */
  ar_cache_kept?: boolean;
  unloaded: boolean;   // true only if something was actually resident
  loaded: boolean;     // always false on success
  freed_bytes?: number;
  freed_mb?: number;
}

// ── Helpers ──

/** The engine reports every /mm3 failure as {"error": "..."} with a real HTTP
 *  status. Surface that message verbatim — the validation errors are written
 *  for humans ("\"duration\" must be greater than 0 seconds"). */
async function mm3ErrorMessage(res: Response, endpoint: string): Promise<string> {
  const body = await res.text().catch(() => '');
  let msg = body;
  try {
    const parsed = JSON.parse(body) as { error?: string };
    if (parsed?.error) msg = parsed.error;
  } catch { /* non-JSON body — use it raw */ }
  return `ace-server ${endpoint} failed (${res.status}): ${msg || 'unknown error'}`;
}

async function mm3Post<T>(endpoint: string, body: unknown, timeoutMs: number): Promise<T> {
  const res = await fetch(`${base()}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(await mm3ErrorMessage(res, `POST ${endpoint}`));
  return await res.json() as T;
}

// ── Props (with last-known-good cache) ──

let lastGoodProps: Mm3Props | null = null;
let lastGoodAt = 0;

/** GET /mm3/props with a short timeout and a last-known-good fallback.
 *
 *  A timeout here almost always means "an MM3 generation is running and holds
 *  the engine-side mutex" — which is the opposite of down. Treating it as down
 *  would make the backend toggle flicker to unavailable mid-song, so the cached
 *  manifest is returned with `stale: true` instead. */
export async function mm3Props(timeoutMs = TIMEOUT_PROPS): Promise<Mm3PropsResult> {
  try {
    const res = await fetch(`${base()}/mm3/props`, { signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) throw new Error(await mm3ErrorMessage(res, 'GET /mm3/props'));
    const props = await res.json() as Mm3Props;
    lastGoodProps = props;
    lastGoodAt = Date.now();
    return { props, stale: false, fetchedAt: lastGoodAt };
  } catch (err: any) {
    return {
      props: lastGoodProps,
      stale: true,
      error: err?.message || String(err),
      fetchedAt: lastGoodAt,
    };
  }
}

/** The cached manifest without touching the engine (never blocks). */
export function mm3PropsCached(): Mm3Props | null {
  return lastGoodProps;
}

// ── Generation ──

/** POST /mm3/synth — submit a generation. Every validation failure (blank
 *  caption, bad type, over-long prompt, bad duration) is a synchronous 400, so
 *  a resolved promise here means the job will actually run. */
export async function mm3Synth(req: Mm3SynthRequest): Promise<Mm3SynthResponse> {
  return mm3Post<Mm3SynthResponse>('/mm3/synth', req, TIMEOUT_QUICK);
}

/** The engine URL of a job's live audio stream — a chunked body of
 *  concatenated self-contained WAVs, same transport as STORM's.
 *
 *  Deliberately a URL rather than a fetch helper: the Node route pipes the body
 *  through untouched, and buffering it here to hand back a Buffer would undo
 *  the entire point of the feature. */
export function mm3StreamUrl(jobId: string): string {
  return `${base()}/mm3/stream?id=${encodeURIComponent(jobId)}`;
}

/** GET /mm3/job?id= — MM3 stage detail. Soft-fails to null: this is progress
 *  garnish on top of the authoritative GET /job status, never a failure signal. */
export async function mm3JobDetail(jobId: string): Promise<Mm3JobDetail | null> {
  try {
    const res = await fetch(`${base()}/mm3/job?id=${encodeURIComponent(jobId)}`, {
      signal: AbortSignal.timeout(TIMEOUT_JOB),
    });
    if (!res.ok) return null;
    return await res.json() as Mm3JobDetail;
  } catch {
    return null;
  }
}

/** POST /mm3/tokenize-check — cold-capable prompt-token pre-flight against the
 *  5,000-token limit. Works without the weights resident. */
export async function mm3TokenizeCheck(
  caption: string, lyrics = '', echoPrompt = false,
): Promise<Mm3TokenizeCheck> {
  return mm3Post<Mm3TokenizeCheck>(
    '/mm3/tokenize-check', { caption, lyrics, prompt: echoPrompt }, TIMEOUT_TOKENIZE,
  );
}

// ── Model selection ──

/** POST /mm3/select-model — choose the quant of each role to run.
 *
 *  '' means auto (engine picks best-first). Idempotent: posting the current
 *  selection returns changed:false and touches nothing. A real change evicts
 *  the resident weights, so the next generation re-warms. Throws on an unknown
 *  quant (400) — unlike the residency helpers below, a failed model switch must
 *  be visible, not swallowed, or the user silently keeps generating on the old
 *  weights. */
export async function mm3SelectModel(sel: Mm3Selection): Promise<Mm3SelectModelResult> {
  // Per-role when any split key is present; otherwise the legacy synth shape.
  const body: Record<string, string> =
    sel.dit !== undefined || sel.depth !== undefined || sel.cond !== undefined || sel.voc !== undefined
      ? {
          lm:    sel.lm ?? '',
          depth: sel.depth ?? '',
          cond:  sel.cond ?? '',
          dit:   sel.dit ?? '',
          voc:   sel.voc ?? '',
        }
      : { lm: sel.lm ?? '', synth: sel.synth ?? '' };
  return mm3Post<Mm3SelectModelResult>('/mm3/select-model', body, TIMEOUT_QUICK);
}

// ── Residency ──

/** POST /mm3/unload — free MM3 weights + KV cache. Idempotent and cheap when
 *  cold. Soft-fails: VRAM arbitration must never break a generation or a
 *  backend switch, so failures are reported, not thrown.
 *
 *  `keepArCache` frees VRAM without dropping the engine's host-side AR slot —
 *  use it for eviction the user did not ask for (a backend switch), never for
 *  an explicit "unload" they clicked. */
export async function mm3Unload(
  timeoutMs = TIMEOUT_QUICK,
  keepArCache = false,
): Promise<Mm3UnloadResult | null> {
  try {
    const res = await fetch(`${base()}/mm3/unload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(keepArCache ? { keep_ar_cache: true } : {}),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return null;
    return await res.json() as Mm3UnloadResult;
  } catch {
    return null;
  }
}
