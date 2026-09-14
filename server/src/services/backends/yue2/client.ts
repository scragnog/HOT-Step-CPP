// backends/yue2/client.ts — typed HTTP client for the engine's /yue2/* family
//
// YuE2 lives INSIDE the same ace-server process as ACE and MiniMax-Music3
// (docs/plans/yue2/06-engine-port-plan.md §7 Endpoints), so this client talks
// to the same base URL aceClient/minimax's client do — there is no second
// engine port. Only the family-scoped endpoints differ:
//
//   GET  /yue2/props            discovery + residency (see the BLOCKING warning)
//   POST /yue2/synth            production generation -> the COMMON shared job id
//   POST /yue2/warm             idempotent pre-load
//   POST /yue2/unload           free YuE2 VRAM (idempotent)
//   POST /yue2/select-model     variant/quant picker
//
// Progress / result / cancel for a YuE2 job go through the SAME shared job
// endpoints ACE and MM3 already use (GET /job?id=, GET /job?id=&result=1,
// POST /job?id=&cancel=1) via aceClient.pollJob / getJobResult / cancelJob —
// per the plan, YuE2 registers NO /yue2/job route of its own (06-engine-port-
// plan.md §7: "A YuE2-specific /yue2/job, /yue2/cancel, or /yue2/result route
// would be new, unnecessary surface — do not add one"). This module therefore
// does not duplicate that machinery; generate.ts calls aceClient directly.
//
// ASSUMPTION (flagged, not settled by either plan doc): end_reason /
// stage_end_reasons and the score.abc / semantic-id artifacts are described
// as "additive result-body keys riding alongside the shared Job struct's
// generic fields" (06-engine-port-plan.md §7). The shared GET /job status
// route today serializes exactly {status, phase, phase_step, phase_total}
// (hot-step-server.cpp:3579) — nothing for arbitrary extra keys yet. This
// client reads end_reason/stage_end_reasons/abc/semantic_ids optimistically
// off that SAME status JSON once status is 'done', typed as optional
// unknown-shaped extras, so it costs nothing today and picks up the fields
// the moment the engine side (queued, not YuE2-specific, per §7 "Queued
// engine-window changes" item 3) starts emitting them. If the engine team
// instead lands a different wire shape for the artifacts, only this file's
// yue2FinalDetail() needs to change.
//
// NOTE on the base URL: aceClient.ts snapshots config.aceServer.url at import
// time (a known latent staleness bug). This module reads it per call instead,
// matching minimax/client.ts's own workaround.

import { config } from '../../../config.js';

const base = () => config.aceServer.url;

/** GET /yue2/props may take an engine-side mutex the same way MM3's does —
 *  never wait on it. 2.5 s, then fall back to the last known-good manifest. */
const TIMEOUT_PROPS = 2_500;
const TIMEOUT_QUICK = 15_000;   // synth submit, warm, unload, select-model

// ── Wire shapes (mirrors engine/src/yue2/yue2-server.h, per the port plan) ──

export interface Yue2PropsFile {
  found?: boolean;
  path?: string;
  [k: string]: unknown;
}

/** One yue2-lm-<type>.gguf found on disk. v1 scope is bf16/f16/f32 dtype only
 *  (docs/plans/yue2/06-engine-port-plan.md §1: "quantized variants ... are
 *  explicitly out of scope for the converter in this pass") — `type` is the
 *  dtype token, not a quant ladder entry the way MM3's `quant` is. */
export interface Yue2VariantFile {
  type: string;       // 'bf16' | 'f16' | 'f32'
  filename: string;
  bytes: number;
}

export interface Yue2LmVariants {
  available: Yue2VariantFile[];
  selected: string;
  requested: string;
}

/** The NAR LoRA state block of GET /yue2/props (yue2-server.h's props handler).
 *
 *  `requested` is the adapter key the engine was last TOLD to use — the
 *  `<path>@<scale, 4dp>` spelling yue2_adapter_key() builds, `;`-joined for a
 *  stack. `merged` is what is actually baked into the resident weights, which
 *  is empty until the next warm/synth after a selection (the merge happens at
 *  load). A UI showing only one of them lies for exactly that window. */
export interface Yue2PropsAdapter {
  requested?: string;
  merged?: string;
  tensors?: number;
  /** Which half the merged adapters landed on: 'ar', 'nar', or 'ar+nar' when a
   *  stack covers both. AR and NAR share no weights, so both are merge targets
   *  and a bare tensor count reads identically whichever one it patched —
   *  which is exactly what loading the wrong file gets wrong. Absent from
   *  engines older than the AR-family loader. */
  family?: string;
  in_force?: boolean;
}

export interface Yue2Props {
  backend: string;                 // 'yue2'
  model: string;                   // 'YuE2'
  available: boolean;
  loaded: boolean;
  synth_ready: boolean;
  models_dir?: string;
  files?: {
    lm?: Yue2PropsFile;
    vae_standard?: Yue2PropsFile;
    vae_legacy?: Yue2PropsFile;
  };
  variants?: {
    lm?: Yue2LmVariants;
  };
  adapter?: Yue2PropsAdapter;
  vram?: Record<string, number>;
  errors?: string[];
  [k: string]: unknown;
}

export interface Yue2PropsResult {
  /** null only if /yue2/props has never answered since server start. */
  props: Yue2Props | null;
  /** true when `props` came from the last-known-good cache (probe timed out
   *  or failed), so callers can degrade honestly instead of reporting "down". */
  stale: boolean;
  error?: string;
  fetchedAt: number;
}

/** Per-role selection. '' (or omitted) = auto/best-first. Mirrors
 *  mm3SelectModel's shape at YuE2's much smaller (two-part) scale. */
export interface Yue2Selection {
  lm?: string;
  vae_variant?: 'standard' | 'legacy';
  /** A NAR LoRA to merge into the LM at load, as an ABSOLUTE path.
   *
   *  The engine opens the path AS GIVEN — yue2-adapter.h resolves it against
   *  nothing, unlike MM3's root-relative adapter references — so a relative
   *  path here means "relative to ace-server's working directory", which is
   *  not a promise anyone should make.
   *
   *  Three states, and the difference between the last two is the point:
   *    undefined (key omitted)  leave whatever the engine has alone
   *    null (or '')             explicitly clear a merged adapter
   *    '<abs path>'             merge this one
   *
   *  A caller that serialises its whole option struct would otherwise clear an
   *  adapter it never meant to touch, which is why yue2_parse_adapter_field
   *  tracks `given` separately from the value. */
  lm_adapter?: string | null;
  /** Merge strength, default 1.0 engine-side. Only sent alongside lm_adapter —
   *  on its own it would change the key the engine compares against and force
   *  a needless teardown. */
  lm_adapter_scale?: number;
}

/** Mirrors yue2_handle_select_model's actual response body
 *  (engine/src/yue2/yue2-server.h) — {selected, vae_variant, lm_type_want,
 *  lm_file, lm_found}, not mm3SelectModel's {changed, lm} shape. */
export interface Yue2SelectModelResult {
  selected: boolean;
  vae_variant: string;
  lm_type_want: string;
  lm_file: string;
  lm_found: boolean;
  /** The adapter key now requested, and the one actually merged right now —
   *  the latter is '' whenever the call itself tore the model down, because
   *  the merge happens on the next warm/synth. Optional: an engine built
   *  before the adapter field existed answers without them. */
  lm_adapter_want?: string;
  lm_adapter_merged?: string;
  error?: string;
}

export interface Yue2WarmRequest {
  lm?: string;
  vae_variant?: 'standard' | 'legacy';
}

export interface Yue2UnloadResult {
  unloaded: boolean;
  loaded: boolean;
  freed_bytes?: number;
  freed_mb?: number;
}

/** POST /yue2/synth request body — field names per
 *  docs/plans/yue2/06-engine-port-plan.md §7's request-fields list verbatim
 *  (style/caption, lyrics, cot, abc, seed, cfg_scale, ode_steps, ode_method,
 *  vae_variant, noise_source). `style` is the wire name for what the UI calls
 *  the caption/prompt field. */
export interface Yue2SynthRequest {
  style: string;
  lyrics?: string;
  cot: 'off' | 'melody' | 'full';
  /** Externally-supplied ABC — skips the ABC-stage model call. Not exposed
   *  by the UI in v1; carried here for completeness/API callers. */
  abc?: string;
  seed?: number;
  /** Optional override; unset = mode-dependent engine default
   *  (1.01 for cot=off, 1.0 otherwise, per protocol.py's own SongRequest). */
  cfg_scale?: number;
  ode_steps?: number;
  ode_method?: 'midpoint';
  vae_variant?: 'standard' | 'legacy';
  /** Validator-only per the plan; never sent by this backend's own mapping. */
  noise_source?: 'native' | 'fixture';
}

export interface Yue2SynthResponse {
  job_id: string;
  id: string;
  seed: number;
  seed_str?: string;
  duration?: number;
  ode_steps?: number;
  cfg_scale?: number;
  vae_variant?: string;
  instrumental?: boolean;
  [k: string]: unknown;
}

/** Named per-stage end reason, per §7's Job result contract. */
export type Yue2StageEndReason = 'eos' | 'limit_hit' | 'skipped';

/** The additive extras this client reads OPTIMISTICALLY off the shared GET
 *  /job status JSON once status is 'done' — see the ASSUMPTION note above. */
export interface Yue2FinalDetail {
  end_reason?: 'completed' | 'cancelled' | 'engine_failed' | 'limit_hit' | 'failed';
  stage_end_reasons?: Partial<Record<'plan' | 'semantic' | 'nar' | 'vae', Yue2StageEndReason>>;
  /** Human-readable decoded ABC text, if the request went through the
   *  ABC-plan stage (absent when the request supplied `abc` itself). */
  abc?: string;
  /** Raw semantic (codec) id array, if the engine returned it. */
  semantic_ids?: number[];
}

// ── Helpers ──

async function yue2ErrorMessage(res: Response, endpoint: string): Promise<string> {
  const body = await res.text().catch(() => '');
  let msg = body;
  try {
    const parsed = JSON.parse(body) as { error?: string };
    if (parsed?.error) msg = parsed.error;
  } catch { /* non-JSON body — use it raw */ }
  return `ace-server ${endpoint} failed (${res.status}): ${msg || 'unknown error'}`;
}

async function yue2Post<T>(endpoint: string, body: unknown, timeoutMs: number): Promise<T> {
  const res = await fetch(`${base()}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(await yue2ErrorMessage(res, `POST ${endpoint}`));
  return await res.json() as T;
}

// ── Props (with last-known-good cache) ──

let lastGoodProps: Yue2Props | null = null;
let lastGoodAt = 0;

/** GET /yue2/props with a short timeout and a last-known-good fallback — same
 *  contract as mm3Props(): a timeout here almost always means a YuE2
 *  generation is running and holding an engine-side lock, not that the
 *  backend is down. */
export async function yue2Props(timeoutMs = TIMEOUT_PROPS): Promise<Yue2PropsResult> {
  try {
    const res = await fetch(`${base()}/yue2/props`, { signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) throw new Error(await yue2ErrorMessage(res, 'GET /yue2/props'));
    const props = await res.json() as Yue2Props;
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
export function yue2PropsCached(): Yue2Props | null {
  return lastGoodProps;
}

// ── Residency ──

/** POST /yue2/warm — idempotent pre-load (06-engine-port-plan.md §7
 *  Residency: "calling it while already warm is a cheap no-op, never a
 *  double-allocate or a crash"). Soft-fails: warming is an optimization,
 *  never something that should throw out of a capability poll or a
 *  reconciliation pass. */
export async function yue2Warm(sel: Yue2WarmRequest = {}): Promise<{ warm: boolean; [k: string]: unknown } | null> {
  try {
    return await yue2Post('/yue2/warm', sel, TIMEOUT_QUICK);
  } catch {
    return null;
  }
}

/** POST /yue2/unload — free YuE2 weights (lm_resident + vae_resident, per
 *  yue2-model.h §1 — there is no AR-cache-keeping concept to preserve, unlike
 *  MM3's keepArCache). Soft-fails like mm3Unload: VRAM arbitration must never
 *  break a generation or a backend switch. */
export async function yue2Unload(timeoutMs = TIMEOUT_QUICK): Promise<Yue2UnloadResult | null> {
  try {
    const res = await fetch(`${base()}/yue2/unload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return null;
    return await res.json() as Yue2UnloadResult;
  } catch {
    return null;
  }
}

// ── Model selection ──

/** POST /yue2/select-model — choose the LM dtype and the VAE variant.
 *  '' means auto (engine picks best-first). Throws on an unknown
 *  type/variant (400) — a failed model switch must be visible, not
 *  swallowed, matching mm3SelectModel's contract.
 *
 *  Wire field is `lm_type`, not `lm` — matches the engine's
 *  yue2_handle_select_model (engine/src/yue2/yue2-server.h), which reads
 *  `req.lm_type` and otherwise silently keeps lm_type_given false, i.e. a
 *  picked LM would never actually apply.
 *
 *  `lm_adapter` is written into the body ONLY when the caller named one (see
 *  Yue2Selection): sending `null` on every call would clear a merged adapter
 *  every time anyone changed the VAE variant, and the omitted-vs-null
 *  distinction is the whole reason the engine parses the field the way it
 *  does. Changing the adapter set costs a full teardown + lazy reload
 *  engine-side, so a no-op repeat must stay a no-op. */
export async function yue2SelectModel(sel: Yue2Selection): Promise<Yue2SelectModelResult> {
  const body: Record<string, unknown> = {
    lm_type: sel.lm ?? '',
    vae_variant: sel.vae_variant ?? '',
  };
  if (sel.lm_adapter !== undefined) {
    // '' and null both clear; send null, which is the engine's documented
    // explicit-clear spelling and cannot be mistaken for "auto".
    body.lm_adapter = sel.lm_adapter === null || sel.lm_adapter === '' ? null : sel.lm_adapter;
    if (typeof sel.lm_adapter_scale === 'number' && Number.isFinite(sel.lm_adapter_scale)) {
      body.lm_adapter_scale = sel.lm_adapter_scale;
    }
  }
  return yue2Post<Yue2SelectModelResult>('/yue2/select-model', body, TIMEOUT_QUICK);
}

// ── Generation ──

/** POST /yue2/synth — submit a generation. Returns the COMMON engine job id
 *  from the shared job system (06-engine-port-plan.md §7: "not a YuE2-scoped
 *  id") — progress/result/cancel all go through aceClient's existing
 *  pollJob/getJobResult/cancelJob against that id, unchanged. */
export async function yue2Synth(req: Yue2SynthRequest): Promise<Yue2SynthResponse> {
  const res = await yue2Post<Yue2SynthResponse>('/yue2/synth', req, TIMEOUT_QUICK);
  // The engine answers {"id": ...} (the ACE /synth spelling); MM3's engine sends
  // both spellings. Normalise so callers can rely on job_id.
  if (!res.job_id && res.id) res.job_id = res.id;
  if (!res.job_id) throw new Error('YuE2 /yue2/synth returned no job id');
  return res;
}

/** Best-effort read of the additive end_reason/stage_end_reasons/artifact
 *  extras off the shared GET /job status JSON (see the ASSUMPTION note at the
 *  top of this file). Never throws: an engine build that doesn't populate
 *  these fields yet simply yields an empty object, and every caller already
 *  treats every field here as optional. */
export async function yue2FinalDetail(jobId: string): Promise<Yue2FinalDetail> {
  try {
    const res = await fetch(`${base()}/job?id=${encodeURIComponent(jobId)}`, {
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) return {};
    return await res.json() as Yue2FinalDetail;
  } catch {
    return {};
  }
}
