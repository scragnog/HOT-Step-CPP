// aceClient.ts — HTTP client for acestep.cpp's ace-server API
//
// Wraps all ace-server endpoints with typed methods.
// Used by the generation orchestrator and model routes.
//
// IMPORTANT: ace-server uses single-threaded httplib. During heavy compute
// (DiT generation, adapter merge, VAE decode) it cannot respond to HTTP
// requests. All fetch calls need generous timeouts to survive these stalls.

import { config } from '../config.js';

const BASE = config.aceServer.url;

// Timeouts (ms) — ace-server is single-threaded httplib, so during heavy
// compute (DiT steps, adapter merge) it can't respond. These must be
// generous enough to survive the longest possible stall.
const TIMEOUT_QUICK  =  15_000;   // health checks, props, job submit
const TIMEOUT_POLL   =  30_000;   // job polling — fail fast, let watchdog decide on stalls
const TIMEOUT_RESULT = 300_000;   // fetching large audio results (encode + transfer)
const TIMEOUT_SA3    = 1_800_000; // /sa3-refine — first call may build a TensorRT engine (many minutes)

/** Props response from GET /props */
/** Sample format for a post-processing endpoint's WAV response (`out_fmt`).
 *
 *  Omitting it gives 16-bit, which is what every UI-facing caller wants. The
 *  post-processing chain asks for 'f32' because it hands each result straight
 *  to the next stage, and a 16-bit hop there re-quantises and bakes in
 *  whatever the previous stage clipped.
 *  See docs/plans/2026-08-28-post-processing-gain-staging.md. */
export type WavOutFormat = 's16' | 's24' | 'f32';

export interface AceProps {
  models: {
    lm: string[];
    embedding: string[];
    dit: string[];
    vae: string[];
  };
  adapters: string[];
  /** Planner-LM adapters from adapters/lm/ (local HOT-Step feature) */
  lm_adapters?: string[];
  cli: {
    max_batch: number;
    mp3_bitrate: number;
  };
  default: Record<string, unknown>;
}

/** AceRequest — matches acestep.cpp's request JSON format */
export interface AceRequest {
  caption: string;
  lyrics?: string;
  bpm?: number;
  duration?: number;
  keyscale?: string;
  timesignature?: string;
  vocal_language?: string;
  seed?: number;
  /** LM-phase sampling seed (caption/lyrics/audio-codes). Independent from
   *  `seed`, which drives DiT synthesis. -1 or omitted lets the engine pick. */
  lm_seed?: number;
  lm_batch_size?: number;
  synth_batch_size?: number;
  lm_temperature?: number;
  lm_cfg_scale?: number;
  lm_cfg_cutoff_ratio?: number; // LM CFG step scheduling: 1.0 = full CFG, 0.5 = CFG for first 50% of tokens
  lm_top_p?: number;
  /** Windowed repetition penalty on audio-code sampling (1.0 = off) */
  lm_rep_penalty?: number;
  lm_rep_window?: number;
  /** How lm_rep_penalty is applied. 'presence' (default) penalises every
   *  distinct code in the window once; 'frequency' scales by occurrence count;
   *  'dry' penalises only codes that would extend a verbatim recent cycle. */
  lm_rep_mode?: 'presence' | 'frequency' | 'dry';
  /** 'dry' only: exponential growth per matched code beyond lm_dry_min_len */
  lm_dry_base?: number;
  /** 'dry' only: matched codes required before any penalty applies */
  lm_dry_min_len?: number;
  lm_top_k?: number;
  lm_negative_prompt?: string;
  negative_prompt?: string;
  use_cot_caption?: boolean;
  audio_codes?: string;
  inference_steps?: number;
  guidance_scale?: number;
  shift?: number;
  /** Self-attention sliding-window override for the DiT's 16 windowed layers,
   *  in tokens (tokens run at 12.5 Hz). -1/undefined = the model's own value
   *  (128 = ±10.2 s), 0 = no window at all (full attention on all 32 layers). */
  dit_sliding_window?: number;
  audio_cover_strength?: number;
  cover_noise_strength?: number;
  cover_noise_method?: string;
  repainting_start?: number;
  repainting_end?: number;
  seed_strength?: number;
  evict_lm?: boolean;
  vae_chunk?: number;
  batch_cfg?: number;
  task_type?: string;
  track?: string;
  infer_method?: string;
  scheduler?: string;       // 'linear' | 'ddim_uniform' | 'sgm_uniform' | etc.
  guidance_mode?: string;   // 'apg' | 'cfg' | etc.
  peak_clip?: number;
  // Server routing fields
  synth_model?: string;
  lm_model?: string;
  /** Planner-LM runtime LoRA (local HOT-Step feature) */
  lm_adapter?: string;
  lm_adapter_scale?: number;
  vae_model?: string;
  emb_model?: string;
  adapter?: string;
  adapter_scale?: number;
  /** Multi-adapter stack. When present and non-empty, supersedes the single
   *  `adapter`/`adapter_scale`: every entry is applied with its own scale
   *  (merged sequentially, or summed in runtime mode). Each `name` is a registry
   *  adapter id (or absolute path) resolved by the engine. */
  adapters?: { name: string; scale: number; gain_curve?: number[]; gain_domain?: 'steps' | 't' }[];
  /** Per-section adapter masking (regional LoRA). Ordered per lyric section; each
   *  entry gives the effective per-adapter scale for that section (indexed to
   *  `adapters`) and a relative size hint. Runtime mode only. */
  adapter_sections?: { weights: number[]; size: number }[];
  /** Per-section masking: fraction of steps before deriving section boundaries from
   *  cross-attention alignment (earlier = identity locks to sections sooner). */
  adapter_section_align_at?: number;
  /** Per-section masking: 0..1 regional self-attention isolation (penalise attention
   *  across section boundaries so sections don't inherit the first section's voice). */
  adapter_section_isolation?: number;
  adapter_group_scales?: {
    self_attn: number;
    cross_attn: number;
    mlp: number;
    cond_embed: number;
    time_embed: number;
    proj_in: number;
  };
  adapter_mode?: string;  // "merge" (default, F32 promoted), "runtime", or "runtime_lowrank" (factor apply, lowest VRAM)
  /** Runtime adapter delta VRAM precision: "bf16" (full), "q8_0" (~½), "q4_k" (~¼).
   *  Quantizes precomputed deltas in VRAM at load; no disk change. Runtime mode only. */
  adapter_runtime_quant?: string;
  /** Merge (low VRAM): re-encode merged weights to the base's native quant instead of
   *  F32 promotion (~¼ the merged-DiT VRAM on a Q8 base). Merge mode only. */
  adapter_merge_lowvram?: boolean;
  // Basin re-base: nudge adapted weights toward the base the adapter was trained
  // on (rebase_source = DiT model name) by rebase_beta*(S - T) before merging.
  rebase_source?: string;
  rebase_beta?: number;
  // Solver sub-parameters
  stork_substeps?: number;
  beat_stability?: number;
  frequency_damping?: number;
  temporal_smoothing?: number;
  // Guidance sub-parameters
  apg_momentum?: number;
  apg_norm_threshold?: number;
  // DCW (Differential Correction in Wavelet domain)
  dcw_enabled?: boolean;
  dcw_mode?: string;         // 'pix' | 'low' | 'high' | 'double'
  dcw_scaler?: number;
  dcw_high_scaler?: number;
  // Latent post-processing (applied after DiT, before VAE decode)
  latent_shift?: number;       // 0.0 = no bias
  latent_rescale?: number;     // 1.0 = no scaling
  custom_timesteps?: string;   // CSV of descending floats, overrides scheduler
  cfg_cutoff_ratio?: number;   // CFG step scheduling: 1.0 = full CFG, 0.5 = 50% CFG then cond-only
  cache_ratio?: number;        // Step-level velocity caching: 0.0 = off, 0.5 = skip ~50% of passes
  // Post-VAE spectral denoiser (HOT-Step)
  denoise_strength?: number;   // 0.0 = off, 1.0 = max suppression
  denoise_smoothing?: number;  // 0.0 = sharp gate, 1.0 = very smooth
  denoise_mix?: number;        // 0.0 = all dry, 1.0 = all denoised
  // LSS: Latent Spectral Suppressor (MDMAchine) — pre-VAE latent channel gate
  lss_strength?: number;       // 0.0 = off, attenuation floor is 1-strength
  lss_var_thresh?: number;     // relative variance threshold (default 0.15)
  lss_dc_remove?: boolean;     // per-channel DC removal while LSS active
  // PP-VAE re-encode (spectral cleanup via post-processing VAE)
  pp_vae_reencode?: boolean;
  // LRC timestamp generation (synchronized lyrics)
  get_lrc?: boolean;
  // Lua plugin dynamic parameters
  plugin_params?: Record<string, string | number | boolean>;
  // Postprocess plugin: name of the Lua postprocess plugin to use for VAE decode
  postprocess_plugin?: string;
  // VAE backend selection: true = ONNX Runtime (+TensorRT), false/undefined = GGML (default)
  use_ort_vae?: boolean;
  // Streaming pipeline (DEMON-style ring buffer)
  stream_mode?: boolean;       // true = route through streaming pipeline
  stream_depth?: number;       // ring buffer depth (default 8)
  stream_chunk_dir?: string;   // directory for preview WAV files
}

/** Job status from ace-server */
export interface AceJobStatus {
  status: 'running' | 'done' | 'failed' | 'cancelled';
  /** Fine-grained engine phase. Optional for back-compat with older
   *  ace-server builds that don't populate it. */
  phase?: AceJobPhase;
  phase_step?: number;
  phase_total?: number;
}

/** Fine-grained engine phase — matches JobPhase in hot-step-server.cpp.
 *  Lowercase snake_case mirrors job_phase_str(). */
export type AceJobPhase =
  | 'queued'
  | 'loading_text_enc'
  | 'encoding_text'
  | 'loading_cond_enc'
  | 'encoding_cond'
  | 'loading_dit'
  | 'loading_adapter'
  | 'adapter_precompute'
  | 'dit_inference'
  | 'loading_vae'
  | 'vae_decode'
  | 'encoding_output'
  | 'done'
  | 'failed'
  | 'cancelled';

/** One row from GET /jobs. */
export interface AceJobsListEntry {
  id: string;
  status: 'running' | 'done' | 'failed' | 'cancelled';
  phase: AceJobPhase;
  phase_step: number;
  phase_total: number;
}

/** Body shape for POST /warm. */
export interface AceWarmRequest {
  dit: string;
  vae?: string;
  adapter?: string;
  adapter_scale?: number;
}

/** Plugin parameter schema from Lua plugin metadata */
export interface PluginParamSchema {
  key: string;
  type: 'slider' | 'select' | 'toggle' | 'text';
  label: string;
  hint?: string;
  transform?: string;
  // slider
  default?: number | string | boolean;
  min?: number;
  max?: number;
  step?: number;
  // select
  options?: { value: string; label: string }[];
  // conditional visibility
  visible_when?: { key: string; equals: string };
}

/** Plugin metadata from Lua plugin files */
export interface PluginInfo {
  name: string;
  display: string;
  description?: string;
  accent?: string;
  // solver-specific
  nfe?: number;
  order?: number;
  needs_model?: boolean;
  stateful?: boolean;
  stochastic?: boolean;
  params: PluginParamSchema[];
}

export interface PluginRegistry {
  solvers: PluginInfo[];
  schedulers: PluginInfo[];
  guidance: PluginInfo[];
  postprocess: PluginInfo[];
}

async function aceGet(path: string, timeoutMs = TIMEOUT_QUICK): Promise<Response> {
  const res = await fetch(`${BASE}${path}`, {
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => 'Unknown error');
    throw new Error(`ace-server ${path} failed (${res.status}): ${body}`);
  }
  return res;
}

async function acePost(path: string, body?: unknown, contentType = 'application/json', timeoutMs = TIMEOUT_QUICK): Promise<Response> {
  const headers: Record<string, string> = {};
  let reqBody: string | undefined;

  if (body !== undefined) {
    headers['Content-Type'] = contentType;
    reqBody = typeof body === 'string' ? body : JSON.stringify(body);
  }

  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers,
    body: reqBody,
    signal: AbortSignal.timeout(timeoutMs),
  });

  if (!res.ok) {
    const errBody = await res.text().catch(() => 'Unknown error');
    throw new Error(`ace-server POST ${path} failed (${res.status}): ${errBody}`);
  }
  return res;
}

export const aceClient = {
  /** GET /health — check if ace-server is alive */
  async health(): Promise<{ status: string }> {
    const res = await aceGet('/health');
    return res.json();
  },

  /** GET /props — available models, config, defaults */
  async props(): Promise<AceProps> {
    const res = await aceGet('/props');
    return res.json();
  },

  /** GET /plugins — dynamic Lua plugin registry (solvers, schedulers, guidance) */
  async plugins(): Promise<PluginRegistry> {
    const res = await aceGet('/plugins');
    return res.json();
  },

  /** POST /warm — pre-load a DiT + VAE + adapter combo so the next /synth with
   *  the same key short-circuits the cold-start adapter precompute. Requires the
   *  engine to be in keep-loaded mode (--keep-loaded or a prior ?keep_loaded=1);
   *  under STRICT the engine returns {warm:false} since modules evict instantly.
   *  Returns the engine job id — poll via pollJob until status=done. */
  async warm(request: AceWarmRequest, keepLoaded = true): Promise<string> {
    const params = new URLSearchParams();
    if (keepLoaded) params.set('keep_loaded', '1');
    const qs = params.toString();
    const path = qs ? `/warm?${qs}` : '/warm';
    const res = await acePost(path, request);
    const data = await res.json() as { id: string };
    return data.id;
  },

  /** GET /jobs — enumerate every job currently in the engine's in-memory job
   *  table. Used to reconcile a still-running engine job after a client
   *  disconnected mid-poll. */
  async listJobs(): Promise<AceJobsListEntry[]> {
    const res = await aceGet('/jobs');
    return res.json();
  },

  /** POST /lm — submit LM generation job, returns job ID.
   *  mode: 'inspire' (Phase 1 only, no codes) | 'format' (reformat, no codes)
   *  keepLoaded: flip store to EVICT_NEVER before LM loads, so the LM stays
   *  cached for subsequent gens instead of being freed under STRICT. */
  async submitLm(request: AceRequest, mode?: 'inspire' | 'format', keepLoaded = false): Promise<string> {
    const body = mode ? { ...request, lm_mode: mode } : request;
    const params = new URLSearchParams();
    if (keepLoaded) params.set('keep_loaded', '1');
    const qs = params.toString();
    const path = qs ? `/lm?${qs}` : '/lm';
    const res = await acePost(path, body);
    const data = await res.json() as { id: string };
    return data.id;
  },

  /** POST /synth — submit synth job, returns job ID.
   *  format: 'wav16'|'wav24'|'wav32'|'mp3' — output format (default: wav16 for lossless) */
  async submitSynth(request: AceRequest | AceRequest[], format: string = 'wav16', keepLoaded = false): Promise<string> {
    const params = new URLSearchParams();
    if (format !== 'mp3') params.set('format', format);
    if (keepLoaded) params.set('keep_loaded', '1');
    const qs = params.toString();
    const path = qs ? `/synth?${qs}` : '/synth';
    const res = await acePost(path, request);
    const data = await res.json() as { id: string };
    return data.id;
  },

  /**
   * POST /synth with multipart — for cover/repaint modes with source audio
   * Sends request JSON + audio file(s) as multipart/form-data
   */
  async submitSynthMultipart(
    request: AceRequest | AceRequest[],
    srcAudio?: Buffer,
    refAudio?: Buffer,
    srcLatents?: Buffer,
    refLatents?: Buffer,
    format: string = 'wav16',
    keepLoaded = false,
    seedLatents?: Buffer,
  ): Promise<string> {
    const params = new URLSearchParams();
    if (format !== 'mp3') params.set('format', format);
    if (keepLoaded) params.set('keep_loaded', '1');
    const qs = params.toString();
    const path = qs ? `/synth?${qs}` : '/synth';
    const boundary = '----HotStepBoundary' + Date.now();

    const parts: Buffer[] = [];
    const addPart = (name: string, content: Buffer, contentType: string, filename?: string) => {
      let header = `--${boundary}\r\nContent-Disposition: form-data; name="${name}"`;
      if (filename) header += `; filename="${filename}"`;
      header += `\r\nContent-Type: ${contentType}\r\n\r\n`;
      parts.push(Buffer.from(header));
      parts.push(content);
      parts.push(Buffer.from('\r\n'));
    };

    // Request JSON part — ace-server multipart expects a single JSON object
    // (uses request_parse_json, not request_parse_json_array)
    const singleReq = Array.isArray(request) ? request[0] : request;
    const reqJson = JSON.stringify(singleReq);
    addPart('request', Buffer.from(reqJson), 'application/json');

    // Source audio part
    if (srcAudio) {
      addPart('audio', srcAudio, 'audio/wav', 'source.wav');
    }

    // Reference audio part
    if (refAudio) {
      addPart('ref_audio', refAudio, 'audio/wav', 'reference.wav');
    }

    // Source latents part (raw float32 — replaces VAE encode of source audio)
    if (srcLatents) {
      addPart('src_latents', srcLatents, 'application/octet-stream', 'source.latent');
    }

    // Reference latents part (raw float32 — replaces VAE encode of timbre ref)
    if (refLatents) {
      addPart('ref_latents', refLatents, 'application/octet-stream', 'reference.latent');
    }

    // Seed latents part (raw float32 — structural seed for repeated sections)
    if (seedLatents) {
      addPart('seed_latents', seedLatents, 'application/octet-stream', 'seed.latent');
    }

    parts.push(Buffer.from(`--${boundary}--\r\n`));
    const body = Buffer.concat(parts);

    const res = await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
      },
      body,
      signal: AbortSignal.timeout(TIMEOUT_RESULT),
    });

    if (!res.ok) {
      const errBody = await res.text().catch(() => 'Unknown error');
      throw new Error(`ace-server POST ${path} multipart failed (${res.status}): ${errBody}`);
    }

    const data = await res.json() as { id: string };
    return data.id;
  },

  /** POST /understand — submit understand job, returns job ID.
   *  `params` (optional) is serialised into the multipart `request` part;
   *  the engine parses it with the same request_parse_json used by /synth,
   *  so any AceRequest field is legal. Engine defaults for understand are
   *  lm_temperature 0.3 / lm_top_p 1.0 and are only overridden by what we send. */
  async submitUnderstand(audioBuffer: Buffer, params?: Partial<AceRequest>): Promise<string> {
    const boundary = '----HotStepBoundary' + Date.now();

    const parts: Buffer[] = [];
    const addPart = (name: string, content: Buffer, contentType: string, filename?: string) => {
      let header = `--${boundary}\r\nContent-Disposition: form-data; name="${name}"`;
      if (filename) header += `; filename="${filename}"`;
      header += `\r\nContent-Type: ${contentType}\r\n\r\n`;
      parts.push(Buffer.from(header));
      parts.push(content);
      parts.push(Buffer.from('\r\n'));
    };

    // Request JSON part — only when there is something to say, so the legacy
    // audio-only call shape stays byte-identical.
    if (params && Object.keys(params).length > 0) {
      addPart('request', Buffer.from(JSON.stringify(params)), 'application/json');
    }
    addPart('audio', audioBuffer, 'audio/wav', 'input.wav');

    parts.push(Buffer.from(`--${boundary}--\r\n`));
    const body = Buffer.concat(parts);

    const res = await fetch(`${BASE}/understand`, {
      method: 'POST',
      headers: { 'Content-Type': `multipart/form-data; boundary=${boundary}` },
      body,
      signal: AbortSignal.timeout(TIMEOUT_RESULT),
    });

    if (!res.ok) {
      const errBody = await res.text().catch(() => 'Unknown error');
      throw new Error(`ace-server POST /understand failed (${res.status}): ${errBody}`);
    }

    const data = await res.json() as { id: string };
    return data.id;
  },

  /** GET /job?id=N — poll job status.
   *  Uses TIMEOUT_POLL because ace-server is single-threaded and may be
   *  mid-DiT-step when we poll, so the response can stall for several seconds. */
  async pollJob(jobId: string): Promise<AceJobStatus> {
    const res = await aceGet(`/job?id=${jobId}`, TIMEOUT_POLL);
    return res.json();
  },

  /** GET /job?id=N&result=1 — fetch completed job result.
   *  Uses TIMEOUT_RESULT because the response contains the full MP3/WAV audio. */
  async getJobResult(jobId: string): Promise<Response> {
    return fetch(`${BASE}/job?id=${jobId}&result=1`, {
      signal: AbortSignal.timeout(TIMEOUT_RESULT),
    });
  },

  /** GET /job?id=N&latent=1 — fetch captured post-DiT latent (raw float32).
   *  Returns null if no latent was captured (non-cover tasks, cancelled, etc). */
  async getJobLatent(jobId: string): Promise<Buffer | null> {
    try {
      const res = await fetch(`${BASE}/job?id=${jobId}&latent=1`, {
        signal: AbortSignal.timeout(TIMEOUT_RESULT),
      });
      if (!res.ok) return null;
      const buf = await res.arrayBuffer();
      return buf.byteLength > 0 ? Buffer.from(buf) : null;
    } catch {
      return null;
    }
  },

  /** POST /job?id=N&cancel=1 — cancel a running job */
  async cancelJob(jobId: string): Promise<void> {
    await fetch(`${BASE}/job?id=${jobId}&cancel=1`, {
      method: 'POST',
      signal: AbortSignal.timeout(TIMEOUT_POLL),
    });
  },

  /** Check if ace-server is reachable */
  async isReachable(): Promise<boolean> {
    try {
      await this.health();
      return true;
    } catch {
      return false;
    }
  },

  /** POST /spectral-lifter — synchronous C++ Spectral Lifter processing.
   *  Sends WAV audio body with SL params as query string.
   *  Returns processed WAV buffer. */
  async submitSpectralLifter(
    wavBuffer: Buffer,
    params: {
      denoise_strength?: number;
      noise_floor?: number;
      hf_mix?: number;
      transient_boost?: number;
      shimmer_reduction?: number;
      /** Response sample format. Omitted = s16, which is what every UI-facing
       *  caller wants. The post-processing chain passes 'f32' so its
       *  intermediate files are not re-quantised at every hop. */
      outFmt?: WavOutFormat;
    },
  ): Promise<Buffer> {
    const qs = new URLSearchParams();
    if (params.denoise_strength !== undefined) qs.set('denoise_strength', String(params.denoise_strength));
    if (params.noise_floor !== undefined) qs.set('noise_floor', String(params.noise_floor));
    if (params.hf_mix !== undefined) qs.set('hf_mix', String(params.hf_mix));
    if (params.transient_boost !== undefined) qs.set('transient_boost', String(params.transient_boost));
    if (params.shimmer_reduction !== undefined) qs.set('shimmer_reduction', String(params.shimmer_reduction));
    if (params.outFmt) qs.set('out_fmt', params.outFmt);
    const qsStr = qs.toString();
    const path = qsStr ? `/spectral-lifter?${qsStr}` : '/spectral-lifter';

    const res = await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: wavBuffer,
      signal: AbortSignal.timeout(TIMEOUT_RESULT),
    });
    if (!res.ok) {
      const errBody = await res.text().catch(() => 'Unknown error');
      throw new Error(`ace-server POST /spectral-lifter failed (${res.status}): ${errBody}`);
    }
    const arrayBuf = await res.arrayBuffer();
    return Buffer.from(arrayBuf);
  },

  /** POST /pp-vae-reencode — synchronous PP-VAE re-encode processing.
   *  Sends WAV audio body. Returns processed WAV buffer with RMS-matched gain.
   *  blend: 0.0 = fully PP-VAE, 1.0 = fully original (wet/dry mix). */
  async submitPpVaeReencode(wavBuffer: Buffer, blend = 0.0, useOnnx?: boolean,
                            outFmt?: WavOutFormat): Promise<Buffer> {
    const params = new URLSearchParams();
    if (blend > 0) params.set('blend', blend.toFixed(3));
    if (useOnnx === true) params.set('backend', 'onnx');
    else if (useOnnx === false) params.set('backend', 'gguf');
    if (outFmt) params.set('out_fmt', outFmt);
    const qs = params.toString();
    const url = qs ? `${BASE}/pp-vae-reencode?${qs}` : `${BASE}/pp-vae-reencode`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: wavBuffer,
      signal: AbortSignal.timeout(TIMEOUT_RESULT),
    });
    if (!res.ok) {
      const errBody = await res.text().catch(() => 'Unknown error');
      throw new Error(`ace-server POST /pp-vae-reencode failed (${res.status}): ${errBody}`);
    }
    const arrayBuf = await res.arrayBuffer();
    return Buffer.from(arrayBuf);
  },

  /** POST /sa3-refine — synchronous SA3 (Stable Audio 3) SDEdit refine.
   *  Sends WAV audio body; the pre-tokenized T5Gemma prompt and sampler
   *  options ride in the query string (the engine cannot tokenize
   *  SentencePiece itself — see sa3Tokenizer.ts).
   *  Returns processed WAV at the input sample rate.
   *  NOTE: the first-ever call may take many minutes (TensorRT engine build),
   *  hence the dedicated 30-minute timeout. */
  async submitSa3Refine(
    wavBuffer: Buffer,
    opts: {
      tokens: number[];       // exactly 256 padded T5Gemma token ids
      nTokens: number;        // real (non-pad) token count
      strength?: number;      // 0..1 init noise level (engine default 0.3)
      steps?: number;         // sampler steps (engine default 8)
      sampler?: 'pingpong' | 'euler';
      seed?: number;          // uint64 RNG seed (engine default: random)
      rmsMatch?: boolean;     // match output RMS to input (engine default true)
      outSr?: number;         // output sample rate (engine default: input rate)
      /** Engine backend: 'onnx' (ONNX Runtime/TensorRT) or 'gguf' (GGML —
       *  CUDA/Vulkan/CPU). 'auto'/undefined lets the engine pick. */
      backend?: 'auto' | 'onnx' | 'gguf';
      /** StableStep DoRA adapters (models/sa3-adapters/<name>.gguf), merged
       *  into the SA3 DiT at load. Forces the GGUF backend engine-side. */
      adapters?: Array<{ name: string; scale: number }>;
      /** Windowed envelope match: output follows the source's short-term RMS
       *  envelope (timbre from the refine, dynamics from the source). */
      envMatch?: boolean;
      /** Wet/dry blend with the source: 0 = pure source, 1 = pure refined.
       *  Mutually exclusive with the band splice (mix wins engine-side). */
      mix?: number;
      /** Spectral band splice: source below the crossover, refined above. */
      bandBlend?: boolean;
      bandFreq?: number;   // crossover center Hz (engine default 250)
      bandWidth?: number;  // transition width Hz (engine default 200)
      /** Lua plugin routing for the SA3 sampler loop — the same registry the
       *  ACE /synth dropdowns use. Orthogonal to the adapters/env/mix/band
       *  fields above: those change load-time weights and post-hoc audio
       *  blending, these change the run-time denoise. Absent = the original
       *  pingpong/euler path, bit-identical.
       *  NOTE: naming a solver REPLACES the pingpong renoise, and SA3 has no
       *  CFG uncond, so APG-family guidance is a pass-through (sa3-refine.h). */
      solver?: string;
      scheduler?: string;
      guidanceMode?: string;
      guidanceScale?: number;                          // only sent when > 1
      pluginParams?: Record<string, string | number>;  // "<plugin>:<key>" -> value
      /** Response sample format. Omitted = s16. */
      outFmt?: WavOutFormat;
    },
  ): Promise<Buffer> {
    const params = new URLSearchParams();
    params.set('tokens', opts.tokens.join(','));
    params.set('n_tokens', String(opts.nTokens));
    if (opts.backend && opts.backend !== 'auto') params.set('backend', opts.backend);
    if (opts.adapters && opts.adapters.length > 0) {
      params.set('adapters', opts.adapters.map(a => `${a.name}:${a.scale}`).join(','));
    }
    if (opts.envMatch) params.set('env_match', '1');
    if (opts.mix !== undefined) params.set('mix', String(opts.mix));
    if (opts.bandBlend) {
      params.set('band_blend', '1');
      if (opts.bandFreq !== undefined) params.set('band_freq', String(opts.bandFreq));
      if (opts.bandWidth !== undefined) params.set('band_width', String(opts.bandWidth));
    }
    if (opts.strength !== undefined) params.set('strength', String(opts.strength));
    if (opts.steps !== undefined) params.set('steps', String(opts.steps));
    if (opts.sampler) params.set('sampler', opts.sampler);
    if (opts.seed !== undefined) params.set('seed', String(opts.seed));
    if (opts.rmsMatch !== undefined) params.set('rms_match', opts.rmsMatch ? '1' : '0');
    if (opts.outSr !== undefined) params.set('out_sr', String(opts.outSr));
    // Lua plugin routing. Empty strings are omitted rather than sent — the
    // engine treats a missing key and an explicit '' identically, and omitting
    // keeps the query string short.
    if (opts.solver) params.set('solver', opts.solver);
    if (opts.scheduler) params.set('scheduler', opts.scheduler);
    if (opts.guidanceMode) params.set('guidance_mode', opts.guidanceMode);
    if (opts.guidanceScale !== undefined && opts.guidanceScale > 1) {
      params.set('guidance_scale', String(opts.guidanceScale));
    }
    if (opts.pluginParams && Object.keys(opts.pluginParams).length > 0) {
      params.set('plugin_params', JSON.stringify(opts.pluginParams));
    }
    if (opts.outFmt) params.set('out_fmt', opts.outFmt);

    const res = await fetch(`${BASE}/sa3-refine?${params.toString()}`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: wavBuffer,
      signal: AbortSignal.timeout(TIMEOUT_SA3),
    });
    if (!res.ok) {
      const errBody = await res.text().catch(() => 'Unknown error');
      throw new Error(`ace-server POST /sa3-refine failed (${res.status}): ${errBody}`);
    }
    const arrayBuf = await res.arrayBuffer();
    return Buffer.from(arrayBuf);
  },

  /** POST /supersep/separate?level=N — start async stem separation.
   *  Body: WAV/MP3 audio. Returns the SuperSep job id.
   *  level: 0=BASIC (6 stems), 1=VOCAL_SPLIT, 2=FULL, 3=MAXIMUM,
   *         4=VOCALS_ONLY (BS-RoFormer 2-stem: Vocals incl. backing + complement Instrumental). */
  async submitSuperSepSeparate(audioBuffer: Buffer, level = 0): Promise<string> {
    const res = await fetch(`${BASE}/supersep/separate?level=${level}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: audioBuffer,
      signal: AbortSignal.timeout(TIMEOUT_RESULT),
    });
    if (!res.ok) {
      const errBody = await res.text().catch(() => 'Unknown error');
      throw new Error(`ace-server POST /supersep/separate failed (${res.status}): ${errBody}`);
    }
    const data = await res.json() as { id: string };
    return data.id;
  },

  /** GET /supersep/progress?id=... — poll a SuperSep separation job. */
  async superSepProgress(jobId: string): Promise<{
    status: string; progress: number; message: string; error?: string; n_stems?: number;
  }> {
    const res = await aceGet(`/supersep/progress?id=${jobId}`, TIMEOUT_POLL);
    return res.json();
  },

  /** GET /supersep/result?id=... — stem list metadata for a completed job. */
  async superSepResult(jobId: string): Promise<{
    id: string;
    stems: Array<{
      name: string; category: string; stem_type: string;
      n_frames: number; stage: number; index: number; hidden: boolean;
    }>;
  }> {
    const res = await aceGet(`/supersep/result?id=${jobId}`, TIMEOUT_POLL);
    return res.json();
  },

  /** GET /supersep/serve?id=...&stem=N — download one stem as a 44.1 kHz WAV. */
  async superSepStem(jobId: string, index: number, outFmt?: WavOutFormat): Promise<Buffer> {
    const fmtQs = outFmt ? `&out_fmt=${outFmt}` : '';
    const res = await fetch(`${BASE}/supersep/serve?id=${jobId}&stem=${index}${fmtQs}`, {
      signal: AbortSignal.timeout(TIMEOUT_RESULT),
    });
    if (!res.ok) {
      const errBody = await res.text().catch(() => 'Unknown error');
      throw new Error(`ace-server GET /supersep/serve failed (${res.status}): ${errBody}`);
    }
    const arrayBuf = await res.arrayBuffer();
    return Buffer.from(arrayBuf);
  },

  /** POST /supersep/recombine — mix a completed job's stems with per-stem
   *  volume/mute controls. Returns a 48 kHz 16-bit stereo WAV. */
  async superSepRecombine(
    jobId: string,
    stems: Array<{ index: number; volume?: number; muted?: boolean }>,
    outFmt?: WavOutFormat,
  ): Promise<Buffer> {
    const fmtQs = outFmt ? `?out_fmt=${outFmt}` : '';
    const res = await fetch(`${BASE}/supersep/recombine${fmtQs}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: jobId, stems }),
      signal: AbortSignal.timeout(TIMEOUT_RESULT),
    });
    if (!res.ok) {
      const errBody = await res.text().catch(() => 'Unknown error');
      throw new Error(`ace-server POST /supersep/recombine failed (${res.status}): ${errBody}`);
    }
    const arrayBuf = await res.arrayBuffer();
    return Buffer.from(arrayBuf);
  },

  /** POST /vae with multipart — VAE encode: sends audio, returns raw f32 latent bytes.
   *  Polls until the engine job completes, then fetches the result.
   *  Returns raw f32 [T*64] latent buffer.
   *  @param vaeModel Optional VAE model name — sent as {"vae":"..."} in request part. */
  async vaeEncode(audioBuffer: Buffer, vaeModel?: string): Promise<Buffer> {
    const boundary = '----HotStepBoundary' + Date.now();
    const parts: Buffer[] = [];
    const addPart = (name: string, content: Buffer, contentType: string, filename?: string) => {
      let header = `--${boundary}\r\nContent-Disposition: form-data; name="${name}"`;
      if (filename) header += `; filename="${filename}"`;
      header += `\r\nContent-Type: ${contentType}\r\n\r\n`;
      parts.push(Buffer.from(header));
      parts.push(content);
      parts.push(Buffer.from('\r\n'));
    };

    // Request JSON part — tells the engine which VAE to use for encoding.
    // The C++ /vae handler reads "vae" (not "vae_model") via request_parse_json.
    if (vaeModel) {
      addPart('request', Buffer.from(JSON.stringify({ vae: vaeModel })), 'application/json');
    }

    // Audio part
    addPart('audio', audioBuffer, 'audio/wav', 'input.wav');

    parts.push(Buffer.from(`--${boundary}--\r\n`));
    const body = Buffer.concat(parts);

    const res = await fetch(`${BASE}/vae`, {
      method: 'POST',
      headers: {
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
      },
      body,
      signal: AbortSignal.timeout(TIMEOUT_RESULT),
    });
    if (!res.ok) {
      const errBody = await res.text().catch(() => 'Unknown error');
      throw new Error(`ace-server POST /vae failed (${res.status}): ${errBody}`);
    }
    const data = await res.json() as { id: string };
    const jobId = data.id;

    // Poll until done
    for (;;) {
      const status = await this.pollJob(jobId);
      if (status.status === 'done') break;
      if (status.status === 'failed') throw new Error('VAE encode failed');
      if (status.status === 'cancelled') throw new Error('VAE encode cancelled');
      await new Promise(r => setTimeout(r, 200));
    }

    // Fetch raw latent result
    const resultRes = await this.getJobResult(jobId);
    if (!resultRes.ok) throw new Error(`VAE encode result fetch failed (${resultRes.status})`);
    const arrayBuf = await resultRes.arrayBuffer();
    return Buffer.from(arrayBuf);
  },

  /** POST /codes-decode — 5 Hz FSQ codes straight to audio through the
   *  detokenizer + VAE. No DiT, no sampler, no adapter: this is the LM's plan
   *  rendered literally. Polls the engine job and returns the encoded audio.
   *
   *  The request is built FRESH by the caller from {audio_codes, synth_model,
   *  vae, output_format, peak_clip} — never from an AceRequest that /lm echoed
   *  back. Server-only sideband fields do not survive that round trip, and
   *  forwarding the echo is the known way to lose them. */
  async codesDecode(
    request: {
      audio_codes: string; synth_model?: string; vae?: string;
      output_format?: string; peak_clip?: number;
    },
    timeoutMs = 90_000,
    isCancelled?: () => boolean,
  ): Promise<Buffer> {
    const started = Date.now();
    const res = await acePost('/codes-decode', request);
    const data = await res.json() as { id: string };
    const jobId = data.id;

    for (;;) {
      // Both exits cancel the ENGINE job. Throwing without cancelling leaves the
      // GPU decoding a result nobody will fetch, and the finished WAV then sits
      // in the engine's in-memory job ring until a restart.
      if (isCancelled?.()) {
        await this.cancelJob(jobId).catch(() => { /* engine may already be gone */ });
        throw new Error('codes-decode cancelled');
      }
      if (Date.now() - started > timeoutMs) {
        await this.cancelJob(jobId).catch(() => { /* best effort */ });
        throw new Error('codes-decode timed out');
      }
      const status = await this.pollJob(jobId);
      if (status.status === 'done') break;
      if (status.status === 'failed') throw new Error('codes-decode failed');
      if (status.status === 'cancelled') throw new Error('codes-decode cancelled');
      await new Promise(r => setTimeout(r, 200));
    }

    const resultRes = await this.getJobResult(jobId);
    if (!resultRes.ok) throw new Error(`codes-decode result fetch failed (${resultRes.status})`);
    const arrayBuf = await resultRes.arrayBuffer();
    return Buffer.from(arrayBuf);
  },

  /** POST /models/restore-policy — full eviction pass + back to EVICT_STRICT
   *  after a ?keep_loaded=1 latch (codes audition). Best-effort; false when
   *  the engine refused (CLI --keep-loaded), something was still in use, or
   *  the call failed. */
  async restoreEvictPolicy(): Promise<boolean> {
    try {
      const res = await fetch(`${BASE}/models/restore-policy`, {
        method: 'POST',
        signal: AbortSignal.timeout(TIMEOUT_QUICK),
      });
      if (!res.ok) return false;
      const body = await res.json().catch(() => ({})) as { restored?: boolean };
      return body.restored === true;
    } catch {
      return false;
    }
  },

  /** GET /models/loaded — the VRAM-resident ModelStore entries. `in_use` marks
   *  a module the worker is mid-call on, which unloadLabel() will refuse to
   *  evict. Best-effort: returns [] rather than throwing, so a caller using
   *  this to make room can always proceed. */
  async listLoadedModels(): Promise<Array<{ label: string; mb: number; in_use: boolean }>> {
    try {
      const res = await fetch(`${BASE}/models/loaded`, { signal: AbortSignal.timeout(TIMEOUT_QUICK) });
      if (!res.ok) return [];
      const data = await res.json().catch(() => ({})) as {
        loaded?: Array<{ label: string; mb: number; in_use: boolean }>;
      };
      return data.loaded ?? [];
    } catch {
      return [];
    }
  },

  /** POST /models/unload — evict one ModelStore label ('LM', 'DiT', 'VAE-Dec',
   *  'FSQ-Detok', …). Best-effort: a false/failed reply is not an error. */
  async unloadLabel(label: string): Promise<boolean> {
    try {
      const res = await fetch(`${BASE}/models/unload`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label }),
        signal: AbortSignal.timeout(TIMEOUT_QUICK),
      });
      if (!res.ok) return false;
      const body = await res.json().catch(() => ({})) as { ok?: boolean; unloaded?: boolean };
      return body.ok !== false && body.unloaded !== false;
    } catch {
      return false;
    }
  },
};
