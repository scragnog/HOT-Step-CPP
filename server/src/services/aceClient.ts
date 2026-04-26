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
const TIMEOUT_POLL   = 120_000;   // job polling (ace-server may be busy with a DiT step)
const TIMEOUT_RESULT = 300_000;   // fetching large audio results (encode + transfer)

/** Props response from GET /props */
export interface AceProps {
  models: {
    lm: string[];
    embedding: string[];
    dit: string[];
    vae: string[];
  };
  adapters: string[];
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
  lm_batch_size?: number;
  synth_batch_size?: number;
  lm_temperature?: number;
  lm_cfg_scale?: number;
  lm_top_p?: number;
  lm_top_k?: number;
  lm_negative_prompt?: string;
  use_cot_caption?: boolean;
  audio_codes?: string;
  inference_steps?: number;
  guidance_scale?: number;
  shift?: number;
  audio_cover_strength?: number;
  cover_noise_strength?: number;
  repainting_start?: number;
  repainting_end?: number;
  task_type?: string;
  track?: string;
  infer_method?: string;
  scheduler?: string;       // 'linear' | 'ddim_uniform' | 'sgm_uniform' | etc.
  guidance_mode?: string;   // 'apg' | 'cfg' | etc.
  peak_clip?: number;
  // Server routing fields
  synth_model?: string;
  lm_model?: string;
  vae_model?: string;
  adapter?: string;
  adapter_scale?: number;
  adapter_group_scales?: {
    self_attn: number;
    cross_attn: number;
    mlp: number;
    cond_embed: number;
  };
  adapter_mode?: string;  // "merge" (default) or "runtime" (fast, for K-quant models)
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
  // Post-VAE spectral denoiser (HOT-Step)
  denoise_strength?: number;   // 0.0 = off, 1.0 = max suppression
  denoise_smoothing?: number;  // 0.0 = sharp gate, 1.0 = very smooth
  denoise_mix?: number;        // 0.0 = all dry, 1.0 = all denoised
}

/** Job status from ace-server */
export interface AceJobStatus {
  status: 'running' | 'done' | 'failed' | 'cancelled';
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

  /** POST /lm — submit LM generation job, returns job ID */
  async submitLm(request: AceRequest, mode?: 'inspire' | 'format'): Promise<string> {
    const path = mode ? `/lm?mode=${mode}` : '/lm';
    const res = await acePost(path, request);
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
    format: string = 'wav16',
    keepLoaded = false,
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

  /** POST /understand — submit understand job, returns job ID */
  async submitUnderstand(audioBuffer: Buffer): Promise<string> {
    const boundary = '----HotStepBoundary' + Date.now();
    const header = `--${boundary}\r\nContent-Disposition: form-data; name="audio"; filename="input.wav"\r\nContent-Type: audio/wav\r\n\r\n`;
    const footer = `\r\n--${boundary}--\r\n`;
    const body = Buffer.concat([Buffer.from(header), audioBuffer, Buffer.from(footer)]);

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
};
