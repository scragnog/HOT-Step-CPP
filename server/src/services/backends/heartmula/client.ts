import { config } from '../../../config.js';
import type { HeartmulaCoverRolls } from '../../../contracts/studioWorkflows.js';

export const HEARTMULA_ROLES = ['global', 'local', 'flow', 'scalar_decoder'] as const;
export type HeartmulaRole = typeof HEARTMULA_ROLES[number];
export interface HeartmulaPick {
  variant: string;
  global_type: string;
  local_type: string;
  flow_type: string;
  scalar_decoder_type: string;
  adapter: string;
  adapter_scale: number;
}
export interface HeartmulaRequest extends HeartmulaPick {
  adapter_scale_attn?: number;
  adapter_scale_mlp?: number;
  adapter_scale_early?: number;
  adapter_scale_mid?: number;
  adapter_scale_late?: number;
  adapter_rebase?: number;
  infer_method?: string;
  scheduler?: string;
  guidance_mode?: string;
  flow_shift?: number;
  apg_norm_threshold?: number;
  plugin_params?: Record<string, unknown>;
  tags: string;
  lyrics: string;
  duration_s: number;
  seed?: number | string;
  noise_seed?: number | string;
  temperature: number;
  topk: number;
  cfg_scale: number;
  codec_steps: number;
  codec_guidance: number;
}
/** Stage 3c's /heartmula/cover body. The Node submission's heartmulaCover
 * becomes rolls; tags and sectioned lyrics use the ordinary top-level fields.
 * Submission remains disabled in mapHeartmulaRequest until that route exists. */
export interface HeartmulaCoverRequest extends HeartmulaRequest {
  rolls: HeartmulaCoverRolls;
}
export interface HeartmulaProps {
  synth_ready: boolean;
  available: boolean;
  encode_ready: boolean;
  selected: Record<string, string>;
  requested: Record<string, string>;
  errors: string[];
}
export interface HeartmulaFile {
  type: string;
  filename: string;
  bytes: number;
  variant: string;
  compatible: boolean;
  error: string;
}
export interface HeartmulaModels {
  variants: string[];
  components: Record<string, HeartmulaFile[]>;
}
export interface HeartmulaJob {
  status: string;
  stage: string;
  prompt_tokens?: number;
  prefill_done?: number;
  frames_done?: number;
  max_frames?: number;
  codec_chunk?: number;
  codec_chunks?: number;
  codec_step?: number;
  codec_steps?: number;
  seed: string;
  noise_seed: string;
  pick: Record<string, unknown>;
  end_reason: string;
  admission?: Record<string, unknown> | null;
  error?: { code: string; message: string } | null;
}

export class HeartmulaEngineError extends Error {
  constructor(readonly status: number, readonly code: string, message: string,
    readonly detail: unknown) { super(message); }
}

async function request(route: string, body?: unknown, timeout = 15_000): Promise<Response> {
  const res = await fetch(`${config.aceServer.url}${route}`, {
    method: body === undefined ? 'GET' : 'POST',
    ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
    signal: AbortSignal.timeout(timeout),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({})) as Record<string, unknown>;
    throw new HeartmulaEngineError(res.status, String(detail.code ?? 'engine_error'),
      String(detail.error ?? detail.message ?? `HeartMuLa HTTP ${res.status}`), detail);
  }
  return res;
}

let cached: HeartmulaProps | null = null;
export const heartmulaClient = {
  async props(): Promise<{ props: HeartmulaProps | null; stale: boolean }> {
    try {
      cached = await (await request('/heartmula/props', undefined, 2500)).json() as HeartmulaProps;
      return { props: cached, stale: false };
    } catch { return { props: cached, stale: true }; }
  },
  cachedProps: () => cached,
  async models(): Promise<HeartmulaModels> {
    return (await request('/heartmula/models')).json();
  },
  async select(pick: Partial<HeartmulaPick>): Promise<Record<string, unknown>> {
    return (await request('/heartmula/select-model', pick)).json();
  },
  async unload(): Promise<void> {
    // /heartmula/unload waits in the engine's GPU worker queue, behind whatever
    // runs there (an inline YuE2 render, say), so skip it when nothing is
    // resident (GH #341). Props answers off the worker; no answer = unload anyway.
    const props = await request('/heartmula/props', undefined, 3000)
      .then(res => res.json() as Promise<{ vram?: { resident_bytes?: number } }>).catch(() => null);
    if (props?.vram?.resident_bytes === 0) return;
    // Older engine builds have no HeartMuLa residency to release.
    try {
      const result = await (await request('/heartmula/unload', {})).json() as { unloaded?: boolean };
      if (result.unloaded !== true) throw new Error('HeartMuLa did not confirm that its models were unloaded');
    }
    catch (err) { if (!(err instanceof HeartmulaEngineError && err.status === 404)) throw err; }
  },
  async synth(wire: HeartmulaRequest): Promise<string> {
    const result = await (await request('/heartmula/synth', wire)).json() as { id?: string };
    if (!result.id) throw new Error('HeartMuLa returned no job id');
    return result.id;
  },
  async job(id: string): Promise<HeartmulaJob> {
    return (await request(`/heartmula/job?id=${encodeURIComponent(id)}`)).json();
  },
};
