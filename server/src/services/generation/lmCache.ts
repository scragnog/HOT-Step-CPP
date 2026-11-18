// generation/lmCache.ts — LM audio code cache
//
// Caches ONLY LM-generated output fields keyed by lm_seed + LM-affecting params.
// Non-LM parameters (DiT, adapter, DCW, etc.) are NEVER cached.

import crypto from 'crypto';
import type { AceRequest } from '../../services/aceClient.js';

export interface LmCacheEntry {
  audio_codes: string;
  caption: string;
  lyrics: string;
  bpm: number;
  duration: number;
  keyscale: string;
  timesignature: string;
  /** Resolved engine LM seed for this output (base lm_seed + batch index) —
   *  cached so cache-hit tracks display the seed that produced their codes. */
  lm_seed?: number;
  /** DiT seed the engine paired with this output (seed + batch index). */
  seed?: number;
}

const LM_CACHE_MAX = 20;
const lmCache = new Map<string, { lmOutputs: LmCacheEntry[]; timestamp: number }>();

/** Compute a stable hash key from LM-affecting parameters */
export function computeLmCacheKey(req: AceRequest): string {
  // lm_seed is left unset on the request when the LM seed is tied to the
  // DiT seed (the engine's own fallback then ties it) — use the effective
  // value here too, or every tied request would collide on `undefined`
  // regardless of the actual (possibly random) DiT seed.
  const effectiveLmSeed = req.lm_seed !== undefined ? req.lm_seed : req.seed;
  const keyObj = {
    lm_seed: effectiveLmSeed,
    caption: req.caption,
    lyrics: req.lyrics,
    bpm: req.bpm,
    duration: req.duration,
    keyscale: req.keyscale,
    timesignature: req.timesignature,
    vocal_language: req.vocal_language,
    lm_model: req.lm_model,
    lm_batch_size: req.lm_batch_size,
    lm_temperature: req.lm_temperature,
    lm_cfg_scale: req.lm_cfg_scale,
    lm_top_p: req.lm_top_p,
    lm_top_k: req.lm_top_k,
    lm_negative_prompt: req.lm_negative_prompt,
    use_cot_caption: req.use_cot_caption,
    // Sampling-time anti-loop settings change the emitted codes, so they have
    // to key the cache — omitting them made A/B-ing the repetition penalty at a
    // fixed seed return the previous run's codes and look like a dead knob.
    lm_rep_penalty: req.lm_rep_penalty,
    lm_rep_window: req.lm_rep_window,
    lm_rep_mode: req.lm_rep_mode,
    lm_dry_base: req.lm_dry_base,
    lm_dry_min_len: req.lm_dry_min_len,
    lm_min_p: req.lm_min_p,
    lm_cfg_min_p: req.lm_cfg_min_p,
    lm_cfg_logprob: req.lm_cfg_logprob,
    lm_cfg_rescale: req.lm_cfg_rescale,
    // Same reasoning for the LM adapter: it changes the code distribution as
    // surely as lm_model does.
    lm_adapter: req.lm_adapter,
    lm_adapter_scale: req.lm_adapter_scale,
  };
  return crypto.createHash('sha256')
    .update(JSON.stringify(keyObj))
    .digest('hex')
    .substring(0, 16);
}

/** Evict oldest entries when cache exceeds max size */
export function evictLmCache(): void {
  if (lmCache.size <= LM_CACHE_MAX) return;
  const entries = [...lmCache.entries()].sort((a, b) => a[1].timestamp - b[1].timestamp);
  const toRemove = entries.slice(0, lmCache.size - LM_CACHE_MAX);
  for (const [key] of toRemove) {
    lmCache.delete(key);
  }
}

export function getLmCache(key: string) {
  return lmCache.get(key);
}

export function setLmCache(key: string, lmOutputs: LmCacheEntry[]) {
  lmCache.set(key, { lmOutputs, timestamp: Date.now() });
  evictLmCache();
}

export function getLmCacheSize(): number {
  return lmCache.size;
}
