import type { AceRequest } from '../../aceClient.js';
import { config } from '../../../config.js';
import { listGpusSync } from '../../gpuDevices.js';

/** Split a batch of N songs into LM passes of at most `cap`. Pass k starts at
 *  song `offset`, so its seeds are shifted by `offset`: the engine gives song b
 *  of a pass lm_seed + b, and every song keeps the LM seed it would get in one
 *  pass of N. A seed the engine resolves itself (absent or negative) stays
 *  unset, and those passes draw their own. */
export function planLmPasses(req: AceRequest, n: number, cap: number): AceRequest[] {
  const total = Math.max(1, Math.floor(n) || 1);
  const size = Math.max(1, Math.floor(cap) || 1);
  const shift = (v: number | undefined, k: number) => (typeof v === 'number' && v >= 0 ? v + k : v);
  const passes: AceRequest[] = [];
  for (let offset = 0; offset < total; offset += size) {
    const pass: AceRequest = { ...req, lm_batch_size: Math.min(size, total - offset) };
    if (offset > 0) {
      // lm_seed absent = the engine ties it to seed, so shifting seed shifts both.
      if (req.lm_seed !== undefined) pass.lm_seed = shift(req.lm_seed, offset);
      else pass.seed = shift(req.seed, offset);
    }
    passes.push(pass);
  }
  return passes;
}

// Measured on an RTX 4090 with the 4B LM, from the engine process's own GPU
// memory (HOTSTEP-1121). Sized for the worst case, the 4B LM in BF16.
const LM_BASE_MB = 8857;       // weights and buffers at batch 1, less its KV
const LM_SONG_MB = 2304;       // KV per song: two 1152 MB sets (CFG cond + uncond)
const SYNTH_STACK_MB = 12660;  // turbo DiT, text and condition encoders, VAE, all resident
const SPARE_MB = 1536;         // desktop, driver and rounding

/** Songs one LM pass may hold on a card of `totalMb`. Co-resident (keep
 *  models loaded) keeps the synth stack in VRAM beside the LM. No card found:
 *  one song per pass, the memory a batch always used before; songs still each
 *  get their own plan, in more passes. */
export function lmSongsPerPass(totalMb: number | null, coResident: boolean, cap: number): number {
  if (!totalMb || totalMb <= 0) return 1;
  const free = totalMb - LM_BASE_MB - SPARE_MB - (coResident ? SYNTH_STACK_MB : 0);
  return Math.max(1, Math.min(Math.max(1, Math.floor(cap)), Math.floor(free / LM_SONG_MB)));
}

let smallestCardMb: number | null | undefined;
/** lmSongsPerPass for this machine: the smallest NVIDIA card, read once. */
export function aceLmSongsPerPass(coResident: boolean): number {
  if (smallestCardMb === undefined) {
    const mb = listGpusSync().map(g => g.memoryMB).filter(n => n > 0);
    smallestCardMb = mb.length ? Math.min(...mb) : null;
  }
  return lmSongsPerPass(smallestCardMb, coResident || config.aceServer.keepLoaded, config.aceServer.lmMaxBatch);
}
