// training/yue2Tokenize.ts — `ace-train yue2-tokenize`: model resolution,
// defaults and the argv builder for the second of the three cache stages.
//
// Split from yue2Train.ts for the reason that file gives for standing apart
// from mm3Train.ts: the CLI surface shares nothing. This stage takes no audio
// directory, no dataset, no captions and no VAE — it takes the manifest
// `yue2-preprocess` already wrote and rewrites it in place.
//
// WHAT IT PRODUCES, and why the AR trainer cannot run without it: the semantic
// tokenizer (MERT-v2-FullSong block 20 + Mothersuperior's 8-layer head) reads
// every SOURCE the manifest lists and writes `<manifest dir>/codes/<stem>.i32`
// — the whole-song code array — plus one sliced `codes/<clip id>.i32` per clip.
// It then sets `codec_ids_present`, gives each source and each clip its own
// `codec_ids` path, and stamps the tokenizer filename into the manifest.
// `yue2-ar-train` reads the manifest as SOURCES and trains next-token CE over
// the codec positions of that whole-song array, so without this stage there is
// nothing for the AR half to predict. (The NAR trainer degrades instead: it
// trains text-only and says so.)
//
// The codes are RAW, in [0, 32768). The `+ codec_offset` (151853) is added
// once, at conditioning time, inside the trainer. Nothing on this side ever
// touches them, which is the point of writing it down here as well: a second
// offset would be finite, in-vocabulary and completely wrong.
//
// TWO ENGINE FACTS THAT LOOK LIKE MISSING OPTIONS:
//
//   1. There is no output directory. `codes/` is created beside the manifest
//      and the manifest-relative path is what the readers resolve, so an
//      `--out` would only let the two halves disagree.
//   2. There is no TF32 knob. cmd_yue2_tokenize sets NVIDIA_TF32_OVERRIDE=0 as
//      its first statement and yue2_tokenize_run REFUSES to start otherwise —
//      the driver reads that variable when the CUDA context is CREATED, so it
//      cannot be a flag the server passes late.
//
// LICENCE: the tokenizer weights are derived from YuE2-3B and are CC BY-NC
// 4.0, like the rest of the family. The notice lives in
// services/backends/yue2/index.ts.

import fs from 'fs';
import path from 'path';

import { config, getFFmpegPath } from '../../config.js';
import { YUE2_TOKENIZER_IDS, type Yue2TokenizerBody, type Yue2TokenizerId } from '../../contracts/preferences.js';
import { yue2ModelDir } from './yue2Train.js';

export { YUE2_TOKENIZER_IDS, type Yue2TokenizerId };

// ── Model files ─────────────────────────────────────────────────────────────

/** The two directories the engine's own discovery scans, in its order
 *  (`yue2_tok_head_find`: `<models>/yue2` first, then `<models>`). Mirrored
 *  rather than assumed, because the readiness check below must agree with what
 *  `--models` will actually find. */
function tokenizerSearchDirs(): string[] {
  return [yue2ModelDir(), config.aceServer.models];
}

/** Each tokenizer's file prefix and registry id. v9 is the engine's own
 *  discovery target (`yue2-tok-*`); the WannabeTokenizer files have their own
 *  prefixes, so discovery never picks one by accident and `--tok` names them. */
export const YUE2_TOKENIZERS: Record<Yue2TokenizerId, { prefix: string; registryId: string; label: string }> = {
  v9: { prefix: 'yue2-tok-', registryId: 'yue2-tok-f16', label: 'v9 (Mothersuperior)' },
  'wannabe-v1': { prefix: 'yue2-wannabe-v1-', registryId: 'yue2-wannabe-v1-f16', label: 'WannabeTokenizer V1' },
  'wannabe-v2': { prefix: 'yue2-wannabe-v2-', registryId: 'yue2-wannabe-v2-f16', label: 'WannabeTokenizer V2' },
};

/** Which tokenizer a file name belongs to (the manifest's codec_ids_tokenizer),
 *  or '' when it is none of them. */
export function yue2TokenizerOfFile(file: string): Yue2TokenizerId | '' {
  const base = path.basename(file);
  return YUE2_TOKENIZER_IDS.find(id => base.startsWith(YUE2_TOKENIZERS[id].prefix)) ?? '';
}

/** The stored choice (preferences family `yue2-tokenizer`): routes/preferences.ts
 *  registers the reader; until it does, and on any read failure, v9 for both. */
let tokenizerSource: () => Yue2TokenizerBody | null = () => null;
export function setYue2TokenizerSource(read: () => Yue2TokenizerBody | null): void { tokenizerSource = read; }
export function yue2TokenizerSetting(): Yue2TokenizerBody {
  try { return tokenizerSource() ?? { training: 'v9', cover: 'v9' }; } catch { return { training: 'v9', cover: 'v9' }; }
}

/** The tokenizer's GGUF, or '' when it is not installed.
 *
 *  For v9 this is NOT passed to the engine — `buildYue2TokenizeArgs` sends
 *  `--models` and lets discovery choose, so the quant ranking stays in one
 *  place; it resolves the same file only to name it in a status payload and to
 *  answer "is the stage installed". The ranking differs in one harmless way:
 *  discovery prefers f16 over f32, while this returns whichever the first
 *  search dir holds. A WannabeTokenizer file is passed as `--tok`. */
export function resolveYue2TokenizerModel(id: Yue2TokenizerId = 'v9'): string {
  const prefix = YUE2_TOKENIZERS[id].prefix;
  for (const dir of tokenizerSearchDirs()) {
    let hits: string[] = [];
    try {
      hits = fs.readdirSync(dir).filter(f => f.startsWith(prefix) && f.endsWith('.gguf')).sort();
    } catch {
      continue;  // a missing models dir reads as a missing file, never a throw
    }
    if (hits.length) return path.join(dir, hits[0]);
  }
  return '';
}

/** Which required files are missing, as user-facing names. Empty = ready.
 *
 *  One file, and it is optional to the rest of the app: nothing in generation
 *  needs the tokenizer, so a fresh install will not have it. Registry id
 *  `yue2-tok-f16`, which the Model Manager can download. */
export function missingYue2TokenizeModels(id: Yue2TokenizerId = 'v9'): string[] {
  const t = YUE2_TOKENIZERS[id];
  return resolveYue2TokenizerModel(id)
    ? []
    : [id === 'v9' ? 'the YuE2 semantic tokenizer (yue2-tok-*.gguf)' : `the ${t.label} (${t.registryId})`];
}

/** The tokenize stage is done FOR THIS CHOICE: codes exist and the manifest
 *  says the chosen tokenizer made them (a cache stamped before the choice
 *  existed is v9's). The chains use it so a changed choice re-runs the stage. */
export function yue2CodesReady(t: { done: boolean; status?: { tokenizer?: string } | null },
  choice: Yue2TokenizerId): boolean {
  return t.done && (yue2TokenizerOfFile(t.status?.tokenizer ?? '') || 'v9') === choice;
}

/** Every tokenizer with whether it is installed, for the pickers. */
export function yue2TokenizerChoices(): Array<{ id: Yue2TokenizerId; label: string; installed: boolean; registryId: string }> {
  return YUE2_TOKENIZER_IDS.map(id => ({
    id, label: YUE2_TOKENIZERS[id].label, registryId: YUE2_TOKENIZERS[id].registryId,
    installed: !!resolveYue2TokenizerModel(id),
  }));
}

// ── Defaults ────────────────────────────────────────────────────────────────
//
// The engine's own, unchanged: this stage has no recipe to tune. It is a
// deterministic re-encode of audio the manifest already names, and every knob
// below only chooses how much of it to do.
export const YUE2_TOKENIZE_DEFAULTS = {
  /** auto = the repo's WAV/MP3 decoder for those two and ffmpeg for the rest;
   *  ffmpeg = one resampler across the whole corpus. Must match what
   *  preprocess used: codes and latents are only frame-aligned because both
   *  come from identically decoded 48 kHz samples. */
  decode: 'auto' as 'auto' | 'ffmpeg',
  /** Case-insensitive substring of the source name, and a cap on how many
   *  matching sources are taken. A partial run is legal — the clips it missed
   *  stay text-only and the trainer reports that — so these are the way to do
   *  less work, not a failure mode. */
  only: '',
  limit: 0,
  /** Re-encode sources whose codes are already cached. Without it a source
   *  whose `codes/<stem>.i32` is already exactly `frames` values long is
   *  skipped, which is what makes a resumed run cheap. */
  force: false,
} as const;

// ── Manifest state ──────────────────────────────────────────────────────────

/** What the manifest says about this stage, for a status endpoint and for the
 *  guard that refuses an AR run against a cache with no codes. */
export interface Yue2CodecIdsStatus {
  /** The `codec_ids_present` flag the stage sets. */
  present: boolean;
  /** Basename of the tokenizer that produced them, as stamped into the
   *  manifest — the one thing that identifies a partially re-tokenized cache. */
  tokenizer: string;
  createdAt: string;
  sources: number;
  /** Sources carrying a `codec_ids` path. The AR trainer reads THESE; a source
   *  without one is not trainable, however many of its clips have slices. */
  sourcesWithCodes: number;
  clips: number;
  clipsWithCodes: number;
}

/** Read that state. Null when there is no manifest, or when it is mid-write:
 *  this feeds a polled endpoint and a route guard, and neither may 500 over a
 *  partial file. Same contract as readYue2PreprocessSummary. */
export function readYue2CodecIdsStatus(manifestPath: string): Yue2CodecIdsStatus | null {
  try {
    const j = JSON.parse(fs.readFileSync(manifestPath, 'utf-8')) as Record<string, unknown>;
    const sources = Array.isArray(j.sources) ? j.sources as Array<Record<string, unknown>> : [];
    const clips = Array.isArray(j.clips) ? j.clips as Array<Record<string, unknown>> : [];
    const has = (r: Record<string, unknown>) => typeof r.codec_ids === 'string' && r.codec_ids !== '';
    return {
      present: j.codec_ids_present === true,
      tokenizer: typeof j.codec_ids_tokenizer === 'string' ? j.codec_ids_tokenizer : '',
      createdAt: typeof j.codec_ids_created_at === 'string' ? j.codec_ids_created_at : '',
      sources: sources.length,
      sourcesWithCodes: sources.filter(has).length,
      clips: clips.length,
      clipsWithCodes: clips.filter(has).length,
    };
  } catch {
    return null;
  }
}

// ── Arg building ────────────────────────────────────────────────────────────
//
// Every flag below is one cmd_yue2_tokenize actually parses
// (engine/tools/ace-train.cpp). An unknown option is a hard exit 2, so nothing
// speculative belongs here.

export interface ResolvedYue2TokenizeOptions {
  /** `<latents>/yue2_preprocess.json` — REWRITTEN IN PLACE. Not an audio
   *  directory: the sources are the ones this manifest already names, and the
   *  codes have to land beside the latents they are aligned to. */
  manifest: string;
  decode: 'auto' | 'ffmpeg';
  only: string;
  limit: number;
  force: boolean;
  /** Which tokenizer makes the codes. A change from the one the manifest
   *  names re-encodes every source (the engine compares the two). */
  tokenizer: Yue2TokenizerId;
  /** Carried for the runner's own log lines and the run record. */
  datasetSlug: string;
}

export function buildYue2TokenizeArgs(o: ResolvedYue2TokenizeOptions): string[] {
  const args = [
    'yue2-tokenize',
    '--manifest', o.manifest,
    '--models', config.aceServer.models,
    '--decode', o.decode,
  ];
  // The engine's default is the bare string "ffmpeg" (i.e. PATH), which is not
  // a safe assumption on a portable Windows install — so the bundled binary is
  // passed explicitly when there is one, exactly as buildYue2PreprocessArgs
  // does it. Without it, FLAC/OGG/M4A sources cannot be decoded at all.
  const ff = getFFmpegPath();
  if (ff) args.push('--ffmpeg', ff);
  if (o.only) args.push('--only', o.only);
  if (o.limit > 0) args.push('--limit', String(o.limit));
  if (o.force) args.push('--force');
  // v9 sends no --tok: `--models` lets the engine's quant ranking pick, and it
  // prints which file it chose. A WannabeTokenizer file is not discoverable by
  // design, so it is named.
  // (A job persisted before the choice existed carries none: v9.)
  if (o.tokenizer && o.tokenizer !== 'v9') args.push('--tok', resolveYue2TokenizerModel(o.tokenizer));
  return args;
}
