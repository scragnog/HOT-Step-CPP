// sa3Tokenizer.ts — T5Gemma tokenization for the StableStep (SA3 refine) feature
//
// The C++ engine's POST /sa3-refine endpoint requires a pre-tokenized prompt
// (256 padded T5Gemma token ids as CSV) because the engine's bpe.h cannot
// parse SentencePiece tokenizer.json. Tokenization happens here in Node via
// @lenml/tokenizers (pure-JS port of the transformers.js tokenizer — chosen
// over @huggingface/transformers because that package transitively pulls
// onnxruntime-node native binaries, which broke the esbuild release bundle;
// verified token-for-token identical to the Python tokenizer on all 256 ids),
// loading tokenizer.json + tokenizer_config.json from <modelsDir>/onnx/sa3/ —
// the same directory ace-server scans for the SA3 ONNX graphs.

import fs from 'fs';
import path from 'path';
import { config } from '../config.js';

type Sa3Tokenizer = ReturnType<typeof import('@lenml/tokenizers').TokenizerLoader.fromPreTrained>;

/** The engine's SA3_TOK_LEN — /sa3-refine expects exactly this many ids. */
const SA3_TOK_LEN = 256;

/** Directory holding the SA3 ONNX graphs + tokenizer files.
 *  Mirrors the engine: <models_dir>/onnx/sa3 (ACESTEPCPP_MODELS override
 *  flows through config.aceServer.models). */
function sa3Dir(): string {
  return path.join(config.aceServer.models, 'onnx', 'sa3');
}

/** True if the SA3 model set appears installed (DiT graph + tokenizer). */
export function sa3ModelsInstalled(): boolean {
  const dir = sa3Dir();
  return fs.existsSync(path.join(dir, 'sa3-dit.onnx'))
      && fs.existsSync(path.join(dir, 'tokenizer.json'));
}

// Lazy singleton — the 34MB tokenizer.json parse is deferred to first use.
let tokenizerPromise: Promise<Sa3Tokenizer> | null = null;

async function getTokenizer(): Promise<Sa3Tokenizer> {
  if (!tokenizerPromise) {
    tokenizerPromise = (async () => {
      const dir = sa3Dir();
      const tokenizerJSON = JSON.parse(
        fs.readFileSync(path.join(dir, 'tokenizer.json'), 'utf-8'));
      const tokenizerConfig = JSON.parse(
        fs.readFileSync(path.join(dir, 'tokenizer_config.json'), 'utf-8'));
      const { TokenizerLoader } = await import('@lenml/tokenizers');
      // Construct directly from the parsed JSON files — no hub resolution.
      return TokenizerLoader.fromPreTrained({ tokenizerJSON, tokenizerConfig });
    })();
    // On failure, allow a retry on the next call instead of caching the error.
    tokenizerPromise.catch(() => { tokenizerPromise = null; });
  }
  return tokenizerPromise;
}

/** Normalize whatever the tokenizer returns (number[], BigInt64Array, or a
 *  Tensor with a .data typed array) into a plain number[]. */
function toNumberArray(value: unknown): number[] {
  const raw: unknown =
    (value !== null && typeof value === 'object' && 'data' in (value as any))
      ? (value as any).data
      : value;
  return Array.from(raw as ArrayLike<number | bigint>, v => Number(v));
}

/**
 * Tokenize a prompt for the engine's /sa3-refine endpoint.
 * Truncates to 256 tokens and pads to exactly 256 with the tokenizer's pad id.
 * Returns the padded ids (length 256) and the real (non-pad) token count.
 */
export async function tokenizeForSa3(prompt: string): Promise<{ ids: number[]; nTokens: number }> {
  const tokenizer = await getTokenizer();
  const enc = tokenizer(prompt, {
    truncation: true,
    padding: 'max_length',
    max_length: SA3_TOK_LEN,
    return_tensor: false,
  });

  let ids = toNumberArray(enc.input_ids);
  const mask = toNumberArray(enc.attention_mask);
  let nTokens = mask.length > 0
    ? mask.reduce((a, b) => a + (b ? 1 : 0), 0)
    : ids.length;

  // Enforce exactly SA3_TOK_LEN ids regardless of tokenizer quirks.
  const padId = tokenizer.pad_token_id ?? 0;
  if (ids.length > SA3_TOK_LEN) ids = ids.slice(0, SA3_TOK_LEN);
  while (ids.length < SA3_TOK_LEN) ids.push(padId);
  if (nTokens > SA3_TOK_LEN) nTokens = SA3_TOK_LEN;
  if (nTokens < 1) nTokens = 1;

  return { ids, nTokens };
}

/** Comma-segments matching this are considered vocal descriptors and dropped
 *  from StableStep prompts (the SA3 refine targets instrumentals). */
const VOCAL_SEGMENT_RE =
  /\b(vocals?|singers?|singing|sung|sing|voice|voices|choir|rap|rapper|rapping|spoken|acapella|a cappella|lyrics|verse|chorus line)\b/i;

/**
 * Build the SA3 refine prompt from a track caption: strip vocal-related
 * descriptors (comma-segment-wise), then append the instrumental suffix and
 * target length. Falls back to "Instrumental track" if everything is stripped.
 */
export function buildStableStepPrompt(caption: string, durationSec: number): string {
  const kept = (caption || '')
    .split(',')
    .map(s => s.trim())
    .filter(s => s.length > 0 && !VOCAL_SEGMENT_RE.test(s));
  const base = kept.length > 0 ? kept.join(', ') : 'Instrumental track';
  return `${base}. Instrumental only, no vocals. Length: ${Math.round(durationSec)} seconds`;
}
