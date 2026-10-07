// resolve/content.ts — wildcard expansion, compose-time caption helpers and
// duration, ported from the UI so Node produces the same text.
//
// Sources, which these must keep matching:
//   expandWildcards, hasWildcards, randomWildcardSeed  ui/src/utils/wildcardUtils.ts
//   composeCreateCaption                               ui/src/components/create/CreatePanel.tsx (buildCaption)
//   estimateDuration, resolveDuration                  ui/src/utils/estimateDuration.ts

import { randomInt } from 'node:crypto';

// ── Wildcards ────────────────────────────────────────────────────────────────

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s += 0x6d2b79f5;
    let z = s;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
}

function mixSeed(seed: number, slot = 0): number {
  const combined = seed ^ slot;
  return ((combined ^ (combined / 0x100000000)) >>> 0);
}

const INNER_BRACES = /\{([^{}]+)\}/g;
const MAX_PASSES = 64;

export function hasWildcards(text: string): boolean {
  return /\{[^{}]*\}/.test(text);
}

/** Expand {A|B|C} groups innermost first, deterministically from `seed`. */
export function expandWildcards(text: string, seed: number, slot = 0, delimiter = '|'): string {
  if (!text || !hasWildcards(text)) return text;
  const rng = mulberry32(mixSeed(seed, slot));
  let current = text;
  for (let pass = 0; pass < MAX_PASSES; pass++) {
    if (!hasWildcards(current)) break;
    let matched = false;
    current = current.replace(INNER_BRACES, (_full, content: string) => {
      matched = true;
      const options = content.split(delimiter).map(o => o.trim()).filter(o => o.length > 0);
      if (options.length === 0) return '';
      if (options.length === 1) return options[0];
      return options[Math.floor(rng() * options.length)];
    });
    INNER_BRACES.lastIndex = 0;
    if (!matched) break;
  }
  return current;
}

/** Same range as the browser's randomWildcardSeed: a safe integer up to 2^53-1. */
export function randomWildcardSeed(): number {
  const hi = randomInt(0, 0x200000);          // top 21 bits
  const lo = randomInt(0, 0x100000000);       // bottom 32 bits
  return Math.min(hi * 0x100000000 + lo, Number.MAX_SAFE_INTEGER);
}

/** Create's rule: the DiT seed when it is fixed, a fresh draw when random. */
export function createWildcardSeed(randomSeed: unknown, seed: unknown): { seed: number; seedFrom: 'dit-seed' | 'random' } {
  if (randomSeed) return { seed: randomWildcardSeed(), seedFrom: 'random' };
  return { seed: typeof seed === 'number' ? seed : Number(seed) || 0, seedFrom: 'dit-seed' };
}

// ── Create caption helpers ───────────────────────────────────────────────────

/** LoRA trigger prepended unless the caption already starts with it as a whole
 *  word; beat intro/outro request appended. */
export function composeCreateCaption(
  base: string,
  opts: { loraTrigger?: string; beatIntro?: boolean; introBars?: number } = {},
): string {
  const trigger = (opts.loraTrigger ?? '').trim();
  const start = base.trimStart();
  const hasTrigger = trigger.length > 0
    && start.slice(0, trigger.length).toLowerCase() === trigger.toLowerCase()
    && (start.length === trigger.length || /[,\s]/.test(start[trigger.length]));
  const loraText = trigger && !hasTrigger ? `${trigger}, ` : '';
  const beatText = opts.beatIntro ? `, with a clean ${opts.introBars ?? 2}-bar percussive intro and outro for DJ mixing` : '';
  return `${loraText}${base}${beatText}`;
}

// ── Duration ─────────────────────────────────────────────────────────────────

const SECTION_RE = /^\[.+\]$/;

export function estimateDuration(lyrics: string, bpm: number): number {
  if (!lyrics.trim() || bpm <= 0) return 0;
  const barDuration = 240.0 / Math.max(bpm, 40);
  let sectionCount = 0;
  let lyricLineCount = 0;
  for (const line of lyrics.trim().split('\n')) {
    const stripped = line.trim();
    if (!stripped) continue;
    if (SECTION_RE.test(stripped)) sectionCount++;
    else lyricLineCount++;
  }
  const vocalSeconds = lyricLineCount * 3.5;
  const breakSeconds = Math.max(sectionCount - 1, 0) * 4 * barDuration;
  return Math.max(90, Math.min(Math.floor(vocalSeconds + breakSeconds), 360));
}

/** The LLM's duration when allowed and positive, else the lyric estimate, else
 *  `fallback`. `useLlmDuration` defaults to true, as the browser setting does. */
export function resolveDuration(
  llmDuration: number | undefined | null,
  lyrics: string,
  bpm: number,
  useLlmDuration = true,
  fallback = 180,
): { value: number; source: 'llm' | 'estimate' | 'fallback' } {
  if (useLlmDuration && llmDuration && llmDuration > 0) return { value: llmDuration, source: 'llm' };
  const estimated = estimateDuration(lyrics, bpm);
  return estimated > 0 ? { value: estimated, source: 'estimate' } : { value: fallback, source: 'fallback' };
}
