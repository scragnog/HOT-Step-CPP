// training/sidecarIO.ts — byte-compatible Side-Step Option-A sidecar I/O
//
// A sidecar is `<stem>.txt` sitting next to the audio file (D4). Side-Step and
// HOT-Step both read and write it, so the parse/write algorithms here must match
// Side-Step's byte for byte — a deviation silently corrupts an existing corpus.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §6.1–§6.5

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import type { MergePolicy } from './types.js';

/** Write order — these six are ALWAYS emitted, even when empty (§6.2). */
export const FIELD_ORDER: readonly string[] = ['caption', 'genre', 'bpm', 'key', 'signature', 'is_instrumental'];

/** The gate that stops a lyric line like `Time: 4:32 in the video` from being
 *  read as a field. Never widen casually (§6.4). */
export const KNOWN_SIDECAR_KEYS: ReadonlySet<string> = new Set([
  'caption', 'genre', 'bpm', 'key', 'signature', 'lyrics',
  'tags', 'custom_tag', 'trigger', 'tag_position', 'is_instrumental',
  'repeat', 'prompt_override', 'duration',
  'keyscale', 'scale',
  'timesignature', 'time_signature', 'sig',
  'language', 'lang',
  'tempo',
  'repeats',
]);

export const SIDECAR_ALIASES: Readonly<Record<string, string>> = {
  keyscale: 'key',
  scale: 'key',
  timesignature: 'signature',
  time_signature: 'signature',
  sig: 'signature',
  language: 'language',
  lang: 'language',
  tempo: 'bpm',
  repeats: 'repeat',
};

/** Fields the merge policies treat as machine-generated (§6.5). */
export const GENERATED_FIELDS: ReadonlySet<string> = new Set([
  'caption', 'lyrics', 'genre', 'bpm', 'key', 'signature', 'is_instrumental',
]);

/** The only keys this server may ever write into a sidecar (§6.4).
 *  Notably NOT `audio_codes` — that lives in the label JSON (D7). */
export const WRITABLE_SIDECAR_KEYS: ReadonlySet<string> = new Set([
  ...FIELD_ORDER, 'lyrics', 'language', 'custom_tag', 'repeat', 'prompt_override', 'duration',
]);

// ── Parse ────────────────────────────────────────────────────────────────

/**
 * Parse an Option-A sidecar body into a lowercase key → value map.
 *
 * Once `lyrics:` starts, EVERY subsequent line is lyrics — including lines
 * containing colons. That is safe because write_sidecar always emits lyrics
 * last (§6.3).
 */
export function parseTxtMetadata(content: string): Record<string, string> {
  const meta: Record<string, string> = {};
  if (!content) return meta;

  let text = content;
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  text = text.trim();
  if (!text) return meta;

  const lines = text.split(/\r\n|\r|\n/);
  let currentKey: string | null = null;
  let currentLines: string[] = [];

  const flush = () => {
    if (currentKey !== null) {
      meta[currentKey] = currentLines.join('\n').trim();
    }
  };

  for (const line of lines) {
    if (currentKey === 'lyrics') {
      currentLines.push(line.replace(/\s+$/, ''));
      continue;
    }

    const lstripped = line.replace(/^\s+/, '');
    const idx = lstripped.indexOf(':');
    if (idx !== -1 && !lstripped.startsWith('[')) {
      const candidate = lstripped.slice(0, idx).trim().toLowerCase();
      if (KNOWN_SIDECAR_KEYS.has(candidate)) {
        flush();
        const value = lstripped.slice(idx + 1).trim();
        currentKey = SIDECAR_ALIASES[candidate] ?? candidate;
        currentLines = value ? [value] : [];
        continue;
      }
    }

    if (currentKey !== null) {
      currentLines.push(line.replace(/\s+$/, ''));
    }
  }

  flush();
  return meta;
}

/** Read + parse a sidecar. Returns `{}` when missing or unreadable (never throws). */
export function readSidecar(sidecarPath: string): Record<string, string> {
  try {
    if (!fs.existsSync(sidecarPath)) return {};
    return parseTxtMetadata(fs.readFileSync(sidecarPath, 'utf-8'));
  } catch {
    return {};
  }
}

// ── Write ────────────────────────────────────────────────────────────────

/** boolean → 'true'|'false'; a bare true/false string → lowercased; else String(). */
function normalizeValue(v: unknown): string {
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'string') {
    return /^(true|false)$/i.test(v) ? v.toLowerCase() : v;
  }
  return String(v ?? '');
}

/** Where a sidecar's previous contents go: `<sidecar>.bak`, or with
 *  `keepBackups` and a `.bak` already there, `<sidecar>.<yyyymmdd-hhmmss>.bak`. */
export function backupPath(sidecarPath: string, keepBackups: boolean, now = new Date()): string {
  const plain = `${sidecarPath}.bak`;
  if (!keepBackups || !fs.existsSync(plain)) return plain;
  const p = (n: number) => String(n).padStart(2, '0');
  const stamp = `${now.getFullYear()}${p(now.getMonth() + 1)}${p(now.getDate())}-${p(now.getHours())}${p(now.getMinutes())}${p(now.getSeconds())}`;
  return `${sidecarPath}.${stamp}.bak`;
}

/**
 * Rewrite a sidecar in full: `.bak` copy, the six ordered fields, sorted extras,
 * then `lyrics:` last. Atomic (temp file in the same directory + rename).
 * `keepBackups` never replaces an existing backup, so every such write keeps
 * the file it replaced (see backupPath).
 */
export async function writeSidecar(sidecarPath: string, data: Record<string, string>,
  opts: { keepBackups?: boolean } = {}): Promise<void> {
  // 1. Best-effort backup of the previous contents
  try {
    if (fs.existsSync(sidecarPath)) fs.copyFileSync(sidecarPath, backupPath(sidecarPath, !!opts.keepBackups));
  } catch (err) {
    if (opts.keepBackups) throw err;  // promised a backup: no write without one
    /* otherwise best-effort */
  }

  const lines: string[] = [];

  // 2. All six ordered fields, always, even when empty
  for (const key of FIELD_ORDER) {
    lines.push(`${key}: ${normalizeValue(data[key] ?? '')}`);
  }

  // 3. Remaining truthy keys (never lyrics), plain ascending string sort.
  //    D7/§6.4 — a key Side-Step's parser cannot read back (notably
  //    `audio_codes`) must NEVER reach a sidecar: the parser would glue it onto
  //    the previous field's value and corrupt it. Keys the parser does know are
  //    kept so a round trip never strips a user's `tags`/`trigger`/`tag_position`.
  const extras = Object.keys(data)
    .filter(k => !FIELD_ORDER.includes(k) && k !== 'lyrics' && !!data[k]
      && (WRITABLE_SIDECAR_KEYS.has(k) || KNOWN_SIDECAR_KEYS.has(k)))
    .sort();
  for (const key of extras) {
    lines.push(`${key}: ${normalizeValue(data[key])}`);
  }

  // 4. Lyrics always last
  const lyrics = normalizeValue(data['lyrics'] ?? '');
  lines.push(lyrics ? `lyrics:\n${lyrics}` : 'lyrics:');

  const body = `${lines.join('\n')}\n`;

  // 5-6. Atomic write into the same directory
  const tmp = path.join(path.dirname(sidecarPath), `.sidecar_${crypto.randomBytes(6).toString('hex')}.tmp`);
  try {
    fs.writeFileSync(tmp, body, 'utf-8');
    fs.renameSync(tmp, sidecarPath);
  } catch (err) {
    try { fs.unlinkSync(tmp); } catch { /* already gone */ }
    throw err;
  }
}

// ── Merge ────────────────────────────────────────────────────────────────

/** `{ ... }`-ish caption blob detection (§6.5). */
export function looksLikeMappingBlob(v: string): boolean {
  if (!v) return false;
  const s = v.trim();
  if (s.startsWith('{') && (
    s.includes("'caption'") || s.includes('"caption"') || s.includes("'ok'") || s.includes('"ok"')
  )) return true;
  return /^caption\s*:\s*\{/i.test(s);
}

/** Best-effort JSON.parse of an object literal, retrying with single quotes
 *  normalised to double quotes (the JS stand-in for Python's literal_eval). */
export function parseLooseObject(raw: string): Record<string, unknown> | null {
  const s = (raw ?? '').trim();
  if (!s.startsWith('{')) return null;
  try {
    const parsed = JSON.parse(s);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed as Record<string, unknown>;
  } catch { /* fall through to the quote-normalised attempt */ }
  try {
    const normalised = s
      .replace(/([{,]\s*)'([^'\\]*)'(\s*:)/g, '$1"$2"$3')   // 'key': → "key":
      .replace(/:\s*'((?:[^'\\]|\\.)*)'/g, (_m, body: string) => `: "${body.replace(/"/g, '\\"')}"`)
      .replace(/\bNone\b/g, 'null')
      .replace(/\bTrue\b/g, 'true')
      .replace(/\bFalse\b/g, 'false');
    const parsed = JSON.parse(normalised);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed as Record<string, unknown>;
  } catch { /* not recoverable */ }
  return null;
}

/** Pull the real caption out of a mapping blob. `''` when nothing usable. */
export function extractCaptionFromBlob(v: string): string {
  if (!v) return '';
  let s = v.trim();
  const prefix = /^caption\s*:\s*/i.exec(s);
  if (prefix) s = s.slice(prefix[0].length).trim();

  const obj = parseLooseObject(s);
  if (obj) {
    for (const key of ['caption', 'ok']) {
      const val = obj[key];
      if (typeof val === 'string' && val.trim()) return val.trim();
    }
  }

  const hunted = /["']caption["']\s*:\s*["']([\s\S]*?)["']\s*(?:,|\})/.exec(s);
  if (hunted && hunted[1].trim()) return hunted[1].trim();

  return '';
}

/**
 * Apply `incoming` onto `existing` under a merge policy, returning the full
 * merged map. Falsy incoming values are skipped entirely — clearing a field is
 * an explicit user edit, never a merge outcome (§6.5).
 */
export function mergeFields(
  existing: Record<string, string>,
  incoming: Record<string, string>,
  policy: MergePolicy,
): Record<string, string> {
  const out: Record<string, string> = { ...existing };

  for (const [key, rawValue] of Object.entries(incoming)) {
    if (!rawValue) continue;

    let value = rawValue;
    if (key === 'caption' && looksLikeMappingBlob(value)) {
      value = extractCaptionFromBlob(value);
      if (!value.trim()) continue;
    }

    const occupied = !!(out[key] ?? '').trim();
    let write: boolean;
    if (policy === 'overwrite_all') write = GENERATED_FIELDS.has(key) || !occupied;
    else if (policy === 'overwrite_caption') write = key === 'caption' || !occupied;
    else if (policy === 'overwrite_lyrics') write = key === 'lyrics' || !occupied;
    else write = !occupied;

    if (write) out[key] = value;
  }

  return out;
}
