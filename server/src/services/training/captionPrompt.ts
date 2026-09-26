// training/captionPrompt.ts — Side-Step's 5-line structured caption prompt
//
// CAPTION_INSTRUCTIONS is a character-for-character copy of Side-Step's
// `sidestep_engine/data/caption_config.py::_DEFAULT_PROMPT_INSTRUCTIONS`, so a
// caption produced here is indistinguishable from one produced there.
//
// D14 said no provider in our registry accepts audio. That is no longer true on
// either side: Gemini takes an inlined MP3 (enhanceService.ts), and the local
// `moss` provider runs MOSS-Music-8B through ace-caption and captions straight
// from the waveform (mossCaption.ts). The text-only prompt below is now the
// FALLBACK — it carries `Audio attached to this request: no` plus the local
// analysis block, so the LLM rewrites from local evidence rather than listening.
//
// When MOSS drives this prompt it is passed verbatim via `--prompt-file`, which
// is why the wording must stay byte-identical to Side-Step's: the same string
// has to serve a cloud chat model and a local 8B, and drift in either direction
// would make captions from the two paths non-interchangeable in one dataset.
//
// Spec: docs/plans/2026-07-27-dataset-studio-implementation.md §4.10

import { parseLooseObject } from './sidecarIO.js';

/**
 * The dimensions an ACE-Step caption covers, in order.
 *
 * Lyric Studio's metadata planner reuses this so the captions it invents for a
 * new song describe the same things, in the same order, as the captions a sound
 * adapter was trained on (see `lireek/prompts.ts`). `CAPTION_INSTRUCTIONS` is
 * rebuilt from it, so the two can never drift.
 *
 * ── Why this replaced Side-Step's nine-sentence plan (2026-08-16) ────────────
 *
 * The previous version was a copy of Side-Step's `caption_config.py`, which
 * mandates EXACTLY nine sentences, one per prescribed dimension. Measured
 * against the 32 reference captions in ACE-Step's own
 * `examples/text2music/*.json`, that is nothing like what the model was trained
 * on:
 *
 *   reference captions   median 31 words, median 2 sentences (range 23-107 / 2-5)
 *   nine-sentence plan   ~150-200 words, nine sentences, always
 *   captions in the reference set with 9+ sentences   0 of 32
 *   captions in the reference set stating BPM or key  0 of 32
 *
 * The length was not the only harm. A nine-sentence quota is a heavy
 * instruction-following load, and on an 8B local captioner it measurably
 * crowded out the audio: MOSS described a 2Pac track as "a high-energy Electro
 * House track... four-on-the-floor... festival-oriented production" while its
 * OWN `genre:` field in the same response correctly said `Hip-Hop`. Asked for
 * two sentences of what it heard, there is no quota left to pad.
 */
export const CAPTION_DIMENSIONS: readonly string[] = [
  'genre and subgenre, named plainly',
  'the instruments actually present, named concretely',
  'vocal character (or state that the track is instrumental and name what carries the lead line)',
  'mood and atmosphere',
  'production style and sonic character',
];

export const CAPTION_INSTRUCTIONS: string =
  "Write music dataset metadata grounded in the song's audible content. If audio " +
  'is attached, describe what you actually HEAR and use title, artist, and lyrics ' +
  'only as weak secondary context.\n\n' +
  'Return EXACTLY 5 lines in plain text and nothing else. Each field must start at ' +
  'the beginning of its own new line. Never place two fields on the same line.\n\n' +
  // `genre:` comes FIRST, before the caption prose — this is load-bearing, not
  // cosmetic. Measured on MOSS-Music-8B over the 9-track americanfootball set
  // (2026-08-18): with caption first, the model opens with its stock phrase
  // "A brooding, atmospheric…" and that phrase drags the genre token toward
  // electronic/trap (teacher-forced, the reference's own top-3 after that
  // prefix is `electronic`/`instrumental`/`trap` within 0.25 logits — a coin
  // flip). Genre named before any prose exists to bias it: badly-wrong genres
  // went 6/9 -> 2/9 and the dark-trap/808/cloud-rap hallucinations vanished.
  // The HF reference fails identically with caption first, so this is a
  // property of the model, not of our GGML port.
  'Use this exact output template:\n' +
  // No example genre: a concrete example gets copied verbatim as the answer
  // when the model is unsure — 333 sidecars carried the literal phrase
  // 'bass house, electro house' after the 2026-08-18/19 bulk run. The trigger
  // was the broken whole-track encode (fixed in ace-caption --encoder-window),
  // but the fallback target was this line; a placeholder gives an unsure model
  // nothing to copy.
  'genre: <comma-separated genre/style tags, most specific first>\n' +
  'caption: <2 to 4 sentences on one line>\n' +
  'bpm: <estimated BPM as integer, e.g. 120>\n' +
  "key: <note plus lowercase mode, e.g. 'C minor' or 'F# major'>\n" +
  "signature: <numerator only — one of 2, 3, 4, 6>\n\n" +
  'Caption rules:\n' +
  // This rule is FIRST and stated as a requirement because a smaller local model
  // will otherwise drop the caption line entirely and answer with the metadata
  // fields alone — measured on MOSS-Music-8B, which returned exactly
  // "Genre: ...\nBPM: ...\nKey: ...\nSignature: ..." for 12 of 13 tracks. Leading
  // with the length limit reads as permission to be brief; leading with "the
  // caption is required" does not.
  '- Line 1 is REQUIRED and must begin with `genre:`. Line 2 is REQUIRED and must begin with `caption:` followed by the description. Never omit it, never leave it blank, and never answer with the metadata fields alone. If you are unsure of everything else, still write the caption.\n' +
  '- The caption is 2 to 4 sentences, roughly 25 to 60 words, on a single line. Reference captions average about 30 words; a longer caption is not a better one, and padding it with invented detail is worse than stopping.\n' +
  '- Cover these, woven into flowing description rather than listed:\n' +
  CAPTION_DIMENSIONS.map(s => `    - ${s}\n`).join('') +
  '- NEVER state BPM, key, or time signature in the caption text. They have dedicated fields below, and repeating them in the caption does not match how this model was trained.\n' +
  '- The genre you name in the caption MUST agree with the `genre:` field. Contradicting yourself between the two is worse than naming neither.\n' +
  // Gemini Flash captioned a whole metal album from its intros (2026-09-26):
  // a piano/woodblock intro became "minimalist neoclassical", a body-percussion
  // intro became "Ambient". The full track was uploaded; it over-weighted the start.
  '- Describe the WHOLE track, weighted by how much of it each part occupies. Intros, outros, interludes and breakdowns are often unrepresentative: a quiet piano or percussion intro before a heavy song does not make the song ambient or classical. Base the genre and caption on the style that dominates most of the running time; mention a contrasting intro or outro only as a secondary detail.\n' +
  '- Name things concretely: `808 bass`, `brushed snare`, `detuned saw lead`, `palm-muted guitar`, `upright piano` — not `interesting textures` or `lush soundscapes`.\n' +
  "- No vague imagery or stacked adjectives ('neon skies, electric hearts'), no marketing copy, and no listener-reaction language ('keeps you moving', 'emotionally resonant').\n" +
  "- Avoid generic openings like 'This track is' when more specific wording can be used immediately.\n" +
  '- If the track is instrumental, say so and name the instrument carrying the lead line.\n' +
  '- Start `genre:` on line 1, `caption:` on line 2, `bpm:` on line 3, `key:` on line 4, and `signature:` on line 5.\n' +
  '- Do not merge fields together. For example, do not output `genre: ... bpm: ... key: ...` on one line.\n' +
  '- Do not use markdown, bullets, numbering, code fences, labels before the template, or commentary after the template.\n' +
  '- Do not mention the artist name or song title in the caption.\n' +
  '- If audio is not attached or a field cannot be determined from available evidence, write N/A for that field instead of guessing.';

/** Gen params Side-Step uses for the caption call. */
export const CAPTION_TEMPERATURE = 0.45;
export const CAPTION_TOP_P = 0.9;

export interface UserPromptArgs {
  title: string;
  artist: string;
  lyricsExcerpt?: string;
  audioAttached: boolean;
  localCaption?: string;
  bpm?: number | null;
  key?: string;
  signature?: string;
}

/** Side-Step's user prompt, extended with the local-label block (D14). */
export function buildUserPrompt(args: UserPromptArgs): string {
  const lines: string[] = [];
  lines.push(CAPTION_INSTRUCTIONS);
  lines.push('');
  lines.push('Song metadata:');
  lines.push(`Audio attached to this request: ${args.audioAttached ? 'yes' : 'no'}`);
  lines.push(`Title: ${args.title || 'unknown'}`);
  lines.push(`Artist: ${args.artist || 'unknown'}`);
  lines.push(
    `Local analysis: bpm ${args.bpm ?? 'unknown'}, key ${args.key || 'unknown'}, ` +
    `signature ${args.signature || 'unknown'}`,
  );
  if (!args.audioAttached) {
    // Text-only mode leans on the local caption. With real audio attached it is
    // OMITTED entirely — a weak local caption anchors the model on wrong genres.
    lines.push('Existing caption (from local audio analysis — treat as primary audible evidence):');
    lines.push(args.localCaption || '(none)');
  }
  if (args.lyricsExcerpt && args.lyricsExcerpt.trim()) {
    lines.push(`Lyrics excerpt:\n${args.lyricsExcerpt.slice(0, 500)}`);
  }
  lines.push('');
  lines.push(args.audioAttached
    ? 'Reminder: describe what you HEAR in the attached audio; metadata and lyrics are secondary context only.'
    : 'Reminder: audible evidence comes first; metadata and lyrics are secondary.');
  return lines.join('\n');
}

// ── Response parsing ─────────────────────────────────────────────────────

type CaptionField = 'caption' | 'genre' | 'bpm' | 'key' | 'signature';
const FIELDS: readonly CaptionField[] = ['caption', 'genre', 'bpm', 'key', 'signature'];

type CaptionFields = Partial<Record<CaptionField, string>>;

function cleanValue(raw: string): string {
  let v = raw.trim();
  v = v.replace(/^["'`]+/, '').replace(/["'`,]+$/, '');
  return v.trim();
}

function isNA(v: string): boolean {
  return v.trim().toLowerCase() === 'n/a';
}

function fromMapping(obj: Record<string, unknown>): CaptionFields {
  const out: CaptionFields = {};
  for (const field of FIELDS) {
    const v = obj[field];
    if (v === null || v === undefined) continue;
    const s = typeof v === 'string' ? v : String(v);
    if (s.trim()) out[field] = cleanValue(s);
  }
  return out;
}

/** Regex hunt through a blob that would not parse as an object. */
function huntBlob(text: string): CaptionFields {
  const out: CaptionFields = {};
  for (const field of FIELDS) {
    const re = new RegExp(`["']?${field}["']?\\s*:\\s*["']([\\s\\S]*?)["']\\s*(?:,|\\}|$)`, 'i');
    const m = re.exec(text);
    if (m && m[1].trim()) out[field] = cleanValue(m[1]);
  }
  return out;
}

/** Plaintext label scan — the normal, well-behaved path. */
function scanLabels(text: string): CaptionFields {
  const out: CaptionFields = {};
  const re = /(?<!\w)(caption|genre|bpm|key|signature)\s*:/gi;
  const hits: Array<{ field: CaptionField; start: number; end: number }> = [];
  for (const m of text.matchAll(re)) {
    hits.push({
      field: m[1].toLowerCase() as CaptionField,
      start: m.index ?? 0,
      end: (m.index ?? 0) + m[0].length,
    });
  }
  if (hits.length === 0) return out;

  for (let i = 0; i < hits.length; i++) {
    const hit = hits[i];
    if (out[hit.field] !== undefined) continue;   // first occurrence wins
    const stop = i + 1 < hits.length ? hits[i + 1].start : text.length;
    const value = cleanValue(text.slice(hit.end, stop));
    if (value) out[hit.field] = value;
  }

  // Tail recovery: prose ahead of the first label is the caption when the
  // response never labelled one.
  if (!out.caption) {
    const head = cleanValue(text.slice(0, hits[0].start));
    if (head) out.caption = head;
  }
  return out;
}

function dropNA(fields: CaptionFields): CaptionFields {
  const out: CaptionFields = {};
  for (const field of FIELDS) {
    const v = fields[field];
    if (v === undefined) continue;
    if (!v.trim() || isNA(v)) continue;
    out[field] = v;
  }
  return out;
}

/**
 * Six-step resolution chain (§4.10): mapping input → JSON object string →
 * blob-regex hunt → plaintext label scan (+ tail recovery) → drop "n/a" →
 * total-failure fallback of `{ caption: raw }`.
 */
export function parseStructuredResponse(raw: string): CaptionFields {
  const input: unknown = raw;

  // 1. Already a mapping (some providers hand back parsed JSON)
  if (input && typeof input === 'object' && !Array.isArray(input)) {
    return dropNA(fromMapping(input as Record<string, unknown>));
  }
  if (typeof input !== 'string') return {};

  const text = input.trim();
  if (!text) return {};

  // 2. A `{...}`-shaped string
  if (text.startsWith('{')) {
    const obj = parseLooseObject(text);
    if (obj) {
      const mapped = dropNA(fromMapping(obj));
      if (Object.keys(mapped).length > 0) return mapped;
    }
    // 3. Blob-regex hunt when the object would not parse
    const hunted = dropNA(huntBlob(text));
    if (Object.keys(hunted).length > 0) return hunted;
  }

  // 4. Plaintext label scan
  const scanned = dropNA(scanLabels(text));
  if (Object.keys(scanned).length > 0) return scanned;

  // 6. Total failure — hand the whole response back as the caption
  return { caption: text };
}
