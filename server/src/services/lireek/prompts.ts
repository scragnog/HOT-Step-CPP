// prompts.ts — CANONICAL single source of truth for all Lyric Studio prompts.
//
// Consumed by BOTH:
//   - the in-app pipeline (llm/orchestration.ts, profilerService.ts)
//   - the MCP server (tools/mcp-lyricstudio/src/prompts.ts re-exports this file)
// Do NOT fork these prompts elsewhere — edit them here.
//
// Prompt philosophy: rules are split into two tiers.
//   TIER 1 (pipeline requirements) — mechanical constraints of the downstream
//     music model (section headers, punctuation, line budget). Hard.
//   TIER 2 (style targets) — artist-calibration guidance. The writer has
//     explicit creative latitude here; screaming MUST at a capable model
//     produces formulaic lyrics, which the anti-slop rules then fight.

import { BLACKLISTED_WORDS, BLACKLISTED_PHRASES, OVERUSED_WORDS } from './slopDetector.js';
import { stripLyricQuotes } from './llm/postprocess.js';
// The nine-sentence plan the dataset captions were written to. Imported rather
// than restated so a planned caption and a training caption stay one format.
import { CAPTION_DIMENSIONS } from '../training/captionPrompt.js';

/** Loose profile shape — accepts both the app's LyricsProfile and the MCP's raw profile_data JSON. */
export type PromptProfile = Record<string, any>;

// ── Blueprint helpers ───────────────────────────────────────────────────────

// ── Section-tag vocabulary ──────────────────────────────────────────────────
//
// Taken from the BASE MODEL's own documentation — ACE-Step-1.5 docs/en
// Tutorial.md and ace_step_musicians_guide.md — not invented here. Those docs
// list the tags below and state that "structure tags can be combined with `-`
// for finer control" ([Chorus - anthemic], [Bridge - whispered]), with one
// caution: "Don't stack too many tags."
//
// Two corrections this encodes (2026-08-08), both measured over the 1487
// existing generations:
//
//   * [Build], [Drop] and [Breakdown] are DOCUMENTED base tags, but the prompts
//     used to ban them as "invented labels". Result: zero uses of Build, Drop,
//     Breakdown, Instrumental, Guitar Solo, Piano Interlude, Fade Out and
//     Silence across all 1487 songs — the entire dynamic and instrumental half
//     of the vocabulary was suppressed. Those are exactly the markers an
//     electronic/metal arrangement needs.
//   * [Instrumental Break: Guitar Solo] was given as an EXAMPLE, so it was
//     copied 54 times. Neither the head nor the colon is documented; the docs
//     specify a dash and the head [Instrumental] or [Guitar Solo].
//
// [Post-Chorus] and [Interlude] are not in the docs but appear 408 and 251
// times in the real-song training lyrics, so they are demonstrably understood
// and stay permitted.
export const SECTION_LABEL_RULE: string =
  '- SECTION HEADERS use square brackets. Recognised labels: [Intro], [Verse 1], [Verse 2], [Verse 3], ' +
  '[Pre-Chorus], [Chorus], [Post-Chorus], [Bridge], [Interlude], [Outro], ' +
  '[Build], [Drop], [Breakdown], [Instrumental], [Guitar Solo], [Piano Interlude], [Fade Out], [Silence].\n' +
  '  [Build], [Drop] and [Breakdown] are first-class labels — use them where the arrangement calls for them, ' +
  'especially in electronic, dance and metal styles.\n' +
  '  You may append ONE short performance annotation after a DASH: [Chorus - High Energy], ' +
  '[Bridge - Sparse and Quiet], [Instrumental - Saxophone Solo]. Never use a colon ([Instrumental Break: Guitar Solo] is wrong), ' +
  'and do not stack several annotations on one tag.\n' +
  '  Do NOT invent labels like [X], [Hook] or [Solo] — for a solo use [Guitar Solo] or [Instrumental - <instrument> Solo].\n' +
  '  INTROS: most records play a few bars before the first vocal, so DEFAULT to opening the song with an empty [Intro - Instrumental] — header, no lyric lines. Roughly one song in five should instead open straight on [Verse 1]; make that a deliberate choice for songs that want to slam in. If the song opens on SUNG material, tag it [Intro] and write the words.\n' +
  '  NEVER write a BARE [Intro] or [Outro] with no lyric lines and no descriptor. The descriptor is what tells the music model to PLAY there rather than sing; without it an empty boundary tag is an open invitation the model fills with an arbitrarily long instrumental of its own choosing (a bare [Intro] once rendered 105 seconds of riffing). [Intro - Instrumental] and [Outro - Instrumental] are correct; a naked [Intro] above a blank line is not.';

export const BLUEPRINT_LABEL_NAMES: Record<string, string> = {
  V: 'Verse', C: 'Chorus', B: 'Bridge', PC: 'Pre-Chorus',
  POC: 'Post-Chorus', I: 'Intro', O: 'Outro', IL: 'Interlude',
};

const BLUEPRINT_CODES_LEGEND =
  'I=Intro, V=Verse, PC=Pre-Chorus, C=Chorus, POC=Post-Chorus, B=Bridge, IL=Interlude, O=Outro';

const FALLBACK_BLUEPRINT = 'V-C-V-C-B-C';

/** Parse a structure string (e.g. "I-V-C-V-C-B-C-O") into a canonical blueprint, or null if unusable. */
export function normalizeBlueprint(structure?: string | null): string | null {
  if (!structure) return null;
  const tokens = structure.toUpperCase().split(/[-,>\s]+/).filter(Boolean);
  const valid = tokens.filter(t => BLUEPRINT_LABEL_NAMES[t]);
  if (valid.length < 3 || !valid.includes('C')) return null;
  return valid.join('-');
}

/** Dedupe + sanity-filter observed blueprints (truncating anything after the first Outro). */
function cleanBlueprints(blueprints?: string[]): string[] {
  if (!blueprints?.length) return [];
  return [...new Set(blueprints.map(bp => {
    const parts = bp.split('-');
    const oi = parts.indexOf('O');
    return (oi >= 0 ? parts.slice(0, oi + 1) : parts).join('-');
  }))].filter(bp => {
    const parts = bp.split('-');
    return parts.length >= 3 && parts.includes('C') && parts.every(p => BLUEPRINT_LABEL_NAMES[p]);
  });
}

/**
 * Sample a blueprint from the artist's observed structures.
 * Unlike the old selectBestBlueprint (deterministic argmax that gave every
 * generation for an artist the identical structure), this keeps every observed
 * structure in play — richer shapes (more distinct sections, a bridge) just get
 * a mild edge. Pass `rand` for deterministic tests.
 */
export function pickBlueprint(blueprints?: string[], rand: () => number = Math.random): string {
  const cleaned = cleanBlueprints(blueprints);
  if (!cleaned.length) return FALLBACK_BLUEPRINT;
  const weights = cleaned.map(bp => {
    const parts = bp.split('-');
    return 1 + new Set(parts).size * 0.5 + (parts.includes('B') ? 1 : 0);
  });
  const total = weights.reduce((a, b) => a + b, 0);
  let r = rand() * total;
  for (let i = 0; i < cleaned.length; i++) {
    r -= weights[i];
    if (r <= 0) return cleaned[i];
  }
  return cleaned[cleaned.length - 1];
}

/** Expand "I-V-C-V-C-O" into ["[Intro]", "[Verse 1]", "[Chorus]", ...]. */
export function blueprintToSections(bp: string): string[] {
  let verseNum = 0;
  return bp.split('-').map(part => {
    if (part === 'V') { verseNum++; return `[Verse ${verseNum}]`; }
    return `[${BLUEPRINT_LABEL_NAMES[part] || part}]`;
  });
}

// ── Album audio enrichment ──────────────────────────────────────────────────
//
// A lyrics set exported from the Training Studio carries per-song audio truth
// measured from the source recordings: caption + genre (Gemini, audio-grounded)
// and bpm/key/signature (Essentia). When present, that data grounds the
// metadata planner (BPM range, key palette, genre, caption format) far better
// than the LLM's guess — and matching the training captions' format keeps a
// sound adapter trained on the same album accurate. Sets without enrichment
// (plain Genius fetches) get null here and every consumer skips silently.

/** Per-example ceiling for the verbatim training captions shown to the planner. */
const CAPTION_EXAMPLE_MAX_CHARS = 2000;

// ── Vocal pacing ────────────────────────────────────────────────────────────
//
// The LM renders a song at EXACTLY its stated duration (98.4% of 1248 logged
// runs stop within ±1s), so duration and lyric word count must agree or the
// song either cuts off mid-phrase or pads with improvised filler. The
// conversion rate is words-per-second of TOTAL duration — measured over 2361
// real vocal songs it already prices in intros, solos and outros.
//
// CRITICAL: pacing is an ARTIST property, not a universal constant. Per-artist
// medians run 0.51 w/s (Muse) to 3.29 (a fast rapper) — 6.5x — so always prefer the
// artist's own measured rate (AlbumEnrichment.wordsPerSec) and use this global
// median only when no measured rate exists.
export const GLOBAL_WORDS_PER_SECOND = 1.20;

// The vocal-idle floor. An artist's measured words/TOTAL-duration bakes their
// INSTRUMENTAL share into the rate — and the model cannot render instrumental
// time it is not told about. Deriving duration = words / 0.71 for Funeral For
// A Friend stretched 235 words over 331s and the extra ~2 minutes rendered as
// aimless looping riffs (Rob, 2026-08-08 — the first real listen after the
// per-artist retime). Real FFAF fills that time with COMPOSED instrumental
// passages; a generation's only channel for those is declared section tags.
// So for DURATION DERIVATION the effective rate is floored here: undeclared
// time is filler risk, not artistry. Declared instrumental sections still add
// real time on top via the allowance.
//
// 1.25 is EAR-MEASURED, not assumed (2026-08-08 evening, n=2 but tight): with
// the empty [Intro] stripped, the model sang 235 words in ~187s (1.26 w/s) and
// 252 in ~205s (1.23 w/s), parking ALL remaining duration in the declared
// instrumental outro — a 60s and a 30s tail respectively. The first floor of
// 0.95 was still 25% below the model's real singing rate, and that 25% is
// exactly the aimless tail Rob heard. The model sings at ~1.25 regardless of
// the artist's TOTAL-duration rate; artists measured faster than 1.25
// (rap: a fast rapper 3.29) carry their density in the lyrics themselves, which is
// why the floor is max(), not a constant.
export const VOCAL_FLOOR_WORDS_PER_SECOND = 1.25;

// How much of a song may be INSTRUMENTAL, as a multiple of its sung time.
//
// The 1.25 floor above is the rate the model SINGS at, and treating it as the
// rate the song RUNS at was an error: it priced every artist's instrumental
// time at zero. Measured against 2009 real recordings (real lyrics, real
// durations), `words / max(rate, 1.25)` reproduces the length of artists at or
// above 1.25 w/s (median 1.00x) but shortens everyone below it (median 0.77x,
// worst 0.26x) — and 90 of 160 albums measure below it. That is where the 1:38
// Muse songs and 1:47 Pink Floyd songs came from.
//
// An artist's measured words/TOTAL-duration rate already IS their real total
// length for a given word count, so `words / rate` is the unbiased answer
// (median 1.00x across the corpus). It is capped here rather than used raw
// because the extreme low rates are the ones that produce aimless filler:
// Muse at 0.51 w/s would demand a song that is 59% empty. At 1.5 no song is
// more than a third instrumental, which recovers essentially all of the bias
// (median 0.99x, median error 2s vs 25s today — 1.6 buys 1s more and nothing
// past that changes) while keeping a hard ceiling on undeclared empty time.
export const INSTRUMENTAL_FACTOR_CAP = 1.5;

// The effective floor on an artist's TOTAL-duration rate — the two constants
// above expressed as the single number both the word budget and the duration
// derivation are built from. Keeping them derived from one another is the
// point: when the word target and the duration used different floors, a
// compliant writer still produced a song whose length disagreed with its own
// word count by the ratio between them.
export const DURATION_RATE_FLOOR = VOCAL_FLOOR_WORDS_PER_SECOND / INSTRUMENTAL_FACTOR_CAP;

const SECTION_TAG_LINE = /^[ \t]*\[[^\]]{1,60}\][ \t]*$/;

/** Words of singable lyric in a lyrics text — section-tag lines excluded. */
export function countLyricWords(lyrics: string): number {
  let words = 0;
  for (const line of String(lyrics ?? '').split(/\r?\n/)) {
    const t = line.trim();
    if (!t || SECTION_TAG_LINE.test(t)) continue;
    words += t.split(/\s+/).filter(Boolean).length;
  }
  return words;
}

/** Section heads that occupy time without carrying lyrics. Bare [Intro]/[Outro]
 *  are deliberately absent — an EMPTY bare boundary tag is an open invitation
 *  the model fills from its adapter prior (a bare [Intro] on a Funeral For A
 *  Friend song rendered 105s of riffing). But a boundary tag with an
 *  explicitly instrumental descriptor ([Outro - Instrumental], [Intro - Guitar
 *  Feedback]) is a DECLARED section and earns allowance time like any other. */
const INSTRUMENTAL_HEADS = /^(instrumental|guitar solo|piano interlude|build|drop|breakdown|interlude|fade out)$/i;
const INSTRUMENTAL_BOUNDARY = /^(intro|outro)$/i;
const INSTRUMENTAL_DESC = /instrumental|solo|breakdown|build|riff|feedback|jam|ambient|drum/i;

/** Bars a declared empty instrumental section is worth, by head. A flat 8 bars
 *  for everything was too small for the passages that carry real weight — a
 *  guitar solo or a breakdown is a 16-bar event, a bookend or a turnaround is
 *  8, a fade is 4. At 120 BPM that is 32s / 16s / 8s. */
function instrumentalSectionBars(head: string): number {
  if (/^(guitar solo|instrumental|breakdown)$/i.test(head)) return 16;
  if (/^fade out$/i.test(head)) return 4;
  return 8;   // build, drop, interlude, piano interlude, declared intro/outro
}

/** Seconds of DECLARED empty instrumental time in a lyric — the passages the
 *  writer explicitly asked for, which the model renders at its learned size
 *  rather than filling with improvisation. */
export function declaredInstrumentalSeconds(lyrics: string, bpm: number): number {
  const barSeconds = bpm > 0 ? 240 / bpm : 2.0;
  const lines = String(lyrics ?? '').split(/\r?\n/);
  let bars = 0;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim();
    if (!SECTION_TAG_LINE.test(t)) continue;
    const [head, ...descParts] = t.slice(1, -1).split(/\s+[-–—]\s+/).map(s => s.trim());
    const declared = INSTRUMENTAL_HEADS.test(head) ||
      (INSTRUMENTAL_BOUNDARY.test(head) && descParts.length > 0 && INSTRUMENTAL_DESC.test(descParts.join(' ')));
    if (!declared) continue;
    let hasLyric = false;
    for (let j = i + 1; j < lines.length; j++) {
      const u = lines[j].trim();
      if (SECTION_TAG_LINE.test(u)) break;
      if (u) { hasLyric = true; break; }
    }
    if (!hasLyric) bars += instrumentalSectionBars(head);
  }
  return bars * barSeconds;
}

/**
 * The duration the written lyrics actually need, at this artist's pacing.
 *
 * Called AFTER the lyrics are final, because the writer LLM cannot reliably
 * count its own words while writing — the word target gets it near, this makes
 * the duration==content invariant hold by construction.
 *
 * Returns the planned duration unchanged when the two already agree within
 * 15s (don't churn the planner's musical intent over noise), else the
 * lyric-derived duration clamped to [60, 480]. The upper bound is 480 rather
 * than 400 because the instrumental factor legitimately pushes long songs by
 * slow artists past 400s — the corpus has real 7-minute Pink Floyd and
 * Pendulum tracks, and clipping them at 400 would reintroduce the same
 * truncation this change exists to remove.
 */
export function reconcileDurationToLyrics(
  lyrics: string, bpm: number, plannedDuration: number, wordsPerSec?: number,
): number {
  const derived = lyricsDurationSeconds(lyrics, bpm, wordsPerSec);
  if (!derived) return plannedDuration;
  if (plannedDuration > 0 && Math.abs(derived - plannedDuration) <= 15) return plannedDuration;
  return Math.max(60, Math.min(480, derived));
}

/** The UNCLAMPED seconds the written lyrics need at the given pacing —
 *  reconcileDurationToLyrics without the stability window or bounds. 0 when
 *  there are no lyric words. Exposed so batch tooling can see the true need
 *  (a derived 430s that the clamp would hide is a "regenerate the lyrics"
 *  signal, not a retiming). */
export function lyricsDurationSeconds(lyrics: string, bpm: number, wordsPerSec?: number): number {
  const rate = wordsPerSec && wordsPerSec > 0 ? wordsPerSec : GLOBAL_WORDS_PER_SECOND;
  const words = countLyricWords(lyrics);
  if (!words) return 0;

  // A song is SUNG time plus INSTRUMENTAL time, and the two are measured
  // differently. Sung time comes from the model's own ear-measured sing rate
  // (1.25 w/s) — that part of the old formula was right and is unchanged.
  const sung = words / Math.max(rate, VOCAL_FLOOR_WORDS_PER_SECOND);

  // Instrumental time comes from either of two sources, whichever is larger —
  // never their sum, which would double-count. The artist's own rate is words
  // over TOTAL duration, so it ALREADY prices in their typical intro, solo and
  // outro; adding declared sections on top of it charges for the same bars
  // twice. Taking the max instead means the artist's measured shape is the
  // baseline, and a writer who explicitly asks for MORE than that (three
  // breakdowns, a long declared outro) gets the extra time.
  const artistTotal = words / Math.max(rate, DURATION_RATE_FLOOR);
  const declaredTotal = sung + declaredInstrumentalSeconds(lyrics, bpm);

  return Math.round(Math.max(artistTotal, declaredTotal));
}

// ── Instrumental intro policy ───────────────────────────────────────────────
//
// Most records let the music establish itself before the first vocal. The
// 2026-08-09 boundary-tag sweep removed every empty [Intro] because a BARE one
// is an open invitation the model fills at whatever size its adapter prefers
// (105s of riffing on one FFAF song). That was right about the bare tag and
// wrong about the conclusion: with no tag at all the vocal now starts at 0s on
// essentially every song. The fix is the DESCRIPTORED form — [Intro -
// Instrumental] declares both that there is an intro and that nothing is sung
// in it, which is the one shape the ear tests showed the model renders at a
// sane, learned size.
//
// Not every song, though: some records genuinely slam straight into the first
// verse. The share below is applied deterministically per song (hashed from a
// stable seed, not random) so the same song always makes the same choice and
// a re-run is reproducible.
export const INSTRUMENTAL_INTRO_SHARE = 0.8;

export const INSTRUMENTAL_INTRO_TAG = '[Intro - Instrumental]';

/** Stable 0..1 from a seed string (FNV-1a). Deterministic across processes —
 *  Math.random() here would make re-runs of the migration disagree. */
function seedFraction(seed: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h / 0x100000000;
}

/**
 * Give a lyric an explicit instrumental intro, unless it already opens with an
 * intro of its own or the seed puts it in the ~20% that start on the vocal.
 *
 * Three cases, in order:
 *  - opens with an [Intro...] carrying sung lines  → untouched (a sung intro
 *    is a deliberate arrangement, not a missing one)
 *  - opens with an EMPTY [Intro] (bare or descriptored non-instrumentally) →
 *    upgraded to [Intro - Instrumental], the form the model handles sanely
 *  - opens with anything else → the tag is inserted, subject to the share
 *
 * Returns null when nothing changed.
 */
export function ensureInstrumentalIntro(lyrics: string, seed: string): string | null {
  const text = String(lyrics ?? '');
  if (!text.trim()) return null;
  const lines = text.split(/\r?\n/);

  // First section tag in the lyric, and whether anything is sung under it.
  let firstTag = -1;
  for (let i = 0; i < lines.length; i++) {
    if (SECTION_TAG_LINE.test(lines[i].trim())) { firstTag = i; break; }
    if (lines[i].trim()) break;   // sung line before any tag — no header to read
  }

  if (firstTag >= 0) {
    const label = lines[firstTag].trim().slice(1, -1);
    const [head, ...descParts] = label.split(/\s+[-–—]\s+/).map(s => s.trim());
    if (INSTRUMENTAL_BOUNDARY.test(head) && /^intro$/i.test(head)) {
      let hasLyric = false;
      for (let j = firstTag + 1; j < lines.length; j++) {
        const u = lines[j].trim();
        if (SECTION_TAG_LINE.test(u)) break;
        if (u) { hasLyric = true; break; }
      }
      if (hasLyric) return null;                                    // sung intro, leave it
      const desc = descParts.join(' ');
      if (desc && INSTRUMENTAL_DESC.test(desc)) return null;        // already declared
      lines[firstTag] = INSTRUMENTAL_INTRO_TAG;                     // upgrade bare/vague
      return lines.join('\n');
    }
  }

  if (seedFraction(seed) >= INSTRUMENTAL_INTRO_SHARE) return null;  // the ~20% with no intro

  const at = firstTag >= 0 ? firstTag : 0;
  lines.splice(at, 0, INSTRUMENTAL_INTRO_TAG, '');
  return lines.join('\n');
}

export interface AlbumEnrichment {
  bpmMin: number;               // 0 when no track had a BPM
  bpmMax: number;
  keys: string[];               // unique, most-frequent first
  genres: string[];             // unique tags (comma-split), most-frequent first
  signatures: string[];         // unique, most-frequent first
  captionExamples: string[];    // up to 3 verbatim training captions
  /** Up to 3 of the album's own one-sentence YuE2 captions (`<stem>.yue2.txt`),
   *  when the dataset has been captioned for YuE2. Empty otherwise. */
  yue2CaptionExamples: string[];
  enrichedSongs: number;        // songs carrying at least one enriched field
  totalSongs: number;
  /** This artist's measured vocal pacing: median words-per-second over songs
   *  carrying both lyrics and a real duration. 0 = unknown (fall back to
   *  GLOBAL_WORDS_PER_SECOND). Per-artist medians span 0.51 (Muse) to 3.29
   *  (a fast rapper) — a 6.5x spread — so a global constant misprices most artists. */
  wordsPerSec: number;
  /** How many songs the pacing median was computed over. */
  pacedSongs: number;
}

/**
 * Canonical key spelling: note letter upper-case, mode lower-case — "Bb minor".
 *
 * The engine's metadata FSM builds its keyscale vocabulary from
 * `modes[] = { "major", "minor" }` (engine/src/metadata-fsm.h), 70 values, all
 * lower-case. "D Major" is therefore OUT OF VOCABULARY, not merely untidy.
 *
 * The dataset sidecars store the mode capitalised ("E Minor", "C# Major"), so
 * an album's measured key list reached the metadata planner in a form the
 * engine cannot accept — while the system prompt two lines away demanded
 * lower-case. The planner reasonably copied the examples it could see, which is
 * how 1,487 of 1,493 stored generations ended up with a capitalised mode.
 * Normalising here fixes every consumer of AlbumEnrichment at once.
 */
export function normalizeKeyScale(key?: string | null): string {
  const m = String(key ?? '').trim().match(/^([A-Ga-g])\s*([#b]|♯|♭)?\s+([Mm]ajor|[Mm]inor)$/);
  if (!m) return String(key ?? '').trim();
  return `${m[1].toUpperCase()}${m[2] ?? ''} ${m[3].toLowerCase()}`;
}

/** Unique non-empty values ordered by frequency (ties keep first-seen order). */
function freqRank(values: string[]): string[] {
  const counts = new Map<string, { display: string; n: number; at: number }>();
  values.forEach((raw, i) => {
    const v = String(raw ?? '').trim();
    if (!v) return;
    const k = v.toLowerCase();
    const hit = counts.get(k);
    if (hit) hit.n++;
    else counts.set(k, { display: v, n: 1, at: i });
  });
  return [...counts.values()].sort((a, b) => b.n - a.n || a.at - b.at).map(c => c.display);
}

/**
 * Distill one lyrics set's per-song enrichment into album-level facts.
 * null when NO song carries any enriched field — callers must then behave
 * exactly as before enrichment existed.
 */
export function computeAlbumEnrichment(
  songs: Array<Record<string, any>> | null | undefined,
): AlbumEnrichment | null {
  if (!Array.isArray(songs) || !songs.length) return null;

  const bpms: number[] = [];
  const keys: string[] = [];
  const genres: string[] = [];
  const signatures: string[] = [];
  const captions: string[] = [];
  const yue2Captions: string[] = [];
  const paceRates: number[] = [];
  let enriched = 0;

  for (const s of songs) {
    if (!s || typeof s !== 'object') continue;
    const bpm = Number(s.bpm);
    const key = typeof s.key === 'string' ? s.key.trim() : '';
    const genre = typeof s.genre === 'string' ? s.genre.trim() : '';
    const signature = typeof s.signature === 'string' ? s.signature.trim() : '';
    const caption = typeof s.caption === 'string' ? s.caption.trim() : '';
    if (typeof s.yue2Caption === 'string' && s.yue2Caption.trim()) yue2Captions.push(s.yue2Caption.trim());
    // Vocal pacing needs BOTH real lyrics and a real duration on the same song.
    const dur = Number(s.duration);
    if (Number.isFinite(dur) && dur > 30 && typeof s.lyrics === 'string') {
      const words = countLyricWords(s.lyrics);
      const rate = words / dur;
      // Per-song clamp guards against corrupt metadata (one dataset measures a
      // physically impossible 8 w/s — bad sidecar durations, not fast rapping).
      if (words >= 40 && rate >= 0.3 && rate <= 3.6) paceRates.push(rate);
    }
    const has = (Number.isFinite(bpm) && bpm > 0) || !!key || !!genre || !!caption || !!signature;
    if (!has) continue;
    enriched++;
    if (Number.isFinite(bpm) && bpm > 0) bpms.push(Math.round(bpm));
    if (key) keys.push(normalizeKeyScale(key));
    // Genre fields are usually comma-separated tag lists — count each tag.
    if (genre) genres.push(...genre.split(',').map(g => g.trim()).filter(Boolean));
    if (signature) signatures.push(signature);
    if (caption) captions.push(caption);
  }

  if (!enriched && !paceRates.length) return null;

  // Median over at least 4 songs; fewer is too noisy to trust over the global.
  let wordsPerSec = 0;
  if (paceRates.length >= 4) {
    const sorted = [...paceRates].sort((a, b) => a - b);
    wordsPerSec = Math.round(sorted[Math.floor(sorted.length / 2)] * 100) / 100;
  }

  return {
    bpmMin: bpms.length ? Math.min(...bpms) : 0,
    bpmMax: bpms.length ? Math.max(...bpms) : 0,
    keys: freqRank(keys),
    genres: freqRank(genres),
    signatures: freqRank(signatures),
    // Cap generously, not tightly. These examples are read off the CORPUS, so
    // the cap has to clear whatever style that corpus holds — and the corpus is
    // mid-migration: captions written before 2026-08-16 follow Side-Step's
    // nine-sentence plan at ~1100-1600 chars, while new ones follow ACE-Step's
    // own reference style at ~150-350. A tight cap amputates the old ones, and
    // the planner faithfully copies the amputation: truncating to 500 produced
    // captions back at ~520 chars with no structure at all. Leave it generous
    // until the corpus is fully recaptioned.
    captionExamples: [...new Set(captions)].slice(0, 3).map(c => c.slice(0, CAPTION_EXAMPLE_MAX_CHARS)),
    yue2CaptionExamples: [...new Set(yue2Captions)].slice(0, 3),
    enrichedSongs: enriched,
    totalSongs: songs.length,
    wordsPerSec,
    pacedSongs: paceRates.length,
  };
}

// ── Caption re-planning ─────────────────────────────────────────────────────
//
// Rewrites the caption of an EXISTING generation into the nine-sentence format
// the sound adapters were actually trained on, leaving its lyrics, title,
// subject, bpm, key and duration untouched. Used by the bulk migration of
// captions written before the 2026-08-08 format fix, which came out at
// 258-986 chars against training captions of 1149-1537.
//
// The caption describes music that does NOT exist yet, so the only evidence is
// the lyrics — above all their section tags, which are what sentences 7-9 have
// to be derived from. A weak model's failure mode here is reaching for "the
// drop" and "the breakdown" on a ballad; the rules below name that explicitly.

export const CAPTION_REPLAN_SYSTEM_PROMPT = `You write music-dataset captions that condition an AI music generator.

You will be given an artist's real training captions, then a batch of planned songs. For each song write ONE new caption in exactly the same format as those training captions.

Return ONLY a JSON object mapping each song id to its caption string:
{"123": "<caption>", "124": "<caption>"}

No markdown, no code fences, no commentary.`;

/** One song whose caption is being re-planned. Lyrics carry the section tags. */
export interface CaptionReplanSong {
  id: number;
  title: string;
  subject?: string | null;
  bpm?: number | null;
  key?: string | null;
  duration?: number | null;
  lyrics: string;
}

/**
 * Build the user prompt for a batch of songs from ONE lyrics set.
 *
 * Batching per set is deliberate: the caption examples are the bulk of the
 * prompt and are identical for every song in the set, so sending them once per
 * set instead of once per song is a large token saving across ~190 sets.
 */
export function buildCaptionReplanPrompt(
  profile: PromptProfile, songs: CaptionReplanSong[], captionExamples?: string[],
): string {
  const enrich: AlbumEnrichment | null = profile.audio_enrichment ?? null;
  const examples = captionExamples?.length ? captionExamples : (enrich?.captionExamples ?? []);
  const lines: string[] = [];

  if (examples.length) {
    lines.push("REAL TRAINING CAPTIONS — these describe the actual recordings this artist's sound adapter was trained on. Match their format, register, vocabulary and level of detail exactly:", '');
    examples.forEach((c, i) => lines.push(`  ${i + 1}. "${c}"`, ''));
  }
  if (enrich) lines.push(...formatAlbumEnrichment(enrich), '');

  lines.push(
    'CAPTION RULES (all mandatory):',
    '- ONE line of 2 to 4 prose sentences, roughly 25-60 words. Not a comma-separated tag list, and not padded to length.',
      '- Never state BPM, key or time signature in the caption text — they have dedicated fields.',
    '- Each sentence covers one topic, in this order:',
    ...CAPTION_DIMENSIONS.map((s: string) => `    - ${s}`),
    '- Never name the artist, the band or the song title.',
    '- Never state the BPM number, the key name or the time signature in the prose — they are separate fields.',
    "- Avoid review or marketing language ('captivating', 'emotionally resonant', 'a journey'). Use concrete audio detail.",
    '',
    'DERIVE SENTENCES 7, 8 AND 9 FROM THE SONG\'S OWN SECTION TAGS — not from a template:',
    '- The lyrics below carry tags such as [Intro], [Verse], [Pre-Chorus], [Chorus], [Bridge - Heavy Breakdown], [Bridge - Sparse and Quiet], [Outro - Whispered]. They tell you what actually happens and when.',
    '- If a song has no breakdown, do NOT invent one. A ballad or a quiet track must describe the lift into its chorus and its sparse bridge, not "the drop".',
    '- Match the stated tempo and key quality: a slow minor song and a fast major one must not get the same energy language.',
    '- Sentence 9 must describe how THIS song ends, which its final tag tells you.',
    '',
    `SONGS (${songs.length}):`,
  );

  for (const s of songs) {
    lines.push(
      '',
      `--- id ${s.id} ---`,
      `Title: ${s.title}`,
      ...(s.subject ? [`Subject: ${s.subject}`] : []),
      `Tempo: ${s.bpm ?? 'unknown'} BPM | Key: ${s.key ?? 'unknown'} | Duration: ${s.duration ?? 'unknown'}s`,
      'Lyrics:',
      s.lyrics,
    );
  }

  lines.push('', `Return the JSON object now, with exactly ${songs.length} entries.`);
  return lines.join('\n');
}

// ── MiniMax-Music3 Structured Caption ───────────────────────────────────────
//
// MM3 is the second generation backend and it does NOT want an ACE-Step
// caption. It was trained on a three-heading "Structured Caption" whose thirteen
// labelled sub-fields are 100% consistent across all 1,000 reference captions
// MiniMax ship (.claude/skills/mm3-captioning/upstream/templates/ — every one of
// the 1,000 carries all thirteen labels, verbatim). Adherence to that format is
// the dominant lever on MM3 output, not a stylistic preference:
//
//   Controlled A/B, 2026-08-14 — one track, identical lyrics, 5 seeds per arm,
//   f16/f16, no adapter. Arm A was a genuinely rich 219-word ACE-style caption
//   already covering groove, per-instrument detail, timbre, mix and arrangement.
//   Arm B was the SAME content restructured into this format. Verdict by ear:
//   B better "by a massive margin" — most A takes were not even the right
//   genre (1 of 5 was), B was on-genre throughout.
//
// So rich descriptive prose is NOT a substitute for the format, and a caption
// written for ACE cannot be handed to MM3. Hence a second, separate field.
//
// Two rules below are load-bearing, and each one cost a wrong result before it
// was understood:
//
//   * LEAD THE ARRANGEMENT WITH WHATEVER OPENS THE TRACK. A piano intro named
//     two thirds of the way down a 503-word caption vanished from the audio;
//     the same instrument named FIRST under Primary survived into it.
//   * THE GENRE WORD MUST BE SPECIFIC. Collapsing "Pop-Punk" to the umbrella
//     "Rock" is precisely the failure that produced flat plain-rock takes.
//
// This caption gets its OWN LLM call, made AFTER the lyrics exist, because its
// Arrangement is a section-by-section timeline of THIS song and the lyric's
// section tags are the only evidence of what that timeline is. The metadata
// planner runs before a single lyric line has been written and cannot do it.
//
// Field lengths below are MEASURED over those 1,000 reference captions (median
// words; whole caption runs 436-714 words, median 574) rather than guessed, so
// a model that hits them produces something the same size and density as what
// MM3 was actually trained on.

/** The thirteen labelled sub-fields of an MM3 Structured Caption, in order,
 *  with the measured median word count of each across MiniMax's 1,000
 *  reference captions. `heading` marks the three top-level lines. */
export const MM3_CAPTION_FIELDS: ReadonlyArray<
  { heading: string } | { label: string; words: number; what: string; opener?: string }
> = [
  { heading: 'Global Metadata' },
  { label: 'Basic Attributes', words: 15, what: 'the fixed facts, in the exact shape shown below' },
  { label: 'Global Emotional Progression', words: 62, what: 'how the emotional intensity moves from the opening through to the ending', opener: 'The piece opens... / The track opens...' },
  { label: 'Application Scenarios & Imagery', words: 27, what: 'where this music would be heard, or the scene it paints', opener: 'Ideal for...' },
  { label: 'Sonics & Production Profile', words: 59, what: 'soundstage width, frequency balance, dynamic aesthetic, production era and character', opener: 'The production features... / The soundstage is... / The mix features...' },
  { heading: 'Vocal Details' },
  { label: 'Vocal Gender & Timbre', words: 28, what: 'the texture, weight and register of the voice', opener: 'Singer A (Male). / Singer A (Female).' },
  { label: 'Vocal Style', words: 49, what: 'phrasing, delivery, dynamics, and how the performance changes across the song', opener: 'The performance begins... / The delivery is...' },
  { label: 'Harmony/Backing Vocals', words: 40, what: 'what harmony or backing exists and where it enters - state plainly when there is none', opener: 'Layered backing vocals... / No distinct backing vocals...' },
  { label: 'Vocal FX', words: 47, what: 'processing on the voice: reverb, delay, doubling, saturation. Restraint is the norm', opener: 'The lead vocal... / The vocal track...' },
  { heading: 'Arrangement' },
  { label: 'Instrument Lifecycle Description (Primary/Secondary Layering)', words: 0, what: 'a bare label on its own line - Primary and Secondary follow it' },
  { label: 'Primary', words: 35, what: 'the instruments carrying the harmonic and melodic weight, and when they are present', opener: 'the instrument itself, as a noun phrase - NAME WHATEVER OPENS THE TRACK' },
  { label: 'Secondary', words: 45, what: 'the supporting layers, and which sections introduce or drop them', opener: 'the instrument itself, as a noun phrase' },
  { label: 'Groove & Foundation Progression', words: 65, what: 'drums, bass and rhythmic feel, and how they develop section by section', opener: 'The rhythm section... / The track begins...' },
  { label: 'Embellishments, Textures & Spatial FX', words: 53, what: 'fills, risers, pads, transitions, ambience - or state plainly that there are none' },
];

/** The literal skeleton, rendered from MM3_CAPTION_FIELDS so the prompt and the
 *  validator can never drift apart. */
function mm3Skeleton(): string {
  return MM3_CAPTION_FIELDS.map((f) => {
    if ('heading' in f) return f.heading;
    if (f.label === 'Basic Attributes') {
      return 'Basic Attributes: bpm is <N>. key is <note>, and scale is <major|minor>. <Specific Genre>.';
    }
    if (!f.words) return `${f.label}:`;
    return `${f.label}: <${f.what}> (~${f.words} words)`;
  }).join('\n');
}

/** How the 1,000 reference captions actually OPEN each field.
 *
 *  Measured, not stylistic preference. The reference corpus is far more
 *  formulaic than it first looks: 69% of Global Emotional Progression fields
 *  begin "The piece opens", 64% of Groove fields begin "The rhythm section",
 *  82% of Vocal FX fields begin with some form of "The lead vocal / The vocal
 *  track / The vocals are". Captions written to the field DESCRIPTIONS alone
 *  matched those openers 0-4 times out of 8, i.e. they were in-format but out
 *  of dialect, and that is the layer nothing in the earlier prompt addressed. */
function mm3Phrasing(): string {
  return MM3_CAPTION_FIELDS
    .filter((f): f is { label: string; words: number; what: string; opener?: string } =>
      !('heading' in f) && !!f.opener)
    .map(f => `  ${f.label}: ${f.opener}`)
    .join('\n');
}

export const MM3_CAPTION_SYSTEM_PROMPT = `You write Structured Captions for MiniMax-Music3, an AI music generator. This caption is the ONLY description the model receives: it alone decides the genre, the voice and the arrangement of the track that gets rendered. A caption that drifts off format produces off-genre music — that is measured, not theoretical.

Return ONE caption in EXACTLY this shape. All three heading lines and all thirteen labels are mandatory, in this order, spelled character-for-character as shown:

${mm3Skeleton()}

HOUSE PHRASING (measured over MiniMax's own 1,000 reference captions — these are
the openings the model was trained on, and matching them matters as much as the
labels do). Open each field the way the reference does:

${mm3Phrasing()}

FORMAT RULES (mechanical — a violation breaks the model's expectations):
- PLAIN TEXT ONLY. No markdown, no '#' headings, no '**bold**', no bullets or dashes at the start of lines, no code fences, no numbering.
- One field per line. No blank lines anywhere, including between the three sections.
- The heading lines "Global Metadata", "Vocal Details" and "Arrangement" stand alone with no colon and no text after them.
- "Instrument Lifecycle Description (Primary/Secondary Layering):" is a bare label; its content is the "Primary:" and "Secondary:" lines beneath it.
- Output the caption and NOTHING else — no preamble, no explanation, no closing remark.

CONTENT RULES (musical — each one is a measured failure mode):
- THE BASIC ATTRIBUTES LINE ENDS ON THE GENRE. No time signature, no trailing clause, nothing after it — all 1,000 reference captions close that line on the genre itself, and it is the only line that states the genre at all.
- THE GENRE MUST BE SPECIFIC. Write "Pop-Punk", "Post-Hardcore", "Contemporary R&B", "Melodic Dubstep" — never the umbrella term "Rock", "Pop" or "Electronic" when a narrower one is true. Collapsing a genre to its umbrella is the single most reliable way to get a generic, wrong-sounding track. Two or three slash-joined genres are normal: "Synth-Pop / Mandopop".
- NAME WHATEVER OPENS THE TRACK FIRST under "Primary:", as a noun phrase, exactly the way the reference does it: "A grand piano serves as the harmonic core...", "Heavily distorted electric guitars carry...". An instrument mentioned late in a long caption does not survive into the audio; the one named first does. Do NOT invert it into "X opens the track" — that construction does not appear in the reference corpus at all.
- Describe the song SECTION BY SECTION. The lyrics you are given carry section tags — [Intro], [Verse], [Chorus], [Bridge - Heavy Breakdown], [Outro] and so on. They are directives: they tell you what actually happens and when. Follow the real tags; do not invent a drop for a ballad or a quiet bridge for a thrash number.
- NEVER quote, paraphrase or summarise the lyrics, and never state the song title, the artist or the band name. Use the lyrics only as evidence of the arrangement and the emotional arc.
- Do NOT fabricate precision. If something is not supported by what you were given, describe it in broader terms instead of inventing an exact technique.
- An instrumental track stays instrumental. Never add vocals that were not asked for.
- Write concrete audio detail, not review copy. "The snare moves from rimclick to a full backbeat at the first chorus" is useful; "an emotionally resonant journey" is not.
- English, roughly 450-650 words in total. The per-field figures above are medians from those same reference captions and sum to about 525 words of prose; with the heading and label text that lands near 570, comfortably inside the range. They agree — treat the per-field numbers as the shape and the range as the bound, and do not pad to reach 650.`;

/** Everything the MM3 caption call knows about the song being described. */
export interface Mm3CaptionContext {
  /** The ACE-Step caption planned for the same song — the best available
   *  evidence of the intended sound, and the only place a vocal gender is
   *  usually stated. Used as EVIDENCE, never restructured mechanically:
   *  scripted restructuring of ACE captions was ear-tested five ways and
   *  never reached the target genre (see docs in the mm3-captioning skill). */
  aceCaption?: string;
  subject?: string;
  bpm?: number;
  /** "E minor" / "Bb major" — as stored on the generation. */
  key?: string;
  /** "4/4", or a bare numerator ("4") as the training sidecars store it. */
  signature?: string;
  lyrics: string;
  /** No lyrics were written — the track is instrumental. */
  instrumental?: boolean;
}

/** Bracketed section tags in the order they appear, duplicates kept — the
 *  Arrangement timeline is exactly this sequence. */
export function extractSectionTags(lyrics: string): string[] {
  const out: string[] = [];
  for (const line of (lyrics || '').split('\n')) {
    const m = line.match(/^[ \t]*\[([^\]]{1,60})\][ \t]*$/);
    if (m) out.push(`[${m[1].trim()}]`);
  }
  return out;
}

/** "4" -> "4/4"; "4/4" -> "4/4"; anything unparseable -> "4/4". */
export function normalizeTimeSignature(sig?: string | null): string {
  const s = String(sig ?? '').trim();
  if (/^\d{1,2}\s*\/\s*\d{1,2}$/.test(s)) return s.replace(/\s+/g, '');
  if (/^\d{1,2}$/.test(s)) return `${s}/4`;
  return '4/4';
}

/** "C# minor" -> ["C#", "minor"]; unparseable -> [null, null].
 *  Deliberately NOT case-insensitive on the accidental: with /i, `[#b]` also
 *  matches the note letter B, so "Bb minor" parses as note "B" + accidental "b"
 *  only while the flag is off. Mode is matched case-insensitively by hand. */
function parseKeyScale(key?: string | null): [string | null, string | null] {
  const m = normalizeKeyScale(key).match(/^([A-G][#b♯♭]?) (major|minor)$/);
  return m ? [m[1], m[2]] : [null, null];
}

export function buildMm3CaptionPrompt(profile: PromptProfile, ctx: Mm3CaptionContext): string {
  const enrich: AlbumEnrichment | null = profile.audio_enrichment ?? null;
  const sig = normalizeTimeSignature(ctx.signature || enrich?.signatures?.[0]);
  const [note, scale] = parseKeyScale(ctx.key);
  const tags = extractSectionTags(ctx.lyrics);

  const lines: string[] = [];

  lines.push(
    'FIXED FACTS — reproduce these EXACTLY in the "Basic Attributes:" line. Do not round them, do not substitute your own, and do not repeat them anywhere else in the caption:',
    `  bpm: ${ctx.bpm && ctx.bpm > 0 ? Math.round(ctx.bpm) : '(unknown — omit the bpm sentence)'}`,
    `  key / scale: ${note && scale ? `${note} / ${scale}` : '(unknown — omit the key sentence)'}`,
    '',
    'The Basic Attributes line ENDS on the genre. Do not state a time signature there or anywhere else in the caption — the reference captions never do.',
    '',
  );

  if (enrich?.genres?.length) {
    lines.push(
      `GENRE — detected on this album's actual recordings: ${enrich.genres.slice(0, 6).join(', ')}.`,
      'Name the most SPECIFIC of these that fits this song. If one of them is an umbrella term and a narrower one is also true, use the narrower one.',
      '',
    );
  }

  if (ctx.aceCaption) {
    lines.push(
      "EVIDENCE — how this song is meant to sound. This describes the same track for a different music model, so it is source material, NOT a template: do not restructure it sentence by sentence. Read what it says about instruments, voice, production and energy, then write the Structured Caption from scratch.",
      `  "${ctx.aceCaption}"`,
      '',
    );
  } else if (enrich?.captionExamples?.length) {
    lines.push(
      "EVIDENCE — captions describing this album's actual recordings. Source material for the sound of this record, NOT a template to restructure:",
      ...enrich.captionExamples.slice(0, 2).map((c, i) => `  ${i + 1}. "${c}"`),
      '',
    );
  }

  if (profile.tone_and_mood) lines.push(`Tone & mood of this artist: ${profile.tone_and_mood}`, '');
  if (ctx.subject) lines.push(`What this song is about (context for the emotional arc only — never state it in the caption): ${ctx.subject}`, '');

  if (ctx.instrumental) {
    lines.push(
      'THIS TRACK IS INSTRUMENTAL. There is no singing. Under "Vocal Gender & Timbre:" state that the piece is instrumental and name the instrument carrying the lead melodic line; keep the other three Vocal Details fields consistent with that. Never add a vocalist.',
      '',
    );
  }

  if (tags.length) {
    lines.push(
      `ARRANGEMENT TIMELINE — this song's own sections, in order (${tags.length}):`,
      `  ${tags.join(' → ')}`,
      `The track OPENS on ${tags[0]} and ENDS on ${tags[tags.length - 1]}. Name whatever plays under ${tags[0]} first under "Primary:", and make the Arrangement follow this exact sequence.`,
      '',
    );
  }

  lines.push(
    'LYRICS — evidence of the arrangement and the emotional arc ONLY. Never quote, paraphrase or summarise a single line of them in the caption:',
    ctx.lyrics,
    '',
    'Write the Structured Caption now. Output the caption only.',
  );

  return lines.join('\n');
}

/**
 * Deterministic cleanup of a model-written MM3 caption.
 *
 * Two jobs. First, strip the markdown a chat model reflexively adds — the
 * upstream skill states its own three sections as `###` headings, so models
 * that have seen it copy that, and the 1,000 real templates use plain labels.
 *
 * Second, REBUILD the "Basic Attributes:" line from the facts we hold exactly.
 * This is the same correction `engine/tools/mm3-caption-hybrid.py` applies to
 * MOSS's audio captions, for the same reason: models are unreliable on bpm and
 * key even when told them (MOSS read 102 BPM / C# minor off a track Essentia
 * measures at 90 / E major), and here the bpm and key are not observations at
 * all — they are the values the song will actually be rendered at. The GENRE
 * clause the model wrote is preserved, because genre is the one part of that
 * line the model is the better judge of.
 */
export function normalizeMm3Caption(
  raw: string,
  facts: { bpm?: number; key?: string; signature?: string; fallbackGenre?: string } = {},
): string {
  let text = String(raw ?? '');

  // Fenced block: keep only what is inside the first fence.
  const fence = text.match(/```(?:[a-z]*)\n([\s\S]*?)```/i);
  if (fence) text = fence[1];

  const cleaned = text
    .split('\n')
    .map((l) =>
      l
        .replace(/^\s{0,8}#{1,6}\s*/, '')      // markdown headings
        .replace(/^\s{0,8}[-*+]\s+/, '')       // bullet markers
        .replace(/\*\*/g, '')                  // bold
        .replace(/^\s+/, '')                   // leading indent
        .replace(/\s+$/, ''),
    )
    .filter((l) => l.length > 0);               // templates carry no blank lines

  const [note, scale] = parseKeyScale(facts.key);
  const bpm = facts.bpm && facts.bpm > 0 ? Math.round(facts.bpm) : 0;

  const idx = cleaned.findIndex((l) => /^Basic Attributes\s*:/i.test(l));
  const genre = idx >= 0
    ? (extractGenreClause(cleaned[idx].replace(/^Basic Attributes\s*:/i, '')) || facts.fallbackGenre || '')
    : (facts.fallbackGenre || '');

  // Only rebuild when we actually hold a fact worth asserting; otherwise the
  // model's own line, whatever it says, is better than a stub.
  if (bpm || (note && scale) || genre) {
    // The line ENDS on the genre. Every one of MiniMax's 1,000 reference
    // captions does; none of them states a time signature here. An earlier
    // version appended ", in 4/4." — lifted from mm3-caption-hybrid.py without
    // checking that convention against the corpus, which it does not match —
    // so the genre was never the terminal token of the only line that names it.
    const parts: string[] = [];
    if (bpm) parts.push(`bpm is ${bpm}.`);
    if (note && scale) parts.push(`key is ${note}, and scale is ${scale}.`);
    if (genre) parts.push(`${genre}.`);
    const line = `Basic Attributes: ${parts.join(' ')}`;
    if (idx >= 0) cleaned[idx] = line;
    else {
      // Model omitted the line entirely — insert it under Global Metadata
      // rather than dropping the facts on the floor.
      const g = cleaned.findIndex((l) => /^Global Metadata\s*$/i.test(l));
      cleaned.splice(g + 1, 0, line);
    }
  }

  return cleaned.join('\n').trim();
}

/** Strip the bpm and key sentences out of a Basic Attributes body, leaving the
 *  genre clause the model chose. */
function extractGenreClause(body: string): string {
  return body
    .replace(/\bbpm\s+is\s+[^.]*\.?/i, '')
    .replace(/\bkey\s+is\s+[^.]*\.?/i, '')
    .replace(/\bscale\s+is\s+(major|minor)\s*\.?/i, '')
    .replace(/,?\s*\bin\s+\d{1,2}\s*\/\s*\d{1,2}\b\s*\.?/i, '')
    .replace(/\s+/g, ' ')
    .replace(/^[\s.,]+|[\s.,]+$/g, '')
    .trim();
}

/**
 * Format problems in a written MM3 caption. Empty array = structurally sound.
 * Advisory only: it logs and drives ONE retry, and never rejects a caption —
 * a partly-malformed Structured Caption still beats handing MM3 an ACE one.
 *
 * Checked against MiniMax's own 1,000 reference captions: 998 pass. The two
 * that do not (big-band-jazz-swing_0003, musical-theatre-cinematic-folk_0001)
 * run "* Primary:" inline on the Instrument Lifecycle line instead of starting
 * its own. That is a genuine 0.2% corpus variation, not a bug here — we ask the
 * model for the dominant form, so requiring it is correct.
 */
export function validateMm3Caption(caption: string): string[] {
  const issues: string[] = [];
  const text = String(caption ?? '');
  for (const f of MM3_CAPTION_FIELDS) {
    if ('heading' in f) {
      if (!new RegExp(`^${f.heading}\\s*$`, 'm').test(text)) issues.push(`missing heading "${f.heading}"`);
    } else {
      const label = f.label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      if (!new RegExp(`^${label}\\s*:`, 'im').test(text)) issues.push(`missing field "${f.label}:"`);
    }
  }
  const words = text.split(/\s+/).filter(Boolean).length;
  if (words < 300) issues.push(`too short (${words} words; reference captions run 436-714)`);
  if (/^#{1,6}\s/m.test(text) || text.includes('**')) issues.push('contains markdown');
  return issues;
}

// ── Lyric density rewrite ───────────────────────────────────────────────────
//
// Fixes songs whose lyrics are the wrong WORD DENSITY for their artist: written
// before per-artist pacing existed, a sparse lyric on a fast artist (needs
// EXPANSION) or a wordy lyric on a slow artist (needs CONDENSING) cannot be
// fixed by retiming — the derived duration lands outside anything the artist
// actually records. The song's identity (title, subject, hooks, section
// sequence) is kept; only the quantity of lyric per section changes. The
// section SEQUENCE must survive because each song's caption describes that
// arc (sentences 7-9) and was written against it.

export const LYRIC_DENSITY_REWRITE_SYSTEM_PROMPT = `You are a professional songwriter performing a surgical rewrite. Each song below already has its identity — title, subject, hook, section structure — but the WRONG AMOUNT of lyric for how this artist actually sings. Your job is to expand or condense each lyric to a target word count while keeping the song recognisably the same song, in the artist's voice.

Rules, all mandatory and machine-validated:
- Hit each song's TARGET WORD COUNT within ±10%. Count words in lyric lines only (section tags don't count).
- KEEP the song's title concept, subject, narrative and main hook lines. Keep the SAME SECTION SEQUENCE in the same order — you may change how many lines each section holds, but not reorder, add or remove sections (exception: when EXPANDING you may add [Pre-Chorus] before an existing [Chorus], or repeat the final [Chorus], if the target is otherwise unreachable).
- EXPANDING means adding CONTENT — concrete images, narrative detail, call-and-response, ad-libs, hook repetitions — never filler words stretched thin. Study the reference lyric for how densely this artist packs a line.
- CONDENSING means cutting to the strongest material — keep the hook and the sharpest images, drop whole lines rather than thinning every line.
- Match the reference lyric's line lengths and phrasing density. It is a real song by this artist.
- Every lyric line must end with punctuation (period, comma, exclamation, question mark, dash, or ellipsis).
- No "Title:" line, no commentary. Lyrics start at the first section tag.
- Avoid AI-cliché words: neon, ethereal, embers, silhouette, static, void, shimmering, tapestry, gasoline, halogen.

Return ONLY a JSON object mapping each song id to its complete rewritten lyrics string (with \\n newlines):
{"123": "[Intro]\\n\\n[Verse 1]\\n...", ...}
No markdown fences, no commentary.`;

export interface DensityRewriteSong {
  id: number;
  title: string;
  subject?: string | null;
  bpm?: number | null;
  key?: string | null;
  targetDuration: number;
  targetWords: number;
  currentWords: number;
  lyrics: string;
}

export interface DensityRewriteRef {
  title: string;
  duration: number;
  words: number;
  lyrics: string;
}

/** Per-set batch prompt: artist pacing context + one real reference lyric sent
 *  once, then every song needing a rewrite in that set. */
export function buildDensityRewritePrompt(
  artist: string, rate: number, albumMedianSec: number,
  ref: DensityRewriteRef | null, songs: DensityRewriteSong[],
): string {
  const lines: string[] = [
    `Artist: ${artist}`,
    `Measured vocal pacing: ${rate.toFixed(2)} words per second of song time (from their real recordings). A typical song by this artist runs ~${albumMedianSec}s.`,
    '',
    ...SECTION_LABEL_RULE.split('\n'),
  ];
  if (ref) {
    lines.push('', `=== REFERENCE — a real lyric by this artist ("${ref.title}", ${ref.duration}s, ${ref.words} words). Match its density and voice, do NOT copy its content: ===`, '', ref.lyrics);
  }
  lines.push('', `=== SONGS TO REWRITE (${songs.length}) ===`);
  for (const s of songs) {
    const dir = s.targetWords > s.currentWords ? 'EXPAND' : 'CONDENSE';
    lines.push(
      '',
      `--- id ${s.id} ---`,
      `Title: ${s.title}`,
      ...(s.subject ? [`Subject: ${s.subject}`] : []),
      `Tempo: ${s.bpm ?? 'unknown'} BPM | Key: ${s.key ?? 'unknown'} | Target duration: ${s.targetDuration}s`,
      `${dir}: currently ${s.currentWords} words -> TARGET ${s.targetWords} words (±10%).`,
      'Current lyrics:',
      s.lyrics,
    );
  }
  lines.push('', `Return the JSON object now, with exactly ${songs.length} entries.`);
  return lines.join('\n');
}

/** The shared "=== AUDIO ANALYSIS ===" prompt block (facts only, no guidance). */
export function formatAlbumEnrichment(e: AlbumEnrichment): string[] {
  const lines = ["=== AUDIO ANALYSIS (measured from this album's source recordings) ==="];
  if (e.bpmMax > 0) {
    lines.push(e.bpmMin === e.bpmMax
      ? `BPM: ${e.bpmMax} on every analysed track`
      : `BPM range: ${e.bpmMin}–${e.bpmMax}`);
  }
  if (e.keys.length) lines.push(`Keys used: ${e.keys.join(', ')}`);
  if (e.genres.length) lines.push(`Detected genre: ${e.genres.join(', ')}`);
  if (e.signatures.length) lines.push(`Time signatures: ${e.signatures.join(', ')}`);
  if (e.wordsPerSec > 0) {
    lines.push(`Vocal pacing: ~${e.wordsPerSec.toFixed(2)} words/second of song time (median of ${e.pacedSongs} tracks) — plan duration and lyric quantity around this rate.`);
  }
  lines.push(`(from ${e.enrichedSongs} of ${e.totalSongs} tracks)`);
  return lines;
}

// ── System Prompts ──────────────────────────────────────────────────────────

export const GENERATION_SYSTEM_PROMPT = `You are a talented, creative songwriter who specialises in emulating specific artistic styles with uncanny accuracy.

You will be given a detailed stylistic profile of an artist's lyrics, including:
- Statistical analysis (rhyme patterns, meter, vocabulary metrics, line length distributions)
- Repetition and hook analysis (how the artist uses repeated lines)
- Deep stylistic analysis (themes, tone, narrative techniques, imagery)
- Representative lyric excerpts showing the artist's actual voice
- The artist's structural vocabulary (section structures observed in their real songs)

Your task is to write a completely new, original song that could convincingly pass as an unreleased track by this artist.

The rules below come in two tiers. TIER 1 rules are mechanical requirements of the music-generation pipeline that parses your lyrics — breaking them produces broken audio, so they are absolute. TIER 2 rules are style targets: calibrate them to THIS artist and use your own creative judgement. When you deviate from a target, do it deliberately and in the artist's service — never by accident.

=== TIER 1: PIPELINE REQUIREMENTS (HARD — the music model breaks if violated) ===

- NO TITLE: Write ONLY the lyrics — no "Title:" line, no heading. Start directly with the first section header.
${SECTION_LABEL_RULE}
- PUNCTUATION: Every lyric line MUST end with punctuation (period, comma, exclamation mark, question mark, dash, or ellipsis). The vocal model uses it for phrasing.
- CHORUS: Every song needs at least one [Chorus]. If a section repeats throughout the song, it is a chorus — label it [Chorus], not [Bridge]. A bridge is a one-time contrasting section, typically appearing once before the final chorus.
- LINE BUDGET: If the user prompt gives a duration budget with a maximum total line count, treat it as a hard ceiling — the music model skips lines beyond it and the song comes out truncated.
- INSTRUMENTAL INTRO: Let the music establish itself before the vocals enter. Open the song with an empty [Intro - Instrumental] header — no lyric lines under it — on MOST songs; roughly one in five should instead slam straight into [Verse 1], as a deliberate choice for songs that want it. The descriptor is not optional: a bare [Intro] with nothing under it invites an arbitrarily long instrumental of the model's own choosing, while [Intro - Instrumental] renders at a sane, learned size. If this song opens on SUNG material, tag it [Intro] and write those words instead. NEVER write count-ins like "One, two, three, four!".

=== TIER 2: STYLE TARGETS (calibrate to THIS artist — deviate only with intent) ===

- STRUCTURE: The user prompt lists the section structures observed in this artist's real songs, plus a suggested structure for this one. Treat that vocabulary as your palette, not a cage: you may add or drop a Pre-Chorus, use two verses instead of three, or move the Bridge — as long as the song still reads like this artist's structural habits and stays close to the suggested section count (the duration budget is planned around it).
- SECTION LENGTHS: Keep verse and chorus line counts EVEN — 4, 6, or 8 lines — so musical phrases resolve cleanly; odd-length sections (5 or 7 lines) clash with the music model's phrasing. Within that, match the artist's typical section lengths from the profile. Bridges may be shorter (2-6 lines).
- METER: vary line lengths according to the syllable distribution shown. Some lines short, some long — NOT uniform.
- RHYME STYLE: use the same mix of perfect, slant, and assonance rhymes.
- PERSPECTIVE: use the same pronoun patterns (first/second/third person balance).
- VOCABULARY LEVEL: same contraction frequency, same register, same slang level.
- SIGNATURE DEVICES: capture the artist's verbal tics, recurring imagery, distinctive phrasing.
- EMOTIONAL ARC: how the song builds, shifts, or resolves emotionally.

LYRIC QUALITY RULES:
- *** NO COPYING — ABSOLUTE RULE ***
  NEVER reuse ANY phrase, line, or distinctive word combination from the source artist's lyrics.
  The excerpts are STYLE REFERENCE ONLY — absorb the cadence and feel, then write 100% original words.
  If a phrase reminds you of something from the excerpts, DO NOT USE IT. Write something new.
  Reusing the artist's actual phrases is plagiarism and ruins the generation.

REPETITION / HOOK RULES (CRITICAL):
- Every chorus MUST have a clear HOOK — one memorable line or phrase that repeats at least twice within the chorus.
- The hook should be the emotional anchor of the chorus. Build the other chorus lines around it.
- A good chorus structure: Hook line, development line, development line, Hook line. Or: Hook line, Hook line, development, resolution.
- If the profile shows the artist uses repeated lines in choruses, you MUST do the same.
- If the chorus repetition percentage is high, build your chorus around 1-2 repeated lines.
- Parenthetical echo lines (e.g. "(you know it's true)") count as separate lines — use them if the artist's style calls for it.
- It's OK to repeat key phrases across verses and choruses for thematic cohesion.

HOOK SPECIFICITY RULES (CRITICAL — READ CAREFULLY):
- The chorus hook MUST be SPECIFIC to this song's subject matter. It should contain a concrete noun, image, or scenario from the verses — NOT a generic emotional statement.
- BANNED HOOK FORMULAS — the following structural patterns are FORBIDDEN in chorus hooks because they produce identical-sounding songs across all genres:
  • "[Verb] it [all/down/away/out]" (e.g. "Burn it all down", "Wash it all away", "Tear it all down", "Watch it fade away")
  • "Watch [me/it/them] [verb]" (e.g. "Watch it burn", "Watch me break", "Watch it fade")
  • "Don't let them [verb]" (e.g. "Don't let them see", "Don't let them take")
  • "Nothing left to [verb]" / "Nowhere left to [verb]"
  • "Let it [burn/fade/go/fall/break/die]"
  • Any hook that could apply to ANY song by ANY artist. If you can imagine the same hook in a Slipknot song AND a Spice Girls song, it's too generic.
- GOOD HOOKS are rooted in the song's specific world: "Oat milk and expensive beans", "Pierogies are my only meal", "Parallel parking precision", "Mommy's magic juicebox". These work because they could ONLY belong to THAT specific song.
- The hook doesn't have to be quirky — it just has to be SPECIFIC. "California castaway" is simple but specific. "Watch it burn" is not.

Do NOT include any commentary or explanations — just the lyrics.

The representative excerpts are there to show you the FEEL, not to be copied. Absorb the cadence, word choices, and line-to-line flow, then create something new in that exact voice.

ANTI-SLOP RULES (CRITICAL — ZERO TOLERANCE):
- You MUST avoid ALL clichéd, generic, AI-sounding language.
- BANNED WORDS (using any of these = failed generation): ${Array.from(BLACKLISTED_WORDS).sort().join(', ')}
- BANNED PHRASES (using any of these = failed generation): ${Array.from(BLACKLISTED_PHRASES).sort().join('; ')}
- Use the artist's ACTUAL vocabulary and phrasing style, not generic poetic language.
- If a word or phrase sounds like it came from an AI writing assistant, do NOT use it.
- Specifically NEVER use: neon, fluorescent, streetlights, embers, silhouette, static, void, ethereal, shimmering.
- OVERUSED VOCABULARY — MINIMIZE (using any of these more than ONCE in a song = sloppy writing):
  ${Array.from(OVERUSED_WORDS).sort().join(', ')}
  These words are not banned, but the model tends to lean on them as a crutch across every genre. A Britney Spears song should NOT share vocabulary DNA with a Metallica song. Use the artist's ACTUAL vocabulary, not these generic defaults. If you catch yourself writing "heavy" or "cold" or "broken" or "nothing left" — STOP and find a word that fits THIS artist's voice.
- The "a-" prefix (e.g. "a-walkin'", "a-staring") is ONLY valid before verbs/gerunds (-ing words). NEVER put "a-" before adjectives, nouns, articles, or adverbs (e.g. "a-rusty", "a-this", "a-highly" are WRONG). Use it SPARINGLY — at most 1-2 times per song.
`;

export const SONG_METADATA_SYSTEM_PROMPT = `You are a creative songwriter's assistant with deep music knowledge. Your job is to plan the metadata for a new song.

You will be given:
- The artist's stylistic profile (themes, tone, typical subjects, observed song structures)
- Subjects, BPMs, and keys that have already been used in previous generations (to ensure variety)

Return ONLY a JSON object with exactly this format:
{
  "subject": "one sentence describing what this new song should be about",
  "bpm": 120,
  "key": "C minor",
  "caption": "genre, instruments, emotion, atmosphere, timbre, vocal characteristics, production style",
  "duration": 217,
  "structure": "I-V-C-V-C-B-C-O"
}

Rules for each field:

SUBJECT:
- Must fit the artist's typical range of topics
- Be SPECIFIC and CONCRETE — not vague themes like "love" or "life"
- Do NOT repeat any subject that has already been used
- Think of a fresh angle or scenario the artist might explore

BPM:
- Choose a realistic tempo (30-300) that fits the artist's typical style and genre
- Just pick a BPM that feels right for the song — don't overthink it or try to avoid previous values
- Genre norms for reference: ballads ~60-80, pop ~100-130, rock ~110-140, punk ~150-180, EDM ~120-150, hip-hop ~80-100, folk ~90-120

KEY:
- Pick a musical key that fits the artist and genre (e.g. "C minor", "A minor", "F# minor", "Bb major")
- Key notation is note name + LOWERCASE mode: "C minor", "F# major". ACE-Step's metadata FSM only accepts lowercase major/minor, so "C Major" is out of vocabulary.
- Vary the key across generations — try not to repeat recently used keys
- Consider the artist's typical tonal palette

CAPTION:
- This is a description of the track's MUSICAL characteristics for an AI music generator
- Write it as a comma-separated list of descriptive tags/phrases
- Cover these dimensions: genre/style, instruments, emotion/atmosphere, timbre/texture, vocal characteristics (gender, style), production style, era/reference
- Be specific: "breathy female vocal" not just "female vocal"; "distorted electric guitar" not just "guitar"
- Match the artist's known sound and production aesthetic
- Keep it to 1-3 sentences of comma-separated descriptors
- Example: "indie rock, driving electric guitars, male vocal, raw and energetic, garage production, anthemic chorus, 2010s alternative"
- OVERRIDE: if the user prompt supplies "Captions describing this album's actual recordings", ignore the two rules above (comma-separated list, 1-3 sentences) and follow the caption format given there instead — matching the training captions matters more than brevity

DURATION:
- Estimate the total track duration in seconds (any integer value is fine — do NOT round to multiples of 5)
- Consider: the BPM, the number of lyric sections in your chosen structure, and typical intro/outro/instrumental break lengths
- At the chosen BPM, estimate how long each section takes (a bar of 4/4 = 240/BPM seconds)
- Include typical intro (4-8 bars), instrumental breaks between sections, and an outro
- Genre norms: punk/pop-punk ~150-180s, pop ~200-240s, ballads ~240-300s, rock ~210-270s, hip-hop ~180-240s
- A song with 3 verses, 3 choruses, and a bridge at 120 BPM is typically around 210-240 seconds

STRUCTURE:
- Plan this song's section structure as dash-joined codes: ${BLUEPRINT_CODES_LEGEND}
- If the artist's observed structures are listed, choose one of them or a close variation — this is a creative decision: pick the shape that best serves the subject (a sprawling story wants 3 verses; a punchy single wants 2; not every song needs a bridge)
- Vary the structure across generations — do not default to the same shape every time
- Must contain at least one C (chorus)

Do NOT include any text outside the JSON object.

SUBJECT ANTI-SLOP RULES:
- The subject description MUST NOT contain any of these AI-cliché words: ${Array.from(BLACKLISTED_WORDS).slice(0, 30).sort().join(', ')}.
- Do NOT use these overused subject framings: "The sensation of", "The feeling of", "A person watching", "The terrifying realization that". Start with a specific, cinematic scenario instead.
- AVOID these subject themes unless the artist profile specifically calls for them: identity dissolution, mirror reflections, masks/disguises, industrial decay, suffocation metaphors, surveillance/being watched.
- Be SPECIFIC and SENSORY: "A fight with a taxi driver over a $3 fare at 4am" beats "The suffocating sensation of urban disconnection".
- Think like the ARTIST would think, not like an AI writing assistant.
`;

export const TITLE_DERIVATION_PROMPT = `You are a song-titling expert. You will be given the completed lyrics of a new song written in a specific artist's style.

Your ONLY job: choose the best possible title for this song.

TITLE RULES (MANDATORY):
1. DERIVE FROM THE LYRICS. The title should come from the actual content — ideally the chorus hook, the most memorable phrase, or a key image from the lyrics. Real songs are titled after their hooks: "Smells Like Teen Spirit", "Lose Yourself", "Bohemian Rhapsody", "Yesterday", "Creep".
2. PREFER THE HOOK. If the chorus has a clear repeated phrase or hook line, that IS the title. Don't overthink it.
3. SHORT AND PUNCHY. 1-5 words is ideal. Rarely more than 6. If the hook phrase is long, trim to its strongest fragment.
4. NO AI CLICHÉ TITLES. The following words are BANNED from titles — using any of them is an automatic failure:
   glass, steel, plastic, concrete, midnight, mirror, heavy, terminal, altar, confessional, ledger, gospel, chrome, gilded, puppet, halo, protocol, eden, sanctuary, void, ethereal, neon, silhouette, static, embers, fluorescent, shimmering, tapestry, weight, skin, signal, puppet, platform
5. BE SPECIFIC, NOT VAGUE. "Pizza Hut and Existential Dread" beats "The Empty Feeling". "Don't Let Your Legs Quit" beats "The Journey Continues".
6. MATCH THE ARTIST'S STYLE. A punk band's title should sound punk. A soul singer's title should sound soulful. Don't impose indie-rock titling on a hip-hop track.

Return ONLY the title — no quotes, no "Title:" prefix, no explanation. Just the title text on a single line.
`;

export const REFINEMENT_SYSTEM_PROMPT = `You are a professional songwriting editor. Your job is to take a rough song draft and make it feel finished, singable, emotionally precise, and true to its intended artistic lane.

You will receive:
1. The original generated lyrics
2. A description of the intended artist/genre lane (style profile)

Your task is to REFINE, not replace.
Default to minimal intervention. Preserve as much of the original wording, imagery, and structure as possible.

EDITING PRIORITY ORDER
When rules conflict, use this order:

1. Preserve the song's core meaning, emotional intent, and strongest images.
2. Preserve the original voice, tone, and worldview.
3. Improve singability, cadence, and section function.
4. Improve hook strength and memorability.
5. Improve rhyme, line economy, and structural neatness.
6. Add stylistic flavor only if it feels native and does not weaken the lyric.

CORE EDIT POLICY
- Preserve at least 70-85% of the original lines unless a line is weak, redundant, tonally false, structurally broken, or obviously artificial.
- Prefer local edits over full rewrites.
- Repair vivid lines rather than replacing them with safer generic lines.
- Do not rewrite for the sake of rewriting.

REFINEMENT RULES

1. VERSE SHAPE
   Prefer even line counts (4, 6, or 8 lines) for verses — musical phrases resolve in even numbers of lines.
   Do not force line counts if doing so weakens meaning, cadence, or imagery.

2. CHORUS DESIGN (CRITICAL)
   The chorus must contain:
   - one central hook phrase
   - clear emotional payoff
   - strong rhythmic and vowel shape
   - at least one line that is instantly memorable after one listen
   Repetition should feel deliberate, not mechanical.
   If the chorus lacks a strong hook, strengthen the best existing line rather than inventing a totally new one.

3. SONG STRUCTURE
   The song must have a clear, logical structure with at least one chorus.
   Typical structures include V-C-V-C-B-C and I-V-C-V-C-B-C-O, but do not add sections unless they improve the song.

4. INTRO (CRITICAL — MUSIC MODEL REQUIREMENT)
   If the song does not already start with an [Intro] section, you MUST ADD ONE before the first verse.
   The downstream music model produces cleaner audio with an instrumental opening before vocals begin.
   The intro should typically be just the [Intro] header with no lyrics (instrumental), unless the artistic choice strongly calls for a vocal intro.

5. RHYME
   Match the intended lane's rhyme behavior.
   Do not force perfect rhyme if looser rhyme sounds more natural.

6. CHORUS CONSISTENCY
   Repeated choruses should be identical or near-identical unless a small change creates meaningful escalation.

7. NO FILLER
   Every line must earn its place.
   Cut throat-clearing, explanatory padding, and duplicate ideas.

8. PRESERVE THE STORY
   The refined version must tell the same story or emotional arc as the original.

9. PRESERVE THE VOICE
   Keep the same level of directness, slang, contraction, profanity, melodrama, and emotional temperature.

10. SECTION FUNCTION
    Each section must do a distinct job:
    - Intro: atmosphere, angle, or motif
    - Verse: story, image set, or argument
    - Pre-Chorus: tension or lift
    - Chorus: emotional thesis and hook
    - Bridge: contrast, reversal, confession, escalation, or revelation
    - Outro: final image, hook, or aftertaste
    If a section does not perform a distinct function, compress or locally rewrite it.

11. PROSODY AND SINGABILITY (CRITICAL)
    Prioritize lines that feel natural when spoken or sung.
    Check for:
    - clunky stress patterns
    - awkward filler words
    - too many function words in a row
    - lines that over-explain
    - page-poetry that does not sing well
    A line may stay slightly rough if it sounds better aloud and suits the voice.

12. VARIED OPENINGS
    Avoid starting multiple sections the same way, especially with "You" or "You're."
    Vary openings through imagery, action, setting, thought, time, or sound.
    The song's FIRST lyric line (after [Intro]) is especially important — it sets the tone. Make it vivid and distinctive.
    It is OK for ONE section to start with "You" — just not multiple sections, and ideally not the very first verse.

13. DYNAMIC LINE LENGTHS
    Avoid machine-like uniformity. Smaller generation models produce lines that are all roughly the same length — this is the #1 tell of AI-generated lyrics.
    Mix short, medium, and long lines in a musically natural way.
    Short lines hit harder for emotional punctuation. Longer lines build narrative momentum.
    Variation should feel performative, not random.

14. DO NOT GENERICISE
    Do not replace specific, vivid, unusual, or emotionally sharp lines with broader, flatter, or more cliché alternatives.
    If a line is memorable but imperfect, repair it instead of simplifying it.

15. AUTHENTICITY SIGNALS
    Aim for authenticity through underlying writing behavior, not imitation by catchphrase.
    Reflect the intended lane through:
    - cadence and line density
    - rhyme looseness/tightness
    - image categories
    - emotional stance
    - repetition habits
    - narrative distance
    - level of theatricality, wit, or bluntness
    Do not rely on trademark phrases, signature ad-libs, or recognisable verbal tics unless they are already present in the draft and feel fully natural.

16. NO SPEAKER IDENTIFIERS
    NEVER include speaker identifiers like "DJ:", "Singer:", "Rapper:", "[Rapper Name]:", etc. Ace-Step 1.5 does not understand these and will speak them literally. Strip them out completely.

17. NO AUDIENCE CUES / PERFORMANCE NOTES
    NEVER include audience cues like "(Crowd: WHO!)", "(Applause)", "(Cheering)", "(Laughter)", or performance notes like "(Spoken)". These disrupt the vocal generation. If they exist in the original, REMOVE them.

18. NO NONSENSE OR CIRCULAR PHRASING
    Fix lines that are grammatically broken or logically circular. Examples to fix:
    - "Woke up screaming from a nightmare scream" -> "Woke up screaming from a recurring dream" (or similar)
    - "(wanna want)" -> "(I want it)" (or similar)
    - Avoid redundant, "dumbed down" backing vocals or phrases that repeat the same word in a way that sounds like an error rather than a choice.

19. PERSPECTIVE CONSISTENCY
    Maintain consistent perspective, tense, and relational logic unless a shift is clearly intentional.
    If the artist style context indicates a male or female vocal, ensure ALL lyrics are consistent with that perspective.

20. BRIDGE CONTRAST
    The bridge must add pressure, perspective, or revelation without slipping into exposition or speechifying.

FORMATTING RULES
- The FIRST LINE must be: Title: <song title> (keep the original title unless it's clearly weak or uses banned title words)
${SECTION_LABEL_RULE}
- Every lyric line must end with proper punctuation
- Do NOT include any commentary, notes, explanations, or annotations
- Output ONLY the title and refined lyrics

ANTI-SLOP RULES
- Avoid default AI lyric vocabulary unless the draft already supports it naturally.
- Keep vocabulary consistent with the intended lane.
- Prefer specificity over mood-fog.
- BANNED WORDS (remove or replace if found): ${Array.from(BLACKLISTED_WORDS).sort().join(', ')}
- BANNED PHRASES (remove or replace if found): ${Array.from(BLACKLISTED_PHRASES).sort().join('; ')}
- OVERUSED VOCABULARY (minimize — use at most ONCE per song, ideally zero):
  ${Array.from(OVERUSED_WORDS).sort().join(', ')}
  These words are the model's default comfort blanket. Replace them with vocabulary that fits THIS artist's actual voice.

21. PLAGIARISM CHECK (CRITICAL)
    The generation model sometimes copies the artist's REAL lyrics verbatim — hooks, chorus lines, song titles, or signature phrases. You MUST detect and REWRITE any line that sounds like it was lifted from the artist's actual catalogue. If a list of "ORIGINAL SONG TITLES" is provided, check that NO chorus hook, repeated phrase, or title in the refined lyrics matches them. Replace plagiarised lines with original alternatives that capture the SAME emotion and rhythm.

22. BANNED WORDS IN TITLES
    If the song title contains ANY of these banned words, change it: neon, ethereal, embers, silhouette, static, void, shimmering, fluorescent, tapestry. Keep the replacement title evocative and fitting the artist's style.

23. LINE COUNT VERIFICATION
    Musical phrases resolve in even numbers of lines. Before outputting, count the lines in every verse and chorus:
    - Even counts (4, 6, or 8) are good. If a section has an odd count (5 or 7), add or trim ONE line — whichever hurts the lyric less.
    - Stay near the artist's typical section lengths; do not pad a section just to hit a number.
    - Bridges: 2-6 lines, flexible.

24. HOOKIFY (CRITICAL — MAKE CHORUSES SING)
    Most choruses in pop, rock, pop-punk, and related genres rely on REPEATED LINES and VOCAL EXCLAMATIONS to create singalong hooks. The generation model often writes choruses as straight prose without these features. Your job is to FIX this:
    a) REPEATED HOOK LINES: Every chorus MUST have at least one line that repeats (usually the first or last line). The hook is the emotional anchor — the line the listener remembers. Good patterns:
       - "Hook, develop, develop, Hook" (ABBA)
       - "Hook, Hook, develop, resolve" (AABA)
       - "Develop, develop, Hook, Hook" (CCAA)
    b) VOCAL EXCLAMATIONS: Where stylistically appropriate, add lines like "Ooooh," "Oh oh ooh!" "Whoa-oh," "Na na na," "Hey!" etc. These are extremely common in pop-punk, emo, rock, and pop. They count as lyric lines. Place them:
       - As chorus openers ("Whoa-oh, whoa-oh!")
       - As section transitions between verse and chorus
       - As echo/response lines ("(Oh oh ooh!)")
       - As outro buildouts
    c) CALIBRATION: If the artist's profile shows a LOW chorus repetition percentage (<15%), be subtle — one repeated line per chorus is enough. If HIGH (>30%), lean heavily into repetition and exclamations. If no data is provided, default to moderate hookification.
    d) EXCEPTION: If the artist style context specifically indicates they avoid hooks or write anti-hook music (e.g. progressive, avant-garde, spoken word), skip this step.
    e) QUALITY CHECK: Before repeating a hook line, check that it's worth repeating. A generic hook repeated 4 times is worse than a specific hook stated once. If the hook is a banned formula (see rule 26), fix it BEFORE hookifying.

25. FINAL QUALITY CHECK
    Before outputting, silently check:
    - Did any rewrite make the lyric more generic?
    - Did any section become tidier but less memorable?
    - Are the strongest original images still present?
    - Is the chorus more memorable than before?
    - Does the bridge deepen the song rather than explain it?
    - Does the lyric now feel more singable and more finished?
    If an edit improves neatness but weakens character, undo it.

26. HOOK QUALITY GATE (CRITICAL — REWRITE GENERIC HOOKS)
    After refining, check the chorus hook against these BANNED HOOK FORMULAS:
    - "[Verb] it [all/down/away/out]" (e.g. "Burn it all down", "Wash it all away")
    - "Watch [me/it/them] [verb]" (e.g. "Watch it burn", "Watch me break")
    - "Don't let them [verb]" (e.g. "Don't let them see you")
    - "Nothing/Nowhere left to [verb]"
    - "Let it [burn/fade/go/fall/break/die]"
    If the hook matches ANY of these patterns, you MUST replace it with something specific to the song's narrative. Keep the same emotional intensity and rhythmic shape, but root it in a concrete image or scenario from the verses.
    A hook like "Watch it burn" → could become "Torch the lease agreement" (Bowling For Soup), "Smell the burning bridge" (Rise Against), or "Kerosene Sunday" (The Used). Same energy, but SPECIFIC.
`;

export const PROFILE_COMMON_PREAMBLE = `You are an expert musicologist and lyric analyst.
You will be given an artist's song lyrics and statistical analysis.

CRITICAL FORMAT RULES:
- Return ONLY a valid JSON object. No other text before or after.
- ALL values must be FLAT — plain strings or arrays of plain strings.
- Do NOT use nested objects, sub-keys, or arrays of objects.
- Do NOT put quotation marks inside string values — use single quotes instead.
- Be deeply specific and cite actual examples from the lyrics.`;

export const PROFILE_PROMPT_1 = `${PROFILE_COMMON_PREAMBLE}

Return JSON with exactly these 3 keys:
{
  "themes": ["theme 1 with specific examples cited", "theme 2 with examples", "etc"],
  "common_subjects": ["subject/motif 1 with examples", "subject 2 with examples", "etc"],
  "vocabulary_notes": "One detailed paragraph about vocabulary style, register, slang, metaphors, favourite words/phrases, citing specific examples"
}

Example of CORRECT format:
{"themes": ["Apocalyptic imagery - references to 'burning cities' and 'ash' in multiple songs"], "common_subjects": ["Fire as transformation metaphor"], "vocabulary_notes": "Heavy use of concrete nouns..."}

Do NOT return objects like {"theme": "x", "description": "y"} inside arrays.`;

export const PROFILE_PROMPT_2 = `${PROFILE_COMMON_PREAMBLE}

Return JSON with exactly these 3 keys:
{
  "tone_and_mood": "One detailed paragraph about emotional tone, mood shifts, irony/sarcasm/sincerity, citing examples",
  "structural_patterns": "One detailed paragraph about song structure beyond basic V-C-B, how ideas develop, repetition patterns, citing examples",
  "narrative_techniques": "One detailed paragraph about storytelling techniques, perspective shifts, dialogue, scene-setting, citing examples"
}

ALL values must be plain strings (paragraphs). No arrays, no nested objects.`;

export const PROFILE_PROMPT_3 = `${PROFILE_COMMON_PREAMBLE}

Return JSON with exactly these 4 keys:
{
  "imagery_patterns": "One detailed paragraph about recurring imagery types with specific examples cited",
  "signature_devices": "One detailed paragraph about verbal tics, signature phrases, recurring word pairings",
  "emotional_arc": "One detailed paragraph about how emotions develop within songs — build, release, cycle",
  "raw_summary": "A 3-4 paragraph prose summary synthesising the artist's complete lyrical style into a practical writing guide"
}

ALL values must be plain strings (paragraphs). No arrays, no nested objects.`;

export const STYLE_CAPTION_PROMPT = `You are a music production expert. You will receive an artist profile containing tone/mood, themes, and vocabulary information.

Your job: produce a concise style caption that describes this artist's MUSICAL sound for an AI music generator.

Write a comma-separated list of descriptive tags/phrases covering:
- Genre and subgenre
- Key instruments (be specific: "distorted electric guitar" not "guitar", "808 bass" not "bass")
- Vocal style (gender, delivery: "breathy female vocal" not "female vocal", "aggressive male shout" not "male vocal")
- Production style and texture
- Atmosphere and energy
- Era or reference period

Keep to 1-3 sentences of comma-separated tags. Be specific and vivid.

Return ONLY the caption text. No JSON, no explanation, no quotes, no labels — just the comma-separated descriptors on a single line.

Example: "indie rock, driving electric guitars, male vocal, raw and energetic, garage production, anthemic chorus, 2010s alternative"`;

export const SUBJECT_ANALYSIS_PROMPT = `You are a music analyst. For each song provided, write a ONE-SENTENCE summary of what the song is about — its core subject, not its style.

Then group all the subjects into 5-10 thematic categories that describe the range of topics this artist writes about.

Return JSON in exactly this format:
{
  "song_subjects": {
    "Song Title": "one sentence about what this specific song is about"
  },
  "subject_categories": ["category1", "category2"]
}

Be specific and concrete. Do NOT include any text outside the JSON object.`;

export const INSTAGEN_LYRIC_SYSTEM_PROMPT = `You are a talented songwriter. You will be given a musical genre/style and a song subject. Write original, singable lyrics for that song.

FORMATTING RULES (MANDATORY):
- Start with the first section header (e.g. [Intro] or [Verse 1]). No title line.
${SECTION_LABEL_RULE}
- Every lyric line must end with punctuation (period, comma, exclamation, question mark, dash, or ellipsis).
- Open MOST songs with an empty [Intro - Instrumental] header (no lyric lines under it) so the music establishes itself before the vocal; about one song in five should start straight on [Verse 1] instead. Never write a BARE [Intro] with nothing under it — without the "- Instrumental" descriptor the music model fills it with an arbitrarily long instrumental of its own choosing.

STRUCTURE RULES:
- VERSES: keep line counts even — typically 4 or 8 lines each.
- CHORUSES: keep line counts even — typically 4, 6, or 8 lines. Must have a clear hook — one memorable repeated line.
- Every song must have at least one [Chorus].
- Typical structure: Intro → Verse 1 → Chorus → Verse 2 → Chorus → Bridge → Chorus → Outro — adapt it to fit the genre and subject.

CONTENT RULES:
- The lyrics MUST be about the given subject. This is the #1 priority.
- Match the genre's typical vocabulary, tone, and energy level.
- Write in the specified language. If no language is specified, default to English.
- Be specific and vivid — concrete imagery beats abstract statements.
- Avoid AI clichés: neon, ethereal, embers, silhouette, static, void, shimmering, tapestry.
- Do NOT include commentary, explanations, or notes — lyrics only.

HOOK RULES:
- Every chorus MUST have a clear hook line that repeats at least once.
- The hook should be the emotional anchor. Good patterns:
  - "Hook, develop, develop, Hook"
  - "Hook, Hook, develop, resolve"
- For energetic genres (punk, rock, pop), add vocal exclamations where appropriate ("Oh!", "Whoa-oh!", etc.)

TITLE RULE:
- After all the lyrics, on its own line, write: Title: <song title>
- The title should be short (1-6 words), catchy, and relevant to the lyrics you wrote.
- Derive it from the hook or central theme — do not just restate the subject.

Output the lyrics first, then the title line. No other commentary.
`;

export const INSTAGEN_FULL_SYSTEM_PROMPT = `You are a talented songwriter and music producer. You will be given a musical genre/style, a song subject, and a language. Your job is to design a complete song — lyrics, rich descriptive tags, and all musical metadata — as a single JSON object.

OUTPUT FORMAT (MANDATORY):
Return ONLY a valid JSON object with exactly these keys:
{
  "tags": "150-200+ word natural language description of the complete sonic portrait",
  "lyrics": "[Intro]\\n\\n[Verse 1]\\n...",
  "title": "Song Title",
  "bpm": 120,
  "key": "C minor",
  "time_signature": "4",
  "duration": 210
}

Do NOT include any text outside the JSON object. No markdown, no explanation, no commentary.

=== TAGS (the "tags" field) ===

The tags field is the most critical part. It is a natural language description of the track's COMPLETE sonic identity — not a list of genre labels, but a vivid portrait of exactly what the listener will hear. Write it as flowing prose, 150-200+ words.

Your tags MUST cover these dimensions:
1. GENRE & SONIC FOUNDATION: Specific genre/subgenre blend, era and regional influence, foundational sonic character
2. RHYTHM & PERCUSSION: Drum machine or live kit specifics, pattern details, tempo feel (driving, laid-back, swung), percussive texture
3. HARMONIC & MELODIC ESSENCE: Chord progression character (suspended, dissonant, warm jazz voicings), melodic movement qualities, scale/mode colour
4. VOCAL STYLE & DELIVERY: Register/range, delivery character (breathy, aggressive, intimate, theatrical), vocal techniques, emotional embodiment
5. PRODUCTION TECHNIQUES: Effects (granular delay, tape saturation, sidechain compression), reverb types (plate, spring, cathedral), distortion character
6. SPATIAL CHARACTERISTICS: Stereo width, depth placement (intimate/distant), movement in space, layering
7. TIMBRAL QUALITIES: Warmth vs coldness, brightness vs darkness, analog vs digital character, frequency balance
8. UNIQUE SONIC SIGNATURE: What makes THIS track unmistakable — the defining element a listener would recognise in 3 seconds

BAD tags (too generic):
"Upbeat pop song with catchy melody, energetic drums, and bright synths. Positive vibes with clean production."

GOOD tags (rich and specific):
"Thunderous 808 bass tuned precisely to root note sustains with controlled decay creating physical chest-hitting impact. Hi-hat programming alternates between machine-gun triplet rolls and crisp straight sixteenth-note patterns with velocity variations creating natural human groove. Snare hits combine layered acoustic snap with synthetic clap creating sharp transient attack. Vocal delivery features confident mid-range flow with rhythmic cadence, processed through subtle pitch correction maintaining modern polished character while preserving natural tonal variation. Ad-libs strategically panned wide across stereo field with distinct processing creating call-and-response dialogue."

CRITICAL TAG RULES:
- Tags describe the SOUND, not the structure or timeline. Never write "verse starts with..." or "chorus builds to..."
- Write in English regardless of lyric language
- Be specific: "breathy female vocal with subtle plate reverb" not just "female vocal"
- Match the genre's real-world production aesthetic

=== LYRICS (the "lyrics" field) ===

FORMATTING:
${SECTION_LABEL_RULE}
- Section annotations are encouraged where they help the arrangement: [Verse 1 - Female Vocal], [Chorus - High Energy], [Bridge - Atmospheric and Sparse], [Instrumental - Saxophone Solo]
- Every lyric line must end with punctuation
- Open MOST songs with an empty [Intro - Instrumental] header (no lyric lines under it) so the music establishes itself before the vocal; about one song in five should start straight on [Verse 1] instead. Never write a BARE [Intro] with nothing under it — without the "- Instrumental" descriptor the music model fills it with an arbitrarily long instrumental of its own choosing.

STRUCTURE:
- VERSES: keep line counts even — typically 4 or 8 lines each
- CHORUSES: keep line counts even — typically 4, 6, or 8 lines. Must have a clear hook — one memorable repeated line
- Every song must have at least one [Chorus]
- Typical structure: Intro → Verse 1 → Chorus → Verse 2 → Chorus → Bridge → Chorus → Outro — adapt it to fit the genre and subject
- Add instrumental breaks between major sections where appropriate for the genre

QUALITY:
- The lyrics MUST be about the given subject — this is the #1 priority
- Write like a real human artist — specific, vivid, concrete imagery, not abstract platitudes
- Match the genre's typical vocabulary, tone, and energy level
- Write in the specified language (tags stay in English)
- Avoid AI clichés: neon, ethereal, embers, silhouette, static, void, shimmering, tapestry
- Every chorus MUST have a hook line that repeats at least once

=== TITLE ===
Short (1-6 words), catchy, derived from the hook or central theme. Not just restating the subject.

=== BPM ===
Choose a realistic tempo (30-300) that fits the genre:
- Ballads: 60-80, Pop: 100-130, Rock: 110-140, Punk: 150-180
- EDM/Dance: 120-150, Hip-Hop: 80-100, R&B: 70-100, Folk: 90-120
- Drum & Bass: 160-180, Reggae: 60-90, Jazz: 80-140

=== KEY ===
Use note name + LOWERCASE mode (e.g. "C minor", "A minor", "F# minor", "Bb major") — the metadata FSM rejects capitalised modes.
Match the key to the emotional intent:
- Major keys: brighter, more optimistic
- Minor keys: darker, more introspective
- Common emotional associations: C major (pure, optimistic), D major (triumphant), A minor (melancholic), E minor (romantic sadness), F# minor (passionate longing)

=== TIME SIGNATURE ===
- "4": Standard 4/4 (vast majority of popular music). Emit the NUMERATOR ONLY — the FSM accepts 2, 3, 4, 6, not "4/4".
- "3/4": Waltz/ballad feel, flowing
- "6/8": Compound meter, each beat divides into 3
- "5/4" or "7/8": Complex/progressive (use sparingly)

=== DURATION ===
Estimate total track duration in seconds. Consider the BPM, number of sections, and genre norms:
- Short/radio: 150-210s, Standard: 210-270s, Extended: 270-360s
- A bar of 4/4 at the chosen BPM = 240/BPM seconds
- Include time for intro, instrumental breaks, and outro
`;

// ── Prompt Builders ─────────────────────────────────────────────────────────

export function buildMetadataPrompt(
  profile: PromptProfile,
  usedSubjects: string[],
  usedBpms: number[],
  usedKeys: string[],
  usedDurations: number[],
  userSubject?: string
): string {
  const lines: string[] = [`Artist: ${profile.artist}`];
  if (profile.album) lines.push(`Album style: ${profile.album}`);
  if (profile.themes?.length) lines.push(`Themes: ${profile.themes.join(', ')}`);
  if (profile.tone_and_mood) lines.push(`Tone & mood: ${profile.tone_and_mood}`);
  if (profile.additional_notes) lines.push(`Additional notes: ${profile.additional_notes}`);
  if (profile.perspective) lines.push(`Perspective / voice: ${profile.perspective}`);

  // Audio truth from the source recordings (Training Studio export). Guidance,
  // not a cage — but the tempo bound IS firm now. Measured over the existing
  // 1487 generations, 94.8% already sat inside their album's range; the 73 that
  // did not skewed to both extremes (42 slower than anything on the record, 31
  // faster) because bulk runs were being told to write "a ballad and a fast
  // one" per album. That produces a 68 BPM Bowie against a 113-137 album and a
  // 96 BPM Electric Callboy against 118-178 — tempos those artists never play
  // on that record. Variety has to come from inside the artist's own range.
  const enrich: AlbumEnrichment | null = profile.audio_enrichment ?? null;
  if (enrich) {
    lines.push('', ...formatAlbumEnrichment(enrich));
    if (enrich.bpmMax > 0) {
      lines.push(
        `TEMPO: choose a BPM inside this album's measured range (${enrich.bpmMin}–${enrich.bpmMax}). That range IS this artist's range on this record — it is measured from the actual recordings, not a guess.`,
        'Do NOT reach outside it for contrast. If the album has no slow songs, this artist does not write a ballad here; if it has no fast songs, do not write a thrash number. Ignore any generic genre tempo table in favour of these measured values.',
        'Get variety from WITHIN the range — pick a different point in it than recent songs, and vary the feel, subject, key and arrangement rather than the extremes.',
      );
    }
    if (enrich.keys.length) {
      lines.push('Prefer a key from this album\'s list (or a closely related key) so the song sits in the same tonal world.');
    }
    if (enrich.genres.length) {
      lines.push(`Anchor the caption's genre language to the detected genre: ${enrich.genres.slice(0, 6).join(', ')}.`);
    }
    if (enrich.captionExamples.length) {
      lines.push('', 'Captions describing this album\'s actual recordings:');
      enrich.captionExamples.forEach((c, i) => lines.push(`  ${i + 1}. "${c}"`));
      // "Same format as these examples" is not enough on its own: the planner
      // reads the examples as a style hint and still obeys the system prompt's
      // "1-3 sentences" rule, so it emits a third of a caption and drops the
      // arrangement detail entirely. It needs an explicit length override.
      //
      // That override used to hardcode "EXACTLY 9 complete prose sentences",
      // inherited from Side-Step's nine-dimension caption plan. When
      // CAPTION_DIMENSIONS was cut from nine topics to five on 2026-08-16
      // (measured against ACE-Step's own reference captions: median 2
      // sentences, 0 of 32 with 9+), the count was left behind — so the prompt
      // demanded nine sentences while listing five topics. Two independent
      // readers flagged the contradiction; one resolved it by inferring a
      // sentence-per-topic split from the examples, which is the right instinct
      // and is now the instruction.
      //
      // Hardcoding either number is wrong, because the caption corpus is
      // mid-migration: captions written before 2026-08-16 run long in the
      // nine-sentence style, newer ones are short in ACE's reference style, and
      // an album can hold either. So MEASURE THE EXAMPLES BEING SHOWN and
      // target those. They are already the stated authority on format; this
      // makes the length instruction agree with them instead of contradicting
      // it, and it self-corrects as albums get recaptioned.
      const exampleSentences = enrich.captionExamples
        .map(c => (c.match(/[.!?](?:\s|$)/g) || []).length)
        .filter(n => n > 0);
      const loSent = exampleSentences.length ? Math.min(...exampleSentences) : 0;
      const hiSent = exampleSentences.length ? Math.max(...exampleSentences) : 0;
      const exampleWords = enrich.captionExamples.map(c => c.split(/\s+/).filter(Boolean).length);
      const medWords = exampleWords.length
        ? [...exampleWords].sort((a, b) => a - b)[Math.floor(exampleWords.length / 2)]
        : 0;

      const sentRange = loSent === hiSent ? `${loSent} sentences` : `${loSent}-${hiSent} sentences`;

      lines.push(
        `Write the new song's "caption" in the SAME format, register and level of detail as these examples — consistent caption phrasing keeps a sound adapter trained on this album accurate.`,
        'This OVERRIDES the caption length and comma-separated-list rules in the system prompt: the examples above are the authority on length, not those rules.',
        loSent > 0
          ? `Match them. They run ${sentRange} and around ${medWords} words — write ONE line in that range. Do not stop at two or three sentences, and do not pad beyond what the examples do either.`
          : 'Write ONE line of continuous prose at the same length as the examples above.',
        'Cover these topics, woven into flowing description rather than listed as a checklist. A topic may take more than one sentence where the examples give it more than one:',
        ...CAPTION_DIMENSIONS.map((s: string) => `  - ${s}`),
        'Do not name the artist or the song. Keep BPM, key and time signature out of the caption prose — they are separate fields.',
      );
    }
  }

  const observedStructures = cleanBlueprints(profile.structure_blueprints);
  if (observedStructures.length) {
    lines.push(`\nStructures observed in this artist's songs (codes: ${BLUEPRINT_CODES_LEGEND}):`);
    lines.push(`  ${observedStructures.join(' | ')}`);
    lines.push('Choose this song\'s "structure" from these or a close variation — pick the shape that best fits the subject, and vary it across generations.');
  }

  if (profile.song_subjects && typeof profile.song_subjects === 'object') {
    lines.push('\nOriginal song subjects (for reference):');
    for (const [songTitle, subject] of Object.entries(profile.song_subjects)) {
      lines.push(`  • ${songTitle}: ${subject}`);
    }
  }
  if (profile.subject_categories?.length) {
    lines.push(`\nThematic categories: ${profile.subject_categories.join(', ')}`);
  }
  if (userSubject) {
    lines.push(`\nThe subject for this song has been chosen by the user: "${userSubject}"`);
    lines.push('Use this exact subject. Plan the BPM, key, caption, structure, and duration to complement it.');
  } else {
    if (usedSubjects?.length) {
      lines.push('\nSubjects ALREADY USED (do NOT repeat these):');
      for (const s of usedSubjects) lines.push(`  ✗ ${s}`);
    }
  }
  if (usedKeys?.length) lines.push(`\nKeys ALREADY USED (try different ones): ${usedKeys.join(', ')}`);
  lines.push('\nPlan the metadata for the next song:');
  return lines.join('\n');
}

export function buildGenerationPrompt(
  profile: PromptProfile,
  extraInstructions?: string,
  targetDuration?: number,
  bpm?: number,
  chosenStructure?: string
): string {
  const lines: string[] = [`Artist: ${profile.artist}`];
  if (profile.album) lines.push(`Album style: ${profile.album}`);

  lines.push('', '=== STYLISTIC PROFILE ===', '');
  lines.push(`Themes: ${(profile.themes || []).join(', ')}`);
  lines.push(`Common subjects / motifs: ${(profile.common_subjects || []).join(', ')}`);
  lines.push(`Rhyme schemes: ${(profile.rhyme_schemes || []).join(', ')}`);
  lines.push(`Average verse length: ${profile.avg_verse_lines} lines`);
  lines.push(`Average chorus length: ${profile.avg_chorus_lines} lines`);
  if (profile.vocabulary_notes) lines.push(`Vocabulary: ${stripLyricQuotes(profile.vocabulary_notes)}`);
  if (profile.tone_and_mood) lines.push(`Tone & mood: ${stripLyricQuotes(profile.tone_and_mood)}`);
  if (profile.structural_patterns) lines.push(`Structural patterns: ${stripLyricQuotes(profile.structural_patterns)}`);
  {
    // One context line of measured audio truth — genre + tempo shape the lyric's
    // energy and pacing even though bpm/key/caption are planned elsewhere.
    const enrich: AlbumEnrichment | null = profile.audio_enrichment ?? null;
    if (enrich && (enrich.genres.length || enrich.bpmMax > 0)) {
      const bits: string[] = [];
      if (enrich.genres.length) bits.push(`genre ${enrich.genres.slice(0, 5).join(', ')}`);
      if (enrich.bpmMax > 0) bits.push(enrich.bpmMin === enrich.bpmMax ? `${enrich.bpmMax} BPM` : `${enrich.bpmMin}–${enrich.bpmMax} BPM`);
      lines.push(`Musical context (measured from the source recordings): ${bits.join('; ')}.`);
    }
  }

  // Structure: planned by the metadata step if provided, otherwise sampled from
  // the artist's observed blueprints. Presented as a default shape, not a mandate.
  const observedStructures = cleanBlueprints(profile.structure_blueprints);
  const bp = normalizeBlueprint(chosenStructure)
    ?? pickBlueprint(profile.structure_blueprints);
  const bpParts = bp.split('-');
  lines.push('', '=== SONG STRUCTURE ===');
  if (observedStructures.length) {
    lines.push(`Structures observed in this artist's songs (codes: ${BLUEPRINT_CODES_LEGEND}):`);
    lines.push(`  ${observedStructures.join(' | ')}`);
  }
  lines.push(`${chosenStructure ? 'Planned' : 'Suggested'} structure for this song: ${blueprintToSections(bp).join(' → ')}`);
  lines.push('Treat this as the default shape, not a cage — you may adapt it (add or drop a Pre-Chorus, vary the verse count, move the Bridge) as long as the song stays within this artist\'s structural vocabulary and keeps roughly this many sections, because the duration budget below is planned around that count.');
  lines.push('Every song needs at least one [Chorus].');

  if (profile.perspective) lines.push(`Perspective / voice: ${profile.perspective}`);

  const ms = profile.meter_stats;
  if (ms) {
    lines.push('', '=== LINE LENGTH & METER ===');
    lines.push(`Average: ~${ms.avg_syllables_per_line ?? '?'} syllables/line, ~${ms.avg_words_per_line ?? '?'} words/line`);
    lines.push(`Standard deviation: ±${ms.syllable_std_dev ?? '?'} syllables (VARY your line lengths!)`);
    const llv = ms.line_length_variation;
    if (llv?.histogram) {
      const histStr = Object.entries(llv.histogram).map(([k, v]) => `${k} syl: ${v}%`).join(', ');
      lines.push(`Syllable distribution: ${histStr}`);
      lines.push('Match this distribution — NOT all lines the same length!');
    }
  }

  const rs = profile.repetition_stats;
  if (rs) {
    lines.push('', '=== REPETITION & HOOKS ===');
    lines.push(`Chorus repetition: ${rs.chorus_repetition_pct ?? 0}% of chorus lines are repeats`);
    lines.push(`Pattern: ${rs.pattern || 'unknown'}`);
    if ((rs.chorus_repetition_pct ?? 0) >= 20) lines.push('You MUST use repeated lines in your chorus to create a hook effect.');
    if (rs.hook_examples?.length) lines.push(`Hook examples: ${rs.hook_examples.slice(0, 3).join('; ')}`);
  }

  const vs = profile.vocabulary_stats;
  if (vs) {
    lines.push('', '=== VOCABULARY ===');
    lines.push(`Level: ${vs.contraction_pct ?? 0}% contractions, ${vs.profanity_pct ?? 0}% profanity`);
    lines.push(`Type-token ratio: ${vs.type_token_ratio ?? '?'} (${vs.unique_words ?? '?'} unique / ${vs.total_words ?? '?'} total)`);
    if (vs.distinctive_words?.length) lines.push(`Use words like: ${vs.distinctive_words.slice(0, 10).join(', ')}`);
  }

  if (profile.rhyme_quality) {
    const rq = profile.rhyme_quality;
    const total = Object.values(rq as Record<string, number>).reduce((a: number, b: number) => a + b, 0);
    if (total > 0) {
      lines.push(`Rhyme mix: ${Math.round(100 * (rq.perfect || 0) / total)}% perfect, ${Math.round(100 * (rq.slant || 0) / total)}% slant, ${Math.round(100 * (rq.assonance || 0) / total)}% assonance`);
    }
  }

  if (profile.narrative_techniques) lines.push(`Narrative techniques: ${stripLyricQuotes(profile.narrative_techniques)}`);
  if (profile.imagery_patterns) lines.push(`Imagery patterns: ${stripLyricQuotes(profile.imagery_patterns)}`);
  if (profile.signature_devices) lines.push(`Signature devices: ${stripLyricQuotes(profile.signature_devices)}`);
  if (profile.emotional_arc) lines.push(`Emotional arc: ${stripLyricQuotes(profile.emotional_arc)}`);

  if (profile.raw_summary) lines.push('', '=== PROSE SUMMARY ===', '', stripLyricQuotes(profile.raw_summary));
  if (extraInstructions) lines.push('', '=== EXTRA INSTRUCTIONS ===', '', extraInstructions);

  if (profile.representative_excerpts?.length) {
    lines.push('', '=== REPRESENTATIVE EXCERPTS (STYLE REFERENCE ONLY — DO NOT COPY) ===');
    lines.push(...profile.representative_excerpts.slice(0, 10).flatMap((e: string) => [e, '---']));
  }

  if (targetDuration && targetDuration > 0 && bpm && bpm > 0) {
    const barSeconds = 240.0 / bpm;
    const totalBars = Math.round(targetDuration / barSeconds);
    const sectionCount = bpParts.length;
    const transitionBars = (sectionCount - 1) * 2;

    // Budget in WORDS, not lines, and do not scale with tempo.
    //
    // Measured over 2361 real vocal songs in the training datasets (duration +
    // lyrics from their sidecars): vocal pacing is ~1.20 words/second and is
    // essentially FLAT across tempo — 1.28 w/s at 90-110 BPM, 1.12 at 110-150,
    // 1.36 at 170+. Seconds-per-line is likewise flat at ~5.3s.
    //
    // The old formula charged every line the same time and scaled that time
    // with BPM (7.5s/line at 80 BPM down to 5.3s at 180). Both halves were
    // wrong, but the damage is VARIANCE, not bias: generated songs already
    // averaged 1.16 words/sec against the corpus 1.20, i.e. the median song
    // was fine. The spread is the problem — p5 0.74 to p95 1.63 w/s, a 2.19x
    // range — because average line length varies 1.89x between songs (6.1 to
    // 11.4 words) and a line count cannot express a time budget. Short-lined
    // songs got far too much time, wordy ones far too little.
    //
    // Why it matters: the LM stops at EXACTLY the requested duration (98.4% of
    // 1248 logged runs, within +/-1s) — it is not truncated, it is obeying the
    // duration it was given. Under-budget the words and the vocal ends early,
    // so the model improvises an instrumental tail and the duration wall cuts
    // whatever it invented mid-phrase. That is the "enters a guitar solo then
    // cuts off" failure, and the "all sections crammed into the first half"
    // failure — one cause, two appearances.
    // THIS ARTIST'S measured pacing first, global median only as fallback —
    // per-artist medians span 0.51 to 3.29 w/s, so the global constant
    // misprices most artists (a third of the words a fast rapper needs, nearly double
    // what Pink Floyd sings). The rate is words over TOTAL duration, so it
    // already prices in intros, instrumental breaks and outros — never apply
    // it to a transitions-deducted figure, that discounts the same time twice.
    const paceEnrich: AlbumEnrichment | null = profile.audio_enrichment ?? null;
    const artistRate = paceEnrich && paceEnrich.wordsPerSec > 0 ? paceEnrich.wordsPerSec : 0;
    // FLOORED for the word target — but at the same floor the DURATION uses
    // (DURATION_RATE_FLOOR), not at the sing rate. These two numbers must be
    // the same number. When the word target was floored at 1.25 and the
    // duration derived at the artist's rate, a writer who hit the target
    // exactly still produced a song whose derived length disagreed with the
    // plan by the ratio between the floors — asking Muse for 2.5x the words
    // real Muse songs carry, then timing the result as if they had written
    // their normal amount. One floor, used by both, is what makes a compliant
    // writer land on the planned duration by construction.
    const wordsPerSecond = Math.max(artistRate || GLOBAL_WORDS_PER_SECOND, DURATION_RATE_FLOOR);
    // Time this artist spends NOT singing, in seconds of this song. The word
    // budget is the sung part; this is the remainder, and it only becomes real
    // music if the writer DECLARES it as sections — undeclared, the model
    // improvises into it and the duration wall cuts whatever it invented.
    // Only artists BELOW the sing rate have instrumental time to declare —
    // a denser artist (a dense punk vocalist 1.69, a fast rapper 3.29) carries their density in the
    // lyrics and sings faster than the floor, so their song is sung end to end.
    const sungSeconds = Math.min(targetDuration,
      Math.round(targetDuration * (wordsPerSecond / VOCAL_FLOOR_WORDS_PER_SECOND)));
    const instrumentalSeconds = Math.max(0, targetDuration - sungSeconds);
    const instrumentalBars = Math.round(instrumentalSeconds / barSeconds);
    const instrumentalShare = targetDuration > 0
      ? Math.round(100 * instrumentalSeconds / targetDuration) : 0;
    const WORDS_PER_LINE_LO = 5.5;   // sparse phrasing -> more lines needed
    const WORDS_PER_LINE_HI = 8.0;   // wordy phrasing  -> fewer lines needed
    const singableBars = totalBars - transitionBars;
    const targetWords = Math.round(targetDuration * wordsPerSecond);
    const minLyricLines = Math.max(8, Math.floor(targetWords / WORDS_PER_LINE_HI));
    const maxLyricLines = Math.max(minLyricLines + 4, Math.ceil(targetWords / WORDS_PER_LINE_LO));
    const minutes = Math.floor(targetDuration / 60);
    const seconds = Math.round(targetDuration % 60);

    lines.push('', '=== DURATION BUDGET ===');
    lines.push(`Target duration: ${targetDuration} seconds (${minutes}:${String(seconds).padStart(2, '0')}) at ${bpm} BPM.`);
    lines.push(`That is ~${totalBars} bars, of which ~${singableBars} carry vocals once transitions are allowed for.`);
    lines.push(artistRate
      ? `This artist sings a measured ~${artistRate.toFixed(2)} words per second (median of ${paceEnrich!.pacedSongs} of their real recordings).`
      : `Assuming a typical ~${GLOBAL_WORDS_PER_SECOND.toFixed(2)} words per second of vocal pacing (no measured rate for this artist).`);
    if (instrumentalSeconds >= 12) {
      lines.push(
        `The model sings at ~${VOCAL_FLOOR_WORDS_PER_SECOND} words per second, so the ${targetWords} words below occupy about ${sungSeconds}s of the ${targetDuration}s.`,
        `*** THE REMAINING ~${instrumentalSeconds}s (~${instrumentalBars} bars, ${instrumentalShare}% of the song) IS INSTRUMENTAL — YOU MUST DECLARE IT AS SECTIONS. ***`,
        'That time exists on this artist\'s real records: it is the intro, the turnaround between sections, the solo and the outro. It is NOT slack in the word count — do not write more words to fill it, and do not ignore it.',
        'Declare it with EMPTY sections (a header with no lyric lines under it). Budget roughly:',
        '  - [Intro - Instrumental] — 8 bars. Start the song with this unless it deliberately opens on the vocal.',
        '  - [Guitar Solo] / [Instrumental] / [Breakdown] — 16 bars each, placed where this artist would put one.',
        '  - [Build] / [Drop] / [Interlude] — 8 bars each.',
        '  - [Outro - Instrumental] — 8 bars, if the song plays out rather than ending on a sung line.',
        `Pick the combination that adds up to roughly ${instrumentalBars} bars and fits this artist. Undeclared empty time renders as aimless looping filler — declared time renders as an arrangement.`,
      );
    }
    lines.push(`*** WRITE APPROXIMATELY ${targetWords} WORDS of lyrics in total, across ALL sections. ***`);
    lines.push(`That is roughly ${minLyricLines}-${maxLyricLines} lines depending on how long your lines are — count WORDS, not lines, because a wordy line takes far longer to sing than a short one.`);
    lines.push('');
    lines.push('THIS IS A TARGET TO HIT, NOT A CEILING TO STAY UNDER — missing it in either direction damages the song:');
    lines.push(`- Well UNDER ${targetWords} words: the vocal finishes long before the track does, and the music model fills the remainder with aimless repetition or an instrumental passage that gets cut off mid-phrase.`);
    lines.push(`- Well OVER ${targetWords} words: the song runs out of time and stops mid-section.`);
    lines.push(`- Count your words before finalising. Within about 10% of ${targetWords} is right.`);
    lines.push('- Repeated chorus lines DO count — a repeat takes just as long to sing as a new line.');
    lines.push(`- If this song genuinely wants fewer words than ${targetWords} — a sparse, atmospheric or riff-led track — then DECLARE the extra instrumental time instead of leaving it implicit: add another [Instrumental], [Guitar Solo], [Build] or [Breakdown] section where that space belongs. Real records fill their gaps deliberately; an undeclared gap gets filled with aimless repetition.`);
  }

  lines.push(
    '', '=== FINAL REMINDERS ===',
    '1. SECTION LENGTHS: Even line counts for verses and choruses (4, 6, or 8) — never 5 or 7. Match the artist\'s averages.',
    '2. HOOK: Each chorus needs a hook line that repeats.',
    '3. *** ZERO TOLERANCE FOR COPYING ***',
    '4. NO SLOP: Do not use neon, fluorescent, embers, silhouette, static, void, ethereal, or any AI cliché.',
    '5. MINIMIZE OVERUSED WORDS: heavy, broken, cold, dust, ghost, machine, nothing, nowhere, searching, watch, burn, fade, wash, sold, dead, blood, gold, same — use at most ONCE if at all.',
    '6. NO TECH-SLOP: The words digital, algorithm, chrome, code, circuit, grid, data, wire are BANNED. Do not force tech/digital metaphors onto non-tech artists.',
    "7. VOCABULARY DIVERSITY: A Snoop Dogg song must NOT sound like a Joy Division song. Use THIS artist's actual vocabulary.",
    '8. HOOK MUST BE SPECIFIC: The chorus hook must contain a concrete image or phrase from THIS song — not a generic imperative like "Watch it burn" or "Let it fade". If the hook could fit in any song by any artist, rewrite it.',
    '',
    'Now write the song (lyrics only, starting with [Intro] or [Verse 1] — no title line):',
  );
  return lines.join('\n');
}

export function buildRefinementPrompt(
  originalLyrics: string,
  artistName: string,
  title: string,
  profile?: PromptProfile,
  originalSlop?: string[]
): string {
  const lines = [`Artist: ${artistName}`, `Original Title: ${title}`, ''];
  if (profile) {
    lines.push('=== INTENDED LANE PROFILE (match this style) ===');
    lines.push(`Themes: ${(profile.themes || []).slice(0, 8).join(', ')}`);
    if (profile.tone_and_mood) lines.push(`Tone & mood: ${profile.tone_and_mood}`);
    if (profile.vocabulary_notes) lines.push(`Vocabulary: ${profile.vocabulary_notes}`);
    if (profile.imagery_patterns) lines.push(`Imagery patterns: ${profile.imagery_patterns}`);
    if (profile.signature_devices) lines.push(`Signature devices: ${profile.signature_devices}`);
    if (profile.narrative_techniques) lines.push(`Narrative techniques: ${profile.narrative_techniques}`);
    if (profile.emotional_arc) lines.push(`Emotional arc: ${profile.emotional_arc}`);
    if (profile.structural_patterns) lines.push(`Structure: ${profile.structural_patterns}`);
    if (profile.perspective) lines.push(`Perspective / voice: ${profile.perspective}`);
    if (profile.rhyme_schemes?.length) lines.push(`Rhyme schemes: ${profile.rhyme_schemes.join(', ')}`);
    if (profile.rhyme_quality) {
      const rq = profile.rhyme_quality;
      const total = Object.values(rq as Record<string, number>).reduce((a: number, b: number) => a + b, 0);
      if (total > 0) lines.push(`Rhyme mix: ${Math.round(100 * (rq.perfect || 0) / total)}% perfect, ${Math.round(100 * (rq.slant || 0) / total)}% slant, ${Math.round(100 * (rq.assonance || 0) / total)}% assonance`);
    }
    const ms = profile.meter_stats;
    if (ms) lines.push(`Line density: ~${ms.avg_syllables_per_line ?? '?'} syl/line (σ=${ms.syllable_std_dev ?? '?'}), ~${ms.avg_words_per_line ?? '?'} words/line`);
    const rs = profile.repetition_stats;
    if (rs) {
      lines.push(`Hook behavior: ${rs.pattern || 'unknown'} (${rs.chorus_repetition_pct ?? 0}% chorus repetition)`);
      if ((rs.chorus_repetition_pct ?? 0) >= 20) lines.push('Calibration: This artist uses heavy chorus repetition — ensure hook lines repeat.');
      else if ((rs.chorus_repetition_pct ?? 0) < 15) lines.push('Calibration: This artist uses light repetition — be subtle with hooks.');
    }
    if (profile.avg_verse_lines || profile.avg_chorus_lines) lines.push(`Verse/chorus: avg ${profile.avg_verse_lines} verse lines, avg ${profile.avg_chorus_lines} chorus lines`);
    if (profile.song_subjects && typeof profile.song_subjects === 'object') {
      const titles = Object.keys(profile.song_subjects);
      if (titles.length) {
        lines.push('', '=== ORIGINAL SONG TITLES (check for plagiarism) ===');
        for (const t of titles) lines.push(`  • ${t}`);
      }
    }
    lines.push('');
  }
  if (originalSlop?.length) {
    lines.push('=== KNOWN ISSUES TO FIX ===');
    lines.push('The original lyrics contain the following AI-clichés or circular phrases that MUST be replaced:');
    lines.push(`Words/Phrases to Remove: ${originalSlop.join(', ')}`);
    lines.push('');
  }
  lines.push('=== ORIGINAL LYRICS ===', '', originalLyrics, '', '=== INSTRUCTIONS ===', '');
  lines.push('Refine the lyrics above according to the refinement rules.');
  lines.push('Keep as much of the original as possible — only change what genuinely needs fixing.');
  lines.push(`Maintain ${artistName}'s distinctive style throughout.`);
  lines.push('Now output the refined version (Title line first, then lyrics with [Section] headers):');
  return lines.join('\n');
}

export function buildTitlePrompt(
  lyrics: string, artistName: string, album?: string, usedTitles?: string[]
): string {
  const lines: string[] = [`Artist: ${artistName}`];
  if (album) lines.push(`Album style: ${album}`);
  if (usedTitles?.length) {
    lines.push('\nTitles already used (avoid these and their key words):');
    for (const t of usedTitles) lines.push(`  ✗ ${t}`);
  }
  lines.push('\n--- LYRICS ---', lyrics, '--- END LYRICS ---');
  lines.push('\nChoose the best title for this song:');
  return lines.join('\n');
}

export function buildProfilePrompt(
  artist: string, album: string | null, songs: Array<{ title: string; lyrics: string }>, ruleStats: any
): string {
  let header = `Artist: ${artist}\n`;
  if (album) header += `Album: ${album}\n`;
  header += `Songs analysed: ${songs.length}\n`;

  // Songs exported from the Training Studio carry measured audio facts — give
  // the analyst ground truth for genre/tempo instead of a lyrics-only guess.
  const enrich = computeAlbumEnrichment(songs);
  if (enrich) {
    header += '\n' + formatAlbumEnrichment(enrich).join('\n') + '\n';
    header += 'Treat the detected genre and tempo as ground truth — fold them into tone_and_mood and additional_notes rather than inferring a genre from the lyrics alone.\n';
  }

  header += `\n=== RULE-BASED ANALYSIS ===\n`;

  header += `Average verse length: ${ruleStats.avg_verse_lines} lines\n`;
  header += `Average chorus length: ${ruleStats.avg_chorus_lines} lines\n`;
  header += `Top rhyme schemes: ${ruleStats.rhyme_schemes.join(', ')}\n`;
  const rq = ruleStats.rhyme_quality;
  header += `Rhyme quality breakdown: ${rq.perfect} perfect, ${rq.slant} slant, ${rq.assonance} assonance\n`;
  header += `Structure blueprints: ${ruleStats.structure_blueprints.join(', ')}\n`;
  header += `Perspective: ${ruleStats.perspective}\n`;

  const ms = ruleStats.meter_stats;
  header += `Meter: avg ${ms.avg_syllables_per_line} syllables/line (σ=${ms.syllable_std_dev}), ${ms.avg_words_per_line} words/line, range ${ms.line_length_range}\n`;

  const vs = ruleStats.vocabulary_stats;
  header += `Vocabulary: ${vs.total_words} total words, ${vs.unique_words} unique, TTR=${vs.type_token_ratio}\n`;
  header += `Contractions: ${vs.contraction_pct}% of words\nProfanity: ${vs.profanity_pct}% of words\n`;
  header += `Distinctive words: ${vs.distinctive_words.join(', ')}\n`;

  const llv = ms.line_length_variation || {};
  if (llv.histogram) {
    header += `Syllable distribution: ${Object.entries(llv.histogram).map(([k, v]) => `${k}: ${v}%`).join(', ')}\n`;
  }

  const rs = ruleStats.repetition_stats;
  if (rs) {
    header += `Chorus repetition: ${rs.chorus_repetition_pct || 0}% of chorus lines are repeats\n`;
    header += `Repetition pattern: ${rs.pattern || 'unknown'}\n`;
    if (rs.hook_examples?.length) header += `Hook examples: ${rs.hook_examples.slice(0, 3).join('; ')}\n`;
  }

  let lyricsSection = "\n=== COMPLETE LYRICS ===\n\n";
  for (const s of songs) lyricsSection += `--- ${s.title} ---\n${s.lyrics}\n\n`;

  return header + lyricsSection;
}

export function buildSubjectAnalysisPrompt(songs: Array<{ title: string; lyrics: string }>): string {
  const songList = songs.map(s => `--- ${s.title} ---\n${s.lyrics.substring(0, 500)}`).join('\n\n');
  return `Analyse the subjects of these ${songs.length} songs:\n\n${songList}`;
}

// ── YuE2 planner caption ────────────────────────────────────────────────────
//
// The YuE2 planner is prompted with ONE sentence in a fixed order — language,
// genre, vocal, instruments, mood, production, BPM — because that is the order
// its training captions take (becausereasons' CNZN model card, September 2026:
// "This order matches planner training data. Tag lists produce odd plans.").
// Our own dataset captions for YuE2 (`<stem>.yue2.txt`) follow the same shape,
// so a song that carries one of these no longer has to borrow a training
// track's caption to stay in distribution.
//
// It is a THIRD caption, not a trimming of the ACE one: the ACE caption is
// 2-9 sentences with no fixed order and no BPM; the MM3 one is a thirteen-field
// structured block. Neither is what the planner saw.

export const YUE2_CAPTION_ORDER: ReadonlyArray<string> = [
  'language', 'genre', 'vocal', 'instruments', 'mood', 'production', 'BPM',
];

export const YUE2_CAPTION_SYSTEM_PROMPT = `You write the style caption for the YuE2 music planner. The planner reads ONE descriptive sentence and writes the song's lead sheet from it, so the caption decides genre, voice, arrangement and tempo.

Return exactly ONE sentence, plain text, no line breaks, no quotes, no label, nothing before or after it. Build it in THIS order, each part a short comma-separated phrase, and keep the order even when a part is brief:

  1. language      — the language the vocal is sung in ("English", "Italian"). For an instrumental write "instrumental" here and skip the vocal part.
  2. genre         — the specific style, with era words where they help ("early 90s pop punk", "classic Sanremo ballad", "dark synth-pop"). Never a bare umbrella like "rock" or "pop".
  3. vocal         — register, gender and delivery of the lead voice ("nasal male tenor lead vocal with gang-vocal shouts"), or what carries the lead line if instrumental.
  4. instruments   — the instruments actually present, named concretely ("distorted power-chord guitars, driving eighth-note bass, punchy live drums").
  5. mood          — two to four plain words ("restless, sarcastic and buoyant").
  6. production    — the mix and era character ("tight dry mid-90s rock mix with little reverb").
  7. BPM           — the number followed by " BPM" ("168 BPM"). This is the ONLY place a number appears.

Rules:
- One sentence. Roughly 35-70 words. Every part present, in order.
- Concrete nouns, not review copy: "LinnDrum", "gated snare", "arpeggiated synth bass" — never "lush soundscapes" or "keeps you moving".
- Do not name the artist, the band, the song title, the key, or the time signature. Do not quote or summarise the lyrics.
- The genre must agree with the evidence you are given; do not collapse it to an umbrella term.
- Output the sentence and NOTHING else.`;

export interface Yue2CaptionContext {
  /** The ACE-Step caption for the same song — evidence of the intended sound. */
  aceCaption?: string;
  subject?: string;
  bpm?: number;
  key?: string;
  lyrics: string;
  instrumental?: boolean;
  /** Language the lyrics are in, when known ("en"/"English"). */
  language?: string;
}

/** Everything the YuE2 caption call knows about the song. The album's own
 *  YuE2 captions, when the dataset has them, are the strongest evidence of the
 *  dialect: they are literally what the adapter trained on. */
export function buildYue2CaptionPrompt(profile: PromptProfile, ctx: Yue2CaptionContext): string {
  const enrich = profile.audio_enrichment as AlbumEnrichment | undefined;
  const lines: string[] = [];
  lines.push(`Artist style: ${profile.artist_name ?? profile.artist ?? ''}${profile.album ? ` — ${profile.album}` : ''}`.trim(), '');
  if (enrich?.yue2CaptionExamples?.length) {
    lines.push(
      "HOUSE DIALECT — YuE2 captions written for this album's own recordings. Match their order, density and vocabulary; do not copy one verbatim:",
      ...enrich.yue2CaptionExamples.slice(0, 3).map((c, i) => `  ${i + 1}. ${c}`),
      '',
    );
  }
  if (ctx.aceCaption) {
    lines.push(
      'EVIDENCE — how this song is meant to sound (an ACE-Step caption for the same track; source material, not a template):',
      `  "${ctx.aceCaption}"`,
      '',
    );
  } else if (enrich?.captionExamples?.length) {
    lines.push(
      "EVIDENCE — captions describing this album's actual recordings:",
      ...enrich.captionExamples.slice(0, 2).map((c, i) => `  ${i + 1}. "${c}"`),
      '',
    );
  }
  if (enrich?.genres?.length) lines.push(`Measured genres on this album: ${enrich.genres.slice(0, 4).join(', ')}`);
  if (profile.tone_and_mood) lines.push(`Tone & mood of this artist: ${profile.tone_and_mood}`);
  if (ctx.subject) lines.push(`What this song is about (context only — never state it): ${ctx.subject}`);
  lines.push('');
  lines.push(`Language: ${ctx.instrumental ? 'instrumental (no vocal)' : (ctx.language || 'English')}`);
  if (ctx.bpm) lines.push(`BPM: ${Math.round(ctx.bpm)} — end the sentence with exactly "${Math.round(ctx.bpm)} BPM".`);
  if (ctx.instrumental) lines.push('This track is INSTRUMENTAL: write "instrumental" as the language part and name the lead instrument in the vocal part.');
  lines.push('');
  const tags = extractSectionTags(ctx.lyrics);
  if (tags.length) lines.push(`Section structure (evidence of the arrangement only): ${tags.join(' ')}`, '');
  lines.push('Write the one-sentence YuE2 caption now.');
  return lines.join('\n');
}

/** Strip the things a model adds anyway (quotes, labels, fences, line breaks)
 *  and rebuild the BPM tail from the number we hold exactly. */
export function normalizeYue2Caption(raw: string, facts: { bpm?: number } = {}): string {
  let text = String(raw ?? '');
  const fence = text.match(/```(?:[a-z]*)\n([\s\S]*?)```/i);
  if (fence) text = fence[1];
  text = text
    .replace(/^\s*(caption|yue2 caption|style)\s*:\s*/i, '')
    .replace(/[\r\n]+/g, ' ')
    .replace(/\*\*/g, '')
    .replace(/^["'“”‘’\s]+|["'“”‘’\s]+$/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (facts.bpm && facts.bpm > 0) {
    const bpm = Math.round(facts.bpm);
    text = text.replace(/,?\s*(?:at\s+|around\s+|~\s*)?\d{2,3}\s*bpm\.?\s*$/i, '').replace(/[.,;\s]+$/, '');
    text = `${text}, ${bpm} BPM`;
  }
  return text;
}

export function validateYue2Caption(caption: string): string[] {
  const issues: string[] = [];
  const text = String(caption ?? '').trim();
  if (!text) return ['empty'];
  if (/[\r\n]/.test(text)) issues.push('contains a line break (must be one sentence)');
  const words = text.split(/\s+/).filter(Boolean).length;
  if (words < 20) issues.push(`too short (${words} words; expect roughly 35-70)`);
  if (words > 110) issues.push(`too long (${words} words; expect roughly 35-70)`);
  if (!/\d{2,3}\s*bpm\s*\.?$/i.test(text)) issues.push('must END with the tempo as "<N> BPM"');
  if ((text.match(/\d{2,3}\s*bpm/gi) ?? []).length > 1) issues.push('states BPM more than once');
  if (/^#{1,6}\s/m.test(text) || text.includes('**')) issues.push('contains markdown');
  return issues;
}
