// mm3Compose.ts — deterministic MiniMax-Music3 Structured Caption composer
//
// Plain-English brief in, MM3-native Structured Caption out, with NO language
// model anywhere in the path.
//
// WHY THIS IS NOT AN LLM CALL
// MM3 reads the caption's PROSE, not its genre label. A caption whose Basic
// Attributes said "Hardcore Punk." rendered as southern rock across every seed,
// because its body used southern-rock vocabulary ("live-room", "baritone",
// "close-miked") that a model had invented. Selecting real prose from the
// target genre's own reference captions makes that failure structurally
// impossible: the composer can only emit vocabulary the genre's own templates
// contain.
//
// PIPELINE
//   1. route()        brief -> one of 18 families, via TF-IDF over MiniMax's
//                     index cards plus their documented modifier/fallback rules
//   2. parseBrief()   pull bpm / key / scale / gender out of the user's prose
//   3. resolveSlots() precedence: explicit UI control > brief prose > corpus
//   4. compose()      pick each of 11 columns from the family neighbourhood
//                     under section / gender / tempo compatibility filters
//   5. renderCaption() emit the 3 headings + 13 labels, one label per line
//
// WHAT THE CAPTION CANNOT CARRY (measured across all 1,000 templates)
//   time signature  26/1000 state a meter, all inside Groove prose and never in
//                   Basic Attributes; they are 4/4 x24, 3/4 x1, 6/8 x1. MM3 also
//                   has no wire slot for it (no hits in engine/src/minimax/).
//   language        no template states a lyric language. MM3 has no language
//                   input at all — its tokenizer is a byte-level BPE, so the
//                   language follows the characters of the lyrics.
//   duration        no template references track length. It IS a real wire
//                   param, but a CEILING not a target: max_frames =
//                   min(duration * 25, 9000), and the AR loop can hit EOS early.
// The first two are reported back as dropped rather than silently ignored.

import { getMm3Corpus, type Mm3Card } from './mm3Corpus.js';
import { validateMm3Caption } from './prompts.js';

// ── types ───────────────────────────────────────────────────────────────────

export type VocalGender = 'male' | 'female' | 'duet';

/** The Create-page metadata controls, using their existing unset sentinels. */
export interface Mm3Controls {
  /** 0 = unset. */
  bpm?: number;
  /** '' = unset. "E minor" / "Bb major". */
  keyScale?: string;
  /** '' = unset. Not expressible — reported as dropped. */
  timeSignature?: string;
  /** -1 = unset. Not expressible in the caption; it is a wire param instead. */
  duration?: number;
  /** Not expressible — drives lyric writing, not MM3. */
  vocalLanguage?: string;
  /** '' = unset. */
  vocalGender?: VocalGender | '';
}

export interface Mm3ComposeOptions {
  controls?: Mm3Controls;
  /** Reproducible variation. Same brief + controls + seed => same caption. */
  seed?: number;
  /**
   * Section names the arrangement may reference. Defaults to the universal
   * pop/rock skeleton; a brief asking for a solo/breakdown/drop unlocks it.
   * Lyrics are deliberately NOT an input to this composer.
   */
  sections?: string[];
  /** Size of the family neighbourhood each column is drawn from. */
  neighbourhood?: number;
}

export interface Mm3SlotResolution {
  value: string | number | null;
  source: 'control' | 'brief' | 'corpus-default';
}

export interface Mm3ComposeResult {
  caption: string;
  family: string;
  /** True when no genre evidence was found and the documented fallback fired. */
  fallback: boolean;
  genre: string;
  slots: Record<string, Mm3SlotResolution>;
  /** label -> template slug the prose came from. */
  provenance: Record<string, string>;
  /** Distinct templates the caption drew on. */
  sourceCount: number;
  /** Conflicts, dropped controls, thin-pool borrows — surface these in the UI. */
  notes: string[];
  /** validateMm3Caption output; empty means the caption is in-format. */
  validation: string[];
}

// ── constants ───────────────────────────────────────────────────────────────

/**
 * Genre-router rule 6: these are modifiers, not genre evidence. Without this,
 * "a sad piano ballad" routes on the word "ballad" (which appears in dozens of
 * style names) and "happy song about my dog" routes to Happy Hardcore.
 */
const MODIFIER_WORDS = new Set([
  'ballad', 'emotional', 'epic', 'modern', 'dark', 'cinematic', 'happy', 'sad',
  'song', 'track', 'tune', 'music', 'about', 'with', 'and', 'the', 'for', 'my',
  'that', 'this', 'some', 'like', 'very', 'really', 'make', 'write', 'create',
]);

/** Genre-router: the designated family when no genre evidence is present. */
const FALLBACK_FAMILY = 'general-pop-ballad';

/** Sections a caption may reference when no explicit arrangement is given. */
const DEFAULT_SECTIONS = ['intro', 'verse', 'pre-chorus', 'chorus', 'bridge', 'outro'];

/** Extra sections a brief can unlock by asking for them. */
const UNLOCKABLE: Array<[RegExp, string]> = [
  [/\bsolo\b/i, 'solo'],
  [/\bbreakdown\b/i, 'breakdown'],
  [/\bdrops?\b/i, 'drop'],
  [/\bhooks?\b/i, 'hook'],
  [/\binterlude\b/i, 'interlude'],
  [/\b(finale|climax)\b/i, 'finale'],
  [/\brefrain\b/i, 'refrain'],
  [/\bpost-?chorus\b/i, 'post-chorus'],
];

/** Tempo words -> a representative bpm, used only when no number is given. */
const TEMPO_WORDS: Array<[RegExp, number]> = [
  [/\b(ballad|slow|sparse|tender|lullaby|languid|dirge)\w*\b/i, 70],
  [/\b(mid-?tempo|moderate|steady|laid-?back|relaxed|groovy)\b/i, 100],
  [/\b(up-?tempo|driving|energetic|dance|danceable|bouncy|punchy)\b/i, 128],
  [/\b(fast|frantic|breakneck|high-?octane|blistering|thrash|frenetic|furious)\b/i, 165],
];

const FEMININE = /\b(female|woman|women|girl|girls|she|her|feminine|soprano|alto|mezzo|chanteuse|frontwoman|diva)\b/i;
const MASCULINE = /\b(male|man|men|guy|boy|boys|he|his|him|masculine|tenor|baritone|frontman)\b/i;
const DUET_RE = /\b(duet|male and female|female and male|two singers|his and her|mixed vocals)\b/i;

/** Columns whose prose states or implies the singer's gender. */
const VOCAL_LABELS = ['Vocal Gender & Timbre', 'Vocal Style', 'Harmony/Backing Vocals', 'Vocal FX'];

/** The 11 columns the composer selects. Basic Attributes is synthesised. */
const SELECTED_LABELS = [
  'Global Emotional Progression',
  'Application Scenarios & Imagery',
  'Sonics & Production Profile',
  'Vocal Gender & Timbre',
  'Vocal Style',
  'Harmony/Backing Vocals',
  'Vocal FX',
  'Primary',
  'Secondary',
  'Groove & Foundation Progression',
  'Embellishments, Textures & Spatial FX',
];

/**
 * Below this many same-gender cards in a family, the vocal columns are borrowed
 * from families the neighbourhood declares as secondary routes. MiniMax's corpus
 * is heavily male-skewed in guitar genres — metal-heavy-rock has 2 female cards
 * out of 78, hip-hop-rap 2 of 74 — so without this, every female metal caption
 * would be built from the same two templates.
 */
const THIN_POOL = 6;

// ── deterministic RNG ───────────────────────────────────────────────────────

function mulberry32(a: number): () => number {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── routing ─────────────────────────────────────────────────────────────────

const normalise = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();

interface RouteIndex {
  docs: string[][];
  styleSets: Array<Set<string>>;
  idf: Map<string, number>;
}
let routeIndex: RouteIndex | null = null;

function getRouteIndex(cards: Mm3Card[]): RouteIndex {
  if (routeIndex) return routeIndex;
  // Style is weighted x3: it is the only column that names the genre outright.
  const docs = cards.map((c) =>
    normalise(
      `${c.style} ${c.style} ${c.style} ${c.fields['Vocal Gender & Timbre'] || ''} ${c.fields['Primary'] || ''}`,
    ).split(' '),
  );
  const styleSets = cards.map((c) => new Set(normalise(c.style).split(' ')));
  const df = new Map<string, number>();
  for (const d of docs) for (const w of new Set(d)) df.set(w, (df.get(w) || 0) + 1);
  const idf = new Map<string, number>();
  for (const [w, n] of df) idf.set(w, Math.log(1 + cards.length / (1 + n)));
  routeIndex = { docs, styleSets, idf };
  return routeIndex;
}

interface Scored { score: number; styleHit: number; card: Mm3Card; }

/** Ranks every card against the brief and picks the winning family. */
function route(brief: string, cards: Mm3Card[]): { family: string; fallback: boolean; ranked: Scored[] } {
  const { docs, styleSets, idf } = getRouteIndex(cards);
  const terms = normalise(brief).split(' ').filter((w) => w.length > 2 && !MODIFIER_WORDS.has(w));

  const ranked: Scored[] = cards.map((card, i) => {
    const counts = new Map<string, number>();
    for (const w of docs[i]) counts.set(w, (counts.get(w) || 0) + 1);
    let score = 0;
    let styleHit = 0;
    for (const w of terms) {
      const n = counts.get(w);
      if (n) score += (idf.get(w) || 0) * (1 + Math.log(n));
      if (styleSets[i].has(w)) styleHit++;
    }
    return { score, styleHit, card };
  }).sort((a, b) => b.score - a.score || b.styleHit - a.styleHit);

  // The winner must have matched an actual STYLE word. Matching only mood or
  // instrument words is not genre evidence, and routing on it is how "a sad
  // piano ballad" ends up as Mandopop.
  const winner = ranked.find((r) => r.styleHit > 0);
  if (!winner) return { family: FALLBACK_FAMILY, fallback: true, ranked };
  return { family: winner.card.family, fallback: false, ranked };
}

// ── brief parsing ───────────────────────────────────────────────────────────

interface BriefFacts { bpm?: number; key?: string; scale?: string; gender?: VocalGender; }

/** Extracts bpm / key / scale / gender stated in the user's own prose. */
export function parseBrief(brief: string): BriefFacts {
  const b = ` ${brief} `;
  const out: BriefFacts = {};

  const bpmMatch = b.match(/\b(\d{2,3})\s*bpm\b/i);
  if (bpmMatch) {
    out.bpm = parseInt(bpmMatch[1], 10);
  } else {
    for (const [re, v] of TEMPO_WORDS) if (re.test(b)) { out.bpm = v; break; }
  }

  // Case-sensitive on the note letter: with /i, [#b] also matches the letter B,
  // so "Bb minor" only parses as Bb while the flag is off.
  const keyMatch = b.match(/\b(?:in\s+)?([A-G][#b♯♭]?)\s*[- ]?\s*(major|minor|maj|min|m)\b/);
  if (keyMatch) {
    out.key = keyMatch[1].replace('♯', '#').replace('♭', 'b');
    out.scale = /^(min|m)/i.test(keyMatch[2]) ? 'minor' : 'major';
  }

  const fem = FEMININE.test(b);
  const masc = MASCULINE.test(b);
  if (DUET_RE.test(b) || (fem && masc)) out.gender = 'duet';
  else if (fem) out.gender = 'female';
  else if (masc) out.gender = 'male';

  return out;
}

/** Sections the caption may reference: the default skeleton plus anything asked for. */
function sectionsFor(brief: string, explicit?: string[]): Set<string> {
  if (explicit?.length) {
    return new Set(explicit.map((s) => s.toLowerCase().replace(/^\[|\]$/g, '').trim()));
  }
  const set = new Set(DEFAULT_SECTIONS);
  for (const [re, name] of UNLOCKABLE) if (re.test(brief)) set.add(name);
  return set;
}

// ── slot resolution ─────────────────────────────────────────────────────────

function isSet(v: unknown): boolean {
  return v !== undefined && v !== null && v !== '' && v !== 0 && v !== -1;
}

/**
 * Resolves each caption fact from, in order: an explicitly-set UI control, the
 * user's own prose, then a genre-appropriate default from the routed card.
 *
 * A control still on its unset sentinel is not a user decision, so the brief
 * wins over it. When both are set and disagree the CONTROL wins and the conflict
 * is reported — a silently-ignored typed instruction is worse than a visible one.
 */
function resolveSlots(
  brief: string,
  controls: Mm3Controls,
  topCard: Mm3Card,
): { slots: Record<string, Mm3SlotResolution>; notes: string[]; gender: VocalGender | null } {
  const facts = parseBrief(brief);
  const notes: string[] = [];
  const slots: Record<string, Mm3SlotResolution> = {};

  const pick = (
    name: string,
    control: string | number | undefined,
    fromBrief: string | number | undefined,
    corpusDefault: string | number | null,
  ) => {
    if (isSet(control)) {
      if (fromBrief !== undefined && String(fromBrief) !== String(control)) {
        notes.push(`${name}: the control says ${control} but the prompt says ${fromBrief} — using the control`);
      }
      slots[name] = { value: control as string | number, source: 'control' };
      return;
    }
    if (fromBrief !== undefined) {
      slots[name] = { value: fromBrief, source: 'brief' };
      return;
    }
    slots[name] = { value: corpusDefault, source: 'corpus-default' };
  };

  const ck = (controls.keyScale || '').match(/^([A-G][#b]?)\s+(major|minor)$/i);

  pick('bpm', controls.bpm, facts.bpm, topCard.bpm || 110);
  pick('key', ck ? ck[1] : '', facts.key, topCard.key || 'C');
  pick('scale', ck ? ck[2].toLowerCase() : '', facts.scale, topCard.scale || 'minor');
  pick('gender', controls.vocalGender || '', facts.gender, null);

  // Controls the Structured Caption format cannot express. Reported, not hidden.
  if (controls.timeSignature && controls.timeSignature !== '4/4') {
    notes.push(
      `time signature ${controls.timeSignature} not applied: 26 of 1,000 reference captions state a meter ` +
      `(4/4 x24, 3/4 x1, 6/8 x1) and never in Basic Attributes — MM3 has no wire slot for it either`,
    );
  }
  if (controls.vocalLanguage && controls.vocalLanguage !== 'en') {
    notes.push(
      `language "${controls.vocalLanguage}" not applied: no reference caption states a lyric language, and MM3 ` +
      `has no language input — it follows the characters of the lyrics`,
    );
  }

  const gender = (slots.gender.value as VocalGender | null) || null;
  return { slots, notes, gender };
}

// ── composition ─────────────────────────────────────────────────────────────

/**
 * Builds the candidate pool for one column: cards from the family neighbourhood
 * whose prose is compatible with the requested arrangement, singer and tempo.
 */
function candidatesFor(
  label: string,
  pool: Scored[],
  allowedSections: Set<string>,
  gender: VocalGender | null,
  bpm: number,
): Scored[] {
  return pool.filter(({ card }) => {
    if (!card.fields[label]) return false;

    // The prose must not reference a section this arrangement does not have.
    for (const s of card.sections[label] || []) if (!allowedSections.has(s)) return false;

    if (gender && VOCAL_LABELS.includes(label) && card.gender !== 'unknown' && card.gender !== gender) {
      return false;
    }

    // Groove describes tempo in prose 48% of the time, so a card from a very
    // different tempo contradicts the bpm printed in Basic Attributes.
    if (label === 'Groove & Foundation Progression' && bpm > 0 && card.bpm > 0) {
      if (Math.abs(Math.log2(card.bpm / bpm)) > 0.28) return false; // ~±21%
    }
    return true;
  });
}

/** Picks one candidate, favouring genre rank and unused sources, seeded. */
function choose(cands: Scored[], used: Map<string, number>, rnd: () => number): Scored {
  let best = cands[0];
  let bestW = -Infinity;
  cands.forEach((c, i) => {
    // Gentle rank decay so the seed can reorder near-equal candidates; a steeper
    // curve makes rank 0 always win and the seed do nothing.
    const w = Math.pow(0.94, i) * (used.has(c.card.id) ? 0.55 : 1) * (0.55 + 0.9 * rnd());
    if (w > bestW) { bestW = w; best = c; }
  });
  return best;
}

function renderCaption(f: Record<string, string>): string {
  return [
    'Global Metadata',
    `Basic Attributes: ${f['Basic Attributes']}`,
    `Global Emotional Progression: ${f['Global Emotional Progression']}`,
    `Application Scenarios & Imagery: ${f['Application Scenarios & Imagery']}`,
    `Sonics & Production Profile: ${f['Sonics & Production Profile']}`,
    'Vocal Details',
    `Vocal Gender & Timbre: ${f['Vocal Gender & Timbre']}`,
    `Vocal Style: ${f['Vocal Style']}`,
    `Harmony/Backing Vocals: ${f['Harmony/Backing Vocals']}`,
    `Vocal FX: ${f['Vocal FX']}`,
    'Arrangement',
    'Instrument Lifecycle Description (Primary/Secondary Layering):',
    `Primary: ${f['Primary']}`,
    `Secondary: ${f['Secondary']}`,
    `Groove & Foundation Progression: ${f['Groove & Foundation Progression']}`,
    `Embellishments, Textures & Spatial FX: ${f['Embellishments, Textures & Spatial FX']}`,
  ].join('\n');
}

/**
 * Composes a Structured Caption for a plain-English brief. No LLM, no network,
 * no lyrics. Same brief + controls + seed always yields the same caption.
 */
export function composeMm3Caption(brief: string, opts: Mm3ComposeOptions = {}): Mm3ComposeResult {
  const { cards } = getMm3Corpus();
  const controls = opts.controls ?? {};
  const K = opts.neighbourhood ?? 24;
  const rnd = mulberry32(((opts.seed ?? 1) >>> 0) || 1);

  const routed = route(brief, cards);
  const inFamily = routed.ranked.filter((r) => r.card.family === routed.family);
  if (!inFamily.length) throw new Error(`no reference captions for family ${routed.family}`);

  const { slots, notes, gender } = resolveSlots(brief, controls, inFamily[0].card);
  const allowedSections = sectionsFor(brief, opts.sections);
  const bpm = Number(slots.bpm.value) || 0;

  let pool = inFamily.slice(0, K);

  // A top-K neighbourhood rarely holds K cards of the requested gender, so widen
  // within the family FIRST — that is not a thin corpus, just a narrow slice.
  // Only when the family itself is short of that gender do we leave it.
  if (gender) {
    const chosen = new Set(pool.map((p) => p.card.id));
    if (pool.filter((p) => p.card.gender === gender).length < THIN_POOL) {
      const moreInFamily = inFamily
        .filter((r) => r.card.gender === gender && !chosen.has(r.card.id))
        .slice(0, K);
      pool = pool.concat(moreInFamily);
      moreInFamily.forEach((r) => chosen.add(r.card.id));
    }

    const familyTotal = inFamily.filter((r) => r.card.gender === gender).length;
    if (familyTotal < THIN_POOL) {
      const secondaries = new Set(inFamily.map((p) => p.card.secondary).filter(Boolean));
      const borrowed = routed.ranked
        .filter((r) => secondaries.has(r.card.family) && r.card.gender === gender && !chosen.has(r.card.id))
        .slice(0, K);
      if (borrowed.length) {
        pool = pool.concat(borrowed);
        notes.push(
          `${routed.family} has only ${familyTotal} ${gender} reference caption(s) — borrowed ` +
          `${borrowed.length} from its declared secondary route(s) so the vocal columns can vary`,
        );
      }
    }
  }

  const fields: Record<string, string> = {};
  const provenance: Record<string, string> = {};
  const used = new Map<string, number>();

  for (const label of SELECTED_LABELS) {
    let cands = candidatesFor(label, pool, allowedSections, gender, bpm);
    if (!cands.length) {
      // Relax rather than fail: the format demands every column be emitted.
      cands = pool.filter((p) => p.card.fields[label]);
      if (!cands.length) cands = routed.ranked.filter((r) => r.card.fields[label]).slice(0, K);
    }
    const chosen = choose(cands, used, rnd);
    fields[label] = chosen.card.fields[label];
    provenance[label] = chosen.card.id;
    used.set(chosen.card.id, (used.get(chosen.card.id) || 0) + 1);
  }

  const genre = inFamily[0].card.style;
  fields['Basic Attributes'] =
    `bpm is ${bpm}. key is ${slots.key.value}, and scale is ${slots.scale.value}. ${genre}.`;
  provenance['Basic Attributes'] = '(synthesised from the resolved facts + routed genre)';

  const caption = renderCaption(fields);

  return {
    caption,
    family: routed.family,
    fallback: routed.fallback,
    genre,
    slots,
    provenance,
    sourceCount: used.size,
    notes,
    validation: validateMm3Caption(caption),
  };
}
