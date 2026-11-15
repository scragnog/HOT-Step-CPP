// The arrangement contract for build by section (yue2Takes.ts). A build used
// to compose the score's sections in source order; an arrangement lets the
// output skip, reorder and repeat them. Nothing imports this file yet: the
// session work that adopts it (cover-local lyrics, then arrangement support)
// follows these rules rather than inventing its own.
//
// Why the composed prompt is enough (engine/src/yue2/yue2-pipeline.h): every
// /yue2/synth request tokenizes style + lyrics + abc into one prompt and
// prefills the cache from scratch (yue2_run_semantic_stage), then
// teacher-forces codec_ids and samples on (continue_codec_ids). Nothing reads
// sections apart from the score text itself, and the NAR chunks copy the same
// prompt rows (yue2_seal_chunks). So a take of a reordered song is an ordinary
// take whose abc and lyrics are written out in output order. Three things
// follow:
//  - The prompt for occurrences already kept must not change, so their text
//    is a function of the kept prefix only (composeArrangement guarantees it).
//    Later text may change between takes; the engine re-prefills the prefix
//    codes under the new prompt and checks they come back verbatim.
//  - Save song must send the same composed abc and lyrics. Its NAR reads the
//    whole stream under the request's prompt, and generate.ts builds that
//    prompt from the job params, which hold the source score.
//  - A lyric schedule (C6) is built from the score's text, so it must be
//    rebuilt from the composed text, and a resumed take replays its prefix
//    under it.
//
// Identity, in the session file:
//  - Source sections are the score's own, in order: ids s0, s1, ... The score
//    never changes within a session (a new score is a new session).
//  - Occurrences are the output, in order: ids o0, o1, ... from a counter that
//    never reuses one, so two choruses, or a chorus removed and added back,
//    are different occurrences.
//  - A take's from/to are occurrence indexes in the plan, and its starts are
//    indexed by occurrence. For a legacy session the plan is the identity,
//    so its from/to/starts already mean that.
//  - `revision` counts changes to what the next take would be sent (plan,
//    lyrics, settings). Every take records the revision and the parent (the
//    spine it resumed) it was made from. A mutation names the revision it
//    expects and fails on a mismatch; a take is kept only if it matches the
//    current revision and spine and composed the next planned occurrences.
//  - Keeping appends; undo pops. Neither changes the revision or the plan, so
//    a take made from a spine is keepable again after undoing back to it.

import { splitLyricBlocks } from './sectionMatch.js';
import { sectionKind } from './coverDrift.js';
import { scoreSections } from './scoreSections.js';
import { barsIn, hasSoundingVocalBar, isScoreVocalVoice, scoreBarSegments, vocalVoiceIds } from './scoreHealth.js';
import type { Yue2Take } from '../../yue2Takes.js';

export const BUILD_SESSION_VERSION = 2;

export interface SourceSection {
  id: string;
  label: string;
  startBar: number;
  bars: number;
  /** Index of its lyric block among splitLyricBlocks' non-empty blocks. */
  lyricBlock: number | null;
}

/** `lyrics`, when set, is this occurrence's own sung text (the block's lines,
 *  without its tag), so a repeated chorus can sing other words than the first.
 *  An improv occurrence has section IMPROV and its own bar count instead of a
 *  source section: the score gets that many bars of rests, labelled Improv,
 *  and its take composes them without the score (yue2Takes' blind window).
 *  `octave`, when set (-1 or 1), moves the occurrence's vocal line that many
 *  octaves in the composed score, so one chorus can be sung higher than the last. */
export interface Occurrence { id: string; section: string; lyrics?: string; bars?: number; octave?: number }

export const IMPROV = 'improv';
export const IMPROV_BARS = [2, 4, 8, 16];

/** A plan entry for composeArrangement: a source section index, or improv bars. */
export type PlanEntry = number | { improv: number };

export interface Arrangement {
  sections: SourceSection[];
  plan: Occurrence[];
  nextOccurrence: number;
  /** The plan ends the song. Until then a take's prompt runs on into the
   *  source section after the last planned one (lookaheadSection), so the
   *  model composes a seam there instead of an ending. */
  finished: boolean;
}

export interface TakeIdentity {
  revision: number;
  /** The spine it resumed; null when it starts the song (or was orphaned
   *  before migration, then only from 0 is keepable). */
  parent: string | null;
  /** Occurrence ids it composed, from..to. */
  occurrences: string[];
}

export type ArrangedTake = Yue2Take & TakeIdentity;

/** The fields a version 2 session adds or narrows. */
export interface ArrangedSession {
  version: typeof BUILD_SESSION_VERSION;
  revision: number;
  arrangement: Arrangement;
  takes: ArrangedTake[];
  kept: string[];
}

const FIELD = /^([A-Za-z]):/;
const label = (line: string) => (line.startsWith('%') ? line.slice(1).trim() : '');

/** Non-empty lyric blocks, as yue2Takes pairs them. */
const sungBlocks = (lyrics: string) => splitLyricBlocks(lyrics).filter(block => block.body.trim());

/** Each section's lyric block: the next unused tag of its kind, as a sung
 *  block index, or null when that tag is empty or there is none. Empty tags
 *  count, so of two intros the first takes Prepare's empty [Intro] and the
 *  second its words, not the first both. */
function pairSections(labels: string[], lyrics: string): Array<number | null> {
  const blocks = splitLyricBlocks(lyrics);
  const sung = blocks.filter(block => block.body.trim());
  let next = 0;
  return labels.map(name => {
    const at = blocks.findIndex((block, i) => i >= next && !!block.tag && sectionKind(block.tag) === sectionKind(name));
    if (at < 0) return null;
    next = at + 1;
    return blocks[at].body.trim() ? sung.indexOf(blocks[at]) : null;
  });
}

export function sourceSections(abc: string, lyrics: string): SourceSection[] {
  const found = scoreSections(abc);
  // A trailing marker reads the total bar count the way scoreSections counts.
  const total = scoreSections(`${abc}\n% end`).at(-1)!.startBar - 1;
  const blocks = pairSections(found.map(s => s.label), lyrics);
  return found.map((s, i) => ({
    id: `s${i}`, label: s.label.trim(), startBar: s.startBar,
    bars: (found[i + 1]?.startBar ?? total + 1) - s.startBar, lyricBlock: blocks[i],
  }));
}

export interface ComposedOccurrence {
  section: number;
  startBar: number;
  bars: number;
  /** Code-point offset of its first line in the composed abc. */
  abcStart: number;
  /** Code-point span of its block in the composed lyrics, or null (unsung). */
  lyric: [number, number] | null;
}

export interface Composed {
  abc: string;
  lyrics: string;
  occurrences: ComposedOccurrence[];
  /** Source lyric blocks no occurrence sings. */
  unplacedBlocks: number[];
  /** Whether pairing the composed text again (yue2Takes' sectionStarts and
   *  lyricsThrough, the lyric schedule) gives each occurrence its own block.
   *  When false, use `occurrences` instead; an unsung section placed before a
   *  sung one of the same kind would take its lyrics. */
  textPairingMatches: boolean;
}

/** Writes the score and lyrics out in the order of `plan` (source section
 *  indexes; repeats allowed). The source order gives the source text back
 *  (lyrics gain a final newline if they lack one). `sing[k]`, when set, is
 *  sung by occurrence k in place of its block's lines; an unsung section has
 *  none to replace. `octaves[k]`, when set, moves occurrence k's vocal lines
 *  that many octaves (shiftOctave). */
export function composeArrangement(abc: string, lyrics: string, plan: PlanEntry[], sing: Array<string | undefined> = [],
  octaves: Array<number | undefined> = []): Composed {
  abc = abc.replace(/\r\n?/g, '\n');
  lyrics = lyrics.replace(/\r\n?/g, '\n');
  const lines = abc.split('\n');
  const starts = lines.flatMap((line, i) => (label(line.trim()) ? [i] : []));
  if (!starts.length) throw new Error('The score has no section labels.');
  if (plan.some(s => (typeof s === 'number' ? !Number.isInteger(s) || s < 0 || s >= starts.length : !IMPROV_BARS.includes(s.improv)))) {
    throw new Error('No such section in the plan.');
  }
  // Every voice the score uses, for an improv's rest bars.
  const voiceIds = [...new Set(lines.flatMap(line => {
    const f = FIELD.exec(line.trim());
    return f?.[1] === 'V' ? [line.trim().slice(2).trim().split(/\s+/)[0] ?? ''] : [];
  }))].filter(Boolean);
  // The header ends at its K: line; music between it and the first label
  // travels with the first section.
  const k = lines.slice(0, starts[0]).findIndex(line => /^K:/.test(line.trim()));
  const headerEnd = k < 0 ? starts[0] : k + 1;
  // A trailing newline belongs to the file, not the last section.
  const trailer = lines.at(-1) === '' ? [lines.pop()!] : [];
  const chunks = starts.map((at, i) => lines.slice(i ? at : headerEnd, starts[i + 1] ?? lines.length));

  // The M:/L:/Q:/K: fields and voice in force where each section begins.
  type State = { fields: Record<string, string>; voice: string | null };
  const header: State = { fields: {}, voice: null };
  for (const line of lines.slice(0, headerEnd)) {
    const f = FIELD.exec(line.trim());
    if (f && 'MLQK'.includes(f[1])) header.fields[f[1]] = line.trim();
  }
  const walk = (state: State, chunk: string[]): State => {
    const next = { fields: { ...state.fields }, voice: state.voice };
    for (const raw of chunk) {
      const line = raw.trim(), f = FIELD.exec(line);
      if (f?.[1] === 'V') next.voice = line.slice(2).trim().split(/\s+/)[0] ?? null;
      else if (f && 'MLQK'.includes(f[1])) next.fields[f[1]] = line;
    }
    return next;
  };
  const entry: State[] = [];
  for (let i = 0; i < chunks.length; i++) entry.push(i ? walk(entry[i - 1], chunks[i - 1]) : header);

  const out = lines.slice(0, headerEnd);
  const blocks = sungBlocks(lyrics);
  const chars = Array.from(lyrics);
  const labels = starts.map(at => label(lines[at].trim()));
  const pairs = pairSections(labels, lyrics);
  // Each section owns a verbatim slice of the lyrics, from its own block (its
  // sung one, else an empty tag of its kind between its neighbours' blocks)
  // to the next owned block. Text before that belongs to the first section.
  // Every plan, the source order included, is a concatenation of the same
  // slices, so kept text and offsets never depend on what follows.
  const anchor = pairs.map(at => (at === null ? null : blocks[at].start));
  const all = splitLyricBlocks(lyrics);
  for (let i = 0; i < anchor.length; i++) {
    if (anchor[i] !== null) continue;
    const lo = anchor.slice(0, i).reduce<number>((last, a) => a ?? last, -1);
    const hi = anchor.slice(i + 1).find(a => a !== null) ?? Infinity;
    anchor[i] = all.find(b => b.tag && !b.body.trim() && b.start > lo && b.start < hi
      && sectionKind(b.tag) === sectionKind(labels[i]))?.start ?? null;
  }
  const cuts = anchor.map((a, i) => (i === 0 ? 0 : a ?? anchor.slice(i + 1).find(n => n !== null) ?? chars.length));
  cuts.push(chars.length);
  const slices = cuts.slice(0, -1).map((c, i) => chars.slice(c, cuts[i + 1]).join(''));
  // The last slice may follow another in a new order, so it ends its line.
  const tail = slices.findLastIndex(slice => slice);
  if (tail >= 0 && !slices[tail].endsWith('\n')) slices[tail] += '\n';
  const lyricParts: string[] = [];
  const occurrences: ComposedOccurrence[] = [];
  let state = header, bar = 1, lyricAt = 0;
  const sections = sourceSections(abc, lyrics);
  const vocalIds = vocalVoiceIds(abc);
  // An unvoiced score is one melody line: the vocal.
  const sung = (voice: string | null) => (voice === null ? !voiceIds.length : isScoreVocalVoice(`V: ${voice}`, vocalIds));
  for (const [k, s] of plan.entries()) {
    if (typeof s !== 'number') {
      const rests = `${restBar(state.fields)}|`.repeat(s.improv);
      const chunk = ['% Improv', ...(voiceIds.length ? voiceIds.flatMap(v => [`V: ${v}`, rests]) : [rests])];
      occurrences.push({ section: -1, startBar: bar, bars: s.improv, abcStart: Array.from(out.join('\n')).length + (out.length ? 1 : 0), lyric: null });
      out.push(...chunk);
      state = walk(state, chunk);
      bar += s.improv;
      continue;
    }
    const want = entry[s];
    // ponytail: restates changed fields in every voice the section uses, which
    // is how the transcriber writes meter changes; a voice absent from the
    // section keeps its own stale meter.
    const changed = Object.entries(want.fields).filter(([f, v]) => state.fields[f] !== v).map(([, v]) => v);
    const voices = new Set<string>();
    const abcStart = Array.from(out.join('\n')).length + (out.length ? 1 : 0);
    let voiced = false, voice = want.voice;
    for (const raw of chunks[s]) {
      const line = raw.trim(), f = FIELD.exec(line);
      const music = !!line && !line.startsWith('%') && !f;
      if (music && !voiced && want.voice && (state.voice !== want.voice || changed.length)) {
        out.push(`V: ${want.voice}`, ...changed);
        voices.add(want.voice);
      }
      out.push(music && octaves[k] && sung(voice) ? shiftOctave(raw, octaves[k]!) : raw);
      if (f?.[1] === 'V') {
        const id = line.slice(2).trim().split(/\s+/)[0] ?? '';
        voice = id;
        if (!voices.has(id)) { voices.add(id); out.push(...changed); }
      }
      voiced ||= music || f?.[1] === 'V';
    }
    if (!voices.size && changed.length) out.push(...changed);
    state = walk(want, chunks[s]);
    const at = pairs[s];
    let slice = slices[s];
    let lyric: [number, number] | null = at === null ? null
      : [lyricAt + blocks[at].start - cuts[s], lyricAt + Math.min(blocks[at].end, cuts[s + 1]) - cuts[s]];
    if (at !== null && sing[k] !== undefined) {
      // The block's tag line and trailing blank lines stay; its lines change.
      const end = Math.min(blocks[at].end, cuts[s + 1]);
      const block = chars.slice(blocks[at].start, end).join('');
      const tag = block.slice(0, block.indexOf('\n') + 1 || block.length);
      const text = `${tag}${tag.endsWith('\n') ? '' : '\n'}${sing[k]!.trim()}${/\s*$/.exec(block)![0] || '\n'}`;
      const pre = chars.slice(cuts[s], blocks[at].start).join('');
      slice = pre + text + Array.from(slice).slice(end - cuts[s]).join('');
      const from = lyricAt + Array.from(pre).length;
      lyric = [from, from + Array.from(text).length];
    }
    lyricParts.push(slice);
    lyricAt += Array.from(slice).length;
    occurrences.push({ section: s, startBar: bar, bars: sections[s].bars, abcStart, lyric });
    bar += sections[s].bars;
  }
  out.push(...trailer);
  const composedLyrics = lyricParts.join('');
  // A sung block is placed when the section owning its slice is.
  const owner = (start: number) => cuts.findLastIndex((c, i) => i < slices.length && c <= start);
  const used = new Set(blocks.flatMap((b, i) => (plan.includes(owner(b.start)) ? [i] : [])));
  const composedAbc = out.join('\n');
  const repaired = pairSections(scoreSections(composedAbc).map(s => s.label), composedLyrics);
  const own = occurrences.map(o => (o.lyric ? sungBlocks(composedLyrics).findIndex(b => b.start === o.lyric![0]) : null));
  return {
    abc: composedAbc, lyrics: composedLyrics, occurrences,
    unplacedBlocks: blocks.flatMap((_, i) => (used.has(i) ? [] : [i])),
    textPairingMatches: repaired.length === own.length && repaired.every((b, i) => b === own[i]),
  };
}

// Skipped whole: comments, chord symbols and annotations, inline fields, decorations.
const NOTE = /%.*$|"[^"]*"|\[[A-Za-z]:[^\]]*\]|!.*?!|([A-Ga-g])([,']*)/g;

/** One music line with every note moved `octaves` octaves (C -> c -> c' up,
 *  C -> C, down). Accidentals stay as written: a whole line moves together,
 *  so what a bar's accidental carries to moves with it. */
export function shiftOctave(line: string, octaves: number): string {
  return line.replace(NOTE, (token, letter: string | undefined, marks: string) => {
    if (!letter) return token;
    const o = (letter === letter.toLowerCase() ? 1 : 0) + [...marks].reduce((n, m) => n + (m === "'" ? 1 : -1), 0) + octaves;
    return o >= 1 ? letter.toLowerCase() + "'".repeat(o - 1) : letter.toUpperCase() + ','.repeat(-o);
  });
}

/** One bar of rest in the meter in force (M:, L:; ABC's defaults when absent). */
function restBar(fields: Record<string, string>): string {
  const ratio = (text: string | undefined, fallback: [number, number]): [number, number] => {
    const m = /(\d+)\s*\/\s*(\d+)/.exec(text ?? '');
    return m ? [Number(m[1]), Number(m[2])] : fallback;
  };
  const m = fields.M?.slice(2).trim();
  const meter = m === 'C' ? [4, 4] : m === 'C|' ? [2, 2] : ratio(m, [4, 4]);
  const unit = ratio(fields.L?.slice(2), meter[0] / meter[1] < 0.75 ? [1, 16] : [1, 8]);
  let num = meter[0] * unit[1], den = meter[1] * unit[0];
  const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);
  const g = gcd(num, den);
  num /= g; den /= g;
  return den === 1 ? (num === 1 ? 'z' : `z${num}`) : `z${num}/${den}`;
}

/** The plan's entries for composeArrangement: source section indexes and improvs. */
export function planSections(arrangement: Arrangement): PlanEntry[] {
  return arrangement.plan.map(o => (o.section === IMPROV ? { improv: o.bars ?? 0 }
    : arrangement.sections.findIndex(s => s.id === o.section)));
}

/** Occurrence i's name: its source section's label, or Improv. */
export function occurrenceLabel(arrangement: Arrangement, i: number): string | undefined {
  const o = arrangement.plan[i];
  return o?.section === IMPROV ? 'Improv' : arrangement.sections.find(s => s.id === o?.section)?.label;
}

/** The plan's own lyrics per occurrence, for composeArrangement. */
export const planLyrics = (arrangement: Arrangement): Array<string | undefined> => arrangement.plan.map(o => o.lyrics);

/** The plan's vocal octave shift per occurrence, for composeArrangement. */
export const planOctaves = (arrangement: Arrangement): Array<number | undefined> => arrangement.plan.map(o => o.octave);

/** The source section a take's prompt adds after the last planned occurrence
 *  while the plan is open: the one that follows it in the score. The take
 *  stops a few seconds into it, so the seam has something to run into
 *  without the user choosing what comes next. None after the score's last
 *  section (that take composes an ending) or once the plan is finished. */
export function lookaheadSection(arrangement: Arrangement): number | null {
  if (arrangement.finished || !arrangement.plan.length) return null;
  // After an improv, the source runs on from the last source section before it.
  const last = planSections(arrangement).filter((s): s is number => typeof s === 'number').at(-1) ?? -1;
  return last + 1 < arrangement.sections.length ? last + 1 : null;
}

/** Plan index of the next occurrence to build: the one after the spine's last. */
export function nextOccurrence(session: Pick<ArrangedSession, 'takes' | 'kept'>): number {
  const spine = session.takes.find(t => t.id === session.kept.at(-1));
  return spine ? spine.to + 1 : 0;
}

/** Why takes of occurrences [next, to] cannot be generated now, or null. */
export function generateRefusal(session: ArrangedSession, to: number): string | null {
  const from = nextOccurrence(session), { plan } = session.arrangement;
  if (!Number.isInteger(to) || to < from || to >= plan.length) return 'Pick an occurrence from the next one to build onward.';
  return null;
}

/** The identity a take queued now records. */
export function takeIdentity(session: ArrangedSession, to: number): TakeIdentity {
  const from = nextOccurrence(session);
  return { revision: session.revision, parent: session.kept.at(-1) ?? null,
    occurrences: session.arrangement.plan.slice(from, to + 1).map(o => o.id) };
}

/** Why `take` cannot be kept now, or null. */
export function keepRefusal(session: ArrangedSession, take: ArrangedTake): string | null {
  const from = nextOccurrence(session);
  if (take.parent !== (session.kept.at(-1) ?? null) || take.from !== from) return 'That take does not follow the kept build; undo back to it first.';
  if (take.revision !== session.revision) return 'That take was made before the last change to the build.';
  const want = session.arrangement.plan.slice(from, take.to + 1).map(o => o.id);
  if (want.length !== take.occurrences.length || want.some((id, i) => id !== take.occurrences[i])) {
    return 'That take composed a different arrangement.';
  }
  return null;
}

/** Why a mutation sent against `expected` must be refused, or null. */
export function revisionRefusal(session: ArrangedSession, expected: unknown): string | null {
  return expected === session.revision ? null : 'The build changed since this page loaded; reload it and try again.';
}

/** Replaces the unbuilt part of the plan. `upcoming` lists source section ids,
 *  or existing occurrence ids to keep their identity (reorder, remove). Kept
 *  occurrences cannot change. The caller bumps the revision and refuses the
 *  edit while takes are pending. */
export function editPlan(session: ArrangedSession, upcoming: string[], finished: boolean): Arrangement {
  const { arrangement } = session;
  const built = nextOccurrence(session);
  const kept = arrangement.plan.slice(0, built);
  const spare = new Map(arrangement.plan.slice(built).map(o => [o.id, o]));
  let counter = arrangement.nextOccurrence;
  const plan = kept.concat(upcoming.map(id => {
    const existing = spare.get(id);
    if (existing) { spare.delete(id); return existing; }
    const improv = /^improv:(\d+)$/.exec(id);
    if (improv) {
      const bars = Number(improv[1]);
      if (!IMPROV_BARS.includes(bars)) throw new Error(`An improv is ${IMPROV_BARS.join(', ')} bars long.`);
      return { id: `o${counter++}`, section: IMPROV, bars };
    }
    if (!arrangement.sections.some(s => s.id === id)) throw new Error(`No such section or upcoming occurrence: ${id}`);
    return { id: `o${counter++}`, section: id };
  }));
  if (!plan.length) throw new Error('The arrangement needs at least one section.');
  // A kept take that composed an ending locks what follows it. One that ran on
  // into a lookahead section (a start recorded past its last occurrence) did
  // not, even if the plan was marked as ending after it was made, so the song
  // can still grow or stop ending there.
  const spine = session.takes.find(t => t.id === session.kept.at(-1));
  const ended = !!spine && spine.starts.length <= spine.to + 1;
  if (spine && ended && arrangement.finished && spine.to === arrangement.plan.length - 1 && (plan.length > built || !finished)) {
    throw new Error('The kept build already ends the song; undo its last keep to change what follows.');
  }
  return { ...arrangement, plan, nextOccurrence: counter, finished };
}

/** A version 1 session file (no version field): sections in source order. */
export interface LegacySession {
  version?: undefined;
  sections: Array<{ label: string; startBar: number }>;
  takes: Yue2Take[];
  kept: string[];
  request: { abc?: string; lyrics?: string };
}

/** Upgrades a legacy session in place of reading it as version 2. The plan is
 *  the source order, finished, at revision 0. Kept takes chain by keep order.
 *  Any other take's parent is the take whose to is just before its from and
 *  whose codes it starts with up to that take's start for it; none such means
 *  orphaned (parent null), so a take after section 0 is never keepable. */
export function migrateLegacySession<T extends LegacySession>(stored: T, codesOf: (takeId: string) => number[]):
  Omit<T, keyof ArrangedSession> & ArrangedSession {
  const abc = stored.request.abc ?? '', lyrics = stored.request.lyrics ?? '';
  const sections = sourceSections(abc, lyrics);
  if (sections.length !== stored.sections.length) throw new Error('Session sections no longer match its score.');
  const plan = sections.map((s, i) => ({ id: `o${i}`, section: s.id }));
  const codes = new Map<string, number[]>();
  const read = (id: string) => codes.get(id) ?? (codes.set(id, codesOf(id)), codes.get(id)!);
  const takes = stored.takes.map((take): ArrangedTake => {
    const keptAt = stored.kept.indexOf(take.id);
    let parent: string | null = null;
    if (keptAt > 0) parent = stored.kept[keptAt - 1];
    else if (keptAt < 0 && take.from > 0) {
      const own = read(take.id);
      const candidates = [...stored.takes].sort((a, b) => Number(!stored.kept.includes(a.id)) - Number(!stored.kept.includes(b.id)));
      parent = candidates.find(p => {
        const cut = p.to === take.from - 1 ? p.starts[take.from] : null;
        if (cut === null || cut === undefined || cut > own.length) return false;
        const theirs = read(p.id);
        return theirs.length >= cut && theirs.slice(0, cut).every((c, i) => c === own[i]);
      })?.id ?? null;
    }
    return { ...take, revision: 0, parent, occurrences: plan.slice(take.from, take.to + 1).map(o => o.id) };
  });
  return { ...stored, version: BUILD_SESSION_VERSION, revision: 0, takes,
    arrangement: { sections, plan, nextOccurrence: plan.length, finished: true } };
}

// ── New chords per take (plan_continue) ─────────────────────────────────────
// A take of a section with no kept occurrence may write its own lines for it:
// the engine plans one section on from the score as it stands (the composed
// score cut before that occurrence, plus its label line), the draw is checked
// here, and the take's score is the session score with that section's lines
// replaced. Keep makes that score the session's, so later repeats follow it.

/** A planned section may be this much shorter or longer than the slot it replaces. */
export const PLAN_BAND = [0.5, 2] as const;

/** The score a plan_continue request starts from for occurrence `k` of `plan`:
 *  the composed score up to its first line plus its label line and newline. */
export function planPrefix(abc: string, lyrics: string, plan: PlanEntry[], sing: Array<string | undefined>, k: number): string {
  const composed = composeArrangement(abc, lyrics, plan, sing);
  const chars = Array.from(composed.abc);
  const at = composed.occurrences[k].abcStart;
  const rest = chars.slice(at).join('').split('\n');
  const end = rest.findIndex(line => label(line.trim()));
  if (end < 0) throw new Error('The occurrence has no label line.');
  return `${chars.slice(0, at).join('')}${rest.slice(0, end + 1).join('\n')}\n`;
}

/** Bars written per voice kind; a voice named neither Vocal nor Ins is not counted. */
function voiceBars(abc: string): { vocal: number; ins: number } {
  const ids = vocalVoiceIds(abc), n = { vocal: 0, ins: 0 };
  let voice: 'vocal' | 'ins' | null = null;
  for (const raw of abc.split('\n')) {
    const line = raw.trim();
    if (/^V:/.test(line)) { voice = isScoreVocalVoice(line, ids) ? 'vocal' : /^V:\s*Ins/i.test(line) ? 'ins' : null; continue; }
    if (!voice || !line || line.startsWith('%') || /^[A-Za-z]:/.test(line)) continue;
    for (const bar of scoreBarSegments(line)) n[voice] += barsIn(bar);
  }
  return n;
}

/** Why a plan_continue draw cannot replace a section of `slot.bars` bars, or null.
 *  `text` is the result score, `prefix` what was sent, `end` the plan stage's
 *  end reason. `eos` and `section_end` both pass: the section is complete. */
export function planDrawRefusal(prefix: string, text: string, end: string | undefined, slot: { bars: number; sung: boolean }): string | null {
  if (end === 'limit_hit') return 'it ran past the plan limit';
  if (!text.startsWith(prefix)) return 'it did not keep the score before it';
  const added = text.slice(prefix.length);
  const head = prefix.trimEnd().split('\n').at(-1)!;
  if (scoreSections(`${head}\n${added}`).length !== 1) return 'it wrote more than one section';
  if (/^\s*[MLQ]:/m.test(added)) return 'it changed the meter, note length or tempo';
  const all = voiceBars(text), before = voiceBars(prefix);
  const vocal = all.vocal - before.vocal, ins = all.ins - before.ins;
  if (vocal <= 0) return 'it wrote no vocal bars';
  if (vocal !== ins) return 'its vocal and instrument bars differ';
  if (vocal < slot.bars * PLAN_BAND[0] || vocal > slot.bars * PLAN_BAND[1]) return `its ${vocal} bars are far from the ${slot.bars} planned`;
  if (slot.sung && !hasSoundingVocalBar(added)) return 'it has no sounding vocal bar for the lyrics';
  return null;
}

/** `abc` with the lines of source section `section` (after its label) replaced by `added`. */
export function spliceSection(abc: string, section: number, added: string): string {
  const lines = abc.replace(/\r\n?/g, '\n').split('\n');
  const starts = lines.flatMap((line, i) => (label(line.trim()) ? [i] : []));
  if (section < 0 || section >= starts.length) throw new Error('No such section in the score.');
  const trailer = lines.at(-1) === '' ? [lines.pop()!] : [];
  const end = starts[section + 1] ?? lines.length;
  lines.splice(starts[section] + 1, end - starts[section] - 1, ...added.replace(/\r\n?/g, '\n').replace(/\n+$/, '').split('\n'));
  return [...lines, ...trailer].join('\n');
}
