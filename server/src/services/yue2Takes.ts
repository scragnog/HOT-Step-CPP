// YuE2 build by section: compose a cover a few sections at a time. A session
// holds the exact /yue2/synth request a one-shot render of the cover would
// send (built by the route from the same params), so every take is
// conditioned on the same prompt, score and lyrics. The "spine" is the last
// kept take. A take of sections [from, to] resumes the spine's codes up to
// where `from` starts in the spine's own audio (codec_ids +
// continue_codec_ids; the first takes resume nothing), composes with its own
// seed, and stops shortly after `to` (preview_max_frames). Keeping a take
// makes it the spine. Save song is an ordinary render job that prefills the
// spine's whole stream (generate.ts, params.yue2Build), so it gets the usual
// post-processing and library save. Section starts come from aligning the
// lyrics on each take's audio, because the score clock drifts from what the
// composer sings. Sessions are files under <audioDir>/takes/<id>/, so the
// /audio static route plays the takes. Sections are built in the
// arrangement's order (coverArrangement.ts): every take and Save song sends
// the score and lyrics written out in that order, and a take's from, to and
// starts index the arrangement's occurrences, never the score's sections.
// The spine can be cropped (cropBuildTake): the cut is a new take in its
// place, so the next take and Save song resume the cut stream and the
// original stays to be put back. A take may carry its own caption
// (generateBuildTakes): it changes what the AR composes for that take only.
// Every preview and Save song still decode the whole stream through NAR under
// one caption: the take's own for its preview, the session's for Save song.

import fs from 'node:fs';
import path from 'node:path';
import { randomInt, randomUUID } from 'node:crypto';
import { config } from '../config.js';
import { aceClient } from './aceClient.js';
import { runOnGpuLane, type LaneLease } from './generation/gpuLane.js';
import { evictOtherFamilies } from './generation/residency.js';
import { yue2Align, yue2Props, yue2Synth, splitMultipartMixed, type Yue2SynthRequest, type Yue2TrackDetail, type Yue2Seed, yue2Seed } from './backends/yue2/client.js';
import { splitLyricBlocks } from './backends/yue2/sectionMatch.js';
import type { Yue2AlignWord } from './backends/yue2/align.js';
import { scoreSections } from './backends/yue2/scoreSections.js';
import { scoreBarClock } from './backends/yue2/scoreClock.js';
import { blindYue2Schedule } from './backends/yue2/lyricSchedule.js';
import {
  BUILD_SESSION_VERSION, IMPROV, composeArrangement, editPlan, generateRefusal, keepRefusal, lookaheadSection, migrateLegacySession, nextOccurrence, planLyrics,
  occurrenceLabel, planDrawRefusal, planPrefix, planSections, revisionRefusal, spliceSection, sourceSections, takeIdentity, type ArrangedSession, type ArrangedTake, type LegacySession, type PlanEntry,
} from './backends/yue2/coverArrangement.js';
import { preparedParams } from '../contracts/yue2Prepare.js';

/** Codec frames per second (the composer's frame clock). */
export const YUE2_FPS = 25;
/** A take runs this much past its sections' score length before stopping. */
const TAKE_SLACK = 1.3;
const TAKE_SLACK_FRAMES = 4 * YUE2_FPS;
const MAX_TAKES = 10;
const MAX_LYRICS = 20_000;
const MAX_CAPTION = 4000;
/** Bars in a take's waveform, over its new part (from its lead-in). */
const PEAKS = 120;
const LEAD_IN_S = 4;

export class TakesError extends Error {
  constructor(message: string, public status = 400) { super(message); }
}

/** A mutation names the revision it was made against; refuse it once the build moved on. */
function checkRevision(stored: Stored, expected: unknown): void {
  const refusal = revisionRefusal(stored, expected);
  if (refusal) throw new TakesError(refusal, 409);
}

export interface Yue2Take {
  id: string;
  /** The occurrences it composed, plan indexes, 0-based and inclusive. */
  from: number;
  to: number;
  seed: Yue2Seed;
  /** Played through /audio. */
  audioUrl: string;
  frames: number;
  /** The frame each occurrence starts at in this take's audio, or null where
   *  the take does not reach it or no start was found. */
  starts: Array<number | null>;
  /** Loudness of its new part, 0..1, from LEAD_IN_S before `prefix` to its end. */
  peaks?: number[];
  /** Codec frames it resumed from its parent; its new part starts here.
   *  Missing on takes from older servers: starts[from] then. */
  prefix?: number;
  /** A crop (cropBuildTake): the take whose audio and codes it cuts short. */
  cropOf?: string;
  /** Where the next take resumes, when not at starts[to + 1]: a crop inside
   *  an occurrence that the next take composes again, or one that keeps part
   *  of the section the take ran on into (`after`). */
  resume?: number;
  /** The section its prompt ran on into after `to` (a source section id, or
   *  improv): the planned one after it, else the lookahead. Missing when it
   *  composed an ending, and on older takes. */
  after?: string;
  /** The style prompt it was sent, adapter trigger included. Missing on older takes. */
  style?: string;
  /** Its own caption, as typed; missing when it used the session's. */
  caption?: string;
  /** The codec frame its audio file starts at: a take renders only from a
   *  lead-in before its new part. Missing or 0: the file starts at the song's start. */
  audioStart?: number;
  /** The whole score (source order) it was composed under, when it planned its
   *  own chords (generateBuildTakes' newChords). Keep makes it the session's
   *  score. Missing: it followed the score it was sent, and keeping it leaves
   *  the score as it was. Not sent to the page. */
  abc?: string;
  /** What did not go to plan: a section it followed the score for, or why no section was planned. */
  planNote?: string;
}

/** Session file version 2 (coverArrangement.ts): revisions, take parents, the
 *  plan. `kept` is the kept takes, oldest first; the last is the spine. */
export interface Yue2BuildSession extends ArrangedSession {
  id: string;
  sections: Array<{ label: string; startBar: number }>;
  /** Takes still rendering. */
  pending: Array<{ from: number; to: number; count: number; activeCount?: number; phase?: string; progress?: number }>;
  error: string | null;
  /** The params a Save song render submits. Applying lyrics or settings
   *  replaces them and the stored request together. */
  params: Record<string, unknown>;
  /** The Prepare confirmations the session started from, when it started
   *  from a Prepare document (contracts/yue2Prepare.ts). */
  prepared?: { structureHash: string; lyricRevision: number };
  /** Where Save song starts the song, in seconds, with a short fade in: set on
   *  a first-section take (`take` names the uncropped original) and applied
   *  only while a version of that take is the first kept one. The codec stream
   *  is not cut, so later takes still continue from all of it. */
  trim?: { take: string; seconds: number } | null;
  /** The whole build so far rendered once, for Play build (takes hold only their own part).
   *  `spine` is the kept take it rendered; it is stale once another take is the spine. */
  preview?: { spine: string; status: 'rendering' | 'done' | 'failed'; progress?: number; audioUrl?: string; error?: string } | null;
}

/** `request` holds the score and lyrics in source order; arranged() writes them out in plan order.
 *  `request.abc` is the score the next take starts from: the last kept take's
 *  own (applyScore), else `baseAbc`, the one the build began with. */
interface Stored extends Yue2BuildSession { userId: string; request: Yue2SynthRequest; baseAbc?: string }

interface BuildRun { cancelled: boolean; jobId?: string }
const runs = new Map<string, BuildRun>();
/** Build previews rendering now, by session id. */
const previewRuns = new Set<string>();

const root = () => path.join(config.data.audioDir, 'takes');
const sessionDir = (id: string) => path.join(root(), id);
const sessionFile = (id: string) => path.join(sessionDir(id), 'session.json');
const codesFile = (id: string, take: string) => path.join(sessionDir(id), `${take}.codes.json`);

function load(id: string, userId: string): Stored {
  if (!/^[0-9a-f-]{36}$/.test(id)) throw new TakesError('Build session not found', 404);
  let stored: Stored | (LegacySession & Pick<Stored, 'id' | 'userId' | 'params' | 'pending' | 'error' | 'request'>);
  try { stored = JSON.parse(fs.readFileSync(sessionFile(id), 'utf8')); }
  catch { throw new TakesError('Build session not found', 404); }
  if (stored.userId !== userId) throw new TakesError('Build session not found', 404);
  if (stored.version === BUILD_SESSION_VERSION) {
    if (stored.preview?.status === 'rendering' && !previewRuns.has(id)) stored.preview = { ...stored.preview, status: 'failed', error: 'The render was interrupted.' };
    return repaired(stored);
  }
  // A session from before revisions: upgrade it once, on first open.
  let upgraded: Stored;
  try {
    upgraded = migrateLegacySession(stored as Exclude<typeof stored, Stored>, take =>
      JSON.parse(fs.readFileSync(codesFile(id, take), 'utf8')) as number[]) as unknown as Stored;
  } catch (err) { throw new TakesError(`This build session cannot be opened: ${(err as Error).message}`, 409); }
  save(upgraded);
  return upgraded;
}

/** Sessions saved before empty tags counted in pairing (coverArrangement's
 *  pairSections) read their sections' lyric blocks afresh. */
function repaired(stored: Stored): Stored {
  const fresh = sourceSections(stored.request.abc ?? '', stored.request.lyrics ?? '');
  if (fresh.length === stored.arrangement.sections.length) {
    stored.arrangement.sections = stored.arrangement.sections.map((s, i) => ({ ...s, lyricBlock: fresh[i].lyricBlock }));
  }
  return stored;
}

function save(stored: Stored): void {
  const tmp = `${sessionFile(stored.id)}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(stored));
  fs.renameSync(tmp, sessionFile(stored.id));
}

// Older session files also carry scoreStarts, which arranged() replaced.
const view = ({ request: _r, scoreStarts: _s, userId: _u, baseAbc: _b, takes, ...rest }: Stored & { scoreStarts?: unknown }): Yue2BuildSession =>
  ({ ...rest, takes: takes.map(({ abc: _abc, ...take }) => take) });

/** Sets the score the next take starts from: the last kept take's own, else the
 *  build's first. A take that followed the score has none of its own and
 *  leaves the score its parent had, so the last kept one that has one wins.
 *  Run after any change to what is kept (keep, undo). */
function applyScore(stored: Stored): void {
  stored.baseAbc ??= stored.request.abc;
  const own = stored.kept.map(id => stored.takes.find(t => t.id === id)?.abc).findLast(abc => abc !== undefined);
  const abc = own ?? stored.baseAbc;
  if (abc === stored.request.abc) return;
  stored.request = { ...stored.request, abc };
  stored.arrangement = { ...stored.arrangement, sections: sourceSections(abc!, stored.request.lyrics ?? '') };
  stored.sections = scoreSections(abc!).map(x => ({ label: x.label.trim(), startBar: x.startBar }));
}

/** A lyric span per occurrence (code points in the composed lyrics), or null when unsung. */
type Spans = Array<[number, number] | null>;

/** What a take or Save song sends now: the session's request with its score
 *  and lyrics written out in plan order, plus each occurrence's lyric span
 *  and its frame on the score clock (and the end's). The source order sends
 *  the stored text unchanged, so sessions from before arrangements render as
 *  they did. A lyric schedule is rebuilt from the composed text, or dropped
 *  when that text would pair a block with the wrong occurrence. */
function arranged(stored: Pick<Stored, 'request' | 'arrangement'>, lookahead = false): { request: Yue2SynthRequest; lyrics: string; spans: Spans; clock: number[] } {
  const abc = stored.request.abc ?? '', lyrics = stored.request.lyrics ?? '';
  const plan = planSections(stored.arrangement);
  // A take of an open plan runs on into the section after it (lookaheadSection).
  const extra = lookahead ? lookaheadSection(stored.arrangement) : null;
  if (extra !== null) plan.push(extra);
  const own = planLyrics(stored.arrangement);
  const composed = composeArrangement(abc, lyrics, plan, own);
  const request = { ...stored.request };
  // Lyric Timing (C6) is off for builds: with a resumed prefix the engine
  // replays it one frame per step, which made late sections crawl.
  delete request.lyric_schedule;
  if (plan.length !== stored.arrangement.sections.length || plan.some((s, i) => s !== i) || own.some(text => text !== undefined)) {
    request.abc = composed.abc;
    if (request.lyrics) request.lyrics = composed.lyrics;
  }
  const marks = scoreSections(request.abc!);
  if (marks.length !== plan.length) throw new TakesError('The arranged score lost a section label; edit the arrangement again.', 409);
  const bars = scoreBarClock(request.abc!);
  const clock = marks.map(m => Math.round((bars[m.startBar - 1]?.start ?? 0) * YUE2_FPS))
    .concat(Math.round((bars.at(-1)?.end ?? 0) * YUE2_FPS));
  return { request, lyrics: composed.lyrics, spans: composed.occurrences.map(o => o.lyric), clock };
}

/** Where each occurrence starts in a take, from the lyrics aligned on its
 *  audio: halfway through the gap before its first sung word. `spans` gives
 *  each occurrence its own lyric block (composeArrangement), so a repeated
 *  chorus is timed by its own copy. Occurrences with no sung block (intro,
 *  solo) start just after the previous block's last word. Occurrence 0
 *  starts at 0. Word times are read as the section matcher reads them
 *  (median of the first three starts; squashed words mean not sung there),
 *  not by the aligner's scores, which read near zero on a full mix. */
export function sectionStarts(spans: Spans, words: Yue2AlignWord[], frames: number): Array<number | null> {
  const timed = (block: { start: number; end: number }) => {
    const own = words.filter(word => word.char0 >= block.start && word.char0 < block.end &&
      Number.isFinite(word.start) && Number.isFinite(word.end) && word.end > word.start);
    if (!own.length) return null;
    const durations = own.map(word => word.end - word.start).sort((a, b) => a - b);
    if (durations[Math.floor(durations.length / 2)] < 0.04) return null;
    const firsts = own.slice(0, 3).map(word => word.start).sort((a, b) => a - b);
    return { start: firsts[Math.floor(firsts.length / 2)], end: Math.max(...own.map(word => word.end)) };
  };
  // The end of the last sung word before each section, unless a sung section
  // in between could not be placed (then what follows it is unknown too).
  let lastEnd: number | null = null;
  let lastStart = 0;
  return spans.map((at, i) => {
    const span = at === null ? null : timed({ start: at[0], end: at[1] });
    const seconds = i === 0 ? 0 : span ? (lastEnd !== null && lastEnd < span.start ? (lastEnd + span.start) / 2
      : Math.max(0, span.start - 0.5)) : at === null && lastEnd !== null ? lastEnd + 0.5 : null;
    if (span) lastEnd = span.end;
    else if (at !== null) lastEnd = null;
    // Sections run in order: a start at or before the last one is a misread
    // (words squashed into a take's tail), not a section.
    const frame = seconds === null ? null : Math.round(seconds * YUE2_FPS);
    if (frame === null || frame >= frames || (i > 0 && frame <= lastStart)) return null;
    lastStart = frame;
    return frame;
  });
}

/** The lyrics up to the end of the block of occurrence `through`. */
export function lyricsThrough(lyrics: string, spans: Spans, through: number): string {
  if (through >= spans.length - 1) return lyrics;
  const end = spans.slice(0, through + 1).filter(span => span !== null).at(-1)?.[1];
  return end === undefined ? lyrics : Array.from(lyrics).slice(0, end).join('');
}

/** An aligned start this much shorter than the score's length for the
 *  section before it is a misread (words squashed early), not a fast take. */
const EARLY_START = 0.75;

/** Checks a take's aligned starts for occurrences (from, through] against the
 *  score clock: a start that comes too soon after the one before it is
 *  dropped, and a missing start is the score clock's estimate from
 *  `sectionStart` while it is inside the take. Edits `starts` in place. */
export function fillStarts(starts: Array<number | null>, clock: number[], from: number, through: number,
  sectionStart: number, frames: number): void {
  for (let i = from + 1; i <= through; i++) {
    const prev = starts[i - 1], own = starts[i];
    if (own !== null && prev !== null && own - prev < EARLY_START * (clock[i] - clock[i - 1])) starts[i] = null;
    const estimate = sectionStart + clock[i] - clock[from];
    if (starts[i] === null && estimate < frames) starts[i] = estimate;
  }
}

// A take that stops early is aligned against the lyrics only up to the
// block of the occurrence after its own: words it never sang would be
// squashed into its tail and drag the earlier ones with them. Fewer words
// fail the other way (the aligner spreads them to fill the audio), so
// fillStarts checks the result against the score clock.
/** Each occurrence's start frame in a take, from aligning its lyrics to its audio.
 *  A take's audio starts at codec frame `audioStart`, so only the lyrics from
 *  occurrence `from` on are aligned, and the words are moved back onto the
 *  song's clock and the whole lyrics' code points. */
async function alignStarts(lyrics: string, spans: Spans, wav: Buffer, frames: number, through?: number,
  from = 0, audioStart = 0): Promise<Array<number | null>> {
  if (!lyrics.trim()) return spans.map((_, i) => (i === 0 ? 0 : null));
  const chars = Array.from(through === undefined ? lyrics : lyricsThrough(lyrics, spans, through));
  const first = audioStart ? spans.slice(from).find(span => span !== null)?.[0] ?? chars.length : 0;
  const text = chars.slice(first).join('');
  if (!text.trim()) return spans.map((_, i) => (i === 0 ? 0 : null));
  const aligned = await yue2Align(wav, text);
  const offset = audioStart / YUE2_FPS;
  const words = aligned.words.map(word => ({ ...word, char0: word.char0 + first, start: word.start + offset, end: word.end + offset }));
  const starts = sectionStarts(spans, words, frames);
  // Past `through` the lyrics were cut, so those occurrences read as unsung.
  return spans.map((_, i) => (through !== undefined && i > through ? null : starts[i] ?? null));
}

async function waitJob(jobId: string, run: BuildRun, lease: LaneLease,
  progress: (phase: string, progress: number) => void): Promise<Yue2TrackDetail[]> {
  const started = Date.now();
  let cancelSent = false;
  for (;;) {
    if ((run.cancelled || !lease.isCurrent()) && !cancelSent) {
      run.cancelled = true;
      await aceClient.cancelJob(jobId);
      cancelSent = true;
    }
    const status = await aceClient.pollJob(jobId);
    if (status.status === 'done') {
      if (run.cancelled) throw new Error('YuE2 takes cancelled');
      return (status as { tracks?: Yue2TrackDetail[] }).tracks ?? [];
    }
    if (status.status === 'failed' || status.status === 'cancelled') {
      throw new Error(`YuE2 take ${status.status}${(status as { error?: string }).error ? `: ${(status as { error?: string }).error}` : ''}`);
    }
    if (Date.now() - started > 15 * 60_000) {
      await aceClient.cancelJob(jobId).catch(() => {});
      throw new Error('YuE2 take timed out after 15 minutes');
    }
    progress(status.phase ?? 'rendering', status.phase_total ? Math.round(100 * (status.phase_step ?? 0) / status.phase_total) : 0);
    await new Promise(r => setTimeout(r, 750));
  }
}

/** A waveform of a 16-bit PCM WAV between two times: PEAKS peak levels. */
export function wavPeaks(wav: Buffer, fromSec: number, toSec: number): number[] {
  // Walk the chunks to fmt and data; the engine writes plain 16-bit PCM.
  let off = 12, channels = 2, rate = 48000, dataOff = -1, dataLen = 0;
  while (off + 8 <= wav.length) {
    const id = wav.toString('ascii', off, off + 4), len = wav.readUInt32LE(off + 4);
    if (id === 'fmt ') { channels = wav.readUInt16LE(off + 10); rate = wav.readUInt32LE(off + 12); }
    if (id === 'data') { dataOff = off + 8; dataLen = Math.min(len, wav.length - dataOff); break; }
    off += 8 + len + (len % 2);
  }
  if (dataOff < 0) return [];
  const frameBytes = 2 * channels, frames = Math.floor(dataLen / frameBytes);
  const a = Math.max(0, Math.floor(fromSec * rate)), b = Math.min(frames, Math.floor(toSec * rate));
  if (b <= a) return [];
  const peaks: number[] = [];
  for (let k = 0; k < PEAKS; k++) {
    const lo = a + Math.floor(((b - a) * k) / PEAKS), hi = a + Math.floor(((b - a) * (k + 1)) / PEAKS);
    let peak = 0;
    for (let f = lo; f < hi; f += 16) peak = Math.max(peak, Math.abs(wav.readInt16LE(dataOff + f * frameBytes)));
    peaks.push(Math.round((peak / 32768) * 100) / 100);
  }
  return peaks;
}

/** The take count controls songs; every song has exactly one noise variation. */
function plainRequest(request: Yue2SynthRequest): Yue2SynthRequest {
  request = { ...request };
  delete request.lm_batch_size;
  delete request.synth_batch_size;
  delete request.songs;
  delete request.seed;
  return request;
}

/** Starts a build session for a cover's request. With `params.yue2Prepare`
 *  (the Prepare document), the shared submission check runs once here: the
 *  score must be the confirmed one and the lyrics the ones it exports. The
 *  session keeps the checked pair and the confirmations, not the document. */
export function createBuildSession(params: Record<string, unknown>, request: Yue2SynthRequest, userId: string): Yue2BuildSession {
  let prepared: Yue2BuildSession['prepared'];
  if (params.yue2Prepare !== undefined) {
    const pair = preparedParams(params);
    if ('error' in pair) throw new TakesError(pair.error, 409);
    const { yue2Prepare: _, ...rest } = params;
    params = rest;
    prepared = { structureHash: pair.structureHash, lyricRevision: pair.lyricRevision };
  }
  if (!request.abc) throw new TakesError('Build by section needs an approved score.');
  const sections = scoreSections(request.abc).map(s => ({ label: s.label.trim(), startBar: s.startBar }));
  if (!sections.length) throw new TakesError('The score has no section labels.');
  request = plainRequest(request);
  const source = sourceSections(request.abc!, request.lyrics ?? '');
  const id = randomUUID();
  fs.mkdirSync(sessionDir(id), { recursive: true });
  const stored: Stored = {
    id, userId, sections, takes: [], kept: [], pending: [], error: null, params, request, baseAbc: request.abc,
    version: BUILD_SESSION_VERSION, revision: 0,
    arrangement: { sections: source, plan: source.map((s, i) => ({ id: `o${i}`, section: s.id })), nextOccurrence: source.length, finished: true },
    ...(prepared ? { prepared } : {}),
  };
  save(stored);
  return view(stored);
}

export function getBuildSession(id: string, userId: string): Yue2BuildSession {
  return view(load(id, userId));
}

/** Queues `count` takes of occurrences [next, to], resumed from the spine
 *  under the arranged prompt. Returns at once; the session lists the takes
 *  as they finish. A non-empty `caption` replaces the session's for these
 *  takes only; `toStyle` composes the style prompt a render of params would
 *  send (adapter trigger and template included). Empty or missing inherits.
 *  It is not session state, so the revision stays.
 *
 *  `newChords: true` has each take write its own lines for the sections with
 *  no kept occurrence (planTakeScores) under `planAdapter`, the planner-strength
 *  adapter stack; anything else, and any section that cannot be planned,
 *  follows the score. */
export function generateBuildTakes(id: string, userId: string, to: number, count: number, revision: unknown,
  caption?: unknown, toStyle?: (params: Record<string, unknown>) => string, newChords?: unknown,
  planAdapter?: (params: Record<string, unknown>) => Yue2SynthRequest['lm_adapter']): Yue2BuildSession {
  const stored = load(id, userId);
  checkRevision(stored, revision);
  if (caption !== undefined && caption !== null && typeof caption !== 'string') throw new TakesError('caption must be text.');
  if (typeof caption === 'string' && caption.length > MAX_CAPTION) throw new TakesError(`A caption is limited to ${MAX_CAPTION.toLocaleString('en-US')} characters.`);
  const own = typeof caption === 'string' && caption.trim() ? caption.trim() : undefined;
  if (own && !toStyle) throw new TakesError('A take caption needs the style composer.', 500);
  const from = nextOccurrence(stored);
  const refusal = generateRefusal(stored, to);
  if (refusal) throw new TakesError(refusal);
  if (!Number.isInteger(count) || count < 1 || count > MAX_TAKES) throw new TakesError(`Generate 1 to ${MAX_TAKES} takes at a time.`);
  const spine = stored.takes.find(t => t.id === stored.kept.at(-1));
  // Where occurrence `from` starts, and where this take resumes: the same
  // place unless a crop cut into `from` (cropBuildTake).
  const sectionStart = spine ? spine.starts[from] : 0;
  const start = spine?.resume ?? sectionStart;
  if (start === null || start === undefined || sectionStart === null || sectionStart === undefined) {
    throw new TakesError('The kept take does not reach the next section. Undo the keep and try another take.');
  }
  // A crop that kept the start of the section the spine ran on into resumes
  // that audio, so the next occurrence must be that section.
  const base = spine && stored.takes.find(t => t.id === spine.cropOf);
  if (spine?.resume !== undefined && spine.after && base?.to === spine.to && stored.arrangement.plan[from]?.section !== spine.after) {
    const label = spine.after === IMPROV ? 'Improv' : stored.arrangement.sections.find(s => s.id === spine.after)?.label ?? 'the section it ran on into';
    throw new TakesError(`The kept take carries on into ${label}: put ${label} next, or crop the kept take again.`);
  }
  const prefix = spine ? (JSON.parse(fs.readFileSync(codesFile(id, spine.id), 'utf8')) as number[]).slice(0, start) : [];
  const { request, lyrics, spans, clock } = arranged(stored, true);
  // prompt is the first caption field a render reads (generate.ts), so it wins.
  if (own) request.style = toStyle!({ ...stored.params, prompt: own });
  const occurrences = spans.length;
  const last = to + 1 >= occurrences;
  const capOf = (marks: number[]) => (last ? undefined
    : start + Math.round(Math.max(marks[to + 1] - marks[from], 1) * TAKE_SLACK) + TAKE_SLACK_FRAMES);
  const name = (i: number) => occurrenceLabel(stored.arrangement, i) ?? `#${i + 1}`;
  const extra = lookaheadSection(stored.arrangement);
  const after = last ? undefined : stored.arrangement.plan[to + 1]?.section ?? (extra === null ? undefined : stored.arrangement.sections[extra].id);
  // The audio renders from the lead-in a take plays from: the NAR and VAE skip
  // what is already kept, which is what made late sections slow.
  const renderFrom = start ? Math.max(0, start - LEAD_IN_S * YUE2_FPS) : 0;
  // An Improv occurrence (or a score section labelled Improv) is composed with
  // the score and lyrics hidden (a blind window), from where this take starts
  // sampling to the section's end on the score clock, so its prefix still
  // prefills in one pass.
  const blind: Array<[number, number]> = [];
  for (let k = from; k <= to; k++) {
    const a = Math.max(clock[k], start + 1), b = clock[k + 1];
    if (/^improv/i.test(name(k)) && b > a) blind.push([a / YUE2_FPS, b / YUE2_FPS]);
  }
  if (blind.length) {
    request.lyric_schedule = blindYue2Schedule(request.abc!, request.lyrics ?? '', blind);
  }
  // The sections this take writes its own chords for: each source section
  // with no kept occurrence, once. A crop that resumes inside `from` keeps
  // part of its audio, composed under the score as it was.
  const heard = new Set(stored.arrangement.plan.slice(0, from).map(o => o.section));
  const todo: Array<{ k: number; s: number; name: string }> = [];
  for (let k = from; k <= to; k++) {
    const section = stored.arrangement.plan[k].section;
    if (section === IMPROV || heard.has(section) || (k === from && spine?.resume !== undefined)) continue;
    heard.add(section);
    todo.push({ k, s: stored.arrangement.sections.findIndex(x => x.id === section), name: name(k) });
  }
  const planning = newChords === true && todo.length > 0;
  const sourceLyrics = stored.request.lyrics ?? '';
  const baseScore = stored.request.abc!;
  const planPlan = planSections(stored.arrangement), planOwn = planLyrics(stored.arrangement);
  const planWire = planAdapter?.(stored.params);
  if (stored.pending.length) throw new TakesError('Wait for the pending takes or cancel them first.');
  if (previewRuns.has(id)) throw new TakesError('Wait for the build render to finish first.');
  const identity = takeIdentity(stored, to);
  const pending = { from, to, count, activeCount: 0, phase: 'queued', progress: 0 };
  stored.pending.push(pending);
  stored.error = null;
  save(stored);
  const run: BuildRun = { cancelled: false };
  runs.set(id, run);

  void (async () => {
    try {
      for (let n = 0; n < count; n += 4) {
        if (run.cancelled) throw new Error('YuE2 takes cancelled');
        const seeds = Array.from({ length: Math.min(4, count - n) }, () => randomInt(1, 2 ** 31 - 1));
        await runOnGpuLane(async lease => {
          // Free every other family before YuE2 loads on the lane.
          await evictOtherFamilies('yue2');
          const checkCancelled = () => {
            if (run.cancelled || !lease.isCurrent()) throw new Error('YuE2 takes cancelled');
          };
          checkCancelled();
          const update = (phase: string, progress: number) => {
            const fresh = load(id, userId);
            Object.assign(fresh.pending[0], { activeCount: seeds.length, phase, progress });
            save(fresh);
          };
          update('starting', 0);
          // Each take's own score, and what its clock and request are made from it.
          let scores = seeds.map(() => baseScore), notes = seeds.map(() => ''), planned = seeds.map(() => false);
          if (planning) {
            update('plan', 0);
            const plans = await planTakeScores({ run, lease, checkCancelled, base: baseScore, sourceLyrics, lyrics, plan: planPlan, own: planOwn,
              request, todo, seeds, planAdapter: planWire });
            scores = plans.map(p => p.abc);
            notes = plans.map(p => p.note);
            planned = plans.map(p => p.planned);
            update('starting', 0);
          }
          const views = scores.map(abc => (abc === baseScore ? { request, lyrics, spans, clock }
            : arranged({ request: { ...stored.request, abc }, arrangement: { ...stored.arrangement, sections: sourceSections(abc, sourceLyrics) } }, true)));
          const cap = Math.max(...views.map(v => capOf(v.clock) ?? 0)) || undefined;
          const req: Yue2SynthRequest = { ...request, synth_batch_size: 1,
            ...(prefix.length ? { codec_ids: prefix, continue_codec_ids: true } : {}),
            ...(cap ? { preview_max_frames: cap } : {}),
            ...(renderFrom ? { render_from_frame: renderFrom } : {}) };
          // Sessions saved by older servers may still carry batch fields.
          delete req.lm_batch_size;
          delete req.songs;
          if (seeds.length === 1) { req.seed = seeds[0]; req.abc = views[0].request.abc; }
          else {
            delete req.seed;
            req.songs = seeds.map((seed, i) => ({ style: request.style, lyrics: views[i].request.lyrics, abc: views[i].request.abc, seed }));
          }
          // Retry winners use a different seed, which breaks the take identity.
          delete req.semantic_retries;
          const sub = await yue2Synth(req);
          run.jobId = sub.job_id;
          try {
            const tracks = await waitJob(sub.job_id, run, lease, update);
            checkCancelled();
            const body = await aceClient.getJobResult(sub.job_id);
            if (!body.ok) throw new Error(`YuE2 take result fetch failed (${body.status})`);
            const audio = Buffer.from(await body.arrayBuffer());
            const contentType = body.headers.get('content-type') ?? 'audio/wav';
            const parts = contentType.startsWith('multipart/mixed') ? splitMultipartMixed(audio, contentType) : [audio];
            // Metadata and WAV parts share their order, but song indices may have
            // gaps. Validate the entire chunk before attaching any of its audio.
            const mapped = mapTakeTracks(seeds, tracks, parts);
            update('aligning', 100);
            const takes: ArrangedTake[] = [];
            for (const [i, { seed, ids, wav, audioStart }] of mapped.entries()) {
              checkCancelled();
              const starts = await alignStarts(lyrics, spans, wav, ids.length, last ? undefined : to + 1, from, audioStart);
              checkCancelled();
              for (let i = 0; i <= from; i++) starts[i] = spine ? spine.starts[i] : i === 0 ? 0 : null;
              fillStarts(starts, views[i].clock, from, Math.min(to + 1, occurrences - 1), sectionStart, ids.length);
              const takeId = randomUUID().slice(0, 8);
              const peaks = wavPeaks(wav, Math.max(0, start / YUE2_FPS - (start ? LEAD_IN_S : 0)) - audioStart / YUE2_FPS, (ids.length - audioStart) / YUE2_FPS);
              fs.writeFileSync(path.join(sessionDir(id), `${takeId}.wav`), wav);
              fs.writeFileSync(codesFile(id, takeId), JSON.stringify(ids));
              takes.push({ id: takeId, from, to, seed, audioUrl: `/audio/takes/${id}/${takeId}.wav`, frames: ids.length, starts, peaks, prefix: start, ...(after ? { after } : {}), style: request.style, ...(own ? { caption: own } : {}), ...(audioStart ? { audioStart } : {}),
                ...(planned[i] ? { abc: scores[i] } : {}), ...(notes[i] ? { planNote: notes[i] } : {}), ...identity });
            }
            const fresh = load(id, userId);
            fresh.takes.push(...takes);
            Object.assign(fresh.pending[0], { count: count - n - seeds.length, activeCount: 0, phase: 'queued', progress: 0 });
            save(fresh);
          } catch (err) {
            await aceClient.cancelJob(sub.job_id).catch(() => {});
            throw err;
          } finally {
            run.jobId = undefined;
          }
        }, { label: `yue2 build: ${name(from)}–${name(to)}`, family: 'yue2' });
      }
    } catch (err) {
      const fresh = load(id, userId);
      fresh.error = (err as Error).message;
      save(fresh);
    } finally {
      runs.delete(id);
      const fresh = load(id, userId);
      fresh.pending = [];
      save(fresh);
    }
  })();
  return view(stored);
}

interface PlanTakes {
  run: BuildRun; lease: LaneLease; checkCancelled: () => void;
  /** The session score (source order) every take starts from, and the lyrics it pairs with. */
  base: string; sourceLyrics: string;
  /** The composed lyrics a take sends, and the plan they are composed in. */
  lyrics: string; plan: PlanEntry[]; own: Array<string | undefined>;
  /** The take's request: style and settings for the plan pass. */
  request: Yue2SynthRequest;
  /** Plan indexes of the sections to write, with their source section index, in order. */
  todo: Array<{ k: number; s: number; name: string }>;
  seeds: number[];
  planAdapter?: Yue2SynthRequest['lm_adapter'];
}

/** The most tokens one planned section may take: a section is a few hundred. */
const PLAN_SECTION_TOKENS = 1024;

/** One take's own score for each seed: the session score with the lines of each
 *  `todo` section planned by the engine (plan_continue) after the score as the
 *  take has it so far, so a chorus is written knowing the verse before it.
 *  One batched pass per section serves every seed; a draw that fails the
 *  checks (planDrawRefusal) gets one more pass with a new seed. A seed still
 *  without a plan follows the score for that section and says so; the planner
 *  never blocks a take. `planned` is false for a seed that did not try. */
export async function planTakeScores(job: PlanTakes): Promise<Array<{ abc: string; note: string; planned: boolean }>> {
  const unplanned = (note: string) => job.seeds.map(() => ({ abc: job.base, note, planned: false }));
  if (job.request.cot === 'off') return unplanned('Chain of Thought is off, so there is no plan to write: this take follows the score.');
  if (!(await yue2Props()).props?.plan_continue) return unplanned('This engine cannot plan from the kept score yet (update it): this take follows the score.');
  const scores = job.seeds.map(() => job.base), notes = job.seeds.map((): string[] => []);
  for (const { k, s, name } of job.todo) {
    let open = job.seeds.map((_, i) => i);
    const why = new Map<number, string>();
    for (let attempt = 0; attempt < 2 && open.length; attempt++) {
      job.checkCancelled();
      const sent = open.map(i => ({ i, prefix: planPrefix(scores[i], job.sourceLyrics, job.plan, job.own, k),
        seed: attempt ? randomInt(1, 2 ** 31 - 1) : job.seeds[i] }));
      const req = plainRequest(job.request);
      for (const key of ['abc', 'lyrics', 'noise_seed', 'codec_ids', 'continue_codec_ids', 'preview_max_frames', 'render_from_frame',
        'semantic_retries', 'lyric_schedule'] as const) delete req[key];
      Object.assign(req, { plan_only: true, plan_continue: true, plan_max_tokens: PLAN_SECTION_TOKENS,
        songs: sent.map(x => ({ style: job.request.style, lyrics: job.lyrics, abc: x.prefix, seed: x.seed })) });
      if (job.planAdapter !== undefined) req.lm_adapter = job.planAdapter;
      let tracks: Yue2TrackDetail[] = [];
      try {
        const sub = await yue2Synth(req);
        job.run.jobId = sub.job_id;
        try { tracks = await waitJob(sub.job_id, job.run, job.lease, () => {}); }
        catch (err) { await aceClient.cancelJob(sub.job_id).catch(() => {}); throw err; }
        finally { job.run.jobId = undefined; }
      } catch (err) {
        job.checkCancelled();
        for (const x of sent) why.set(x.i, `the plan pass failed (${(err as Error).message})`);
        open = sent.map(x => x.i);
        continue;
      }
      open = [];
      for (const [n, { i, prefix }] of sent.entries()) {
        const track = tracks.find(t => t.song === n);
        const slot = sourceSections(scores[i], job.sourceLyrics)[s];
        const refusal = !track?.abc ? 'the planner returned no score'
          : planDrawRefusal(prefix, track.abc, track.stage_end_reasons?.plan, { bars: slot.bars, sung: slot.lyricBlock !== null });
        if (refusal) { why.set(i, refusal); open.push(i); continue; }
        scores[i] = spliceSection(scores[i], s, track!.abc!.slice(prefix.length));
      }
    }
    for (const i of open) notes[i].push(`${name} follows the score: ${why.get(i)}.`);
  }
  return scores.map((abc, i) => ({ abc, note: notes[i].join(' '), planned: true }));
}

/** Validate every track before any take is saved. Parts follow track order. */
export function mapTakeTracks(seeds: number[], tracks: Yue2TrackDetail[], parts: Buffer[]): Array<{ seed: Yue2Seed; ids: number[]; wav: Buffer; audioStart: number }> {
  if (tracks.length !== seeds.length || parts.length !== tracks.length) {
    throw new Error(`YuE2 take chunk expected ${seeds.length} tracks, received ${tracks.length} metadata rows and ${parts.length} audio parts. No takes saved.`);
  }
  const mapped: Array<{ seed: Yue2Seed; ids: number[]; wav: Buffer; audioStart: number }> = [];
  for (let i = 0; i < tracks.length; i++) {
    const track = tracks[i];
    if (!Number.isInteger(track.song) || track.song < 0 || track.song >= seeds.length || mapped[track.song]
      || yue2Seed(track.seed) !== seeds[track.song] || track.variation !== 0) {
      throw new Error(`YuE2 take chunk has a missing, duplicate or mismatched song/seed at track ${i}. No takes saved.`);
    }
    if (!track.semantic_ids?.length || parts[i].length < 44
      || parts[i].toString('ascii', 0, 4) !== 'RIFF' || parts[i].toString('ascii', 8, 12) !== 'WAVE') {
      throw new Error(`YuE2 take chunk song ${track.song} has no codec stream or valid WAV audio. No takes saved.`);
    }
    mapped[track.song] = { seed: yue2Seed(track.seed)!, ids: track.semantic_ids, wav: parts[i], audioStart: track.audio_from ?? 0 };
  }
  return mapped;
}

/** Stops this session's current chunk and prevents later chunks from starting. */
export async function cancelBuildTakes(id: string, userId: string): Promise<Yue2BuildSession> {
  load(id, userId);
  const run = runs.get(id);
  if (run) {
    run.cancelled = true;
    if (run.jobId) await aceClient.cancelJob(run.jobId);
  }
  return getBuildSession(id, userId);
}

/** Keeps a take of the next sections: it becomes the spine. */
export function keepBuildTake(id: string, userId: string, takeId: string, revision: unknown): Yue2BuildSession {
  const stored = load(id, userId);
  checkRevision(stored, revision);
  const take = stored.takes.find(t => t.id === takeId);
  if (!take) throw new TakesError('No such take.');
  const refusal = keepRefusal(stored, take);
  if (refusal) throw new TakesError(refusal, 409);
  stored.kept.push(take.id);
  applyScore(stored);
  save(stored);
  return view(stored);
}

/** Drops kept takes from the end: one, or until occurrence `section` is the
 *  next to build again. The revision stays, so a take made from an earlier
 *  spine is keepable again. */
export function undoBuildKeep(id: string, userId: string, section: number | undefined, revision: unknown): Yue2BuildSession {
  const stored = load(id, userId);
  checkRevision(stored, revision);
  if (section === undefined) stored.kept.pop();
  else {
    if (!Number.isInteger(section) || section < 0) throw new TakesError('No such section.');
    while (stored.kept.length && nextOccurrence(stored) > section) stored.kept.pop();
  }
  applyScore(stored);
  save(stored);
  return view(stored);
}

/** Deletes a take that is not part of the build, with its audio and codes. */
export function deleteBuildTake(id: string, userId: string, takeId: string, revision: unknown): Yue2BuildSession {
  const stored = load(id, userId);
  checkRevision(stored, revision);
  if (!stored.takes.some(t => t.id === takeId)) throw new TakesError('No such take.');
  if (stored.kept.includes(takeId)) throw new TakesError('That take is part of the build; undo back past it first.');
  if (stored.takes.some(t => t.cropOf === takeId && stored.kept.includes(t.id))) {
    throw new TakesError('A crop of that take is part of the build; undo the crop first.');
  }
  stored.takes = stored.takes.filter(t => t.id !== takeId);
  save(stored);
  for (const file of [`${takeId}.wav`, `${takeId}.codes.json`]) {
    try { fs.unlinkSync(path.join(sessionDir(id), file)); } catch { /* already gone */ }
  }
  return view(stored);
}

/** Deletes the whole build: its session folder with every take's audio and codes. Refused while takes or the build render are rendering, or while `hasActiveSave` reports a Save song job that will read the folder. */
export function deleteBuild(id: string, userId: string, hasActiveSave: (id: string) => boolean = () => false): void {
  load(id, userId);
  if (runs.has(id) || previewRuns.has(id)) throw new TakesError('Takes are still rendering; cancel them first.', 409);
  if (hasActiveSave(id)) throw new TakesError('A Save song render is queued or running for this build.', 409);
  fs.rmSync(sessionDir(id), { recursive: true, force: true });
}

/** A 16-bit PCM WAV cut to its first `seconds`; chunks after the data are dropped. */
export function cropWav(wav: Buffer, seconds: number): Buffer {
  let off = 12, frameBytes = 4, rate = 48000;
  while (off + 8 <= wav.length) {
    const id = wav.toString('ascii', off, off + 4), len = wav.readUInt32LE(off + 4);
    if (id === 'fmt ') { frameBytes = 2 * wav.readUInt16LE(off + 10); rate = wav.readUInt32LE(off + 12); }
    if (id === 'data') {
      const keep = Math.min(Math.round(seconds * rate) * frameBytes, Math.floor(Math.min(len, wav.length - off - 8) / frameBytes) * frameBytes);
      const out = Buffer.from(wav.subarray(0, off + 8 + keep));
      out.writeUInt32LE(keep, off + 4);
      out.writeUInt32LE(out.length - 8, 4);
      return out;
    }
    off += 8 + len + (len % 2);
  }
  throw new TakesError('The take\'s audio has no data chunk.', 409);
}

/** A 16-bit PCM WAV without its first `seconds`, faded in over `fade` seconds so it does not start on a click. */
export function trimWavStart(wav: Buffer, seconds: number, fade: number): Buffer {
  let off = 12, frameBytes = 4, rate = 48000;
  while (off + 8 <= wav.length) {
    const id = wav.toString('ascii', off, off + 4), len = wav.readUInt32LE(off + 4);
    if (id === 'fmt ') { frameBytes = 2 * wav.readUInt16LE(off + 10); rate = wav.readUInt32LE(off + 12); }
    if (id === 'data') {
      const total = Math.floor(Math.min(len, wav.length - off - 8) / frameBytes);
      const skip = Math.min(total, Math.max(0, Math.round(seconds * rate)));
      const data = Buffer.from(wav.subarray(off + 8 + skip * frameBytes, off + 8 + total * frameBytes));
      const ramp = Math.min(total - skip, Math.round(fade * rate));
      for (let f = 0; f < ramp; f++) for (let b = 0; b < frameBytes; b += 2) {
        data.writeInt16LE(Math.round(data.readInt16LE(f * frameBytes + b) * f / ramp), f * frameBytes + b);
      }
      const head = Buffer.from(wav.subarray(0, off + 8));
      head.writeUInt32LE(data.length, off + 4);
      const out = Buffer.concat([head, data]);
      out.writeUInt32LE(out.length - 8, 4);
      return out;
    }
    off += 8 + len + (len % 2);
  }
  throw new TakesError('The audio has no data chunk.', 409);
}

/** The fade in at a trimmed start. */
export const TRIM_FADE_S = 0.5;

/** Sets where Save song starts the song (null: from the top). Only a take of
 *  the first section can carry it, and at least a second of it must remain.
 *  It leaves the revision: nothing a take is sent changes. */
export function setBuildTrim(id: string, userId: string, takeId: string, revision: unknown, seconds: unknown): Yue2BuildSession {
  const stored = load(id, userId);
  checkRevision(stored, revision);
  if (seconds === null) {
    if (!stored.trim) return view(stored);
    stored.trim = null;
  } else {
    const take = stored.takes.find(t => t.id === takeId);
    if (!take) throw new TakesError('No such take.');
    if (take.from !== 0 || take.parent != null) throw new TakesError('Only a take of the first section can start later.', 409);
    const length = take.frames / YUE2_FPS;
    if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds <= 0 || seconds > length - 1) {
      throw new TakesError(`Start between 0 and ${(length - 1).toFixed(2)} s.`);
    }
    stored.trim = { take: take.cropOf ?? take.id, seconds: Math.round(seconds * 1000) / 1000 };
  }
  save(stored);
  return view(stored);
}

/** Seconds Save song cuts from the start, when a version of the trimmed take is the first kept one. */
function trimStart(stored: Stored): number | undefined {
  const first = stored.takes.find(t => t.id === stored.kept[0]);
  return first && stored.trim && (first.cropOf ?? first.id) === stored.trim.take ? stored.trim.seconds : undefined;
}

type CropMode = 'regenerate' | 'advance';

/** What a take cut at codec frame `frame` covers. The cut lies in the part the
 *  take composed. Cut inside an occurrence, `mode` says whether the next take
 *  composes that occurrence again from the cut ('regenerate') or starts the
 *  one after it there ('advance'); refused without one. A cut at an
 *  occurrence's start needs no choice. Past the next one's start (the part
 *  the take ran on into) the next take replaces that part, unless 'regenerate'
 *  keeps it up to the cut. */
export function cropPlan(take: Pick<Yue2Take, 'from' | 'to' | 'frames' | 'starts' | 'prefix'>, frame: unknown, mode: unknown,
  label: (occurrence: number) => string = i => `section ${i + 1}`): { to: number; starts: Array<number | null>; resume?: number; prefix: number } {
  const prefix = take.prefix ?? take.starts[take.from] ?? 0;
  if (typeof frame !== 'number' || !Number.isInteger(frame) || frame <= prefix || frame >= take.frames) {
    throw new TakesError(`Crop between ${(prefix / YUE2_FPS).toFixed(2)} s and ${(take.frames / YUE2_FPS).toFixed(2)} s: inside the part this take composed.`);
  }
  // The occurrence the cut falls in: the last one known to start at or before it.
  let k = take.from;
  for (let i = take.from + 1; i <= Math.min(take.to + 1, take.starts.length - 1); i++) {
    const s = take.starts[i];
    if (s !== null && s !== undefined && s <= frame) k = i;
  }
  let to = k - 1, resume: number | undefined;
  if (k <= take.to && take.starts[k] !== frame) {
    if (mode === 'advance') to = k;
    else if (mode === 'regenerate') resume = frame;
    else throw new TakesError(`The cut falls inside ${label(k)}: choose whether the next take composes it again or moves on.`);
  }
  // Past the next section's start, in the part the take ran on into: by
  // default the next take composes that section from its start; 'regenerate'
  // keeps it up to the cut and the next take carries on from there.
  if (k > take.to) {
    to = take.to;
    if (mode === 'regenerate' && take.starts[k] !== frame) resume = frame;
  }
  const starts = take.starts.map((s, i) => (i > to + 1 ? null : s));
  if (to === k && k + 1 < starts.length) starts[k + 1] = frame;
  return { to, starts, resume, prefix };
}

/** Crops the last kept take at codec frame `frame` (cropPlan). The crop is a
 *  new take that replaces it as the spine, so the original keeps its audio
 *  and codes; `frame` null puts the original back. A later crop always cuts
 *  the original. Takes made from the replaced spine stop being keepable
 *  through their parent and come back with it. Crops leave the revision. */
export function cropBuildTake(id: string, userId: string, takeId: string, revision: unknown, frame: unknown, mode: unknown): Yue2BuildSession {
  const stored = load(id, userId);
  checkRevision(stored, revision);
  if (stored.pending.length) throw new TakesError('Wait for the pending takes or cancel them first.', 409);
  const spine = stored.takes.find(t => t.id === stored.kept.at(-1));
  if (!spine || spine.id !== takeId) throw new TakesError('Only the last kept take can be cropped.', 409);
  const base = stored.takes.find(t => t.id === spine.cropOf) ?? spine;
  // Reaching past the spine re-adds occurrences a regenerate crop reopened,
  // which the plan or lyrics may have changed since: keep's own rules apply.
  const refuseReach = (take: ArrangedTake) => {
    const refusal = take.to > spine.to ? keepRefusal({ ...stored, kept: stored.kept.slice(0, -1) }, take) : null;
    if (refusal) throw new TakesError(`${refusal} Its cut part no longer matches the build: crop within what is built, or generate that section again.`, 409);
  };
  let next: ArrangedTake;
  if (frame === null) {
    if (base === spine) throw new TakesError(spine.cropOf ? 'The uncropped take was deleted.' : 'That take is not cropped.', 409);
    refuseReach(base);
    next = base;
  } else {
    const name = (i: number) => occurrenceLabel(stored.arrangement, i) ?? `section ${i + 1}`;
    const plan = cropPlan(base, frame, mode, name);
    const cut = frame as number;
    const occurrences = base.occurrences.slice(0, Math.max(0, plan.to - base.from + 1));
    refuseReach({ ...base, to: plan.to, occurrences });
    const same = stored.takes.find(t => t.cropOf === base.id && t.frames === cut && t.to === plan.to && t.resume === plan.resume);
    if (same) next = same;
    else {
      const cropId = randomUUID().slice(0, 8);
      const audioStart = base.audioStart ?? 0;
      const wav = cropWav(fs.readFileSync(path.join(sessionDir(id), `${base.id}.wav`)), (cut - audioStart) / YUE2_FPS);
      const codes = (JSON.parse(fs.readFileSync(codesFile(id, base.id), 'utf8')) as number[]).slice(0, cut);
      fs.writeFileSync(path.join(sessionDir(id), `${cropId}.wav`), wav);
      fs.writeFileSync(codesFile(id, cropId), JSON.stringify(codes));
      next = { ...base, id: cropId, audioUrl: `/audio/takes/${id}/${cropId}.wav`, frames: cut, to: plan.to, starts: plan.starts,
        prefix: plan.prefix, resume: plan.resume, cropOf: base.cropOf ?? base.id, occurrences,
        peaks: wavPeaks(wav, Math.max(0, plan.prefix / YUE2_FPS - (plan.prefix ? LEAD_IN_S : 0)) - audioStart / YUE2_FPS, (cut - audioStart) / YUE2_FPS) };
      stored.takes.push(next);
    }
  }
  stored.kept[stored.kept.length - 1] = next.id;
  // A replaced crop goes, unless takes were made from it.
  const drop = spine.cropOf && spine !== next && !stored.takes.some(t => t.parent === spine.id);
  if (drop) stored.takes = stored.takes.filter(t => t !== spine);
  save(stored);
  if (drop) for (const file of [`${spine.id}.wav`, `${spine.id}.codes.json`]) {
    try { fs.unlinkSync(path.join(sessionDir(id), file)); } catch { /* already gone */ }
  }
  return view(stored);
}

/** What the Save song render sends in place of the job's own score and
 *  lyrics: the spine's whole codec stream under the arranged prompt it was
 *  composed with. */
export function buildSessionRender(id: string, userId: string):
  Pick<Yue2SynthRequest, 'codec_ids' | 'abc' | 'lyrics' | 'lyric_schedule'> & { trimStart?: number } {
  const stored = load(id, userId);
  const spine = stored.kept.at(-1);
  if (!spine) throw new TakesError('Keep a take before saving the song.');
  const { request } = arranged(stored);
  return { codec_ids: JSON.parse(fs.readFileSync(codesFile(id, spine), 'utf8')) as number[],
    abc: request.abc, lyrics: request.lyrics, lyric_schedule: request.lyric_schedule, trimStart: trimStart(stored) };
}

/** Renders everything kept so far as one audio file, for Play build: the
 *  spine's whole codec stream prefilled under the arranged prompt, as Save
 *  song renders it but without post-processing or a library song. Returns at
 *  once; the session's `preview` tracks it. */
export function renderBuildPreview(id: string, userId: string): Yue2BuildSession {
  const stored = load(id, userId);
  const spine = stored.kept.at(-1);
  if (!spine) throw new TakesError('Keep a take before rendering the build.');
  if (stored.pending.length) throw new TakesError('Wait for the pending takes or cancel them first.', 409);
  if (previewRuns.has(id)) return view(stored);
  const codes = JSON.parse(fs.readFileSync(codesFile(id, spine), 'utf8')) as number[];
  const { request } = arranged(stored);
  stored.preview = { spine, status: 'rendering', progress: 0 };
  save(stored);
  previewRuns.add(id);
  const finish = (preview: NonNullable<Yue2BuildSession['preview']>) => {
    const fresh = load(id, userId);
    const old = fresh.preview?.audioUrl;
    fresh.preview = preview;
    save(fresh);
    if (old && old !== preview.audioUrl) {
      try { fs.unlinkSync(path.join(sessionDir(id), path.basename(old))); } catch { /* already gone */ }
    }
  };
  void (async () => {
    try {
      await runOnGpuLane(async lease => {
        await evictOtherFamilies('yue2');
        const req: Yue2SynthRequest = { ...request, codec_ids: codes, synth_batch_size: 1 };
        delete req.lm_batch_size;
        delete req.songs;
        delete req.semantic_retries;
        delete req.preview_max_frames;
        const sub = await yue2Synth(req);
        const run: BuildRun = { cancelled: false, jobId: sub.job_id };
        try {
          await waitJob(sub.job_id, run, lease, (_phase, progress) => {
            const fresh = load(id, userId);
            if (fresh.preview?.status === 'rendering') { fresh.preview.progress = progress; save(fresh); }
          });
          const body = await aceClient.getJobResult(sub.job_id);
          if (!body.ok) throw new Error(`YuE2 build render result fetch failed (${body.status})`);
          const audio = Buffer.from(await body.arrayBuffer());
          const contentType = body.headers.get('content-type') ?? 'audio/wav';
          const wav = contentType.startsWith('multipart/mixed') ? splitMultipartMixed(audio, contentType)[0] : audio;
          const file = `build-${spine}-${randomUUID().slice(0, 6)}.wav`;
          fs.writeFileSync(path.join(sessionDir(id), file), wav);
          finish({ spine, status: 'done', audioUrl: `/audio/takes/${id}/${file}` });
        } catch (err) {
          await aceClient.cancelJob(sub.job_id).catch(() => {});
          throw err;
        }
      }, { label: 'yue2 build: render so far', family: 'yue2' });
    } catch (err) {
      finish({ spine, status: 'failed', error: (err as Error).message });
    } finally {
      previewRuns.delete(id);
    }
  })();
  return view(load(id, userId));
}

const MAX_OCCURRENCES = 100;

/** Replaces the occurrences not yet built (coverArrangement's editPlan):
 *  `upcoming` lists source section ids (s0...) for new occurrences, or
 *  existing upcoming occurrence ids (o3...) to keep their identity, and
 *  `finished` says the plan ends the song. Kept occurrences stay. A take not
 *  yet kept stays keepable only when its own sections are untouched (carryTakes). */
export function editBuildPlan(id: string, userId: string, revision: unknown, upcoming: unknown, finished: unknown): Yue2BuildSession {
  const stored = load(id, userId);
  checkRevision(stored, revision);
  if (stored.pending.length) throw new TakesError('Wait for the pending takes or cancel them first.', 409);
  if (!Array.isArray(upcoming) || upcoming.some(x => typeof x !== 'string') || typeof finished !== 'boolean') {
    throw new TakesError('upcoming must list section or occurrence ids, and finished must be true or false.');
  }
  let arrangement;
  try { arrangement = editPlan(stored, upcoming as string[], finished); }
  catch (err) { throw new TakesError((err as Error).message, 409); }
  if (arrangement.plan.length > MAX_OCCURRENCES) throw new TakesError(`An arrangement holds at most ${MAX_OCCURRENCES} sections.`);
  const same = (a: typeof arrangement) => JSON.stringify([a.plan, a.finished]);
  if (same(arrangement) === same(stored.arrangement)) return view(stored);
  arranged({ request: stored.request, arrangement });
  const before = stored.revision;
  stored.arrangement = arrangement;
  stored.revision++;
  carryTakes(stored, before);
  save(stored);
  return view(stored);
}

/** Takes made at revision `before` whose own sections a plan edit left alone
 *  stay keepable: keeping one first and editing the plan after gives the same
 *  build. A take that composed an ending stays only while the song still ends
 *  after it. */
export function carryTakes(stored: Pick<ArrangedSession, 'takes' | 'kept' | 'arrangement' | 'revision'>, before: number): void {
  const from = nextOccurrence(stored);
  const parent = stored.kept.at(-1) ?? null;
  const { plan, finished } = stored.arrangement;
  for (const take of stored.takes) {
    if (take.revision !== before || take.parent !== parent || take.from !== from) continue;
    const want = plan.slice(from, take.to + 1).map(o => o.id);
    if (want.length !== take.occurrences.length || want.some((id, i) => id !== take.occurrences[i])) continue;
    const ending = take.starts.length <= take.to + 1;
    if (!ending || (finished && plan.length === take.to + 1)) take.revision = stored.revision;
  }
}

/** Why `lyrics` cannot be sung by a build, or null. Text above the first tag
 *  and malformed tag lines would be sung as words or dropped silently. */
export function lyricsProblem(lyrics: unknown): string | null {
  if (typeof lyrics !== 'string') return 'Lyrics must be text.';
  if (lyrics.length > MAX_LYRICS) return `Lyrics are limited to ${MAX_LYRICS.toLocaleString('en-US')} characters.`;
  if (!lyrics.trim()) return 'Lyrics cannot be empty.';
  if (/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(lyrics)) return 'Lyrics contain control characters.';
  const lines = lyrics.replace(/\r\n?/g, '\n').split('\n');
  const bad = lines.findIndex(line => { const t = line.trim(); return t.startsWith('[') && !/^\[[^\]\n]*\S[^\]\n]*\]$/.test(t); });
  if (bad >= 0) return `Line ${bad + 1} looks like a section tag but is not one; use [verse], [chorus] and so on.`;
  const blocks = splitLyricBlocks(lyrics);
  if (!blocks.some(b => b.tag)) return 'Lyrics need section tags such as [verse].';
  if (blocks[0].tag === null) return 'Put a section tag above the first line of lyrics.';
  if (blocks.some(b => (b.tag?.length ?? 0) > 40)) return 'A section tag is longer than 40 characters.';
  return null;
}

/** Why the lyrics of the sections already built cannot change, or null: the
 *  kept audio was composed under them. Later text is free. */
function builtLyricsRefusal(stored: Stored, lyrics: string): string | null {
  const built = nextOccurrence(stored);
  if (!built) return null;
  const abc = stored.request.abc ?? '', old = stored.request.lyrics ?? '';
  try {
    // The kept occurrences, in plan order: same text and same lyric spans.
    const plan = planSections(stored.arrangement).slice(0, built), own = planLyrics(stored.arrangement).slice(0, built);
    const kept = (text: string) => { const c = composeArrangement(abc, text, plan, own); return JSON.stringify([c.lyrics, c.occurrences.map(o => o.lyric)]); };
    const same = kept(old) === kept(lyrics);
    return same ? null : 'The lyrics of built sections are locked; undo back to a section to change its lyrics.';
  } catch { return 'These lyrics do not fit the score; keep the section tags the score expects.'; }
}

/** Replaces the lyrics of the next takes. `toRequest` rebuilds the engine
 *  request from the new params the way a render would, so the stored params
 *  (what Save song submits) and the request agree. Kept audio stays; every
 *  take not yet kept becomes ineligible through the revision. Nothing here
 *  touches a dataset. */
export function applyBuildLyrics(id: string, userId: string, revision: unknown, lyrics: unknown,
  toRequest: (params: Record<string, unknown>) => Yue2SynthRequest): Yue2BuildSession {
  const stored = load(id, userId);
  checkRevision(stored, revision);
  if (stored.pending.length) throw new TakesError('Wait for the pending takes or cancel them first.', 409);
  if (stored.params.instrumental === true) throw new TakesError('An instrumental build has no lyrics.');
  const problem = lyricsProblem(lyrics);
  if (problem) throw new TakesError(problem);
  const text = (lyrics as string).replace(/\r\n?/g, '\n');
  const params = { ...stored.params, lyrics: text };
  const request = plainRequest(toRequest(params));
  if (request.abc !== (stored.baseAbc ?? stored.request.abc)) throw new TakesError('The score changed; start a new build.', 409);
  if (request.lyrics === stored.request.lyrics && text === stored.params.lyrics) return view(stored);
  const locked = builtLyricsRefusal(stored, request.lyrics ?? '');
  if (locked) throw new TakesError(locked, 409);
  stored.params = params;
  stored.baseAbc ??= stored.request.abc;
  stored.request = { ...request, abc: stored.request.abc };
  stored.revision++;
  // New lyrics for the whole build replace the unbuilt occurrences' own too.
  const built = nextOccurrence(stored);
  stored.arrangement = { ...stored.arrangement, sections: sourceSections(stored.request.abc!, request.lyrics ?? ''),
    plan: stored.arrangement.plan.map((o, i) => (i < built ? o : (({ lyrics: _, ...rest }) => rest)(o))) };
  save(stored);
  return view(stored);
}

/** Gives unbuilt occurrences their own sung lines: `edits` maps occurrence
 *  ids to the lines (the block's tag stays) or null to sing the build's
 *  lyrics again. A repeated section can then sing other words than an
 *  earlier, built copy. Every take not yet kept becomes ineligible through
 *  the revision; nothing changes, nothing moves. */
export function setOccurrenceLyrics(id: string, userId: string, revision: unknown, edits: unknown): Yue2BuildSession {
  const stored = load(id, userId);
  checkRevision(stored, revision);
  if (stored.pending.length) throw new TakesError('Wait for the pending takes or cancel them first.', 409);
  if (stored.params.instrumental === true) throw new TakesError('An instrumental build has no lyrics.');
  if (!edits || typeof edits !== 'object' || Array.isArray(edits)) throw new TakesError('edits must map occurrence ids to lyrics.');
  const built = nextOccurrence(stored);
  const plan = stored.arrangement.plan.map(o => ({ ...o }));
  for (const [occurrence, value] of Object.entries(edits as Record<string, unknown>)) {
    const at = plan.findIndex(o => o.id === occurrence);
    if (at < 0) throw new TakesError(`No such occurrence: ${occurrence}`);
    const name = stored.arrangement.sections.find(x => x.id === plan[at].section);
    if (at < built) throw new TakesError(`${name?.label ?? 'That section'} is built; undo back to it to change its lyrics.`, 409);
    if (value === null) { delete plan[at].lyrics; continue; }
    if (typeof value !== 'string') throw new TakesError('Lyrics must be text.');
    if (name?.lyricBlock === null || name?.lyricBlock === undefined) throw new TakesError(`${name?.label ?? 'That section'} is instrumental here; it has no lyrics to change.`);
    const text = value.replace(/\r\n?/g, '\n').trim();
    if (!text) throw new TakesError(`${name.label} needs words; reset it to sing the build's.`);
    if (text.length > MAX_LYRICS) throw new TakesError(`Lyrics are limited to ${MAX_LYRICS.toLocaleString('en-US')} characters.`);
    if (/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(text)) throw new TakesError('Lyrics contain control characters.');
    if (text.split('\n').some(line => line.trim().startsWith('['))) throw new TakesError(`${name.label}'s lyrics cannot hold a section tag; edit one section at a time.`);
    plan[at].lyrics = text;
  }
  if (JSON.stringify(plan) === JSON.stringify(stored.arrangement.plan)) return view(stored);
  const arrangement = { ...stored.arrangement, plan };
  arranged({ request: stored.request, arrangement });
  stored.arrangement = arrangement;
  stored.revision++;
  save(stored);
  return view(stored);
}

/** Replaces a build's settings before its first keep. A cover build takes voices, tempo,
 *  key (with its keyShift), condition strength and AR strength, as the
 *  one-shot cover params carry them. `toRequest` rebuilds the request the way
 *  a render would and validates the params, so params (what Save song
 *  submits) and request agree. A rewritten score keeps the plan when its
 *  sections match, else the plan returns to the score's order. Every take not
 *  yet kept becomes ineligible through the revision. After a keep the kept
 *  audio was composed under these settings: a change needs a fresh build. */
export function applyBuildSettings(id: string, userId: string, revision: unknown, settings: unknown,
  toRequest: (params: Record<string, unknown>) => Yue2SynthRequest): Yue2BuildSession {
  const stored = load(id, userId);
  checkRevision(stored, revision);
  if (stored.pending.length) throw new TakesError('Wait for the pending takes or cancel them first.', 409);
  if (stored.kept.length) throw new TakesError('The kept sections were built with these settings; start a fresh build to change them.', 409);
  const cover = stored.params.yue2Cover;
  const isCover = !!cover && typeof cover === 'object';
  if (!settings || typeof settings !== 'object' || Array.isArray(settings)) throw new TakesError('settings are required.');
  const { voices, tempo, key, keyShift, cfgScale, arStrength } = settings as Record<string, unknown>;
  if (typeof arStrength !== 'number' || !Number.isFinite(arStrength) || arStrength < 0 || arStrength > 10) {
    throw new TakesError('AR strength must be a number from 0 to 10.');
  }
  const pick = { ...(stored.params.yue2Pick as Record<string, unknown> | undefined), lmAdapterArScale: arStrength };
  let params: Record<string, unknown>;
  if (isCover) {
    const nextCover: Record<string, unknown> = { ...(cover as Record<string, unknown>), voices, tempo, key, cfgScale };
    if (keyShift === undefined) delete nextCover.keyShift; else nextCover.keyShift = keyShift;
    params = { ...stored.params, yue2Cover: nextCover, yue2CfgScale: cfgScale, yue2Pick: pick };
  } else {
    // A Create build has no cover transforms: only condition strength and AR strength.
    if (typeof cfgScale !== 'number' || !Number.isFinite(cfgScale) || cfgScale <= 0 || cfgScale > 25) {
      throw new TakesError('Condition strength must be a number above 0, up to 25.');
    }
    params = { ...stored.params, yue2CfgScale: cfgScale, yue2Pick: pick };
  }
  if (JSON.stringify(params) === JSON.stringify(stored.params)) return view(stored);
  const request = plainRequest(toRequest(params));
  const sections = scoreSections(request.abc ?? '').map(s => ({ label: s.label.trim(), startBar: s.startBar }));
  if (!sections.length) throw new TakesError('The score has no section labels.');
  const source = sourceSections(request.abc!, request.lyrics ?? '');
  let arrangement = stored.arrangement;
  if (JSON.stringify(source) !== JSON.stringify(arrangement.sections)) {
    const n = arrangement.nextOccurrence;
    arrangement = { sections: source, plan: source.map((s, i) => ({ id: `o${n + i}`, section: s.id })), nextOccurrence: n + source.length, finished: true };
  }
  arranged({ request, arrangement });
  Object.assign(stored, { params, request, baseAbc: request.abc, sections, arrangement, revision: stored.revision + 1 });
  save(stored);
  return view(stored);
}
