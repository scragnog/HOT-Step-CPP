// What the read-outs show, worked out from the values they read (docs/09, package 2): the beat lamps at the BPM, the
// re-planning dice, MiniMax-Music3's flow windows and their count, the speed a run gains from its trades, the plan
// strength curve with its Plan CFG band and CADS ramp, the plan the planner starts from, the lyric timestamps' preview
// and the cover's colour. Each is the concept's own formula (25-module-tempo.js pulseHTML, 25-module-replan.js,
// 25-module-flow.js flowViz and flowCap, 25-module-perf.js, 25-module-adhere.js adhereSVG, 25-module-planner.js
// planText, 28-side-units.js), moved here so the view draws numbers and words it is given. None has a branch for any
// module or engine: what differs per engine is said in the definitions (a node's reads and options, per engine).

import type { Node, Value } from '../definitions/types.ts';
import { type Desk, type Index, has, isAuto, mmss, valueOf, wordsOf } from './core.ts';
import { type Msg, type Slot, list, msg } from './messages.ts';
import { textMsg } from './strings.ts';

/** A plan line: its key (printed in a column `pad` characters wide) and its value, in the terminal's colour for whose
 *  it is: v the user's, y the planner's pick, k nothing yet. */
export interface PlanLine { key: Slot; value: Slot; tone: 'v' | 'y' | 'k' }

export interface ReadoutModel {
  /** beatPulse: one lamp a beat of the bar, a beat's length as CSS seconds, and whether the lamps rest (on auto). */
  pulse?: { beats: number; beat: string; idle: boolean };
  /** dice: one die per allowed attempt, drawn off while the switch that allows them is off, and those used. */
  dice?: { n: number; off: boolean; used: number; tip: Slot };
  /** flowWindows: each window's start and width along the span shown (0 to 1) and its lane, the axis words, and the
   *  caption's two halves. */
  flow?: { windows: { x: number; w: number; lane: number }[]; axis: string[]; count: number; compare: Slot; seam: Slot };
  /** speedGauge: how much faster, as it is printed. */
  speed?: { text: string };
  /** adhereCurve: the curve's points (in a 300 x 110 box), the Plan CFG band and the CADS ramp when they are on, and the
   *  legend's words. */
  curve?: {
    points: string; grid: number[]; unity: number;
    band: { x: number; width: number; opacity: number } | null;
    cads: { width: number; opacity: number; stop: number } | null;
    legend: { cls: 'l-ps' | 'l-pc' | 'l-cd'; words: Slot }[];
  };
  /** planText: the command after "$" (and words after it in the terminal's quiet ink: no tags yet), its lines, or one
   *  note in their place (the planner off). */
  plan?: { command: Slot; after: Slot | null; lines: PlanLine[]; note: Slot | null; pad: number };
  /** lrcPreview: its two timed lines. */
  lrc?: { lines: Slot[] };
  /** coverSwatch: its hue. */
  cover?: { hue: number };
}

/* ---------------------------------------------------------------- reading what a node reads */

const read = (ix: Index, desk: Desk, k: string | undefined): Value | null => (k && ix.params.has(k) && has(ix, k, desk.engine) ? valueOf(ix, k, desk) ?? null : null);
const num = (ix: Index, desk: Desk, k: string | undefined, d = 0): number => { const v = read(ix, desk, k); return typeof v === 'number' ? v : d; };

/** The sections of a lyric text, in order: a [Tag] line starts one; words before any tag are a verse
 *  (25-module-lyrics.js sections). */
export function sectionNames(text: string): string[] {
  const out: string[] = []; let open = false;
  for (const l of text.split('\n')) {
    const m = /^\[([^\]]+)\]/.exec(l.trim());
    if (m) { out.push(m[1]!.trim()); open = true; } else if (l.trim() && !open) { out.push('Verse'); open = true; }
  }
  return out;
}

/** A key as a musician spells it: the flat side where it is read so (D♭ major, B♭ minor), with true sharps and flats
 *  (25-module-tempo.js pretty). The key wheel's package spells its segments the same way. */
export function spellKey(s: string): string {
  const FLAT_MAJ: Record<string, string> = { 'C#': 'Db', 'G#': 'Ab', 'D#': 'Eb', 'A#': 'Bb' }, FLAT_MIN: Record<string, string> = { 'D#': 'Eb', 'A#': 'Bb' };
  return s.replace(/^([A-G]#?)( major| minor)$/, (_m, n: string, q: string) => ((q === ' major' ? FLAT_MAJ : FLAT_MIN)[n] ?? n) + q).replace(/#/g, '♯').replace(/([A-G])b/g, '$1♭');
}

/* ---------------------------------------------------------------- each read-out */

/** The beat lamps: as many as the bar has beats (the time signature's top, 4 on auto), pulsing a beat apart at the
 *  BPM; on auto they rest (the planner picks the tempo). Reads the BPM, then the time signature. */
function pulse(ix: Index, node: Node, desk: Desk): NonNullable<ReadoutModel['pulse']> {
  const [bpmKey, sigKey] = node.reads ?? [];
  const bpm = bpmKey && ix.params.has(bpmKey) && has(ix, bpmKey, desk.engine) && !isAuto(ix, bpmKey, desk) ? num(ix, desk, bpmKey) : 0;
  const top = Number(String(read(ix, desk, sigKey) ?? '').split('/')[0]) || 4;
  return { beats: top, beat: bpm ? `${(60 / bpm).toFixed(3)}s` : '0s', idle: !bpm };
}

/** The dice: as many as the attempts it reads; off while the switch named by `offWhen` is off. Those used are the tries
 *  the run that renders reports (desk.songs, told by the generation service). */
function dice(ix: Index, node: Node, desk: Desk): NonNullable<ReadoutModel['dice']> {
  const off = (node.opts as { offWhen?: string } | undefined)?.offWhen;
  return { n: Math.min(50, Math.max(0, Math.round(num(ix, desk, node.reads?.[0])))), off: !!off && read(ix, desk, off) === false, used: Math.min(desk.songs?.attempts ?? 0, 50), tip: msg('dice.tip') };
}

/** MiniMax-Music3's flow windows: an 8-second window every hop (frames at 25 a second) over the first 48 s, in two
 *  lanes; and the count a 3:00 song needs against the checkpoint's own 4 s hop, and the overlap at each seam. */
function flow(ix: Index, node: Node, desk: Desk): NonNullable<ReadoutModel['flow']> {
  const key = node.reads?.[0], frames = num(ix, desk, key, 100), hs = Math.max(1, frames) / 25, span = 48, win = 8;
  const n = Math.floor((span - win) / hs + 1e-6) + 1;
  const count = Math.ceil((180 - win) / hs) + 1, ref = Math.ceil((180 - win) / 4) + 1, d = Math.round((1 - count / ref) * 100);
  return {
    windows: Array.from({ length: n }, (_, i) => ({ x: +(i * hs / span).toFixed(4), w: +(win / span).toFixed(4), lane: i % 2 })),
    axis: ['0 s', `${span / 2} s`, `${span} s`], count,
    compare: count === ref ? msg('flow.same', { n: count }) : msg('flow.compare', { n: count, pct: Math.abs(d), more: d > 0 ? 'fewer' : 'more', ref }),
    seam: msg(frames > 150 ? 'flow.seam.listen' : 'flow.seam', { s: Math.max(0, win - hs).toFixed(1) }),
  };
}

/** How much faster the trades it reads make a run (CFG cutoff, step cache, LM CFG cutoff, in that order). */
function speed(ix: Index, node: Node, desk: Desk): NonNullable<ReadoutModel['speed']> {
  const [cut, cache, lm] = node.reads ?? [], sp = (1 - num(ix, desk, cut, 1)) * 40 + num(ix, desk, cache) * 55 + (1 - num(ix, desk, lm, 1)) * 15;
  return { text: sp < 1 ? '0%' : `+${Math.round(sp)}%` };
}

/** Plan strength across the run, noise on the left and clean on the right, with the Plan CFG band where it acts and the
 *  CADS ramp where it adds noise. Reads start, end, curve, Plan CFG, its from and to, CADS, its clean-below and
 *  full-above, in that order. */
function curve(ix: Index, node: Node, desk: Desk): NonNullable<ReadoutModel['curve']> {
  const [ks, ke, kc, kp, kp1, kp2, kd, kd1, kd2] = node.reads ?? [];
  const L = 8, R = 292, T = 8, B = 88, X = (t: number) => L + (R - L) * (1 - t), Y = (v: number) => B - (B - T) * v / 2;
  const s0 = num(ix, desk, ks, 1), s1 = num(ix, desk, ke, 1);
  /* the curve's second option eases (cosine); the first is a straight line */
  const opts = kc && ix.params.has(kc) ? wordsOf(ix, ix.params.get(kc)!.p, desk).options : undefined;
  const eased = Array.isArray(opts) && opts.length > 1 && read(ix, desk, kc) === opts[1]!.value;
  const points = Array.from({ length: 41 }, (_, i) => { const u = i / 40, k = eased ? (1 - Math.cos(Math.PI * u)) / 2 : u; return `${(L + (R - L) * u).toFixed(1)},${Y(s0 + (s1 - s0) * k).toFixed(1)}`; }).join(' ');
  const pc = num(ix, desk, kp, 1), t1 = num(ix, desk, kp1), t2 = num(ix, desk, kp2, 1), cs = num(ix, desk, kd), c1 = num(ix, desk, kd1), c2 = num(ix, desk, kd2);
  const legend: NonNullable<ReadoutModel['curve']>['legend'] = [{ cls: 'l-ps', words: msg('curve.strength', { a: s0.toFixed(2), b: s1.toFixed(2) }) }];
  if (pc > 1) legend.push({ cls: 'l-pc', words: msg('curve.cfg', { v: pc.toFixed(1) }) });
  if (cs > 0) legend.push({ cls: 'l-cd', words: msg('curve.cads', { v: cs.toFixed(2) }) });
  return {
    points, grid: Array.from({ length: 11 }, (_, i) => L + (R - L) * i / 10), unity: Y(1),
    band: pc > 1 ? { x: X(Math.max(t1, t2)), width: Math.abs(X(t1) - X(t2)), opacity: +(0.45 + (pc - 1) / 3 * 0.55).toFixed(3) } : null,
    cads: cs > 0 ? { width: X(c1) - L, opacity: +(0.4 + 0.6 * cs).toFixed(3), stop: +Math.min(1, Math.max(0, (1 - c2) / Math.max(0.0001, 1 - c1))).toFixed(3) } : null,
    legend,
  };
}

/** The caption as the planner gets it: on one line, a long one cut at a word with an ellipsis (B-16). */
const captionOf = (text: string): string => { const raw = text.trim().replace(/\n/g, ' '); return raw.length > 64 ? raw.slice(0, 63).replace(/\s+\S*$/, '') + '…' : raw; };

/** The plan the planner will start from, by its look: `plan` (caption, tempo, key and form, each the user's or the
 *  planner's pick; with its switch off, a note), `prose` (tempo and key come from the caption's prose), `compose` (the
 *  chain of thought writes a score) or `tags` (the tags, the lyrics' sections, the length). Reads the caption, the
 *  lyrics, then by look: plan the BPM, the key, the planner's switch and chain-of-thought switch; compose the chain of
 *  thought's mode; tags the length. */
function plan(ix: Index, node: Node, desk: Desk): NonNullable<ReadoutModel['plan']> {
  const look = String((node.opts as { look?: string } | undefined)?.look ?? 'plan'), r = node.reads ?? [];
  const cap = captionOf(String(read(ix, desk, r[0]) ?? '')), secs = sectionNames(String(read(ix, desk, r[1]) ?? ''));
  const capLine: PlanLine = cap ? { key: msg('plan.caption'), value: cap, tone: 'v' } : { key: msg('plan.caption'), value: msg('plan.none'), tone: 'k' };
  const form = (none: string): PlanLine => (secs.length ? { key: msg('plan.form'), value: list(secs, 'dot'), tone: 'v' } : { key: msg('plan.form'), value: msg(none), tone: 'y' });
  const out = (command: Msg, lines: PlanLine[], note: Slot | null = null, after: Slot | null = null) => ({ command, after, lines, note, pad: 9 });
  if (look === 'compose') {
    const k = r[2], opts = k && ix.params.has(k) ? wordsOf(ix, ix.params.get(k)!.p, desk).options : undefined, i = Array.isArray(opts) ? opts.findIndex(o => o.value === read(ix, desk, k)) : 0;
    return out(msg('plan.compose', { mode: k && has(ix, k, desk.engine) ? textMsg(ix, k, desk) : '' }), [capLine, { key: msg('plan.score'), value: msg(`plan.score.${Math.max(0, i)}`), tone: 'v' }, form('plan.form.composer')]);
  }
  if (look === 'tags') {
    const k = r[2], range = k && ix.params.has(k) ? wordsOf(ix, ix.params.get(k)!.p, desk).range : undefined;
    const length: PlanLine = k && has(ix, k, desk.engine) && !isAuto(ix, k, desk) ? { key: msg('plan.length'), value: msg('plan.upTo', { v: mmss(num(ix, desk, k)) }), tone: 'y' } : { key: msg('plan.length'), value: msg('plan.modelEnds', { v: mmss(range?.[1] ?? 240) }), tone: 'y' };
    return out(cap ? msg('plan.tags', { cap }) : msg('plan.tags.none'), [{ key: msg('plan.lyrics'), value: secs.length ? list(secs, 'dot') : msg('plan.instrumental'), tone: secs.length ? 'v' : 'y' }, length], null, cap ? null : msg('plan.none'));
  }
  if (look === 'prose') return out(msg('plan.model'), [capLine, { key: msg('plan.tempo'), value: msg('plan.prose'), tone: 'y' }, { key: msg('plan.key'), value: msg('plan.prose'), tone: 'y' }, form('plan.form.planner')]);
  const [, , kb, kk, kon, kcot] = r;
  if (kon && read(ix, desk, kon) === false) return out(msg('plan.off'), [], msg('plan.off.note'));
  const yoursBpm = !!kb && has(ix, kb, desk.engine) && !isAuto(ix, kb, desk), key = String(read(ix, desk, kk) ?? '');
  return out(msg(read(ix, desk, kcot) === true ? 'plan.cot' : 'plan.plain'), [
    capLine,
    yoursBpm ? { key: msg('plan.tempo'), value: msg('plan.bpm.yours', { v: num(ix, desk, kb) }), tone: 'v' } : { key: msg('plan.tempo'), value: msg('plan.bpm.pick', { v: 90 + (cap.length * 7) % 40 }), tone: 'y' },
    key ? { key: msg('plan.key'), value: msg('plan.key.yours', { v: spellKey(key) }), tone: 'v' } : { key: msg('plan.key'), value: msg('plan.key.pick', { v: spellKey('D major') }), tone: 'y' },
    form('plan.form.planner'),
  ]);
}

/** The lyric timestamps' preview: the first section of the lyrics it reads (a verse when there is none), then a chorus. */
function lrc(ix: Index, node: Node, desk: Desk): NonNullable<ReadoutModel['lrc']> {
  const first = sectionNames(String(read(ix, desk, node.reads?.[0]) ?? ''))[0] ?? 'verse';
  return { lines: [msg('lrc.line', { t: '00:12.40', name: first }), msg('lrc.line', { t: '00:18.95', name: msg('lrc.chorus') })] };
}

/** The cover swatch's hue: the newest song's colour, which the generation package's list gives (`newest`); until a
 *  song is listed, the concept's own violet. */
export const coverHue = (newest?: { hue: number } | null): number => newest?.hue ?? 300;

/** The extra model of a read-out's node, or undefined for a part that needs none. */
export function readoutModel(ix: Index, node: Node, desk: Desk): ReadoutModel | undefined {
  switch (node.part) {
    case 'beatPulse': return { pulse: pulse(ix, node, desk) };
    case 'dice': return { dice: dice(ix, node, desk) };
    case 'flowWindows': return { flow: flow(ix, node, desk) };
    case 'speedGauge': return { speed: speed(ix, node, desk) };
    case 'adhereCurve': return { curve: curve(ix, node, desk) };
    case 'planText': return { plan: plan(ix, node, desk) };
    case 'lrcPreview': return { lrc: lrc(ix, node, desk) };
    case 'coverSwatch': return { cover: { hue: coverHue(desk.songs?.hue === undefined ? null : { hue: desk.songs.hue }) } };
  }
  return undefined;
}
