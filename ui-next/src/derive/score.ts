// The lead sheet (YuE2's Lead sheet tile, the concept's 25-module-score.js and 34-score-modal.js): the ABC foot under
// the staff paper, and the score preview. What the foot offers, whether the text reads as ABC, what the preview asks the
// planner for, and the preview's drawing (where every line, note, mark and playhead goes, its words, the notes it
// plays) are worked out here; the score itself is the planner's (services/score.ts plans one, the mock beside the
// real planner). Nothing here has a branch for a module: the foot's node says what it reads.
//
//   abcFoot (instrumentModel)   InstrumentModel.abc: which of its node's actions apply, the preview's words, its note
//   scoreAsk(ix, desk)          what the preview shows: the ABC on the sheet, or what to ask the planner for
//   abcInfo(abc)                a pasted lead sheet read from its own header
//   scoreLayout(score)          where the staff's systems, bars and notes go (shared by the drawing and the playhead)
//   scoreModel(score)           the preview's words and drawing

import type { Node } from '../definitions/types.ts';
import { type Desk, type Index, applies, has, mmss, valueOf } from './core.ts';
import { EN, type Slot, list, msg } from './messages.ts';
import { sectionNames } from './readouts.ts';
import { sectionColour } from './build.ts';
import { defineInstrument } from './instruments.ts';

Object.assign(EN, {
  'abc.lines': '{n, plural, one {# line} other {# lines}}. The composer follows this instead of planning one.',
  'abc.not': "This doesn't read as ABC (no key line, no bars). The planner will write its own.",
  'abc.none': 'The Planner is set to write nothing down.',
  'abc.preview': 'Preview this score',
  'abc.plan': 'Plan and preview a score',
  'score.untitled': 'Untitled',
  'score.meta': 'seed {seed} · {len} · {n} sections',
  'score.len.loops': 'plays {t}, then loops',
  'score.v.healthy': 'healthy',
  'score.v.long': 'long',
  'score.v.runaway': 'runaway',
  'score.why.healthy': 'Every section has a vocal line and the plan ends where the lyrics do.',
  'score.why.long': 'The plan runs {n} bars past the lyrics (shaded). It will render, but the ending may drag.',
  'score.why.runaway': 'The plan loops its last section and never closes. Retry draws a new one.',
  'score.order': 'Sections: play from one',
  'score.from': 'Play from {name}',
  'score.staff': 'Lead sheet, {n} sections in G major, 4/4, {bars} bars',
  'score.again': '{name} again',
  'score.past': 'past the lyrics',
  'score.more': 'and again …',
  'score.key': 'Key {key}',
  'score.bars': '{n, plural, one {# bar} other {# bars}}',
  'score.pastedOn': 'pasted on the lead sheet',
  'score.continue': 'Rendering this score with its seed, {seed}. Your seed setting is unchanged.',
  'score.saved': 'Saved the lead sheet',
  'score.noAudio': "This browser can't play audio here",
});

/* ---------------------------------------------------------------- the ABC foot */

export interface AbcModel {
  /** Its node's actions that apply now, by their place, and what each is: Insert example (with the text it fills),
   *  the preview, the way back to a score. */
  show: { at: number; kind: 'example' | 'preview' | 'back'; key?: string }[];
  /** What the preview's button says: the score on the sheet, or one to plan. */
  preview: Slot;
  note: { text: Slot; warn: boolean } | null;
}
const textOf = (ix: Index, desk: Desk, k: string | undefined) => (k && ix.params.has(k) && has(ix, k, desk.engine) ? String(valueOf(ix, k, desk) ?? '') : '');
/** A text reads as ABC with a key line and bar lines (E-26). */
export const readsAsAbc = (abc: string): boolean => /^K:/m.test(abc) && /\|/.test(abc);

function abcOf(ix: Index, node: Node, desk: Desk): AbcModel {
  const [abcKey] = node.reads ?? [], abc = textOf(ix, desk, abcKey), n = abc.trim() ? abc.trim().split('\n').length : 0;
  /* with Chain of Thought off the planner writes no score, so there is none to plan and preview: the lead sheet's text
     does not apply, and the foot offers the way back (E-12) */
  const none = !!abcKey && ix.params.has(abcKey) && !applies(ix, abcKey, desk);
  const acts = node.actions ?? [], KIND = { example: 'example', bespoke: 'preview', set: 'back' } as const;
  const at = (d: keyof typeof KIND) => acts.flatMap((a, i) => (a.do === d ? [{ at: i, kind: KIND[d], ...('param' in a && d === 'example' ? { key: a.param } : {}) }] : []));
  if (!n && none) return { show: [...at('example'), ...at('set')], preview: msg('abc.plan'), note: { text: msg('abc.none'), warn: false } };
  const ok = readsAsAbc(abc);
  return {
    show: [...(n ? [] : at('example')), ...at('bespoke')], preview: msg(n ? 'abc.preview' : 'abc.plan'),
    note: n ? { text: ok ? msg('abc.lines', { n }) : msg('abc.not'), warn: !ok } : null,
  };
}
/* abcFoot: what the foot offers (InstrumentModel's field, added where it is drawn) */
declare module './instruments.ts' {
  interface InstrumentModel {
    abc?: AbcModel;
  }
}

defineInstrument('abcFoot', (ix, node, desk) => ({ abc: abcOf(ix, node, desk) }));

/* ---------------------------------------------------------------- what the preview shows */

/** What the preview shows on this desk: the ABC on the lead sheet as written, or what to ask the planner for (the
 *  sections the lyrics hold, the title as Generate names a song, whether the plan carries chords), read by the ABC
 *  foot's node: the sheet's text, the plan mode, the lyrics, the prompt and the song's title. Null with no lead sheet
 *  on the desk. */
export interface ScoreAsk { abc: string; names: string[]; title: string; chords: boolean }
export function scoreAsk(ix: Index, desk: Desk): ScoreAsk | null {
  const d = ix.defs.find(x => desk.on.has(x.id) && x.engines.includes(desk.engine) && (x.face ?? []).some(n => n.part === 'abcFoot'));
  const node = d?.face?.find(n => n.part === 'abcFoot');
  if (!node) return null;
  const [abcK, modeK, lyricsK, promptK, titleK] = node.reads ?? [];
  const mode = modeK && ix.params.has(modeK) && has(ix, modeK, desk.engine) ? String(valueOf(ix, modeK, desk)) : '';
  const t = (textOf(ix, desk, titleK).trim() || textOf(ix, desk, promptK).trim() || '').split(/[,.\n]/)[0]!.trim().slice(0, 40);
  return {
    abc: textOf(ix, desk, abcK).trim(), names: sectionNames(textOf(ix, desk, lyricsK)), title: t ? t[0]!.toUpperCase() + t.slice(1) : '',
    chords: mode === String((node.opts as { chords?: string } | undefined)?.chords ?? ''),
  };
}

/** A pasted lead sheet, read from its own header (G-B-02): its title, key, metre and bars. */
export function abcInfo(abc: string): { title: string; meta: Slot; file: string } {
  const h = (k: string) => (new RegExp(`^${k}:\\s*(.+)$`, 'm').exec(abc) ?? [])[1]?.trim();
  const body = abc.split('\n').filter(l => l.trim() && !/^[A-Za-z]:/.test(l) && !/^%/.test(l));
  const bars = body.reduce((a, l) => a + (l.replace(/\|\]|\|\||:\||\|:/g, '|').match(/\|/g) ?? []).length, 0), key = h('K'), metre = h('M'), title = h('T') ?? '';
  return {
    title, file: abcFile(title || 'lead-sheet'),
    meta: list([...(key ? [msg('score.key', { key })] : []), ...(metre ? [metre] : []), msg('score.bars', { n: bars }), msg('score.pastedOn')], 'dot'),
  };
}
/** The file a lead sheet downloads as. */
export const abcFile = (title: string): string => title.replace(/[^\w.-]+/g, '_') + '.abc.txt';

/* ---------------------------------------------------------------- a planned score and its drawing */

/** A score as the planner writes it: its sections in order, its bars (each a section's, four notes as scale degrees;
 *  `past` the lyrics' end in a long plan, `loop` the replayed last section of a runaway one), and its verdict. */
export interface Score {
  seed: number; order: string[]; meas: { b: number[]; si: number; first: boolean; past?: boolean; loop?: boolean }[];
  lyricBars: number; verdict: 'healthy' | 'long' | 'runaway'; est: number; abc: string; title: string; chords: boolean;
}
/** The tempo the preview plays at, the scale's notes as ABC and as heard, and the chords (G major). */
export const SCORE_BPM = 96, SCORE_Q = 60 / SCORE_BPM;
export const SCORE_SCALE = ['G', 'A', 'B', 'c', 'd', 'e', 'f', 'g', 'a'], SCORE_FREQ = [196, 220, 247, 262, 294, 330, 370, 392, 440];
const CHORDS = ['G', 'Am', 'Bm', 'C', 'D', 'Em', 'D7'];
export const barChord = (m: { b: number[] }): string => CHORDS[Math.min(8, m.b[0] ?? 0) % 7]!;

/** Where everything on the staff goes, shared by the drawing and the playhead (E-01): systems of six bars, each bar's
 *  notes and each system's end. */
export function scoreLayout(sc: Score) {
  const perRow = 6, W = 640, RH = 72, LS = 7, rows: Score['meas'][] = [];
  for (let i = 0; i < sc.meas.length; i += perRow) rows.push(sc.meas.slice(i, i + perRow));
  const run = sc.verdict === 'runaway';
  const out = rows.map((row, ri) => {
    const top = ri * RH, y0 = top + 26, last = ri === rows.length - 1;
    const avail = W - (run ? 26 : 0), pre = 26 + 12 + (ri === 0 ? 18 : 0), bw = (avail - pre) / perRow, end = last ? pre + bw * row.length + (run ? 26 : 0) : avail;
    const bars = row.map((m, mi) => { const bx = pre + mi * bw; return { m, bx, bw, notes: m.b.map((_, ni) => bx + 12 + ni * (bw - 20) / 4) }; });
    return { ri, top, y0, pre, bw, end, last, bars };
  });
  return { W, RH, LS, rows: out, H: rows.length * RH - 6 };
}

/** One mark of the staff drawing: an SVG element by its tag, its attributes and class, and its words. `si` marks a
 *  section's rehearsal mark and name (lit while it plays). */
export interface StaffMark { tag: 'line' | 'rect' | 'text' | 'ellipse' | 'circle'; cls?: string; a: { [attr: string]: number | string }; text?: Slot; si?: number; colour?: string }

export interface ScoreModel {
  title: string; meta: Slot; verdict: { cls: string; text: Slot }; why: Slot; run: boolean;
  order: { name: string; colour: string; tip: Slot }[];
  staff: { W: number; H: number; aria: Slot; marks: StaffMark[]; head: { y: number; h: number } };
  /** Where the playhead stands for each note (its system and place), and the notes as heard, in order. */
  play: { x: number; row: number; y: number; rowEnd: number; si: number }[];
  freqs: number[];
  /** Each system's top and height in the drawing's units, so the paper can scroll to the one being played. */
  rows: { top: number; h: number }[];
  file: string;
}

/* coordinates as the concept writes them, unrounded, so the drawing is the same to the last place */
const r2 = (x: number) => x;
/** A section's ink on the score's paper (E-37): the trail above uses the same mix, so a section has one colour. */
const secInk = (n: string) => `color-mix(in srgb, ${sectionColour(n)} 62%, #2a2433)`;

/** The preview's words and drawing (34-score-modal.js staffSVG and openScore): each system has a label row above it
 *  (rehearsal marks in the section colour over the bar they start), a treble clef and the key signature, 4/4 on the
 *  first; noteheads on their lines with a ledger line for A5, stems down from the middle line up; a final double bar
 *  where the music ends. A long plan's bars past the lyrics are washed amber; a runaway's repeat ends on a repeat sign
 *  and runs on into "…" with no final bar (E-23). */
export function scoreModel(sc: Score): ScoreModel {
  const G = scoreLayout(sc), { LS } = G, run = sc.verdict === 'runaway', marks: StaffMark[] = [];
  const line = (x1: number, x2: number, y1: number, y2: number, cls: string) => marks.push({ tag: 'line', cls, a: { x1: r2(x1), x2: r2(x2), y1: r2(y1), y2: r2(y2) } });
  for (const R of G.rows) {
    const { top, y0, end, last } = R, yb = y0 + 4 * LS;
    for (const { m, bx, bw } of R.bars) if (m.past) marks.push({ tag: 'rect', cls: 'st-past', a: { x: r2(bx), y: y0 - 9, width: r2(bw), height: 4 * LS + 18 } });
    const solid = run && last ? end - 26 : end;
    for (let l = 0; l < 5; l++) { line(0, solid, y0 + l * LS, y0 + l * LS, 'st-line'); if (solid < end) line(solid, end, y0 + l * LS, y0 + l * LS, 'st-line open'); }
    line(0, 0, y0, yb, 'st-bar');
    marks.push({ tag: 'text', cls: 'st-clef', a: { x: 2, y: y0 + 31 }, text: '\u{1D11E}' }, { tag: 'text', cls: 'st-acc', a: { x: 28, y: y0 + 8 }, text: '♯' });
    if (R.ri === 0) marks.push({ tag: 'text', cls: 'st-time', a: { x: 46, y: y0 + 13 }, text: '4' }, { tag: 'text', cls: 'st-time', a: { x: 46, y: y0 + 27 }, text: '4' });
    R.bars.forEach(({ m, bx, bw, notes }, mi) => {
      const isEnd = last && mi === R.bars.length - 1, x = bx + bw;
      const dots = (cx: number) => marks.push({ tag: 'circle', cls: 'st-dot', a: { cx: r2(cx), cy: y0 + 1.5 * LS, r: 1.6 } }, { tag: 'circle', cls: 'st-dot', a: { cx: r2(cx), cy: y0 + 2.5 * LS, r: 1.6 } });
      if (isEnd) { line(x - 5, x - 5, y0, yb, 'st-bar'); line(x - 1.5, x - 1.5, y0, yb, 'st-bar fin'); if (run) { dots(x - 9); marks.push({ tag: 'text', cls: 'sc-sec st-more', a: { x: r2(x + 26), y: top + 11, 'text-anchor': 'end' }, text: msg('score.more') }); } }
      else line(x, x, y0, yb, 'st-bar');
      if (m.loop && m.first) { line(bx + 1.5, bx + 1.5, y0, yb, 'st-bar fin'); dots(bx + 6); }
      const name = sc.order[m.si]!, label: Slot = m.past ? msg('score.past') : m.loop ? msg('score.again', { name }) : name;
      if (m.first && m.past) marks.push({ tag: 'text', cls: 'sc-sec st-pastlbl', a: { x: r2(bx + 2), y: top + 11 }, text: label });
      else if (m.first) marks.push({ tag: 'rect', cls: 'sc-mark', si: m.si, colour: sectionColour(name), a: { x: r2(bx - 2), y: top, width: 10, height: 15, rx: 3 } }, { tag: 'text', cls: 'sc-sec', si: m.si, colour: secInk(name), a: { x: r2(bx + 2), y: top + 11 }, text: label });
      /* on a bar that opens a section the chord follows the section's name on that line */
      const chars = m.past ? 'past the lyrics'.length : (name + (m.loop ? ' again' : '')).length;
      if (sc.chords) marks.push({ tag: 'text', cls: 'st-chord', a: { x: r2(bx + 4 + (m.first ? Math.round(chars * 6.6) + 8 : 0)), y: y0 - 12 }, text: barChord(m) });
      m.b.forEach((deg, ni) => {
        const dg = Math.min(8, deg), nx = notes[ni]!, ny = y0 + 3 * LS - dg * LS / 2, down = dg >= 2;
        if (ny <= y0 - LS + 0.1) line(nx - 6, nx + 6, y0 - LS, y0 - LS, 'st-line led');
        marks.push({ tag: 'ellipse', a: { cx: r2(nx), cy: r2(ny), rx: 4, ry: 2.8, transform: `rotate(-20 ${r2(nx)} ${r2(ny)})`, fill: 'currentColor' } });
        if (down) line(nx - 3.6, nx - 3.6, ny + 1, ny + 19, 'st-stem'); else line(nx + 3.6, nx + 3.6, ny - 1, ny - 19, 'st-stem');
      });
    });
  }
  const r0 = G.rows[0]!, play = G.rows.flatMap(R => R.bars.flatMap(B => B.notes.map(x => ({ x: x - 5, row: R.ri, y: R.y0 - 6, rowEnd: R.end - 3, si: B.m.si }))));
  const len = run ? msg('score.len.loops', { t: mmss(sc.est) }) : mmss(sc.est);
  return {
    title: sc.title, meta: msg('score.meta', { seed: sc.seed, len, n: sc.order.length }), run,
    verdict: { cls: `v-${sc.verdict}`, text: msg(`score.v.${sc.verdict}`) },
    why: sc.verdict === 'long' ? msg('score.why.long', { n: sc.meas.length - sc.lyricBars }) : msg(`score.why.${sc.verdict}`),
    order: sc.order.map(n => ({ name: n, colour: sectionColour(n), tip: msg('score.from', { name: n }) })),
    staff: { W: G.W, H: G.H, aria: msg('score.staff', { n: sc.order.length, bars: sc.meas.length }), marks, head: { y: r0.y0 - 6, h: 4 * LS + 12 } },
    play, freqs: sc.meas.flatMap(x => x.b).map(d => SCORE_FREQ[Math.min(8, d)]!), rows: G.rows.map(R => ({ top: R.top, h: G.RH })),
    file: abcFile(sc.title || 'lead-sheet'),
  };
}
