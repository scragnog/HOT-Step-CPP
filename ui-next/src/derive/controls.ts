// What the shared controls draw beyond their parameters' own models (docs/09, package 1): a number box's text, a
// ruler's scale, fill, caption and markers, a fader's travel, a look-back strip's bars and caption, a power chip's
// state, where a Quick row's value stands between its choices, and each option's face in a picker (a plugin's family,
// symbol, file and curve; a model's quantisation; a language's own name). Each is worked out from the definitions and
// the desk, as the concept works it out (20-controls.js, 21-numbox.js, 21-ruler.js, 21-fader.js, 22-pick.js,
// 25-module-tempo.js, 25-module-repetition.js, 25-module-adhere.js); none has a branch for any module.

import type { Node, Value } from '../definitions/types.ts';
import { ENGINES } from '../definitions/engines.ts';
import {
  LANGUAGE_NATIVE, PLUGIN_BUILT_IN, PLUGIN_COMMUNITY, PLUGIN_FAMILIES, PLUGIN_FAMILY_OF, PLUGIN_FILES, PLUGIN_OTHER, PLUGIN_SYMBOLS,
  SCHEDULE_CURVES,
} from '../definitions/option-faces.ts';
import { type Desk, type Index, choiceLists, has, isAuto, mmss, resolve, valueOf, wordsOf } from './core.ts';
import { type Msg, type Slot, msg } from './messages.ts';
import { paramWords } from './strings.ts';

/** A picker's option, as its variant draws it beside its words. */
export interface OptionFace {
  /** plugin: its place in the list (1 is the first), its symbol, its family and that family's hue, its file, and for a
   *  schedule its curve (25 points of sigma, 1 to 0). */
  n?: number; sym?: string; family?: string; hue?: number; file?: string; curve?: number[];
  /** model: its name before the quantisation, the quantisation's badge and its class (lo, mid, hi, fp4), its size. */
  name?: string; quant?: string; qc?: string; gb?: number;
  /** model: its size as written ("2.5 GB"), its share of the card (0 to 1), and its quantisation chip's tip (its file
   *  and size, "Turbo · Q8_0 · 2.5 GB"). */
  gbText?: string; mem?: number; title?: string;
  /** stamp: the language's own name, and its ink: the stamp's pink for the first, then hues a golden angle apart. */
  native?: string; ink?: string;
}

export interface RulerModel {
  /** On auto: the fill is the estimate, drawn as auto. */
  auto: boolean;
  /** Has an auto switch: the head carries its auto button. */
  hasAuto: boolean;
  /** Where the fill ends on the scale (0 to 1), the cap's words and whether it stands flush at an end. */
  at: number; caption: Slot; capAt: 'start' | 'end' | null;
  /** The scale: the spacing of its major and minor ticks (percent of the scale) and its labels. */
  maj: number; min: number; labels: { at: number; text: string }[];
  /** The seconds the scale spans. */
  span: number;
  /** Beat I/O and the trim tag, where the node draws markers: each end's place and width (0 to 1 of the scale), its tip,
   *  the trim tag's place, and the line under the ruler when Beat I/O is longer than the song. */
  markers: { ioHere: boolean; io: { left: number; width: number }[] | null; ioTip: Slot; trimHere: boolean; trim: { at: number; flip: boolean } | null; note: Slot | null } | null;
  /** Its parameter can be held by an engine switch (its note under the ruler; drawn empty until the engines' book
   *  reaches a part's model). */
  held: boolean;
}

export interface ControlModel {
  /** A number box or an LCD number: the number as it is typed in the box ("" on auto), the box's words while empty,
   *  whether the parameter is on auto (null: it has none), and the seconds it takes (a step count's estimate). */
  box?: { text: string; empty: Slot; auto: boolean | null; seconds: Slot | null };
  ruler?: RulerModel;
  /** A fader: where the cap stands on its travel (0 to 1), where the unity line (1.00) falls, where the range stops on
   *  the travel (1: at the top), and how far along the parameter's range the top of the travel is (for a hand). */
  fader?: { at: number; unity: number; top: number; span: number; stops: boolean; name: Slot; tape: Slot | null };
  /** A look-back strip: how many of its bars are lit, out of how many, and its caption and tip. */
  lookback?: { lit: number; bars: number; far: Slot; mid: Slot | null; now: Slot; tip: Slot };
  /** A power chip: whether its group is on, the value pressing it sets, its words and its tip. */
  power?: { on: boolean; to: number; words: Slot; tip: Slot };
  /** A Quick row whose value falls between its choices: its place as a fractional index, and whether it lies beyond
   *  either end (-1, 1) or between (0). */
  between?: { pos: number; beyond: -1 | 0 | 1 };
  /** A picker's options' faces, in the order of its parameter's options. */
  faces?: OptionFace[];
}

/* ---------------------------------------------------------------- numbers in boxes */

/** A number as it is typed into a box: two places for a step under 1, none otherwise (21-numbox.js numVal). */
export const boxText = (v: number, step: number): string => String(+v.toFixed(step < 1 ? 2 : 0));

/** How long a run of steps takes on an engine (25-module-sampler.js stepSec), for a step count's estimate. */
export const stepSeconds = (desk: Desk, steps: number): Msg => {
  const per = ENGINES.find(e => e.id === desk.engine)?.stepSeconds ?? 0.11;
  return msg('box.seconds', { s: Math.max(0.1, steps * per).toFixed(1) });
};

/* ---------------------------------------------------------------- the ruler (21-ruler.js) */

/** The scale of a ruler spanning `span` seconds: labels on a round step (30 s, 1, 2, 3, 5 or 10 min) giving at most
 *  six, minor ticks halving it. */
export function rulerScale(span: number): { maj: number; min: number; labels: { at: number; text: string }[] } {
  const st = [30, 60, 120, 180, 300, 600].find(s => span / s <= 5) ?? 600, n = Math.floor(span / st + 1e-9), pct = (v: number) => +(v / span * 100).toFixed(4);
  return { maj: pct(st), min: pct(st / 2), labels: Array.from({ length: n + 1 }, (_, i) => ({ at: pct(i * st), text: mmss(i * st) })) };
}

/** The song's tempo: the BPM field when it is set, else a BPM written in the prompt or the tags, else none (S-07). */
export function songTempo(bpm: { value: Value | null; auto: boolean } | undefined, text: Value | null | undefined): number | null {
  if (bpm && !bpm.auto && typeof bpm.value === 'number') return bpm.value;
  const typed = /(\d{2,3})\s*BPM/i.exec(String(text ?? ''))?.[1];
  return typed ? +typed : null;
}

/** What auto stands for on a number: its default, or its maximum where its auto words say "up to {max}" (the model
 *  ends the song, up to that). Turning auto off starts the value there, so the press changes nothing the song gets
 *  until the control is moved (S-17). */
export function autoEstimate(ix: Index, key: string, desk: Desk): number | null {
  const f = wordsOf(ix, ix.params.get(key)!.p, desk);
  if (!f.range || !f.auto) return null;
  return /\{max\}/.test((typeof f.auto === 'object' && f.auto.says) || '') ? f.range[1] : Number(f.default);
}

/** A ruler on the desk. Its scale spans the node's `scaleMax`, else the parameter's maximum. On auto its fill is the
 *  estimate: the default, or the whole span where the auto words say "up to" (the model ends the song). With
 *  `markers`, it reads Beat I/O, its bars, the BPM, Auto-trim and the prompt (in that order) and draws their markers. */
export function rulerModel(ix: Index, node: Node, desk: Desk): RulerModel | undefined {
  const key = node.param; if (!key || !has(ix, key, desk.engine)) return undefined;
  const f = wordsOf(ix, ix.params.get(key)!.p, desk); if (!f.range) return undefined;
  const opts = (node.opts ?? {}) as { scaleMax?: number; markers?: boolean };
  const span = opts.scaleMax ?? f.range[1], auto = isAuto(ix, key, desk), hasAuto = !!f.auto;
  /* on auto the ruler shows the estimate; where the auto words say "up to" (the model ends the song) the whole span it may use */
  const est = auto ? autoEstimate(ix, key, desk) : null, upTo = est !== null && est !== Number(f.default);
  const len = est ?? Number(valueOf(ix, key, desk));
  const at = Math.min(1, Math.max(0, len / span));
  const caption: Slot = auto ? (upTo ? msg('ruler.upTo', { v: mmss(len) }) : /exact/.test(f.auto && typeof f.auto === 'object' ? f.auto.says ?? '' : '') ? mmss(len) : msg('value.about', { v: mmss(len) })) : mmss(len);
  return { auto, hasAuto, held: !!ix.params.get(key)!.p.held, at, caption, capAt: at > 0.9 ? 'end' : at < 0.1 ? 'start' : null, span, ...rulerScale(span), markers: opts.markers && desk.tier !== 'quick' ? rulerMarkers(ix, node, desk, len, span) : null };
}

function rulerMarkers(ix: Index, node: Node, desk: Desk, len: number, span: number): RulerModel['markers'] {
  const [io, bars, bpmKey, trim, text] = node.reads ?? [];
  const read = (k: string | undefined) => (k && ix.params.has(k) && has(ix, k, desk.engine) ? valueOf(ix, k, desk) ?? null : null);
  const on = read(io) === true, n = Number(read(bars) ?? 2);
  const bpm = songTempo(bpmKey && ix.params.has(bpmKey) && has(ix, bpmKey, desk.engine) ? { value: read(bpmKey), auto: isAuto(ix, bpmKey, desk) } : undefined, read(text));
  const beat = n * 4 * 60 / (bpm ?? 120), need = beat * 2;
  const w = Math.min(len / 2, beat) / span;
  const box = text && ix.params.has(text) ? (paramWords(ix, text, desk).undoName ?? paramWords(ix, text, desk).label) : '';
  const trimOn = !!trim && read(trim) === true;
  const here = (k: string | undefined) => !!k && ix.params.has(k) && has(ix, k, desk.engine);
  return {
    ioHere: here(io), trimHere: here(trim),
    io: on ? [{ left: 0, width: w }, { left: len / span - w, width: w }] : null,
    ioTip: bpm ? msg('ruler.io', { n }) : msg('ruler.io.at120', { n, box }),
    trim: trimOn ? { at: len / span, flip: len / span > 0.8 } : null,
    note: on && need > len ? msg('ruler.io.short', { n, need: mmss(Math.ceil(need)), bpm: bpm ?? 120 }) : null,
  };
}

/** Where a hand dragging a length ruler sets it: snapped to 5 s (15 s above 2:00), and to a labelled tick within 6px
 *  (S-31). `f` is the hand's place along the scale (0 to 1), `px` the scale's width in pixels; null for a ruler whose
 *  parameter is not a length (it takes the hand's place as it is). */
export function rulerSnap(ix: Index, key: string, desk: Desk, f: number, span: number, px: number): number | null {
  const face = resolve(ix.params.get(key)!.p, desk.engine);
  if (face.unit !== 's' || !face.range) return null;
  const v = Math.min(1, Math.max(0, f)) * span, tick = rulerScale(span).labels.map(l => l.at / 100 * span).find(t => Math.abs(t - v) / span * px <= 6);
  const round = (x: number) => { const r = x > 120 ? 15 : 5; return Math.round(x / r) * r; };
  return Math.min(face.range[1], Math.max(face.range[0], tick ?? round(v)));
}

/* ---------------------------------------------------------------- the scrub token (20-controls.js) */

/** Where a hand dragging a scrub token sideways sets it: from the value it started at, `px` pixels on (Shift: a tenth
 *  as far). A range of 500 steps or fewer is 220px end to end; a wider one moves with the value, a hundredth of it a
 *  pixel (one a pixel at 100, ten at 1000, a hundred at 10000: S-28), never less than a step, on the step grid. */
export function scrubValue(ix: Index, key: string, desk: Desk, from: number, px: number, shift: boolean): number | null {
  const r = resolve(ix.params.get(key)!.p, desk.engine).range; if (!r) return null;
  const [lo, hi, step] = r, k = shift ? 0.1 : 1;
  if ((hi - lo) / step <= 500) return from + px / 220 * k * (hi - lo);
  /* walk the pixels: each moves by the rate where the value is then */
  let v = from; const dir = Math.sign(px);
  for (let i = 0; i < Math.abs(px); i++) { v += dir * Math.max(step, Math.abs(v) / 100) * k; if (v <= lo || v >= hi) break; }
  return Math.min(hi, Math.max(lo, v));
}

/* ---------------------------------------------------------------- a part's extra model */

const numberFace = (ix: Index, key: string, desk: Desk) => wordsOf(ix, ix.params.get(key)!.p, desk);

/** The extra model of a shared control's node, or undefined for a part that needs none. `words` are the node's own
 *  words the core has already keyed (a look-back strip's unit and who; a power chip's label). */
export function controlModel(ix: Index, node: Node, desk: Desk, words: { [k: string]: Slot }, label: Slot | undefined, rowAt?: { values: Value[]; checked: number | null }): ControlModel | undefined {
  const key = node.param, here = !!key && has(ix, key, desk.engine);
  const opts = (node.opts ?? {}) as Record<string, unknown>;
  switch (node.part) {
    case 'numberBox': case 'lcdNumber': {
      if (!here) return undefined;
      const f = numberFace(ix, key!, desk), v = Number(valueOf(ix, key!, desk)), auto = f.auto ? isAuto(ix, key!, desk) : null, step = f.range?.[2] ?? 1;
      const empty: Slot = node.part === 'lcdNumber' ? msg('box.about', { v: boxText(Number(f.default), step) }) : f.auto ? msg('box.auto') : '';
      return { box: { text: auto ? '' : boxText(v, step), empty, auto, seconds: opts.seconds ? stepSeconds(desk, v) : null } };
    }
    case 'ruler': { const r = rulerModel(ix, node, desk); return r ? { ruler: r } : undefined; }
    case 'fader': {
      if (!here) return undefined;
      const f = numberFace(ix, key!, desk); if (!f.range) return undefined;
      const [lo, hi] = f.range, travel = typeof opts.travel === 'number' && opts.travel > hi ? opts.travel : hi, v = Number(valueOf(ix, key!, desk));
      const along = (x: number) => Math.min(1, Math.max(0, (x - lo) / (travel - lo)));
      /* its accessible name is its parameter's full name and strength; its tape, said short, carries that name as its
         tip (V-01, 25-module-adapters.js) */
      const full = paramWords(ix, key!, desk).label;
      return { fader: { at: along(v), unity: along(1), top: (hi - lo) / (travel - lo), span: (travel - lo) / (hi - lo), stops: travel > hi, name: msg('fader.name', { label: full }), tape: full } };
    }
    case 'lookback': {
      if (!here) return undefined;
      const f = numberFace(ix, key!, desk); if (!f.range) return undefined;
      const hi = f.range[1], v = Number(valueOf(ix, key!, desk)), bars = 160, fps = typeof opts.perSecond === 'number' ? opts.perSecond : null;
      const unit = words.unit ?? '', who = words.who ?? '';
      return { lookback: {
        lit: v === 0 ? bars : Math.round(Math.min(hi, v) / hi * bars), bars,
        far: fps ? msg('lookback.farSeconds', { s: Math.round(hi / fps) }) : msg('lookback.far', { n: hi, unit }),
        mid: fps ? msg('lookback.rate', { n: fps, unit }) : null, now: msg('lookback.now'),
        tip: msg(fps ? 'lookback.tip.rate' : 'lookback.tip', { who, n: fps ?? 0, unit }),
      } };
    }
    case 'powerChip': {
      if (!here) return undefined;
      const v = valueOf(ix, key!, desk), off = Number(opts.off ?? 0), on = v !== off, l = label ?? '';
      return { power: { on, to: on ? off : Number(opts.on ?? off), words: on ? l : msg('power.off', { label: l }), tip: msg(on ? 'power.tip.on' : 'power.tip.off') } };
    }
    case 'choiceRow': {
      if (!rowAt || rowAt.checked !== null || !key || !here) return undefined;
      const nums = rowAt.values.map(Number), v = Number(valueOf(ix, key, desk));
      if (!nums.length || nums.some(Number.isNaN) || Number.isNaN(v)) return undefined;
      return { between: choicePos(nums, v) };
    }
    case 'pick': case 'chips': case 'termChips': case 'lcdChips': {
      if (!here || ix.params.get(key!)!.p.type !== 'choice') return undefined;
      const variant = node.part === 'pick' ? String(opts.variant ?? 'lcd') : '';
      if (variant !== 'plugin' && variant !== 'model' && variant !== 'stamp') return undefined;
      return { faces: optionFaces(ix, key!, desk, variant) };
    }
  }
  return undefined;
}

/** Where a value falls along a row of numeric choices, as a fractional index in the row's order (2.4: between the third
 *  and fourth, nearer the third), and whether it lies beyond an end (20-controls.js choicePos). */
export function choicePos(nums: number[], v: number): { pos: number; beyond: -1 | 0 | 1 } {
  const up = nums[nums.length - 1]! >= nums[0]!, at = (i: number) => (up ? nums[i]! : -nums[i]!), w = up ? v : -v, n = nums.length;
  if (w < at(0)) return { pos: 0, beyond: -1 };
  if (w > at(n - 1)) return { pos: n - 1, beyond: 1 };
  const j = nums.findIndex((_, i) => at(i) > w);
  if (j <= 0) return { pos: j < 0 ? n - 1 : 0, beyond: 0 };
  return { pos: j - 1 + (w - at(j - 1)) / (at(j) - at(j - 1)), beyond: 0 };
}

/* ---------------------------------------------------------------- an option's face in a picker */

/** The card a model's size is drawn against until the models service tells the real one (desk.vram.card). */
export const CARD_GB = 31.8;
const QUANT = /(IQ\d_X+S-imat|Q\d_K_M-imat|Q\d_K_[MS]|Q\d_K|Q\d_0|BF16|F16|MXFP4|NVFP4)/;
/** A model's quantisation as a badge and its class (22-pick.js quantOf). */
export function quantOf(label: string): { quant: string; qc: string } | null {
  const q = QUANT.exec(label)?.[1]; if (!q) return null;
  return { quant: q, qc: /Q4|Q2/.test(q) ? 'lo' : /Q8|Q6/.test(q) ? 'mid' : /FP4/.test(q) ? 'fp4' : 'hi' };
}
/** A schedule's curve, 25 points of sigma from the start to the end of the run. */
const curveOf = (label: string): number[] => { const f = SCHEDULE_CURVES[label] ?? ((t: number) => 1 - t); return Array.from({ length: 25 }, (_, i) => +f(i / 24).toFixed(4)); };

/** Every option's face in a picker's variant: a plugin's place, symbol, family, file and (a schedule's) curve; a
 *  model's name, quantisation and size; a language's own name. Words are the options' own (English labels as the
 *  server lists them); the tables are src/definitions/option-faces.ts. */
export function optionFaces(ix: Index, key: string, desk: Desk, variant: 'plugin' | 'model' | 'stamp'): OptionFace[] {
  const { p } = ix.params.get(key)!, f = wordsOf(ix, p, desk), card = desk.vram?.card ?? CARD_GB;
  const from = !Array.isArray(f.options) && f.options ? f.options : null;
  const { fixed, listed } = choiceLists(f, desk);
  const all = [...fixed.map(o => ({ value: o.value, label: o.label })), ...listed.map(o => ({ value: o.value, label: o.label, gb: o.gb }))];
  /* a list of schedules draws each one's curve: the scheduler plugins, or a list whose options the curves table knows */
  const schedule = (!!from && from.from === 'plugins.scheduler') || all.some(o => o.label in SCHEDULE_CURVES);
  return all.map((o, i) => {
    if (variant === 'stamp') return { native: LANGUAGE_NATIVE[o.label] ?? o.label, ink: i === 0 ? '#e0225a' : `hsl(${Math.round(i * 137.508) % 360} 62% 42%)` };
    if (variant === 'model') {
      const q = quantOf(o.label), gb = 'gb' in o && typeof o.gb === 'number' ? o.gb : null;
      return { name: q ? o.label.split(' · ')[0]! : o.label, ...(q ?? {}), title: `${o.label} · ${(gb ?? 0).toFixed(1)} GB`, ...(gb ? { gb, gbText: `${gb.toFixed(1)} GB`, mem: Math.min(1, gb / card) } : {}) };
    }
    const fam = PLUGIN_FAMILY_OF.params[key] ?? PLUGIN_FAMILY_OF.engines[desk.engine] ?? PLUGIN_FAMILIES.find(x => x.members.includes(o.label)) ?? (/^MD /.test(o.label) ? PLUGIN_COMMUNITY : PLUGIN_OTHER);
    const file = PLUGIN_FILES[o.label] ?? (/exp\./.test(o.label) ? PLUGIN_BUILT_IN.experimental : /^MD /.test(o.label) ? `md_${o.label.slice(3).toLowerCase().replace(/[^a-z0-9.]+/g, '_')}.lua` : PLUGIN_BUILT_IN.plain);
    return { n: i + 1, sym: PLUGIN_SYMBOLS[o.label] ?? o.label.replace(/[^A-Za-z0-9]/g, '').slice(0, 2), family: fam.name, hue: fam.hue, file, ...(schedule ? { curve: curveOf(o.label) } : {}) };
  });
}
