// What each piece of the page draws, as plain data the core works out from the definitions and the desk: a part's
// model (its values, words, options, scale and state, every one already decided), a tile's, a pedal's, a side unit's,
// the board's, the desk's and a Modules card's; and how a hand or a key moves a value (keyStep, fractionValue,
// parseTyped), what the desk's make-up does when something is added or removed (addTo, removeFrom). The React layer
// draws these and hands back what the user did; it decides nothing about a value (docs/09, "The one rule").
//
// A model is plain JSON: a view that draws one compares it with the last and redraws only when it changed, so a value
// that moves redraws the parts that show it and nothing else (docs/09, "Rendering"). Nothing here has a branch for
// any one module.

import type { Role } from '../definitions/roles.ts';
import { CATEGORIES, type Definition, type EngineId, type ModuleDef, type Node, type PedalDef, type Tier, type UnitDef, type Value } from '../definitions/types.ts';
import { PARTS, type PartSpec } from '../definitions/parts.ts';
import { ENGINES, engineName, engineTag } from '../definitions/engines.ts';
import { type Desk, type Index, applies, changed, has, holds, isAuto, nodeAt, resolve, shownValue, stripOf, valueOf, wordsOf } from './core.ts';
import { type Msg, type Slot, list, msg } from './messages.ts';
import { defName } from './steps.ts';
import {
  actionWords, bareText, fillMsg, forcedOf, formatMsg, keptMsg, lcdMsg, nodeOptWord, nodeWord, optionsOf, paramWords, pickMsg, stateWords, stripWords, textMsg,
  wordKey,
} from './strings.ts';
import { alsoLinkWords, alsoSet } from './tiers.ts';
import { ringKeys, boardNodes, choiceOf, dimOf, hiddenLabModules, idleOf, rowOf, type Cell } from './view.ts';
import { boardCount, nodeShown, parkedOn, tileAtTier, tileName, usedHere } from './desk.ts';
import { autoEstimate, controlModel, type ControlModel } from './controls.ts';
import { kidsOf, type KidModel } from './containers.ts';
import { readoutModel, type ReadoutModel } from './readouts.ts';
import { instrumentModel, type InstrumentModel } from './instruments.ts';
import { mmss, onOf, stateOf } from './core.ts';
import type { Json } from '../definitions/types.ts';
import { modelRows, type ModelRow } from './vram.ts';
import { heldNote } from './engine.ts';
import { rng } from './generate.ts';
import { promptModel, sheetModel, type PromptModel, type SheetModel } from './fields.ts';

/* ---------------------------------------------------------------- a part's model */

/** Where a part stands: the surface its container gives it. A part draws the housing the concept gives it there
 *  (a labelled control is a foot pairing on a tile and an LCD selector on a pedal), and the rebuild's stylesheets give
 *  every part a look on every surface (styles/rebuild/). */
export type Surface = 'tile' | 'quick' | 'rack' | 'foot' | 'well' | 'lcd' | 'pedal' | 'unit'
  /* the containers' own (package 2): a terminal's lines, a rack group, a plugin's controls, a deck, an LCD's row */
  | 'term' | 'group' | 'plugin' | 'deck' | 'lcdrow';

/** How the shell places a laid-out cell (a pedal's knob): the core's words for it, not the view's. */
export interface CellWords {
  small?: boolean;
  /** The read-out prints the number alone and its unit under it. */
  bare?: boolean;
  label?: Slot;
  aria?: Slot;
  title?: Slot;
}

/** A parameter as a part draws it. */
export interface ParamModel {
  key: string;
  /** What the page needs it for, when its definition says (roles.ts): a part that lays out several parameters finds
   *  each by its role, never by its key. */
  role?: Role;
  value: Value;
  text: Slot;
  changed: boolean;
  auto: boolean;
  applies: boolean;
  /** Its own label (not the node's). */
  label: Slot;
  short?: Slot;
  tip?: Slot;
  /** A number: its range on the engine, its default's place and its own place on it (0 to 1). */
  range?: [number, number, number];
  at?: number;
  detent?: number;
  /** A number printed without its unit, the unit under it (a pedal's dB, Hz, LUFS); null otherwise. */
  bare?: { text: Slot; unit: string } | null;
  /** What an LCD read-out prints for it, and what an LCD picker prints. */
  lcd: Slot;
  pick: Slot;
  /** A choice: its options, with their words and whether each is the value in force. */
  options?: { value: string | number; label: Slot; short?: Slot; tip?: Slot; lcd: Slot; pick: Slot; detail?: Slot; on: boolean; off?: Slot }[];
  /** A choice the engine runs at another value meanwhile (ParamFace.forced): what is said beside it. */
  forced?: { note: Msg };
  placeholder?: Slot;
  /** The most characters a text takes. */
  maxLength?: number;
  /** The accessible name of the field a number is typed into. */
  typeName?: Slot;
  /** What its auto button's tip says auto does. */
  autoTip?: Slot;
  /** A field that keeps invalid text: what it says is in force meanwhile, and whether there is a value to put back. */
  kept?: { says: Slot; back: boolean };
  /** It can be automatic (a knob draws its auto pill). */
  canAuto?: true;
  /** Presses within this many milliseconds are one step (taps on TAP, keys on the key wheel). */
  run?: number;
  /** A song value this engine cannot take, held for another: what its control says (held.note: "Your 8 takes return
   *  on ACE-Step 1.5"), until the value is changed here. */
  held?: Msg;
}

/** What a node reads; `changed` when a parameter it reads is changed (an LCD's word carries the changed mark of what it
 *  shows: Tempo's KEY). */
export interface ReadModel { key: string; value: unknown; text: Slot; lcd: Slot; onDesk: boolean; changed?: boolean }

export interface PartModel {
  part: string;
  tier: Tier;
  engine: EngineId;
  surface: Surface;
  /** The node's options as given (sizes, looks); words among them are in `words`. */
  opts: Record<string, unknown>;
  /** The control's name: the shell's, else the node's, else its first parameter's. */
  label?: Slot;
  /** Its accessible name where that is not its label (a pedal knob's full name). */
  aria?: Slot;
  /** Its tooltip: the shell's, else its tip then how to work it. */
  title?: Slot;
  /** The node's own words: tip, text, between, and each option that is words (an XY pad's axes, a prefix). */
  words: Record<string, Slot>;
  params: ParamModel[];
  reads: ReadModel[];
  /** Its actions' words; a named action (parts.ts ACTIONS) says its name, so a part can run the one it means; `auto`, the
   *  parameter an action puts back on auto (the concept's data-auto hook, package 3c). */
  actions: { label: Slot; tip?: Slot; tier?: Tier; ring?: string; name?: string; auto?: string; item?: string }[];
  /** A Quick choice row: its label, the choice checked (null: the values match none) and what the row says. */
  row?: { label: Slot; checked: number | null; says: Slot; choices: Slot[]; sets: Slot[] };
  /** A meter: its scale. */
  meter?: MeterScale;
  /** The lyric sheet's sections, line count, tabs and stamp; the prompt card's suggestions, caption source and tools
   *  (src/derive/fields.ts). */
  sheet?: SheetModel;
  prompt?: PromptModel;
  /** An LCD's lines, each with the case it is printed in. */
  lines?: { text: Slot; upper: boolean }[];
  /** What a shared control draws beyond its parameters (src/derive/controls.ts): a box's text, a ruler, a fader, a
   *  look-back strip, a power chip, a row's value between its choices, a picker's option faces. */
  control?: ControlModel;
  /** The engine panel's model rows (a modelPicker): each model's role, words and option faces (derive/vram.ts). */
  models?: ModelRow[];
  /** Why the node stands back, when it does. */
  dim: Slot | null;
  small: boolean;
  /** What the node says when it stands back, whether or not it does now (a guard's "guard off", drawn hidden until it
   *  holds); absent for a node that never stands back. */
  dimWords?: Slot;
  /** A container: its children, in the order they are drawn (src/derive/containers.ts). */
  kids?: KidModel[];
  /** A read-out: what it shows, worked out from what it reads (src/derive/readouts.ts). */
  readout?: ReadoutModel;
  /** The state its node names (a drop zone's file), and whether another definition only lends it to this node. */
  state?: StateModel[];
  /** A bespoke instrument: what it draws, worked out from what it binds and reads (src/derive/instruments.ts). */
  instrument?: InstrumentModel;
}

/** A file's length, as the recording service told it (desk.sessions.recording): "3:51 long". */
function fileDetail(kind: string, v: Json | undefined, desk: Desk): { detail?: Slot } {
  const told = desk.sessions?.recording as { [file: string]: number } | undefined, t = kind === 'file' && typeof v === 'string' ? told?.[v] : undefined;
  return typeof t === 'number' ? { detail: msg('file.long', { t: mmss(t) }) } : {};
}

/** A file's waveform on its sleeve (the concept's drawMini, 25-module-timbre.js): bars seeded by the name, the same on
 *  every visit; enough for the widest sleeve, a bar every 3px. */
const WAVE_BARS = 400;
function fileWave(v: Json | undefined): { wave?: number[] } {
  if (typeof v !== 'string' || !v) return {};
  const r = rng(v.length * 13);
  return { wave: Array.from({ length: WAVE_BARS }, () => 0.3 + 0.7 * r()) };
}

/** A state a node names, as a part draws it. */
/** `detail`: a file's length, when the page was told it (a recording given to Continue from recording: package 3c). */
/** `wave`: a loaded file's waveform as its sleeve draws it (dropZone's `wave`), each bar's height from 0 to 1. */
export interface StateModel { key: string; kind: string; value: Json | null; lent: boolean; owner: string; noun: Slot; detail?: Slot; wave?: number[] }


/** A parameter on the desk as a part draws it. */
export function paramModel(ix: Index, key: string, desk: Desk, cell?: CellWords): ParamModel {
  const { p } = ix.params.get(key)!, f = wordsOf(ix, p, desk), w = paramWords(ix, key, desk), v = valueOf(ix, key, desk)!;
  const out: ParamModel = {
    key, value: v, text: textMsg(ix, key, desk), changed: changed(ix, key, desk), auto: isAuto(ix, key, desk), applies: applies(ix, key, desk),
    label: w.label, lcd: lcdMsg(ix, key, desk), pick: pickMsg(ix, key, desk),
    ...(w.short ? { short: w.short } : {}), ...(w.tip ? { tip: w.tip } : {}), ...(p.role ? { role: p.role } : {}),
  };
  if (p.type === 'number' && f.range) {
    /* while automatic, a number stands at the value auto stands for (L-11) */
    const est = out.auto ? autoEstimate(ix, key, desk) : null;
    const [lo, hi] = f.range, n = est ?? (typeof v === 'number' ? v : lo), d = typeof f.default === 'number' ? f.default : lo;
    out.range = f.range; out.at = (n - lo) / (hi - lo); out.detent = Math.min(1, Math.max(0, (d - lo) / (hi - lo)));
    out.typeName = msg('knob.type', { label: w.label, min: lo, max: hi });
    out.bare = cell?.bare ? bareText(ix, key, desk) : null;
    /* a number of twelve steps or fewer has its steps as options too, each written as the page writes it (a part that
       draws stops: a stepped slider, chips) */
    const [, , step] = f.range, steps = Math.round((hi - lo) / step);
    if (steps <= 12) out.options = Array.from({ length: steps + 1 }, (_, i) => { const x = +(lo + i * step).toFixed(10), w = formatMsg(ix, key, desk, x); return { value: x, label: w, lcd: w, pick: w, on: x === v }; });
  }
  /* a model choice's "" is shown as the file it stands for */
  const shown = shownValue(ix, key, desk, v);
  /* while forced, the value the engine runs is pressed and the others are disabled with why (A-30) */
  const forced = p.type === 'choice' ? forcedOf(ix, key, desk) : null;
  if (p.type === 'choice') out.options = optionsOf(ix, key, desk).map(o => ({ value: o.value, label: o.label, ...(o.short ? { short: o.short } : {}), ...(o.tip ? { tip: o.tip } : {}), lcd: o.lcd ?? o.label, pick: o.pick ?? o.lcd ?? o.label, ...(o.detail ? { detail: o.detail } : {}), on: forced ? o.value === forced.value : o.value === shown, ...(forced && o.value !== forced.value ? { off: forced.tip } : {}) }));
  if (forced) out.forced = { note: forced.note };
  if (f.placeholder) out.placeholder = fillMsg(ix, wordKey(ix, key, desk, 'placeholder'), f.placeholder, desk);
  if (f.maxLength) out.maxLength = f.maxLength;
  if (f.auto && f.auto.tip) out.autoTip = fillMsg(ix, `create.${ix.params.get(key)!.d.id}.p.${key.replace(/\./g, '-')}.autoTip${(p.per || []).some(l => 'engines' in l && l.engines.includes(desk.engine) && (l as { auto?: { tip?: string } }).auto?.tip) ? '@' + ((p.per || []).find(l => 'engines' in l && l.engines.includes(desk.engine)) as { engines: string[] }).engines.join('+') : ''}`, f.auto.tip, desk, f);
  const kept = p.type === 'text' ? keptMsg(ix, key, desk) : undefined;
  if (kept) out.kept = kept;
  if (f.auto) out.canAuto = true;
  if (p.run) out.run = p.run;
  const held = p.held ? heldNote(ix, key, desk) : null;
  if (held) out.held = held;
  return out;
}

/** What a node's part draws on the desk, or null when the node is not drawn (nodeShown). `key` is the node's
 *  translation key (nodeKeys; '' for a cell the shell lays out). */
export function partModel(ix: Index, d: Definition, node0: Node, desk: Desk, key: string, surface: Surface, cell?: CellWords): PartModel | null {
  if (!nodeShown(ix, node0, desk)) return null;
  const node = nodeAt(ix, node0, desk)!, spec = (PARTS as Record<string, PartSpec>)[node.part];
  const binds = [node.param, ...(node.params ?? [])].filter((k): k is string => !!k && has(ix, k, desk.engine));
  const params = binds.map(k => paramModel(ix, k, desk, cell));
  const words: Record<string, Slot> = {};
  if (key) {
    for (const f of ['tip', 'text', 'between'] as const) { const m = nodeWord(ix, node, desk, key, f); if (m) words[f] = m; }
    for (const [o, s] of Object.entries(spec?.opts ?? {})) if (s.type === 'text') { const m = nodeOptWord(ix, node, desk, key, o); if (m) words[o] = m; }
    /* a list of words (an XY pad's corners): each by its place, `corners.0` ... */
    else if (s.type === 'texts') ((node.opts as Record<string, unknown> | undefined)?.[o] as string[] | undefined ?? []).forEach((_, i) => { const m = nodeOptWord(ix, node, desk, key, o, i); if (m) words[`${o}.${i}`] = m; });
  }
  const nodeLabel = key ? nodeWord(ix, node, desk, key, 'label') : undefined;
  const label = cell?.label ?? nodeLabel ?? params[0]?.label;
  const tip = words.tip ?? params[0]?.tip, how = spec?.howTo ? { id: `part.${node.part}.howTo`, en: spec.howTo } : undefined;
  const title = cell?.title ?? (how ? (tip ? msg('part.tip', { tip, how }) : how) : tip);
  const opts = (node.opts ?? {}) as Record<string, unknown>;
  const m: PartModel = {
    part: node.part, tier: desk.tier, engine: desk.engine, surface, opts, words, params,
    reads: (node.reads ?? []).map(k => readModel(ix, k, desk)),
    actions: key ? actionWords(ix, node, desk, key).map(({ action: a, label: l, tip: t }) => ({ label: l, ...(t ? { tip: t } : {}), ...(a.do === 'tier' ? { tier: a.tier, ...(a.ring ? { ring: ringKeys(ix, a.ring).join(' ') } : {}) } : {}), ...(a.do === 'bespoke' ? { name: a.name } : {}), ...(a.do === 'auto' ? { auto: a.param } : {}), ...(a.do === 'add' ? { item: a.item } : {}) })) : [],
    dim: key ? dimOf(ix, node, desk, key) : null,
    small: !!cell?.small || opts.small === true,
    ...(label !== undefined ? { label } : {}), ...(cell?.aria ?? label) !== undefined ? { aria: (cell?.aria ?? label)! } : {}, ...(title !== undefined ? { title } : {}),
  };
  if (node.choices?.length && key) {
    const r = rowOf(ix, d, node, desk, key), c = choiceOf(ix, d, node, desk, key);
    /* each choice's tip says what it sets ("Steps: 8"), as the concept's buttons do */
    const sets = (v: Value | { [k: string]: Value }) => list(Object.entries(typeof v === 'object' ? v : { [r.key]: v }).filter(([k]) => has(ix, k, desk.engine)).map(([k, x]) => msg('row.sets', { label: paramWords(ix, k, desk).label, value: formatMsg(ix, k, desk, x) })), 'comma');
    m.row = { label: r.label, checked: c.checked, says: c.says, choices: r.choices.map(([w]) => w), sets: r.choices.map(([, v]) => sets(v)) };
    const ctl = controlModel(ix, node, desk, words, label, { values: r.choices.map(([, v]) => (typeof v === 'object' ? NaN : v)), checked: c.checked });
    if (ctl) m.control = ctl;
  } else {
    const ctl = controlModel(ix, node, desk, words, label);
    if (ctl) m.control = ctl;
  }
  if (node.dim && key) m.dimWords = m.dim ?? fillMsg(ix, `${key}.dim`, node.dim.says, desk);
  /* a container's children, and a bespoke part's that holds children it places itself (the cartridges' stack bar) */
  if ((spec?.kind === 'container' || (spec?.kind === 'bespoke' && node0.children?.length)) && key) m.kids = kidsOf(ix, node0, desk, key);
  const ro = readoutModel(ix, node, desk);
  if (ro) m.readout = ro;
  const ins = instrumentModel(ix, node, desk, key);
  if (ins) m.instrument = ins;
  if (node.state?.length) m.state = node.state.filter(k => ix.states.has(k)).map(k => { const hit = ix.states.get(k)!; return { key: k, kind: hit.s.kind, value: stateOf(ix, k, desk) ?? null, lent: hit.d.id !== d.id || !!opts.from, owner: hit.d.id, noun: stateWords(ix, k).noun, ...fileDetail(hit.s.kind, stateOf(ix, k, desk), desk), ...(opts.wave && hit.s.kind === 'file' ? fileWave(stateOf(ix, k, desk)) : {}) }; });
  /* a meter that reads a number is on that number's scale (its range on the engine): stated once, in the parameter */
  if (node.part === 'meter') { const r = node.reads?.[0], range = r && ix.params.has(r) && has(ix, r, desk.engine) ? resolve(ix.params.get(r)!.p, desk.engine).range : undefined; m.meter = meterScale(range ? { min: range[0], max: range[1], ...opts } : opts, m.reads[0]?.value); }
  if (node.part === 'modelPicker') m.models = modelRows(ix, node, desk);
  if (node.part === 'unitLcd') m.lines = lcdLines(ix, node, desk, key, m);
  /* the lyric sheet and the prompt card (src/derive/fields.ts) */
  if (node.part === 'lyricSheet') { const s = sheetModel(ix, node, desk); if (s) m.sheet = s; }
  if (node.part === 'promptBox' && node.param) m.prompt = promptModel(ix, node.param, desk);
  return m;
}

function readModel(ix: Index, k: string, desk: Desk): ReadModel {
  if (ix.params.has(k)) { const on = has(ix, k, desk.engine); return { key: k, value: on ? valueOf(ix, k, desk) ?? null : null, text: on ? textMsg(ix, k, desk) : '', lcd: on ? lcdMsg(ix, k, desk) : '', onDesk: true, ...(on && changed(ix, k, desk) ? { changed: true } : {}) }; }
  return { key: k, value: desk.state[k] ?? null, text: '', lcd: '', onDesk: true };
}

/** A side unit's LCD: its text as one line, else each of what it reads on a line of its own, the first in capitals
 *  after its prefix and the rest in small letters (28-side-units.js). */
function lcdLines(ix: Index, node: Node, desk: Desk, key: string, m: PartModel): { text: Slot; upper: boolean }[] {
  if (m.words.text) return [{ text: m.words.text, upper: false }];
  const prefix = key ? nodeOptWord(ix, node, desk, key, 'prefix') : undefined;
  return m.reads.map((r, i) => ({ text: i === 0 && prefix ? msg('lcd.prefixed', { prefix, text: r.lcd }) : r.lcd, upper: i === 0 }));
}

/* ---------------------------------------------------------------- a meter's scale */


/** A meter's printed scale (27-pedals.js vuHTML): its marks (where each stands from 0 to 1, its words, whether its tick
 *  is long, whether it is in the red), where the red starts, and where the needle points (degrees from upright), or null
 *  at rest. A meter that reads a value has a scale of its own from its node's min to max in five marks; one that reads
 *  nothing is the concept's VU, -20 to +3, red from 0. Numbers are written as the core writes them (a true minus). */
export interface MeterScale { marks: { at: number; words: string | null; long: boolean; red: boolean }[]; red: number; needle: number | null; caption: Msg | null }
const VU: [number, number][] = [[-20, 0], [-10, 0.26], [-7, 0.38], [-5, 0.48], [-3, 0.58], [-1, 0.67], [0, 0.72], [1, 0.79], [2, 0.88], [3, 1]];
const signed = (n: number, minus = '−') => (n > 0 ? '+' + n : n < 0 ? minus + -n : '0');
export function meterScale(opts: Record<string, unknown>, value: unknown): MeterScale {
  const lo = typeof opts.min === 'number' ? opts.min : null, hi = typeof opts.max === 'number' ? opts.max : null;
  if (typeof value === 'number' && lo !== null && hi !== null) {
    const at = (x: number) => (x - lo) / (hi - lo), red = at(typeof opts.red === 'number' ? opts.red : hi);
    const marks = Array.from({ length: 5 }, (_, i) => { const x = Math.round((lo + (hi - lo) * i / 4) * 10) / 10, f = i / 4; return { at: f, words: signed(x), long: i % 2 === 0, red: f >= red }; });
    return { marks, red, needle: -48 + Math.min(1, Math.max(0, at(value))) * 96, caption: null };
  }
  /* the concept's VU, as it prints it: long ticks at every fifth decibel and above 0, words at -20, -10, -5, 0, +3
     (with a hyphen-minus), red from 0, and its caption */
  return { marks: VU.map(([db, f]) => ({ at: f, words: [-20, -10, -5, 0, 3].includes(db) ? signed(db, '-') : null, long: db % 5 === 0 || db > 0, red: db >= 0 })), red: 0.72, needle: null, caption: msg('meter.vu') };
}

/* ---------------------------------------------------------------- how a hand or a key moves a value */

/** Where a key takes a number (arrows a step, Shift ten, Page Up and Down ten, Home and End the ends), or null when the
 *  key is not one of the value contract's. The one statement of the contract: every number part asks it. */
export function keyStep(ix: Index, key: string, desk: Desk, k: string, shift: boolean): number | null {
  const f = resolve(ix.params.get(key)!.p, desk.engine), v = valueOf(ix, key, desk);
  if (!f.range || typeof v !== 'number') return null;
  const [lo, hi, step] = f.range;
  if (k === 'Home') return lo;
  if (k === 'End') return hi;
  const d = ({ ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1, PageUp: 10, PageDown: -10 } as Record<string, number>)[k];
  if (!d) return null;
  return +(v + d * (shift && Math.abs(d) === 1 ? 10 : 1) * step).toFixed(10);
}

/** The value a hand's place on a control stands for (0 to 1 along its range). */
export function fractionValue(ix: Index, key: string, desk: Desk, f: number): number | null {
  const r = resolve(ix.params.get(key)!.p, desk.engine).range;
  return r ? r[0] + Math.min(1, Math.max(0, f)) * (r[1] - r[0]) : null;
}

/** A typed number (the inverse of how the page writes it): a true minus, a unit after it and m:ss for seconds are
 *  read; anything else that is no number is refused with the words to say. */
export function parseTyped(ix: Index, key: string, desk: Desk, typed: string): { value: number } | { error: Msg } {
  const f = wordsOf(ix, ix.params.get(key)!.p, desk);
  let t = typed.trim().replace(/−/g, '-');
  if (f.unit && t.endsWith(f.unit)) t = t.slice(0, -f.unit.length).trim();
  const mmss = /^(-?\d+):([0-5]?\d(?:\.\d+)?)$/.exec(t);
  const n = mmss ? +mmss[1]! * 60 + +mmss[2]! : /^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(t) ? parseFloat(t) : NaN;
  return Number.isFinite(n) ? { value: n } : { error: msg('knob.typeNumber', { label: paramWords(ix, key, desk).label }) };
}

/* ---------------------------------------------------------------- the shells */

export interface TileModel {
  id: string;
  /** Its width classes and the classes it stands with. */
  cls: string;
  attrs: Record<string, string>;
  name: Slot;
  group: Slot;
  /** The engine does not use it: what the tile says. */
  unsup: Slot | null;
  /** `held`: what a value of its own held for another engine says, the one place it shows while the tile is a strip
   *  (RD-03: "Your 4 takes return on ACE-Step 1.5" under Takes' one-song strip). */
  strip: { says: Slot; actions: { label: Slot; tip?: Slot }[]; held: Slot | null } | null;
  also: { text: Slot; link: Tier; ring: string; linkWords: Slot } | null;
  board: boolean;
}
/** A tile's shell (30-tile.js tileHTML): its width at the tier and on the engine, its group and name, the note when
 *  the engine does not use it, the strip it becomes when it has nothing to show, and the "Also set" line. */
export function tileModel(ix: Index, d: ModuleDef, desk: Desk): TileModel {
  const unsup = !usedHere(d, desk.engine), strip = unsup ? null : stripOf(ix, d, desk);
  const size = { ...d.size, ...((d.per || []).find(l => l.engines.includes(desk.engine))?.size ?? {}) };
  let w = desk.tier === 'quick' ? (size.quickW === 4 ? 'w4' : size.quickW === 2 ? 'w2' : '') : size.w === 2 ? 'w2' : size.w === 4 ? 'w4' : '';
  if (size.narrowWhen && desk.tier !== 'lab' && holds(ix, size.narrowWhen, desk)) w = '';
  if (size.foldAt3 && desk.tier !== 'quick') w += ' fold3';
  if (strip) w = size.stripKeepsWidth ? `${w} is-strip` : 'is-strip';
  const attrs: Record<string, string> = {};
  if (size.grow && !strip && !/w4/.test(w)) attrs['data-grow'] = '';
  if (size.fold && /w2/.test(w)) attrs['data-fold'] = '';
  if (size.stretch && !unsup && !strip) attrs['data-stretch'] = String(size.stretch);
  if (strip && !unsup) attrs['data-span'] = '';
  const parent = d.labUnder ? ix.defs.find(x => x.id === d.labUnder) : undefined;
  /* a strip says what is set at another tier too, as the concept's every tile does (E-25: Build by section's strip at
     Quick names Studio) */
  const a = unsup ? null : alsoSet(ix, d, desk);
  const s = strip ? stripWords(ix, `create.${d.id}.strip`, strip, desk) : null;
  return {
    /* a source module (Repaint) carries the whole-box ring of the module working on a song (RD-09) */
    id: d.id, cls: `tile ${w} ${d.labUnder ? 'labmod' : ''} ${d.source ? 'source' : ''} ${unsup ? 'unsup' : ''}`, attrs,
    name: tileName(d, desk), group: parent ? msg('tile.group.lab', { parent: defName(parent, desk) }) : categoryWord(d.category),
    unsup: unsup ? msg('tile.unsup', { engine: engineName(desk.engine) }) : null,
    strip: s ? { says: s.says, actions: s.actions.map(x => ({ label: x.label, ...(x.tip ? { tip: x.tip } : {}) })), held: d.params.map(p => (p.held ? heldNote(ix, p.key, desk) : null)).find(x => !!x) ?? null } : null,
    also: a ? { text: a.text, link: a.link, ring: a.keys.join(' '), linkWords: alsoLinkWords(a) } : null,
    board: !!d.board,
  };
}

/** The module that holds the board, and its chain switch: its one switch, the module's `on` (post.ts `pp.on`). */
export const boardOf = (ix: Index): ModuleDef => ix.defs.find((d): d is ModuleDef => d.kind === 'module' && !!d.board)!;
export const chainOf = (ix: Index): string | null => boardOf(ix).params.find(p => p.type === 'boolean')?.key ?? null;
const chainOn = (ix: Index, desk: Desk) => { const c = chainOf(ix); return c === null || valueOf(ix, c, desk) === true; };

/** Where a pedal or unit stands: on, bypassed (its switch off), held (on, but the chain is bypassed), idle (on with
 *  nothing to run), always (no switch: it always runs). */
export type PedalState = 'on' | 'bypassed' | 'held' | 'idle' | 'always';
export function pedalState(ix: Index, d: PedalDef | UnitDef, desk: Desk): PedalState {
  if (!d.on) return 'always';
  if (valueOf(ix, d.on, desk) !== true) return 'bypassed';
  if (d.kind === 'pedal' && !chainOn(ix, desk)) return 'held';
  if (d.kind === 'unit' && d.gated && !chainOn(ix, desk)) return 'held';
  return idleOf(ix, d, desk) ? 'idle' : 'on';
}

export interface BoardModel { chainOn: boolean; count: Slot; pedals: string[]; units: string[] }
/** The board: whether the chain runs, its count, and the pedals and units the engine uses, in chain order. */
export function boardModel(ix: Index, desk: Desk, board: readonly string[]): BoardModel {
  const used = (k: 'pedal' | 'unit') => board.filter(id => { const x = ix.defs.find(d => d.id === id); return x?.kind === k && usedHere(x, desk.engine); });
  return { chainOn: chainOn(ix, desk), count: boardCount(ix, desk, chainOf(ix)), pedals: used('pedal'), units: used('unit') };
}

/** A laid-out cell of a pedal or unit, with the core's words for it. */
export interface CellModel { part: string; param?: string; node?: Node; plate?: Slot | null; beside?: string; shown: boolean; words: CellWords; label?: Slot; text: boolean }
const cellOf = (ix: Index, d: Definition, c: Cell, desk: Desk): CellModel => {
  const base = { part: c.part, shown: c.shown, ...(c.param ? { param: c.param } : {}), ...(c.node ? { node: c.node } : {}), ...(c.plate !== undefined ? { plate: c.plate } : {}), ...(c.beside ? { beside: c.beside } : {}), text: !!c.param && ix.params.get(c.param)?.p.type === 'text' };
  if (!c.param || c.part === 'plate') return { ...base, words: {} };
  const w = paramWords(ix, c.param, desk), f = w.face;
  /* a pedal knob prints its short name below Lab and its full name at Lab, says its full name, and its tip names it
     before how to work it; a unit that is not a percent prints on its own line under the number */
  if (c.part === 'knob') {
    const how = { id: 'part.knob.howTo', en: PARTS.knob.howTo };
    return { ...base, words: { small: true, bare: !!f.unit && f.unit !== '%', label: f.short && w.short ? w.short : w.label, aria: w.label, title: msg('pedal.tip', { a: w.label, b: how }) } };
  }
  if (c.part === 'ledButton') return { ...base, words: { small: true } };
  return { ...base, words: {}, label: w.label };
  void d;
};

export interface PedalModel {
  id: string; kind: 'pedal' | 'unit'; state: PedalState; cls: string; hue: number | null;
  name: Slot; full: Slot; tip: Slot | null; idle: Slot | null;
  /** The foot switch (or a unit's power): its key and whether it is on, with its words; null: always on. */
  foot: { key: string; on: boolean; label: Slot; title: Slot | null } | null;
  units: 1 | 2; cols: number; tier: Tier;
  knobs: CellModel[]; plates: CellModel[]; sels: CellModel[]; nodes: CellModel[]; switches: CellModel[]; mid: CellModel[];
  gated: boolean; waits: boolean; wide: boolean;
}
/** A pedal or a side unit on the board (27-pedals.js pedalHTML, 28-side-units.js): its state and classes, its name at
 *  the tier, its tip, its foot switch, and its body laid out at the tier: knobs (and the plates of hidden ones),
 *  selectors, its own nodes, switches, and at Studio its main knob or its Studio nodes. */
export function pedalModel(ix: Index, d: PedalDef | UnitDef, desk: Desk): PedalModel {
  const state = pedalState(ix, d, desk), lab = desk.tier === 'lab', full = defName(d, desk);
  const cells = boardNodes(ix, d, desk).cells.map(c => cellOf(ix, d, c, desk));
  const idle = d.kind === 'pedal' ? idleOf(ix, d, desk) : null;
  const name = d.kind === 'pedal' && d.short && !lab ? { id: `create.${d.id}.short`, en: d.short } as Msg : full;
  const knobs = d.kind === 'unit' && !lab ? [] : cells.filter(c => c.part === 'knob' && !c.beside && c.shown), plates = cells.filter(c => c.part === 'plate');
  const sels = cells.filter(c => (c.part === 'pick' || c.part === 'textField' || c.part === 'chips') && !c.beside);
  const nodes = cells.filter(c => c.part === 'node');
  const switches = d.kind === 'unit'
    ? (lab ? cells.filter(c => c.part !== 'node' && c.part !== 'knob' && c.part !== 'plate' && c.param !== d.on).sort((a, b) => d.params.findIndex(p => p.key === a.param) - d.params.findIndex(p => p.key === b.param)) : [])
    : cells.filter(c => (c.part === 'ledButton' && c.param !== d.on) || !!c.beside);
  const n = knobs.length + plates.length;
  const main = desk.tier === 'studio' ? cells.find(c => c.part === 'knob') : undefined;
  const on = !!d.on && valueOf(ix, d.on, desk) === true;
  const tip = [d.kind === 'pedal' && d.short && !lab ? full : null, idle ? idle.says : null].filter((x): x is Msg => !!x);
  /* a pedal runs when it is on (or has no switch) and the chain runs; held: on, while the chain is bypassed */
  const runs = (!d.on || on) && chainOn(ix, desk);
  const cls = d.kind === 'pedal'
    ? `pedal u${lab && knobs.length >= 3 ? 2 : 1} ${runs ? '' : 'off'} ${state === 'held' ? 'held' : ''} ${idle ? 'idle' : ''} ${state === 'bypassed' ? 'byp' : ''} ${d.on ? '' : 'neutral'}`
    : `pedal xunit ${d.wide ? 'u-wide' : ''} ${state === 'on' ? '' : 'off'}`;
  return {
    id: d.id, kind: d.kind, state, cls, hue: d.kind === 'pedal' ? d.hue : null, name, full, idle: idle ? idle.says : null,
    tip: tip.length ? list(tip, 'sentences') : null,
    foot: d.on ? { key: d.on, on, label: msg(on ? 'pedal.off' : 'pedal.on', { name: full }), title: d.kind === 'unit' ? msg(on ? 'unit.off' : 'unit.on') : null } : null,
    units: lab && knobs.length >= 3 ? 2 : 1, cols: n <= 5 ? n : Math.ceil(n / 2), tier: desk.tier,
    knobs, plates, sels, nodes, switches, mid: main ? (main.shown ? [main] : nodes) : desk.tier === 'studio' ? nodes : [],
    gated: d.kind === 'unit' && !!d.gated, waits: d.kind === 'unit' && !!d.gated && !chainOn(ix, desk), wide: d.kind === 'unit' && !!d.wide,
  };
}

/* ---------------------------------------------------------------- the desk */

export interface DeskModel {
  tier: Tier; engine: EngineId;
  /** The tiles drawn, in the user's order: on the desk, at this tier, used by this engine. */
  tiles: string[];
  /** The lyric sheet's place: a tile of its own on this engine, or a row the desk keeps for it. */
  lyricsHere: boolean;
  empty: { quick: boolean; says: Slot | null; starters: { id: string; name: Slot }[] } | null;
  parked: { says: Slot; modules: { id: string; name: Slot; tag: ParkTag | null }[]; board: { id: string; name: Slot; what: Slot; tag: ParkTag | null }[] } | null;
  hidden: Slot | null;
  /** The desk's make-up: what, when it changes, lays the desk out afresh (layout.ts; the view adds the window's
   *  width). */
  make: string;
}
export interface ParkTag { words: Slot; tip: Slot }
/** The engines something works on, as a parked chip or a Modules card tags it: their short names for one or two, a
 *  count for more, and the full names in its tip. */
const enginesTag = (d: Definition, words: 'parked' | 'card'): ParkTag | null => d.engines.length >= ENGINES.length ? null : {
  words: d.engines.length > 2 ? msg(`${words}.engines`, { n: d.engines.length }) : list(d.engines.map(e => ({ id: `engine.${e}.tag`, en: engineTag(e) })), 'dot'),
  tip: msg('parked.worksOn', { engines: list(d.engines.map(engineName)) }),
};
/** A group's words: a module's category, or Pedals, or Side units (cardGroups). */
export const categoryWord = (c: string): Msg => ({ id: `cat.${c}`, en: c });
/** The empty desk's offers: the modules marked `starter`, in that order, that the engine uses and that are not on the
 *  desk; three at most. */
const starterLine = (d: ModuleDef, desk: Desk) => { const l = (d.per || []).find(x => x.engines.includes(desk.engine) && x.starter); return l ? `@${l.engines.join('+')}` : ''; };
export function startersOn(ix: Index, desk: Desk): ModuleDef[] {
  const at = (d: ModuleDef) => ((d.per || []).find(l => l.engines.includes(desk.engine) && l.starter)?.starter ?? d.starter)!;
  return ix.defs.filter((d): d is ModuleDef => d.kind === 'module' && !!d.starter && usedHere(d, desk.engine) && !desk.on.has(d.id)).sort((a, b) => at(a).at - at(b).at).slice(0, 3);
}
export function deskModel(ix: Index, desk: Desk, mods: readonly string[]): DeskModel {
  const defs = mods.map(id => ix.defs.find(d => d.id === id)).filter((d): d is ModuleDef => d?.kind === 'module');
  const tiles = defs.filter(d => tileAtTier(d, desk) && usedHere(d, desk.engine)).map(d => d.id);
  const p = parkedOn(ix, desk), lyrics = ix.defs.find(d => d.kind === 'module' && d.builtIn && d.face.some(n => n.part === 'lyricSheet'));
  const st = tiles.length ? [] : startersOn(ix, desk);
  const first = st[0] ? ((st[0].per || []).find(l => l.engines.includes(desk.engine) && l.starter)?.starter ?? st[0].starter) : undefined;
  return {
    tier: desk.tier, engine: desk.engine, tiles, lyricsHere: !!lyrics && usedHere(lyrics, desk.engine),
    empty: tiles.length ? null : {
      quick: desk.tier === 'quick',
      says: desk.tier === 'quick' ? msg('qempty', { first: first?.noun ? { id: `create.${st[0]!.id}.starter${starterLine(st[0]!, desk)}`, en: first.noun } : '' }) : null,
      starters: st.map(d => ({ id: d.id, name: tileName(d, desk) })),
    },
    parked: p.says ? {
      says: p.says,
      modules: p.modules.map(d => ({ id: d.id, name: tileName(d, desk), tag: enginesTag(d, 'parked') })),
      board: p.board.map(d => ({ id: d.id, name: defName(d, desk), what: msg(d.kind === 'unit' ? 'parked.unit' : 'parked.pedal'), tag: enginesTag(d, 'parked') })),
    } : null,
    hidden: hiddenLabModules(ix, desk),
    make: [desk.tier, desk.engine, mods.join()].join('|'),
  };
}

/* ---------------------------------------------------------------- the desk's make-up */

/** What adding something does to the desk's make-up (30-desk.js addItem): a module goes before `before` (or back where
 *  it stood when it was removed, or at the end); a pedal or unit takes its place on the board, and the first brings
 *  the board's module. 'there': it is already on the desk; 'lab': a Lab module, outside Lab. */
export function addTo(ix: Index, desk: Desk, mods: readonly string[], board: readonly string[], id: string, before: string | null | undefined, gone: Readonly<Record<string, string | null>>):
  { did: 'added'; mods: string[]; board: string[] } | { did: 'there' | 'lab' } {
  const d = ix.defs.find(x => x.id === id); if (!d) return { did: 'there' };
  if (d.kind === 'module' && d.labUnder && desk.tier !== 'lab') return { did: 'lab' };
  if (desk.on.has(d.id)) return { did: 'there' };
  let m = [...mods], b = [...board];
  const insert = (x: string, at: string | null | undefined) => { m = m.filter(y => y !== x); const i = at ? m.indexOf(at) : -1; if (i >= 0) m.splice(i, 0, x); else m.push(x); };
  if (d.kind === 'module') { const back = gone[d.id]; insert(d.id, before === undefined && back && m.includes(back) ? back : before); }
  else { b = boardOrder(ix, [...b, d.id]); const host = boardOf(ix).id; if (!m.includes(host)) insert(host, before); }
  return { did: 'added', mods: m, board: b };
}
/** What removing something does (30-desk.js removeMod, 27-pedalboard.js): a module leaves (the board's module takes
 *  its pedals and units with it), and where it stood is kept so adding it back puts it there (the next module that is
 *  not a source); a pedal or unit leaves the board, and the last takes the board's module with it. */
export function removeFrom(ix: Index, mods: readonly string[], board: readonly string[], id: string):
  { mods: string[]; board: string[]; stood: string | null; pedals: number; last: boolean } {
  const d = ix.defs.find(x => x.id === id)!;
  if (d.kind === 'module') {
    const at = mods.indexOf(id), stood = mods.slice(at + 1).find(x => !(ix.defs.find(y => y.id === x) as ModuleDef | undefined)?.source) ?? null;
    return { mods: mods.filter(x => x !== id), board: d.board ? [] : [...board], stood, pedals: d.board ? board.length : 0, last: false };
  }
  const b = board.filter(x => x !== id), last = !b.length;
  return { mods: last ? mods.filter(x => x !== boardOf(ix).id) : [...mods], board: b, stood: null, pedals: 0, last };
}
/** Pedals in chain order, then side units (each by `at`). */
export function boardOrder(ix: Index, ids: Iterable<string>): string[] {
  const set = new Set(ids), of = (k: 'pedal' | 'unit') => ix.defs.filter(d => d.kind === k && set.has(d.id)).sort((a, b) => a.at - b.at).map(d => d.id);
  return [...of('pedal'), ...of('unit')];
}
/** The desk a first visit starts on. */
export const emptyDesk = (ix: Index): Desk => ({ engine: ENGINES[0]!.id, tier: 'quick', vals: {}, auto: {}, state: {}, on: onOf(ix, [], []) });

/* ---------------------------------------------------------------- the Modules panel */

export interface CardModel { id: string; name: Slot; tip: Slot; why: Slot; locked: boolean; notHere: boolean; on: boolean; lab: boolean; tag: ParkTag | null; rank: number; group: Slot; thumbName: string }
/** The groups of the Modules panel: the modules' categories in the order the definitions give them, then pedals, then
 *  side units. */
export const cardGroups = (): string[] => [...CATEGORIES, 'Pedals', 'Side units'];
export const groupOf = (d: Definition): string => (d.kind === 'pedal' ? 'Pedals' : d.kind === 'unit' ? 'Side units' : d.category);
/** A card of the Modules panel: its name at the tier, its tip, whether it is locked (a Lab module outside Lab), not
 *  used here, already on the desk, its engines' tag, and how well it matches the filter (`words`: the name and
 *  description as the page shows them, in its language): 1 a name that starts so, 2 one that holds it, 3 the words
 *  around it, 0 none. */
export function cardModel(d: Definition, desk: Desk, query: string, words: { name: string; desc: string; group: string }): CardModel {
  const name = tileName(d, desk), locked = d.kind === 'module' && !!d.labUnder && desk.tier !== 'lab', notHere = !usedHere(d, desk.engine), on = desk.on.has(d.id);
  const q = query.toLowerCase().trim(), nm = words.name.toLowerCase();
  const rank = !q ? 1 : nm.startsWith(q) || nm.split(/[\s&-]+/).some(w => w.startsWith(q)) ? 1 : nm.includes(q) ? 2
    : q.length >= 3 && new RegExp('(^|[^\\p{L}\\p{N}])' + q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'u').test((words.desc + ' ' + words.group).toLowerCase()) ? 3 : 0;
  const engines = list(d.engines.map(engineName));
  const tip = notHere ? msg(locked ? 'card.notUsedLab' : 'card.notUsed', { engine: engineName(desk.engine), engines }) : locked ? msg('card.lab') : on ? msg('card.on') : { id: `create.${d.id}.desc`, en: d.desc };
  return { id: d.id, name, tip: msg('card.label', { name, tip }), why: tip, locked, notHere, on, lab: d.kind === 'module' && !!d.labUnder, tag: enginesTag(d, 'card'), rank, group: categoryWord(groupOf(d)), thumbName: words.name };
}

