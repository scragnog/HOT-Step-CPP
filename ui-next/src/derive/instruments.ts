// What the bespoke instruments of the Tempo, Seed, Takes, Sampler, Repaint, Timbre reference and Latent & correction
// tiles draw, and what a hand on them does to a value (docs/09, package 3a). Each is the concept's own formula
// (25-module-tempo.js wheelSVG and its keys, the tap and the prompt chips; 25-module-seed.js; 25-module-takes.js and
// 35-gobar.js vramNeed; 25-module-sampler.js sigmas, scopeSVG and scopeRead; 25-module-repaint.js; 25-module-timbre.js;
// 25-module-latent.js distSVG), moved here so the parts draw numbers and words they are given and hand back what the
// hand did. What differs per engine or per module is said in the definitions (a node's reads, params, state and
// options); the formulas are keyed by the part that draws them, as the inventory names it (parts.ts PARTS), never by a
// module or an engine.
//
//   instrumentModel(ix, node, desk)   PartModel.instrument: what a bespoke part draws
//   partAct(ix, node, props, desk, name, arg, tell)   what a hand on a bespoke part does, through its own setters

import type { Json, Node, Value } from '../definitions/types.ts';
import type { PartProps } from '../definitions/parts.ts';
import { CHECKS } from '../definitions/parts.ts';
import { SCHEDULE_CURVES } from '../definitions/option-faces.ts';
import { CARD_GB } from './controls.ts';
import { type Desk, type Index, applies, formatValue, has, isAuto, mmss, nodeAt, resolve, stateOf, valueOf } from './core.ts';
import { nodeWord } from './strings.ts';
import { EN, type Msg, type Slot, msg } from './messages.ts';
import { spellKey } from './readouts.ts';
import { nodeShown } from './desk.ts';

/* the core's words for these instruments, added to its table (messages.ts EN) where this file is loaded */
Object.assign(EN, {
  'flap.random': 'Random seed',
  'flap.title': 'Click to type or paste a seed',
  'flap.title.rand': 'Random: the next run draws a fresh seed. Lock to keep {seed}, or click to type one',
  'flap.aria': 'Seed {seed}. Click to type another',
  'flap.aria.rand': 'Seed: random. Click to type one',
  'flap.aria.randKept': 'Seed: random. Lock to keep {seed}. Click to type one',
  'flap.bad': 'A seed is a whole number from 0 to {max}. Leave it empty for random.',
  'flap.bad.left': 'A seed is a whole number from 0 to {max}, so it was left as it was. Leave it empty for random.',
  'flap.lm.off': 'LM seed: random',
  'flap.lm.rand': 'random, like the DiT',
  'flap.lm.uses': 'uses {seed} too',
  'stars.first': 'Roll or load a seed first. Random draws are saved with each song.',
  'stars.saveFirst': 'Save a seed first',
  'stars.loaded': 'Loaded “{name}” ({seed})',
  'stars.loadedPlain': 'Loaded saved seed {seed}',
  'stars.saved': 'Saved seed {seed}',
  'stars.updated': 'Updated saved seed {seed}',
  'stars.deleted': 'Deleted {name}',
  'stars.name': 'seed_{seed}',
  'takes.cost': '{need} of {card} GB',
  'takes.bar': 'About {need} GB of the {card} GB card',
  'takes.each': '+{gb}\u00a0GB per take',
  'takes.eachExtra': '+{gb}\u00a0GB per\u00a0extra take',
  'takes.wontFit': "Won't fit. Try {n} takes.",
  'takes.passes': 'Plans in {n} passes to fit the card',
  'takes.tight': 'Tight on the card',
  'takes.valuetext': '{n, plural, one {# take} other {# takes}}',
  'scope.native': 'native sampling',
  'scope.ts': '{n} custom timesteps from Latent & correction',
  'scope.everyStep': 'guidance on every step',
  'scope.from': 'guidance from {a} to {b}',
  'scope.until': 'guidance until {b}',
  'scope.off': 'guidance off: the window holds no step',
  'scope.cut': ' (CFG cutoff)',
  'region.cap': 'Will repaint {a} to {b}',
  'region.text': '{a} to {b}',
  'region.song': '{style} · {dur}',
  'blend.pct': '{v}%',
});

/* ---------------------------------------------------------------- what each part draws */

export interface InstrumentModel {
  /** keyWheel: each segment (its option, its ring and place, its letter, accidental and whether it is a minor, its
   *  spelled name, chosen or not), the centre's Auto, and whether its LCD stands without it (`solo`: it edits nothing
   *  at this tier or on this engine). */
  wheel?: { segs: { value: string; ring: 'maj' | 'min'; i: number; letter: string; acc: string; minor: boolean; name: string; sel: boolean }[]; auto: { value: string; sel: boolean }; solo: boolean };
  /** capChips: each chip's words, whether the text it edits holds it as an item, and the step a press takes. */
  caps?: { chips: { text: string; on: boolean }[] };
  /** splitFlap: the ten cells' characters, random or a kept seed, the field's text, and its words. */
  flap?: { cells: string[]; rand: boolean; edit: string; title: Slot; aria: Slot; lm: Slot | null; lock: boolean };
  /** savedSeeds: the saved seeds, starred first, each with what a search finds it by; the seed shown; the save form's
   *  defaults (null when there is no seed to save). */
  stars?: { rows: { v: number | string; name: string; desc: string; tags: string[]; star: boolean; cur: boolean; find: string }[]; form: { old: boolean; name: string; desc: string; tags: string } | null };
  /** takesReels: how many reels, how many lit, the grid's columns, and the VRAM the count costs. */
  takes?: { max: number; n: number; cols: number; valuetext: Slot; labels: [Slot, Slot]; cost: { bar: number; text: Slot; tip: Slot; cls: '' | 'split' | 'over'; msg: Slot | null; each: Slot | null } };
  /** samplerScope: the schedule as points in a 300 x 110 box, the step dots (percent of the box) while 32 or fewer,
   *  the guidance band and the hatched CFG cutoff, the read-out's pieces (with the node's action each offers), and the
   *  interval handles' places (percent across the box). */
  scope?: { points: string; dots: { x: number; y: number }[]; band: { x: number; w: number } | null; cut: { x: number; w: number } | null; read: { text: Slot; tone: '' | 'warn' | 'amber' | 'ts'; action: number | null }[]; handles: { key: string; at: number; flip: boolean }[] };
  /** latentCurve: the grid's x places, the zero line, and the curves before and after (points in a 300 x 80 box). */
  dist?: { grid: number[]; zero: number; before: string; after: string };
  /** repaintRegion: the song (its title, style line, length, hue), the region as fractions of it, and its words. */
  region?: { title: string; style: Slot; dur: number; durText: string; hue: number; id: number; a: number; b: number; cap: Slot; text: Slot; ctl: boolean; one: boolean };
  /** timbreBlend: the blend as its percent. */
  blend?: { pct: Slot };
}

/** What a bespoke part draws on the desk, or undefined for a part this file does not draw. */
export function instrumentModel(ix: Index, node: Node, desk: Desk, key: string): InstrumentModel | undefined {
  const f = DRAW[node.part];
  return f ? f(ix, node, desk, key) : undefined;
}

const DRAW: { [part: string]: (ix: Index, node: Node, desk: Desk, key: string) => InstrumentModel } = {
  keyWheel: (ix, node, desk) => ({ wheel: wheelOf(ix, node, desk) }),
  capChips: (ix, node, desk) => ({ caps: { chips: capOffer(node, desk).map(text => ({ text, on: capHas(textOf(ix, node.param, desk), text) })) } }),
  splitFlap: (ix, node, desk) => ({ flap: flapOf(ix, node, desk) }),
  savedSeeds: (ix, node, desk) => ({ stars: starsOf(ix, node, desk) }),
  takesReels: (ix, node, desk, key) => ({ takes: takesOf(ix, node, desk, key) }),
  samplerScope: (ix, node, desk) => ({ scope: scopeOf(ix, node, desk) }),
  latentCurve: (ix, node, desk) => ({ dist: distOf(num(ix, desk, node.reads?.[0], 0), num(ix, desk, node.reads?.[1], 1)) }),
  repaintRegion: (ix, node, desk) => { const r = regionOf(ix, node, desk); return r ? { region: r } : {}; },
  timbreBlend: (ix, node, desk) => ({ blend: { pct: msg('blend.pct', { v: Math.round(num(ix, desk, node.param, 0) * 100) }) } }),
};

/* ---------------------------------------------------------------- reading */

const read = (ix: Index, desk: Desk, k: string | undefined): Value | null => (k && ix.params.has(k) && has(ix, k, desk.engine) ? valueOf(ix, k, desk) ?? null : null);
const num = (ix: Index, desk: Desk, k: string | undefined, d: number): number => { const v = read(ix, desk, k); return typeof v === 'number' ? v : d; };
const textOf = (ix: Index, k: string | undefined, desk: Desk): string => String(read(ix, desk, k) ?? '');
/** A choice's words as the server or the definition lists them (a schedule's curve is keyed by them). */
const labelOf = (ix: Index, k: string | undefined, desk: Desk): string | null => {
  if (!k || !ix.params.has(k) || !has(ix, k, desk.engine)) return null;
  return formatValue(resolve(ix.params.get(k)!.p, desk.engine), valueOf(ix, k, desk)!, desk.lists);
};
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

/* ---------------------------------------------------------------- the key wheel */

/** A key as the wheel prints it: its letter, its accidental as a musician spells it (B♭, F♯), whether it is minor. */
function glyphOf(value: string): { letter: string; acc: string; minor: boolean } {
  const s = spellKey(value), m = /^([A-G])([♯♭]?) (major|minor)$/.exec(s);
  return m ? { letter: m[1]!, acc: m[2]!, minor: m[3] === 'minor' } : { letter: s, acc: '', minor: false };
}
function wheelOf(ix: Index, node: Node, desk: Desk): NonNullable<InstrumentModel['wheel']> {
  const k = node.param ?? node.params?.[0];
  if (!k || !has(ix, k, desk.engine)) return { segs: [], auto: { value: '', sel: false }, solo: true };
  const f = resolve(ix.params.get(k)!.p, desk.engine), v = String(valueOf(ix, k, desk) ?? '');
  const opts = Array.isArray(f.options) ? f.options : [];
  const ring = (r: 'major' | 'minor') => opts.filter(o => String(o.value).endsWith(' ' + r)).map(o => String(o.value));
  const segs = (['maj', 'min'] as const).flatMap(rg => ring(rg === 'maj' ? 'major' : 'minor').map((value, i) => ({ value, ring: rg, i, ...glyphOf(value), name: spellKey(value), sel: value === v })));
  const auto = opts.find(o => !/ (major|minor)$/.test(String(o.value)));
  return { segs, auto: { value: String(auto?.value ?? ''), sel: v === String(auto?.value ?? '') }, solo: false };
}
/** Where a key takes the wheel from the value in force (not the focused segment: two keys inside one frame both count,
 *  Q-33): round the ring, between the rings, to Auto and back. Null for a key it does not take. */
export function wheelWalk(values: string[], autoValue: string, cur: string, key: string): string | null {
  const maj = values.filter(x => x.endsWith(' major')), min = values.filter(x => x.endsWith(' minor'));
  const on = cur === autoValue ? null : cur.endsWith(' major') ? maj : min, i = on ? on.indexOf(cur) : 0;
  const at = (r: string[], j: number) => r[(j + 12) % r.length] ?? null;
  switch (key) {
    case 'ArrowRight': return on ? at(on, i + 1) : at(maj, 0);
    case 'ArrowLeft': return on ? at(on, i - 1) : at(maj, maj.length - 1);
    case 'ArrowUp': return on === null ? at(min, 0) : on === min ? at(maj, i) : null;
    case 'ArrowDown': return on === null ? null : on === maj ? at(min, i) : autoValue;
    case 'Home': return autoValue;
    default: return null;
  }
}

/* ---------------------------------------------------------------- the prompt chips (Stable Audio 3, HeartMuLa) */

const CAP_ADD = ['90 BPM', '120 BPM', '140 BPM', 'D major', 'A minor'];
/** The chips offered: the tempo and key the node reads first (the user's own, kept in the song's values when they were
 *  set on another engine: S-11), then the usual ones. */
function capOffer(node: Node, desk: Desk): string[] {
  const own: string[] = [], [bk, kk] = node.reads ?? [];
  const b = bk ? desk.vals[bk] : undefined, k = kk ? desk.vals[kk] : undefined;
  if (bk && desk.auto[bk] === false && typeof b === 'number') own.push(`${b} BPM`);
  if (typeof k === 'string' && k) own.push(spellKey(k).replace(/♯/g, '#').replace(/♭/g, 'b'));
  return [...new Set([...own, ...CAP_ADD])];
}
/** Which group a chip is in: a tempo or a key; an item of the same group is replaced by it. */
const capItem = (s: string) => (/BPM$/i.test(s) ? /^\d{2,3}\s*BPM$/i : /^[A-G][#♯b♭]?\s+(major|minor)$/i);
/** The text's items: its words between commas. Only an item of its own counts, never words inside a sentence (S-04). */
const capParts = (v: string) => v.split(',').map(x => x.trim()).filter(Boolean);
export const capHas = (text: string, chip: string): boolean => capParts(text).some(x => x.replace(/\s+/g, ' ').toLowerCase() === chip.toLowerCase());
/** The text after pressing a chip: it replaces any other of its group, or comes out when it was in. */
export function capToggle(text: string, chip: string): { text: string; on: boolean } {
  const on = capHas(text, chip), rest = capParts(text).filter(x => !capItem(chip).test(x));
  return { text: (on ? rest : [...rest, chip]).join(', '), on: !on };
}

/* ---------------------------------------------------------------- the seed */

const FLAPS = 10, SEED_MAX = '18446744073709551615';
/** The seed in force and whether the next run draws a fresh one: one rule at every tier (Q-06), the flaps show a
 *  number only when the next run will use it. */
function seedOf(ix: Index, node: Node, desk: Desk): { seed: number | string | null; rand: boolean } {
  const key = node.state?.find(k => ix.states.get(k)?.s.kind === 'seed'), s = key ? stateOf(ix, key, desk) : undefined;
  const seed = typeof s === 'number' || typeof s === 'string' ? s : null, lock = read(ix, desk, node.param);
  return { seed, rand: seed === null || lock !== true };
}
function flapOf(ix: Index, node: Node, desk: Desk): NonNullable<InstrumentModel['flap']> {
  const { seed, rand } = seedOf(ix, node, desk), show = rand ? '  RANDOM  ' : String(seed).slice(-FLAPS).padStart(FLAPS, ' ');
  /* the LM seed's link: its switch (the flaps' second child) and what it does, while that switch is shown */
  const lmKey = node.reads?.[0], link = node.children?.[1], lm = lmKey && link && nodeShown(ix, link, desk) ? read(ix, desk, lmKey) : null;
  return {
    cells: [...show], rand, edit: seed === null ? '' : String(seed), lock: !!node.children?.[0] && nodeShown(ix, node.children[0], desk),
    title: rand && seed !== null ? msg('flap.title.rand', { seed }) : msg('flap.title'),
    aria: rand ? (seed !== null ? msg('flap.aria.randKept', { seed }) : msg('flap.aria.rand')) : msg('flap.aria', { seed: seed! }),
    lm: lm === null ? null : lm === true ? (rand ? msg('flap.lm.rand') : msg('flap.lm.uses', { seed: seed! })) : msg('flap.lm.off'),
  };
}
/** A seed typed or pasted: spaces, commas and underscores ignored; empty is random; a whole number up to 2^64-1 is
 *  kept (its decimal text past 2^53). */
export function seedTyped(text: string): { seed: number | string | null } | { error: true } {
  const raw = text.replace(/[\s,_]/g, '');
  if (raw === '') return { seed: null };
  if (!/^\d{1,20}$/.test(raw) || BigInt(raw) > BigInt(SEED_MAX)) return { error: true };
  const big = BigInt(raw);
  return { seed: big <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(big) : big.toString() };
}

/** A saved seed, as the list keeps it (the concept's S.stars entry), cleaned. */
export interface SavedSeed { v: number | string; n: string; d: string; tags: string[]; star?: boolean }
export function cleanSeeds(raw: Json | undefined): SavedSeed[] {
  if (!Array.isArray(raw)) return [];
  return raw.flatMap(x => {
    const o = x as { v?: unknown; n?: unknown; d?: unknown; tags?: unknown; star?: unknown };
    if (!o || typeof o !== 'object') return [];
    const v = typeof o.v === 'number' ? o.v : typeof o.v === 'string' ? seedTyped(o.v) : null;
    const seed = typeof v === 'number' ? (Number.isSafeInteger(v) && v >= 0 ? v : null) : v && 'seed' in v ? v.seed : null;
    if (seed === null) return [];
    return [{ v: seed, n: String(o.n ?? '').slice(0, 40), d: String(o.d ?? '').slice(0, 120), tags: Array.isArray(o.tags) ? o.tags.filter((t): t is string => typeof t === 'string').slice(0, 12) : [], ...(o.star === true ? { star: true } : {}) }];
  }).slice(0, 200);
}
const listKey = (ix: Index, node: Node) => node.state?.find(k => ix.states.get(k)?.s.kind === 'list');
const seedName = (s: SavedSeed) => s.n || `seed_${s.v}`;
function starsOf(ix: Index, node: Node, desk: Desk): NonNullable<InstrumentModel['stars']> {
  const list = cleanSeeds(stateOf(ix, listKey(ix, node) ?? '', desk)), { seed } = seedOf(ix, node, desk);
  const rows = [...list].sort((a, b) => (b.star ? 1 : 0) - (a.star ? 1 : 0)).map(s => ({
    v: s.v, name: seedName(s), desc: s.d, tags: s.tags, star: !!s.star, cur: seed !== null && String(s.v) === String(seed),
    find: `${s.n} ${s.d} ${s.tags.join(' ')} ${s.v}`.toLowerCase(),
  }));
  const old = seed === null ? undefined : list.find(s => String(s.v) === String(seed));
  return { rows, form: seed === null ? null : { old: !!old, name: old ? seedName(old) : `seed_${seed}`, desc: old?.d ?? '', tags: (old?.tags ?? []).join(', ') } };
}

/* ---------------------------------------------------------------- takes */

/** What a run costs the card (35-gobar.js vramNeed): the models, the fixed parts, and what each take adds. The engine
 *  panel's package gives the loaded models' real figures; until then the engines' default models stand for them. */
const VRAM: { [e: string]: { models: number; fixed: number; take: number; each: 'all' | 'extra'; lm?: string; split?: boolean; refuses?: boolean } } = {
  ace: { models: 5.3 + 4.5 + 0.34, fixed: 1.2 + 2.0, take: 2.3, each: 'all', lm: 'lm.on', split: true },
  'minimax-m3': { models: 2.6 + 9.1 + 0.69, fixed: 0.8 + 3.0, take: 3.4, each: 'extra', refuses: true },
  yue2: { models: 3.9 + 0.3, fixed: 1.5, take: 1.8, each: 'extra' },
  sa3: { models: 2.9, fixed: 1.6 + 1.2, take: 0, each: 'extra' },
  heartmula: { models: 3.7 + 0.53 + 1.6, fixed: 0.3 + 1.5, take: 0, each: 'extra' },
};
/** The memory a count of takes needs on the desk's engine. */
export function takesCost(ix: Index, desk: Desk, n: number): { models: number; need: number; each: number; passes: number; refuses: boolean } {
  const e = VRAM[desk.engine] ?? VRAM.ace!, slots = stateOf(ix, 'slots', desk);
  /* each loaded adapter adds a little (34-engine-panel.js vramParts) */
  const models = e.models + e.fixed + (Array.isArray(slots) ? 0.15 * Math.min(slots.length, 3) : 0);
  const each = e.lm ? (read(ix, desk, e.lm) === false ? 0 : e.take) : e.take;
  const extra = e.each === 'all' ? n * each : Math.max(0, n - 1) * each, need = models + extra;
  const passes = e.split && need > CARD_GB && each ? Math.ceil(extra / Math.max(each, CARD_GB - models)) : 1;
  return { models, need, each, passes, refuses: !!e.refuses && need > CARD_GB };
}
function takesOf(ix: Index, node: Node, desk: Desk, key: string): NonNullable<InstrumentModel['takes']> {
  const k = node.param!, f = resolve(ix.params.get(k)!.p, desk.engine), max = f.range?.[1] ?? 1, n = Math.min(num(ix, desk, k, 1), max);
  /* one row up to five; past that a grid as near square as the count allows (nine are three by three, as the
     concept's; ACE-Step's eight keep that footprint, three a row: the tile stands as tall as the concept's) */
  const cols = max <= 5 ? max : Math.ceil(Math.sqrt(max));
  const v = takesCost(ix, desk, n), over = v.need > CARD_GB, e = VRAM[desk.engine] ?? VRAM.ace!;
  const gb = (x: number) => x.toFixed(1);
  const m = over ? (v.refuses ? msg('takes.wontFit', { n: Math.max(1, Math.floor((CARD_GB - v.models) / (v.each || 1)) + 1) }) : e.split ? msg('takes.passes', { n: v.passes }) : msg('takes.tight')) : null;
  return {
    max, n, cols, valuetext: msg('takes.valuetext', { n }),
    /* the count's words for one take and for more, as the node says them at each (a reel's preview says its count) */
    labels: [1, 2].map(x => { const at = { ...desk, vals: { ...desk.vals, [k]: x } }, n0 = nodeAt(ix, node, at); return (n0 && key ? nodeWord(ix, n0, at, key, 'label') : undefined) ?? ''; }) as [Slot, Slot],
    cost: {
      bar: Math.min(1, v.need / CARD_GB), text: msg('takes.cost', { need: gb(v.need), card: CARD_GB }), tip: msg('takes.bar', { need: gb(v.need), card: CARD_GB }),
      cls: v.refuses ? 'over' : over ? 'split' : '', msg: m, each: v.each ? msg(e.each === 'all' ? 'takes.each' : 'takes.eachExtra', { gb: v.each }) : null,
    },
  };
}

/* ---------------------------------------------------------------- the Sampler's scope */

/** The two stages of a composite schedule (25-module-sampler.js STAGEF), by their words. */
const STAGES: { [label: string]: (u: number) => number } = {
  'Tangent': u => (1 - Math.tanh(3 * (u - 0.5)) / Math.tanh(1.5)) / 2, 'Linear': u => 1 - u, 'Cosine': u => Math.cos(u * Math.PI / 2),
  'DDIM Uniform': u => 1 - (3 * u * u - 2 * u ** 3), 'SGM-Uniform (Karras)': u => (1 - u) ** 3, 'Linear-Quadratic': u => (u < 0.5 ? 1 - u * 0.5 : 0.75 * (1 - (u - 0.5) * 2) ** 2),
};
function betaCDF(t: number, a: number, b: number): number { const n = 64; let s = 0, tot = 0; for (let i = 0; i < n; i++) { const u = (i + 0.5) / n, f = u ** (a - 1) * (1 - u) ** (b - 1); tot += f; if (u <= t) s += f; } return tot ? s / tot : t; }
/** What the scope reads, by place in its node's `reads`: steps, shift and schedule first (the inventory's signature),
 *  then the schedule shapes' values, the custom timesteps, the CFG cutoff, the solver and the engine's own sampler
 *  choice, and (where the node does not edit them) the CFG interval. */
const SCOPE_READS = ['steps', 'shift', 'sched', 'alpha', 'beta', 'exp', 'a', 'b', 'x', 'split', 'ts', 'cut', 'solver', 'own', 'cs', 'ce'] as const;
type ScopeRead = typeof SCOPE_READS[number];
function scopeKeys(node: Node): { [r in ScopeRead]?: string } {
  const out: { [r in ScopeRead]?: string } = {}, reads = node.reads ?? [], params = node.params ?? [];
  SCOPE_READS.slice(0, 14).forEach((r, i) => { if (reads[i]) out[r] = reads[i]; });
  /* the CFG interval: the node's two params at Lab, else its last two reads */
  const ivl = params.length === 2 ? params : reads.slice(14, 16);
  if (ivl[0]) out.cs = ivl[0]; if (ivl[1]) out.ce = ivl[1];
  return out;
}
/** The custom timesteps in force (two or more valid ones), which replace the steps and the schedule. */
function customTs(ix: Index, desk: Desk, k: string | undefined): number[] | null {
  const t = textOf(ix, k, desk);
  if (!k || !CHECKS.timesteps.test(t)) return null;
  return t.split(',').map(x => x.trim()).filter(Boolean).map(x => +x);
}
/** The noise schedule the scope draws (sigmas): the schedule its menu pictures, with its shape's values, warped by the
 *  shift; or the custom timesteps. Shift 0 is the limiting curve (S-32). */
export function sigmasOf(ix: Index, node: Node, desk: Desk): number[] {
  const k = scopeKeys(node), ts = customTs(ix, desk, k.ts);
  if (ts) return ts;
  const n = clamp(num(ix, desk, k.steps, 12), 1, 300), shOk = !!k.shift && applies(ix, k.shift, desk);
  const sh = !shOk ? (desk.engine === 'ace' ? 3 : 1) : isAuto(ix, k.shift!, desk) ? 3 : num(ix, desk, k.shift, 3);
  const sc = k.sched && applies(ix, k.sched, desk) ? labelOf(ix, k.sched, desk) ?? 'Linear' : 'Linear';
  const g = (r: ScopeRead, d: number) => num(ix, desk, k[r], d);
  let f: (t: number) => number;
  if (sc === 'Power') { const p = g('exp', 2); f = t => (1 - t) ** p; }
  else if (sc === 'Beta (Custom)') { const a = g('alpha', 0.5), b = g('beta', 0.7); f = t => 1 - betaCDF(t, a, b); }
  else if (sc === 'Composite (2-Stage)') {
    const fa = STAGES[labelOf(ix, k.a, desk) ?? ''] ?? STAGES.Linear!, fb = STAGES[labelOf(ix, k.b, desk) ?? ''] ?? STAGES.Linear!, cx = g('x', 0.5), sp = g('split', 0.5);
    f = t => (t < sp ? cx + (1 - cx) * fa(t / sp) : cx * fb((t - sp) / (1 - sp)));
  } else f = SCHEDULE_CURVES[sc] ?? (t => 1 - t);
  const N = Math.min(n, 120);
  return Array.from({ length: N + 1 }, (_, i) => { const b = f(i / N), d = 1 + (sh - 1) * b; return d > 1e-9 ? sh * b / d : 1; });
}
function scopeOf(ix: Index, node: Node, desk: Desk): NonNullable<InstrumentModel['scope']> {
  const k = scopeKeys(node), sg = sigmasOf(ix, node, desk), n = sg.length - 1, L = 6, R = 294, T = 8, B = 100;
  const X = (i: number) => L + (R - L) * i / Math.max(1, n), Y = (s: number) => B - (B - T) * s;
  const ivl = !!k.cs && !!k.ce && has(ix, k.cs, desk.engine) && has(ix, k.ce, desk.engine);
  const cs = ivl ? num(ix, desk, k.cs, 0) : 0, ce = ivl ? num(ix, desk, k.ce, 1) : 1;
  const cutV = k.cut && has(ix, k.cut, desk.engine) ? num(ix, desk, k.cut, 1) : 1;
  const band = ivl && (cs > 0 || ce < 1) ? { x: L + (R - L) * Math.min(cs, ce), w: (R - L) * Math.abs(ce - cs) } : null;
  const cut = cutV < 1 ? { x: L + (R - L) * cutV, w: (R - L) * (1 - cutV) } : null;
  /* the read-out says only what the controls under it do not (V-07): native sampling, the custom timesteps, the
     guidance window its handles set and where Performance trades' cutoff ends it (L-12, A-31) */
  const read: NonNullable<InstrumentModel['scope']>['read'] = [], ts = customTs(ix, desk, k.ts);
  /* native: no solver plugin applies, and the engine has no sampler choice of its own (Stable Audio 3's) */
  if (k.solver && has(ix, k.solver, desk.engine) && !applies(ix, k.solver, desk) && !(k.own && applies(ix, k.own, desk))) read.push({ text: msg('scope.native'), tone: '', action: null });
  if (ts) read.push({ text: msg('scope.ts', { n: ts.length }), tone: 'ts', action: 0 });
  if (ivl) {
    const z = Math.min(ce, cutV), steps = ts ? ts.length - 1 : num(ix, desk, k.steps, 12), why = cutV < 1 && cutV <= ce;
    /* its numbers are set in bold: each between two U+0001 marks, which the part draws as <b> */
    const b = (x: number) => `\u0001${x.toFixed(2)}\u0001`;
    const say = (z - cs) * steps < 1 ? msg('scope.off') : cs > 0 ? msg('scope.from', { a: b(cs), b: b(z) }) : z < 1 ? msg('scope.until', { b: b(z) }) : msg('scope.everyStep');
    const tone = (z - cs) * steps < 1 ? 'warn' : cs > 0 || z < 1 ? 'amber' : '';
    read.push({ text: why ? { id: 'scope.why', en: '{s}{c}', slots: { s: say, c: msg('scope.cut') } } : say, tone, action: why ? 1 : null });
  }
  const handles = (node.params ?? []).length === 2 && ivl ? [{ key: k.cs!, at: (L + (R - L) * cs) / 300 * 100, flip: false }, { key: k.ce!, at: (L + (R - L) * ce) / 300 * 100, flip: false }] : [];
  return {
    points: sg.map((s, i) => `${X(i).toFixed(1)},${Y(s).toFixed(1)}`).join(' '),
    dots: n <= 32 ? sg.map((s, i) => ({ x: +(X(i) / 3).toFixed(2), y: +(Y(s) / 1.1).toFixed(2) })) : [],
    band, cut, read, handles,
  };
}

/* ---------------------------------------------------------------- the latent curve */

/** The latent distribution as rendered (a unit normal, dashed grey) and after the shift and rescale (green). */
export function distOf(shift: number, rescale: number): NonNullable<InstrumentModel['dist']> {
  const X = (x: number) => (x + 4) / 8 * 300;
  const curve = (m: number, sd: number) => { const pts: string[] = []; for (let x = -4; x <= 4.001; x += 0.08) { const y = Math.exp(-((x - m) ** 2) / (2 * sd * sd)); pts.push(`${X(x).toFixed(1)},${(74 - y * 60 / Math.max(sd, 1)).toFixed(1)}`); } return pts.join(' '); };
  return { grid: Array.from({ length: 9 }, (_, i) => X(i - 4)), zero: X(0), before: curve(0, 1), after: curve(shift, rescale) };
}

/* ---------------------------------------------------------------- the repaint region */

/** The song a repaint works on, as its record holds it (the song menu's "Repaint a region" writes it). */
export interface SourceSong { song: number | string; title: string; style: string; dur: number; hue: number }
const sourceOf = (v: Json | undefined): SourceSong | null => {
  const o = v as Partial<SourceSong> | null | undefined;
  return o && typeof o === 'object' && typeof o.dur === 'number' && o.dur > 0 ? { song: o.song ?? 0, title: String(o.title ?? ''), style: String(o.style ?? ''), dur: o.dur, hue: typeof o.hue === 'number' ? o.hue : 215 } : null;
};
/** The region in force, as fractions of the song: its state (seconds), else the concept's first region. */
function regionAt(ix: Index, node: Node, desk: Desk): { src: SourceSong | null; a: number; b: number } {
  const [rec, span] = [node.state?.find(k => ix.states.get(k)?.s.kind === 'record'), node.state?.find(k => ix.states.get(k)?.s.kind === 'span')];
  const src = sourceOf(rec ? stateOf(ix, rec, desk) : undefined), r = span ? stateOf(ix, span, desk) as { start?: number; end?: number } | undefined : undefined;
  if (!src || !r || typeof r.start !== 'number' || typeof r.end !== 'number') return { src, a: 0.35, b: 0.55 };
  return { src, a: clamp(r.start / src.dur, 0, 1), b: clamp(r.end / src.dur, 0, 1) };
}
function regionOf(ix: Index, node: Node, desk: Desk): NonNullable<InstrumentModel['region']> | undefined {
  const { src, a, b } = regionAt(ix, node, desk);
  if (!src) return undefined;
  const t = (f: number) => mmss(f * src.dur), id = typeof src.song === 'number' ? src.song : [...String(src.song)].reduce((h, c) => h * 31 + c.charCodeAt(0) | 0, 0);
  return {
    title: src.title, style: msg('region.song', { style: src.style, dur: mmss(src.dur) }), dur: src.dur, durText: mmss(src.dur), hue: src.hue, id,
    a, b, cap: msg('region.cap', { a: t(Math.min(a, b)), b: t(Math.max(a, b)) }), text: msg('region.text', { a: t(a), b: t(b) }),
    /* its children: the first two stand on the wave's caption line where shown (Lab), the rest beside the wave; with
       nothing beside it but the note (Quick), the wave takes the width (Q-28) */
    ctl: !!node.children?.[0] && nodeShown(ix, node.children[0], desk),
    one: !(node.children ?? []).slice(2, -1).some(c => nodeShown(ix, c, desk)),
  };
}
/** Where a key or a drag takes the region (25-module-repaint.js): arrows move it a hundredth of the song (Alt: a
 *  twentieth), Shift moves its end, Home and End take it to the song's ends with its length kept; a drag carries an
 *  edge, the whole region, or draws a new one; it is never narrower than a fiftieth. Fractions in, fractions out. */
export function regionMove(a: number, b: number, how: { key?: string; shift?: boolean; alt?: boolean; kind?: 'a' | 'b' | 'body' | 'out'; x0?: number; x?: number }): [number, number] | null {
  const MIN = 0.02;
  if (how.key) {
    const w = b - a;
    if (how.key === 'Home') return [0, w];
    if (how.key === 'End') return [1 - w, 1];
    const dir = how.key === 'ArrowRight' || how.key === 'ArrowUp' ? 1 : how.key === 'ArrowLeft' || how.key === 'ArrowDown' ? -1 : 0;
    if (!dir) return null;
    const st = how.alt ? 0.05 : 0.01;
    if (how.shift) return [a, clamp(b + dir * st, a + MIN, 1)];
    const na = clamp(a + dir * st, 0, 1 - w); return [na, na + w];
  }
  const p = clamp(how.x ?? 0, 0, 1), x0 = clamp(how.x0 ?? 0, 0, 1), d = (how.x ?? 0) - (how.x0 ?? 0);
  if (how.kind === 'a') return [clamp(p, 0, b - MIN), b];
  if (how.kind === 'b') return [a, clamp(p, a + MIN, 1)];
  if (how.kind === 'body') { const w = b - a, na = clamp(a + d, 0, 1 - w); return [na, na + w]; }
  const lo = Math.min(x0, p), hi = Math.max(x0, p);
  return hi - lo < MIN ? [lo, Math.min(1, lo + MIN)] : [lo, hi];
}

/* ---------------------------------------------------------------- what a hand on a part does */

/** A hand on a bespoke part, worked out here and made through the part's own setters (bindPart), so the change is
 *  cleaned, bound and named as every other: `name` is what the hand did, `arg` what it gives. `tell` says what it did or
 *  why it did nothing (with `undo`, the toast offers to undo the step just taken). False when it was refused. */
export function partAct(ix: Index, node: Node, v: PartProps, desk: Desk, name: string, arg: unknown, tell: Tell): boolean {
  const f = ACTS[node.part]?.[name];
  if (!f) throw new Error(`${node.part} has no act "${name}"`);
  return f({ ix, node, v, desk, arg, tell }) !== false;
}
/** `undo`: the toast offers to undo the step just taken; 'say': it is said in the live region only, as the concept's say()
 *  (package 3c). */
export type Tell = (m: Msg, undo?: boolean | 'say') => void;
type Acts = { [name: string]: (x: ActIn) => boolean | void };
export interface ActIn { ix: Index; node: Node; v: PartProps; desk: Desk; arg: unknown; tell: Tell }

/** Sets the seed and its lock in one step (the seed state's `set`): a number kept, or null for random. */
function setSeed(x: ActIn, seed: number | string | null): void {
  const key = x.node.state?.find(k => x.ix.states.get(k)?.s.kind === 'seed') ?? x.ix.defs.flatMap(d => d.state ?? []).find(s => s.kind === 'seed')?.key;
  const st = key ? x.v.state[key] : undefined, lock = x.node.param;
  if (!st || !lock) return;
  st.apply('set', seed, { [lock]: seed !== null });
}
function seedList(x: ActIn): { key: string; list: SavedSeed[] } | null {
  const key = listKey(x.ix, x.node);
  return key && x.v.state[key] ? { key, list: cleanSeeds(x.v.state[key]!.value) } : null;
}

const ACTS: { [part: string]: Acts } = {
  keyWheel: {
    /* a key on the wheel, from the value in force */
    walk: x => {
      const p = x.v.params[x.node.param ?? x.node.params?.[0] ?? '']; if (!p) return;
      const f = resolve(x.ix.params.get(p.key)!.p, x.desk.engine), opts = Array.isArray(f.options) ? f.options.map(o => String(o.value)) : [];
      const auto = opts.find(o => !/ (major|minor)$/.test(o)) ?? '', to = wheelWalk(opts, auto, String(p.value), String(x.arg));
      if (to !== null) p.set(to);
    },
  },
  tapTempo: {
    /* the taps' mean interval, in milliseconds: the BPM it beats at, put in range by the setter (A-25) */
    tap: x => { const p = x.v.params[x.node.param ?? ''], iv = Number(x.arg); if (p && iv > 0) p.set(Math.round(60000 / iv)); },
  },
  capChips: {
    chip: x => {
      const p = x.v.params[x.node.param ?? '']; if (!p) return;
      const chip = String(x.arg), r = capToggle(String(p.value ?? ''), chip);
      p.set(r.text, r.on ? 'addChip' : 'removeChip', { chip });
    },
  },
  splitFlap: {
    /* a seed typed or pasted (empty is random); refused and said when it is no seed */
    type: x => { const r = seedTyped(String(x.arg ?? '')); if ('error' in r) { x.tell(msg('flap.bad', { max: SEED_MAX })); return false; } setSeed(x, r.seed); },
    /* left with no seed in it: kept as it was, and said */
    left: x => { const r = seedTyped(String(x.arg ?? '')); if ('error' in r) { x.tell(msg('flap.bad.left', { max: SEED_MAX })); return; } setSeed(x, r.seed); },
    /* Roll once: a fresh seed, kept; `arg` is the hand's draw, 0 to 1 */
    roll: x => setSeed(x, Math.floor(clamp(Number(x.arg) || 0, 0, 0.999999999) * 2147483647)),
    random: x => setSeed(x, null),
  },
  savedSeeds: {
    /* the save form opens on a seed: there is none while each run draws its own */
    open: x => { if (seedOf(x.ix, x.node, x.desk).seed === null) { x.tell(msg('stars.first')); return false; } },
    /* save the seed shown with the form's words (`arg`: name, description, tags), or update it when it is saved */
    save: x => {
      const l = seedList(x), { seed } = seedOf(x.ix, x.node, x.desk); if (!l || seed === null) { x.tell(msg('stars.first')); return; }
      const o = (x.arg ?? {}) as { name?: string; desc?: string; tags?: string }, old = l.list.find(s => String(s.v) === String(seed));
      const entry: SavedSeed = { v: seed, n: (String(o.name ?? '').trim() || `seed_${seed}`).slice(0, 40), d: String(o.desc ?? '').trim().slice(0, 120), tags: String(o.tags ?? '').split(',').map(t => t.trim()).filter(Boolean).slice(0, 12), ...(old?.star ? { star: true } : {}) };
      x.v.state[l.key]!.apply(old ? 'update' : 'save', [entry, ...l.list.filter(s => String(s.v) !== String(seed))] as unknown as Json, undefined, { seed: String(seed) });
      x.tell(msg(old ? 'stars.updated' : 'stars.saved', { seed: String(seed) }), !!old);
    },
    star: x => {
      const l = seedList(x), at = l?.list.findIndex(s => String(s.v) === String(x.arg)) ?? -1; if (!l || at < 0) return;
      const s = l.list[at]!, next = l.list.map((y, i) => (i === at ? { ...y, star: !s.star } : y));
      x.v.state[l.key]!.apply(s.star ? 'unstar' : 'star', next as unknown as Json, undefined, { seed: seedName(s) });
    },
    delete: x => {
      const l = seedList(x), s = l?.list.find(y => String(y.v) === String(x.arg)); if (!l || !s) return;
      x.v.state[l.key]!.apply('delete', l.list.filter(y => y !== s) as unknown as Json, undefined, { seed: seedName(s) });
      x.tell(msg('stars.deleted', { name: seedName(s) }), true);
    },
    load: x => { const l = seedList(x), s = l?.list.find(y => String(y.v) === String(x.arg)); if (s) setSeed(x, s.v); },
    /* a random saved seed: with two or more, never the one already loaded (L-22); said with its name (L-20). `arg` is
       the hand's draw, 0 to 1 */
    random: x => {
      const l = seedList(x); if (!l || !l.list.length) { x.tell(msg('stars.saveFirst')); return false; }
      const { seed } = seedOf(x.ix, x.node, x.desk), pool = l.list.length > 1 ? l.list.filter(s => String(s.v) !== String(seed)) : l.list;
      const s = pool[Math.floor(clamp(Number(x.arg) || 0, 0, 0.999999999) * pool.length)]!;
      setSeed(x, s.v);
      x.tell(s.n ? msg('stars.loaded', { name: s.n.length > 40 ? s.n.slice(0, 39) + '…' : s.n, seed: String(s.v) }) : msg('stars.loadedPlain', { seed: String(s.v) }));
    },
  },
  repaintRegion: {
    /* a key or a drag on the waveform (`arg`: the key and its modifiers, or the drag's kind, its places as fractions and
       the region it started from, `a0` and `b0`) */
    move: x => {
      const { src, a, b } = regionAt(x.ix, x.node, x.desk), span = x.node.state?.find(k => x.ix.states.get(k)?.s.kind === 'span');
      if (!src || !span || !x.v.state[span]) return;
      const how = (x.arg ?? {}) as Parameters<typeof regionMove>[2] & { a0?: number; b0?: number };
      const r = regionMove(how.a0 ?? a, how.b0 ?? b, how);
      if (!r || (Math.abs(r[0] - a) < 1e-12 && Math.abs(r[1] - b) < 1e-12)) return;
      x.v.state[span]!.apply('move', { start: r[0] * src.dur, end: r[1] * src.dur });
    },
  },
};

/* ---------------------------------------------------------------- instruments a core file of its own adds
   A package's core file draws its own parts and works out their hands, and adds them here when it is loaded
   (derive.ts loads it): derive/cartridges.ts, the Adapters' cartridges and the XY pad's off rail (package 3b);
   derive/build.ts, recording.ts and score.ts, Build by section, Continue from recording and the ABC foot (package 3c).
   Each adds its model's field to InstrumentModel by declaring it (declare module './instruments.ts'). */

/** Adds a part's drawing (instrumentModel) and the hands it takes (partAct). */
export function defineInstrument(part: string, draw?: (ix: Index, node: Node, desk: Desk, key: string) => InstrumentModel, acts?: { [name: string]: (x: ActIn) => boolean | void }): void {
  if (draw) { if (DRAW[part]) throw new Error(`the instrument "${part}" is drawn twice`); DRAW[part] = draw; }
  if (acts) ACTS[part] = { ...ACTS[part], ...acts };
}
