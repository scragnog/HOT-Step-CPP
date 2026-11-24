// What the Adapters' cartridges draw, and what a hand on them does (docs/09, package 3b): each loaded adapter as a
// cassette with its label art, its strength fader, its kind's housing, and in a stack its share, its active phase and
// its section masks written into the lyrics; the empty slot, the planner adapter, the column beside one cartridge, the
// adapters loaded but not used. The concept's own formulas (25-module-adapters.js: cartHTML, adArt, slotLabels,
// stackOn, effScale, sectionDirectives, setDirective, dropDirectiveSlot, maskTag, phaseHTML and its keys, unusedNote,
// the load and remove handlers and the two stepName rules), moved here so the part draws numbers and words it is given.
//
// What differs per engine is said in the definition (adapters.ts): the kinds of its slots and the words beside one
// cartridge are the node's options per engine; the parameters it binds are each slot's strength in order, then the
// first slot's own faders where the engine has them (YuE2's Plan and Compose), then the planner adapter and the stack
// switch where it binds them; what it reads is, by place: the stack switch, the stack mode, the combined strength, the
// planner adapter and the lyrics. Nothing here names a module, an engine or a parameter key.
//
//   cartsOf(ix, node, desk, key)   InstrumentModel.carts
//   directivesOf, setDirective, dropDirectiveSlot, phaseMove   the rules, tested in tools/test-cartridges.ts
//
// Also here, the XY pad's off rail (MiniMax-Music3's top-p 0 is "off", E-28, E-32): where the puck stands and where a
// hand takes the value (xyRail, xyRailValue), the pad's hand `y`.

import type { Json, Node, Value } from '../definitions/types.ts';
import { type Desk, type Index, applies, factOf, has, resolve, stateOf, valueOf } from './core.ts';
import { controlModel, type ControlModel } from './controls.ts';
import { EN, type Msg, type Slot, list, msg } from './messages.ts';
import { formatMsg, nodeOptWord, paramWords } from './strings.ts';
import { defineInstrument, type ActIn, type InstrumentModel } from './instruments.ts';
import { stepChanges } from './steps.ts';
import { engineName } from '../definitions/engines.ts';

Object.assign(EN, {
  'fader.name': '{label} strength',
  'forced.returns': '{says} · {value} returns after',
  'carts.type.LoRA.note': 'low-rank', 'carts.type.LoRA.desc': 'Two small matrices added onto the weights.',
  'carts.type.LoKr.note': 'Kronecker', 'carts.type.LoKr.desc': 'One block pattern tiled across the weights.',
  'carts.type.DoRA.note': 'mag + dir', 'carts.type.DoRA.desc': 'Adapts magnitude and direction separately.',
  'carts.title': '{type}: {desc}',
  'carts.em': '{type} · {note}',
  'carts.slot.n': '#{n} · {kind}',
  'carts.add': '+ {kind} adapter',
  'carts.add.plain': '+ Load adapter',
  'carts.second': '+ Second adapter',
  'carts.second.kind': '+ Second {kind} adapter',
  'carts.second.tip': 'Loading a second one stacks them: each gets its own fader and section masks',
  'carts.remove': 'Remove {file}',
  'carts.planner.remove': 'Remove the planner adapter',
  'carts.strength': '{file} strength',
  'carts.strength.label': 'Strength',
  'carts.strength.of': '{file} {label|lower} strength',
  'carts.sends': 'sends {v}',
  'carts.phase.title': 'Active phase: {a}% to {b}% of the run. Early steps shape structure, late steps timbre.',
  'carts.phase.group': 'Active phase for adapter {n}',
  'carts.phase.start': 'Adapter {n} phase start',
  'carts.phase.end': 'Adapter {n} phase end',
  'carts.phase.from': 'from {v}% of the run',
  'carts.phase.to': 'to {v}% of the run',
  'carts.phase.word': 'phase',
  'carts.masks': 'Sections this adapter colours',
  'carts.mask.aria': '{section}: {file} at {lv}',
  'carts.mask.title': '{section} · {file} at {lv}',
  'carts.lv.full': 'full', 'carts.lv.half': 'half', 'carts.lv.off': 'off',
  'carts.unused.single': '{names} {n, plural, one {is} other {are}} loaded but not used while {stack} is {single}.',
  'carts.unused.takes': '{names} {n, plural, one {is} other {are}} loaded but not used: {engine} takes {max}.',
  'carts.more': '{n} adapters.',
  'carts.more.stacked': '{n} adapters, stacked.',
  'carts.more.first': '{n} adapters, the first one used.',
  'carts.list': '{kind} adapters for {engine}',
  'carts.list.planner': '{kind} adapters',
  'carts.removed': 'Removed {file}',
  'carts.removed.renumbered': 'Removed {file}. The lyric directives were renumbered to match',
  'carts.away': '{words}.',
  'carts.adapter': 'adapter {n}',
});

/* ---------------------------------------------------------------- what the part draws */

/** A loaded adapter's label art: a tape for a LoRA, a chip for a LoKr, a vector for a DoRA; `p` is its strength's
 *  share of 2, its reels' spin, and whether it stands still (strength 0). */
export interface CartArt { kind: 'tape' | 'chip' | 'vec'; p: number; spin: string; still: boolean; cells: { x: string; y: string; v: string }[]; head: { x: string; y: string } | null }
/** A fader in a cartridge: its parameter, where its cap stands, its tape (YuE2's pair) and its accessible name. */
export interface CartFader { key: string; fader: NonNullable<ControlModel['fader']>; label: Slot | null; aria: Msg }
export interface CartPhase { a: number; b: number; span: string; title: Msg; group: Msg; word: Msg; start: { aria: Msg; max: number; text: Msg }; end: { aria: Msg; min: number; text: Msg } }
export interface CartMask { i: number; j: number; tag: string; lv: 'full' | 'half' | 'off'; color: string; n: number; next: number; aria: Msg; title: Msg }
export interface Cart {
  /** Its slot, or -1 for the planner adapter. */
  i: number;
  file: string; full: string; name: string[]; type: string; cls: string; housing: 'sound' | 'plan';
  label: Slot; title: Msg; em: Slot; remove: Msg;
  art: CartArt; faders: CartFader[]; split: boolean;
  /** The planner adapter's note where a fader would be, and its parameter (the changed mark on its label). */
  note: Slot | null; ctl: string | null;
  eff: Msg | null; phase: CartPhase | null; masks: CartMask[] | null;
}
export interface CartsModel {
  /** Quick's flat cartridge, or the rig. */
  look: 'flat' | 'rig';
  carts: Cart[];
  /** The empty slot that loads the next adapter, and the planner adapter's (its button, or its cartridge). */
  add: { label: Slot; housing: 'sound' | 'plan'; bar: boolean } | null;
  planner: { cart: Cart } | { label: Slot; bar: boolean } | null;
  /** One cartridge and no stack: its words and its choices stand in the column beside it (Q-19). */
  one: boolean;
  /** The foot stands in that column (Studio). */
  sideFoot: boolean;
  side: { desc: Slot; second: { label: Msg; tip: Msg } | null; note: Slot | null } | null;
  unused: { text: Msg; stack: { label: Slot; value: Value; key: string } | null } | null;
  away: Msg | null;
  /** Quick: the adapters it does not show, and the way to Studio. */
  more: Msg | null;
  /** The adapter lists a hand opens: the next slot's, and the planner's (their heading, the files offered). */
  list: AdapterList | null; plannerList: AdapterList | null;
  /** The section strip's accessible name. */
  masksLabel: Msg;
}
export interface AdapterList { head: Msg; note: Slot | null; items: { file: string; label: string; type: string; cls: string; title: Msg; art: CartArt; value: string }[] }

declare module './instruments.ts' {
  interface InstrumentModel {
    /** cartridges: the Adapters' cartridges, the empty slot, the planner adapter and the words around them. */
    carts?: CartsModel;
    /** xyPad with an off rail: whether Y is off, and its place on the plot (1 at the plot's top; past it, on the rail). */
    rail?: { off: boolean; at: number };
  }
}

/* ---------------------------------------------------------------- reading the node */

interface Slot0 { file: string; type: string; ph?: number[] }
const slotsOf = (ix: Index, node: Node, desk: Desk): Slot0[] => {
  const k = node.state?.[0], v = k ? stateOf(ix, k, desk) : undefined;
  return Array.isArray(v) ? (v as unknown as Slot0[]).filter(s => s && typeof s.file === 'string') : [];
};
const read = (ix: Index, desk: Desk, k: string | undefined): Value | null => (k && ix.params.has(k) && has(ix, k, desk.engine) ? valueOf(ix, k, desk) ?? null : null);
const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));
const J = (x: unknown) => JSON.stringify(x);

/** The node's reads by place, and its kinds (each slot's label on this engine). */
function shape(ix: Index, node: Node, desk: Desk, key: string) {
  const [adv, mode, sigma, planner, lyrics] = node.reads ?? [];
  const kinds = ((node.opts as { kinds?: string[] } | undefined)?.kinds ?? []).map((_, i) => (key ? nodeOptWord(ix, node, desk, key, 'kinds', i) : undefined) ?? '');
  const bound = [node.param, ...(node.params ?? [])].filter((k): k is string => !!k && has(ix, k, desk.engine));
  const nums = bound.filter(k => ix.params.get(k)!.p.type === 'number');
  const strengths = nums.slice(0, kinds.length), split = nums.slice(kinds.length);
  const switchHere = !!adv && has(ix, adv, desk.engine);
  const advFace = switchHere ? resolve(ix.params.get(adv!)!.p, desk.engine) : null;
  const stackedValue = advFace && Array.isArray(advFace.options) ? advFace.options.find(o => o.value !== advFace.default)?.value ?? null : null;
  /* every slot the engine has is used, but where a stack switch decides: then one, unless it is on stacked */
  const adv1 = !switchHere || read(ix, desk, adv) === stackedValue;
  const slots = slotsOf(ix, node, desk);
  const stacked = switchHere && adv1 && slots.length >= 2;
  const plannerKey = planner && bound.includes(planner) ? planner : null;
  return { adv, mode, sigma, planner: plannerKey, lyrics: lyrics && has(ix, lyrics, desk.engine) ? lyrics : null, kinds, strengths, split, switchHere, stackedValue, adv1, slots, stacked };
}

/* ---------------------------------------------------------------- section directives, written into the lyrics */

const DIRECTIVE = /^(\s*\[[^\]]+\])\s*\{([^}]*)\}/;
const TAG = /^\s*\[[^\]]+\]/;
/** One entry per tagged line of the lyrics: null when it has no directive, else each slot's weight by its index. */
export function directivesOf(text: string): ({ [slot: number]: number } | null)[] {
  const out: ({ [slot: number]: number } | null)[] = [];
  for (const l of String(text || '').split('\n')) {
    if (!TAG.test(l)) continue;
    const m = l.match(DIRECTIVE); if (!m) { out.push(null); continue; }
    const w: { [slot: number]: number } = {}; let any = false;
    for (const pair of m[2]!.split(/[;,]/)) { const mm = pair.match(/^\s*#?(\d+)\s*=\s*(-?[\d.]+)\s*$/); if (mm) { w[+mm[1]! - 1] = Math.max(0, +mm[2]!); any = true; } }
    out.push(any ? w : null);
  }
  return out;
}
/** The lyrics with section j's weight for slot i set to w, among n slots: a section at full for every slot loses its
 *  directive. */
export function setDirective(text: string, j: number, i: number, w: number, n: number): string {
  const lines = String(text).split('\n'); let k = -1;
  for (let li = 0; li < lines.length; li++) {
    if (!TAG.test(lines[li]!)) continue; k++; if (k !== j) continue;
    const head = lines[li]!.match(TAG)![0], rest = lines[li]!.replace(DIRECTIVE, '$1').slice(head.length), cur = directivesOf(lines[li]!)[0];
    const ws = Array.from({ length: n }, (_, x) => (cur ? cur[x] ?? 0 : 1)); ws[i] = w;
    lines[li] = ws.every(v => v === 1) ? head + rest : `${head}{${ws.map((v, x) => `#${x + 1}=${+v.toFixed(2)}`).join('; ')}}${rest}`;
    break;
  }
  return lines.join('\n');
}
/** The lyrics with slot i taken out of every directive and the later slots moved up; null when nothing changes. */
export function dropDirectiveSlot(text: string, i: number): string | null {
  const t = String(text); let hit = false;
  const out = t.split('\n').map(l => {
    const m = l.match(DIRECTIVE); if (!m) return l;
    const w = directivesOf(l)[0]; if (!w) return l; hit = true;
    const ks = Object.keys(w).map(Number).filter(x => x !== i).sort((a, b) => a - b), ws = ks.map(x => [x > i ? x - 1 : x, w[x]!] as const), rest = l.replace(DIRECTIVE, '$1');
    return !ws.length || ws.every(([, v]) => v === 1) ? rest : rest.replace(/^(\s*\[[^\]]+\])/, `$1{${ws.map(([x, v]) => `#${x + 1}=${+v.toFixed(2)}`).join('; ')}}`);
  }).join('\n');
  return hit && out !== t ? out : null;
}
const levelOf = (w: number): 'full' | 'half' | 'off' => (w >= 0.75 ? 'full' : w > 0 ? 'half' : 'off');
const weightAt = (d: { [slot: number]: number } | null | undefined, i: number) => (d ? d[i] ?? 0 : 1);
/** A mask chip's next weight: full, half, off, and round again. */
const STEPW = [1, 0.5, 0];
const nextWeight = (w: number) => STEPW[(STEPW.indexOf(w >= 0.75 ? 1 : w > 0 ? 0.5 : 0) + 1) % 3]!;

/** The lyrics' sections, with their lines (25-module-lyrics.js sections). */
function sectionsOf(text: string): { name: string; n: number }[] {
  const out: { name: string; n: number }[] = []; let cur: { name: string; n: number } | null = null;
  for (const l of text.split('\n')) {
    const m = l.trim().match(/^\[([^\]]+)\]/);
    if (m) { cur = { name: m[1]!.trim(), n: 0 }; out.push(cur); } else if (l.trim()) { if (!cur) { cur = { name: 'Verse', n: 0 }; out.push(cur); } cur.n++; }
  }
  return out;
}
/** A section's colour (25-module-lyrics.js SECTION_COL), the same on the lyric sheet's strip. */
const SECTION_COL: { [name: string]: string } = { verse: '#3f5ef0', chorus: '#e0225a', 'pre-chorus': '#9440f5', bridge: '#0d7f62', intro: '#6b7280', outro: '#6b7280', hook: '#a85a00' };
const secColor = (n: string) => SECTION_COL[n.toLowerCase().replace(/\s*\d+$/, '')] ?? '#8b8496';
/** A section's tag on its chip: its initial, numbered when the song has more than one of it (V2, C3) (L-25). */
function maskTag(secs: { name: string }[], j: number): string {
  const ini = (n: string) => n.replace(/\s*\d+$/, '').slice(0, 1).toUpperCase(), me = ini(secs[j]!.name), same = secs.filter(x => ini(x.name) === me);
  return same.length > 1 ? me + secs.slice(0, j + 1).filter(x => ini(x.name) === me).length : me;
}

/* ---------------------------------------------------------------- active phase */

/** A slot's active-phase window, 0 to 100 on a 5% grid. */
const phaseOf = (s: Slot0 | undefined): [number, number] => (s?.ph && s.ph.length === 2 ? [s.ph[0]!, s.ph[1]!] : [0, 100]);
const span = (p: [number, number]) => `${p[0]}–${p[1]}%`;
/** Where a key or a hand takes a phase handle (h 0 the start, 1 the end): keys step 5%, Page 25%, Home and End to the
 *  ends of the handle's own range; a hand's place `f` (0 to 1 along the strip) snaps to 5%. The handles keep 5% apart. */
export function phaseMove(ph: [number, number], h: 0 | 1, how: { key?: string; f?: number }): [number, number] | null {
  const out: [number, number] = [ph[0], ph[1]];
  if (how.key !== undefined) {
    const d = ({ ArrowLeft: -5, ArrowDown: -5, ArrowRight: 5, ArrowUp: 5, PageDown: -25, PageUp: 25, Home: -1000, End: 1000 } as { [k: string]: number })[how.key];
    if (d === undefined) return null;
    out[h] = clamp(ph[h] + d, h ? ph[0] + 5 : 0, h ? 100 : ph[1] - 5);
  } else {
    const v = clamp(Math.round(clamp(how.f ?? 0, 0, 1) * 20) * 5, 0, 100);
    if (h === 0) out[0] = Math.min(v, ph[1] - 5); else out[1] = Math.max(v, ph[0] + 5);
  }
  return out;
}

/* ---------------------------------------------------------------- the art */

const TYPES = ['LoRA', 'LoKr', 'DoRA'];
const typeOf = (t: string) => (TYPES.includes(t) ? t : 'LoRA');
function artOf(type: string, file: string, strength: number | null): CartArt {
  const p = strength === null ? 0.6 : clamp(strength / 2, 0, 1), t = typeOf(type);
  /* a cartridge without a strength of its own (the planner's) stands at .6 and does not spin */
  const base = { p, spin: strength === null ? '' : p > 0 ? (1.6 / p).toFixed(2) + 's' : '4s', still: strength !== null && p === 0, cells: [], head: null };
  if (t === 'LoKr') {
    const A = [[1, 0.45], [0.7, 0.9]], B = [[1, 0.3, 0.8], [0.5, 1, 0.35]], cells: CartArt['cells'] = [];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 6; c++) cells.push({ x: String(+(18 + c * 8.2).toFixed(1)), y: String(+(9 + r * 8.2).toFixed(1)), v: (A[r >> 1]![(c / 3) | 0]! * B[r & 1]![c % 3]!).toFixed(2) });
    return { ...base, kind: 'chip', cells };
  }
  if (t === 'DoRA') {
    const h = [...(file || 'x')].reduce((a, ch) => a + ch.charCodeAt(0), 0), ang = (35 + h % 110) * Math.PI / 180;
    return { ...base, kind: 'vec', head: { x: (42 + 15 * Math.cos(ang)).toFixed(1), y: (36 - 15 * Math.sin(ang)).toFixed(1) } };
  }
  return { ...base, kind: 'tape' };
}
/** A file's name as a cartridge prints it: without its extension, its middle cut past `n` characters (the version at its
 *  end stays), in pieces that may break after a hyphen, an underscore or a dot. */
function nameOf(file: string, n: number): string[] {
  let s = String(file).replace(/\.(safetensors|ckpt|pt|pth|bin|gguf)$/i, '');
  if (s.length > n) { const k = n - 1, a = Math.ceil(k * 0.55); s = s.slice(0, a) + '…' + s.slice(s.length - (k - a)); }
  return s.split(/(?<=[-_.])/);
}
const typeWords = (t: string) => ({ note: msg(`carts.type.${typeOf(t)}.note`), desc: msg(`carts.type.${typeOf(t)}.desc`) });

/* ---------------------------------------------------------------- the model */

export function cartsOf(ix: Index, node: Node, desk: Desk, key: string): CartsModel {
  const s = shape(ix, node, desk, key), quick = desk.tier === 'quick', lab = desk.tier === 'lab';
  const { slots, kinds, stacked } = s;
  /* an adapter is called by its kind only where the page offers a second kind beside it (Lab's planner adapter where a
     stack switch decides; every slot elsewhere: Q-11) */
  const named = !quick && (!s.switchHere || !!s.planner);
  const kindAt = (i: number) => (named ? kinds[i] ?? '' : '');
  const max = quick ? 1 : s.adv1 ? kinds.length : 1, shown = slots.slice(0, max);
  const blend = !!s.sigma && applies(ix, s.sigma, desk);
  const raw = slots.slice(0, 3).map((_, x) => Number(read(ix, desk, s.strengths[x]) ?? 0)), sum = raw.reduce((a, b) => a + b, 0), sigma = Number(read(ix, desk, s.sigma) ?? 0);
  const eff = (i: number) => (blend ? (sum ? raw[i]! / sum * sigma : 0) : raw[i]!);
  const lyrics = String(read(ix, desk, s.lyrics ?? undefined) ?? ''), masks = stacked && !quick && !!s.lyrics;
  const secs = masks ? sectionsOf(lyrics) : [], dirs = masks ? directivesOf(lyrics) : [];
  const fileOrN = (i: number) => slots[i]?.file ?? msg('carts.adapter', { n: i + 1 });
  const fader = (k: string, label: Slot | null, aria: Msg): CartFader | null => {
    const f = controlModel(ix, { part: 'fader', param: k } as Node, desk, {}, undefined)?.fader;
    /* named for its adapter, its tape (YuE2's pair) without a tip */
    return f ? { key: k, fader: { ...f, name: aria, tape: null }, label, aria } : null;
  };
  const cartOf = (x: Slot0, i: number): Cart => {
    const t = typeOf(x.type), w = typeWords(t), k = s.strengths[i] ?? null, strength = k ? Number(read(ix, desk, k) ?? 0) : null;
    const split = i === 0 && s.split.length > 0 && !quick;
    const faders = !k ? [] : split
      ? [fader(k, msg('carts.strength.label'), msg('carts.strength', { file: x.file })), ...s.split.map(q => fader(q, paramWords(ix, q, desk).label, msg('carts.strength.of', { file: x.file, label: paramWords(ix, q, desk).label })))]
      : [fader(k, null, msg('carts.strength', { file: x.file }))];
    const ph = phaseOf(x), phase = stacked && lab ? {
      a: ph[0], b: ph[1], span: span(ph), title: msg('carts.phase.title', { a: ph[0], b: ph[1] }), group: msg('carts.phase.group', { n: i + 1 }), word: msg('carts.phase.word'),
      start: { aria: msg('carts.phase.start', { n: i + 1 }), max: ph[1] - 5, text: msg('carts.phase.from', { v: ph[0] }) },
      end: { aria: msg('carts.phase.end', { n: i + 1 }), min: ph[0] + 5, text: msg('carts.phase.to', { v: ph[1] }) },
    } : null;
    return {
      i, file: x.file, full: x.file, name: nameOf(x.file, quick ? 26 : 30), type: t, cls: t.toLowerCase(), housing: housingOf(kinds[i] ?? kinds[0] ?? ''),
      label: quick ? '' : stacked ? msg('carts.slot.n', { n: i + 1, kind: kinds[i] ?? '' }) : kinds[i] ?? '',
      title: msg('carts.title', { type: t, desc: w.desc }), em: quick ? t : msg('carts.em', { type: t, note: w.note }), remove: msg('carts.remove', { file: x.file }),
      art: artOf(t, x.file, strength), faders: faders.filter((f): f is CartFader => !!f), split, note: null, ctl: null,
      eff: stacked && blend ? msg('carts.sends', { v: eff(i).toFixed(2) }) : null, phase,
      masks: masks ? secs.map((sec, j) => {
        const w0 = weightAt(dirs[j], i), lv = levelOf(w0), lvw = msg(`carts.lv.${lv}`);
        return { i, j, tag: maskTag(secs, j), lv, color: secColor(sec.name), n: Math.max(1, sec.n), next: nextWeight(w0), aria: msg('carts.mask.aria', { section: sec.name, file: fileOrN(i), lv: lvw }), title: msg('carts.mask.title', { section: sec.name, file: fileOrN(i), lv: lvw }) };
      }) : null,
    };
  };
  const carts = shown.map(cartOf);
  const one = !quick && shown.length === 1 && !stacked;
  /* the planner adapter: its cartridge, or the empty slot that loads one */
  let planner: CartsModel['planner'] = null;
  if (s.planner && !quick) {
    const v = String(read(ix, desk, s.planner) ?? ''), o = key ? nodeOptWord(ix, node, desk, key, 'planner') : undefined;
    if (!v) planner = { label: msg('carts.add', { kind: o ?? '' }), bar: !slots.length };
    else {
      const hit = (desk.lists?.['adapters.lm'] ?? []).find(x => String(x.value) === v), t = typeOf(hit?.kind ?? 'LoRA'), w = typeWords(t);
      planner = { cart: {
        i: -1, file: v, full: v, name: nameOf(v, 30), type: t, cls: t.toLowerCase(), housing: 'plan', label: o ?? '', title: msg('carts.title', { type: t, desc: w.desc }),
        em: msg('carts.em', { type: t, note: w.note }), remove: msg('carts.planner.remove'), art: artOf(t, v, null), faders: [], split: false,
        note: (key ? nodeOptWord(ix, node, desk, key, 'plannerNote') : undefined) ?? null, ctl: s.planner, eff: null, phase: null, masks: null,
      } };
    }
  }
  const add = !quick && shown.length < max ? { label: kindAt(shown.length) ? msg('carts.add', { kind: kindAt(shown.length) }) : msg('carts.add.plain'), housing: housingOf(kinds[shown.length] ?? ''), bar: !shown.length || one } : null;
  const unused = slots.length > max && !quick ? unusedOf(ix, desk, s, slots, max) : null;
  const sideWords = key ? nodeOptWord(ix, node, desk, key, 'side') : undefined;
  const side = one ? {
    desc: typeWords(shown[0]!.type).desc,
    second: s.switchHere && !s.adv1 && slots.length === 1 ? { label: kindAt(1) ? msg('carts.second.kind', { kind: kindAt(1) }) : msg('carts.second'), tip: msg('carts.second.tip') } : null,
    note: max > 1 || !s.switchHere ? sideWords ?? null : null,
  } : null;
  const away = String(factOf(ix, 'adaptersAway', desk) ?? '').replace(/^\s*·\s*/, '');
  const more = quick && slots.length > 1 ? msg(!s.switchHere ? 'carts.more' : s.adv1 ? 'carts.more.stacked' : 'carts.more.first', { n: slots.length }) : null;
  return {
    look: quick ? 'flat' : 'rig', masksLabel: msg('carts.masks'), carts, add, planner, one, sideFoot: one && !lab && !quick, side, unused,
    away: away && !quick ? msg('carts.away', { words: away }) : null, more,
    list: listOf(desk, s, slots.length), plannerList: s.planner ? plannerListOf(ix, node, desk, key) : null,
  };
}

/** Which housing a slot's kind is: the planner's (a planner LM, YuE2's AR composer, HeartMuLa's global LM) or the
 *  sound's. Its kind's words, as the definition writes them, say it. */
const housingOf = (kind: string | Slot): 'sound' | 'plan' => (/^(Planner|AR\b|Global)/.test(typeof kind === 'string' ? kind : (kind as Msg).en ?? '') ? 'plan' : 'sound');

function unusedOf(ix: Index, desk: Desk, s: ReturnType<typeof shape>, slots: Slot0[], max: number): CartsModel['unused'] {
  const rest = slots.slice(max).map(x => x.file), names = list(rest);
  if (s.switchHere && !s.adv1 && s.adv && s.stackedValue !== null) {
    const pw = paramWords(ix, s.adv, desk), single = formatMsg(ix, s.adv, desk, valueOf(ix, s.adv, desk)!);
    return { text: msg('carts.unused.single', { names, n: rest.length, stack: pw.label, single }), stack: { label: formatMsg(ix, s.adv, desk, s.stackedValue), value: s.stackedValue, key: s.adv } };
  }
  return { text: msg('carts.unused.takes', { names, n: rest.length, engine: engineWords(desk), max }), stack: null };
}
const engineWords = (desk: Desk): string => engineName(desk.engine);

/** The adapter list for the next slot: the installed files of its kind (the server's list for that slot), but those
 *  loaded already, and never a DoRA in a planner's housing (B-02). */
function listOf(desk: Desk, s: ReturnType<typeof shape>, n: number): AdapterList | null {
  if (n >= 3) return null;
  const kind = s.kinds[n] ?? s.kinds[0] ?? '', plan = housingOf(kind) === 'plan';
  const files = (desk.lists?.[`adapters.${desk.engine}.${n}`] ?? []).filter(x => !s.slots.some(y => y.file === String(x.value)) && !(plan && x.kind === 'DoRA'));
  return { head: msg('carts.list', { kind, engine: engineWords(desk) }), note: null, items: files.map(x => item(String(x.value), x.label, x.kind)) };
}
function plannerListOf(ix: Index, node: Node, desk: Desk, key: string): AdapterList {
  const o = (key ? nodeOptWord(ix, node, desk, key, 'planner') : undefined) ?? '';
  return { head: msg('carts.list.planner', { kind: o }), note: (key ? nodeOptWord(ix, node, desk, key, 'plannerList') : undefined) ?? null, items: (desk.lists?.['adapters.lm'] ?? []).map(x => item(String(x.value), x.label, x.kind)) };
}
function item(value: string, label: string, kind: string | undefined) {
  const t = typeOf(kind ?? 'LoRA');
  return { file: value, label, value, type: t, cls: t.toLowerCase(), title: msg('carts.title', { type: t, desc: typeWords(t).desc }), art: artOf(t, value, null) };
}

/** The installed adapters as the page is told them (desk.lists): each slot's files on an engine under
 *  `adapters.<engine>.<slot>`, and the planner LM adapters under `adapters.lm`, named with their type as the planner
 *  adapter's choice lists them ("folk-voice-planner (LoRA)"). */
export function adapterLists(engine: string, slots: { file: string; type: string }[][], planner?: { file: string; type: string }[]): { [source: string]: { value: string; label: string; kind: string }[] } {
  const out: { [source: string]: { value: string; label: string; kind: string }[] } = {};
  slots.forEach((files, i) => { out[`adapters.${engine}.${i}`] = files.map(f => ({ value: f.file, label: f.file, kind: typeOf(f.type) })); });
  if (planner) out['adapters.lm'] = planner.map(f => ({ value: f.file, label: `${f.file} (${typeOf(f.type)})`, kind: typeOf(f.type) }));
  return out;
}

/** The desk fact `adaptersAway` (parts.ts DESK_FACTS): the other engines that keep adapters for when you switch back,
 *  from the engines' book, as words after a separator, or nothing (25-module-adapters.js awayText). */
export function adaptersAway(book: { stash: { [engine: string]: { state: { [k: string]: Json } } } }, engine: string, key = 'slots'): string {
  const away = Object.entries(book.stash).filter(([e, st]) => e !== engine && Array.isArray(st.state?.[key]) && (st.state[key] as Json[]).length);
  if (!away.length) return '';
  return ' · ' + away.map(([e, st]) => { const n = (st.state[key] as Json[]).length; return `${engineName(e as Parameters<typeof engineName>[0])} keeps its ${n === 1 ? 'adapter' : `${n} adapters`} for when you switch back`; }).join('; ');
}

/* ---------------------------------------------------------------- what a hand does */

/** The words of a mask change, from the lyrics `a` to `b`: one section of one adapter, or several. */
function maskStep(slots: Slot0[], a: string, b: string): { step: 'mask' | 'masks'; slots: { [k: string]: string | number } } | null {
  const da = directivesOf(a), db = directivesOf(b), secs = sectionsOf(b), hit: [number, number][] = [];
  db.forEach((d, j) => { for (let i = 0; i < Math.min(3, slots.length); i++) if (levelOf(weightAt(d, i)) !== levelOf(weightAt(da[j], i))) hit.push([j, i]); });
  if (!hit.length) return null;
  const file = (i: number) => slots[i]?.file ?? `adapter ${i + 1}`;
  if (hit.length > 1) { const fs = [...new Set(hit.map(h => file(h[1])))]; return { step: 'masks', slots: { file: fs.length > 1 ? fs.slice(0, -1).join(', ') + ' and ' + fs.at(-1) : fs[0]! } }; }
  const [j, i] = hit[0]!;
  return { step: 'mask', slots: { section: secs[j]?.name ?? `Section ${j + 1}`, file: file(i), from: levelOf(weightAt(da[j], i)), to: levelOf(weightAt(db[j], i)) } };
}

const stateKey = (x: ActIn) => x.node.state?.[0] ?? '';
defineInstrument('cartridges', (ix, node, desk, key) => ({ carts: cartsOf(ix, node, desk, key) }), {
  /* an adapter chosen from the list goes into the next slot; a second one loaded while the stack switch is off stacks
     them, in the same step (L-22) */
  load: x => {
    const o = (x.arg ?? {}) as { file?: string; type?: string }, st = x.v.state[stateKey(x)], s = shape(x.ix, x.node, x.desk, '');
    if (!st || !o.file || s.slots.length >= 3 || s.slots.some(y => y.file === o.file)) return false;
    const also = s.switchHere && !s.adv1 && s.slots.length === 1 && s.adv && s.stackedValue !== null ? { [s.adv]: s.stackedValue } : undefined;
    st.apply('load', [...s.slots, { file: o.file, type: typeOf(o.type ?? 'LoRA') }] as unknown as Json, also, { file: o.file });
  },
  /* × on a cartridge: the later strengths move up, the lyric directives are renumbered, and the toast offers Undo */
  remove: x => {
    const i = Number(x.arg), st = x.v.state[stateKey(x)], s = shape(x.ix, x.node, x.desk, '');
    const gone = s.slots[i]; if (!st || !gone) return false;
    /* every slot's strength, in order, as the remove step lists them (Quick binds the first alone) */
    const def = x.ix.states.get(stateKey(x))!.s.steps?.remove, keys = def ? stepChanges(def).filter(k => x.ix.params.get(k)?.p.type === 'number' && has(x.ix, k, x.desk.engine)) : s.strengths;
    const also: { [k: string]: Value } = {};
    for (let k = i; k < keys.length; k++) {
      const next = keys[k + 1];
      also[keys[k]!] = next ? valueOf(x.ix, next, x.desk)! : resolve(x.ix.params.get(keys[k]!)!.p, x.desk.engine).default;
    }
    const lyr = s.lyrics ? dropDirectiveSlot(String(valueOf(x.ix, s.lyrics, x.desk) ?? ''), i) : null;
    if (lyr !== null && s.lyrics) also[s.lyrics] = lyr;
    st.apply('remove', s.slots.filter((_, j) => j !== i) as unknown as Json, also, { file: gone.file });
    const left = lyr ?? String((s.lyrics && valueOf(x.ix, s.lyrics, x.desk)) ?? '');
    x.tell(msg(directivesOf(left).some(Boolean) ? 'carts.removed.renumbered' : 'carts.removed', { file: gone.file }), true);
  },
  /* the planner adapter chosen, or taken out ("") */
  planner: x => { const s = shape(x.ix, x.node, x.desk, ''), p = s.planner ? x.v.params[s.planner] : undefined; if (!p) return false; p.set(String(x.arg ?? '')); },
  /* a phase handle moved by a key or a hand (`arg`: the slot, the handle, the key or the place, and the window the
     gesture began from, so a drag is named from where it started) */
  phase: x => {
    const o = (x.arg ?? {}) as { i: number; h: 0 | 1; key?: string; f?: number; from?: [number, number] }, st = x.v.state[stateKey(x)], s = shape(x.ix, x.node, x.desk, '');
    const sl = s.slots[o.i]; if (!st || !sl || !s.stacked) return false;
    const ph = phaseOf(sl), to = phaseMove(ph, o.h, o);
    if (!to || J(to) === J(ph)) return to !== null;
    const next = s.slots.map((y, j) => (j === o.i ? { ...y, ph: to } : y));
    st.apply('phase', next as unknown as Json, undefined, { file: sl.file, from: span(o.from ?? ph), to: span(to) });
  },
  /* a section's mask chip set to the weight `w` (a press: its next; a drag: the first chip's), named from the lyrics the
     gesture began with (`from`) */
  mask: x => {
    const o = (x.arg ?? {}) as { i: number; j: number; w: number; from?: string }, st = x.v.state[stateKey(x)], s = shape(x.ix, x.node, x.desk, '');
    if (!st || !s.lyrics || !s.stacked) return false;
    const now = String(valueOf(x.ix, s.lyrics, x.desk) ?? ''), next = setDirective(now, o.j, o.i, o.w, Math.min(s.slots.length, 3));
    if (next === now) return true;
    const words = maskStep(s.slots, o.from ?? now, next);
    st.apply(words?.step ?? 'mask', s.slots as unknown as Json, { [s.lyrics]: next }, words?.slots ?? { section: '', file: '', from: '', to: '' });
  },
});

/* ---------------------------------------------------------------- the XY pad's off rail */

/** A pad whose Y minimum means off (its node's `yOff`): the plot runs from one step above the minimum, and the puck
 *  goes onto the rail at the minimum (E-28); `at` is the place on the plot, 0 at its foot and 1 at its top. */
export function xyRail(lo: number, hi: number, step: number, v: number): { off: boolean; at: number } {
  return v <= lo ? { off: true, at: 1 } : { off: false, at: clamp((v - lo - step) / (hi - lo - step), 0, 1) };
}
/** Where a hand at `f` along the plot (past 1: over the rail) takes such a value. */
export function xyRailValue(lo: number, hi: number, step: number, f: number, over: boolean): number {
  if (over) return lo;
  return +clamp(Math.round((lo + step + clamp(f, 0, 1) * (hi - lo - step)) / step) * step, lo + step, hi).toFixed(4);
}
/** Where a key takes such a value: off, up stays off and down goes to the widest; on, up past the widest is off; else a
 *  step (Shift ten, Page ten) within the plot (21-xy-pad.js). Null for a key that does not move Y. */
export function xyRailKey(lo: number, hi: number, step: number, v: number, key: string, shift: boolean): number | null {
  const d = ({ ArrowDown: -1, ArrowUp: 1, PageUp: 10, PageDown: -10 } as { [k: string]: number })[key];
  if (d === undefined) return null;
  if (v <= lo) return d > 0 ? lo : hi;
  if (d > 0 && v >= hi) return lo;
  return +clamp(v + d * (/^Page/.test(key) ? 1 : shift ? 10 : 1) * step, lo + step, hi).toFixed(4);
}
defineInstrument('xyPad', (ix, node, desk) => {
  const y = node.params?.[1], o = (node.opts as { yOff?: string } | undefined)?.yOff;
  if (!o || !y || !has(ix, y, desk.engine)) return {} as InstrumentModel;
  const r = resolve(ix.params.get(y)!.p, desk.engine).range; if (!r) return {} as InstrumentModel;
  return { rail: xyRail(r[0], r[1], r[2], Number(valueOf(ix, y, desk))) };
}, {
  /* a hand on a pad with an off rail: Y from its place on the plot, or off over the rail */
  y: x => {
    const k = x.node.params?.[1], p = k ? x.v.params[k] : undefined, r = k && has(x.ix, k, x.desk.engine) ? resolve(x.ix.params.get(k)!.p, x.desk.engine).range : undefined;
    const o = (x.arg ?? {}) as { f: number; over: boolean };
    if (!p || !r) return false;
    p.set(xyRailValue(r[0], r[1], r[2], o.f, o.over));
  },
  /* a key on such a pad's Y: true when it was one */
  yKey: x => {
    const k = x.node.params?.[1], p = k ? x.v.params[k] : undefined, r = k && has(x.ix, k, x.desk.engine) ? resolve(x.ix.params.get(k)!.p, x.desk.engine).range : undefined;
    const o = (x.arg ?? {}) as { key: string; shift: boolean };
    const to = p && r ? xyRailKey(r[0], r[1], r[2], Number(valueOf(x.ix, k!, x.desk)), o.key, o.shift) : null;
    if (to === null) return false;
    p!.set(to);
  },
});
