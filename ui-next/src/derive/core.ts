// The core of what is derived from the definitions: a parameter on an engine, its value, its words, whether a
// condition holds, which nodes show at a tier and what they edit, and how a number is written.
// docs/05-what-the-definition-drives.md describes each; the validator, the tools and (later) the page call these
// rather than restating them. Nothing here has a branch for any one module.

import type {
  Condition, Definition, EngineId, EngineOverride, Json, Node, NodeOverride, ParamDef, ParamFace, StateDef, Tier,
  Value, WordsOverride,
} from '../definitions/types.ts';
import { engineName } from '../definitions/engines.ts';
import { DESK_FACTS } from '../definitions/parts.ts';

/* ---------------------------------------------------------------- the index */

export interface Index {
  defs: readonly Definition[];
  params: Map<string, { p: ParamDef; d: Definition }>;
  states: Map<string, { s: StateDef; d: Definition }>;
  /** Former keys (`was`) to their parameter. */
  former: Map<string, string>;
  /** Former ids (`was`) to their definition's id, and former state keys to their state's. */
  formerIds: Map<string, string>;
  formerStates: Map<string, string>;
}

export function indexOf(defs: readonly Definition[]): Index {
  const params = new Map<string, { p: ParamDef; d: Definition }>(), states = new Map<string, { s: StateDef; d: Definition }>();
  const former = new Map<string, string>(), formerIds = new Map<string, string>(), formerStates = new Map<string, string>();
  for (const d of defs) {
    (d.was || []).forEach(w => formerIds.set(w, d.id));
    for (const p of d.params) { params.set(p.key, { p, d }); (p.was || []).forEach(w => former.set(w, p.key)); }
    for (const s of d.state || []) { states.set(s.key, { s, d }); (s.was || []).forEach(w => formerStates.set(w, s.key)); }
  }
  return { defs, params, states, former, formerIds, formerStates };
}

/* ---------------------------------------------------------------- a parameter on an engine */

export const enginesOf = (p: ParamDef, d: Definition): readonly EngineId[] => p.engines ?? d.engines;
export const isEngineLine = (l: EngineOverride | WordsOverride): l is EngineOverride => 'engines' in l;

/** The base fields, then that engine's line (shallow, per field). Word lines are not applied: see wordsOf. */
export function resolve(p: ParamDef, e: EngineId): ParamDef {
  const line = (p.per || []).find((l): l is EngineOverride => isEngineLine(l) && l.engines.includes(e));
  if (!line) return p;
  const { engines: _e, ...fields } = line;
  return { ...p, ...fields } as ParamDef;
}

export const WORD_FIELDS = ['label', 'short', 'tip', 'auto'] as const;

/** The parameter's words on the desk: its engine's face, then every word line whose condition holds. */
export function wordsOf(ix: Index, p: ParamDef, desk: Desk): ParamFace {
  let f: ParamFace = resolve(p, desk.engine);
  /* a word line changes words only: its label, short, tip and auto words, nothing else (types.ts WordsOverride) */
  for (const l of p.per || []) if (!isEngineLine(l) && holds(ix, l.when, desk)) for (const k of WORD_FIELDS) if (l[k] !== undefined) f = { ...f, [k]: l[k] } as ParamFace;
  return f;
}

/** A definition's name on an engine: a module's engine line may rename it (Tempo is "Length" on Stable Audio 3). */
export const nameOn = (d: Definition, e: EngineId): string => (d.kind === 'module' ? (d.per || []).find(l => l.engines.includes(e))?.name : undefined) ?? d.name;

/* ---------------------------------------------------------------- the desk, and conditions */

/** What conditions, values and the request read. */
export interface Desk {
  engine: EngineId;
  tier: Tier;
  /** Stored values, by parameter key, in the definition's units. Absent: the default. */
  vals: { [key: string]: Value };
  /** Auto switches, by parameter key. Absent: on auto. */
  auto: { [key: string]: boolean };
  state: { [key: string]: Json };
  /** The desk facts (parts.ts DESK_FACTS) the page works out. */
  facts: { [name: string]: number | boolean };
  /** Modules, pedals and units on the desk. Built-in modules always are. */
  on: Set<string>;
  /** The models the server has selected, by `<engine>.<bucket>` (GET /api/backends/models): what a model choice
   *  left at "" (the engine's default) stands for. */
  selection?: { [bucket: string]: string };
  /** The lists the server gives for server-listed choices (parts.ts OPTION_SOURCES), with their words, so a value
   *  from one is written by its name ("DPM++ 3M" for dpm3m). */
  lists?: { [source: string]: { value: string | number; label: string }[] };
}

export const has = (ix: Index, key: string, e: EngineId): boolean => {
  const hit = ix.params.get(key);
  return !!hit && hit.d.engines.includes(e) && enginesOf(hit.p, hit.d).includes(e);
};

export const isAuto = (ix: Index, key: string, desk: Desk): boolean => {
  const hit = ix.params.get(key);
  return !!hit && !!resolve(hit.p, desk.engine).auto && desk.auto[key] !== false;
};

/** The value in force on the desk's engine: the stored value when the engine can take it, else the engine's limit
 *  for a number (a held value), else the default. Reading never writes. Undefined when the engine lacks it. */
export function valueOf(ix: Index, key: string, desk: Desk): Value | undefined {
  const hit = ix.params.get(key);
  if (!hit || !has(ix, key, desk.engine)) return undefined;
  const f = resolve(hit.p, desk.engine), v = desk.vals[key];
  if (v === undefined) return f.default;
  if (hit.p.type === 'number') {
    if (typeof v !== 'number' || !f.range) return f.default;
    return Math.min(f.range[1], Math.max(f.range[0], v));
  }
  if (hit.p.type === 'boolean') return typeof v === 'boolean' ? v : f.default;
  if (hit.p.type === 'text') return typeof v === 'string' ? v : f.default;
  const o = f.options;
  if (Array.isArray(o)) return o.some(x => x.value === v) ? v : f.default;
  return typeof v === 'string' || typeof v === 'number' ? v : f.default;
}

export function holds(ix: Index, c: Condition, desk: Desk): boolean {
  if ('all' in c) return c.all.every(x => holds(ix, x, desk));
  if ('any' in c) return c.any.some(x => holds(ix, x, desk));
  if ('not' in c && !('param' in c)) return !holds(ix, c.not, desk);
  if ('engine' in c) return c.engine.includes(desk.engine);
  if ('onDesk' in c) return desk.on.has(c.onDesk);
  if ('state' in c) {
    const v = c.state in desk.facts ? desk.facts[c.state] : desk.state[c.state];
    const n = typeof v === 'number' ? v : Array.isArray(v) ? v.length : v ? 1 : 0;
    if (c.atLeast !== undefined) return n >= c.atLeast;
    return c.set === undefined ? n > 0 : (n > 0) === c.set;
  }
  if (!has(ix, c.param, desk.engine)) return false;
  if (c.auto !== undefined) return isAuto(ix, c.param, desk) === c.auto;
  const v = valueOf(ix, c.param, desk)!, list = (x: Value | Value[]) => (Array.isArray(x) ? x : [x]);
  if (c.is !== undefined && !list(c.is).includes(v)) return false;
  if (c.not !== undefined && list(c.not).includes(v)) return false;
  if (c.above !== undefined && !(typeof v === 'number' && v > c.above)) return false;
  if (c.below !== undefined && !(typeof v === 'number' && v < c.below)) return false;
  return true;
}

/** The desk facts worked out from the desk (parts.ts DESK_FACTS, each one's `of`), over those the page was told. */
export function factsOf(ix: Index, desk: Desk): Desk['facts'] {
  const out = { ...desk.facts };
  const input = { value: (k: string) => valueOf(ix, k, desk), state: desk.state, on: desk.on, engine: desk.engine };
  for (const [name, f] of Object.entries(DESK_FACTS) as [string, { of?: (x: typeof input) => number | boolean }][]) if (f.of) out[name] = f.of(input);
  return out;
}

/** It is on the engine and its own `when` holds: the concept's ok(k). */
export function applies(ix: Index, key: string, desk: Desk): boolean {
  if (!has(ix, key, desk.engine)) return false;
  const f = resolve(ix.params.get(key)!.p, desk.engine);
  return !f.when || holds(ix, f.when, desk);
}

export function changed(ix: Index, key: string, desk: Desk): boolean {
  if (!applies(ix, key, desk)) return false;
  const f = resolve(ix.params.get(key)!.p, desk.engine);
  if (f.auto) return !isAuto(ix, key, desk);
  return key in desk.vals && valueOf(ix, key, desk) !== f.default;
}

/* ---------------------------------------------------------------- how a number is written (docs/05) */

const minus = (s: string) => s.replace(/^-/, '−');
export const mmss = (s: number) => { const t = Math.max(0, Math.round(s)); return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`; };
export const seconds = (v: number) => (v >= 60 ? mmss(v) : `${v.toFixed(v < 10 ? 1 : 0)} s`);

export function formatValue(f: ParamFace, v: Value, lists?: Desk['lists']): string {
  const fm = f.format || {};
  const word = (fm.words || []).find(w => w.value === v);
  if (word) return word.text;
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  if (typeof v === 'string') {
    const o = (Array.isArray(f.options) ? f.options : [...(f.options?.include ?? []), ...(f.options ? lists?.[f.options.from] ?? [] : [])]).find(x => x.value === v);
    return o ? o.label : v;
  }
  const step = f.range?.[2] ?? 1;
  const dec = fm.decimals ?? (step < 0.1 ? 2 : step < 1 ? 1 : 0);
  if (fm.template) return fm.template.replace('{v}', minus(v.toFixed(dec))).replace('{s}', fm.perSecond ? seconds(v / fm.perSecond) : '');
  if (f.unit === 's') return v >= 60 ? mmss(v) : `${v.toFixed(step < 1 && v < 10 ? 1 : 0)} s`;
  return minus(v.toFixed(dec)) + (f.unit ? (f.unit === '%' ? '%' : ' ' + f.unit) : '');
}

/** What a parameter says it is: its formatted value, or its auto words. */
export function textOf(ix: Index, key: string, desk: Desk): string {
  const hit = ix.params.get(key)!, f = wordsOf(ix, hit.p, desk);
  if (isAuto(ix, key, desk)) return fill(ix, (f.auto && f.auto.says) || `≈ ${formatValue(f, f.default)}`, desk, f);
  return formatValue(f, valueOf(ix, key, desk)!, desk.lists);
}

/* ---------------------------------------------------------------- words that name values (docs/05) */

export const SLOT = /\{(engine|min|max|default|value:[\w.-]+|count:[\w.-]+|file:[\w.-]+(?:\|[^}]*)?)\}/g;

export function fill(ix: Index, text: string, desk: Desk, own?: ParamFace): string {
  return text.replace(SLOT, (m, slot: string) => {
    if (slot === 'engine') return engineName(desk.engine);
    if (own && (slot === 'min' || slot === 'max')) return own.range ? formatValue(own, own.range[slot === 'min' ? 0 : 1]) : m;
    if (own && slot === 'default') return formatValue(own, own.default);
    const [kind, rest] = [slot.slice(0, slot.indexOf(':')), slot.slice(slot.indexOf(':') + 1)];
    if (kind === 'value') return ix.params.has(rest) && has(ix, rest, desk.engine) ? textOf(ix, rest, desk) : '';
    if (kind === 'count') { const v = rest in desk.facts ? desk.facts[rest] : desk.state[rest]; return String(typeof v === 'number' ? v : Array.isArray(v) ? v.length : v ? 1 : 0); }
    const [path, empty] = rest.split('|') as [string, string | undefined];
    const [key, at] = path.split('.') as [string, string | undefined];
    let v: Json | undefined = desk.state[key];
    if (Array.isArray(v)) v = v[Number(at ?? 0)];
    if (v && typeof v === 'object' && !Array.isArray(v) && 'file' in v) v = (v as { file: Json }).file;
    return typeof v === 'string' && v ? v : empty ?? '';
  });
}

/* ---------------------------------------------------------------- faces */

/** A node as it is on the desk: its lines applied, or null when it is not shown (tiers, engines, when). */
export function nodeAt(ix: Index, n: Node, desk: Desk): Node | null {
  if (n.tiers && !n.tiers.includes(desk.tier)) return null;
  if (n.engines && !n.engines.includes(desk.engine)) return null;
  let out = n;
  for (const l of (n.per || []) as NodeOverride[]) {
    if (l.engines && !l.engines.includes(desk.engine)) continue;
    if (l.tiers && !l.tiers.includes(desk.tier)) continue;
    if (l.when && !holds(ix, l.when, desk)) continue;
    const { engines: _e, tiers: _t, when: _w, ...fields } = l;
    out = { ...out, ...fields } as Node;
  }
  if (out.when && !holds(ix, out.when, desk)) return null;
  return out;
}

/** The parameters a node edits: its `param` and `params`, and the keys its Quick choices set. */
export const editsOf = (n: Partial<Pick<Node, 'param' | 'params' | 'choices'>>): string[] => [...new Set([n.param, ...(n.params || []),
  ...(n.choices || []).flatMap(c => (typeof c[1] === 'object' ? Object.keys(c[1]) : []))].filter((x): x is string => !!x))];

/** The parameters a definition's face lets you change at the desk's tier: the edits of every node shown, whose
 *  parameter applies. This is what "Also set", the changed marker's tests and the coverage check read. */
export function editsAt(ix: Index, d: Definition, desk: Desk): Set<string> {
  const out = new Set<string>();
  /* a tile that is a strip at this tier shows no controls */
  if (stripOf(ix, d, desk)) return out;
  const walk = (n: Node) => {
    const at = nodeAt(ix, n, desk);
    if (!at) return;
    editsOf(at).forEach(k => { if (applies(ix, k, desk)) out.add(k); });
    (at.children || []).forEach(walk);
  };
  ('face' in d ? d.face || [] : []).forEach(walk);
  return out;
}

/** The strip a module's tile is at the desk's tier, or null: the first of its strips that holds there. */
export function stripOf(ix: Index, d: Definition, desk: Desk): NonNullable<Extract<Definition, { kind: 'module' }>['strip']>[number] | null {
  if (d.kind !== 'module') return null;
  return (d.strip || []).find(s => (!s.tiers || s.tiers.includes(desk.tier)) && holds(ix, s.when, { ...desk, facts: factsOf(ix, desk) })) ?? null;
}

/** What a pedal's blanking plate says over the knobs its parameters' `when` hides on the desk (docs/01, "How a pedal
 *  and a side unit are laid out"): for knobs hidden by { param, is }, "with <that parameter's label>: <the labels of
 *  the values that show them, joined by or>". Null when nothing is hidden, or a hidden knob's condition is not of that
 *  form (the plate is then blank). */
export function plateWords(ix: Index, d: Definition, desk: Desk): string | null {
  const hidden = d.params.filter(p => p.type === 'number' && has(ix, p.key, desk.engine) && !applies(ix, p.key, desk));
  if (!hidden.length) return null;
  const by = new Map<string, Set<Value>>();
  for (const p of hidden) {
    const w = resolve(p, desk.engine).when as { param?: string; is?: Value | Value[] } | undefined;
    if (!w || !w.param || w.is === undefined || Object.keys(w).length !== 2) return null;
    const set = by.get(w.param) || new Set<Value>(); [w.is].flat().forEach(v => set.add(v)); by.set(w.param, set);
  }
  return 'with ' + [...by].map(([key, vals]) => {
    const f = wordsOf(ix, ix.params.get(key)!.p, desk);
    const labels = [...vals].map(v => formatValue(f, v));
    return `${f.label}: ${labels.slice(0, -1).join(', ')}${labels.length > 1 ? ' or ' : ''}${labels[labels.length - 1]}`;
  }).join('; ');
}
