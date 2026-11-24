// The core of what is derived from the definitions: a parameter on an engine, its value, its words, whether a
// condition holds, which nodes show at a tier and what they edit, and how a number is written.
// docs/05-what-the-definition-drives.md describes each; the validator, the tools and (later) the page call these
// rather than restating them. Nothing here has a branch for any one module.

import type {
  Condition, Definition, EngineId, EngineOverride, Json, Node, NodeOverride, Option, OptionsFrom, ParamDef, ParamFace,
  StateDef, Tier, Value, WordsOverride,
} from '../definitions/types.ts';
import { engineName } from '../definitions/engines.ts';
import { DESK_FACTS, type FactInput } from '../definitions/parts.ts';

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
  /** The two desk facts the page is told rather than works out (parts.ts DESK_FACTS without `of`): the plugin host's
   *  count and the words for another engine's adapters. Every other fact is worked out when it is read (factOf), so
   *  none can be stale. */
  told?: { vstPlugins?: number; adaptersAway?: string };
  /** Modules, pedals and units on the desk. Built-in modules always are. */
  on: Set<string>;
  /** The models the server has selected, by `<engine>.<bucket>` (GET /api/backends/models): what a model choice
   *  left at "" (the engine's default) stands for. */
  selection?: { [bucket: string]: string };
  /** The lists the server gives for server-listed choices (parts.ts OPTION_SOURCES), with their words, so a value
   *  from one is written by its name ("DPM++ 3M" for dpm3m). A model bucket's list is the engine's own, under
   *  `models.<engine>.<bucket>` (listKey), and each file carries its size in GB. */
  lists?: { [source: string]: { value: string | number; label: string; gb?: number }[] };
  /** What the VRAM figures are worked out from beside the model files' sizes (derive/vram.ts): the card, and what
   *  each engine loads besides its model files and costs per take. Told by the models service. */
  vram?: VramFigures;
  /** What a server session holds, as its service tells it (package 3c: the build session's takes, a recording's
   *  length): never a step, never saved with the desk; the core reads it to draw what the server has done. */
  sessions?: { [service: string]: Json };
}

/** The VRAM figures the server tells the page (the models service): the card's size, and for each engine the parts it
 *  always loads (encoders, working memory), what each adapter it can load adds and how many it can load, and what
 *  each take adds: `gb` a take, only while the switch `when` is on, the first take included or not (`first`), and
 *  what happens when a run does not fit the card (`over`: planned in passes, refused before anything loads, or just
 *  tight). */
export interface VramFigures {
  card: number;
  engines: { [e in EngineId]?: {
    fixed: { name: string; gb: number }[];
    adapters?: { gb: number; most: number };
    take: { gb: number; when?: string; first: boolean };
    over: 'passes' | 'refuse' | 'tight';
  } };
}

/** The key of the server's list a choice draws from in `desk.lists`: its source, or for a model bucket the engine's
 *  own (`models.ace.dit`), since each engine lists its own files. */
export const listKey = (from: OptionsFrom, e: EngineId): string => (from.from === 'models' && from.bucket ? `models.${e}.${from.bucket}` : from.from);
/** Where the server's selected file of an engine's model bucket is kept in `desk.selection` (`ace.dit`). */
export const selectionKey = (e: EngineId, bucket: string): string => `${e}.${bucket}`;
/** A choice's options in two parts: those the definition lists itself (`include`, or its own list) and those the
 *  server lists for it, without a value twice and only the values it names (`only`). A model choice left at ""
 *  (the engine's default) stands for the file the server has selected for its bucket (modelDefault): while that file is
 *  listed, "" is not offered apart from it. */
export function choiceLists(f: ParamFace, desk: Desk, e: EngineId = desk.engine): { fixed: Option[]; listed: { value: string | number; label: string; gb?: number }[] } {
  const own = Array.isArray(f.options) ? f.options : f.options?.include ?? [];
  const from = !Array.isArray(f.options) && f.options ? f.options : null;
  const listed = from ? (desk.lists?.[listKey(from, e)] ?? []).filter(x => !own.some(o => o.value === x.value) && (!from.only || from.only.includes(String(x.value)))) : [];
  const stands = from?.loaded && from.bucket ? desk.selection?.[selectionKey(e, from.bucket)] : undefined;
  const folded = stands !== undefined && listed.some(x => x.value === stands);
  return { fixed: folded ? own.filter(o => o.value !== '') : own, listed };
}
/** The file a loaded model's "" (the engine's default) stands for: the one the server has selected for its bucket,
 *  when the server lists it; undefined for any other choice (docs/05, "What is sent": the request sends the same). */
export function modelDefault(ix: Index, key: string, desk: Desk): string | undefined {
  const hit = ix.params.get(key); if (!hit || !has(ix, key, desk.engine)) return undefined;
  const f = resolve(hit.p, desk.engine), from = !Array.isArray(f.options) && f.options ? f.options : null;
  if (from?.from !== 'models' || !from.loaded || !from.bucket) return undefined;
  const s = desk.selection?.[selectionKey(desk.engine, from.bucket)];
  return s !== undefined && (desk.lists?.[listKey(from, desk.engine)] ?? []).some(x => x.value === s) ? s : undefined;
}
/** The value a choice shows as chosen: a model choice's "" is the file it stands for. */
export const shownValue = (ix: Index, key: string, desk: Desk, v: Value): Value => (v === '' ? modelDefault(ix, key, desk) ?? v : v);

export const has = (ix: Index, key: string, e: EngineId): boolean => {
  const hit = ix.params.get(key);
  return !!hit && hit.d.engines.includes(e) && enginesOf(hit.p, hit.d).includes(e);
};

/* ---------------------------------------------------------------- the value in force
   One answer, used by conditions, desk facts, words, the changed marker, summaries and the request alike: a value is
   in force when its definition is on the desk (a built-in module always is; a module the tier hides is still on the
   desk, and its settings still apply). A definition that is not on the desk reads as its defaults everywhere: its
   values are kept for when it comes back, but nothing reads them, so the page and the request never describe two
   different songs. A pedal's foot switch defaults to off (the validator checks it), so a pedal off the board is off. */

/** Its values are in force: it is built in, or on the desk. */
export const onDesk = (d: Definition, desk: Desk): boolean => (d.kind === 'module' && !!d.builtIn) || desk.on.has(d.id);
/** The same desk with every definition on it: for what is about stored values whatever the desk shows (what an engine
 *  switch puts aside and holds, what a revision saves). */
export const everythingOn = (ix: Index, desk: Desk): Desk => ({ ...desk, on: new Set(ix.defs.map(d => d.id)) });
/** The definition a parameter or a state belongs to is on the desk. */
const keyOnDesk = (ix: Index, key: string, desk: Desk): boolean => {
  const d = ix.params.get(key)?.d ?? ix.states.get(key)?.d;
  return !!d && onDesk(d, desk);
};

export const isAuto = (ix: Index, key: string, desk: Desk): boolean => {
  const hit = ix.params.get(key);
  return !!hit && !!resolve(hit.p, desk.engine).auto && (!onDesk(hit.d, desk) || desk.auto[key] !== false);
};

/** The stored value as the desk's engine takes it, whether or not its definition is on the desk: a number clamped to
 *  the engine's range (a held value reads at the engine's limit), a choice the engine does not offer, or a value of the
 *  wrong type, as the default. What a revision saves and a control shows. Undefined when the engine lacks the
 *  parameter. Reading never writes. A number is put on the engine's step grid when it is set (values.ts setValue), not
 *  when it is read: a value carried from another engine is sent as it was set, as the existing page sends it. */
export function storedOn(ix: Index, key: string, desk: Desk): Value | undefined {
  const hit = ix.params.get(key);
  if (!hit || !has(ix, key, desk.engine)) return undefined;
  const f = resolve(hit.p, desk.engine), v = desk.vals[key];
  if (v === undefined) return f.default;
  if (hit.p.type === 'number') {
    if (typeof v !== 'number' || !Number.isFinite(v) || !f.range) return f.default;
    return Math.min(f.range[1], Math.max(f.range[0], v));
  }
  if (hit.p.type === 'boolean') return typeof v === 'boolean' ? v : f.default;
  if (hit.p.type === 'text') return typeof v === 'string' ? v : f.default;
  const o = f.options;
  if (Array.isArray(o)) return o.some(x => x.value === v) ? v : f.default;
  return typeof v === 'string' || typeof v === 'number' ? v : f.default;
}

/** The value in force on the desk's engine: the stored value as the engine takes it (storedOn) while its definition
 *  is on the desk, else its default. Undefined when the engine lacks it. */
export function valueOf(ix: Index, key: string, desk: Desk): Value | undefined {
  const hit = ix.params.get(key);
  if (!hit || !has(ix, key, desk.engine)) return undefined;
  return onDesk(hit.d, desk) ? storedOn(ix, key, desk) : resolve(hit.p, desk.engine).default;
}

/** A state's value in force: what it holds while its definition is on the desk, else nothing (an adapter loaded in
 *  an Adapters tile that was taken off the desk is not loaded). */
export const stateOf = (ix: Index, key: string, desk: Desk): Json | undefined => (keyOnDesk(ix, key, desk) ? desk.state[key] : undefined);

/** A desk fact's value: worked out from the values and state in force by its rule (parts.ts DESK_FACTS `of`), or as
 *  the page was told it. Undefined when `name` is no fact. */
export function factOf(ix: Index, name: string, desk: Desk): number | boolean | string | undefined {
  const f = (DESK_FACTS as Record<string, { of?: (x: FactInput) => number | boolean }>)[name];
  if (!f) return undefined;
  /* the validator's finite evaluation tries each fact's values directly (TrialDesk); the page never passes these */
  const trial = (desk as Partial<TrialDesk>).trialFacts;
  if (trial && name in trial) return trial[name];
  if (!f.of) return (desk.told as Record<string, number | string | undefined> | undefined)?.[name] ?? (name === 'adaptersAway' ? '' : 0);
  const state = new Proxy(desk.state, { get: (_, k) => (typeof k === 'string' ? stateOf(ix, k, desk) : undefined) });
  return f.of({ value: k => valueOf(ix, k, desk), state, engine: desk.engine });
}
/** A desk on which the validator tries desk facts' values directly instead of working them out, to ask whether a
 *  condition can ever hold. Only tools/validate-definitions.ts makes one. */
export interface TrialDesk extends Desk { trialFacts: { [name: string]: number | boolean | string } }

/** Every desk fact, for a part's reads and for tools that print them. */
export const factsOf = (ix: Index, desk: Desk): { [name: string]: number | boolean | string } =>
  Object.fromEntries(Object.keys(DESK_FACTS).map(n => [n, factOf(ix, n, desk)!]));

/** How many a state or a fact holds: a number as it is, a list by its length, anything else 1 or 0. */
export function countOf(ix: Index, key: string, desk: Desk): number {
  const v = key in DESK_FACTS ? factOf(ix, key, desk) : stateOf(ix, key, desk);
  return typeof v === 'number' ? v : Array.isArray(v) ? v.length : v ? 1 : 0;
}

export function holds(ix: Index, c: Condition, desk: Desk): boolean {
  if ('all' in c) return c.all.every(x => holds(ix, x, desk));
  if ('any' in c) return c.any.some(x => holds(ix, x, desk));
  if ('not' in c && !('param' in c)) return !holds(ix, c.not, desk);
  if ('engine' in c) return c.engine.includes(desk.engine);
  if ('onDesk' in c) return desk.on.has(c.onDesk);
  if ('state' in c) {
    const n = countOf(ix, c.state, desk);
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

export function formatValue(f: ParamFace, v: Value, lists?: Desk['lists'], e?: EngineId): string {
  const fm = f.format || {};
  const word = (fm.words || []).find(w => w.value === v);
  if (word) return word.text;
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  if (typeof v === 'string') {
    const o = (Array.isArray(f.options) ? f.options : [...(f.options?.include ?? []), ...(f.options ? lists?.[e ? listKey(f.options, e) : f.options.from] ?? [] : [])]).find(x => x.value === v);
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
  return formatValue(f, valueOf(ix, key, desk)!, desk.lists, desk.engine);
}

/* ---------------------------------------------------------------- words that name values (docs/05) */

export const SLOT = /\{(engine|min|max|default|value:[\w.-]+|count:[\w.-]+|say:[\w.-]+|file:[\w.-]+(?:\|[^}]*)?)\}/g;

export function fill(ix: Index, text: string, desk: Desk, own?: ParamFace): string {
  return text.replace(SLOT, (m, slot: string) => {
    if (slot === 'engine') return engineName(desk.engine);
    if (own && (slot === 'min' || slot === 'max')) return own.range ? formatValue(own, own.range[slot === 'min' ? 0 : 1]) : m;
    if (own && slot === 'default') return formatValue(own, own.default);
    const [kind, rest] = [slot.slice(0, slot.indexOf(':')), slot.slice(slot.indexOf(':') + 1)];
    if (kind === 'value') return ix.params.has(rest) && has(ix, rest, desk.engine) ? textOf(ix, rest, desk) : '';
    if (kind === 'count') return String(countOf(ix, rest, desk));
    if (kind === 'say') return String(factOf(ix, rest, desk) ?? '');
    const [path, empty] = rest.split('|') as [string, string | undefined];
    const [key, at] = path.split('.') as [string, string | undefined];
    let v: Json | undefined = stateOf(ix, key, desk);
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
  return (d.strip || []).find(s => (!s.tiers || s.tiers.includes(desk.tier)) && holds(ix, s.when, desk)) ?? null;
}
