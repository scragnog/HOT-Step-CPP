// Checks the definitions for what the type check cannot see, and checks their JSON form against the format too, so
// a definition that arrives as JSON (or skips its `satisfies`) gets the same check. Every message names the file,
// the parameter or node, the engine, and what to change.
//
//   node tools/validate-definitions.ts             the checks; exit 1 on any error
//   node tools/validate-definitions.ts --notes     also print the notes (things worth a look that are not errors)
//   node tools/validate-definitions.ts --docs      rewrite the generated parts of docs/ (counts and tables)
//   node tools/validate-definitions.ts --json f    check one definition file in JSON form, as a server might send it
//
// Needs Node 22.18 or later (it runs the TypeScript sources directly) and tools/fixtures/real-app.json (npm run
// fixtures writes it from the real application's source).

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { DEFINITIONS as LISTED } from '../src/definitions/index.ts';
import { FACTORY_WORKFLOWS, FACTORY_REVISIONS } from '../src/definitions/factory.ts';
import { ENGINES, ENGINE_IDS } from '../src/definitions/engines.ts';
import { PARTS, CHECKS, DESK_FACTS, OPTION_SOURCES, ACTIONS, layOut, type PartSpec } from '../src/definitions/parts.ts';
import { TRANSFORMS, type Transform } from '../src/definitions/transforms.ts';
import { RETIRED } from '../src/definitions/retired.ts';
import { ROLES, type Role, type RoleSpec } from '../src/definitions/roles.ts';
import type {
  Action, Condition, Definition, EngineId, Json, Node, NodeOverride, Option, ParamDef, Step, Strip, Tier, Value, Wire, WireKey,
  WireMap,
} from '../src/definitions/types.ts';
import { stepSays, stepChanges } from '../src/derive/steps.ts';
import {
  type Desk, type Index, type TrialDesk, SLOT, applies, cleanState, cleanValues, enginesOf, editsOf, holds, indexOf, isEngineLine, nodeAt, realState, resolve,
  toOf, wiresOf, realDefaultsFrom,
} from '../src/derive.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFDIR = path.join(ROOT, 'src', 'definitions');
const FIX = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools', 'fixtures', 'real-app.json'), 'utf8')) as {
  generationDefaults: Record<string, Json>; createContent: Record<string, Json>; createContentFields: string[];
  extensions: Record<string, { key: string; type: string; default?: Json; min?: number; max?: number; step?: number; options?: Json }[]>;
  reads: Record<string, string[]>; fields: Record<string, string[]>; plugins: Record<string, string[]>; buckets: Record<string, string[]>;
  sliders: Record<string, (number | null)[][]>; values: Record<string, string[]>; intent: { compose: string[] };
  core: Record<string, { batchMax: number | null; durationMax: number | null }>;
  features: Record<string, Record<string, boolean>>;
};
const TIERS: Tier[] = ['quick', 'studio', 'lab'];
/* --json <file>: a definition that arrives as JSON is checked as if it were in the index, every check included; it takes
   the place of a listed definition with its id, or joins them */
const jsonArg = process.argv.indexOf('--json');
const JSON_FILE = jsonArg > 0 ? process.argv[jsonArg + 1]! : null;
const JSON_DEF: Definition | null = JSON_FILE ? JSON.parse(fs.readFileSync(JSON_FILE, 'utf8')) : null;
const DEFINITIONS: readonly Definition[] = JSON_DEF ? [...LISTED.filter(d => d.id !== JSON_DEF.id), JSON_DEF] : LISTED;

const CHANNELS = ['param', 'ext', 'content', 'body', 'intent', 'model', 'ui'];
const errors: string[] = [], notes: string[] = [];
const err = (where: string, msg: string) => errors.push(`${where}: ${msg}`);
const note = (where: string, msg: string) => notes.push(`${where}: ${msg}`);
const J = (x: unknown) => JSON.stringify(x);
const fileOf = (d: Definition) => `${d.kind === 'module' ? 'modules' : d.kind === 'pedal' ? 'pedals' : 'units'}/${d.id}.ts`;

/* ================================================================ 1. the files, and their JSON form */

const listed = new Set(DEFINITIONS.map(d => `${d.kind}/${d.id}`));
for (const [dir, kind] of [['modules', 'module'], ['pedals', 'pedal'], ['units', 'unit']] as const) {
  for (const f of fs.readdirSync(path.join(DEFDIR, dir)).filter(f => f.endsWith('.ts'))) {
    if (!listed.has(`${kind}/${f.slice(0, -3)}`) && !(JSON_DEF && JSON_DEF.id === f.slice(0, -3))) err(`${dir}/${f}`, 'is not listed in index.ts, or its id does not match its file name');
  }
}
for (const d of DEFINITIONS) {
  const walk = (v: unknown, at: string): void => {
    if (typeof v === 'function' || typeof v === 'bigint' || typeof v === 'symbol' || v === undefined) err(fileOf(d), `${at} holds ${typeof v}, which JSON cannot carry`);
    else if (typeof v === 'number' && !Number.isFinite(v)) err(fileOf(d), `${at} is not a finite number`);
    else if (Array.isArray(v)) v.forEach((x, i) => walk(x, `${at}[${i}]`));
    else if (v && typeof v === 'object') for (const [k, x] of Object.entries(v)) walk(x, `${at}.${k}`);
  };
  walk(d, d.id);
}

/** Type-checks JSON documents against the format with the compiler: the one statement of the format is types.ts. */
function checkShapes(docs: { name: string; json: unknown; type: string }[]): void {
  const virt = path.join(ROOT, '__shapes__.ts').replace(/\\/g, '/');
  const lines = [`import type { ModuleDef, PedalDef, UnitDef, Workflow, Revision } from './src/definitions/types.ts';`];
  const at: { name: string; from: number }[] = [];
  let lineNo = 1;
  for (const doc of docs) {
    const chunk = `export const d${at.length}: ${doc.type} = ${JSON.stringify(doc.json, null, 1)};`;
    at.push({ name: doc.name, from: lineNo });
    lines.push(chunk);
    lineNo += chunk.split('\n').length;
  }
  const text = lines.join('\n');
  const opts: ts.CompilerOptions = { strict: true, exactOptionalPropertyTypes: true, noEmit: true, allowImportingTsExtensions: true, module: ts.ModuleKind.NodeNext, moduleResolution: ts.ModuleResolutionKind.NodeNext, target: ts.ScriptTarget.ES2022, skipLibCheck: true, types: [] };
  const host = ts.createCompilerHost(opts);
  const get = host.getSourceFile.bind(host);
  host.getSourceFile = (f, l, ...r) => (f.replace(/\\/g, '/') === virt ? ts.createSourceFile(f, text, l) : get(f, l, ...r));
  const exists = host.fileExists.bind(host);
  host.fileExists = f => f.replace(/\\/g, '/') === virt || exists(f);
  const program = ts.createProgram([virt], opts, host);
  const src = program.getSourceFile(virt)!;
  for (const dg of ts.getPreEmitDiagnostics(program, src)) {
    if (dg.file?.fileName.replace(/\\/g, '/') !== virt || dg.start === undefined) continue;
    const line = src.getLineAndCharacterOfPosition(dg.start).line;
    const owner = [...at].reverse().find(a => line >= a.from) ?? at[0]!;
    const msg = ts.flattenDiagnosticMessageText(dg.messageText, ' ').replace(/\s+/g, ' ');
    const short = msg.length > 300 ? msg.slice(0, msg.indexOf('.', 120) + 1 || 300) : msg;
    const field = /"([\w.]+)"/.exec(src.text.split('\n')[line] ?? '')?.[1];
    err(owner.name, `in its JSON form: ${short}${field ? ` (at "${field}")` : ''}`);
  }
}
const TYPE = { module: 'ModuleDef', pedal: 'PedalDef', unit: 'UnitDef' } as const;
checkShapes([
  ...DEFINITIONS.map(d => ({ name: fileOf(d), json: JSON.parse(J(d)), type: TYPE[d.kind] })),
  ...FACTORY_WORKFLOWS.map(w => ({ name: `factory.ts › workflow "${w.name}"`, json: JSON.parse(J(w)), type: 'Workflow' })),
  ...FACTORY_REVISIONS.map(r => ({ name: `factory.ts › revision "${r.workflow} › ${r.name}"`, json: JSON.parse(J(r)), type: 'Revision' })),
]);

/* ================================================================ 2. the index */

const ix: Index = indexOf(DEFINITIONS);
const ids = new Map<string, Definition>(), keyOwners = new Map<string, string>();
const KEY = /^[a-z][a-z0-9]*(\.[a-z0-9]+)*$/i;
for (const d of DEFINITIONS) {
  const at = fileOf(d);
  if (ids.has(`${d.kind}:${d.id}`)) err(at, `id "${d.id}" is used by two ${d.kind}s`); ids.set(`${d.kind}:${d.id}`, d);
  if (DEFINITIONS.some(x => x !== d && x.id === d.id)) err(at, `id "${d.id}" is also another definition's: a workflow names items by id, so ids are unique across modules, pedals and units`);
  for (const p of d.params) {
    if (keyOwners.has(p.key)) err(at, `parameter key "${p.key}" is also used in ${keyOwners.get(p.key)}`); keyOwners.set(p.key, at);
    if (!KEY.test(p.key)) err(`${at} › ${p.key}`, 'a key is letters and digits in dot-separated parts (no "-"), so its translation key is its own');
    for (const w of p.was || []) { if (keyOwners.has(w) || ix.params.has(w)) err(`${at} › ${p.key}`, `former key "${w}" is a key in use`); }
  }
  for (const s of d.state || []) {
    if (keyOwners.has(s.key)) err(at, `state key "${s.key}" is also used in ${keyOwners.get(s.key)}`); keyOwners.set(s.key, at);
    if (s.key in DESK_FACTS) err(`${at} › ${s.key}`, 'a state key may not be a desk fact\'s name (parts.ts DESK_FACTS)');
  }
}
for (const kind of ['module', 'pedal', 'unit'] as const) {
  const byAt = new Map<number, string>();
  for (const d of DEFINITIONS.filter(x => x.kind === kind && !(x.kind === 'module' && (x.builtIn || x.source)))) {
    if (byAt.has(d.at)) err(fileOf(d), `at ${d.at} is also ${byAt.get(d.at)}'s: pick a number between its neighbours`); byAt.set(d.at, d.id);
  }
}
const itemIds = new Set(DEFINITIONS.map(d => d.id));

/* ---- which engines a definition is on, against the subsystems each engine's server declares
   (capabilities().features). A definition that is a server subsystem is on exactly the engines that have it, except
   where docs/08 question 17 asks the maintainer and the definitions keep the concept's list meanwhile. */
const FEATURE_OF: Record<string, string> = { ss: 'stableStep', wh: 'whisper', lrc: 'lyricTimestamps', fa: 'forcedAlignment' };
const FEATURE_ASKED: Record<string, string[]> = { wh: ['sa3'] };
for (const [id, feature] of Object.entries(FEATURE_OF)) {
  const d = DEFINITIONS.find(x => x.id === id); if (!d) continue;
  for (const { id: e } of ENGINES) {
    const server = !!FIX.features[e]?.[feature], ours = d.engines.includes(e);
    if (server !== ours && !(FEATURE_ASKED[id] ?? []).includes(e)) err(fileOf(d), `${ours ? 'is on' : 'is not on'} ${e}, where the server's capabilities().features.${feature} is ${server}: put it on exactly the engines that have it`);
  }
}

/* ---- a definition put on an engine must say each parameter there: a parameter that lists its own engines and is on
   every other engine of its definition is missing on this one by mistake (a Prompt on a new engine with no prompt
   text), unless it is one of these, which are left off one engine on purpose */
const ABSENT_ON: Record<string, string> = {
  'wild.auto': 'heartmula', 'ad.trigger': 'sa3', 'ad.att': 'ace', 'ad.mlp': 'ace', 'ad.early': 'ace', 'ad.mid': 'ace', 'ad.late': 'ace',
  'ad.beta': 'minimax-m3', 'samp.cfg': 'yue2', 'samp.guide': 'yue2', 'apg.mom': 'minimax-m3',
};
for (const d of DEFINITIONS) for (const p of d.params) {
  if (!p.engines || d.engines.length < 3) continue;
  const missing = d.engines.filter(e => !p.engines!.includes(e));
  if (missing.length === 1 && ABSENT_ON[p.key] !== missing[0]) err(`${fileOf(d)} › ${p.key}`, `is on every engine of ${d.name} but ${missing[0]}: add ${missing[0]} to its engines with a line for it, or, if it is left off on purpose, say so in ABSENT_ON in tools/validate-definitions.ts`);
}

/* ================================================================ 3. conditions */

type Ctx = { where: string; engines: readonly EngineId[]; own?: Definition };
const names = (c: Condition): string[] => 'all' in c ? c.all.flatMap(names) : 'any' in c ? c.any.flatMap(names) : 'not' in c && !('param' in c) ? names(c.not) : 'param' in c ? [c.param] : [];

/** Every name in a condition is real, every value is one the parameter can hold, and it can hold on some engine. */
function checkCond(c: Condition, cx: Ctx, soft = false): void {
  if ('all' in c) {
    const g = c.all.find((x): x is { engine: EngineId[] } => 'engine' in x);
    const narrowed = g ? cx.engines.filter(e => g.engine.includes(e)) : cx.engines;
    c.all.forEach(x => checkCond(x, { ...cx, engines: narrowed.length ? narrowed : cx.engines }, soft));
    return;
  }
  if ('any' in c) { c.any.forEach(x => checkCond(x, cx, true)); return; }
  if ('not' in c && !('param' in c)) { checkCond(c.not, cx, true); return; }
  if ('engine' in c) { c.engine.forEach(e => { if (!ENGINE_IDS.includes(e)) err(cx.where, `names engine "${e}", which engines.ts does not list`); }); return; }
  if ('onDesk' in c) { if (!itemIds.has(c.onDesk)) err(cx.where, `names "${c.onDesk}", which is no module, pedal or unit`); return; }
  if ('state' in c) {
    if (!ix.states.has(c.state) && !(c.state in DESK_FACTS)) err(cx.where, `names state "${c.state}": no definition declares it and it is not a desk fact (parts.ts DESK_FACTS)`);
    return;
  }
  const hit = ix.params.get(c.param);
  if (!hit) { err(cx.where, `names parameter "${c.param}", which no definition has${ix.former.has(c.param) ? ` (it is now "${ix.former.get(c.param)}")` : ''}`); return; }
  const on = cx.engines.filter(e => enginesOf(hit.p, hit.d).includes(e) && hit.d.engines.includes(e));
  if (!on.length) { if (!soft) err(cx.where, `tests "${c.param}", which none of ${cx.engines.join(', ')} has, so it is never true: guard it with an engine test or drop it`); return; }
  const vals = [...[c.is ?? []].flat(), ...[c.not ?? []].flat()] as Value[];
  for (const e of on) {
    const f = resolve(hit.p, e), allowed = allowedValues(hit.p, e);
    for (const v of vals) {
      if (hit.p.type === 'boolean' && typeof v !== 'boolean') err(cx.where, `compares the switch "${c.param}" with ${J(v)}: use true or false`);
      if (hit.p.type === 'number' && typeof v !== 'number') err(cx.where, `compares the number "${c.param}" with ${J(v)}`);
      if (hit.p.type === 'number' && typeof v === 'number' && f.range && (v < f.range[0] || v > f.range[1])) err(cx.where, `compares "${c.param}" with ${v}, outside its range ${f.range[0]}..${f.range[1]} on ${e}`);
      if (hit.p.type === 'choice' && allowed && !allowed.includes(v as string)) {
        const now = Array.isArray(f.options) ? f.options.find(o => (o.was || []).includes(v as string)) : undefined;
        err(cx.where, `compares "${c.param}" with ${J(v)}, which is not one of its options on ${e} ${now ? `(it is now ${J(now.value)}, which lists ${J(v)} in was)` : `(${allowed.slice(0, 8).map(x => J(x)).join(', ')}${allowed.length > 8 ? ', ...' : ''})`}`);
      }
    }
    for (const b of [c.above, c.below]) {
      if (b === undefined) continue;
      if (hit.p.type !== 'number') err(cx.where, `uses above/below on "${c.param}", which is not a number`);
      else if (f.range && (b < f.range[0] || b > f.range[1])) err(cx.where, `tests "${c.param}" against ${b}, outside its range ${f.range[0]}..${f.range[1]} on ${e}`);
    }
    if (c.auto !== undefined && !f.auto) err(cx.where, `tests the auto switch of "${c.param}", which has no auto on ${e}`);
  }
}

/** The values a choice can hold on an engine: its options, or the server's list (from the fixture) and `include`;
 *  null when the list is only known at run time. */
function allowedValues(p: ParamDef, e: EngineId): (string | number)[] | null {
  const o = resolve(p, e).options;
  if (!o) return null;
  if (Array.isArray(o)) return o.map(x => x.value);
  const fx = OPTION_SOURCES[o.from].fixture as string | null;
  const inc = (o.include || []).map(x => x.value);
  if (!fx) return null;
  const list = FIX.plugins[fx.split('.')[1]!] ?? [];
  return [...inc, ...(o.only ?? list)];
}

/* ---- finite evaluation: does a condition ever hold, do two writers ever write together ---- */
function domain(key: string, e: EngineId): Value[] | null {
  const hit = ix.params.get(key); if (!hit) return null;
  const f = resolve(hit.p, e);
  if (hit.p.type === 'boolean') return [true, false];
  if (hit.p.type === 'choice') { const a = allowedValues(hit.p, e); return a ? a : [f.default, '\u0000another']; }
  if (hit.p.type === 'number' && f.range) {
    const pts = new Set<number>([f.range[0], f.range[1], f.default as number]);
    return [...pts];
  }
  return [f.default, ''];
}
function thresholds(c: Condition, acc: Map<string, Set<number>>): void {
  if ('all' in c) c.all.forEach(x => thresholds(x, acc)); else if ('any' in c) c.any.forEach(x => thresholds(x, acc));
  else if ('not' in c && !('param' in c)) thresholds(c.not, acc);
  else if ('param' in c) { const s = acc.get(c.param) || new Set(); [c.above, c.below].forEach(b => { if (b !== undefined) [b - 1e-6, b, b + 1e-6].forEach(x => s.add(x)); }); [c.is, c.not].flat().forEach(v => { if (typeof v === 'number') s.add(v); }); acc.set(c.param, s); }
}
function deskWith(e: EngineId, vals: Record<string, Value>, auto: Record<string, boolean>, state: Record<string, Json>, on: Set<string>): Desk {
  const trial: TrialDesk = { engine: e, tier: 'lab', vals, auto, state, trialFacts: Object.fromEntries(Object.keys(DESK_FACTS).map(k => [k, (state[k] as number) ?? 0])), on };
  return trial;
}
function leaves(c: Condition, pick: (c: Condition) => string | null): string[] {
  if ('all' in c) return c.all.flatMap(x => leaves(x, pick));
  if ('any' in c) return c.any.flatMap(x => leaves(x, pick));
  if ('not' in c && !('param' in c)) return leaves(c.not, pick);
  const v = pick(c); return v === null ? [] : [v];
}
const allOn = new Set(DEFINITIONS.map(d => d.id));
/** Every combination of the values the conditions can tell apart, up to a few thousand. */
function* combos(conds: Condition[], e: EngineId): Generator<Desk> {
  const keys = [...new Set(conds.flatMap(names))].filter(k => ix.params.has(k));
  const thr = new Map<string, Set<number>>(); conds.forEach(c => thresholds(c, thr));
  const doms = keys.map(k => { const d = domain(k, e) ?? []; const t = thr.get(k); return t && ix.params.get(k)!.p.type === 'number' ? [...new Set([...d, ...t])] : d; });
  const stateKeys = [...new Set(conds.flatMap(c => leaves(c, x => ('state' in x ? x.state : null))))];
  const desks = [...new Set(conds.flatMap(c => leaves(c, x => ('onDesk' in x ? x.onDesk : null))))];
  const autos = keys.filter(k => resolve(ix.params.get(k)!.p, e).auto);
  const axes: { set: (o: { vals: Record<string, Value>; auto: Record<string, boolean>; state: Record<string, Json>; on: Set<string> }, v: unknown) => void; values: unknown[] }[] = [
    ...keys.map((k, i) => ({ set: (o: any, v: unknown) => { o.vals[k] = v as Value; }, values: doms[i]! })),
    ...autos.map(k => ({ set: (o: any, v: unknown) => { o.auto[k] = v as boolean; }, values: [true, false] })),
    ...stateKeys.map(k => ({ set: (o: any, v: unknown) => { o.state[k] = v as Json; }, values: [0, 1, 2, 3] })),
    ...desks.map(k => ({ set: (o: any, v: unknown) => { if (!v) o.on.delete(k); }, values: [true, false] })),
  ];
  let count = 1; for (const a of axes) count *= Math.max(1, a.values.length);
  if (count > 20000) return;
  const idx = axes.map(() => 0);
  for (;;) {
    /* everything is on the desk but what a condition asks about: a definition off the desk reads as its defaults */
    const o = { vals: {} as Record<string, Value>, auto: {} as Record<string, boolean>, state: {} as Record<string, Json>, on: new Set<string>(allOn) };
    axes.forEach((a, i) => a.values.length && a.set(o, a.values[idx[i]!]));
    yield deskWith(e, o.vals, o.auto, o.state, o.on);
    let i = 0; for (; i < axes.length; i++) { idx[i]!++; if (idx[i]! < Math.max(1, axes[i]!.values.length)) break; idx[i] = 0; }
    if (i === axes.length) return;
  }
}
const everHolds = (c: Condition, e: EngineId) => { for (const desk of combos([c], e)) if (holds(ix, c, desk)) return true; return false; };

/* ================================================================ 4. parameters, on every engine */

const declared = (e: EngineId, key: string) => FIX.extensions[e]?.find(x => x.key === key);
const sameNum = (a: number, b: number) => Math.abs(a - b) < 1e-9;
const SAY_RE = new RegExp(SLOT.source, 'g');
function checkText(text: string | undefined, where: string, own?: ParamDef): void {
  if (text === undefined) return;
  if (!text.trim()) err(where, 'is empty: give it words, or leave the field out');
  for (const m of text.matchAll(/\{([a-z]+(?::[^}]*)?)\}/gi)) {
    const slot = m[1]!;
    if (['v', 's', 'value', 'shown', 'file', 'from', 'to', 'n', 'section', 'title', 'seed', 'noun'].includes(slot)) continue;
    if (!new RegExp(`^(?:${SAY_RE.source.slice(3, -3)})$`).test(slot) && !/^say:\w+$/.test(slot)) { err(where, `names {${slot}}, which is not a slot words may use (docs/05, "Words that name values")`); continue; }
    const [kind, rest = ''] = slot.split(':') as [string, string?];
    if ((kind === 'min' || kind === 'max' || kind === 'default') && !own) err(where, `uses {${kind}}, which only a parameter's own words may`);
    if (kind === 'value' && !ix.params.has(rest)) err(where, `names {value:${rest}}, and there is no parameter "${rest}"`);
    if (kind === 'count' && !ix.states.has(rest) && !(rest in DESK_FACTS)) err(where, `names {count:${rest}}: no state or desk fact has that name`);
    if (kind === 'file' && !ix.states.has(rest.split('|')[0]!.split('.')[0]!)) err(where, `names {file:${rest}}: no state has that name`);
    if (kind === 'say' && !(rest in DESK_FACTS)) err(where, `names {say:${rest}}, which is not a desk fact`);
  }
}

/* the `when` graph, for cycles */
const whenDeps = new Map<string, Set<string>>();

for (const { p, d } of ix.params.values()) {
  const where = `${fileOf(d)} › ${p.key}`, engs = enginesOf(p, d);
  engs.forEach(e => { if (!d.engines.includes(e)) err(where, `is on ${e}, which its ${d.kind} is not: add ${e} to the ${d.kind}'s engines, or take it off this one`); });
  const lines = p.per || [], engineLines = lines.filter(isEngineLine);
  if (engineLines.length && !p.engines) err(where, `has engine lines in \`per\`, so list its engines in \`engines\` beside them: then adding an engine to the ${d.kind} cannot hand this one another engine's names, ranges and defaults by accident`);
  const seen = new Set<string>();
  engineLines.forEach((l, i) => l.engines.forEach(e => {
    if (seen.has(e)) err(where, `per line ${i + 1} names ${e}, which an earlier line already does`); seen.add(e);
    if (!engs.includes(e)) err(where, `per line ${i + 1} names ${e}, which it is not on`);
  }));
  if (p.staysBelow) {
    const q = ix.params.get(p.staysBelow);
    if (!q) err(where, `stays below "${p.staysBelow}", which no definition has`);
    else if (q.p.key === p.key) err(where, 'stays below itself');
    else if (q.p.type !== 'number' || p.type !== 'number') err(where, 'staysBelow joins two numbers');
    else for (const e of engs) { const a = resolve(p, e).range, b = resolve(q.p, e).range; if (J(a) !== J(b)) err(where, `stays below "${q.p.key}", whose range on ${e} is ${J(b)} where this one's is ${J(a)}: the two ends of a range share it`); }
  }
  if (p.beside) { const b = ix.params.get(p.beside); if (!b || b.d !== d || b.p.type !== 'boolean') err(where, `stands beside "${p.beside}", which is not one of this ${d.kind}'s switches`); }
  if (p.beside && d.kind !== 'pedal') err(where, '`beside` is for a pedal\'s laid-out controls');
  const owner = p.owner || d.owner, stored = p.stored || d.stored;
  if (owner === 'engine' && stored === 'content') err(where, 'is kept per engine and saved as song content: a draft holds one song\'s content for any engine, so content is the song\'s (owner song), or the setting is a revision\'s (stored settings)');
  const ranges = new Set(engs.map(e => J(resolve(p, e).range?.slice(0, 2))));
  if (owner === 'song' && p.type === 'number' && ranges.size > 1 && !p.held) err(where, 'travels with the song, and its range differs per engine: give it `held` (the note and toast for when an engine cannot take the value), or keep it per engine (owner engine)');
  if (p.held && !(owner === 'song' && p.type === 'number' && ranges.size > 1)) err(where, '`held` is for a song number whose range differs per engine, and this one\'s does not');
  if (p.held) { checkText(p.held.note, `${where} held.note`); checkText(p.held.toast, `${where} held.toast`); if (!p.held.note.includes('{value}') || !p.held.note.includes('{engine}')) err(`${where} held.note`, 'should say the held {value} and the {engine} it returns on'); }
  for (const [i, l] of lines.entries()) if (!isEngineLine(l)) checkCond(l.when, { where: `${where} per line ${i + 1}`, engines: engs, own: d });

  for (const e of engs) {
    const f = resolve(p, e), at = `${where} [${e}]`, dv = f.default;
    checkText(f.label, `${at} label`, p); checkText(f.tip, `${at} tip`, p); checkText(f.short, `${at} short`, p); checkText(f.undoName, `${at} undoName`, p);
    if (f.auto && f.auto.says !== undefined) checkText(f.auto.says, `${at} auto.says`, p);
    if (f.when) { checkCond(f.when, { where: `${at} when`, engines: [e], own: d }); if (!everHolds(f.when, e)) err(`${at} when`, 'can never hold, so the parameter never applies on this engine: take the engine off it, or fix the condition'); whenDeps.set(p.key, new Set([...(whenDeps.get(p.key) || []), ...names(f.when)])); }
    if (p.type === 'number') {
      if (!f.range) { err(at, 'is a number without a range: give it [min, max, step]'); continue; }
      const [lo, hi, step] = f.range;
      if (!(lo <= hi)) err(at, `range ${lo}..${hi} is empty`);
      if (!(step > 0)) err(at, `step ${step} is not positive`);
      if (typeof dv !== 'number') err(at, `default ${J(dv)} is not a number`);
      else if (dv < lo || dv > hi) err(at, `default ${dv} is outside its range ${lo}..${hi}`);
      else if (Math.abs((dv - lo) / step - Math.round((dv - lo) / step)) > 1e-6) {
        const x = wiresOf(f.wire).map(w => toOf(w)).filter(t => t.startsWith('ext:')).map(t => declared(e, t.slice(4))).find(Boolean);
        if (x && x.default === dv && x.step === step) note(at, `default ${dv} is off the step grid (${lo} + n × ${step}), as the server declares it`);
        else err(at, `default ${dv} is off the step grid (${lo} + n × ${step}): a keyboard step from it lands off the grid`);
      }
      if (f.options) err(at, 'is a number with options: a number has a range; a choice has options');
      if (f.format) {
        const fm = f.format;
        if (fm.decimals !== undefined && (!Number.isInteger(fm.decimals) || fm.decimals < 0 || fm.decimals > 6)) err(at, `format.decimals ${fm.decimals} is not a whole number from 0 to 6`);
        if (fm.template !== undefined && !fm.template.includes('{v}')) err(at, 'format.template does not say the value: put {v} in it');
        if (fm.template?.includes('{s}') && !fm.perSecond) err(at, 'format.template says {s} without format.perSecond');
        if (fm.perSecond !== undefined && !(fm.perSecond > 0)) err(at, 'format.perSecond must be above 0');
        for (const w of fm.words || []) { if (typeof w.value !== 'number' || w.value < lo || w.value > hi) err(at, `format word for ${J(w.value)} names a value it cannot hold (${lo}..${hi})`); checkText(w.text, `${at} format word`); }
      }
    } else if (p.type === 'boolean') {
      if (typeof dv !== 'boolean') err(at, `default ${J(dv)} is not true or false`);
      if (f.range || f.options) err(at, 'a switch has no range or options');
      for (const w of f.format?.words || []) if (typeof w.value !== 'boolean') err(at, `format word for ${J(w.value)} is not true or false`);
      if (f.format && (f.format.template || f.format.decimals !== undefined || f.format.perSecond)) err(at, 'a switch\'s format has only words');
    } else if (p.type === 'choice') {
      if (!f.options) { err(at, 'is a choice without options'); continue; }
      if (f.range) err(at, 'is a choice with a range');
      if (Array.isArray(f.options)) {
        if (!f.options.length) err(at, 'has an empty option list');
        const vs = f.options.map((o: Option) => o.value), ls = f.options.map((o: Option) => o.label);
        if (new Set(vs).size !== vs.length) err(at, 'has two options with one value');
        if (new Set(ls).size !== ls.length) err(at, 'has two options with one label, which no one can tell apart');
        f.options.forEach((o, i) => { checkText(o.label, `${at} option ${i + 1} label`); (o.was || []).forEach(w => { if (vs.includes(w as string)) err(at, `option "${o.value}" was "${w}", which is another option's value`); }); });
        if (!vs.includes(dv as string | number)) err(at, `default ${J(dv)} is not one of its options`);
      } else {
        const o = f.options, allowed = allowedValues(p, e);
        if (o.from === 'models' && (!o.bucket || !(FIX.buckets[e] ?? []).includes(o.bucket))) err(at, `options from models need a bucket ${e} has (${(FIX.buckets[e] ?? []).join(', ')})`);
        if (o.only) { const list = FIX.plugins[(OPTION_SOURCES[o.from].fixture ?? '.').split('.')[1]!]; if (!list) err(at, '`only` filters a list the fixture does not know'); else o.only.forEach(v => { if (!list.includes(v)) err(at, `only "${v}", which ${o.from} does not list`); }); }
        if (typeof dv !== 'string' && typeof dv !== 'number') err(at, 'the default of a server-listed choice is a string or number');
        else if (allowed && !allowed.includes(dv)) err(at, `default ${J(dv)} is not in ${o.from} (tools/fixtures/real-app.json) nor in its include`);
        else if (!allowed && dv === '' && !(o.include || []).some(x => x.value === '')) err(at, 'default "" is not among the options it always has (include)');
      }
    } else if (p.type === 'text') {
      if (typeof dv !== 'string') err(at, `default ${J(dv)} is not text`);
      if (f.options || f.range) err(at, 'text has no options or range');
      if (f.format) err(at, 'text has no format');
    }
    if (f.auto && p.type !== 'number') err(at, `auto is for numbers; a ${p.type} says "Auto" with an option`);
    if (f.example !== undefined && p.type !== 'text') err(at, 'an example is for text');
    if (f.check && p.type !== 'text') err(at, 'a check is for text');
    if (f.check) checkText(f.check.message, `${at} check message`);
    if (f.placeholder !== undefined) checkText(f.placeholder, `${at} placeholder`);
    checkWire(f.wire, p, d, e, at);
  }
}
for (const [k] of whenDeps) {
  const path: string[] = [];
  const visit = (x: string): boolean => { if (path.includes(x)) return true; path.push(x); for (const y of whenDeps.get(x) || []) if (visit(y)) return true; path.pop(); return false; };
  if (visit(k) && path[0] === k) err(`${keyOwners.get(k)} › ${k}`, `applies only when ${path.slice(1).join(', then ')} does, which applies only when it does: a circle, so neither can ever be shown`);
}

/* ---- wires ---- */
/** A check the wire waives by name (`differs`), which it must explain in its note. A note alone waives nothing. */
function waived(w: WireKey | WireMap, what: NonNullable<WireMap['differs']>[number]): boolean { return typeof w === 'object' && (w.differs || []).includes(what); }
function checkWire(w: Wire, p: ParamDef, d: Definition, e: EngineId, at: string): void {
  if (Array.isArray(w)) { if (!w.length) err(at, 'wire is an empty list'); w.forEach(x => checkWire(x, p, d, e, at)); return; }
  if (typeof w === 'object' && 'none' in w) { if (!w.none.trim()) err(at, 'wire says nothing about the real application: say what the real app does instead'); return; }
  const m: WireMap = typeof w === 'string' ? { to: w } : w;
  const ch = m.to.slice(0, m.to.indexOf(':')), name = m.to.slice(m.to.indexOf(':') + 1), root = name.split(/[.[]/)[0]!;
  if (!CHANNELS.includes(ch) || !name) { err(at, `wire "${m.to}" has no channel: write ${CHANNELS.map(c => c + ':').join(', ')} before the name`); return; }
  if (ch === 'intent' && !(name.startsWith('compose.') && FIX.intent.compose.includes(name.slice(8))) && name !== 'captionSource') err(at, `wire ${m.to} is not a part of the Create intent: intent:compose.<${FIX.intent.compose.join('|')}> or intent:captionSource`);
  const f = resolve(p, e);
  if (ch === 'param' && !(root in FIX.generationDefaults)) err(at, `wire param:${root} is not a GENERATION_DEFAULTS field${suggest(root, Object.keys(FIX.generationDefaults))}`);
  else if (ch === 'param' && !(FIX.fields[e] ?? []).includes(root) && !waived(m, 'unread')) err(at, `wire param:${root}: nothing the ${e} engine runs reads that field, so the control would change nothing there: give ${e} its own field in a per line, take ${e} off the parameter, or add 'unread' to the wire's differs and say why in its note`);
  if (ch === 'content' && !FIX.createContentFields.includes(root)) err(at, `wire content:${root} is not a CreateContent field${suggest(root, FIX.createContentFields)}`);
  if (ch === 'model') { const [eng, bucket] = name.split('.') as [string, string]; if (eng !== e) err(at, `wire ${m.to} names engine ${eng} on ${e}`); else if (!(FIX.buckets[e] ?? []).includes(bucket)) err(at, `wire ${m.to}: ${e} has no model bucket "${bucket}" (${(FIX.buckets[e] ?? []).join(', ')})`); }
  if (ch === 'ext') {
    const x = declared(e, name);
    if (!x) {
      const others = ENGINE_IDS.filter(o => declared(o, name));
      if ((FIX.reads[e] ?? []).includes(name)) { if (!m.note) err(at, `wire ext:${name} is read by the ${e} backend but not declared as an extension: say so in the wire's note`); }
      else err(at, `wire ext:${name} is neither declared nor read by the ${e} backend${others.length ? ` (${others.join(', ')} declares it: is this another engine's name?)` : suggest(name, (FIX.extensions[e] ?? []).map(y => y.key))}`);
    } else {
      const kind = x.type === 'toggle' ? 'boolean' : x.type === 'select' ? 'choice' : x.type === 'text' ? 'text' : 'number';
      const sent = m.via === 'text' ? 'text' : m.map ? null : p.type;
      if (sent && sent !== kind && !(sent === 'choice' && kind === 'text')) err(at, `wire ext:${name} is a ${x.type} on the server, and this is a ${p.type}${m.via ? ` sent through ${m.via}` : ''}`);
      if (p.type === 'number' && !m.via && f.range && !waived(m, 'range')) {
        const k = m.divide ? 1 / m.divide : m.scale ?? 1;
        const mine = [f.range[0] * k, f.range[1] * k, f.range[2] * k];
        const real = [x.min, x.max, x.step];
        if (real.some((r, i) => typeof r === 'number' && !sameNum(r, mine[i]!))) err(at, `range ${J(f.range)} differs from the server's ${J(real)} for ext:${name}: follow it, or add 'range' to the wire's differs and say why in its note`);
      }
      if (p.type === 'choice' && Array.isArray(f.options) && Array.isArray(x.options) && !waived(m, 'options')) {
        const real = (x.options as { value: string }[]).map(o => o.value), mine = f.options.map(o => o.value);
        const missing = mine.filter(v => !real.includes(v as string));
        if (missing.length) err(at, `options ${J(missing)} are not among the server's for ext:${name} (${J(real)})`);
      }
    }
  }
  if (ch === 'param' && e === 'ace' && p.type === 'number' && !m.via && f.range && !waived(m, 'range') && !name.includes('[') && !name.includes('.')) {
    const sl = FIX.sliders[name];
    if (sl) {
      const k = m.divide ? 1 / m.divide : m.scale ?? 1, mine = f.range.map(v => +(v * k).toFixed(6));
      if (!sl.some(r => r[0] === mine[0] && r[1] === mine[1] && (r[2] === null || r[2] === mine[2]))) err(at, `range ${J(f.range)}${k !== 1 ? `, sent as ${J(mine)},` : ''} differs from the existing UI's slider for ${name} (${sl.map(r => J(r)).join(' or ')}): follow it, or add 'range' to the wire's differs and say why in its note`);
    }
  }
  /* the values a field takes, where the real app spells them (tools/fixtures/real-app.json values) */
  const takes = (ch === 'param' || ch === 'ext') && !name.includes('.') && !name.includes('[') ? FIX.values[name] : undefined;
  if (takes && p.type === 'choice' && Array.isArray(f.options) && !m.via && !waived(m, 'options')) {
    for (const o of f.options) {
      const sent = m.map ? (String(o.value) in m.map ? m.map[String(o.value)] : undefined) : o.value;
      if (sent !== undefined && !takes.includes(String(sent))) err(at, `option ${J(o.value)} sends ${J(sent)} to ${m.to}, which takes ${takes.map(x => J(x)).join(', ')}`);
    }
  }
  /* the type of a param: field is the type of its real default */
  if (ch === 'param' && !m.via && !m.map && !name.includes('.') && !name.includes('[') && root in FIX.generationDefaults) {
    const real = FIX.generationDefaults[root], want = typeof real === 'boolean' ? 'boolean' : typeof real === 'number' ? 'number' : typeof real === 'string' ? 'string' : null;
    const mine = p.type === 'boolean' ? 'boolean' : p.type === 'number' ? 'number' : p.type === 'text' ? 'string' : typeof f.default;
    if (want && mine !== want && !('auto' in m)) err(at, `sends a ${p.type} to ${m.to}, whose real value is a ${want} (${J(real)})`);
  }
  if (m.via && (m.map || m.invert || m.scale !== undefined || m.divide !== undefined)) err(at, 'wire.via does the whole change: map, invert, scale and divide beside it are not applied');
  if (m.map && Array.isArray(f.options)) {
    /* an option the map leaves out sends nothing through it, which is right only when another writer of the field
       applies while that option is chosen (the normalizer's Custom knob) */
    const writers = [...ix.params.values()].filter(({ p: q }) => q.key !== p.key && has(q.key, e) && wiresOf(resolve(q, e).wire).some(w => toOf(w) === m.to));
    for (const o of f.options.filter(x => !(String(x.value) in m.map!))) {
      const desk = deskWith(e, { [p.key]: o.value }, {}, {}, new Set(itemIds));
      if (!writers.some(({ p: q }) => applies(ix, q.key, desk))) err(at, `wire.map has no value for option ${J(o.value)}, and nothing else writes ${m.to} while it is chosen, so choosing it sends nothing: map it, or give the field another writer for it`);
    }
  }
  if (m.invert && p.type !== 'boolean') err(at, 'wire.invert is for a switch');
  if ((m.scale !== undefined || m.divide !== undefined) && p.type !== 'number') err(at, 'wire.scale and wire.divide are for numbers');
  if (m.scale !== undefined && m.divide !== undefined) err(at, 'wire has both scale and divide');
  if (m.map) {
    if (p.type !== 'choice' && p.type !== 'boolean') err(at, 'wire.map is for a choice or a switch');
    const vals = Array.isArray(f.options) ? f.options.map(o => String(o.value)) : p.type === 'boolean' ? ['true', 'false'] : null;
    if (vals) Object.keys(m.map).forEach(k => { if (!vals.includes(k)) err(at, `wire.map names "${k}", which is not one of its values`); });
  }
  if (m.via) {
    const t = TRANSFORMS[m.via] as Transform | undefined;
    if (!t) err(at, `wire.via names "${m.via}", which transforms.ts does not have`);
    else if (t.packs && !t.packs.includes(p.key)) err(at, `wire.via "${m.via}" packs ${t.packs.join(', ')}, and not this parameter: add it to the transform's packs, or use another`);
  }
  const hasAuto = !!f.auto;
  if (hasAuto && !('auto' in m)) err(at, 'is automatic, and its wire does not say what auto sends: give the wire `auto` (null sends nothing, so the engine\'s own default applies)');
  if (!hasAuto && 'auto' in m) err(at, 'wire.auto says what auto sends, and the parameter has no auto');
  if (m.when) checkCond(m.when, { where: `${at} wire.when`, engines: [e], own: d });
  if ('auto' in m && m.auto !== null && ch === 'param' && root in FIX.generationDefaults && typeof m.auto !== typeof FIX.generationDefaults[root]) err(at, `wire.auto sends ${J(m.auto)} to ${m.to}, whose real value is a ${typeof FIX.generationDefaults[root]}`);
  if ('auto' in m && m.auto !== null && ch === 'content' && root in FIX.createContent && typeof m.auto !== typeof FIX.createContent[root]) err(at, `wire.auto sends ${J(m.auto)} to ${m.to}, whose real value is a ${typeof FIX.createContent[root]}`);
  if (f.when && !m.via && !m.when && !waived(m, 'default') && !name.includes('.') && !name.includes('[')) {
    /* the default a condition hides is still the one sent when it applies: it must be the real one */
    const real = ch === 'param' ? FIX.generationDefaults[root] : ch === 'ext' ? declared(e, name)?.default : undefined;
    let sent: Json | undefined = f.default;
    if (m.map) sent = String(f.default) in m.map ? m.map[String(f.default)] : undefined;
    if (m.invert && typeof sent === 'boolean') sent = !sent;
    if (typeof sent === 'number') { if (m.scale) sent = +(sent * m.scale).toFixed(10); if (m.divide) sent = +(sent / m.divide).toFixed(10); }
    if (real !== undefined && sent !== undefined && !(typeof real === 'string' && real.startsWith('<')) && J(real) !== J(sent)) err(at, `its default sends ${J(sent)} to ${m.to} whenever it applies, where the real app's default is ${J(real)}: use the real default, or add 'default' to the wire's differs and say why in its note`);
  }
  if ('else' in m && !m.when) err(at, 'wire.else without wire.when');
}
function suggest(name: string, list: string[]): string {
  const near = list.filter(x => x.toLowerCase().startsWith(name.toLowerCase().slice(0, Math.max(4, name.length - 3))) || name.toLowerCase().startsWith(x.toLowerCase()));
  return near.length ? `; did you mean ${near.slice(0, 3).join(' or ')}?` : '';
}

/* ---- the packing transforms are consistent ---- */
for (const [name, t] of Object.entries(TRANSFORMS) as [string, Transform][]) {
  if (!t.packs) continue;
  for (const k of t.packs) {
    if (!ix.params.has(k) && !ix.states.has(k)) { err(`transforms.ts › ${name}`, `packs "${k}", which no definition has`); continue; }
    const hit = ix.params.get(k);
    const wires = hit ? enginesOf(hit.p, hit.d).flatMap(e => wiresOf(resolve(hit.p, e).wire)) : stateWires(ix.states.get(k)!.s).flatMap(x => wiresOf(x.wire));
    if (!wires.some(w => typeof w === 'object' && w.via === name)) err(`transforms.ts › ${name}`, `packs "${k}", whose wire does not name it`);
  }
}
function stateWires(s: NonNullable<Definition['state']>[number]): { engines: readonly EngineId[]; wire: Wire }[] {
  const w = s.wire;
  return Array.isArray(w) && w.length && typeof w[0] === 'object' && 'engines' in (w[0] as object) ? (w as { engines: EngineId[]; wire: Wire }[]) : [{ engines: ENGINE_IDS, wire: w as Wire }];
}

/* ================================================================ 5. state */

/** A definition's named steps: words, and the parameters each may also change, which must exist and be on its engines. */
function checkSteps(steps: { [k: string]: Step } | undefined, where: string, d: Definition): void {
  for (const [step, s] of Object.entries(steps || {})) {
    checkText(stepSays(s), `${where} steps.${step}`);
    for (const k of stepChanges(s)) { const hit = ix.params.get(k); if (!hit) err(`${where} steps.${step}`, `changes "${k}", which no definition has`); else if (!hit.d.engines.some(e => d.engines.includes(e))) err(`${where} steps.${step}`, `changes "${k}", which none of ${d.engines.join(', ')} has`); }
  }
}

for (const { s, d } of ix.states.values()) {
  const where = `${fileOf(d)} › ${s.key}`;
  checkText(s.noun, `${where} noun`);
  checkSteps(s.steps, where, d);
  if (s.owner === 'engine' && s.stored === 'content') err(where, 'is kept per engine and saved as song content; one of the two is wrong');
  for (const { engines, wire } of stateWires(s)) for (const e of engines.filter(x => d.engines.includes(x))) {
    for (const w of wiresOf(wire)) {
      const to = toOf(w), ch = to.split(':')[0]!, root = to.slice(to.indexOf(':') + 1).split(/[.[]/)[0]!;
      if (!CHANNELS.includes(ch)) err(where, `wire "${to}" has no channel`);
      if (ch === 'param' && !(root in FIX.generationDefaults)) err(where, `wire ${to} is not a GENERATION_DEFAULTS field`);
      if (ch === 'content' && !FIX.createContentFields.includes(root)) err(where, `wire ${to} is not a CreateContent field`);
      if (ch === 'ext' && !declared(e, root) && !(FIX.reads[e] ?? []).includes(root)) err(where, `wire ${to} is neither declared nor read by the ${e} backend`);
      if (typeof w === 'object' && w.via && !(w.via in TRANSFORMS)) err(where, `wire.via names "${w.via}", which transforms.ts does not have`);
    }
  }
}

/* ================================================================ 6. faces */

const partUse = new Map<string, Set<string>>();
const stateUse = new Set<string>();
/** For every definition, engine and tier: the parameters some shown control edits. */
const coverage = new Map<string, Set<string>>();
const cover = (d: Definition, e: EngineId, k: string) => { const id = `${d.id}|${e}`; if (!coverage.has(id)) coverage.set(id, new Set()); coverage.get(id)!.add(k); };
const visibleOn = new Map<string, Set<EngineId>>();

interface NodeCx { d: Definition; where: string; engines: readonly EngineId[]; tiers: readonly Tier[] }
function checkNode(n: Node, cx: NodeCx): void {
  const spec = (PARTS as Record<string, PartSpec>)[n.part];
  const where = cx.where;
  if (!spec) { err(where, `names part "${n.part}", which is not in the inventory (parts.ts)`); return; }
  if (!partUse.has(n.part)) partUse.set(n.part, new Set()); partUse.get(n.part)!.add(cx.d.id);
  let engines = cx.engines, tiers = cx.tiers;
  if (n.engines) { n.engines.forEach(e => { if (!ENGINE_IDS.includes(e)) err(where, `names engine "${e}", which engines.ts does not list`); else if (!cx.engines.includes(e)) err(where, `is limited to ${e}, which its ${cx.d.kind} or parent node is not on`); }); engines = n.engines.filter(e => cx.engines.includes(e)); }
  if (n.tiers) {
    if (cx.d.kind === 'module' && cx.d.labUnder) err(where, 'a Lab-only module\'s nodes take no tiers: the module shows at Lab alone');
    if (!n.tiers.length) err(where, 'tiers is empty, so the node is never shown: leave it out to show it at every tier');
    const outside = n.tiers.filter(t => !cx.tiers.includes(t));
    if (outside.length) err(where, `is tiered ${outside.join(', ')}, where its parent is never shown, so it never is`);
    tiers = n.tiers.filter(t => cx.tiers.includes(t));
  }
  if (!engines.length || !tiers.length) return;
  const lines = (n.per || []) as NodeOverride[];
  lines.forEach((l, i) => {
    (l.engines || []).forEach(e => { if (!engines.includes(e)) err(`${where} per line ${i + 1}`, `names ${e}, where the node is not shown`); });
    (l.tiers || []).forEach(t => { if (!tiers.includes(t)) err(`${where} per line ${i + 1}`, `names ${t}, where the node is not shown`); });
    if (l.when) checkCond(l.when, { where: `${where} per line ${i + 1}`, engines });
  });
  if (n.when) checkCond(n.when, { where: `${where} when`, engines });
  if (n.dim) checkCond(n.dim.when, { where: `${where} dim`, engines });
  /* every state the node can be in: each engine and tier, with the lines that can hold there */
  const variants: { e: EngineId; t: Tier; n: Node }[] = [];
  for (const e of engines) for (const t of tiers) {
    const desk = deskWith(e, {}, {}, {}, new Set(itemIds)); desk.tier = t;
    const { when: _w, per: _p, ...bare } = n;
    let acc: Node[] = [nodeAt(ix, bare as unknown as Node, desk)!];
    for (const l of lines) {
      if (l.engines && !l.engines.includes(e)) continue; if (l.tiers && !l.tiers.includes(t)) continue;
      const { engines: _e, tiers: _t, when, ...fields } = l;
      const applied = acc.map(x => ({ ...x, ...fields }) as Node);
      acc = when ? [...acc, ...applied] : applied;
    }
    acc.forEach(v => variants.push({ e, t, n: v }));
    if (!n.when || everHolds(n.when, e)) (visibleOn.get(cx.d.id) ?? visibleOn.set(cx.d.id, new Set()).get(cx.d.id)!).add(e);
  }
  if (n.when && !engines.some(e => everHolds(n.when!, e))) err(`${where} when`, 'can never hold on the engines the node is on, so it is never shown');
  /* a node shown by its own condition where none of the parameters it and its children edit applies draws controls for
     nothing (the Power group shown on Cosine): its condition must imply one of theirs */
  if (n.when) {
    const subtree = (x: Node): string[] => [...editsOf(x), ...(x.children || []).flatMap(subtree)];
    const keys = [...new Set(subtree(n))].filter(k => ix.params.has(k));
    for (const e of engines) {
      const here = keys.filter(k => has(k, e));
      if (!here.length) continue;
      const whens = here.map(k => resolve(ix.params.get(k)!.p, e).when).filter((c): c is Condition => !!c);
      if (whens.length < here.length) continue;
      for (const desk of combos([n.when, ...whens], e)) {
        if (holds(ix, n.when, desk) && !here.some(k => applies(ix, k, desk))) { err(`${where} when`, `is shown on ${e} where none of ${here.join(', ')} applies (${J(desk.vals)}): make its condition imply theirs`); break; }
      }
    }
  }
  const dims = variants.filter(v => v.n.dim);
  if (dims.length && !dims.some(v => everHolds(v.n.dim!.when, v.e))) err(`${where} dim`, 'can never hold on any engine the node is on, so it never stands back: drop it, or fix the condition');
  const seenVar = new Set<string>(), boundSomewhere = new Map<string, boolean>();
  for (const { e, t, n: v } of variants) {
    for (const k of [...editsOf(v), ...(v.reads || [])]) { const hit = ix.params.get(k); boundSomewhere.set(k, (boundSomewhere.get(k) ?? false) || !!(hit && hit.d.engines.includes(e) && enginesOf(hit.p, hit.d).includes(e))); }
    const sig = J([e, v.param, v.params, v.reads, v.state, v.choices, v.opts, v.actions, v.label, v.text, v.dim]);
    if (seenVar.has(sig)) continue; seenVar.add(sig);
    checkBinding(v, spec, cx.d, e, t, where);
  }
  for (const [k, ok] of boundSomewhere) if (!ok && ix.params.has(k) && !(n.part === 'capChips')) err(where, `binds "${k}", which none of the engines it is shown on (${engines.join(', ')}) has: drop it, or show the node where the parameter is`);
  if (n.children?.length && spec.kind !== 'container' && spec.kind !== 'bespoke') err(where, `part "${n.part}" is a ${spec.kind} and holds no children`);
  if (spec.kind === 'container' && !n.children?.length) err(where, `container "${n.part}" holds nothing`);
  (n.children || []).forEach((c, i) => checkNode(c, { ...cx, where: `${cx.where} > ${c.part}#${i}`, engines, tiers }));
}

function checkBinding(n: Node, spec: PartSpec, d: Definition, e: EngineId, t: Tier, where: string): void {
  const at = `${where} [${e}, ${t}]`;
  const edits = editsOf(n);
  if (edits.length && !spec.edits) err(at, `part "${n.part}" is a ${spec.kind} and edits nothing; list what it shows in \`reads\``);
  if (n.reads && !spec.reads) err(at, `part "${n.part}" takes no \`reads\``);
  if (n.state && !spec.state) err(at, `part "${n.part}" holds no state`);
  if ((spec.kind === 'control') && !edits.length && !n.state?.length) err(at, `control "${n.part}" is bound to nothing: give it the param it edits`);
  if (spec.edits?.includes('pair') && edits.length !== 2) err(at, `part "${n.part}" edits two numbers: give it two keys in \`params\``);
  if (n.param && n.params?.length && spec.kind !== 'bespoke') err(at, 'has both param and params: a control edits one, an XY pad two');
  for (const k of edits) {
    const hit = ix.params.get(k);
    if (!hit) { err(at, `edits "${k}", which no definition has${ix.former.has(k) ? ` (it is now "${ix.former.get(k)}")` : ''}`); continue; }
    if (!(hit.d.engines.includes(e) && enginesOf(hit.p, hit.d).includes(e))) continue; /* not drawn on this engine */
    const type = hit.p.type;
    const ok = spec.edits?.includes('pair') ? type === 'number' : spec.edits?.includes(type);
    if (!ok) err(at, `part "${n.part}" cannot edit a ${type} ("${k}"); it edits ${spec.edits?.join(', ') ?? 'nothing'}`);
    if (hit.d === d) cover(d, e, k);
  }
  for (const k of n.reads || []) if (!ix.params.has(k) && !ix.states.has(k) && !(k in DESK_FACTS)) err(at, `reads "${k}", which is no parameter, state or desk fact`);
  if (Array.isArray(spec.reads)) {
    const sig = spec.reads as readonly string[], got = n.reads || [];
    const kind = (k: string) => (ix.params.has(k) ? ix.params.get(k)!.p.type : ix.states.has(k) ? 'state' : 'fact');
    if (got.length < sig.length || sig.some((t, i) => kind(got[i]!) !== t)) err(at, `part "${n.part}" draws from ${sig.join(', ')}, in that order, and its node reads ${got.length ? got.map(k => `${k} (${kind(k)})`).join(', ') : 'nothing'}`);
  }
  /* the named steps a bespoke part takes are declared by its state or its parameter */
  if (spec.steps) {
    const declared = new Set<string>();
    for (const k of n.state || []) { const s = ix.states.get(k)?.s; if (!s) continue; Object.keys(s.steps || {}).forEach(x => declared.add(x)); if (s.kind === 'file') ['load', 'clear'].forEach(x => declared.add(x)); }
    for (const k of editsOf(n)) Object.keys(ix.params.get(k)?.p.steps || {}).forEach(x => declared.add(x));
    const missing = spec.steps.filter(x => !declared.has(x));
    if (missing.length) err(at, `part "${n.part}" takes the steps ${missing.join(', ')}, which neither its state nor its parameters name in \`steps\``);
  }
  for (const k of n.state || []) {
    const hit = ix.states.get(k);
    if (!hit) { err(at, `names state "${k}", which no definition declares`); continue; }
    stateUse.add(k);
    if (!spec.state?.includes(hit.s.kind)) err(at, `part "${n.part}" holds ${spec.state?.join(' or ') ?? 'no'} state, and "${k}" is a ${hit.s.kind}`);
    const lends = n.part === 'dropZone' && n.opts && 'from' in n.opts;
    if (hit.d !== d && !lends) err(at, `names "${k}", the state of ${fileOf(hit.d)}: a part changes only its own definition's state (to show another's file, a drop zone takes opts.from)`);
  }
  if (spec.state && !n.state?.length && spec.kind !== 'bespoke') err(at, `part "${n.part}" needs \`state\`: the ${spec.state.join(' or ')} it holds`);
  checkText(n.label, `${at} label`); checkText(n.tip, `${at} tip`); checkText(n.text, `${at} text`);
  if (n.dim) checkText(n.dim.says, `${at} dim says`);
  for (const t of [n.label, n.tip, n.text, n.dim?.says]) for (const m of (t || '').matchAll(/\{value:([\w.-]+)\}/g)) if (ix.params.has(m[1]!) && !has(m[1]!, e)) err(at, `says {value:${m[1]}}, which ${e} does not have, so the sentence has a hole there: show the node only where it is, or say it another way`);
  /* a control edits its own definition's values; only a bespoke part may edit another's (B24) */
  if (n.part === 'choiceRow') for (const k of editsOf(n)) { const hit = ix.params.get(k); if (hit && hit.d !== d) err(at, `a Quick row sets "${k}", which is ${fileOf(hit.d)}'s: a tile's plain choices set its own values, so its Also set line and the row agree`); }
  /* a tier action rings what that tier shows (B15) */
  for (const a of n.actions || []) if (a.do === 'tier') for (const k of a.ring || []) if (ix.params.has(k) && !tierEdits(ix.params.get(k)!.d, e, a.tier).has(k)) err(`${at} action "${a.label}"`, `goes to ${a.tier} to ring "${k}", which ${a.tier} does not show on ${e}`);
  (n.actions || []).forEach((a, i) => checkAction(a, `${at} action ${i + 1}`, e, n));
  const o = (n.opts || {}) as Record<string, Json>;
  if (n.part === 'dice' && typeof o.offWhen === 'string') { const s = ix.params.get(o.offWhen); if (!s || s.p.type !== 'boolean') err(at, `opts.offWhen names "${o.offWhen}", which is not a switch`); }
  if (n.part === 'powerChip' && n.param) {
    const f = resolve(ix.params.get(n.param)!.p, e);
    for (const k of ['off', 'on']) { const v = o[k]; if (typeof v === 'number' && f.range && (v < f.range[0] || v > f.range[1])) err(at, `opts.${k} ${v} is outside ${n.param}'s range ${f.range[0]}..${f.range[1]}`); }
    if (o.off === undefined || o.on === undefined) err(at, 'a power chip needs opts.off and opts.on');
  }
  if (n.part === 'termLine' && !n.children?.length) err(at, 'a terminal line holds its controls as children');
  if (n.part === 'choiceRow' && (!n.param || has(n.param, e))) checkChoices(n, at, e);
  else if (n.part !== 'choiceRow' && (n.choices || n.between)) err(at, 'choices and between are for a choice row');
  if (n.part === 'meter' && n.reads?.length) { const r = ix.params.get(n.reads[0]!); if (r && r.p.type !== 'number') err(at, 'a meter reads a number'); }
}

function has(key: string, e: EngineId): boolean { const hit = ix.params.get(key); return !!hit && hit.d.engines.includes(e) && enginesOf(hit.p, hit.d).includes(e); }
function checkChoices(n: Node, at: string, e: EngineId): void {
  const list = n.choices;
  if (!list || !list.length) { err(at, `a choice row has no choices on ${e}: give it choices (in a per line for ${e} when they differ)`); return; }
  const hit = n.param ? ix.params.get(n.param) : undefined;
  if (!hit) { if (!n.choices?.every(c => typeof c[1] === 'object')) err(at, 'a choice row edits its param: give it one, or make every choice set several values by key'); }
  const words = list.map(c => c[0]), values = list.map(c => J(c[1]));
  if (new Set(words).size !== words.length) err(at, `two choices share a word on ${e}`);
  if (new Set(values).size !== values.length) err(at, `two choices set the same value on ${e}`);
  if (n.between !== undefined && !n.between.includes('{v}')) err(at, 'between does not say the value: put {v} in it');
  const keys = new Set<string>();
  for (const [word, v] of list) {
    checkText(word, `${at} choice`);
    const sets: [string, Value][] = typeof v === 'object' ? Object.entries(v) : n.param ? [[n.param, v]] : [];
    sets.forEach(([k]) => keys.add(k));
    for (const [k, x] of sets) {
      const h = ix.params.get(k);
      if (!h) { err(at, `choice "${word}" sets "${k}", which no definition has`); continue; }
      if (!enginesOf(h.p, h.d).includes(e)) { err(at, `choice "${word}" sets "${k}", which ${e} does not have`); continue; }
      const f = resolve(h.p, e);
      if (h.p.type === 'number') {
        if (typeof x !== 'number' || !f.range || x < f.range[0] || x > f.range[1]) { err(at, `choice "${word}" = ${J(x)} is outside ${k}'s range on ${e}`); continue; }
        const [lo, , step] = f.range; if (Math.abs((x - lo) / step - Math.round((x - lo) / step)) > 1e-6) err(at, `choice "${word}" = ${x} is off ${k}'s step grid (${lo} + n × ${step}) on ${e}`);
      } else { const allowed = allowedValues(h.p, e); if (allowed && !allowed.includes(x as string)) err(at, `choice "${word}" = ${J(x)} is not an option of ${k} on ${e}`); if (typeof x !== (h.p.type === 'boolean' ? 'boolean' : typeof f.default)) err(at, `choice "${word}" sets ${k} to ${J(x)}, the wrong type`); }
    }
  }
  /* the row checks the choice equal to the values in force, so at defaults one of them must be checked */
  const atDefault = list.some(([, v]) => (typeof v === 'object' ? Object.entries(v) : [[n.param!, v] as [string, Value]]).every(([k, x]) => { const h = ix.params.get(k); return h && enginesOf(h.p, h.d).includes(e) && resolve(h.p, e).default === x; }));
  if (!atDefault) err(at, `no choice is the default on ${e}, so a fresh desk checks none: make one of them the default`);
  if (typeof list[0]![1] === 'object' && list.some(([, v]) => J(Object.keys(v as object).sort()) !== J([...keys].sort()))) err(at, 'every choice sets the same keys, so the row can say which one is in force');
}

function checkAction(a: Action, at: string, e: EngineId, n?: Node): void {
  checkText(a.label, `${at} label`); if ('tip' in a) checkText(a.tip, `${at} tip`);
  if ('param' in a) {
    const hit = ix.params.get(a.param);
    if (!hit) { err(at, `acts on "${a.param}", which no definition has`); return; }
    const f = resolve(hit.p, enginesOf(hit.p, hit.d).includes(e) ? e : enginesOf(hit.p, hit.d)[0]!);
    if (!enginesOf(hit.p, hit.d).includes(e)) err(at, `acts on "${a.param}", which ${e} does not have`);
    if (a.do === 'set') {
      const want = hit.p.type === 'choice' ? typeof f.default : hit.p.type === 'text' ? 'string' : hit.p.type;
      if (typeof a.value !== want) err(at, `sets ${a.param} to ${J(a.value)}, the wrong type`);
      const allowed = allowedValues(hit.p, e); if (hit.p.type === 'choice' && allowed && !allowed.includes(a.value as string)) err(at, `sets ${a.param} to ${J(a.value)}, which is not one of its options`);
      if (hit.p.type === 'number' && f.range && typeof a.value === 'number' && (a.value < f.range[0] || a.value > f.range[1])) err(at, `sets ${a.param} to ${a.value}, outside its range ${f.range[0]}..${f.range[1]}`);
    }
    if (a.do === 'auto' && !f.auto) err(at, `turns on auto for ${a.param}, which has none`);
    if (a.do === 'example' && f.example === undefined) err(at, `inserts ${a.param}'s example, and it has none: give the parameter \`example\``);
  }
  if (a.do === 'tier') a.ring?.forEach(k => { if (!ix.params.has(k) && !ix.states.has(k)) err(at, `rings "${k}", which is no parameter or state`); });
  if (a.do === 'add' && !itemIds.has(a.item)) err(at, `adds "${a.item}", which is no module, pedal or unit`);
  if (a.do === 'bespoke' && !(a.name in ACTIONS)) err(at, `runs "${a.name}", which parts.ts ACTIONS does not list`);
  if (a.do === 'reset' && n && !n.children?.some(function ed(c: Node): boolean { return editsOf(c).length > 0 || (c.children || []).some(ed); })) err(at, 'resets the parameters its node\'s children edit, and they edit none');
  if (a.do === 'tag' && !/^\[[^\]]+\]$/.test(a.tag)) err(at, `tag "${a.tag}" is not a section tag like [Verse]`);
}

function checkStrips(list: Strip[], d: Definition, what: string): void {
  list.forEach((s, i) => {
    const at = `${fileOf(d)} ${what} ${i + 1}`;
    checkCond(s.when, { where: at, engines: d.engines });
    checkText(s.says, `${at} says`);
    if (!d.engines.some(e => everHolds(s.when, e))) err(at, 'can never hold on the engines it is on, so it never shows: fix the condition, or drop it');
    (s.tiers || []).forEach(t => { if (!TIERS.includes(t)) err(at, `names tier "${t}"`); });
    if (s.tiers && !s.tiers.length) err(at, 'tiers is empty, so it never shows');
    (s.actions || []).forEach((a, j) => d.engines.filter(e => everHolds(s.when, e)).forEach(e => checkAction(a, `${at} action ${j + 1}`, e)));
  });
}

for (const d of DEFINITIONS) {
  const at = fileOf(d);
  checkText(d.name, `${at} name`); checkText(d.desc, `${at} desc`);
  if (!d.engines.length) err(at, 'is on no engine');
  d.engines.forEach(e => { if (!ENGINE_IDS.includes(e)) err(at, `names engine "${e}", which engines.ts does not list`); });
  const tiers: Tier[] = d.kind === 'module' && d.labUnder ? ['lab'] : TIERS;
  if (d.kind === 'module') {
    if (d.labUnder) {
      const parent = DEFINITIONS.find(x => x.kind === 'module' && x.id === d.labUnder);
      if (!parent) err(at, `is Lab-only under "${d.labUnder}", which is no module`);
      else if (parent.kind === 'module' && (parent.labUnder || parent.builtIn || parent.source)) err(at, `is Lab-only under "${d.labUnder}", which is not a desk module of its own (it is Lab-only, built in or a source)`);
      else d.engines.forEach(e => { if (!parent.engines.includes(e)) err(at, `is on ${e} under "${d.labUnder}", which ${e} does not have`); });
    }
    (d.per || []).forEach((l, i) => { l.engines.forEach(e => { if (!d.engines.includes(e)) err(`${at} per line ${i + 1}`, `names ${e}, which it is not on`); }); if (l.name) checkText(l.name, `${at} per line ${i + 1} name`); if (l.desc) checkText(l.desc, `${at} per line ${i + 1} desc`); });
    if (d.strip) checkStrips(d.strip, d, 'strip');
  }
  if (d.kind === 'pedal' && d.idle) checkStrips(d.idle, d, 'idle');
  ('face' in d ? d.face || [] : []).forEach((n, i) => checkNode(n, { d, where: `${at} face#${i} ${n.part}`, engines: d.engines, tiers }));
  if (d.kind === 'pedal' || d.kind === 'unit') {
    for (const p of d.params) {
      if (p.key === d.on) continue;
      const part = layOut(d.kind, p.type, Array.isArray(p.options) ? p.options.length : null);
      if (!partUse.has(part)) partUse.set(part, new Set()); partUse.get(part)!.add(`${d.id} (laid out)`);
      enginesOf(p, d).forEach(e => cover(d, e, p.key));
      if (d.kind === 'pedal' && (p.key === d.on || p.key === d.main) && p.when) err(`${at} › ${p.key}`, 'is the pedal\'s switch or main knob, and `when` would hide it');
      if (d.kind === 'pedal' && p.type === 'number' && p.when && !('param' in p.when && 'is' in p.when && Object.keys(p.when).length === 2)) note(`${at} › ${p.key}`, 'its `when` is not { param, is }, so the blanking plate over it says nothing');
    }
    const sw = d.on ? ix.params.get(d.on) : null;
    if (d.on && (!sw || sw.d !== d || sw.p.type !== 'boolean')) err(at, `its switch "${d.on}" is not one of its own switches`);
    if (sw?.p.when) err(`${at} › ${d.on}`, 'is the switch, and `when` would hide it');
    if (d.on) enginesOf(sw!.p, d).forEach(e => cover(d, e, d.on!));
  }
  if (d.kind === 'pedal') {
    if (d.main) { const m = ix.params.get(d.main); if (!m || m.d !== d || m.p.type !== 'number') err(at, `its main knob "${d.main}" is not one of its own numbers`); else if (m.p.when) err(`${at} › ${d.main}`, 'is the main knob, and `when` would hide it'); }
    (d.face || []).forEach((n, i) => { if (!n.tiers || n.tiers.includes('quick')) err(`${at} face#${i}`, 'a pedal\'s face nodes are tiered studio or lab: at Quick a pedal is its name and switch'); });
  }
  /* is anything of it on each of its engines? */
  for (const e of d.engines) {
    const hasParam = d.params.some(p => enginesOf(p, d).includes(e)), shown = visibleOn.get(d.id)?.has(e) || d.kind !== 'module';
    if (!hasParam && !(d.state || []).length) err(at, `is on ${e}, and none of its parameters is: take ${e} off it, or add ${e} to the engines of the parameters it should have there${d.params.some(p => p.per?.length) ? ' (those with engine lines list their engines beside them)' : ''}`);
    else if (!hasParam && !shown) err(at, `is on ${e}, and nothing of it shows there: take ${e} off it`);
    else if (d.kind === 'module' && !shown && !(d.strip || []).some(s => everHolds(s.when, e))) err(at, `is on ${e}, and no node of its face shows there and no strip holds: its tile would be empty`);
  }
  if (!d.thumb) { if (!(d.kind === 'module' && (d.builtIn || d.source))) err(at, 'has no thumbnail, and the Modules panel and the picker show one: add thumb and its file in thumbs/'); }
  else {
    const files = [d.thumb.file, ...(d.thumb.per || []).map(x => x.file)];
    files.forEach(f => {
      const p = path.join(DEFDIR, 'thumbs', f + '.svg');
      if (!fs.existsSync(p)) err(at, `thumbnail thumbs/${f}.svg does not exist${d.kind !== 'module' ? `: \`node tools/thumb.ts ${d.id}\` draws one from its parameters` : ''}`);
      else if (!/<(path|rect|circle|g|ellipse|line|polygon)\b/.test(fs.readFileSync(p, 'utf8'))) err(at, `thumbnail thumbs/${f}.svg draws nothing`);
    });
    (d.thumb.per || []).forEach(x => x.engines.forEach(e => { if (!d.engines.includes(e)) err(at, `thumbnail for ${e}, which it is not on`); }));
  }
}
for (const { s, d } of ix.states.values()) if (!stateUse.has(s.key)) err(`${fileOf(d)} › ${s.key}`, 'is state no node names, so nothing shows or changes it: name it in the node of the part that holds it, or drop it');

/* every parameter of a module has a control on every engine it is on */
for (const { p, d } of ix.params.values()) {
  if (d.kind !== 'module') continue;
  for (const e of enginesOf(p, d)) if (d.engines.includes(e) && !coverage.get(`${d.id}|${e}`)?.has(p.key)) {
    err(`${fileOf(d)} › ${p.key} [${e}]`, `no control edits it on ${e} at any tier, so nothing can change it or show it is changed: add a node that edits it there, or take ${e} off the parameter's engines`);
  }
}
const drawn = new Set(DEFINITIONS.flatMap(d => d.thumb ? [d.thumb.file, ...(d.thumb.per || []).map(x => x.file)] : []));
for (const f of fs.readdirSync(path.join(DEFDIR, 'thumbs'))) if (!drawn.has(f.replace(/\.svg$/, ''))) note(`thumbs/${f}`, 'no definition draws with it');

/* ================================================================ 7. what is sent */

const fieldDefault = (f: string) => FIX.generationDefaults[f];
for (const e of ENGINE_IDS) {
  /* one writer per field at a time */
  const writers = new Map<string, { key: string; w: WireKey | WireMap }[]>();
  for (const { p, d } of ix.params.values()) {
    if (!d.engines.includes(e) || !enginesOf(p, d).includes(e) || (d.kind === 'module' && d.source)) continue;
    for (const w of wiresOf(resolve(p, e).wire)) {
      const to = toOf(w); if (/^(model|ui):/.test(to)) continue;
      const group = typeof w === 'object' && w.via && (TRANSFORMS[w.via] as Transform).packs ? `${to}|${w.via}` : `${to}|${p.key}`;
      const list = writers.get(to) || []; if (!list.some(x => (typeof x.w === 'object' && x.w.via ? `${to}|${x.w.via}` : `${to}|${x.key}`) === group)) list.push({ key: p.key, w }); writers.set(to, list);
    }
  }
  for (const [to, list] of writers) {
    if (list.length < 2) continue;
    for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) {
      const a = list[i]!, b = list[j]!, conds = [a, b].flatMap(x => { const f = resolve(ix.params.get(x.key)!.p, e); return [f.when, typeof x.w === 'object' ? x.w.when : undefined].filter((c): c is Condition => !!c).concat([{ param: x.key }]); });
      let both = false;
      for (const desk of combos(conds, e)) {
        const sends = (x: typeof a) => { const k = x.key; if (!applies(ix, k, desk)) return false; const w = x.w; if (typeof w === 'object') { if (w.when && !holds(ix, w.when, desk) && !('else' in w)) return false; if (w.map && !(String(desk.vals[k] ?? resolve(ix.params.get(k)!.p, e).default) in w.map)) return false; } return true; };
        if (sends(a) && sends(b)) { both = true; break; }
      }
      if (both) err(`${keyOwners.get(a.key)} › ${a.key} [${e}]`, `writes ${to}, and so does "${b.key}" at the same time: make their conditions exclusive, or give one another field`);
    }
  }
  /* at defaults, what is sent agrees with the real app's defaults, or says why */
  const desk: Desk = { engine: e, tier: 'lab', vals: {}, auto: {}, state: {}, on: allOn };
  const rs = realState(ix, desk, fieldDefault, realDefaultsFrom(FIX.generationDefaults, FIX.extensions[e] ?? []));
  for (const [field, v] of Object.entries(rs.params)) {
    const real = FIX.generationDefaults[field];
    if (J(real) === J(v)) continue;
    const who = rs.from[`param:${field}`] ?? Object.entries(rs.from).filter(([to]) => to.startsWith(`param:${field}.`) || to.startsWith(`param:${field}[`)).flatMap(([, k]) => k);
    const noted = who.some(k => { const hit = ix.params.get(k.split('+')[0]!); return hit && wiresOf(resolve(hit.p, e).wire).some(w => waived(w, 'default')); });
    if (!noted) err(`${who.map(k => keyOwners.get(k.split('+')[0]!) + ' › ' + k).join(', ')} [${e}]`, `sends ${field} ${J(v)} at its default, where the real app's default is ${J(real)}: use the real default, or add 'default' to the wire's differs and say why in its note`);
  }
  for (const [field, v] of Object.entries(rs.backendParams)) {
    const x = declared(e, field); if (!x || x.default === undefined || typeof x.default === 'string' && x.default.startsWith('<')) continue;
    if (J(x.default) === J(v)) continue;
    const who = rs.from[`ext:${field}`] ?? [];
    const noted = who.some(k => { const hit = ix.params.get(k); return hit && wiresOf(resolve(hit.p, e).wire).some(w => waived(w, 'default')); });
    if (!noted) err(`${who.map(k => keyOwners.get(k) + ' › ' + k).join(', ')} [${e}]`, `sends ext:${field} ${J(v)} at its default, where the server declares ${J(x.default)}: use the server's default, or add 'default' to the wire's differs and say why in its note`);
  }
}

/* ================================================================ 7b. what a definition says to a person */

/** The keys some node of a definition edits at a tier on an engine, wherever its conditions put it. */
function tierEdits(d: Definition, e: EngineId, t: Tier): Set<string> {
  const out = new Set<string>();
  if (d.kind !== 'module') { d.params.forEach(p => out.add(p.key)); return out; }
  if (d.labUnder && t !== 'lab') return out;
  const walk = (n: Node) => {
    if (n.tiers && !n.tiers.includes(t)) return; if (n.engines && !n.engines.includes(e)) return;
    editsOf(n).forEach(k => out.add(k));
    for (const l of (n.per || []) as NodeOverride[]) if ((!l.engines || l.engines.includes(e)) && (!l.tiers || l.tiers.includes(t))) editsOf(l).forEach(k => out.add(k));
    (n.children || []).forEach(walk);
  };
  d.face.forEach(walk);
  return out;
}
for (const d of DEFINITIONS) {
  const at = fileOf(d);
  for (const e of d.engines) {
    /* two controls of one definition with one name, on one engine (B07) */
    const seen = new Map<string, string>();
    for (const p of d.params) { if (!has(p.key, e) || (d.kind !== 'module' && p.key === d.on)) continue; const l = (resolve(p, e).undoName ?? resolve(p, e).label).toLowerCase(); if (seen.has(l)) err(`${at} › ${p.key} [${e}]`, `is labelled "${resolve(p, e).label}", as "${seen.get(l)}" is: no one can tell them apart, and undo names them alike`); seen.set(l, p.key); }
    /* what Quick or Studio edits, Lab edits too: Lab is every parameter (B09) */
    if (d.kind === 'module' && !d.labUnder) { const lab = tierEdits(d, e, 'lab'); for (const t of ['quick', 'studio'] as Tier[]) for (const k of tierEdits(d, e, t)) if (ix.params.get(k)?.d === d && has(k, e) && !lab.has(k)) err(`${at} › ${k} [${e}]`, `is edited at ${t} and not at Lab, which shows every parameter: show its control at Lab too`); }
    /* the same control twice at one tier (B21) */
    if (d.kind === 'module') for (const t of (d.labUnder ? ['lab'] : ['quick', 'studio', 'lab']) as Tier[]) {
      const count = new Map<string, number>();
      const walk = (n: Node) => { if (n.tiers && !n.tiers.includes(t)) return; if (n.engines && !n.engines.includes(e)) return; if (n.param && !n.when && !(n.per || []).length) { const id = `${n.part}|${n.param}`; count.set(id, (count.get(id) ?? 0) + 1); } (n.children || []).forEach(walk); };
      d.face.forEach(walk);
      for (const [id, c] of count) if (c > 1) err(`${at} [${e}, ${t}]`, `draws ${id.replace('|', ' for ')} ${c} times`);
    }
  }
  /* a text default or example that fails its own check (B10) */
  for (const p of d.params) for (const e of enginesOf(p, d)) { const f = resolve(p, e); if (f.check) for (const [what, v] of [['default', f.default], ['example', f.example]] as const) if (typeof v === 'string' && v !== '' && !CHECKS[f.check.rule].test(v)) err(`${at} › ${p.key} [${e}]`, `its ${what} ${J(v)} fails its own check (${f.check.rule})`); if (f.maxLength !== undefined && typeof f.default === 'string' && f.default.length > f.maxLength) err(`${at} › ${p.key}`, `its default is longer than its maxLength ${f.maxLength}`); }
  /* a line that replaces the parameter's condition keeps what the base tested (B23) */
  for (const p of d.params) if (p.when) for (const l of (p.per || []).filter(isEngineLine)) if (l.when) { const lost = names(p.when).filter(k => !names(l.when!).includes(k)); if (lost.length) err(`${at} › ${p.key} [${l.engines.join(', ')}]`, `its engine line's \`when\` replaces the parameter's, which tests ${lost.join(', ')}; the line no longer does: repeat what still holds there`); }
  /* a pedal shows something at Studio (B17); a side unit shows something at every tier (B27) */
  if (d.kind === 'pedal' && !d.main && !(d.face || []).some(n => n.tiers?.includes('studio'))) err(at, 'has no main knob and no Studio node, so it shows nothing at Studio but its name: give it a main knob or a Studio node');
  if (d.kind === 'unit' && !d.face.length) err(at, 'has an empty face, so it shows nothing outside Lab: give it the object it shows');
  /* two options that claim one former value: an old record could become either */
  for (const p of d.params) for (const e of enginesOf(p, d)) { const o = resolve(p, e).options; if (!Array.isArray(o)) continue; const was = o.flatMap(x => x.was || []); if (new Set(was).size !== was.length) err(`${at} › ${p.key}`, 'two options list one value in `was`: an old record could become either'); }
}
/* a range's two ends: the lower stays below, so their defaults do, and neither stays below the other (B12, B13) */
for (const { p, d } of ix.params.values()) {
  if (!p.staysBelow) continue;
  const q = ix.params.get(p.staysBelow); if (!q) continue;
  if (q.p.staysBelow === p.key) err(`${fileOf(d)} › ${p.key}`, `stays below "${q.p.key}", which stays below it: one of them is the lower end`);
  for (const e of enginesOf(p, d)) { const a = resolve(p, e).default, b = resolve(q.p, e).default; if (typeof a === 'number' && typeof b === 'number' && !(a < b)) err(`${fileOf(d)} › ${p.key} [${e}]`, `stays below "${q.p.key}", and its default ${a} is not below that one's ${b}: put staysBelow on the lower end`); }
}
/* pedals read as a sequence of colours: each one's hue at least 15 degrees from its neighbours' in the chain, on every
   engine (a pedal another engine lacks makes two others neighbours) (B11, DH-24) */
{
  const said = new Set<string>();
  for (const { id: e } of ENGINES) {
    const chain = DEFINITIONS.filter((x): x is Extract<Definition, { kind: 'pedal' }> => x.kind === 'pedal' && x.on !== null && x.engines.includes(e)).sort((a, b) => a.at - b.at);
    for (let i = 1; i < chain.length; i++) {
      const a = chain[i - 1]!, b = chain[i]!, dh = Math.min(Math.abs(a.hue - b.hue), 360 - Math.abs(a.hue - b.hue));
      if (dh < 15 && !said.has(a.id + b.id)) { said.add(a.id + b.id); err(fileOf(b), `its hue ${b.hue} is ${dh} degrees from ${a.id}'s ${a.hue}, its neighbour in the chain on ${e}: pick one at least 15 degrees away`); }
    }
  }
}

/* ================================================================ 8. factory workflows and revisions */

for (const w of FACTORY_WORKFLOWS) {
  const at = `factory.ts › workflow "${w.name}"`;
  [...w.modules, ...w.board].forEach(id => { const d = DEFINITIONS.find(x => x.id === id); if (!d) err(at, `names "${id}", which is no module, pedal or unit`); else if (w.modules.includes(id) && d.kind !== 'module') err(at, `lists ${d.kind} "${id}" among its modules`); else if (w.board.includes(id) && d.kind === 'module') err(at, `lists module "${id}" on its board`); });
  if (FACTORY_WORKFLOWS.filter(x => x.name.toLowerCase() === w.name.toLowerCase()).length > 1) err(at, 'shares its name with another workflow');
}
for (const r of FACTORY_REVISIONS) {
  const at = `factory.ts › revision "${r.workflow} › ${r.name}" [${r.engine}]`;
  if (!FACTORY_WORKFLOWS.some(w => w.name === r.workflow)) err(at, `belongs to workflow "${r.workflow}", which the factory does not have`);
  for (const [k, v] of Object.entries(r.vals)) {
    const hit = ix.params.get(k);
    if (!hit) { err(at, `sets "${k}", which no definition has${ix.former.has(k) ? ` (it is now "${ix.former.get(k)}")` : ''}`); continue; }
    if (!enginesOf(hit.p, hit.d).includes(r.engine) || !hit.d.engines.includes(r.engine)) { err(at, `sets "${k}", which ${r.engine} does not have`); continue; }
    const stored = hit.p.stored || hit.d.stored;
    if (stored === 'none') err(at, `sets "${k}", which a revision does not keep (stored none)`);
    if (stored === 'content' && !r.withContent) err(at, `sets "${k}", which is song content: only a revision saved with content keeps it (withContent)`);
    const f = resolve(hit.p, r.engine), allowed = allowedValues(hit.p, r.engine);
    if (hit.p.type === 'number' && (typeof v !== 'number' || !f.range || v < f.range[0] || v > f.range[1])) err(at, `sets ${k} to ${J(v)}, outside its range on ${r.engine}`);
    if (hit.p.type === 'boolean' && typeof v !== 'boolean') err(at, `sets the switch ${k} to ${J(v)}`);
    if (hit.p.type === 'choice' && allowed && !allowed.includes(v as string)) err(at, `sets ${k} to ${J(v)}, which is not one of its values on ${r.engine}`);
  }
  for (const [k, s] of Object.entries(r.state || {})) { const hit = ix.states.get(k); if (!hit) err(at, `holds state "${k}", which no definition declares`); else if (hit.s.stored !== 'settings') err(at, `holds state "${k}", which a revision does not keep`); else if (J(cleanState(hit.s, s)) !== J(s)) err(at, `holds ${k} ${J(s)}, which is not a ${hit.s.kind}: loading it would drop it`); }
  const c = cleanValues(ix, r.vals, r.auto);
  for (const [k, v] of Object.entries(r.vals)) if (ix.params.has(k) && J(c.vals[k]) !== J(v)) err(at, `sets ${k} to ${J(v)}, which loading cleans to ${J(c.vals[k])}: write what it means`);
  const rd: Desk = { engine: r.engine, tier: 'lab', vals: c.vals, auto: c.auto, state: (r.state || {}) as Record<string, Json>, on: new Set(itemIds) };
  for (const k of Object.keys(r.vals)) if (ix.params.has(k) && has(k, r.engine) && !applies(ix, k, rd)) err(at, `sets ${k}, which does not apply in this revision (its condition does not hold), so it does nothing`);
  for (const k of Object.keys(r.auto || {})) { const hit = ix.params.get(k); if (!hit || !resolve(hit.p, r.engine).auto) err(at, `sets the auto switch of "${k}", which has none on ${r.engine}`); }
}

/* ================================================================ 8b. the roles (definitions/roles.ts)
   What the page needs a particular parameter or state for: what plays each role is of the kind its role says, one per
   engine unless the role is `many`; a needed role is played; a role nothing plays is a note saying what the page does
   without it. The core and the view ask for roles, never keys (tools/test-view.ts refuses a key or an id in their
   code), so a key renamed the documented way keeps its role, and one removed is named here. */

for (const [r, spec] of Object.entries(ROLES) as [Role, RoleSpec][]) {
  const ks = ix.roles.get(r) ?? [], where = `roles.ts › ${r}`;
  if (!ks.length) { (spec.needed ? err : note)(where, `no definition plays it: give a ${spec.of === 'param' ? `${spec.type} parameter` : `${spec.type} state`} \`role: '${r}'\`${spec.needed ? '' : ', or the page does without it'} (${spec.doc})`); continue; }
  for (const k of ks) {
    const hit = spec.of === 'param' ? ix.params.get(k) : ix.states.get(k), at = `${fileOf(hit?.d ?? DEFINITIONS[0]!)} › ${k}`;
    if (!hit) { err(at, `plays the role "${r}", which is a ${spec.of === 'param' ? 'parameter' : 'state'}'s: give it to one`); continue; }
    const kind = 'p' in hit ? hit.p.type : hit.s.kind;
    if (kind !== spec.type) err(at, `plays the role "${r}", which wants a ${spec.type}, and it is a ${kind}: give the role to a ${spec.type}, or change roles.ts`);
  }
  if (spec.many) continue;
  if (spec.of === 'state' && ks.length > 1) err(`roles.ts › ${r}`, `is played by ${ks.join(' and ')}: the page takes the first; give the role to one`);
  if (spec.of === 'param') for (const e of ENGINE_IDS) { const on = ks.filter(k => has(k, e)); if (on.length > 1) err(`roles.ts › ${r}`, `is played by ${on.join(' and ')} on ${e}: give the role to one, or tell them apart by engines`); }
}
const rolesTable = ['| Role | Played by | What the page does with it |', '| --- | --- | --- |',
  ...(Object.entries(ROLES) as [Role, RoleSpec][]).map(([r, s]) => `| \`${r}\`${s.needed ? ' (needed)' : ''}${s.many ? ' (several)' : ''} | ${(ix.roles.get(r) ?? []).map(k => `\`${k}\` (${(ix.params.get(k)?.d ?? ix.states.get(k)?.d)!.id})`).join(', ') || 'nothing'} | ${s.doc.replace(/\|/g, '\\|')} |`)].join('\n');

/* ================================================================ 9. what is in the inventory and the format but unused: notes */

const partsUnused = Object.keys(PARTS).filter(p => !partUse.has(p));
partsUnused.forEach(p => note('parts.ts', `part "${p}" is used by no definition`));
const optsUsed = new Map<string, Set<string>>();
for (const d of DEFINITIONS) { const walk = (n: Node) => { [n, ...((n.per || []) as NodeOverride[])].forEach(x => Object.keys((x as { opts?: object }).opts || {}).forEach(o => { if (!optsUsed.has(n.part)) optsUsed.set(n.part, new Set()); optsUsed.get(n.part)!.add(o); })); (n.children || []).forEach(walk); }; ('face' in d ? d.face || [] : []).forEach(walk); }
for (const [name, spec] of Object.entries(PARTS) as [string, PartSpec][]) for (const o of Object.keys(spec.opts || {})) if (!optsUsed.get(name)?.has(o)) note('parts.ts', `option ${name}.${o} is used by no definition`);
for (const [name, spec] of Object.entries(PARTS) as [string, PartSpec][]) if (spec.edits && !spec.drive) err('parts.ts', `part "${name}" edits and says no \`drive\`, so the generated tests cannot operate it`);
for (const a of Object.keys(ACTIONS)) if (!DEFINITIONS.some(d => J(d).includes(`"name":"${a}"`))) note('parts.ts', `action "${a}" is run by no definition`);
for (const f of Object.keys(DESK_FACTS)) if (!DEFINITIONS.some(d => J(d).includes(`"${f}"`) || J(d).includes(`:${f}}`))) note('parts.ts', `desk fact "${f}" is named by no definition`);
for (const t of Object.keys(TRANSFORMS)) if (!DEFINITIONS.some(d => J(d).includes(`"via":"${t}"`))) note('transforms.ts', `transform "${t}" is named by no wire`);

/* ================================================================ 10. the numbers, and the generated parts of the documents */

const pairs = [...ix.params.values()].flatMap(({ p, d }) => enginesOf(p, d).filter(e => d.engines.includes(e)).map(e => ({ p, d, e, w: resolve(p, e).wire })));
const isNone = (w: Wire) => typeof w === 'object' && !Array.isArray(w) && 'none' in w;
const channels: Record<string, number> = {};
pairs.forEach(x => { if (isNone(x.w)) return; const ch = new Set(wiresOf(x.w).map(w => toOf(w).split(':')[0]!)); ch.forEach(c => channels[c] = (channels[c] || 0) + 1); });
const kinds = { module: 0, pedal: 0, unit: 0 } as Record<string, number>; DEFINITIONS.forEach(d => kinds[d.kind]!++);
const noneKeys = [...new Set(pairs.filter(x => isNone(x.w)).map(x => x.p.key))];
const wireKinds = { plain: 0, mapped: 0, transformed: 0, none: 0 };
pairs.forEach(x => { if (isNone(x.w)) wireKinds.none++; else { const ws = wiresOf(x.w); if (ws.some(w => typeof w === 'object' && w.via)) wireKinds.transformed++; else if (ws.some(w => typeof w === 'object' && (w.map || w.invert || w.scale || w.divide || 'auto' in w || w.when)) || ws.length > 1) wireKinds.mapped++; else wireKinds.plain++; } });
const counts = [
  `- ${DEFINITIONS.length} definitions: ${kinds.module} modules, ${kinds.pedal} pedals, ${kinds.unit} side units, on ${ENGINES.length} engines.`,
  `- ${ix.params.size} parameters; ${pairs.length} parameter-engine pairs. Of those, ${pairs.length - wireKinds.none} say where the value goes (${Object.entries(channels).map(([c, n]) => `${c} ${n}`).join(', ')}): ${wireKinds.plain} as a plain name, ${wireKinds.mapped} through wire fields (invert, scale, divide, map, auto, when, several fields), ${wireKinds.transformed} through a named transform. ${wireKinds.none} pairs, on ${noneKeys.length} parameters, have no counterpart in the real app.`,
  `- ${ix.states.size} state entries; ${Object.keys(PARTS).length} parts in the inventory, ${partUse.size} used; ${Object.keys(TRANSFORMS).length} transforms; ${FACTORY_WORKFLOWS.length} factory workflows and ${FACTORY_REVISIONS.length} factory revisions.`,
].join('\n');

const cell = (s: string) => s.replace(/\|/g, '\\|').replace(/\n/g, ' ');
const inventory = ['| Part | Kind | Edits | State | Reads | What it is for | Options | States | Styles (concept src/css) | Used by |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
  ...(Object.entries(PARTS) as [string, PartSpec][]).map(([name, p]) => `| \`${name}\` | ${p.kind} | ${(p.edits || []).join(', ')} | ${(p.state || []).join(', ')} | ${Array.isArray(p.reads) ? p.reads.join(', ') : p.reads ? 'any' : ''} | ${cell(p.doc + (p.responsible ? ` **Responsible for:** ${p.responsible}` : ''))} | ${cell(Object.entries(p.opts || {}).map(([k, o]) => `\`${k}\` (${Array.isArray(o.type) ? o.type.join(' \\| ') : o.type}): ${o.doc}`).join('<br>'))} | ${cell((p.states || []).join('; '))} | ${p.css.join(', ')} | ${[...(partUse.get(name) || [])].join(', ')} |`)].join('\n');
const SHORT: Record<string, string> = { ace: 'ace', 'minimax-m3': 'mm3', yue2: 'yue2', sa3: 'sa3', heartmula: 'hm' };
const say = (w: Wire): string => {
  if (Array.isArray(w)) return w.map(say).join(' + ');
  if (typeof w === 'string') return '`' + w + '`';
  if ('none' in w) return `none: ${w.none}`;
  const bits = [w.invert && 'inverted', w.scale && `× ${w.scale}`, w.divide && `÷ ${w.divide}`, w.map && `map ${Object.entries(w.map).map(([k, v]) => `${k} → ${J(v)}`).join(', ')}`, w.via && `via ${w.via}`, 'auto' in w && `auto: ${J(w.auto)}`, w.when && `while ${J(w.when)}${'else' in w ? `, else ${J(w.else)}` : ''}`].filter(Boolean);
  return '`' + w.to + '`' + (bits.length ? ` (${bits.join('; ')})` : '') + (w.note ? `. ${w.note}` : '');
};
const namesTable = DEFINITIONS.filter(d => d.params.length).map(d => [`### ${d.name} (\`${d.id}\`, ${d.kind})`, '', '| Concept key | Type | Engines | Real name |', '| --- | --- | --- | --- |',
  ...d.params.map(p => { const engs = enginesOf(p, d), per = engs.map(e => [e, resolve(p, e).wire] as const), one = new Set(per.map(([, w]) => J(w))).size === 1;
    const real = one ? say(per[0]![1]) : per.map(([e, w]) => `${SHORT[e]}: ${say(w)}`).join('<br>');
    return `| \`${p.key}\` | ${p.type} | ${engs.length === ENGINES.length ? 'all' : engs.map(e => SHORT[e]).join(', ')} | ${cell(real)} |`; })].join('\n')).join('\n\n');
const noneTable = ['| Key | Engines | What the real app does instead |', '| --- | --- | --- |', ...noneKeys.map(k => { const ps = pairs.filter(x => x.p.key === k && isNone(x.w)); return `| \`${k}\` | ${ps.map(x => SHORT[x.e]).join(', ')} | ${cell((ps[0]!.w as { none: string }).none)} |`; })].join('\n');
const GENERATED: Record<string, string> = { counts, inventory, names: namesTable, none: noneTable, roles: rolesTable };

/* ================================================================ 11. what saved data may hold: every key, state and id shipped */

const SHIPPED_FILE = path.join(ROOT, 'tools', 'fixtures', 'shipped.json');
type Shipped = { keys: Record<string, string>; states: Record<string, string>; ids: Record<string, string> };
const shipped: Shipped = fs.existsSync(SHIPPED_FILE) ? JSON.parse(fs.readFileSync(SHIPPED_FILE, 'utf8')) : { keys: {}, states: {}, ids: {} };
/* what saved data depends on: where a value is kept and what saves it */
const now: Shipped = {
  keys: Object.fromEntries([...ix.params.values()].map(({ p, d }) => [p.key, `${d.id} ${p.owner ?? d.owner} ${p.stored ?? d.stored}`])),
  states: Object.fromEntries([...ix.states.values()].map(({ s, d }) => [s.key, `${d.id} ${s.owner} ${s.stored}`])),
  ids: Object.fromEntries(DEFINITIONS.map(d => [d.id, d.kind])),
};
for (const [what, former, retired] of [['keys', ix.former, RETIRED.keys], ['states', ix.formerStates, RETIRED.states], ['ids', ix.formerIds, RETIRED.ids]] as const) {
  for (const [k, was] of Object.entries(shipped[what])) {
    if (k in now[what]) { if (now[what][k] !== was && !process.argv.includes('--ship')) err(`tools/fixtures/shipped.json › ${k}`, `was shipped as "${was}" and is now "${now[what][k]}": saved records keep it where it was kept then. If that is meant, the change needs a migration step (src/derive/records.ts) before the next release`); continue; }
    if (former.has(k) || retired.includes(k)) continue;
    err(`tools/fixtures/shipped.json › ${k}`, `saved data may hold ${what === 'ids' ? 'the id' : what === 'states' ? 'the state' : 'the key'} "${k}", which no definition has any more: list it in \`was\` where it went, or in src/definitions/retired.ts if it is gone`);
  }
  for (const k of retired) if (k in now[what]) err('src/definitions/retired.ts', `retires "${k}", which a definition still has`);
}
/* what a release ships is recorded on purpose, by `npm run ship` when a release goes out; nothing else writes it */
if (process.argv.includes('--ship')) {
  const merged: Shipped = { keys: { ...shipped.keys, ...now.keys }, states: { ...shipped.states, ...now.states }, ids: { ...shipped.ids, ...now.ids } };
  /* a key no definition has stays listed: its going is said by `was` or retired.ts, never by forgetting it */
  const sort = (o: Record<string, string>) => Object.fromEntries(Object.entries(o).sort(([a], [b]) => a.localeCompare(b)));
  fs.writeFileSync(SHIPPED_FILE, JSON.stringify({ note: 'Every parameter key, state key and definition id the definitions ship with, and where each value is kept (definition, owner, stored). Written by npm run ship when a release goes out, and by nothing else; the validator checks the definitions against it (docs/07).', keys: sort(merged.keys), states: sort(merged.states), ids: sort(merged.ids) }, null, 1).replace(/\n/g, '\r\n') + '\r\n');
}

const DOCS = path.join(ROOT, 'docs');
const stale: string[] = [];
for (const f of fs.existsSync(DOCS) ? fs.readdirSync(DOCS).filter(f => f.endsWith('.md')) : []) {
  const p = path.join(DOCS, f), text = fs.readFileSync(p, 'utf8');
  const nl = text.includes('\r\n') ? '\r\n' : '\n';
  const next = text.replace(/(<!-- generated: (\w+)[^>]*-->)([\s\S]*?)(\r?\n<!-- end generated -->)/g, (whole, open: string, name: string, _body: string, close: string) => {
    if (!(name in GENERATED)) { err(`docs/${f}`, `has a generated block "${name}", which the validator does not make`); return whole; }
    return open + nl + GENERATED[name]!.replace(/\r?\n/g, nl) + close;
  });
  if (next !== text) { if (process.argv.includes('--docs')) fs.writeFileSync(p, next); else stale.push(f); }
}
if (stale.length) err('docs', `${stale.join(', ')} ${stale.length > 1 ? 'are' : 'is'} out of date with the definitions: run \`npm run docs\``);

/* ================================================================ report */
console.log(counts.replace(/^- /gm, ''));
if (process.argv.includes('--notes') && notes.length) { console.log(`\nnotes (${notes.length}):`); notes.forEach(n => console.log('  ' + n)); }
else if (notes.length) console.log(`\n${notes.length} notes (--notes prints them).`);
if (errors.length) { console.log(`\nerrors (${errors.length}):`); errors.forEach(e => console.log('  ' + e)); process.exit(1); }
console.log('\nOK: no errors.');
