// Checks the definitions for what the type system cannot: unique keys, defaults inside their ranges, non-empty
// option lists, parts that exist and fit what they are bound to, overrides and conditions that refer to real
// engines, parameters and state, thumbnails that exist, and every field of the format in use.
//
//   node tools/validate-definitions.ts            the checks; exit 1 on any error
//   node tools/validate-definitions.ts --usage    also print where each part is used, and the parameter counts
//
// Needs Node 22.18 or later (it runs the TypeScript sources directly).

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { DEFINITIONS } from '../src/definitions/index.ts';
import { PARTS, DESK_STATES, VALIDATORS, type PartSpec } from '../src/definitions/parts.ts';
import type { Condition, Definition, EngineId, Node, ParamDef, ParamFace, Value, WireRef, Option } from '../src/definitions/types.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFDIR = path.join(ROOT, 'src', 'definitions');
const ENGINES: EngineId[] = ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'];
const TIERS = ['quick', 'studio', 'lab'];
const errors: string[] = [], notes: string[] = [];
const err = (where: string, msg: string) => errors.push(`${where}: ${msg}`);

// ---------------------------------------------------------------- the files themselves
const listed = new Set(DEFINITIONS.map(d => `${d.kind}/${d.id}`));
for (const [dir, kind] of [['modules', 'module'], ['pedals', 'pedal'], ['units', 'unit']] as const) {
  for (const f of fs.readdirSync(path.join(DEFDIR, dir)).filter(f => f.endsWith('.ts'))) {
    const id = f.slice(0, -3);
    if (!listed.has(`${kind}/${id}`)) err(`${dir}/${f}`, 'not listed in index.ts (or its id does not match its file name)');
  }
}
for (const d of DEFINITIONS) {
  if (JSON.stringify(JSON.parse(JSON.stringify(d))) !== JSON.stringify(d)) err(d.id, 'is not plain JSON data');
  const walk = (v: unknown, at: string): void => {
    if (typeof v === 'function' || typeof v === 'bigint' || typeof v === 'symbol' || v === undefined) err(d.id, `${at} holds ${typeof v}, which JSON cannot carry`);
    else if (typeof v === 'number' && !Number.isFinite(v)) err(d.id, `${at} is not a finite number`);
    else if (Array.isArray(v)) v.forEach((x, i) => walk(x, `${at}[${i}]`));
    else if (v && typeof v === 'object') for (const [k, x] of Object.entries(v)) walk(x, `${at}.${k}`);
  };
  walk(d, d.id);
}

// ---------------------------------------------------------------- indexes
const ids = new Map<string, Definition>();
const params = new Map<string, { p: ParamDef; d: Definition }>();
const states = new Map<string, Definition>();
for (const d of DEFINITIONS) {
  const key = `${d.kind === 'module' ? '' : d.kind + ':'}${d.id}`;
  if (ids.has(key)) err(d.id, `id used twice (${d.kind})`); ids.set(key, d);
  for (const p of d.params) { if (params.has(p.key)) err(p.key, `parameter key used twice (${params.get(p.key)!.d.id} and ${d.id})`); params.set(p.key, { p, d }); }
  for (const s of d.state || []) { if (states.has(s.key) || params.has(s.key)) err(s.key, 'state key used twice, or also a parameter key'); states.set(s.key, d); }
}
const moduleIds = new Set(DEFINITIONS.filter(d => d.kind === 'module').map(d => d.id));
const itemIds = new Set([...moduleIds, ...DEFINITIONS.filter(d => d.kind !== 'module').map(d => d.id)]);
const engOf = (p: ParamDef, d: Definition): EngineId[] => p.engines || d.engines;
/** A parameter as one engine sees it: base fields, then that engine's override. */
const resolve = (p: ParamDef, e: EngineId): ParamFace & ParamDef => ({ ...p, ...(p.per?.[e] || {}) } as ParamFace & ParamDef);
const typeOf = (v: Value) => typeof v === 'boolean' ? 'boolean' : typeof v === 'number' ? 'number' : 'string';
const optionValues = (o: ParamFace['options']): (string | number)[] | null =>
  !o ? null : Array.isArray(o) ? o.map(x => x.value) : null;
const allowedValues = (o: ParamFace['options']): (string | number)[] | null =>
  !o ? null : Array.isArray(o) ? o.map(x => x.value) : (o.include || []).map(x => x.value);

// ---------------------------------------------------------------- conditions
/* A parameter an engine does not have reads as absent, and a condition on it is false there. That is an error
   only where nothing guards it: at the top of a condition, or in an `all` that no engine test narrows. */
function checkCond(c: Condition, where: string, engines: EngineId[], soft = false): void {
  if ('all' in c) {
    const g = c.all.find((x): x is { engine: EngineId[] } => 'engine' in x), narrowed = g ? engines.filter(e => g.engine.includes(e)) : engines;
    return c.all.forEach(x => checkCond(x, where, narrowed.length ? narrowed : engines, soft || !narrowed.length));
  }
  if ('any' in c) return c.any.forEach(x => checkCond(x, where, engines, true));
  if ('not' in c && !('param' in c)) return checkCond((c as { not: Condition }).not, where, engines, true);
  if ('engine' in c) { c.engine.forEach(e => { if (!ENGINES.includes(e)) err(where, `condition names an unknown engine "${e}"`); }); return; }
  if ('onDesk' in c) { if (!itemIds.has(c.onDesk)) err(where, `condition names an unknown module "${c.onDesk}"`); return; }
  if ('state' in c) { if (!states.has(c.state) && !(c.state in DESK_STATES)) err(where, `condition names unknown state "${c.state}"`); return; }
  const hit = params.get(c.param);
  if (!hit) { err(where, `condition names unknown parameter "${c.param}"`); return; }
  const { p, d } = hit;
  if (!engines.some(e => engOf(p, d).includes(e))) { if (!soft) err(where, `condition on "${c.param}", which none of its engines have`); return; }
  const vals = [...[c.is ?? []].flat(), ...[c.not ?? []].flat()] as Value[];
  for (const e of engOf(p, d).filter(x => engines.includes(x))) {
    const f = resolve(p, e), ok = optionValues(f.options);
    for (const v of vals) {
      if (p.type === 'choice' && ok && !ok.includes(v as string)) err(where, `condition compares "${c.param}" with "${v}", which is not one of its options on ${e}`);
      if (p.type === 'boolean' && typeof v !== 'boolean') err(where, `condition compares boolean "${c.param}" with ${JSON.stringify(v)}`);
      if (p.type === 'number' && typeof v !== 'number') err(where, `condition compares number "${c.param}" with ${JSON.stringify(v)}`);
    }
    if ((c.above !== undefined || c.below !== undefined) && p.type !== 'number') err(where, `above/below on "${c.param}", which is not a number`);
    if (c.auto !== undefined && !f.auto) err(where, `condition on "${c.param}".auto, which has no auto on ${e}`);
  }
}

// ---------------------------------------------------------------- parameters
const wireKeys = (w: WireRef): string[] => typeof w === 'string' ? [w] : 'to' in w ? [w.to] : [];
for (const { p, d } of params.values()) {
  const where = `${d.id} › ${p.key}`, engs = engOf(p, d);
  engs.forEach(e => { if (!d.engines.includes(e)) err(where, `has engine ${e}, which its ${d.kind} does not`); });
  for (const e of Object.keys(p.per || {})) if (!engs.includes(e as EngineId)) err(where, `overrides engine "${e}", which it is not on`);
  for (const ref of [p.below, p.above, p.beside]) if (ref && !params.has(ref)) err(where, `refers to unknown parameter "${ref}"`);
  if (p.beside && params.get(p.beside)?.p.type !== 'boolean') err(where, `stands beside "${p.beside}", which is not a switch`);
  if ((p.below || p.above) && p.type !== 'number') err(where, 'below/above on a parameter that is not a number');
  /* a song value travels between engines; if its range differs per engine it must be held, or a switch would clamp it
     and lose the user's value (the concept's "value left out of range after an engine switch") */
  const owner = p.owner || d.owner, ranges = new Set(engs.map(e => JSON.stringify(resolve(p, e).range)));
  if (owner === 'song' && p.type === 'number' && ranges.size > 1 && !p.held) err(where, 'a song value whose range differs per engine, not marked held');
  if (p.held && (owner !== 'song' || p.type !== 'number')) err(where, 'held is for song numbers whose range differs per engine');
  for (const e of engs) {
    const f = resolve(p, e), at = `${where} [${e}]`;
    const w = f.wire;
    if (typeof w === 'object' && 'none' in w && (!w.none || w.none === 'MISSING')) err(at, 'wire says nothing about the real application');
    wireKeys(w).forEach(k => { if (!/^(param|ext|content|model|ui|plugin):\S/.test(k)) err(at, `wire "${k}" has no channel`); });
    if (f.when) checkCond(f.when, at + ' when', [e]);
    (f.labelWhen || []).forEach(l => checkCond(l.when, at + ' labelWhen', [e]));
    if (f.check && !(f.check.rule in VALIDATORS)) err(at, `unknown check "${f.check.rule}"`);
    if (f.check && p.type !== 'text') err(at, 'a check on a parameter that is not text');
    const dv = f.default;
    if (p.type === 'number') {
      if (!f.range) { err(at, 'a number without a range'); continue; }
      const [lo, hi, step] = f.range;
      if (!(lo <= hi)) err(at, `range ${lo}..${hi} is empty`);
      if (!(step > 0)) err(at, `step ${step} is not positive`);
      if (typeof dv !== 'number') err(at, `default ${JSON.stringify(dv)} is not a number`);
      else if (dv < lo || dv > hi) err(at, `default ${dv} is outside ${lo}..${hi}`);
      else if (Math.abs((dv - lo) / step - Math.round((dv - lo) / step)) > 1e-6) notes.push(`${at}: default ${dv} is off the step grid (${lo} + n × ${step})`);
      if (f.options) err(at, 'a number with options');
    } else if (p.type === 'boolean') {
      if (typeof dv !== 'boolean') err(at, `default ${JSON.stringify(dv)} is not a boolean`);
    } else if (p.type === 'choice') {
      if (!f.options) { err(at, 'a choice without options'); continue; }
      if (Array.isArray(f.options)) {
        if (!f.options.length) err(at, 'option list is empty');
        const vs = f.options.map((o: Option) => o.value);
        if (new Set(vs).size !== vs.length) err(at, 'option values repeat');
        if (!vs.includes(dv as string | number)) err(at, `default ${JSON.stringify(dv)} is not one of its options`);
      } else {
        const inc = (f.options.include || []).map(o => o.value);
        if (f.options.from === 'models' && !f.options.bucket) err(at, 'options from models without a bucket');
        if (typeof dv !== 'string' && typeof dv !== 'number') err(at, 'default of a server-listed choice must be a string or number');
        if (dv === '' && !inc.includes('')) err(at, 'default "" is not among the options it always has (include)');
      }
      if (f.range) err(at, 'a choice with a range');
    } else if (p.type === 'text') {
      if (typeof dv !== 'string') err(at, `default ${JSON.stringify(dv)} is not text`);
      if (f.options || f.range) err(at, 'text with options or a range');
    }
    if (f.auto && p.type === 'boolean') err(at, 'a switch cannot be automatic');
    if (f.format && p.type !== 'number' && p.type !== 'boolean') err(at, 'a format on a parameter that is neither a number nor a switch');
    if (f.format?.template?.includes('{s}') && !f.format.perSecond) err(at, 'format uses {s} without perSecond');
    for (const wd of f.format?.words || []) if (typeOf(wd.value) !== typeOf(dv)) err(at, `format word for ${JSON.stringify(wd.value)} does not match the value type`);
  }
}

// ---------------------------------------------------------------- state
for (const d of DEFINITIONS) for (const s of d.state || []) {
  if (s.load && !s.load.includes('{file}')) err(`${d.id} › ${s.key}`, 'load text has no {file}');
  if (s.load && s.kind !== 'file' && s.kind !== 'slots') err(`${d.id} › ${s.key}`, 'load text on state that is not a file');
  const ws = typeof s.wire === 'string' || 'to' in s.wire || 'none' in s.wire ? [s.wire as WireRef] : Object.entries(s.wire).map(([e, w]) => { if (!d.engines.includes(e as EngineId)) err(`${d.id} › ${s.key}`, `wire for engine "${e}" it is not on`); return w as WireRef; });
  ws.forEach(w => wireKeys(w).forEach(k => { if (!/^(param|ext|content|model|ui|plugin):\S/.test(k)) err(`${d.id} › ${s.key}`, `wire "${k}" has no channel`); }));
}

// ---------------------------------------------------------------- faces
const partUse = new Map<string, Set<string>>();
const shownParams = new Set<string>();
function checkNode(n: Node, d: Definition, where: string, engines: EngineId[], labOnly: boolean): void {
  const spec = (PARTS as Record<string, PartSpec>)[n.part];
  if (!spec) { err(where, `names part "${n.part}", which is not in the inventory`); return; }
  if (!partUse.has(n.part)) partUse.set(n.part, new Set()); partUse.get(n.part)!.add(d.id);
  const here = n.engines ? n.engines.filter(e => engines.includes(e)) : engines;
  if (n.engines) { n.engines.forEach(e => { if (!ENGINES.includes(e)) err(where, `unknown engine "${e}"`); else if (!engines.includes(e)) err(where, `limited to ${e}, which its ${d.kind} or parent node is not on`); }); }
  if (n.tiers) { if (labOnly) err(where, 'tiers on a node of a Lab-only module'); n.tiers.forEach(t => { if (!TIERS.includes(t)) err(where, `unknown tier "${t}"`); }); }
  for (const o of Object.keys(n.opts || {})) if (!spec.opts || !(o in spec.opts)) err(where, `part "${n.part}" takes no option "${o}"`);
  const bound = [n.param, ...(n.params || [])].filter((x): x is string => !!x);
  for (const k of bound) {
    const hit = params.get(k); if (!hit) { err(where, `binds unknown parameter "${k}"`); continue; }
    shownParams.add(k);
    if (!here.some(e => engOf(hit.p, hit.d).includes(e))) err(where, `binds "${k}" on engines that do not have it`);
  }
  if (n.param) {
    const t = params.get(n.param)?.p.type;
    if (t && !spec.binds.includes(t)) err(where, `part "${n.part}" cannot edit a ${t} ("${n.param}"); it binds ${spec.binds.join(', ')}`);
  }
  if (spec.binds.includes('pair') && (n.params?.length !== 2 || !n.params.every(k => params.get(k)?.p.type === 'number'))) err(where, `part "${n.part}" needs two numbers in params`);
  if (spec.binds.includes('state') && spec.kind === 'control' && !(n.opts && typeof n.opts.state === 'string' && (states.has(n.opts.state as string)))) {
    if (!n.opts?.state) err(where, `part "${n.part}" needs opts.state naming the module's state`); else err(where, `opts.state "${n.opts.state}" is not a state`);
  }
  if (n.part === 'choiceRow') {
    if (!n.choices) err(where, 'a choice row without choices');
    const lists = Array.isArray(n.choices) ? [[null, n.choices] as const] : Object.entries(n.choices || {});
    const hit = n.param ? params.get(n.param) : undefined;
    for (const [e, list] of lists) {
      if (e && !here.includes(e as EngineId)) err(where, `choices for engine "${e}", which the node is not on`);
      for (const [word, v] of list) {
        if (!hit) continue;
        for (const eng of (e ? [e as EngineId] : here).filter(x => engOf(hit.p, hit.d).includes(x))) {
          const f = resolve(hit.p, eng);
          if (hit.p.type === 'number' && f.range && (typeof v !== 'number' || v < f.range[0] || v > f.range[1])) err(where, `choice "${word}" = ${v} is outside ${n.param}'s range on ${eng}`);
          const ok = allowedValues(f.options); if (hit.p.type === 'choice' && Array.isArray(f.options) && ok && !ok.includes(v as string)) err(where, `choice "${word}" = ${v} is not an option of ${n.param}`);
        }
      }
    }
    if (n.between && !n.between.includes('{v}')) err(where, 'between has no {v}');
  } else if (n.choices || n.between) err(where, 'choices or between on a part that is not a choice row');
  if (n.part === 'termLine' && Array.isArray(n.opts?.flags) && n.opts.flags.length !== (n.params?.length || 0) + (n.children?.length || 0))
    err(where, `a terminal line with ${n.opts.flags.length} flags for ${(n.params?.length || 0) + (n.children?.length || 0)} controls`);
  if (n.when) checkCond(n.when, where + ' when', here);
  if (n.dim) checkCond(n.dim.when, where + ' dim', here);
  if (n.action) checkAction(n.action, where, here);
  (n.children || []).forEach((c, i) => checkNode(c, d, `${where} > ${c.part}#${i}`, here, labOnly));
  if (n.children?.length && spec.kind !== 'container' && spec.kind !== 'bespoke' && n.part !== 'note') err(where, `part "${n.part}" is a ${spec.kind} and cannot hold children`);
  if (spec.kind === 'container' && !n.children?.length && !n.params?.length) err(where, `container "${n.part}" holds nothing`);
}
function checkAction(a: NonNullable<Node['action']>, where: string, engines: EngineId[]): void {
  if ('param' in a) { const hit = params.get(a.param); if (!hit) err(where, `action on unknown parameter "${a.param}"`);
    else if (a.do === 'set') { const f = resolve(hit.p, engOf(hit.p, hit.d).find(e => engines.includes(e)) || engOf(hit.p, hit.d)[0]!); const ok = allowedValues(f.options);
      if (typeOf(a.value) !== (hit.p.type === 'choice' || hit.p.type === 'text' ? 'string' : hit.p.type)) err(where, `action sets ${a.param} to ${JSON.stringify(a.value)}, the wrong type`);
      if (hit.p.type === 'choice' && Array.isArray(f.options) && ok && !ok.includes(a.value as string)) err(where, `action sets ${a.param} to "${a.value}", not one of its options`); }
    else if (a.do === 'auto' && !resolve(hit.p, engOf(hit.p, hit.d)[0]!).auto) err(where, `action turns on auto for ${a.param}, which has none`); }
  if (a.do === 'tier') a.ring?.forEach(k => { if (!params.has(k)) err(where, `rings unknown parameter "${k}"`); });
  if (a.do === 'add' && !itemIds.has(a.item)) err(where, `adds unknown item "${a.item}"`);
}
for (const d of DEFINITIONS) {
  const labOnly = d.kind === 'module' && !!d.labUnder;
  if (d.kind === 'module' && d.labUnder && !moduleIds.has(d.labUnder)) err(d.id, `labUnder names unknown module "${d.labUnder}"`);
  (d.face || []).forEach((n, i) => checkNode(n, d, `${d.id} face#${i} ${n.part}`, d.engines, labOnly));
  for (const s of [...(d.kind === 'module' ? d.strip || [] : []), ...(d.kind === 'pedal' ? [...(d.idle || []), ...(d.blank || [])] : [])]) {
    checkCond(s.when, `${d.id} strip/idle/blank`, d.engines); if (s.action) checkAction(s.action, `${d.id} strip`, d.engines);
  }
  if (d.kind === 'pedal' || d.kind === 'unit') {
    /* a pedal's and a unit's controls are laid out from its parameters (docs/05, "Pedals and side units"): this is
       the part each one gets, so the inventory counts it as used */
    for (const p of d.params) {
      shownParams.add(p.key); if (p.key === d.on) continue;
      const n = Array.isArray(p.options) ? p.options.length : 99;
      const part = p.type === 'number' ? 'knob' : p.type === 'boolean' ? 'ledButton' : p.type === 'text' ? 'textField' : d.kind === 'unit' && n <= 4 ? 'chips' : 'pick';
      if (!partUse.has(part)) partUse.set(part, new Set()); partUse.get(part)!.add(d.id + ' (laid out)');
    }
    const sw = d.on ? params.get(d.on) : null;
    if (d.on && (!sw || sw.d !== d || sw.p.type !== 'boolean')) err(d.id, `its switch "${d.on}" is not one of its own boolean parameters`);
  }
  if (d.kind === 'pedal' && d.main) { const m = params.get(d.main); if (!m || m.d !== d || m.p.type !== 'number') err(d.id, `its main knob "${d.main}" is not one of its own numbers`); }
  if (d.kind === 'pedal') (d.face || []).forEach((n, i) => { if (!n.tiers || n.tiers.includes('quick')) err(`${d.id} face#${i}`, 'a pedal face node must be tiered studio or lab'); });
  if (d.kind === 'module' && d.per) Object.keys(d.per).forEach(e => { if (!d.engines.includes(e as EngineId)) err(d.id, `overrides engine "${e}", which it is not on`); });
  if (!d.engines.length) err(d.id, 'is on no engine');
  d.engines.forEach(e => { if (!ENGINES.includes(e)) err(d.id, `unknown engine "${e}"`); });
  if (!d.thumb) { if (!(d.kind === 'module' && (d.builtIn || d.source))) err(d.id, 'has no thumbnail, and it is offered in the Modules panel and the picker'); continue; }
  const files = [d.thumb.file, ...Object.values(d.thumb.per || {})];
  files.forEach(f => { if (!fs.existsSync(path.join(DEFDIR, 'thumbs', f + '.svg'))) err(d.id, `thumbnail thumbs/${f}.svg does not exist`); });
  Object.keys(d.thumb.per || {}).forEach(e => { if (!d.engines.includes(e as EngineId)) err(d.id, `thumbnail for engine "${e}", which it is not on`); });
}
const drawn = new Set(DEFINITIONS.flatMap(d => d.thumb ? [d.thumb.file, ...Object.values(d.thumb.per || {})] : []));
for (const f of fs.readdirSync(path.join(DEFDIR, 'thumbs'))) if (!drawn.has(f.replace(/\.svg$/, ''))) err(`thumbs/${f}`, 'no definition draws with it');
for (const [k, { d }] of params) if (!shownParams.has(k) && d.kind === 'module') err(`${d.id} › ${k}`, 'no face shows this parameter at any tier, so nothing can change it or say it is changed');

// ---------------------------------------------------------------- every field of the format is used
const program = ts.createProgram([path.join(DEFDIR, 'types.ts')], { allowImportingTsExtensions: true, noEmit: true });
const src = program.getSourceFile(path.join(DEFDIR, 'types.ts'))!;
const declared = new Map<string, string>();
ts.forEachChild(src, n => {
  const add = (owner: string, members: ts.NodeArray<ts.TypeElement>) => members.forEach(m => { if (m.name && ts.isIdentifier(m.name)) declared.set(`${owner}.${m.name.text}`, m.name.text); });
  if (ts.isInterfaceDeclaration(n)) add(n.name.text, n.members);
  if (ts.isTypeAliasDeclaration(n)) { const visit = (t: ts.Node): void => { if (ts.isTypeLiteralNode(t)) add(n.name.text, t.members); ts.forEachChild(t, visit); }; visit(n.type); }
});
const usedNames = new Set<string>();
const collect = (v: unknown): void => { if (Array.isArray(v)) v.forEach(collect); else if (v && typeof v === 'object') for (const [k, x] of Object.entries(v)) { usedNames.add(k); collect(x); } };
DEFINITIONS.forEach(collect);
const unusedFields = [...declared].filter(([, name]) => !usedNames.has(name)).map(([q]) => q);
unusedFields.forEach(q => err('types.ts', `field ${q} is used by no definition`));
const unusedParts = Object.keys(PARTS).filter(p => !partUse.has(p));
unusedParts.forEach(p => err('parts.ts', `part "${p}" is used by no definition`));

// ---------------------------------------------------------------- report
const kinds = { module: 0, pedal: 0, unit: 0 } as Record<string, number>; DEFINITIONS.forEach(d => kinds[d.kind]!++);
const wireCount = { found: 0, none: 0 }; const channels: Record<string, number> = {};
for (const { p, d } of params.values()) for (const e of engOf(p, d)) { const w = resolve(p, e).wire; if (typeof w === 'object' && 'none' in w) wireCount.none++; else { wireCount.found++; const ch = (typeof w === 'string' ? w : w.to).split(':')[0]!; channels[ch] = (channels[ch] || 0) + 1; } }
const noneParams = [...params.values()].filter(({ p, d }) => engOf(p, d).some(e => { const w = resolve(p, e).wire; return typeof w === 'object' && 'none' in w; })).map(({ p }) => p.key);
console.log(`definitions: ${DEFINITIONS.length} (${kinds.module} modules, ${kinds.pedal} pedals, ${kinds.unit} side units)`);
console.log(`parameters: ${params.size}; parameter-engine pairs: ${wireCount.found + wireCount.none} (${wireCount.found} with a real name: ${Object.entries(channels).map(([c, n]) => `${c} ${n}`).join(', ')}; ${wireCount.none} with none)`);
console.log(`parameters with no real counterpart on at least one engine: ${noneParams.length} (${noneParams.join(', ')})`);
console.log(`state entries: ${states.size}; parts in the inventory: ${Object.keys(PARTS).length}, used: ${partUse.size}; format fields: ${declared.size}, unused: ${unusedFields.length}`);
if (process.argv.includes('--usage')) {
  console.log('\npart usage:');
  for (const p of Object.keys(PARTS)) console.log(`  ${p.padEnd(15)} ${[...(partUse.get(p) || [])].join(', ')}`);
}
if (process.argv.includes('--inventory')) {
  /* the inventory as the markdown table in docs/01-parts-inventory.md */
  const cell = (s: string) => s.replace(/\|/g, '\\|').replace(/\n/g, ' ');
  console.log('\n| Part | Kind | Binds | What it is for | Options | States | Styles (concept src/css) | Used by |\n| --- | --- | --- | --- | --- | --- | --- | --- |');
  for (const [name, p] of Object.entries(PARTS) as [string, PartSpec][]) {
    const opts = Object.entries(p.opts || {}).map(([k, v]) => `\`${k}\`: ${v}`).join('<br>');
    console.log(`| \`${name}\` | ${p.kind} | ${p.binds.join(', ')} | ${cell(p.doc + (p.responsible ? ` **Responsible for:** ${p.responsible}` : ''))} | ${cell(opts)} | ${cell((p.states || []).join('; '))} | ${p.css.join(', ')} | ${[...(partUse.get(name) || [])].join(', ')} |`);
  }
}
if (process.argv.includes('--names')) {
  /* the concept-to-real-name table in docs/04-parameter-names.md */
  const SHORT: Record<string, string> = { ace: 'ace', 'minimax-m3': 'mm3', yue2: 'yue2', sa3: 'sa3', heartmula: 'hm' };
  const say = (w: WireRef): string => typeof w === 'string' ? '`' + w + '`' : 'none' in w ? `none: ${w.none}`
    : '`' + w.to + '`' + (w.invert ? ' (inverted)' : '') + (w.scale ? ` (× ${+w.scale.toFixed(4)})` : '') + (w.auto !== undefined ? ` (auto: ${JSON.stringify(w.auto)})` : '') + `. ${w.note}`;
  for (const d of DEFINITIONS) {
    if (!d.params.length) continue;
    console.log(`\n### ${d.name} (\`${d.id}\`, ${d.kind})\n\n| Concept key | Type | Engines | Real name |\n| --- | --- | --- | --- |`);
    for (const p of d.params) {
      const engs = engOf(p, d), per = engs.map(e => [e, resolve(p, e).wire] as const), one = new Set(per.map(([, w]) => JSON.stringify(w))).size === 1;
      const real = one ? say(per[0]![1]) : per.map(([e, w]) => `${SHORT[e]}: ${say(w)}`).join('<br>');
      console.log(`| \`${p.key}\` | ${p.type} | ${engs.length === ENGINES.length ? 'all' : engs.map(e => SHORT[e]).join(', ')} | ${real.replace(/\|/g, '\\|')} |`);
    }
  }
}
if (notes.length) { console.log(`\nnotes (${notes.length}):`); notes.forEach(n => console.log('  ' + n)); }
if (errors.length) { console.log(`\nerrors (${errors.length}):`); errors.forEach(e => console.log('  ' + e)); process.exit(1); }
console.log('\nOK: no errors.');
