// The request built from the definitions, compared field by field with the one the existing Create page sends for
// the same intent, over the value space of every engine. Both sides end in the server's own resolveCreateIntent()
// (what /api/resolve/preview runs on what Create submits), and both the request and the intent's `compose` and
// `captionSource` are compared.
//
//   node --import ./tools/real-app/load.mjs tools/compare-request.ts            fails on a difference not accepted
//                                                                               in tools/fixtures/request-differences.json
//   ... tools/compare-request.ts --print     every difference class, with an example state
//   ... tools/compare-request.ts --quick     the targeted states only, without the random ones
//
// (npm run request). The definitions' side runs derive.ts createIntent(). The existing side holds the same intent in
// the existing page's stores as tools/existing-ui.ts says its controls hold it, and runs the existing code
// (tools/real-app/app.ts). The states, per engine:
//   - every parameter at its minimum, its maximum and a value off its default, each option of every choice, each
//     switch both ways, auto on and off, text set and empty, each with its conditions made to hold
//   - every parameter with a condition set off its default while the condition does not hold
//   - every module, pedal and side unit taken off the desk with its values changed
//   - the module state: adapters with phase windows, a kept seed, files
//   - 300 random combinations from a fixed seed
// A difference is either a defect in a definition, fixed there, or listed in the fixture with a reason. A listed
// difference that no longer occurs fails the run too, so the list stays true.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFINITIONS } from '../src/definitions/index.ts';
import { ENGINES, type EngineId } from '../src/definitions/engines.ts';
import { CHECKS } from '../src/definitions/parts.ts';
import type { Condition, Json, ParamDef, Value, WireKey, WireMap } from '../src/definitions/types.ts';
import { applies, createIntent, factsOf, has, holds, indexOf, isAuto, liveDefs, resolve, valueOf, wiresOf, type Desk } from '../src/derive.ts';
import { REAL, SELECTION, existingIntent, resolveCreateIntent, type Existing } from './real-app/app.ts';
import { CAPTION_SOURCE, CONTINUATION, HOLDINGS, PACKED, STATE_HOLDINGS, type Hold } from './existing-ui.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const EXPECT_FILE = path.join(ROOT, 'tools', 'fixtures', 'request-differences.json');
const ix = indexOf(DEFINITIONS);
const J = (x: unknown) => JSON.stringify(x);
const QUICK = process.argv.includes('--quick');

/* ================================================================ the states, in the definitions' terms */

interface Case { name: string; cause: string; desk: Desk }
const ALL = new Set(DEFINITIONS.map(d => d.id));
const base = (e: EngineId): Desk => ({ engine: e, tier: 'lab', vals: {}, auto: {}, state: {}, facts: {}, on: new Set(ALL), selection: { ...SELECTION } });
const clone = (d: Desk): Desk => ({ ...d, vals: { ...d.vals }, auto: { ...d.auto }, state: structuredClone(d.state), facts: { ...d.facts }, on: new Set(d.on), selection: { ...d.selection } });
const onGrid = (x: number, step: number) => +x.toFixed(Math.max(0, -Math.floor(Math.log10(step)) + 2));

/** The values a parameter is tried at on an engine. */
function tries(p: ParamDef, e: EngineId): Value[] {
  const f = resolve(p, e);
  if (p.type === 'boolean') return [!f.default, f.default];
  if (p.type === 'number') {
    const [lo, hi, step] = f.range!, d = f.default as number;
    const off = d + step <= hi ? d + step : d - step;
    return [...new Set([lo, hi, onGrid(off, step)])];
  }
  if (p.type === 'choice') {
    if (Array.isArray(f.options)) return f.options.map(o => o.value);
    const inc = (f.options!.include || []).map(o => o.value), from = f.options!.from;
    const fx = from.startsWith('plugins.') ? (REAL_FIX.plugins[from.slice(8)] ?? []) as string[] : [];
    return [...inc, ...(f.options!.only ?? fx).slice(0, 4), ...(!inc.length && !fx.length ? ['picked-model.gguf'] : [])];
  }
  const t = f.check ? sampleText(f.check.rule) : f.example ?? 'x';
  return [t, ''];
}
const REAL_FIX = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools', 'fixtures', 'real-app.json'), 'utf8'));
function sampleText(rule: keyof typeof CHECKS): string {
  return rule === 'timesteps' ? '1,0.6,0.2,0' : '1234';
}

/** The values a condition's names can take, to find a desk where it holds or does not. */
function* settings(c: Condition, e: EngineId): Generator<{ vals: Record<string, Value>; auto: Record<string, boolean>; state: Record<string, Json>; on: string[] }> {
  const keys = new Set<string>(), states = new Set<string>(), desks = new Set<string>();
  const walk = (x: Condition): void => {
    if ('all' in x) x.all.forEach(walk); else if ('any' in x) x.any.forEach(walk); else if ('not' in x && !('param' in x)) walk(x.not);
    else if ('param' in x) keys.add(x.param); else if ('state' in x) states.add(x.state); else if ('onDesk' in x) desks.add(x.onDesk);
  };
  walk(c);
  const axes: { set: (o: any, v: unknown) => void; values: unknown[] }[] = [];
  for (const k of keys) {
    const hit = ix.params.get(k); if (!hit || !has(ix, k, e)) continue;
    const f = resolve(hit.p, e), thr: number[] = [];
    const scan = (x: Condition): void => { if ('all' in x) x.all.forEach(scan); else if ('any' in x) x.any.forEach(scan); else if ('not' in x && !('param' in x)) scan(x.not); else if ('param' in x && x.param === k) { [x.above, x.below].forEach(b => { if (b !== undefined) thr.push(b + (f.range?.[2] ?? 0.01), b - (f.range?.[2] ?? 0.01)); }); [x.is, x.not].flat().forEach(v => { if (v !== undefined && typeof v !== 'object') thr.push(v as number); }); } };
    scan(c);
    const vals = hit.p.type === 'number' ? [...new Set([...tries(hit.p, e), ...thr.filter(t => typeof t === 'number' && f.range && t >= f.range[0] && t <= f.range[1])])] : tries(hit.p, e);
    axes.push({ set: (o, v) => { o.vals[k] = v; }, values: vals });
    if (f.auto) axes.push({ set: (o, v) => { o.auto[k] = v; }, values: [false, true] });
  }
  for (const s of states) axes.push({ set: (o, v) => { if (s === 'adapters' || s === 'slots') o.state.slots = slots(v as number); else if (ix.states.has(s)) o.state[s] = v ? stateSample(s) : null; }, values: [0, 1, 2, 3] });
  for (const d of desks) axes.push({ set: (o, v) => { if (v) o.on.push(d); else o.off = [...(o.off || []), d]; }, values: [true, false] });
  let n = 1; for (const a of axes) n *= a.values.length;
  const idx = axes.map(() => 0);
  for (let i = 0; i < Math.min(n, 4096); i++) {
    const o: any = { vals: {}, auto: {}, state: {}, on: [] };
    axes.forEach((a, j) => a.set(o, a.values[idx[j]!]));
    yield o;
    for (let j = 0; j < axes.length; j++) { idx[j]!++; if (idx[j]! < axes[j]!.values.length) break; idx[j] = 0; }
  }
}
const TRACKS: Json = [{ title: 'Neon Rain', bpm: 96, caption: 'synthwave, female vocal, rain' }, { title: 'Low Tide', bpm: 128, caption: 'house, warm bass, male vocal' }];
const slots = (n: number, ph = false): Json => Array.from({ length: n }, (_, i) => ({ file: `C:/adapters/ad${i}.safetensors`, type: 'LoRA', ...(ph && i === 1 ? { ph: [60, 100] } : {}) }));
function stateSample(key: string): Json {
  const s = ix.states.get(key)!.s;
  if (s.kind === 'file') return `C:/refs/${key}.wav`;
  if (s.kind === 'seed') return 123456;
  if (s.kind === 'span') return { start: 12, end: 40 };
  if (s.kind === 'slots') return slots(2, true);
  if (key === 'capTracks') return TRACKS;
  return null;
}
/** A desk where every condition of `c` is made to hold (or not), starting from `desk`. */
function withCondition(desk: Desk, c: Condition | undefined, want: boolean): Desk | null {
  if (!c) return want ? desk : null;
  for (const s of settings(c, desk.engine)) {
    const d = clone(desk);
    Object.assign(d.vals, s.vals); Object.assign(d.auto, s.auto); Object.assign(d.state, s.state);
    s.on.forEach(x => d.on.add(x)); ((s as any).off || []).forEach((x: string) => d.on.delete(x));
    d.facts = factsOf(ix, { ...d, facts: {} });
    if (holds(ix, c, d) === want) return d;
  }
  return null;
}

function* cases(e: EngineId): Generator<Case> {
  const b = base(e);
  yield { name: 'defaults, everything on the desk', cause: '-', desk: b };
  const params = [...ix.params.values()].filter(({ p, d }) => has(ix, p.key, e) && !(d.kind === 'module' && d.source));
  for (const { p } of params) {
    const f = resolve(p, e);
    /* each condition a wire of it sends under (the High DCW knob goes to another field) */
    /* (the conditions come from the wires and from how the existing page holds it, so a wire that lost its
       condition is still tried where the condition matters) */
    const whens = [...wiresOf(f.wire).flatMap(w => (typeof w === 'object' && w.when ? [{ when: w.when, to: w.to as string }] : [])), ...(holdsOf(p.key, e) || []).flatMap(h => (h.when ? [{ when: h.when, to: h.to as string }] : []))];
    for (const w of whens) {
      const on = withCondition(b, f.when ? { all: [f.when, w.when] } : w.when, true);
      if (on) for (const lit of [false, true]) {
        /* and with every switch of its definition on, so a group's power switch does not hide what the wire does */
        const d = clone(on); d.vals[p.key] = tries(p, e)[0]!; if (f.auto) d.auto[p.key] = false;
        if (lit) for (const q of ix.params.get(p.key)!.d.params) if (q.type === 'boolean' && has(ix, q.key, e) && !(q.key in d.vals)) d.vals[q.key] = true;
        d.facts = factsOf(ix, { ...d, facts: {} });
        if (applies(ix, p.key, d)) yield { name: `${p.key} = ${J(d.vals[p.key])} while its wire to ${w.to} holds${lit ? ', its switches on' : ''}`, cause: p.key, desk: d };
      }
    }
    const ready = withCondition(b, f.when, true);
    if (ready) {
      for (const v of tries(p, e)) {
        const d = clone(ready); d.vals[p.key] = v; if (f.auto) d.auto[p.key] = false; d.facts = factsOf(ix, { ...d, facts: {} });
        if (applies(ix, p.key, d)) yield { name: `${p.key} = ${J(v)}`, cause: p.key, desk: d };
      }
      if (f.auto) { const d = clone(ready); d.auto[p.key] = true; d.vals[p.key] = tries(p, e)[1]!; yield { name: `${p.key} on auto`, cause: p.key, desk: d }; }
    }
    if (f.when) {
      const off = withCondition(b, f.when, false);
      if (off) { const d = clone(off); d.vals[p.key] = tries(p, e)[2] ?? tries(p, e)[0]!; if (f.auto) d.auto[p.key] = false; d.facts = factsOf(ix, { ...d, facts: {} }); yield { name: `${p.key} set while its condition does not hold`, cause: p.key, desk: d }; }
    }
  }
  /* each definition off the desk, its values changed */
  for (const d of DEFINITIONS.filter(x => x.engines.includes(e) && !(x.kind === 'module' && (x.builtIn || x.source)))) {
    const k = clone(b); k.on.delete(d.id);
    for (const p of d.params) if (has(ix, p.key, e)) k.vals[p.key] = tries(p, e)[0]!;
    k.facts = factsOf(ix, { ...k, facts: {} });
    yield { name: `${d.id} off the desk, its values changed`, cause: d.id, desk: k };
  }
  /* module state */
  for (const n of [1, 2, 3]) for (const adv of ['simple', 'advanced']) for (const ph of [false, true]) {
    if (!has(ix, 'ad.adv', e) && adv === 'advanced') continue;
    const d = clone(b); d.state.slots = slots(n, ph); if (has(ix, 'ad.adv', e)) d.vals['ad.adv'] = adv; d.vals.s1 = 0.5; d.facts = factsOf(ix, { ...d, facts: {} });
    yield { name: `${n} adapters, ${adv}${ph ? ', the second in a phase window' : ''}`, cause: 'slots', desk: d };
  }
  for (const lock of [true, false]) { const d = clone(b); d.state.seed = 123456; d.vals['seed.lock'] = lock; yield { name: `seed 123456, kept ${lock}`, cause: 'seed', desk: d }; }
  if (has(ix, 'cap.src', e)) for (const src of ['custom', 'auto', 'Low Tide']) for (const text of ['', 'my own caption']) {
    const d = clone(b); d.state.capTracks = TRACKS; d.vals['cap.src'] = src; d.vals['prompt.text'] = text; d.vals['tempo.bpm'] = 126; d.auto['tempo.bpm'] = false;
    yield { name: `caption source ${src}, caption ${J(text)}`, cause: 'capTracks', desk: d };
  }
  for (const { s } of ix.states.values()) if (s.kind === 'file' || s.kind === 'span') { const d = clone(b); d.state[s.key] = stateSample(s.key); if (s.key === 'contSel') d.state.contFile = stateSample('contFile'); yield { name: `${s.key} set`, cause: s.key, desk: d }; }
  if (QUICK) return;
  /* random combinations */
  let seed = 0x5eed + ENGINES.findIndex(x => x.id === e);
  const rnd = () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const pick = <T>(a: readonly T[]) => a[Math.floor(rnd() * a.length)]!;
  for (let i = 0; i < 300; i++) {
    const d = base(e);
    for (const x of DEFINITIONS) if (!(x.kind === 'module' && x.builtIn) && rnd() < 0.25) d.on.delete(x.id);
    for (const { p } of params) { if (rnd() < 0.3) d.vals[p.key] = pick(tries(p, e)); if (resolve(p, e).auto && rnd() < 0.3) d.auto[p.key] = rnd() < 0.5; }
    if (rnd() < 0.5) d.state.slots = slots(Math.floor(rnd() * 4), rnd() < 0.5);
    if (rnd() < 0.5) d.state.seed = Math.floor(rnd() * 4294967295);
    for (const { s } of ix.states.values()) if ((s.kind === 'file' || s.kind === 'span') && rnd() < 0.3) d.state[s.key] = stateSample(s.key);
    d.facts = factsOf(ix, d);
    yield { name: `random #${i + 1}`, cause: 'random', desk: d };
  }
}

/* ================================================================ the same intent, held by the existing page */

const holdsOf = (key: string, e: EngineId): Hold[] | null => {
  const h = HOLDINGS[key];
  if (!h) return null;
  if (Array.isArray(h) && h.length && 'engines' in (h[0] as object)) {
    const line = (h as { engines: EngineId[]; hold: Hold | Hold[] }[]).find(x => x.engines.includes(e));
    return line ? [line.hold].flat() : null;
  }
  return [h as Hold | Hold[]].flat();
};
const isPlain = (w: WireKey | WireMap) => typeof w === 'string' || Object.keys(w).every(k => k === 'to' || k === 'note' || k === 'differs');
const missingHoldings = new Set<string>();
/** How the existing page holds a parameter on an engine: its line in existing-ui.ts, or its wire when that is a
 *  plain name. */
function holdingFor(p: ParamDef, e: EngineId): Hold[] {
  const listed = holdsOf(p.key, e);
  if (listed) return listed;
  const ws = wiresOf(resolve(p, e).wire);
  if (!ws.every(isPlain)) { missingHoldings.add(`${p.key} [${e}]`); return []; }
  return ws.flatMap((w): Hold[] => {
    const to = typeof w === 'string' ? w : w.to, ch = to.slice(0, to.indexOf(':')), name = to.slice(to.indexOf(':') + 1);
    if (ch === 'param') return [{ to: `store:${name}` }];
    if (ch === 'ext') return [{ to: `store:backendParams.${name}` }];
    if (ch === 'content') return [{ to: `content:${name}` }];
    if (ch === 'intent' && name.startsWith('compose.')) return [{ to: `compose:${name.slice(8)}` }];
    return [];
  });
}
function holdValue(h: Hold, desk: Desk, v: Value | undefined, auto: boolean): Json | undefined {
  if (auto) return 'auto' in h ? h.auto as Json : undefined;
  if (h.when && !holds(ix, h.when, desk)) return 'else' in h ? h.else as Json : undefined;
  if (h.from) v = valueOf(ix, h.from, desk);
  if (v === undefined || (h.leave || []).includes(v)) return undefined;
  let x: Json | undefined = v;
  if (h.map) x = String(v) in h.map ? h.map[String(v)] : undefined;
  if (h.invert && typeof x === 'boolean') x = !x;
  if (h.times !== undefined && typeof x === 'number') x = +(x * h.times).toFixed(10);
  if (h.as === 'text' && x !== undefined) x = String(x);
  if (h.as === 'number' && typeof x === 'string') x = /^-?\d+$/.test(x.trim()) ? Number(x.trim()) : undefined;
  return x;
}
/** The existing page in the state that means the same as the desk. `applying`: hold only the values that apply (the
 *  existing page holds every value the user set; holding only these tells a dead value from a wrong one). */
function existingOf(desk: Desk, applying = false): Existing {
  const e = desk.engine, x: Existing = { store: [], content: {}, panel: {}, selection: { ...desk.selection } };
  const set = (to: Hold['to'], v: Json) => {
    const ch = to.slice(0, to.indexOf(':')), name = to.slice(to.indexOf(':') + 1);
    if (ch === 'store') x.store.push([name, v]);
    else if (ch === 'content') x.content[name] = v;
    else if (ch === 'compose') (x.panel as Record<string, Json>)[name] = v;
    else if (ch === 'selection') x.selection[name] = String(v);
  };
  const live = liveDefs(ix, desk);
  for (const d of live) for (const s of d.state || []) { const f = STATE_HOLDINGS[s.key]; if (f && desk.state[s.key] !== undefined && desk.state[s.key] !== null) f(desk.state[s.key]!, e).forEach(([to, v]) => set(to, v)); }
  const tracks = Array.isArray(desk.state.capTracks) ? desk.state.capTracks as { title: string; bpm?: number; caption: string }[] : [];
  const cs = has(ix, 'cap.src', e) ? CAPTION_SOURCE(e, { source: String(valueOf(ix, 'cap.src', desk)), dataset: String(valueOf(ix, 'cap.ds', desk) ?? ''), caption: String(valueOf(ix, 'prompt.text', desk) ?? ''), tracks, adapter: Array.isArray(desk.state.slots) && desk.state.slots.length > 0 }) : undefined;
  if (cs !== undefined) x.captionSource = cs as Record<string, Json>;
  if (live.some(d => d.id === 'cont')) { const c = CONTINUATION(desk.state.contFile, desk.state.contSel); if (c !== undefined) set('content:yue2Continuation', c); }
  const packedKeys = new Set(PACKED.filter(p => p.engines.includes(e)).flatMap(p => p.keys));
  for (const d of live) for (const p of d.params) {
    if (!has(ix, p.key, e) || packedKeys.has(p.key)) continue;
    if (!(p.key in desk.vals) && !(p.key in desk.auto)) continue;
    if (applying && !applies(ix, p.key, desk)) continue;
    const v = valueOf(ix, p.key, desk), auto = isAuto(ix, p.key, desk);
    for (const h of holdingFor(p, e)) { const hv = holdValue(h, desk, v, auto); if (hv !== undefined) set(h.to, hv); }
  }
  for (const pk of PACKED) {
    if (!pk.engines.includes(e) || !pk.keys.some(k => k in desk.vals && live.includes(ix.params.get(k)!.d))) continue;
    const vals = Object.fromEntries(pk.keys.map(k => [k, has(ix, k, e) && (!applying || applies(ix, k, desk)) ? valueOf(ix, k, desk) : undefined]));
    const v = pk.make(vals); if (v !== undefined) set(pk.to, v);
  }
  /* a pedal or side unit not on the board is off */
  for (const d of DEFINITIONS) {
    if (d.kind === 'module' || !d.on || !d.engines.includes(e) || desk.on.has(d.id) || !has(ix, d.on, e)) continue;
    for (const h of holdingFor(ix.params.get(d.on)!.p, e)) { const hv = holdValue(h, desk, false, false); if (hv !== undefined) set(h.to, hv); }
  }
  return x;
}

/* ================================================================ compare */

const flatten = (o: Record<string, unknown>, prefix = ''): Record<string, Json> =>
  Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined).map(([k, v]) => [prefix + k, v as Json]));
function sides(desk: Desk, applying = false): { ours: Record<string, Json>; theirs: Record<string, Json> } {
  const e = desk.engine;
  const mine = createIntent(ix, desk, REAL), theirs = existingIntent(e, existingOf(desk, applying));
  /* compose as the server reads it: an absent switch is its default (resolveIntent.ts, composeCreateCaption) */
  const COMPOSE = { autoExpand: false, loraTrigger: '', beatIntro: false, introBars: 2 };
  /* the tracks a caption source picks from, as the server loads them for the intent (loadIntentData.ts) */
  const tracks = Array.isArray(desk.state.capTracks) ? desk.state.capTracks as { title: string; bpm: number; caption: string }[] : [];
  const data = { mm3Tracks: tracks, yue2Tracks: tracks.map(t => ({ name: t.title, caption: t.caption, bpm: t.bpm })) };
  const run = (i: { params: unknown; compose: unknown; captionSource?: unknown }) => {
    const r = JSON.parse(J(resolveCreateIntent(i, e, data).request)) as Record<string, Json>;
    return { ...flatten(r), ...flatten({ ...COMPOSE, ...(i.compose ?? {}) as Record<string, unknown> }, 'compose.'), ...flatten((i.captionSource ?? {}) as Record<string, unknown>, 'captionSource.') };
  };
  return { ours: run(mine), theirs: run(theirs) };
}

interface Diff { engine: EngineId; field: string; kind: string; existing?: Json; definitions?: Json; state: string; cause: string }
interface Accepted { engines: EngineId[] | '*'; fields: string[]; kind: string; why: string }
const expected: Accepted[] = fs.existsSync(EXPECT_FILE) ? JSON.parse(fs.readFileSync(EXPECT_FILE, 'utf8')).differences : [];
const matches = (a: Accepted, d: Diff) => (a.engines === '*' || a.engines.includes(d.engine)) && a.fields.includes(d.field) && a.kind === d.kind;

const SHOW = process.argv.indexOf('--show');
if (SHOW > 0) {
  /* --show <engine> <words of a state's name>: both sides of the first state whose name holds the words */
  const e = process.argv[SHOW + 1] as EngineId, words = process.argv[SHOW + 2] ?? '';
  for (const c of cases(e)) if (c.name.includes(words)) {
    const { ours, theirs } = sides(c.desk);
    console.log(`${c.name}
  desk: ${J({ vals: c.desk.vals, auto: c.desk.auto, state: c.desk.state, off: [...ALL].filter(x => !c.desk.on.has(x)) })}
  existing page holds: ${J(existingOf(c.desk))}`);
    for (const k of [...new Set([...Object.keys(ours), ...Object.keys(theirs)])].sort()) console.log(`  ${J(ours[k]) === J(theirs[k]) ? ' ' : '*'} ${k}: existing ${J(theirs[k])} | definitions ${J(ours[k])}`);
    process.exit(0);
  }
  console.log('no such state'); process.exit(1);
}

const all: Diff[] = [], used = new Set<Accepted>();
const report: string[] = [];
for (const { id: e } of ENGINES) {
  let states = 0, fields = 0, found = 0;
  for (const c of cases(e)) {
    states++;
    const { ours, theirs } = sides(c.desk);
    const keys = [...new Set([...Object.keys(ours), ...Object.keys(theirs)])];
    fields += keys.length;
    const differ = keys.filter(k => J(ours[k]) !== J(theirs[k]));
    if (!differ.length) continue;
    const settled = sides(c.desk, true).theirs;
    for (const k of differ) {
      found++;
      const kind = J(ours[k]) === J(settled[k]) ? 'existing sends a value that does not apply'
        : !(k in ours) ? 'only the existing page sends it' : !(k in theirs) ? 'only the definitions send it' : 'value differs';
      const d: Diff = { engine: e, field: k, kind, ...(k in theirs ? { existing: theirs[k] } : {}), ...(k in ours ? { definitions: ours[k] } : {}), state: c.name, cause: c.cause };
      const hit = expected.find(a => matches(a, d)); if (hit) used.add(hit);
      all.push(d);
    }
  }
  report.push(`  ${e}: ${states} states, ${fields} fields compared, ${found} differences`);
}

const fresh = all.filter(d => !expected.some(a => matches(a, d)));
const classes = new Map<string, Diff[]>();
for (const d of fresh) { const k = `${d.engine} | ${d.field} | ${d.kind}`; classes.set(k, [...(classes.get(k) || []), d]); }
const gone = expected.filter(a => !used.has(a));

if (process.argv.includes('--print')) {
  const byClass = new Map<string, Diff[]>();
  for (const d of all) { const k = `${d.engine} | ${d.field} | ${d.kind}`; byClass.set(k, [...(byClass.get(k) || []), d]); }
  for (const [k, list] of byClass) {
    const a = expected.find(x => matches(x, list[0]!)), ex = list[0]!;
    console.log(`${k}: ${list.length} states${a ? ' (accepted)' : ''}; causes ${[...new Set(list.map(d => d.cause))].slice(0, 6).join(', ')}\n    e.g. ${ex.state}: existing ${J(ex.existing)} | definitions ${J(ex.definitions)}`);
  }
}
console.log(`request over the value space, ${ENGINES.length} engines${QUICK ? ' (targeted states only)' : ''}:\n${report.join('\n')}`);
console.log(`  ${all.length} differences in ${new Set(all.map(d => `${d.engine}|${d.field}|${d.kind}`)).size} classes; ${all.length - fresh.length} accepted by ${used.size} entries of tools/fixtures/request-differences.json`);
let bad = false;
if (missingHoldings.size) { bad = true; console.log(`  tools/existing-ui.ts says nothing of how the existing page holds ${[...missingHoldings].join(', ')}, whose wires are more than a plain name: add a line`); }
if (classes.size) { bad = true; for (const [k, list] of classes) console.log(`  new: ${k} (${list.length} states; e.g. ${list[0]!.state}: existing ${J(list[0]!.existing)}, definitions ${J(list[0]!.definitions)})`); }
if (gone.length && !QUICK) { bad = true; gone.forEach(a => console.log(`  gone: ${a.engines === '*' ? 'every engine' : a.engines.join(', ')} ${a.fields.join(', ')} (${a.kind}): the list accepts a difference that no longer occurs`)); }
process.exit(bad ? 1 : 0);
