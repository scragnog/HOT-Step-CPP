// The request built from the definitions, compared field by field with the one the existing Create page sends when a
// user does the same things on it, over the value space of every engine. Both end in the server's own
// resolveCreateIntent() (what /api/resolve/preview runs on what Create submits); the request and the intent's
// `compose` and `captionSource` are compared.
//
// The definitions' side is derive.ts createIntent(). The existing side is the existing page itself
// (tools/real-app/app.ts, page.ts): its bar and CreatePanel are drawn in Node over its real stores, each value is set
// by using the control a user would use (tools/existing-ui.ts says which, by its place and words on the page), and
// Generate is pressed. What the definitions' wires say is never read on the existing side.
//
//   node --import ./tools/real-app/load.mjs tools/compare-request.ts             fails on a difference that is not one
//                                                                                tools/fixtures/request-differences.json
//                                                                                names exactly
//   ... --quick     the targeted states only, without the random ones      ... --print   every difference class
//   ... --live      drive the existing page even when its recording stands  ... --record  drive it and record it
//   ... --show <engine> <words>   both sides of the first state whose name holds the words
//
// (npm run request). The existing page's requests are recorded in tools/fixtures/existing-requests.json with the
// commit and a fingerprint of the files they come from. When those files are as recorded, the recording is used (the
// everyday check); when they changed, the page is driven live and a recorded request it no longer sends fails the run
// until `npm run record` records it again on purpose; when they are gone (the existing frontend removed), the
// recording stands for them.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { Session } from 'node:inspector/promises';
import { DEFINITIONS } from '../src/definitions/index.ts';
import { ENGINES, type EngineId } from '../src/definitions/engines.ts';
import type { Condition, Json, ParamDef, Value } from '../src/definitions/types.ts';
import { applies, createIntent, has, holds, indexOf, isAuto, resolve, valueOf, wiresOf, toOf, type Desk } from '../src/derive.ts';
import { CAPTION_SOURCE, LINKS, RESTATED, STATE_HOLDINGS, type Holding, type Link } from './existing-ui.ts';
import type { Existing } from './real-app/app.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const EXPECT_FILE = path.join(ROOT, 'tools', 'fixtures', 'request-differences.json');
const RECORD_FILE = path.join(ROOT, 'tools', 'fixtures', 'existing-requests.json');
const ix = indexOf(DEFINITIONS);
const J = (x: unknown) => JSON.stringify(x);
const arg = (a: string) => process.argv.includes(a);
const QUICK = arg('--quick'), LIVE = arg('--live'), RECORD = arg('--record');

/* the transforms' branches the run reaches: V8's block coverage of src/definitions/transforms.ts */
const coverage = new Session(); coverage.connect();
await coverage.post('Profiler.enable');
await coverage.post('Profiler.startPreciseCoverage', { callCount: true, detailed: true });
const { REAL, SELECTION, APP, FIX, resolveCreateIntent } = await import('./real-app/real.ts');

/* ================================================================ which run: live, or the recording */

/** The files the existing page's request comes from: a change in any of them stops the recording standing for it. */
const SOURCES = ['ui/src/components/global-bar', 'ui/src/components/create', 'ui/src/stores', 'ui/src/utils', 'ui/src/context', 'ui/src/App.tsx', 'ui/src/components/settings/SettingsPanel.tsx'];
function fingerprint(): string | null {
  if (!fs.existsSync(path.join(APP, 'ui', 'src'))) return null;
  const h = crypto.createHash('sha1');
  const add = (p: string) => {
    const full = path.join(APP, p);
    if (!fs.existsSync(full)) return;
    if (fs.statSync(full).isDirectory()) { for (const f of fs.readdirSync(full).sort()) if (!/\.test\./.test(f)) add(path.join(p, f)); return; }
    h.update(p.split(path.sep).join('/')); h.update(fs.readFileSync(full, 'utf8').replace(/\r\n/g, '\n'));
  };
  SOURCES.forEach(add);
  return h.digest('hex').slice(0, 16);
}
/** The recording: per engine a base request, and per state what differs from it (fields set, fields absent). */
interface Recording { note: string; commit: string; fingerprint: string; base: { [engine: string]: { [field: string]: Json } }; requests: { [input: string]: { e: string; set?: { [field: string]: Json }; absent?: string[]; why?: string } } }
const recorded: Recording | null = fs.existsSync(RECORD_FILE) ? JSON.parse(fs.readFileSync(RECORD_FILE, 'utf8')) : null;
const unpack = (r: Recording['requests'][string]): { [k: string]: Json } => {
  if (r.why) return { $why: r.why };
  const out = { ...recorded!.base[r.e] };
  for (const k of r.absent ?? []) delete out[k];
  return { ...out, ...r.set };
};
const print = fingerprint();
const mode: 'live' | 'recorded' | 'changed' | 'absent' = RECORD || LIVE ? 'live' : print === null || arg('--as-if-absent') ? 'absent' : recorded && recorded.fingerprint === print ? 'recorded' : 'changed';
if (mode === 'absent' && !recorded) { console.log('the existing frontend is not there and nothing is recorded: nothing to compare with'); process.exit(1); }
/* the existing page is loaded only when it may be driven */
const app = mode === 'absent' ? null : await import('./real-app/app.ts');

/* ================================================================ the states, in the definitions' terms */

interface Case { name: string; cause: string; desk: Desk }
const ALL = new Set(DEFINITIONS.map(d => d.id));
const BUILT_IN = DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id);
/* every state has a description and, where the engine sings, lyrics: the existing page offers Generate only then */
const base = (e: EngineId): Desk => ({ engine: e, tier: 'lab', vals: { 'prompt.text': 'a song', ...(has(ix, 'lyrics.text', e) ? { 'lyrics.text': 'la la la' } : {}) }, auto: {}, state: {}, on: new Set(ALL), selection: { ...SELECTION } });
const clone = (d: Desk): Desk => ({ ...d, vals: { ...d.vals }, auto: { ...d.auto }, state: structuredClone(d.state), on: new Set(d.on), selection: { ...d.selection } });
const onGrid = (x: number, step: number) => +x.toFixed(Math.max(0, -Math.floor(Math.log10(step)) + 2));
/** A seed beyond 32 bits, as the application's seeds are (ui/src/utils/seed.ts). */
const BIG_SEED = '18446744073709551615';

/** The values a parameter is tried at on an engine: its ends and a step off its default, every option (every id a
 *  server list has), both ways for a switch, set and empty for text. */
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
    const fx = from.startsWith('plugins.') ? (FIX.plugins[from.slice(8)] ?? []) as string[] : [];
    return [...new Set([...inc, ...(f.options!.only ?? fx), ...(from === 'models' ? ['', 'picked-model.gguf'] : []), ...(!inc.length && !fx.length && from !== 'models' ? ['picked-value'] : [])])];
  }
  const t = f.check ? (f.check.rule === 'timesteps' ? '1,0.6,0.2,0' : '1234') : f.example ?? 'x';
  return [t, ''];
}

/** The values a condition's names can take, to find a desk where it holds or does not. */
function* settings(c: Condition, e: EngineId): Generator<{ vals: Record<string, Value>; auto: Record<string, boolean>; state: Record<string, Json>; off: string[] }> {
  const keys = new Set<string>(), states = new Set<string>(), desks = new Set<string>();
  const walk = (x: Condition): void => {
    if ('all' in x) x.all.forEach(walk); else if ('any' in x) x.any.forEach(walk); else if ('not' in x && !('param' in x)) walk(x.not);
    else if ('param' in x) keys.add(x.param); else if ('state' in x) states.add(x.state); else if ('onDesk' in x) desks.add(x.onDesk);
  };
  walk(c);
  const axes: { set: (o: any, v: unknown) => void; values: unknown[] }[] = [];
  for (const k of keys) {
    const hit = ix.params.get(k); if (!hit || !has(ix, k, e)) continue;
    const f = resolve(hit.p, e), named: Value[] = [];
    const scan = (x: Condition): void => {
      if ('all' in x) x.all.forEach(scan); else if ('any' in x) x.any.forEach(scan); else if ('not' in x && !('param' in x)) scan(x.not);
      else if ('param' in x && x.param === k) {
        [x.above, x.below].forEach(b => { if (b !== undefined) named.push(b + (f.range?.[2] ?? 0.01), b - (f.range?.[2] ?? 0.01)); });
        /* every value a condition names, so a condition on a server-listed choice can hold whatever its place in the list */
        [x.is, x.not].flat().forEach(v => { if (v !== undefined && typeof v !== 'object') named.push(v as Value); });
      }
    };
    scan(c);
    const inRange = (v: Value) => hit.p.type !== 'number' || (typeof v === 'number' && !!f.range && v >= f.range[0] && v <= f.range[1]);
    axes.push({ set: (o, v) => { o.vals[k] = v; }, values: [...new Set([...named.filter(inRange), ...tries(hit.p, e)])] });
    if (f.auto) axes.push({ set: (o, v) => { o.auto[k] = v; }, values: [false, true] });
  }
  for (const s of states) axes.push({ set: (o, v) => { if (s === 'adapters' || s === 'slots') o.state.slots = slots(v as number); else if (s === 'sectionTags' || s === 'sectionDirectives') { if (v) o.vals['lyrics.text'] = v === 1 ? '[Verse]\nla' : '[Verse] {#2=0.5}\nla\n[Chorus]\nla'; } else if (ix.states.has(s)) o.state[s] = v ? stateSample(s) : null; }, values: [0, 1, 2, 3] });
  for (const d of desks) axes.push({ set: (o, v) => { if (!v) o.off.push(d); }, values: [true, false] });
  let n = 1; for (const a of axes) n *= a.values.length;
  const idx = axes.map(() => 0);
  for (let i = 0; i < Math.min(n, 4096); i++) {
    const o: any = { vals: {}, auto: {}, state: {}, off: [] };
    axes.forEach((a, j) => a.set(o, a.values[idx[j]!]));
    yield o;
    for (let j = 0; j < axes.length; j++) { idx[j]!++; if (idx[j]! < axes[j]!.values.length) break; idx[j] = 0; }
  }
}
const TRACKS: Json = [{ title: 'Neon Rain', bpm: 96, caption: 'synthwave, female vocal, rain' }, { title: 'Low Tide', caption: 'house, warm bass, male vocal' }];
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
/** Desks where `c` holds (or not), starting from `desk`: the first `n` that differ. */
function withConditions(desk: Desk, c: Condition | undefined, want: boolean, n: number): Desk[] {
  if (!c) return want ? [desk] : [];
  const out: Desk[] = [];
  for (const s of settings(c, desk.engine)) {
    const d = clone(desk);
    Object.assign(d.vals, s.vals); Object.assign(d.auto, s.auto); Object.assign(d.state, s.state);
    s.off.forEach(x => d.on.delete(x));
    if (holds(ix, c, d) === want) { out.push(d); if (out.length >= n) break; }
  }
  return out;
}
/** A desk where `c` holds (or not), starting from `desk`. */
const withCondition = (desk: Desk, c: Condition | undefined, want: boolean): Desk | null => withConditions(desk, c, want, 1)[0] ?? null;
/** A generator seeded by a string: a random state depends only on what it varies, so adding a definition does not
 *  move the states of the others. */
function rng(seed: string): () => number {
  let s = 0; for (let i = 0; i < seed.length; i++) s = (Math.imul(s, 31) + seed.charCodeAt(i)) | 0;
  return () => { s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function* cases(e: EngineId): Generator<Case> {
  const b = base(e);
  yield { name: 'defaults, everything on the desk', cause: '-', desk: b };
  const nothing = clone(b); nothing.on = new Set(BUILT_IN);
  yield { name: 'defaults, nothing on the desk', cause: '-', desk: nothing };
  const params = [...ix.params.values()].filter(({ p, d }) => has(ix, p.key, e) && !(d.kind === 'module' && d.source));
  for (const { p } of params) {
    const f = resolve(p, e);
    /* each condition a wire of it sends under, alone and with every switch of its definition on */
    const whens = wiresOf(f.wire).flatMap(w => (typeof w === 'object' && w.when ? [{ when: w.when, to: w.to as string }] : []));
    for (const w of whens) {
      const on = withCondition(b, f.when ? { all: [f.when, w.when] } : w.when, true);
      if (on) for (const lit of [false, true]) {
        const d = clone(on); d.vals[p.key] = tries(p, e)[0]!; if (f.auto) d.auto[p.key] = false;
        if (lit) for (const q of ix.params.get(p.key)!.d.params) if (q.type === 'boolean' && has(ix, q.key, e) && !(q.key in d.vals)) d.vals[q.key] = true;
        if (holds(ix, w.when, d)) yield { name: `${p.key} = ${J(d.vals[p.key])} while its wire to ${w.to} holds${lit ? ', its switches on' : ''}`, cause: p.key, desk: d };
      }
    }
    const ready = withCondition(b, f.when, true);
    if (ready) {
      for (const v of tries(p, e)) { const d = clone(ready); d.vals[p.key] = v; if (f.auto) d.auto[p.key] = false; yield { name: `${p.key} = ${J(v)}`, cause: p.key, desk: d }; }
      if (f.auto) { const d = clone(ready); d.auto[p.key] = true; d.vals[p.key] = tries(p, e)[1]!; yield { name: `${p.key} on auto`, cause: p.key, desk: d }; }
    }
    /* each option of a choice with its definition's switches on and its numbers changed: what a mode routes where is
       tried from the choice, not only from the wires that name it (a wire that lost its condition still shows) */
    if (ready && p.type === 'choice' && !p.key.endsWith('.on')) {
      const mine = ix.params.get(p.key)!.d.params.filter(q => q.key !== p.key && has(ix, q.key, e));
      if (mine.some(q => q.type === 'number')) for (const v of tries(p, e)) {
        const d = clone(ready); d.vals[p.key] = v;
        for (const q of mine) {
          if (q.key in d.vals) continue;
          if (q.type === 'boolean') d.vals[q.key] = true;
          else if (q.type === 'number') { const dv = resolve(q, e).default; d.vals[q.key] = tries(q, e).find(x => x !== dv)!; if (resolve(q, e).auto) d.auto[q.key] = false; }
        }
        yield { name: `${p.key} = ${J(v)}, its definition's switches on and numbers changed`, cause: p.key, desk: d };
      }
    }
    /* set where its condition does not hold, in each of the first few ways it can fail (no adapter, one adapter) */
    if (f.when) for (const [i, off] of withConditions(b, f.when, false, 3).entries()) {
      const d = clone(off); d.vals[p.key] = tries(p, e)[2] ?? tries(p, e)[0]!; if (f.auto) d.auto[p.key] = false;
      yield { name: `${p.key} set while its condition does not hold${i ? ` (${i + 1})` : ''}`, cause: p.key, desk: d };
    }
    /* set with its module's state held (adapters loaded, a file chosen), where a module's state is what makes it act */
    const own = ix.params.get(p.key)!.d.state ?? [];
    if (ready && own.length) {
      const d = clone(ready); for (const s of own) if (d.state[s.key] === undefined) d.state[s.key] = stateSample(s.key);
      d.vals[p.key] = tries(p, e)[0]!; if (f.auto) d.auto[p.key] = false;
      if (applies(ix, p.key, d)) yield { name: `${p.key} = ${J(d.vals[p.key])} with its module's state held`, cause: p.key, desk: d };
    }
  }
  /* each definition off the desk, its values changed */
  for (const d of DEFINITIONS.filter(x => x.engines.includes(e) && !(x.kind === 'module' && (x.builtIn || x.source)))) {
    const k = clone(b); k.on.delete(d.id);
    for (const p of d.params) if (has(ix, p.key, e)) k.vals[p.key] = tries(p, e)[0]!;
    yield { name: `${d.id} off the desk, its values changed`, cause: d.id, desk: k };
  }
  /* module state */
  for (const n of [1, 2, 3]) for (const adv of ['simple', 'advanced']) for (const ph of [false, true]) {
    if (!has(ix, 'ad.adv', e) && adv === 'advanced') continue;
    const d = clone(b); d.state.slots = slots(n, ph); if (has(ix, 'ad.adv', e)) d.vals['ad.adv'] = adv; d.vals.s1 = 0.5;
    yield { name: `${n} adapters, ${adv}${ph ? ', the second in a phase window' : ''}`, cause: 'slots', desk: d };
  }
  for (const seed of [123456, 9007199254740991, BIG_SEED]) for (const lock of [true, false]) { const d = clone(b); d.state.seed = seed; d.vals['seed.lock'] = lock; yield { name: `seed ${seed}, kept ${lock}`, cause: 'seed', desk: d }; }
  if (has(ix, 'cap.src', e)) for (const src of ['custom', 'auto', 'Low Tide']) for (const text of ['', 'my own caption']) for (const adapter of [false, true]) {
    const d = clone(b); d.state.capTracks = TRACKS; d.vals['cap.src'] = src; d.vals['prompt.text'] = text; d.vals['tempo.bpm'] = 126; d.auto['tempo.bpm'] = false;
    if (adapter) d.state.slots = slots(1);
    yield { name: `caption source ${src}, caption ${J(text)}${adapter ? ', an adapter loaded' : ''}`, cause: 'capTracks', desk: d };
  }
  for (const { s } of ix.states.values()) if (s.kind === 'file' || s.kind === 'span') { const d = clone(b); d.state[s.key] = stateSample(s.key); if (s.key === 'contSel') d.state.contFile = stateSample('contFile'); yield { name: `${s.key} set`, cause: s.key, desk: d }; }
  if (QUICK) return;
  /* random combinations: each parameter, definition and the state draw from their own generator */
  for (let i = 0; i < 300; i++) {
    const d = base(e);
    for (const x of DEFINITIONS) if (!(x.kind === 'module' && x.builtIn) && rng(`${e}|${i}|on:${x.id}`)() < 0.25) d.on.delete(x.id);
    for (const { p } of params) {
      const g = rng(`${e}|${i}|${p.key}`);
      if (g() < 0.3) { const t = tries(p, e); d.vals[p.key] = t[Math.floor(g() * t.length)]!; }
      if (resolve(p, e).auto && g() < 0.3) d.auto[p.key] = g() < 0.5;
    }
    const s = rng(`${e}|${i}|state`);
    if (s() < 0.5) d.state.slots = slots(Math.floor(s() * 4), s() < 0.5);
    if (s() < 0.5) d.state.seed = Math.floor(s() * 4294967295);
    for (const { s: st } of ix.states.values()) if ((st.kind === 'file' || st.kind === 'span') && s() < 0.3) d.state[st.key] = stateSample(st.key);
    yield { name: `random #${i + 1}`, cause: 'random', desk: d };
  }
}

/* ================================================================ the same things done on the existing page */

const linkOf = (key: string, e: EngineId): Link[] | null => {
  const h: Holding | undefined = LINKS[key];
  if (h === undefined) return null;
  const one = (x: string | Link): Link => (typeof x === 'string' ? { control: x } : x);
  if (typeof h === 'object' && !Array.isArray(h) && 'on' in h) { const l = h.on[e]; return l === undefined ? null : [l].flat().map(one); }
  return [h as string | Link | (string | Link)[]].flat().map(one);
};
const used = new Map<string, number>(), missingControls = new Map<string, number>();
function linkValue(l: Link, desk: Desk, v: Value | undefined, auto: boolean): Json | undefined {
  const all = { ...desk, on: new Set(ALL) };
  if (auto) return l.auto;
  if (l.when && !holds(ix, l.when, all)) return l.else;
  if (l.from) v = valueOf(ix, l.from, all);
  if (v === undefined || (l.leave || []).includes(v)) return undefined;
  let x: Json | undefined = v;
  if (l.map) x = String(v) in l.map ? l.map[String(v)] : undefined;
  if (l.invert && typeof x === 'boolean') x = !x;
  if (l.times !== undefined && typeof x === 'number') x = +(x * l.times).toFixed(10);
  if (l.as === 'text' && x !== undefined) x = String(x);
  if (l.as === 'number' && typeof x === 'string') x = /^-?\d+$/.test(x.trim()) ? Number(x.trim()) : undefined;
  return x;
}
/** What a user of the existing page does for the same desk. The existing page has no desk, so every value the user
 *  set is held whether or not its module is on the desk, and a pedal is as the user left its switch: that is the
 *  difference docs/08 asks about (question 11), shown here rather than assumed on both sides. */
function existingOf(desk: Desk): Existing {
  const e = desk.engine, all = { ...desk, on: new Set(ALL) };
  const x: Existing = { selection: { ...desk.selection }, store: [], storage: [], use: [] };
  const target = (to: string, v: Json) => { if (to.startsWith('selection:')) x.selection[to.slice(10)] = String(v); else x.store.push([to.slice(6), v]); };
  for (const [k, f] of Object.entries(STATE_HOLDINGS)) if (desk.state[k] !== undefined && desk.state[k] !== null && ix.states.get(k)?.d.engines.includes(e)) f(desk.state[k]!, e).forEach(([to, v]) => target(to, v));
  if (e === 'ace' && typeof desk.state.latent === 'string' && desk.state.latent) x.storage.push(['hs-sourceLatentUrl', desk.state.latent]);
  /* the seed is typed into the seed's own field (SeedControl), which keeps a seed beyond 2^53 as its decimal text */
  if (desk.state.seed !== undefined && desk.state.seed !== null) x.use.push(['Generation > @SeedInput', desk.state.seed]);
  if (e === 'yue2' && typeof desk.state.contFile === 'string' && desk.state.contFile) {
    const s = (desk.state.contSel ?? {}) as { start?: number; end?: number };
    x.use.push(['Create > @Yue2Continuation', { sourceAudioUrl: desk.state.contFile, start: s.start ?? 0, end: s.end ?? 30 }]);
  }
  const tracks = Array.isArray(desk.state.capTracks) ? desk.state.capTracks as { title: string; bpm?: number; caption: string }[] : [];
  if (has(ix, 'cap.src', e)) { const cs = CAPTION_SOURCE(e, { source: String(valueOf(ix, 'cap.src', all)), dataset: String(valueOf(ix, 'cap.ds', all) ?? ''), caption: String(valueOf(ix, 'prompt.text', all) ?? ''), tracks, adapter: Array.isArray(desk.state.slots) && desk.state.slots.length > 0 }); if (cs !== undefined) x.captionSource = cs as Record<string, Json>; }
  /* the controls, as a user who wants every value set goes about it: every switch the desk holds turned on first (a
     switch opens what it governs), then the choices and the rest, then each switch to where the desk has it. The
     existing page keeps a value set behind a switch that is then turned off, as the definitions do. */
  const order = (k: string) => ({ boolean: 2, choice: 0, number: 1, text: 1 })[ix.params.get(k)!.p.type];
  const keys = [...new Set([...Object.keys(desk.vals), ...Object.keys(desk.auto)])].filter(k => ix.params.has(k) && has(ix, k, e)).sort((a, b) => order(a) - order(b));
  const revealed = new Set<string>();
  const useLinks = (k: string, v: Value | undefined, auto: boolean, last = false) => {
    for (const l of linkOf(k, e) ?? []) {
      const hv = linkValue(l, { ...desk, vals: { ...desk.vals, [k]: v as Value } }, v, auto);
      if (hv === undefined) continue;
      if (l.reveal && !revealed.has(l.reveal)) { revealed.add(l.reveal); x.use.push([l.reveal, '@press']); }
      if (typeof l.press === 'string') { if (hv === true) x.use.push([l.press, '@press']); }
      else if (l.press) { const btn = l.press[String(hv)]; if (btn) x.use.push([btn, '@press']); }
      else if (l.control) x.use.push(last ? [l.control, hv, 'last'] : [l.control, hv]);
    }
  };
  const switches = keys.filter(k => ix.params.get(k)!.p.type === 'boolean' && !RESTATED[k] && valueOf(ix, k, all) === false);
  for (const k of switches) if (!(linkOf(k, e) ?? []).some(l => typeof l.press === 'string')) useLinks(k, true, false);
  for (const k of keys) {
    const r = RESTATED[k];
    if (r) { r.set(valueOf(ix, k, all)!, e, desk).forEach(([to, v]) => target(to, v)); continue; }
    useLinks(k, valueOf(ix, k, all), isAuto(ix, k, all), switches.includes(k));
  }
  return x;
}

/* ================================================================ both sides */

const flatten = (o: Record<string, unknown>, prefix = ''): Record<string, Json> =>
  Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined).map(([k, v]) => [prefix + k, v as Json]));
/* compose as the server reads it: an absent switch is its default (resolveIntent.ts, composeCreateCaption) */
const COMPOSE = { autoExpand: false, loraTrigger: '', beatIntro: false, introBars: 2 };
type Tracks = { title: string; bpm?: number; caption: string }[];
function finish(e: EngineId, i: { params: unknown; compose?: unknown; captionSource?: unknown }, tracks: Tracks): Record<string, Json> {
  /* the tracks a caption source picks from, as the server loads them for the intent (loadIntentData.ts) */
  const data = { mm3Tracks: tracks, yue2Tracks: tracks.map(t => ({ name: t.title, caption: t.caption, bpm: t.bpm })) };
  const r = JSON.parse(J(resolveCreateIntent(i, e, data).request)) as Record<string, Json>;
  return { ...flatten(r), ...flatten({ ...COMPOSE, ...(i.compose ?? {}) as Record<string, unknown> }, 'compose.'), ...flatten((i.captionSource ?? {}) as Record<string, unknown>, 'captionSource.') };
}
const tracksOf = (desk: Desk) => (Array.isArray(desk.state.capTracks) ? desk.state.capTracks as Tracks : []);
const inputKey = (e: EngineId, x: unknown) => crypto.createHash('sha1').update(J([e, x])).digest('hex').slice(0, 20);
const fresh: { [input: string]: { [k: string]: Json } } = {}, freshEngine: { [input: string]: EngineId } = {};
let drivenLive = 0, fromRecording = 0, changedRecords = 0;
const changedExamples: string[] = [];
/** The same request, whatever order its fields were written in (a recording is a base and a change). */
const sameFields = (a: { [k: string]: Json }, b: { [k: string]: Json }) => { const ka = Object.keys(a); return ka.length === Object.keys(b).length && ka.every(k => J(a[k]) === J(b[k])); };
const fieldsOf = (r: { [k: string]: Json }) => ('$why' in r ? { fields: null, why: String(r.$why) } : { fields: r });
function theirs(desk: Desk): { fields: Record<string, Json> | null; why?: string } {
  const e = desk.engine, x = existingOf(desk), key = inputKey(e, x), rec = recorded?.requests[key];
  if (mode === 'absent' || (mode === 'recorded' && rec)) {
    fromRecording++;
    return rec ? fieldsOf(unpack(rec)) : { fields: null, why: 'nothing recorded for this state' };
  }
  if (fresh[key]) return fieldsOf(fresh[key]!);
  drivenLive++;
  const r = app!.existingIntent(e, x);
  for (const m of r.missing) missingControls.set(`${e}|${m}`, (missingControls.get(`${e}|${m}`) ?? 0) + 1);
  for (const [c] of x.use) if (!r.missing.includes(c)) used.set(`${e}|${c}`, (used.get(`${e}|${c}`) ?? 0) + 1);
  const out: { [k: string]: Json } = r.intent ? finish(e, r.intent, tracksOf(desk)) : { $why: r.why ?? 'no request' };
  fresh[key] = out; freshEngine[key] = e;
  if ((mode === 'changed' || (LIVE && !RECORD)) && rec && !sameFields(unpack(rec), out)) { changedRecords++; if (changedExamples.length < 3) changedExamples.push(`${e}: ${J(x.use).slice(0, 160)}`); }
  return fieldsOf(out);
}
const ours = (desk: Desk) => finish(desk.engine, createIntent(ix, desk, REAL), tracksOf(desk));

/* ================================================================ the accepted differences */

/** An accepted difference names exactly what it covers: one field, the engines it occurs on, the state it occurs in (a
 *  condition on the desk in the definitions' terms, and parameters that must be set, or left alone), and what each
 *  side sends there: a value, "absent", "sent" (any value), or "value:<key>" (that parameter's value), or a list of
 *  these. Any other difference fails, and an entry that no longer occurs on one of its engines fails too. `question`
 *  is the question in docs/08-open-questions.md that decides it. */
interface Accepted { id: string; engines: EngineId[]; field: string; when?: Condition; set?: string[]; unset?: string[]; existing: Json; definitions: Json; why: string; question: number; rule?: 'on the desk' }
const expected: Accepted[] = fs.existsSync(EXPECT_FILE) ? JSON.parse(fs.readFileSync(EXPECT_FILE, 'utf8')).differences : [];
/* "prefix:<key>": this side sends the other's value with that parameter's value and ", " before it (a trigger word) */
const sideIs = (want: Json, got: Json | undefined, desk: Desk, other?: Json): boolean => [want].flat().some(w =>
  w === 'absent' ? got === undefined : w === 'sent' ? got !== undefined : typeof w === 'string' && w.startsWith('value:') ? J(got) === J(valueOf(ix, w.slice(6), desk))
    : typeof w === 'string' && w.startsWith('prefix:') ? got === `${desk.vals[w.slice(7)]}, ${other}` : J(got) === J(w));
/** The one rule that is not a field: a difference that goes away when every module, pedal and unit is put on the desk
 *  is the desk's own (docs/08 question 11), whatever the field; the definitions' side must then send exactly what
 *  the existing page sends. */
const allOnSent = new WeakMap<Desk, Record<string, Json>>();
const onTheDesk = (desk: Desk, field: string, existing: Json | undefined): boolean => {
  if (desk.on.size >= ALL.size) return false;
  if (!allOnSent.has(desk)) allOnSent.set(desk, ours({ ...desk, on: new Set(ALL) }));
  const mine = allOnSent.get(desk)![field], all = { ...desk, on: new Set(ALL) };
  /* with everything on the desk the two agree, or differ only as another entry names exactly (which is counted) */
  if (J(mine) === J(existing)) return true;
  const a = expected.find(x => !x.rule && matches(x, desk.engine, field, all, existing, mine));
  if (a) hits.set(`${a.id}|${desk.engine}`, (hits.get(`${a.id}|${desk.engine}`) ?? 0) + 1);
  return !!a;
};
const matches = (a: Accepted, e: EngineId, field: string, desk: Desk, existing: Json | undefined, definitions: Json | undefined): boolean =>
  a.rule === 'on the desk' ? a.engines.includes(e) && onTheDesk(desk, field, existing) : a.field === field && a.engines.includes(e) && (!a.when || holds(ix, a.when, desk)) && (a.set ?? []).every(k => k in desk.vals) && (a.unset ?? []).every(k => !(k in desk.vals))
  && sideIs(a.existing, existing, desk, definitions) && sideIs(a.definitions, definitions, desk, existing);

/* ================================================================ run */

const SHOW = process.argv.indexOf('--show');
if (SHOW > 0) {
  const e = process.argv[SHOW + 1] as EngineId, words = process.argv[SHOW + 2] ?? '';
  for (const c of cases(e)) if (c.name.includes(words)) {
    const mine = ours(c.desk), t = theirs(c.desk), x = existingOf(c.desk);
    console.log(`${c.name}\n  desk: ${J({ vals: c.desk.vals, auto: c.desk.auto, state: c.desk.state, off: [...ALL].filter(k => !c.desk.on.has(k)) })}\n  on the existing page: ${J(x)}${t.why ? `\n  existing: ${t.why}` : ''}`);
    const th = t.fields ?? {};
    for (const k of [...new Set([...Object.keys(mine), ...Object.keys(th)])].sort()) console.log(`  ${J(mine[k]) === J(th[k]) ? ' ' : '*'} ${k}: existing ${J(th[k])} | definitions ${J(mine[k])}`);
    process.exit(0);
  }
  console.log('no such state'); process.exit(1);
}

interface Diff { engine: EngineId; field: string; existing?: Json; definitions?: Json; state: string; cause: string }
const all: Diff[] = [], hits = new Map<string, number>(), unread: string[] = [];
/* the fields the existing page sends at all, per engine: a field it never sends has no reference (question 19) */
const existingFields = new Map<EngineId, Set<string>>();
let refused = 0, previewed = 0;
const report: string[] = [];
const t0 = performance.now();
for (const { id: e } of ENGINES) {
  let states = 0, fields = 0, found = 0;
  for (const c of cases(e)) {
    states++;
    const mine = ours(c.desk), t = theirs(c.desk);
    if (!t.fields) {
      /* the existing page offers no Generate (nothing to make a song from), or asks the server for a score to preview
         first (YuE2's Preview the score first): nothing is sent to compare */
      if (/disabled/.test(t.why ?? '')) refused++;
      else if (/did not hand/.test(t.why ?? '') && valueOf(ix, 'lm.preview', { ...c.desk, on: new Set(ALL) }) === true) previewed++;
      else unread.push(`${e} ${c.name}: ${t.why}`);
      continue;
    }
    const th = t.fields, keys = [...new Set([...Object.keys(mine), ...Object.keys(th)])];
    if (!existingFields.has(e)) existingFields.set(e, new Set());
    for (const k of Object.keys(th)) existingFields.get(e)!.add(k);
    fields += keys.length;
    for (const k of keys.filter(k => J(mine[k]) !== J(th[k]))) {
      found++;
      /* the exact entries first; the desk's rule only for what none of them names */
      const a = expected.find(x => !x.rule && matches(x, e, k, c.desk, th[k], mine[k])) ?? expected.find(x => x.rule && matches(x, e, k, c.desk, th[k], mine[k]));
      if (a) hits.set(`${a.id}|${e}`, (hits.get(`${a.id}|${e}`) ?? 0) + 1);
      else all.push({ engine: e, field: k, ...(k in th ? { existing: th[k] } : {}), ...(k in mine ? { definitions: mine[k] } : {}), state: c.name, cause: c.cause });
    }
  }
  report.push(`  ${e}: ${states} states, ${fields} fields compared, ${found} differences`);
}

/* which pairs the existing side drove through the page's own controls, restated, or has no control for */
const pairs = { driven: 0, restated: 0, none: [] as string[], linkedNotFound: [] as string[] };
for (const { p, d } of ix.params.values()) for (const e of d.engines) {
  if (!has(ix, p.key, e) || (d.kind === 'module' && d.source)) continue;
  const ws = wiresOf(resolve(p, e).wire);
  if (!ws.length || ws.every(w => /^(ui|model):/.test(toOf(w)))) continue;
  if (RESTATED[p.key]) { pairs.restated++; continue; }
  const links = linkOf(p.key, e);
  if (!links) { pairs.none.push(`${p.key} [${e}]`); continue; }
  const controls = links.flatMap(l => (typeof l.press === 'object' ? Object.values(l.press) : [l.press ?? l.control!]));
  if (mode === 'recorded' || mode === 'absent' || controls.some(c => used.has(`${e}|${c}`))) pairs.driven++; else pairs.linkedNotFound.push(`${p.key} [${e}] ${controls[0]}`);
}

/* the transforms' branches reached: every block of every send */
const cov = await coverage.post('Profiler.takePreciseCoverage') as { result: { url: string; functions: { functionName: string; ranges: { startOffset: number; endOffset: number; count: number }[] }[] }[] };
const tf = cov.result.find(s => s.url.endsWith('/src/definitions/transforms.ts'));
const src = fs.readFileSync(path.join(ROOT, 'src', 'definitions', 'transforms.ts'), 'utf8');
const lineOf = (o: number) => src.slice(0, o).split('\n').length;
const starts = [...src.matchAll(/^  (\w+): \{/gm)].map(m => [m.index!, m[1]!] as const);
const nameAt = (o: number) => starts.filter(([at]) => at < o).pop()?.[1] ?? '';
const reached: string[] = [], missed: string[] = [];
for (const f of tf?.functions ?? []) {
  const first = f.ranges[0]!;
  if (!/send\s*:\s*$/.test(src.slice(Math.max(0, first.startOffset - 12), first.startOffset))) continue;
  const zero = f.ranges.filter(r => r.count === 0), name = nameAt(first.startOffset);
  reached.push(`${name} ${first.count === 0 ? 0 : f.ranges.length - zero.length}/${f.ranges.length}`);
  if (first.count === 0 || zero.length) missed.push(`${name}: ${first.count === 0 ? 'never called' : zero.map(r => `line ${lineOf(r.startOffset)}`).join(', ')}`);
}
for (const name of starts.map(([, n]) => n)) if (!reached.some(r => r.startsWith(name + ' '))) missed.push(`${name}: never called`);

/* a field the existing page never sends on that engine, sent by a parameter it has no control for: new, with no
   reference to compare with. Listed, not failed; question 19 asks whether new fields are sent. A field the existing
   page does send, sent wrongly, still fails, as does a new field written by a parameter the page has a control for. */
const never = (d: Diff) => d.existing === undefined && !existingFields.get(d.engine)?.has(d.field);
const newKeys = new Set(all.filter(d => never(d) && ix.params.has(d.cause) && !linkOf(d.cause, d.engine) && !RESTATED[d.cause]).map(d => `${d.engine}|${d.field}`));
const unseen = all.filter(d => never(d) && newKeys.has(`${d.engine}|${d.field}`));
const newFields = new Map<string, Set<string>>();
for (const d of unseen.filter(x => ix.params.has(x.cause))) { const k = `${d.engine} ${d.field}`; newFields.set(k, (newFields.get(k) ?? new Set()).add(d.cause)); }
const classes = new Map<string, Diff[]>();
for (const d of all.filter(x => !unseen.includes(x))) { const k = `${d.engine} | ${d.field}`; classes.set(k, [...(classes.get(k) || []), d]); }
const gone = QUICK ? [] : expected.flatMap(a => a.engines.filter(e => !hits.has(`${a.id}|${e}`)).map(e => `${a.id} [${e}]`));

if (arg('--print')) for (const [k, list] of classes) console.log(`${k}: ${list.length} states; causes ${[...new Set(list.map(d => d.cause))].slice(0, 6).join(', ')}\n    e.g. ${list[0]!.state}: existing ${J(list[0]!.existing)} | definitions ${J(list[0]!.definitions)}`);
const at = recorded?.commit.slice(0, 10);
const how = { live: 'live: the existing page was driven', recorded: `recorded: the existing frontend is unchanged since ${at}, so its recording stands`, changed: `live: the existing frontend changed since the recording at ${at}`, absent: `recorded: the existing frontend is not there; its recording at ${at} stands` }[mode];
console.log(`request over the value space, ${ENGINES.length} engines${QUICK ? ' (targeted states only)' : ''}; existing side ${how} (${drivenLive} states driven, ${fromRecording} from the recording, ${((performance.now() - t0) / 1000).toFixed(1)} s):\n${report.join('\n')}`);
if (refused || previewed) console.log(`  ${refused + previewed} states left out: ${refused} where the existing page offers no Generate (no description, lyrics or Instrumental), ${previewed} where it asks the server for a score to preview before it sends anything (YuE2)`);
console.log(`  pairs: ${pairs.driven} driven through the existing page's own controls, ${pairs.restated} restated (the engine panel's model choices), ${pairs.none.length} with no control on the existing page: ${pairs.none.join(', ')}`);
console.log(`  transforms reached (blocks of send reached/all): ${reached.join(', ')}`);
if (newFields.size) console.log(`  new fields the existing page never sends, from parameters it has no control for (question 19): ${[...newFields].map(([k, c]) => `${k} (${[...c].join(', ')})`).join('; ')}`);
console.log(`  ${[...hits.values()].reduce((a, b) => a + b, 0)} differences accepted by ${new Set([...hits.keys()].map(k => k.split('|')[0])).size} entries of tools/fixtures/request-differences.json`);
let bad = false;
if (unread.length) { bad = true; console.log(`  the existing page sent nothing in ${unread.length} states: ${unread.slice(0, 4).join('; ')}`); }
if (pairs.linkedNotFound.length && !QUICK) { bad = true; console.log(`  tools/existing-ui.ts names a control the page never showed: ${pairs.linkedNotFound.join('; ')}`); }
if (missingControls.size && arg('--missing')) console.log(`  controls not shown when they were to be used: ${[...missingControls].map(([k, n]) => `${k} (${n})`).join('; ')}`);
if (missed.length && !QUICK) { bad = true; console.log(`  transform branches never reached: ${missed.join('; ')}`); }
if (classes.size) { bad = true; for (const [k, list] of classes) console.log(`  new: ${k} (${list.length} states; e.g. ${list[0]!.state}: existing ${J(list[0]!.existing)}, definitions ${J(list[0]!.definitions)})`); }
if (gone.length) { bad = true; console.log(`  gone: ${gone.join(', ')}: the list accepts a difference that no longer occurs there`); }
if (changedRecords) { bad = true; console.log(`  ${changedRecords} recorded requests are not what the existing page sends now (e.g. ${changedExamples.join('; ')}): \`npm run record\` records it again`); }
if (RECORD) {
  const commit = (() => { try { return execFileSync('git', ['-C', APP, 'log', '-1', '--format=%H', '--', 'ui'], { encoding: 'utf8' }).trim(); } catch { return 'unknown'; } })();
  /* each engine's base is its most common value of each field; a state records what differs from it */
  const base: Recording['base'] = {};
  for (const e of ENGINES.map(x => x.id)) {
    const reqs = Object.keys(fresh).filter(k => freshEngine[k] === e && !('$why' in fresh[k]!)).map(k => fresh[k]!);
    const counts = new Map<string, Map<string, number>>();
    for (const r of reqs) for (const [f, v] of Object.entries(r)) { const m = counts.get(f) ?? new Map<string, number>(); m.set(J(v), (m.get(J(v)) ?? 0) + 1); counts.set(f, m); }
    base[e] = Object.fromEntries([...counts].filter(([, m]) => [...m.values()].reduce((a, b) => a + b, 0) * 2 > reqs.length).map(([f, m]) => [f, JSON.parse([...m].sort((a, b) => b[1] - a[1])[0]![0])]));
  }
  const requests: Recording['requests'] = {};
  for (const k of Object.keys(fresh).sort()) {
    const r = fresh[k]!, e = freshEngine[k]!;
    if ('$why' in r) { requests[k] = { e, why: String(r.$why) }; continue; }
    const set = Object.fromEntries(Object.entries(r).filter(([f, v]) => J(base[e]![f]) !== J(v)));
    const absent = Object.keys(base[e]!).filter(f => !(f in r));
    requests[k] = { e, ...(Object.keys(set).length ? { set } : {}), ...(absent.length ? { absent } : {}) };
  }
  const note = 'What the existing Create page sends for each state the request comparison drives it into, keyed by a hash of what was done on the page (tools/compare-request.ts): per engine a base request, and per state the fields that differ from it. Recorded by `npm run record`; do not edit by hand.';
  const NL = String.fromCharCode(10), lines = Object.entries(requests).map(([k, v]) => `  ${J(k)}: ${J(v)}`);
  fs.writeFileSync(RECORD_FILE, ['{', ` "note": ${J(note)},`, ` "commit": ${J(commit)},`, ` "fingerprint": ${J(print ?? '')},`, ` "base": ${J(base)},`, ' "requests": {', lines.join(',' + NL), ' }', '}', ''].join(NL));
  console.log(`  recorded ${lines.length} requests at ${commit.slice(0, 10)} in tools/fixtures/existing-requests.json`);
}
process.exit(bad ? 1 : 0);
