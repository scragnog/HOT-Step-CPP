// The core's derivations against the concept page: undo step names, each tier's "Also set" line, engine switches,
// what a revision stores and what cleaning keeps, each compared with what the page did in the same state
// (tools/fixtures/concept.json, collected by tools/concept/collect.mjs). Then the parts' contract (bindPart), the
// translation keys, reading the real fields back, and saved records of older shapes.
//
//   node tools/test-core.ts            exit 1 on a disagreement tools/fixtures/concept-differences.json does not list
//   node tools/test-core.ts --print    every disagreement
//
// A disagreement is a defect in the core or a definition, fixed there, or listed with the reason the page is not
// followed (a fault its own behaviour suite records as pending, or a decision of the rebuild).

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFINITIONS } from '../src/definitions/index.ts';
import type { Definition, EngineId, Json, Tier, Value } from '../src/definitions/types.ts';
import {
  alsoSet, bindPart, cleanValues, editsAt, emptyBook, factsOf, fromReal, has, indexOf, loadRevision, migrate, readDesk, realState, resolve, toOf, wiresOf,
  revisionOf, stepWords, stringsOf, switchEngine, valueOf, type Change, type Desk, type EngineBook,
} from '../src/derive.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FIX = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools', 'fixtures', 'concept.json'), 'utf8'));
const ACCEPT_FILE = path.join(ROOT, 'tools', 'fixtures', 'concept-differences.json');
const REAL = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools', 'fixtures', 'real-app.json'), 'utf8'));
const ACCEPT: { derivation: string; match: string; why: string }[] = fs.existsSync(ACCEPT_FILE) ? JSON.parse(fs.readFileSync(ACCEPT_FILE, 'utf8')).differences : [];
const ix = indexOf(DEFINITIONS);
const J = (x: unknown) => JSON.stringify(x);
const ENG: Record<string, EngineId> = { ace: 'ace', mm3: 'minimax-m3', yue: 'yue2', sa3: 'sa3', hm: 'heartmula' };
const ALL = new Set(DEFINITIONS.map(d => d.id));
const faceOf = (d: Definition) => ('face' in d ? d.face ?? [] : []);

/* ================================================================ the page's values in the definitions' terms */

/** A value of the page (a choice is its option's name) as the definitions store it; undefined when the definitions
 *  cannot say it (a server-listed choice whose names the server gives). */
function ours(k: string, v: unknown, e: EngineId): Value | undefined {
  const hit = ix.params.get(k);
  if (!hit) return undefined;
  /* a value kept for another engine is said as that engine says it */
  if (!has(ix, k, e)) { const other = (hit.p.engines ?? hit.d.engines).find(x => hit.d.engines.includes(x)); return other && other !== e ? ours(k, v, other) : undefined; }
  const { p } = hit, f = resolve(p, e);
  if (p.type !== 'choice') return v as Value;
  const page = FIX.controls[Object.keys(ENG).find(b => ENG[b] === e)!]?.[k];
  /* only a value the page's control holds (one of its options), not one of the words it shows for them */
  if (Array.isArray(page?.opts) && !page.opts.includes(v)) return undefined;
  const names = page?.names ?? {};
  const opts = Array.isArray(f.options) ? f.options : f.options?.include ?? [];
  const o = opts.find(x => x.label === v || x.label === names[v as string] || x.short === v || String(x.value).toLowerCase() === String(v).toLowerCase());
  if (o) return o.value;
  /* a plugin the server lists: its id is its name in lower case, letters and digits ("DPM++ 3M" is dpm3m) */
  const from = !Array.isArray(f.options) && f.options?.from.startsWith('plugins.') ? REAL.plugins[f.options.from.slice(8)] as string[] : [];
  return from.find(id => norm(id) === norm(String(v)));
}
interface Mapped { vals: { [k: string]: Value }; auto: { [k: string]: boolean }; lost: string[] }
/** Without the keys the definitions cannot say (both sides drop them, so the rest is compared). */
const keep = (vals: { [k: string]: unknown }, e: EngineId) => Object.fromEntries(Object.entries(vals || {}).filter(([k, v]) => k.endsWith('.auto') || ours(k, v, e) !== undefined));
function mapVals(vals: { [k: string]: unknown }, e: EngineId): Mapped {
  const out: Mapped = { vals: {}, auto: {}, lost: [] };
  for (const [k, v] of Object.entries(vals || {})) {
    if (k.endsWith('.auto') && ix.params.has(k.slice(0, -5))) { out.auto[k.slice(0, -5)] = v as boolean; continue; }
    const x = ours(k, v, e);
    if (x === undefined) out.lost.push(k); else out.vals[k] = x;
  }
  return out;
}
/* the server's plugin lists with the names the page shows for them (its options, matched to the ids by name) */
const norm = (s: string) => s.toLowerCase().replace(/\(exp\.\)/, '').replace(/[^a-z0-9]/g, '');
const LISTS: Desk['lists'] = Object.fromEntries(Object.entries(REAL.plugins as Record<string, string[]>).map(([kind, ids]) => {
  const names = new Set<string>(); for (const bk of Object.keys(FIX.controls)) for (const c of Object.values<{ opts: unknown }>(FIX.controls[bk])) if (Array.isArray(c.opts)) c.opts.forEach(o => typeof o === 'string' && names.add(o));
  return [`plugins.${kind}`, ids.flatMap(id => { const n = [...names].find(x => norm(x) === norm(id)) ?? [...names].find(x => norm(x).length > 3 && norm(id).endsWith(norm(x))); return n ? [{ value: id, label: n }] : []; })];
}));
const deskOf = (e: EngineId, m: Mapped, extra: Partial<Desk> = {}): Desk => ({ engine: e, tier: 'lab', vals: m.vals, auto: m.auto, state: {}, facts: {}, on: new Set(ALL), lists: LISTS, ...extra });

/* ================================================================ compare, count, report */

interface Result { cases: number; agree: number; skipped: number; accepted: number; fails: string[] }
const results: Record<string, Result> = {};
const res = (name: string) => (results[name] ||= { cases: 0, agree: 0, skipped: 0, accepted: 0, fails: [] });
const used = new Set<number>();
function verdict(name: string, same: boolean, what: string): void {
  const r = res(name); r.cases++;
  if (same) { r.agree++; return; }
  /* a step that names a new parameter's owner because the page does not have the parameter (above) is right */
  if (what.startsWith('shares its words with a parameter the page does not have')) { r.accepted++; return; }
  const i = ACCEPT.findIndex(a => a.derivation === name && new RegExp(a.match).test(what));
  if (i >= 0) { r.accepted++; used.add(i); return; }
  r.fails.push(what);
}

/* ---- undo step names */
for (const s of FIX.steps) {
  const e = ENG[s.engine]!;
  const b = mapVals(s.before, e), a = mapVals(s.after, e);
  if (b.lost.length || a.lost.length || !s.words) { res('undo step names').skipped++; continue; }
  const before = deskOf(e, b, { state: s.seedBefore !== undefined ? { seed: s.seedBefore } : {} });
  const after = deskOf(e, a, { state: s.seedAfter !== undefined ? { seed: s.seedAfter } : {} });
  if (s.prompt) { before.vals['prompt.text'] = ''; after.vals['prompt.text'] = 'a new description'; }
  const w = stepWords(ix, before, after);
  /* a parameter added since the page was collected can share a label with one the page has, which rightly names both
     after their owners; that is said, so it is told from a defect */
  const conceptKeys = new Set(Object.values<Record<string, unknown>>(FIX.controls).flatMap(c => Object.keys(c)));
  const shares = Object.keys(a.vals).some(k => { const l = resolve(ix.params.get(k)!.p, e).label.toLowerCase(); return [...ix.params.values()].some(x => !conceptKeys.has(x.p.key) && resolve(x.p, e).label.toLowerCase() === l); });
  const pick = (x: { label?: string; undone?: string; redone?: string; put?: string } | null) => x && { label: x.label, undone: x.undone, redone: x.redone, put: x.put };
  verdict('undo step names', J(pick(w)) === J(pick(s.words)), `${shares ? 'shares its words with a parameter the page does not have: ' : ''}${e} ${J(s.after)}: page ${J(s.words)} | core ${J(w)}`);
}

/* ---- "Also set" */
for (const c of FIX.also) {
  const e = ENG[c.engine]!, m = mapVals(c.vals, e), d = DEFINITIONS.find(x => x.id === c.module);
  if (!d || m.lost.length) { res('also set').skipped++; continue; }
  if (!c.text && false) continue;
  const on = new Set<string>([...DEFINITIONS.filter(x => x.kind === 'module' && x.builtIn).map(x => x.id), c.module, ...c.pedals]);
  const a = alsoSet(ix, d, deskOf(e, m, { tier: c.tier as Tier, on }));
  verdict('also set', J(a ? [a.text, a.link] : null) === J(c.text ? [c.text, c.link] : null), `${e} ${c.tier} ${c.module} ${J(c.vals)}: page ${J(c.text)} → ${c.link} | core ${J(a?.text ?? null)} → ${a?.link}`);
}

/* ---- what each tile lets you change at each tier: the page's controls against editsAt */
for (const t of FIX.tiers) {
  if (!t.keys) continue;
  const e = ENG[t.engine]!, d = DEFINITIONS.find(x => x.id === t.module);
  if (!d) { res('tier contents').skipped++; continue; }
  const m = mapVals(t.vals, e);
  const on = new Set<string>([...DEFINITIONS.filter(x => x.kind === 'module' && x.builtIn).map(x => x.id), t.module, ...(d.kind === 'module' && d.labUnder ? [d.labUnder] : [])]);
  const desk0 = deskOf(e, m, { tier: t.tier as Tier, on, state: t.slots.length ? { slots: t.slots } : {} });
  const desk = { ...desk0, facts: factsOf(ix, desk0) };
  const mine = [...editsAt(ix, d, desk)].sort(), page = (t.keys as string[]).filter(k => ix.params.has(k) && has(ix, k, e)).sort();
  const extra = mine.filter(k => !page.includes(k)), missing = page.filter(k => !mine.includes(k));
  verdict('tier contents', !extra.length && !missing.length, `${e} ${t.tier} ${t.module}${t.loaded ? ' loaded' : ''}: only the page ${J(missing)}, only the definitions ${J(extra)}`);
}

/* ---- engine switches */
for (const w of FIX.switches) {
  const e0 = ENG[w.desk.engine]!, m = mapVals(keep(w.desk.vals, e0), e0);
  const builtIn = new Set(DEFINITIONS.filter(x => x.kind === 'module' && x.builtIn).map(x => x.id));
  let desk = deskOf(e0, m, { state: w.desk.slots.length ? { slots: w.desk.slots } : {}, on: builtIn }), book: EngineBook = emptyBook();
  const lost = new Set(Object.keys(w.desk.vals).filter(k => !(k in m.vals) && !k.endsWith('.auto')));
  for (const h of w.hops) {
    const to = ENG[h.to]!, r = switchEngine(ix, desk, book, to);
    desk = r.desk; book = r.book;
    const want = mapVals(Object.fromEntries(Object.entries(h.vals).filter(([k]) => !lost.has(k))), to);
    const got = { vals: desk.vals, auto: desk.auto };
    const hold = Object.fromEntries(Object.entries(h.hold as Record<string, { v: number; shown: number; from: string }>).map(([k, x]) => [k, { value: x.v, shown: x.shown, from: ENG[x.from] }]));
    /* the toast counts what stays behind, so it is compared only when the definitions can say every value */
    const keys = [...new Set([...Object.keys(got.vals), ...Object.keys(want.vals)])].filter(k => J(got.vals[k]) !== J(want.vals[k]));
    verdict('engine switch', !keys.length && J(sorted(got.auto)) === J(sorted(want.auto)) && J(sorted(book.hold)) === J(sorted(hold)),
      `${desk.engine}: values ${keys.map(k => `${k} page ${J(want.vals[k])} core ${J(got.vals[k])}`).join(', ')} holds page ${J(hold)} core ${J(book.hold)}`);
    /* the toast counts what stays behind, so it is compared only when the definitions can say every value; a count
       alone that differs is said so */
    if (!lost.size) { const n = (s: string) => s.split(/\. (?=[A-Z])/).filter(x => !x.includes(' keeps ')).join('. ').replace(/\.?$/, '.'); verdict('engine switch toast', r.toast === h.news, `${n(r.toast) === n(h.news) ? 'count only: ' : ''}page ${h.news} | core ${r.toast}`); }
  }
}
function sorted<T extends object>(o: T): T { return Object.fromEntries(Object.entries(o).sort(([a], [b]) => a.localeCompare(b))) as T; }

/* ---- what a revision stores: the page keeps what was set; the definitions keep every value, defaults included */
for (const r of FIX.revisions) {
  const e = ENG[r.engine]!, m = mapVals(keep(r.vals, e), e);
  const rev = revisionOf(ix, deskOf(e, m), 'w', 'r', r.withContent);
  const page = mapVals(keep({ ...r.rev.vals, ...(r.rev.content?.vals || {}) }, e), e);
  const bad: string[] = [];
  for (const [k, v] of Object.entries(page.vals)) if (J(rev.vals[k]) !== J(v)) bad.push(`${k}: page ${J(v)}, core ${J(rev.vals[k])}`);
  for (const [k, v] of Object.entries(rev.vals)) if (!(k in page.vals) && J(v) !== J(resolve(ix.params.get(k)!.p, e).default) && k !== 'prompt.text') bad.push(`${k}: core keeps ${J(v)}, the page nothing`);
  for (const k of Object.keys(m.vals)) if (!(k in page.vals) && k in rev.vals && J(rev.vals[k]) === J(m.vals[k]) && J(m.vals[k]) !== J(resolve(ix.params.get(k)!.p, e).default)) bad.push(`${k}: core keeps it, the page does not`);
  verdict('revision', !bad.length, `${e}${r.withContent ? ' with content' : ''}: ${bad.join('; ')}`);
}

/* ---- cleaning */
for (const c of FIX.cleaning) {
  /* the input as the page held it: a choice by its option's name, mapped where the definitions have it, else kept raw */
  const input: { [k: string]: unknown } = {}, auto: { [k: string]: unknown } = {};
  for (const [k, v] of Object.entries(c.input as Record<string, unknown>)) {
    if (k.endsWith('.auto')) { auto[k.slice(0, -5)] = v; continue; }
    const hit = ix.params.get(k), e = hit ? (hit.p.engines ?? hit.d.engines)[0]! : 'ace';
    input[k] = hit?.p.type === 'choice' ? ours(k, v, e) ?? v : v;
  }
  const got = cleanValues(ix, input, auto);
  const want: { [k: string]: unknown } = {}, wantAuto: { [k: string]: unknown } = {};
  for (const [k, v] of Object.entries(c.vals as Record<string, unknown>)) { if (k.endsWith('.auto')) wantAuto[k.slice(0, -5)] = v; else { const hit = ix.params.get(k); want[k] = hit?.p.type === 'choice' ? ours(k, v, (hit.p.engines ?? hit.d.engines)[0]!) ?? v : v; } }
  verdict('cleaning', J(sorted(got.vals)) === J(sorted(want)) && J(sorted(got.auto)) === J(sorted(wantAuto)), `${J(c.input).slice(0, 200)}: page ${J(sorted(want))} ${J(wantAuto)} | core ${J(sorted(got.vals))} ${J(got.auto)}`);
}

/* ================================================================ the parts' contract, by construction */

const contract: string[] = [];
const expectThrow = (what: string, fn: () => void) => { try { fn(); contract.push(`does not refuse: ${what}`); } catch { /* refused, as it must be */ } };
{
  const adapters = DEFINITIONS.find(d => d.id === 'adapters')!, cart = adapters.kind === 'module' ? adapters.face[0]! : null;
  const desk: Desk = { engine: 'ace', tier: 'quick', vals: {}, auto: {}, state: { slots: [{ file: 'a.safetensors', type: 'LoRA' }] }, facts: { adapters: 1 }, on: new Set(ALL) };
  const changes: Change[] = [];
  const props = bindPart(ix, adapters, cart!, desk, c => changes.push(c));
  /* loading a second adapter while Stack is Single stacks them, in the same step, named by the definition */
  props.state.slots!.apply('load', [{ file: 'a.safetensors', type: 'LoRA' }, { file: 'b.safetensors', type: 'LoRA' }], { 'ad.adv': 'advanced' }, { file: 'b.safetensors' });
  if (changes.length !== 1 || changes[0]!.words.label !== 'Load b.safetensors' || changes[0]!.vals['ad.adv'] !== 'advanced') contract.push(`the load step: ${J(changes)}`);
  expectThrow('a value its node does not bind (Quick binds s0 only)', () => props.params['ad.trigger']!.set('x'));
  expectThrow('a value the step does not list', () => props.state.slots!.apply('load', [], { 'ad.mode': 'merge' }));
  expectThrow('a step its state does not name', () => props.state.slots!.apply('swap', []));
  expectThrow('a named step on a parameter that has none', () => props.params.s0!.set(0.5, 'boost'));
  /* the lyric sheet reads the tempo in force: its value, its auto switch, and whether Tempo is on the desk */
  const lyrics = DEFINITIONS.find(d => d.id === 'lyrics')!;
  const sheet = bindPart(ix, lyrics, faceOf(lyrics)[0]!, { ...desk, tier: 'studio', on: new Set([...ALL].filter(x => x !== 'tempo')), vals: { 'tempo.bpm': 90 }, auto: { 'tempo.bpm': false } }, () => {});
  const bpm = sheet.reads['tempo.bpm'];
  if (!bpm || bpm.value !== 90 || bpm.auto || bpm.onDesk) contract.push(`the lyric sheet's tempo: ${J(bpm)}`);
  /* the prompt's named steps are its definition's words */
  const prompt = DEFINITIONS.find(d => d.id === 'prompt')!, got: Change[] = [];
  const box = bindPart(ix, prompt, faceOf(prompt)[0]!, { ...desk, engine: 'sa3' }, c => got.push(c));
  box.params['prompt.text']!.set('lofi, 120 BPM', 'addChip', { chip: '120 BPM' });
  bindPart(ix, prompt, faceOf(prompt)[0]!, desk, c => got.push(c)).params['prompt.text']!.set('a song', 'tempo', { bpm: 96 }, { 'tempo.bpm': 96 });
  if (got[0]?.words.label !== 'Add 120 BPM to the prompt' || got[1]?.words.label !== 'Set the tempo to 96 BPM' || got[1]?.vals['tempo.bpm'] !== 96) contract.push(`the prompt's steps: ${J(got.map(c => c.words.label))}`);
  /* an XY pad's drag is one step naming both values */
  const planner = DEFINITIONS.find(d => d.id === 'planner')!, pads: Change[] = [];
  const walk = (n: { part: string; children?: unknown[] }): { part: string } | undefined => n.part === 'xyPad' ? n : (n.children as never[] || []).map(walk).find(Boolean);
  const pad = faceOf(planner).map(n => walk(n as never)).find(Boolean)!;
  const xy = bindPart(ix, planner, pad as never, { ...desk, tier: 'lab' }, c => pads.push(c));
  xy.batch(() => { for (const k of Object.keys(xy.params)) xy.params[k]!.set(resolve(ix.params.get(k)!.p, 'ace').range![1]); });
  if (pads.length !== 1 || !pads[0]!.words.label.includes(', ')) contract.push(`one step for the pad: ${J(pads.map(c => c.words.label))}`);
}

/* ================================================================ translation keys */

const strings: string[] = [];
{
  const all = DEFINITIONS.flatMap(stringsOf), keys = new Map<string, string>();
  for (const s of all) { if (keys.has(s.key) && keys.get(s.key) !== s.text) strings.push(`two strings share ${s.key}`); keys.set(s.key, s.text); }
  /* the lists whose order is a layout, not a meaning: lines, options, nodes, children, strips, actions, choices */
  const LISTS = new Set(['per', 'options', 'face', 'children', 'strip', 'idle', 'actions', 'choices', 'include']);
  const rev = (x: unknown, k = ''): unknown => Array.isArray(x) ? (LISTS.has(k) ? [...x].reverse() : x).map(v => rev(v)) : x && typeof x === 'object' ? Object.fromEntries(Object.entries(x).map(([kk, v]) => [kk, rev(v, kk)])) : x;
  const again = new Map(DEFINITIONS.flatMap(d => stringsOf(rev(d) as Definition)).map(s => [s.key, s.text]));
  for (const [k, t] of keys) if (again.get(k) !== t) strings.push(`reordering moves ${k} ("${t.slice(0, 40)}") to ${J(again.get(k))}`);
  if (!strings.length) strings.push(`ok: ${keys.size} keys, none shared, none moved by reordering`);
}

/* ================================================================ reading the real fields back */

const back: string[] = [];
{
  let n = 0, ok = 0;
  for (const e of ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'] as EngineId[]) {
    for (const { p } of ix.params.values()) {
      if (!has(ix, p.key, e)) continue;
      const f = resolve(p, e), v = p.type === 'number' ? f.range![1] : p.type === 'boolean' ? !f.default : p.type === 'choice' && Array.isArray(f.options) ? f.options[f.options.length - 1]!.value : undefined;
      if (v === undefined) continue;
      /* what a saved request holds and the reader reads: not the intent's own parts, a conditional field, a list's
         entry, or an option a map leaves to another parameter */
      if (wiresOf(f.wire).some(w => /^(intent|ui|model):|\[\d+\]/.test(toOf(w)) || (typeof w === 'object' && (!!w.when || (!!w.map && !(String(v) in w.map)))))) continue;
      const desk: Desk = { engine: e, tier: 'lab', vals: { [p.key]: v }, auto: f.auto ? { [p.key]: false } : {}, state: {}, facts: {}, on: new Set(ALL) };
      if (!realState(ix, desk).from || !Object.keys(realState(ix, desk).from).some(t => realState(ix, desk).from[t]!.includes(p.key))) continue;
      const rs = realState(ix, desk);
      const r = fromReal(ix, e, { ...rs.params, ...rs.backendParams, ...rs.content, ...rs.body } as { [k: string]: Json });
      n++; if (J(r[p.key]) === J(valueOf(ix, p.key, desk))) ok++; else if (back.length < 12) back.push(`${p.key} [${e}]: sent ${J(v)}, read back ${J(r[p.key])}`);
    }
  }
  back.unshift(`${ok} of ${n} values read back from what they send`);
}

/* ================================================================ saved records of older shapes */

const saved: string[] = [];
{
  /* the concept's own revision, as its page saved it (version 0): engine as `backend`, auto switches among the values,
     content apart, slots at the top */
  const old = { name: 'Tape', backend: 'ace', wf: 'Lo-fi tape', vals: { 'samp.steps': 30, 'tempo.bpm': 82, 'tempo.bpm.auto': false, 'gone.key': 3 }, slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }], content: { vals: { 'lyrics.text': 'la' }, prompt: 'lo-fi' } };
  const r = migrate(old);
  if (!r || r.kind !== 'revision') saved.push('the concept\'s revision is not read as a revision');
  else {
    const loaded = loadRevision(ix, { engine: 'ace', tier: 'lab', vals: {}, auto: {}, state: {}, facts: {}, on: new Set(ALL) }, old);
    const d = loaded.desk;
    if (d.vals['samp.steps'] !== 30 || d.vals['tempo.bpm'] !== 82 || d.auto['tempo.bpm'] !== false || d.vals['prompt.text'] !== 'lo-fi' || d.vals['lyrics.text'] !== 'la' || J(d.state.slots) !== J(old.slots)) saved.push(`the concept's revision loads as ${J(d)}`);
    if (loaded.told !== '1 setting could not be used') saved.push(`its toast says ${J(loaded.told)}`);
  }
  /* a desk saved under a former id, key and state name reads as the renamed ones */
  const renamed = DEFINITIONS.map(d => d.id !== 'ss' ? d : { ...d, id: 'stablestep', was: ['ss'], params: d.params.map(p => p.key !== 'pp.ss.str' ? p : { ...p, key: 'pp.ss.refine', was: ['pp.ss.str'] }) }) as Definition[];
  const ix2 = indexOf(renamed);
  const desk = readDesk(ix2, { v: 1, engine: 'ace', tier: 'lab', modules: [], board: ['ss'], vals: { 'pp.ss.str': 40 }, auto: {}, state: {}, stash: {}, hold: {} });
  if (!desk || !desk.desk.on.has('stablestep') || desk.desk.vals['pp.ss.refine'] !== 40 || desk.told) saved.push(`a desk under former names reads as ${J(desk && { on: [...desk.desk.on], vals: desk.desk.vals, told: desk.told })}`);
  /* a record of a newer shape is not read */
  if (migrate({ v: 99, name: 'x', tier: 'lab', modules: [], board: [] })) saved.push('a record of a newer shape is read');
  if (!saved.length) saved.push('ok: the concept\'s revision, former ids, keys and a newer shape');
}

/* ================================================================ report */

let bad = 0;
console.log('derivations against the concept page (tools/fixtures/concept.json):');
for (const [name, r] of Object.entries(results)) {
  console.log(`  ${name}: ${r.cases} cases, ${r.agree} agree, ${r.accepted} differ as accepted, ${r.fails.length} differ${r.skipped ? `; ${r.skipped} not comparable (a server-listed choice)` : ''}`);
  if (r.fails.length) { bad += r.fails.length; (process.argv.includes('--print') ? r.fails : r.fails.slice(0, 4)).forEach(f => console.log('    ' + f.slice(0, 600))); }
}
const gone = ACCEPT.filter((_, i) => !used.has(i));
gone.forEach(a => { bad++; console.log(`  gone: ${a.derivation} /${a.match}/ no longer differs; take it out of concept-differences.json`); });
console.log(`parts' contract: ${contract.length ? contract.join('; ') : 'ok'}`); bad += contract.length;
console.log(`translation keys: ${strings.join('; ')}`); if (!strings[0]!.startsWith('ok')) bad++;
console.log(`real fields read back: ${back.join('; ')}`); if (back.length > 1) bad++;
console.log(`saved records: ${saved.join('; ')}`); if (!saved[0]!.startsWith('ok')) bad++;
process.exit(bad ? 1 : 0);
