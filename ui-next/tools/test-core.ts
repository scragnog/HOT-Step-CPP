// The core's derivations against the concept page: undo step names, each tier's "Also set" line, engine switches,
// what a revision stores and what cleaning keeps, each compared with what the page did in the same state
// (tools/fixtures/concept.json, recorded from the concept page before the cutover; its collector is in the tag
// rebuild-parity). Then the parts' contract (bindPart), the translation keys, reading the real fields back, and saved
// records of older shapes.
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
import type { Definition, EngineId, Json, Node, Tier, Value } from '../src/definitions/types.ts';
import {
  alsoSet, applies, bindPart, changed, coreStrings, cleanValues, editsAt, emptyBook, everythingOn, fromReal, savedDeskOf, stripOf, has, indexOf, loadRevision, migrate, readDesk, realState, resolve, toOf, wiresOf,
  revisionOf, stepWords, stringsOf, switchEngine, valueOf, type Change, type Desk, type EngineBook,
} from '../src/derive.ts';
import { say, type Slot } from '../src/derive/messages.ts';
import { applyChoice, choiceOf, hiddenLabModules, loadDraft, loadWorkflow } from '../src/derive/view.ts';
import { cleanState, setValue } from '../src/derive/values.ts';
import { FACTORY_REVISIONS } from '../src/definitions/factory.ts';
import { RETIRED } from '../src/definitions/retired.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const FIX = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools', 'fixtures', 'concept.json'), 'utf8'));
const ACCEPT_FILE = path.join(ROOT, 'tools', 'fixtures', 'concept-differences.json');
const REAL = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools', 'fixtures', 'real-app.json'), 'utf8'));
/** An accepted disagreement: the derivation, a pattern the case (its state and both outputs) matches, and, to name
 *  exactly what the core says instead, either `rewrite` (replacements that turn the page's output into the core's,
 *  which must then be equal) or `core` (a pattern the core's output must match). */
const ACCEPT: { derivation: string; match: string; rewrite?: [string, string][]; core?: string; why: string; n?: number }[] = fs.existsSync(ACCEPT_FILE) ? JSON.parse(fs.readFileSync(ACCEPT_FILE, 'utf8')).differences : [];
const ix = indexOf(DEFINITIONS);
/* the definitions as the concept page had them: every parameter added since left out (for the undo names a new
   parameter's label renames: RA-16) */
let conceptIx: ReturnType<typeof indexOf> | null = null;
const ixConcept = (keys: Set<string>) => (conceptIx ??= indexOf(DEFINITIONS.map(d => ({ ...d, params: d.params.filter(p => keys.has(p.key) || (d.kind === 'module' && d.builtIn)) })) as typeof DEFINITIONS));
const newParams = { exact: false, n: 0 };
const pickOf = (x: { label?: Slot; undone?: Slot; redone?: Slot; put?: Slot } | null) => x && { label: x.label === undefined ? undefined : say(x.label), undone: x.undone === undefined ? undefined : say(x.undone), redone: x.redone === undefined ? undefined : say(x.redone), put: x.put === undefined ? undefined : say(x.put) };
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
  /* an option of a fixed list the page shows and the definitions do not name: a misspelt label, never left out */
  if (Array.isArray(f.options)) unlabelled.add(`${e} ${k}: the page's option ${J(v)} is none of the definition's labels (${f.options.map(x => x.label).join(', ')})`);
  /* a plugin the server lists: its id is its name in lower case, letters and digits ("DPM++ 3M" is dpm3m) */
  const from = !Array.isArray(f.options) && f.options?.from.startsWith('plugins.') ? REAL.plugins[f.options.from.slice(8)] as string[] : [];
  return from.find(id => norm(id) === norm(String(v)));
}
const unlabelled = new Set<string>();
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
const deskOf = (e: EngineId, m: Mapped, extra: Partial<Desk> = {}): Desk => ({ engine: e, tier: 'lab', vals: m.vals, auto: m.auto, state: {}, on: new Set(ALL), lists: LISTS, ...extra });

/* ================================================================ compare, count, report */

interface Result { cases: number; agree: number; skipped: number; accepted: number; fails: string[] }
const results: Record<string, Result> = {};
const res = (name: string) => (results[name] ||= { cases: 0, agree: 0, skipped: 0, accepted: 0, fails: [] });
const used = new Set<number>(), absorbed = new Map<number, number>();
function verdict(name: string, same: boolean, what: string, out?: { page: string; core: string }): void {
  const r = res(name); r.cases++;
  if (same) { r.agree++; return; }
  /* a step that names a new parameter's owner because the page does not have the parameter (above) is right only when
     the core without the parameters added since says exactly what the page says (`exactWithout`): counted apart */
  if (what.startsWith('shares its words with a parameter the page does not have') && newParams.exact) { r.accepted++; newParams.n++; return; }
  const exact = (a: (typeof ACCEPT)[number]) => (!a.rewrite || (!!out && a.rewrite.reduce((s, [f, t]) => s.replace(new RegExp(f, 'g'), t), out.page) === out.core))
    && (!a.core || (!!out && new RegExp(a.core).test(out.core)));
  const i = ACCEPT.findIndex(a => a.derivation === name && new RegExp(a.match).test(what) && exact(a));
  if (i >= 0) { r.accepted++; used.add(i); absorbed.set(i, (absorbed.get(i) ?? 0) + 1); if (process.argv.includes('--accepted')) console.log(`[${i}] ${what.slice(0, 400)}`); return; }
  r.fails.push(what);
}

/* the parameters the concept page had when it was collected: one added since is new, and not compared with it (a
   parameter added by 06-how-to.md section 1 leaves the comparison as it was) */
const CONCEPT_KEYS = new Set([...Object.values<Record<string, unknown>>(FIX.controls).flatMap(c => Object.keys(c)),
  /* the built-in modules' (the prompt, the lyrics, the models) are the page's own, whatever its control list says */
  ...DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).flatMap(d => d.params.map(p => p.key))]);
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
  /* RA-16: the core on the definitions as the page had them (the parameters added since taken out) must say exactly
     the page's words, or the difference is not the new parameter's and fails as any other */
  newParams.exact = shares && J(pickOf(stepWords(ixConcept(conceptKeys), before, after))) === J(pickOf(s.words));
  const pick = pickOf;
  verdict('undo step names', J(pick(w)) === J(pick(s.words)), `${shares ? 'shares its words with a parameter the page does not have: ' : ''}${e} ${J(s.after)}: page ${J(s.words)} | core ${J(w)}`, { page: J(pick(s.words)), core: J(pick(w)) });
}

/* ---- "Also set" */
for (const c of FIX.also) {
  const e = ENG[c.engine]!, m = mapVals(c.vals, e), d = DEFINITIONS.find(x => x.id === c.module);
  if (!d || m.lost.length) { res('also set').skipped++; continue; }
  if (!c.text && false) continue;
  const on = new Set<string>([...DEFINITIONS.filter(x => x.kind === 'module' && x.builtIn).map(x => x.id), c.module, ...c.pedals]);
  const a0 = alsoSet(ix, d, deskOf(e, m, { tier: c.tier as Tier, on })), a = a0 && { ...a0, text: say(a0.text) };
  verdict('also set', J(a ? [a.text, a.link] : null) === J(c.text ? [c.text, c.link] : null), `${e} ${c.tier} ${c.module} ${J(c.vals)}: page ${J(c.text)} → ${c.link} | core ${J(a?.text ?? null)} → ${a?.link}`, { page: J(c.text ? [c.text, c.link] : null), core: J(a ? [a.text, a.link] : null) });
}

/* ---- what each tile lets you change at each tier: the page's controls against editsAt */
for (const t of FIX.tiers) {
  if (!t.keys) continue;
  const e = ENG[t.engine]!, d = DEFINITIONS.find(x => x.id === t.module);
  if (!d) { res('tier contents').skipped++; continue; }
  const m = mapVals(t.vals, e);
  /* the adapters the page had loaded are on the desk only with the Adapters tile (the value in force, docs/05) */
  const on = new Set<string>([...DEFINITIONS.filter(x => x.kind === 'module' && x.builtIn).map(x => x.id), t.module, ...(d.kind === 'module' && d.labUnder ? [d.labUnder] : []), ...(t.slots.length ? ['adapters'] : [])]);
  const desk0 = deskOf(e, m, { tier: t.tier as Tier, on, state: t.slots.length ? { slots: t.slots } : {} });
  const desk = desk0;
  const mine = [...editsAt(ix, d, desk)].sort(), page = (t.keys as string[]).filter(k => ix.params.has(k) && has(ix, k, e)).sort();
  const extra = mine.filter(k => !page.includes(k) && CONCEPT_KEYS.has(k)), missing = page.filter(k => !mine.includes(k));
  verdict('tier contents', !extra.length && !missing.length, `${e} ${t.tier} ${t.module}${t.loaded ? ' loaded' : ''}: only the page ${J(missing)}, only the definitions ${J(extra)}`);
}

/* ---- what each tile lets you change when its switches and choices are on their other values (concept-tiers.json,
   recorded from the concept page before the cutover): a module's controls against editsAt, a pedal's or unit's against the
   layout rule of docs/01 (Lab: every parameter that applies and its face's; Studio: switch, main knob and Studio face;
   Quick: its switch). A face condition on a value that disagrees with the page shows here. */
const TIERS = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools', 'fixtures', 'concept-tiers.json'), 'utf8'));
for (const t of TIERS.res as { engine: string; kind: string; id: string; tier: Tier; set?: [string, unknown]; slots?: number; parent?: string | null; strip?: boolean; keys?: string[] | null }[]) {
  const e = ENG[t.engine]!, d = DEFINITIONS.find(x => x.id === t.id);
  if (!d || !t.keys) continue;
  const vals: { [k: string]: Value } = {}, auto: { [k: string]: boolean } = {};
  if (t.set) { const v = ours(t.set[0], t.set[1], e); if (v === undefined) { res('tier contents at other values').skipped++; continue; } vals[t.set[0]] = v; if (resolve(ix.params.get(t.set[0])!.p, e).auto) auto[t.set[0]] = false; }
  const slots = [{ file: 'tape-lofi-v2', type: 'LoRA' }, { file: 'grit', type: 'LoRA' }].slice(0, t.slots ?? 0);
  const on = new Set<string>([...DEFINITIONS.filter(x => x.kind === 'module' && x.builtIn).map(x => x.id), t.id, ...(t.parent ? [t.parent] : []), ...(slots.length ? ['adapters'] : []), ...(t.kind === 'board' ? ['post', ...DEFINITIONS.filter(x => x.kind !== 'module').map(x => x.id)] : [])]);
  const desk: Desk = { engine: e, tier: t.tier, vals, auto, state: slots.length ? { slots } : {}, on, lists: LISTS };
  let mine: string[];
  if (d.kind === 'module') mine = [...editsAt(ix, d, desk)];
  else {
    const face = [...editsAt(ix, d, desk)], lab = d.params.filter(p => applies(ix, p.key, desk)).map(p => p.key);
    mine = t.tier === 'lab' ? [...lab, ...face] : t.tier === 'studio' ? [...(d.on ? [d.on] : []), ...(d.kind === 'pedal' && d.main ? [d.main] : []), ...face] : [...(d.on ? [d.on] : []), ...(d.kind === 'unit' ? face : [])];
  }
  mine = [...new Set(mine)].filter(k => has(ix, k, e) && !(d.kind !== 'module' && k === d.on)).sort();
  const page = t.keys.filter(k => ix.params.has(k) && has(ix, k, e)).sort();
  const onlyPage = page.filter(k => !mine.includes(k)), onlyDefs = mine.filter(k => !page.includes(k) && CONCEPT_KEYS.has(k));
  const strip = d.kind === 'module' && !!stripOf(ix, d, desk) !== !!t.strip;
  verdict('tier contents at other values', !onlyPage.length && !onlyDefs.length && !strip, `${e} ${t.tier} ${t.id}${t.slots ? ` (${t.slots} adapters)` : ''} ${J(t.set)}: only the page ${J(onlyPage)}, only the definitions ${J(onlyDefs)}${strip ? `, a strip on the page ${!!t.strip}` : ''}`);
}

/* ---- engine switches */
for (const w of FIX.switches) {
  const e0 = ENG[w.desk.engine]!, m = mapVals(keep(w.desk.vals, e0), e0);
  const builtIn = new Set(DEFINITIONS.filter(x => x.kind === 'module' && x.builtIn).map(x => x.id));
  let desk = deskOf(e0, m, { state: w.desk.slots.length ? { slots: w.desk.slots } : {}, on: builtIn }), book: EngineBook = emptyBook();
  const lost = new Set(Object.keys(w.desk.vals).filter(k => !(k in m.vals) && !k.endsWith('.auto')));
  for (const h of w.hops) {
    const desk0Engine = desk.engine, before = desk, to = ENG[h.to]!, r0 = switchEngine(ix, desk, book, to), r = { ...r0, toast: say(r0.toast) };
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
    /* "count only": the toasts differ only in how many settings the engine left keeps, the core's fewer (a value whose
       condition does not hold is not changed here); its models and adapters must be the page's */
    if (!lost.size) {
      const n = (s: string) => s.split(/\. (?=[A-Z])/).filter(x => !x.includes(' keeps ')).join('. ').replace(/\.?$/, '.');
      const keeps = (s: string) => /keeps ([^.]*)\./.exec(s)?.[1] ?? '', count = (k: string) => +(/(\d+) settings?/.exec(k)?.[1] ?? 0), rest = (k: string) => k.replace(/(^|,| and) ?\d+ settings?/, '');
      const countOnly = n(r.toast) === n(h.news) && rest(keeps(r.toast)) === rest(keeps(h.news)) && count(keeps(r.toast)) < count(keeps(h.news));
      /* "lists differ": the core counts the changed settings only the engine left has, by the definitions' engine
         lists, the page by its own wider ones, and the page skips a control with no label of its own. Its models and
         adapters must be the page's, and its count may differ from the page's by no more than the number of those
         settings */
      const only = Object.keys(before.vals).filter(k => has(ix, k, desk0Engine) && !has(ix, k, to) && changed(ix, k, everythingOn(ix, before)));
      const listsDiffer = only.length > 0 && n(r.toast) === n(h.news) && rest(keeps(r.toast)) === rest(keeps(h.news)) && Math.abs(count(keeps(r.toast)) - count(keeps(h.news))) <= only.length;
      verdict('engine switch toast', r.toast === h.news, `${countOnly ? 'count only: ' : listsDiffer ? 'lists differ: ' : ''}${J(Object.keys(w.desk.vals))} page ${h.news} | core ${r.toast}`);
    }
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
  /* a key removed on purpose (retired.ts) is dropped by cleaning, where the page, which still has it, kept it */
  for (const [k, v] of Object.entries(c.vals as Record<string, unknown>)) { if (RETIRED.keys.includes(k.replace(/\.auto$/, ''))) continue; if (k.endsWith('.auto')) wantAuto[k.slice(0, -5)] = v; else { const hit = ix.params.get(k); want[k] = hit?.p.type === 'choice' ? ours(k, v, (hit.p.engines ?? hit.d.engines)[0]!) ?? v : v; } }
  verdict('cleaning', J(sorted(got.vals)) === J(sorted(want)) && J(sorted(got.auto)) === J(sorted(wantAuto)), `${J(c.input).slice(0, 200)}: page ${J(sorted(want))} ${J(wantAuto)} | core ${J(sorted(got.vals))} ${J(got.auto)}`);
}

/* ================================================================ the parts' contract, by construction */

const contract: string[] = [];
const expectThrow = (what: string, fn: () => void) => { try { fn(); contract.push(`does not refuse: ${what}`); } catch { /* refused, as it must be */ } };
{
  const adapters = DEFINITIONS.find(d => d.id === 'adapters')!, cart = adapters.kind === 'module' ? adapters.face[0]! : null;
  const desk: Desk = { engine: 'ace', tier: 'quick', vals: {}, auto: {}, state: { slots: [{ file: 'a.safetensors', type: 'LoRA' }] }, on: new Set(ALL) };
  const changes: Change[] = [];
  const props = bindPart(ix, adapters, cart!, desk, c => changes.push(c));
  /* loading a second adapter while Stack is Single stacks them, in the same step, named by the definition */
  props.state.slots!.apply('load', [{ file: 'a.safetensors', type: 'LoRA' }, { file: 'b.safetensors', type: 'LoRA' }], { 'ad.adv': 'advanced' }, { file: 'b.safetensors' });
  if (changes.length !== 1 || say(changes[0]!.words.label) !== 'Load b.safetensors' || changes[0]!.vals['ad.adv'] !== 'advanced') contract.push(`the load step: ${J(changes)}`);
  expectThrow('a value its node does not bind (Quick binds s0 only)', () => props.params['ad.trigger']!.set('x'));
  expectThrow('a value the step does not list', () => props.state.slots!.apply('load', [], { 'ad.mode': 'merge' }));
  expectThrow('a step its state does not name', () => props.state.slots!.apply('swap', []));
  expectThrow('a named step on a parameter that has none', () => props.params.s0!.set(0.5, 'boost'));
  /* the lyric sheet reads the tempo in force: while Tempo is off the desk that is its default (about 120 BPM, on auto),
     whatever Tempo kept, and it says Tempo is not on the desk */
  const lyrics = DEFINITIONS.find(d => d.id === 'lyrics')!;
  const sheet = bindPart(ix, lyrics, faceOf(lyrics)[0]!, { ...desk, tier: 'studio', on: new Set([...ALL].filter(x => x !== 'tempo')), vals: { 'tempo.bpm': 90 }, auto: { 'tempo.bpm': false } }, () => {});
  const bpm = sheet.reads['tempo.bpm'];
  if (!bpm || bpm.value !== 120 || !bpm.auto || bpm.onDesk) contract.push(`the lyric sheet's tempo: ${J(bpm)}`);
  /* the prompt's named steps are its definition's words */
  const prompt = DEFINITIONS.find(d => d.id === 'prompt')!, got: Change[] = [];
  const box = bindPart(ix, prompt, faceOf(prompt)[0]!, { ...desk, engine: 'sa3' }, c => got.push(c));
  box.params['prompt.text']!.set('lofi, 120 BPM', 'addChip', { chip: '120 BPM' });
  bindPart(ix, prompt, faceOf(prompt)[0]!, desk, c => got.push(c)).params['prompt.text']!.set('a song', 'tempo', { bpm: 96 }, { 'tempo.bpm': 96 });
  if ((got[0] && say(got[0].words.label)) !== 'Add 120 BPM to the prompt' || (got[1] && say(got[1].words.label)) !== 'Set the tempo to 96 BPM' || got[1]?.vals['tempo.bpm'] !== 96) contract.push(`the prompt's steps: ${J(got.map(c => c.words.label))}`);
  /* an XY pad's drag is one step naming both values */
  const planner = DEFINITIONS.find(d => d.id === 'planner')!, pads: Change[] = [];
  const walk = (n: { part: string; children?: unknown[] }): { part: string } | undefined => n.part === 'xyPad' ? n : (n.children as never[] || []).map(walk).find(Boolean);
  const pad = faceOf(planner).map(n => walk(n as never)).find(Boolean)!;
  const xy = bindPart(ix, planner, pad as never, { ...desk, tier: 'lab' }, c => pads.push(c));
  xy.batch(() => { for (const k of Object.keys(xy.params)) xy.params[k]!.set(resolve(ix.params.get(k)!.p, 'ace').range![1]); });
  if (pads.length !== 1 || !say(pads[0]!.words.label).includes(', ')) contract.push(`one step for the pad: ${J(pads.map(c => c.words.label))}`);
}

/* ================================================================ the value in force (docs/05, "Values")
   A definition off the desk reads as its defaults in conditions, desk facts, words and the request alike, so the page
   and the request describe one song. The three cases of the third review (DH-01), and a desk read back from its
   autosave, whose desk facts are worked out when they are read (DH-07). */

const inForce: string[] = [];
{
  const ALLON = new Set(ALL), off = (...ids: string[]) => new Set([...ALL].filter(x => !ids.includes(x)));
  const say = (what: string, ok: boolean, got: unknown) => { if (!ok) inForce.push(`${what}: ${J(got)}`); };
  /* 1. the Sampler off the desk with Rescaled CFG kept, the Guidance plugin on it: APG is in force, so its settings
     apply and are sent, and the tile says nothing of Rescaled CFG */
  const g: Desk = { engine: 'ace', tier: 'lab', vals: { 'samp.guide': 'rescaled_cfg', 'apg.norm': 0.5, 'apg.mom': 0.2 }, auto: {}, state: {}, on: off('sampler') };
  const guidance = DEFINITIONS.find(d => d.id === 'guidance')!, rg = realState(ix, g);
  say('APG applies with the Sampler off the desk', applies(ix, 'apg.norm', g) && rg.params.apgNormThreshold === 0.5 && rg.params.apgMomentum === 0.2 && rg.params.guidanceMode === undefined, rg.params);
  say('the Guidance tile is not a strip', stripOf(ix, guidance, g) === null, stripOf(ix, guidance, g));
  /* 2. an adapter loaded, the Adapters tile off the desk, a trigger word typed: no adapter, no trigger word */
  const a: Desk = { engine: 'ace', tier: 'lab', vals: { 'ad.trigger': 'tape', 'prompt.text': 'lofi' }, auto: {}, state: { slots: [{ file: 'a.safetensors', type: 'LoRA' }] }, on: off('adapters') };
  const ra = realState(ix, a);
  say('no trigger word without the Adapters tile', !applies(ix, 'ad.trigger', a) && (ra.intent.compose as { loraTrigger?: string }).loraTrigger === undefined && ra.params.adapter === undefined && ra.params.adapterStack === undefined, [ra.intent, ra.params.adapter]);
  /* 3. YuE2, the Planner off the desk with "writes nothing" kept, a lead sheet typed: the planner's default (a full
     score) is in force, so the lead sheet applies and is sent */
  const y: Desk = { engine: 'yue2', tier: 'lab', vals: { 'lm.cotmode': 'off', 'yue.abc': 'X:1\nK:G\nGABc|' }, auto: {}, state: {}, on: off('planner') };
  const ry = realState(ix, y);
  say('the lead sheet is sent with the Planner off the desk', ry.backendParams.yue2Cot === undefined && typeof ry.backendParams.yue2Abc === 'string', ry.backendParams);
  /* the desk read back from its autosave sends the trigger word and the stack mode, as the desk it was saved from */
  const two: Desk = { engine: 'ace', tier: 'lab', vals: { 'ad.adv': 'advanced', 'ad.stack': 'blend', 'ad.trigger': 'tape', 'prompt.text': 'lofi' }, auto: {}, state: { slots: [{ file: 'a.safetensors', type: 'LoRA' }, { file: 'b.safetensors', type: 'LoRA' }] }, on: ALLON };
  const back = readDesk(ix, savedDeskOf(ix, two, emptyBook()))!.desk;
  say('a desk read back sends what it sent', J(realState(ix, back).intent) === J(realState(ix, two).intent) && realState(ix, back).params.adapterStackMode === realState(ix, two).params.adapterStackMode && realState(ix, two).params.adapterStackMode === 'blend', [realState(ix, back).intent, realState(ix, back).params.adapterStackMode]);
  if (!inForce.length) inForce.push('ok: the three off-desk cases and a desk read back');
}

/* ================================================================ what the view asks the core (view.ts)
   A Quick row that sets several values (the how-to's Dynamics, written here as a definition only this test uses), a
   pedal laid out, an action run, a workflow and a draft loaded, and what a part may not give the core. */

const viewTests: string[] = [];
{
  const ok = (what: string, cond: boolean, got: unknown) => { if (!cond) viewTests.push(`${what}: ${J(got)}`); };
  const dynamics = { kind: 'module', id: 'dynamics', name: 'Dynamics', desc: 'How loud.', at: 215, category: 'Chain', engines: ['ace'], owner: 'song', stored: 'settings', size: { w: 1 },
    params: [
      { key: 'dyn.target', type: 'number', label: 'Target', range: [-24, -6, 0.5], unit: 'LUFS', default: -14, wire: { none: 'x' } },
      { key: 'dyn.ceiling', type: 'number', label: 'Ceiling', range: [-3, 0, 0.1], default: -1, wire: { none: 'x' } },
    ],
    face: [{ part: 'choiceRow', label: 'Loudness', tiers: ['quick'], choices: [['Gentle', { 'dyn.target': -16, 'dyn.ceiling': -1.5 }], ['Standard', { 'dyn.target': -14, 'dyn.ceiling': -1 }], ['Loud', { 'dyn.target': -9, 'dyn.ceiling': -0.3 }]] },
      { part: 'row', tiers: ['studio', 'lab'], children: [{ part: 'knob', param: 'dyn.target' }, { part: 'knob', param: 'dyn.ceiling' }] }],
  } as unknown as Definition;
  const ix3 = indexOf([...DEFINITIONS, dynamics]), row = (dynamics as unknown as { face: Node[] }).face[0]!;
  const quick: Desk = { engine: 'ace', tier: 'quick', vals: {}, auto: {}, state: {}, on: new Set([...ALL, 'dynamics']) };
  const c0 = choiceOf(ix3, dynamics, row, quick, 'create.dynamics.n.choiceRow');
  ok('the default choice is checked', c0.checked === 1, c0);
  const loud = applyChoice(ix3, dynamics, row, quick, 'create.dynamics.n.choiceRow', 2);
  ok('a choice that sets two values is one step named by the row', say(loud.words.label) === 'Loudness: Standard → Loud' && loud.vals['dyn.target'] === -9 && loud.vals['dyn.ceiling'] === -0.3, say(loud.words.label));
  const between = { ...quick, vals: { 'dyn.target': -12 } };
  const c1 = choiceOf(ix3, dynamics, row, between, 'create.dynamics.n.choiceRow');
  ok('a value between the choices checks none and says the values', c1.checked === null && !!c1.says, [c1.checked, c1.says && say(c1.says)]);
  /* a workflow naming a retired id and an unknown one; a draft for another engine; a revision for another engine */
  const lab: Desk = { engine: 'ace', tier: 'lab', vals: {}, auto: {}, state: {}, on: new Set(ALL) };
  const wf = loadWorkflow(ix, lab, { v: 1, name: 'w', tier: 'studio', modules: ['tempo', 'nope'], board: [] });
  ok('a workflow loads its modules and says what it could not use', wf.desk.on.has('tempo') && wf.desk.tier === 'studio' && !!wf.told, [[...wf.desk.on].length, wf.told && say(wf.told)]);
  const dr = loadDraft(ix, lab, { v: 1, name: 'd', engine: 'sa3', vals: { 'prompt.text': 'x' } });
  ok('a draft for another engine is refused with a word', !!dr.told && dr.desk === lab, dr.told);
  const rv = loadRevision(ix, lab, FACTORY_REVISIONS.find(r => r.engine !== 'ace'));
  ok('a revision for another engine is refused with a word', !!rv.told && rv.desk === lab, rv.told && say(rv.told));
  ok('the hidden Lab modules are named at Studio', !!hiddenLabModules(ix, { ...lab, tier: 'studio' }), hiddenLabModules(ix, { ...lab, tier: 'studio' }));
  /* what a part may not give the core: NaN, a seed past 2^64-1; a uint64 seed is kept */
  expectThrow('a value that is not a number (NaN)', () => setValue(ix, lab, 'samp.steps', NaN));
  const seedDef = DEFINITIONS.find(d => d.id === 'seed')!;
  ok('a uint64 seed is kept', cleanState(seedDef.state![0]!, '18446744073709551615') === '18446744073709551615' && cleanState(seedDef.state![0]!, '18446744073709551616') === undefined && cleanState(seedDef.state![0]!, 9007199254740991) === 9007199254740991, null);
  if (!viewTests.length) viewTests.push('ok: a Quick row of several values, a workflow, a draft and a revision loaded, and what a part may not give');
}

/* ================================================================ translation keys */

const strings: string[] = [];
{
  const all = [...DEFINITIONS.flatMap(stringsOf), ...coreStrings()], keys = new Map<string, string>();
  for (const s of all) { if (keys.has(s.key) && keys.get(s.key) !== s.text) strings.push(`two strings share ${s.key}`); keys.set(s.key, s.text); }
  /* the lists whose order is a layout, not a meaning: lines, options, nodes, children, strips, actions, choices */
  /* (a Quick row's choices are keyed by their place: their order is the row's meaning, so it is not reordered here) */
  const LISTS = new Set(['per', 'options', 'face', 'children', 'strip', 'idle', 'actions', 'include']);
  const rev = (x: unknown, k = ''): unknown => Array.isArray(x) ? (LISTS.has(k) ? [...x].reverse() : x).map(v => rev(v)) : x && typeof x === 'object' ? Object.fromEntries(Object.entries(x).map(([kk, v]) => [kk, rev(v, kk)])) : x;
  const again = new Map([...DEFINITIONS.flatMap(d => stringsOf(rev(d) as Definition)), ...coreStrings()].map(s => [s.key, s.text]));
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
      const desk: Desk = { engine: e, tier: 'lab', vals: { [p.key]: v }, auto: f.auto ? { [p.key]: false } : {}, state: {}, on: new Set(ALL) };
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
    const loaded = loadRevision(ix, { engine: 'ace', tier: 'lab', vals: {}, auto: {}, state: {}, on: new Set(ALL) }, old);
    const d = loaded.desk;
    if (d.vals['samp.steps'] !== 30 || d.vals['tempo.bpm'] !== 82 || d.auto['tempo.bpm'] !== false || d.vals['prompt.text'] !== 'lo-fi' || d.vals['lyrics.text'] !== 'la' || J(d.state.slots) !== J(old.slots)) saved.push(`the concept's revision loads as ${J(d)}`);
    if (!loaded.told || say(loaded.told) !== '1 setting could not be used') saved.push(`its toast says ${J(loaded.told && say(loaded.told))}`);
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
  console.log(`  ${name}: ${r.cases} cases, ${r.agree} agree, ${r.accepted} differ as accepted${name === 'undo step names' && newParams.n ? ` (${newParams.n} of them name a parameter added since by its owner, and say the page's words without it)` : ''}, ${r.fails.length} differ${r.skipped ? `; ${r.skipped} not comparable (a server-listed choice)` : ''}`);
  if (r.fails.length) { bad += r.fails.length; (process.argv.includes('--print') ? r.fails : r.fails.slice(0, 4)).forEach(f => console.log('    ' + f.slice(0, 600))); }
}
/* each accepted difference covers exactly the cases it was recorded with (its n): one that now covers more or fewer is
   a change nobody has looked at (SL-13). `--recount` writes the counts, after the change has been looked at. */
if (process.argv.includes('--recount')) {
  const file = JSON.parse(fs.readFileSync(ACCEPT_FILE, 'utf8'));
  file.differences.forEach((a: { n?: number }, i: number) => { a.n = absorbed.get(i) ?? 0; });
  /* an entry that covers no case any more (what it accepted was removed) goes with the recount */
  const kept = file.differences.filter((_: unknown, i: number) => used.has(i));
  if (kept.length < file.differences.length) console.log(`concept-differences.json: ${file.differences.length - kept.length} entries that no longer occur taken out`);
  file.differences = kept;
  fs.writeFileSync(ACCEPT_FILE, JSON.stringify(file, null, 1).split(String.fromCharCode(10)).join(String.fromCharCode(13, 10)) + String.fromCharCode(13, 10));
  console.log(`concept-differences.json: ${ACCEPT.length} counts written`);
} else ACCEPT.forEach((a, i) => {
  /* more cases than recorded is a change nobody has looked at, and fails; fewer (a module removed) is said */
  const n = absorbed.get(i) ?? 0; if (a.n === undefined || !n || n === a.n) return;
  if (n > a.n) bad++;
  console.log(`  accepted: ${a.derivation} /${a.match}/ now covers ${n} cases, recorded with ${a.n}${n > a.n ? ': look at what changed, then npm run core -- --recount' : ' (fewer: npm run core -- --recount records it)'}`);
});
const gone = ACCEPT.filter((_, i) => !used.has(i));
if (!process.argv.includes('--recount')) gone.forEach(a => { bad++; console.log(`  gone: ${a.derivation} /${a.match}/ no longer differs: npm run core -- --recount takes it out`); });
console.log(`parts' contract: ${contract.length ? contract.join('; ') : 'ok'}`); bad += contract.length;
console.log(`translation keys: ${strings.join('; ')}`); if (!strings[0]!.startsWith('ok')) bad++;
console.log(`real fields read back: ${back.join('; ')}`); if (back.length > 1) bad++;
console.log(`saved records: ${saved.join('; ')}`); if (!saved[0]!.startsWith('ok')) bad++;
console.log(`the value in force: ${inForce.join('; ')}`); if (!inForce[0]!.startsWith('ok')) bad++;
console.log(`the view's selectors: ${viewTests.join('; ')}`); if (!viewTests[0]!.startsWith('ok')) bad++;
console.log(`option labels: ${unlabelled.size ? [...unlabelled].join('; ') : 'ok: every option the page shows is one of the labels the definitions give it'}`); bad += unlabelled.size;
process.exit(bad ? 1 : 0);
