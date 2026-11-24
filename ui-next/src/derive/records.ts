// Saved records (docs/05, "Saved data"): what a revision and a workflow hold, loading one, the desk autosave, and
// reading a record of an older shape (migration). Every path that reads a record cleans it the one way (values.ts).

import type { Draft, EngineId, Json, Revision, SavedDesk, Tier, Value, Workflow } from '../definitions/types.ts';
import { SAVED_VERSION } from '../definitions/types.ts';
import { ENGINE_IDS, engineName } from '../definitions/engines.ts';
import { RETIRED } from '../definitions/retired.ts';
import { type Desk, type Index, enginesOf, has, resolve } from './core.ts';
import { cleanStates, cleanValues } from './values.ts';
import { type Msg, msg } from './messages.ts';
import type { EngineBook } from './engine.ts';

const storedOf = (ix: Index, key: string) => { const h = ix.params.get(key)!; return h.p.stored ?? h.d.stored; };

/** What is on the desk, of one kind, in the desk's order: `on` keeps the order things were put on the desk in (a Set
 *  keeps its insertion order), which is the order the tiles read in; a record keeps it. */
const onOrder = (ix: Index, desk: Desk, kind: (d: Index['defs'][number]) => boolean): string[] =>
  [...desk.on].filter(id => ix.defs.some(d => d.id === id && kind(d)));

/** A revision of the desk's engine: every parameter on that engine with stored 'settings' (and 'content' with
 *  content), its value in force and its auto switch, defaults included, and the state with stored 'settings'. A
 *  source module's values (a repaint region's) are never in it. */
export function revisionOf(ix: Index, desk: Desk, workflow: string, name: string, withContent = false): Revision {
  const vals: { [k: string]: Value } = {}, auto: { [k: string]: boolean } = {}, state: { [k: string]: Json } = {};
  for (const { p, d } of ix.params.values()) {
    if (!has(ix, p.key, desk.engine) || (d.kind === 'module' && d.source)) continue;
    const s = storedOf(ix, p.key);
    if (!(s === 'settings' || (withContent && s === 'content'))) continue;
    vals[p.key] = p.key in desk.vals ? desk.vals[p.key]! : resolve(p, desk.engine).default;
    if (resolve(p, desk.engine).auto) auto[p.key] = desk.auto[p.key] !== false;
  }
  for (const { s, d } of ix.states.values()) if (s.stored === 'settings' && d.engines.includes(desk.engine) && desk.state[s.key] !== undefined) state[s.key] = desk.state[s.key]!;
  /* written as it reads back (cleanRevision): no empty auto or state (RC-02) */
  return { v: SAVED_VERSION, workflow, name, engine: desk.engine, vals, ...(Object.keys(auto).length ? { auto } : {}), ...(Object.keys(state).length ? { state } : {}), ...(withContent ? { withContent: true as const } : {}) };
}

/** A draft: one song's content for the desk's engine. */
export function draftOf(ix: Index, desk: Desk, name: string): Draft {
  const vals: { [k: string]: Value } = {};
  for (const { p } of ix.params.values()) if (has(ix, p.key, desk.engine) && storedOf(ix, p.key) === 'content' && p.key in desk.vals) vals[p.key] = desk.vals[p.key]!;
  return { v: SAVED_VERSION, name, engine: desk.engine, vals };
}

/** A workflow: the desk's layout, never a source module. */
export function workflowOf(ix: Index, desk: Desk, name: string): Workflow {
  const mods = onOrder(ix, desk, d => d.kind === 'module' && !d.builtIn && !d.source);
  const board = onOrder(ix, desk, d => d.kind !== 'module');
  return { v: SAVED_VERSION, name, tier: desk.tier, modules: mods, board };
}

/** Loading a revision: every parameter a revision keeps for its engine is reset, then its values (cleaned) applied;
 *  its state replaces the revision-kept state. `told` is what the toast says could not be used. */
export function loadRevision(ix: Index, desk: Desk, rev: unknown): { desk: Desk; told: Msg | null } {
  const r = migrate(rev);
  if (!r || r.kind !== 'revision') return { desk, told: msg('load.unreadable') };
  const rec = r.record as Revision, e = rec.engine;
  /* a revision is one engine's settings: on another engine's desk it is refused with a word, never applied */
  if (e !== desk.engine) return { desk, told: msg('load.engine', { engine: ENGINE_IDS.includes(e) ? engineName(e) : String(e) }) };
  const out: Desk = { ...desk, vals: { ...desk.vals }, auto: { ...desk.auto }, state: { ...desk.state } };
  for (const { p } of ix.params.values()) {
    if (!has(ix, p.key, e) || ix.params.get(p.key)!.d.kind === 'module' && (ix.params.get(p.key)!.d as { source?: true }).source) continue;
    const s = storedOf(ix, p.key);
    /* content a revision holds as empty text replaces nothing: no saved lyrics or caption means yours stay */
    if (s === 'settings' || (rec.withContent && s === 'content' && rec.vals[p.key] !== '')) { delete out.vals[p.key]; delete out.auto[p.key]; }
  }
  const c = cleanValues(ix, rec.vals, rec.auto, RETIRED.keys), st = cleanStates(ix, rec.state, RETIRED.states);
  for (const k of Object.keys(c.vals)) if (c.vals[k] === '' && storedOf(ix, k) === 'content') delete c.vals[k];
  Object.assign(out.vals, c.vals); Object.assign(out.auto, c.auto);
  for (const { s } of ix.states.values()) if (s.stored === 'settings') delete out.state[s.key];
  Object.assign(out.state, st.state);
  const n = c.dropped.length + st.dropped.length;
  return { desk: out, told: n ? msg('load.told', { n }) : null };
}

/** The desk autosave: everything, for a reload or another tab. */
export function savedDeskOf(ix: Index, desk: Desk, book: EngineBook): SavedDesk {
  const mods = onOrder(ix, desk, d => d.kind === 'module' && !d.builtIn);
  const board = onOrder(ix, desk, d => d.kind !== 'module');
  return { v: SAVED_VERSION, engine: desk.engine, tier: desk.tier, modules: mods, board, vals: { ...desk.vals }, auto: { ...desk.auto }, state: { ...desk.state }, stash: structuredClone(book.stash), hold: structuredClone(book.hold) };
}

/** Reading a saved desk: everything checked as it comes in, engine and tier included; held values with a number, a
 *  known engine and a parameter that has `held`. */
export function readDesk(ix: Index, raw: unknown): { desk: Desk; book: EngineBook; told: Msg | null } | null {
  const r = migrate(raw);
  if (!r || r.kind !== 'desk') return null;
  const x = r.record as SavedDesk;
  const engine: EngineId = ENGINE_IDS.includes(x.engine) ? x.engine : 'ace';
  const tier: Tier = (['quick', 'studio', 'lab'] as const).includes(x.tier) ? x.tier : 'quick';
  const ids = (list: unknown, kind: 'module' | 'board') => (Array.isArray(list) ? list : []).map(id => (typeof id === 'string' ? ix.formerIds.get(id) ?? id : null))
    .filter((id): id is string => !!id && ix.defs.some(d => d.id === id && (kind === 'module' ? d.kind === 'module' : d.kind !== 'module')));
  const on = new Set([...ix.defs.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id), ...ids(x.modules, 'module'), ...ids(x.board, 'board')]);
  const c = cleanValues(ix, x.vals, x.auto, RETIRED.keys), st = cleanStates(ix, x.state, RETIRED.states);
  const book: EngineBook = { stash: {}, hold: {}, lastOn: {} };
  for (const [e, s] of Object.entries(x.stash && typeof x.stash === 'object' ? x.stash : {})) {
    if (!ENGINE_IDS.includes(e as EngineId)) continue;
    const cv = cleanValues(ix, s?.vals, s?.auto, RETIRED.keys), cs = cleanStates(ix, s?.state, RETIRED.states);
    const own = (k: string) => { const h = ix.params.get(k)!; return (h.p.owner ?? h.d.owner) === 'engine'; };
    book.stash[e] = { vals: Object.fromEntries(Object.entries(cv.vals).filter(([k]) => own(k))), auto: Object.fromEntries(Object.entries(cv.auto).filter(([k]) => own(k))), state: cs.state };
  }
  for (const [k, h] of Object.entries(x.hold && typeof x.hold === 'object' ? x.hold : {})) {
    const p = ix.params.get(k)?.p;
    if (p?.held && h && typeof h.value === 'number' && Number.isFinite(h.value) && typeof h.shown === 'number' && ENGINE_IDS.includes(h.from)) book.hold[k] = { value: h.value, shown: h.shown, from: h.from };
  }
  const n = c.dropped.length + st.dropped.length;
  return { desk: { engine, tier, vals: c.vals, auto: c.auto, state: st.state, on }, book, told: n ? msg('load.told', { n }) : null };
}

/* ---------------------------------------------------------------- migration
   A record says the version of its shape in `v`. Reading one of an older shape runs each step from its version up to
   SAVED_VERSION, in order; a record of a newer or unknown shape is kept aside unread (the page says so once). Renamed
   keys, ids, state and option values are not migration steps: `was` reads them in every version, and RETIRED drops
   what is gone without a word. A step is added here only when the shape of a record changes. */

type Kind = 'workflow' | 'revision' | 'draft' | 'desk';
const kindOf = (r: Record<string, unknown>): Kind | null =>
  'stash' in r || 'hold' in r ? 'desk' : 'workflow' in r && 'engine' in r ? 'revision' : 'modules' in r && 'board' in r ? 'workflow' : 'engine' in r && 'vals' in r ? 'draft' : null;

/** Steps from one shape to the next, by the version they read. Version 0 is the concept's own saved shape (its
 *  workflows, revisions and desks before this format), so a record exported from the concept can be read too. */
const STEPS: { [from: number]: (r: Record<string, unknown>) => Record<string, unknown> } = {
  0: r => {
    /* the concept keeps an auto switch as "<key>.auto" among the values, the engine as `backend` (mm3, yue, hm), the
       layout as `mods` and `pedals`, and a revision's content under `content` */
    const ENG: Record<string, EngineId> = { ace: 'ace', mm3: 'minimax-m3', yue: 'yue2', sa3: 'sa3', hm: 'heartmula' };
    const out: Record<string, unknown> = { ...r, v: 1 };
    const split = (vals: unknown) => {
      const v: Record<string, unknown> = {}, a: Record<string, unknown> = {};
      for (const [k, x] of Object.entries(vals && typeof vals === 'object' ? vals : {})) (k.endsWith('.auto') ? (a[k.slice(0, -5)] = x) : (v[k] = x));
      return { v, a };
    };
    if ('backend' in r) { out.engine = ENG[String(r.backend)] ?? r.backend; delete out.backend; }
    if ('mods' in r) { out.modules = r.mods; delete out.mods; }
    if ('pedals' in r) { out.board = r.pedals; delete out.pedals; }
    const content = r.content as { vals?: unknown; prompt?: unknown } | undefined;
    const { v, a } = split({ ...(r.vals as object), ...(content?.vals as object) });
    if (content && typeof content.prompt === 'string') v['prompt.text'] = content.prompt;
    if (content) { out.withContent = true; delete out.content; }
    if ('slots' in r) { out.state = { ...(out.state as object), slots: r.slots }; delete out.slots; }
    if ('wf' in r) { out.workflow = r.wf; delete out.wf; }
    out.vals = v; out.auto = { ...(r.auto as object), ...a };
    return out;
  },
};

/** A record read in the current shape: its kind and the record, or null when it cannot be read. */
export function migrate(raw: unknown): { kind: Kind; record: Workflow | Revision | Draft | SavedDesk } | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  let r = raw as Record<string, unknown>;
  let v = typeof r.v === 'number' ? r.v : 0;
  if (v > SAVED_VERSION || !Number.isInteger(v) || v < 0) return null;
  while (v < SAVED_VERSION) { const step = STEPS[v]; if (!step) return null; r = step(r); v = r.v as number; }
  const kind = kindOf(r);
  return kind ? { kind, record: r as unknown as Workflow } : null;
}

/** The engine a stored key is on, for a revision of `e`. */
export const keptOn = (ix: Index, key: string, e: EngineId) => { const h = ix.params.get(key); return !!h && enginesOf(h.p, h.d).includes(e); };
