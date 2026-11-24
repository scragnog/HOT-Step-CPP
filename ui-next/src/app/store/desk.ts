// The desk store: what is on the desk (modules in the user's order, the board in chain order), the tier, the engine,
// the values, undo and redo, and the autosave. Every change goes through the core: a value through the Change a part's
// setter built (bindPart, src/derive/parts.ts), which the core has already cleaned and named; an engine switch through
// switchEngine(); a record through readDesk() and savedDeskOf(). The store decides nothing about a value: it applies
// what the core returns, records it as one undo step, and puts back only what a step changed.

import { create } from 'zustand';
import type { Definition, EngineId, Tier, Value } from '../../definitions/types.ts';
import {
  addTo, applyChange, arrivalChange, boardOrder, deskStep, emptyBook, emptyDesk, engineStep, newOn, readDesk, removeFrom, removedWords, savedDeskOf, stepWords, switchEngine,
  msg, type Change, type Desk, type EngineBook, type Msg, type Slot, type StepWords,
} from '../../derive.ts';
import { BUILT_IN, defOf, ix } from '../model/index.ts';
import { undoables } from '../register/undoable.ts';
import { keepAside, tellOnBoot, writeStored } from './stored.ts';

/** Where the desk is kept between visits. */
export const DESK_KEY = 'hotstep.create.desk';
/** The engines this browser has been on (the concept's hs3-engseen): the first switch to another says what it brings. */
const SEEN_KEY = 'hotstep.create.engines';
const seenEngines = (): string[] => { try { const a = JSON.parse(localStorage.getItem(SEEN_KEY) ?? '[]'); return Array.isArray(a) ? a.filter(x => typeof x === 'string') : []; } catch { return []; } };
const seeEngine = (e: string) => { const a = seenEngines(); if (!a.includes(e)) try { localStorage.setItem(SEEN_KEY, JSON.stringify([...a, e])); } catch { /* not kept */ } };

/* ---------------------------------------------------------------- undo: what a step changed, path by path */

/** The desk as paths (a value, an auto switch, a state, the modules, the board, the engine, the engines' book), each
 *  with its JSON: a step is the paths it changed, and undoing it puts back only those that nothing has changed since. */
type Flat = Map<string, string>;
interface Parts { desk: Desk; book: EngineBook; mods: string[]; board: string[] }
function flatOf(p: Parts): Flat {
  const f: Flat = new Map();
  for (const [k, v] of Object.entries(p.desk.vals)) f.set('v:' + k, JSON.stringify(v));
  for (const [k, v] of Object.entries(p.desk.auto)) f.set('a:' + k, JSON.stringify(v));
  for (const [k, v] of Object.entries(p.desk.state)) f.set('s:' + k, JSON.stringify(v));
  f.set('mods', JSON.stringify(p.mods)); f.set('board', JSON.stringify(p.board)); f.set('engine', JSON.stringify(p.desk.engine)); f.set('book', JSON.stringify(p.book));
  /* the packages' undoable state, as it is now (register/undoable.ts) */
  for (const [n, s] of undoables()) f.set('x:' + n, JSON.stringify(s.read()));
  return f;
}
function unflat(base: Parts, f: Flat): Parts {
  const vals: Desk['vals'] = {}, auto: Desk['auto'] = {}, state: Desk['state'] = {};
  for (const [p, j] of f) {
    if (p.startsWith('v:')) vals[p.slice(2)] = JSON.parse(j) as Value;
    else if (p.startsWith('a:')) auto[p.slice(2)] = JSON.parse(j) as boolean;
    else if (p.startsWith('s:')) state[p.slice(2)] = JSON.parse(j);
    else if (p.startsWith('x:')) { const s = undoables().get(p.slice(2)); if (s && JSON.stringify(s.read()) !== j) s.write(JSON.parse(j)); }
  }
  const mods = JSON.parse(f.get('mods')!) as string[], board = JSON.parse(f.get('board')!) as string[];
  const desk: Desk = { ...base.desk, vals, auto, state, engine: JSON.parse(f.get('engine')!) as EngineId, on: onOf(mods, board) };
  return { desk, book: JSON.parse(f.get('book')!) as EngineBook, mods, board };
}
const onOf = (mods: string[], board: string[]) => new Set([...BUILT_IN, ...mods, ...board]);

export interface Entry {
  words: StepWords;
  /** The desk before the step, for naming a merged run (stepWords from the first desk to the last). */
  first: Desk;
  before: Flat;
  after: Flat;
  /** Steps of one gesture (a drag, a held key, typing in one field) merge into one entry: the gesture's id. */
  gesture: string | null;
  at: number;
  /** Where focus goes when the step is undone (a selector). */
  back?: string;
  /** A step that is no change to the desk (a saved record replaced, renamed or deleted: store/workflows.ts) undoes and
   *  redoes itself, by name, on the records as they are then; either may say what it left alone. */
  run?: { undo(): Slot | void; redo(): Slot | void };
}

export interface Toast { text: Slot; action?: { label: Slot; run: () => void } }

export interface DeskStore extends Parts {
  undo: Entry[];
  redo: Entry[];
  /** The gesture in progress, whose changes merge into one step. */
  gesture: string | null;
  toast: Toast | null;
  /** Spoken in the live region: everything said in one task, together (the concept's say(), F-12). Every toast is said
   *  too, so what the page shows is also told to a screen reader. */
  said: { text: Slot[]; n: number } | null;
  /** Where focus should go once the desk has redrawn: selectors tried in order, the first on the page wins. Set by a
   *  step that removed what had focus. */
  focus: string[] | null;
  /** Where each removed module stood (the module after it), so adding it back puts it there. */
  gone: { [id: string]: string | null };
  /** What the last add should ring because it was already there: a tile (`flash`) or a pedal or unit (`here`). */
  flash: { id: string; ring: 'flash' | 'here'; n: number } | null;

  /** Applies a change a part made and records it as one step. `from` names the part: a gesture is one part's, and a
   *  change from another part while it runs is a step of its own. */
  commit(c: Change, from?: string): void;
  beginGesture(tag: string): void;
  endGesture(tag?: string): void;
  setTier(t: Tier): void;
  switchEngine(to: EngineId): void;
  add(id: string, before?: string | null): 'added' | 'there' | 'lab';
  remove(id: string, focusAfter?: string | string[] | null): void;
  move(id: string, before: string | null, back?: string): void;
  runUndo(entry?: Entry): void;
  runRedo(entry?: Entry): void;
  showToast(t: Toast | null): void;
  say(text: Slot): void;
  setFocus(sel: string | string[] | null): void;
  /** A change to a package's undoable state (register/undoable.ts), made by fn: one step with the core's words. */
  deskStep(words: StepWords, fn: () => void): void;
  /** A step that undoes and redoes itself (Entry.run): a saved record's replacement, rename or deletion. */
  recordStep(words: StepWords, run: NonNullable<Entry['run']>, back?: string): Entry;
  /** What the engine told the page: its lists, the selected models, the plugin host's count, the VRAM figures. Lists
   *  and selections are added to what it was told before (each engine's model buckets arrive on their own). */
  told(o: { lists?: Desk['lists']; selection?: Desk['selection']; vstPlugins?: number; vram?: Desk['vram']; adaptersAway?: string; songs?: Desk['songs'] }): void;
}

const UNDO_MAX = 100;
let gestures = 0;

function initial(): Parts & { told: Msg | null } {
  const empty: Desk = emptyDesk(ix);
  let raw: unknown = null, text: string | null = null;
  try { text = localStorage.getItem(DESK_KEY); raw = text ? JSON.parse(text) : null; } catch { raw = null; }
  const r = raw ? readDesk(ix, raw) : null;
  /* a desk that cannot be used is copied aside before the next autosave writes over it, and said once (F-16) */
  if (!r && text) { keepAside(DESK_KEY, text); tellOnBoot(msg('store.desk')); }
  if (!r) return { desk: empty, book: emptyBook(), mods: [], board: [], told: null };
  const ids = [...r.desk.on];
  const mods = ids.filter(id => { const d = defOf(id); return d?.kind === 'module' && !d.builtIn; });
  const board = boardOrder(ix, ids.filter(id => defOf(id)?.kind !== 'module'));
  return { desk: { ...r.desk, on: onOf(mods, board) }, book: r.book, mods, board, told: r.told };
}

export const useDesk = create<DeskStore>()((set, get) => {
  const start = initial();
  seeEngine(start.desk.engine);
  let saying = false;
  /** Records a step from the parts as they were to the parts as they are now. */
  const record = (prev: Parts, next: Parts, words: StepWords, back?: string, from?: string, named?: Change['named']) => {
    const s = get(), before = flatOf(prev), after = flatOf(next), last = s.undo[s.undo.length - 1];
    /* the gesture in progress is this change's when the change comes from the part that began it */
    const g = s.gesture && (from === undefined || s.gesture.startsWith(from + '|')) ? s.gesture : null;
    if (g && last && last.gesture === g) {
      const merged = stepWords(ix, last.first, next.desk, named ? { named } : undefined) ?? words;
      const undo = [...s.undo.slice(0, -1), { ...last, after, words: merged, at: Date.now() }];
      set({ ...next, undo, redo: [] });
      return;
    }
    const entry: Entry = { words, first: prev.desk, before, after, gesture: g, at: Date.now(), ...(back ? { back } : {}) };
    set({ ...next, undo: [...s.undo, entry].slice(-UNDO_MAX), redo: [] });
  };
  const parts = (): Parts => { const s = get(); return { desk: s.desk, book: s.book, mods: s.mods, board: s.board }; };
  /** Moves the desk across an entry, path by path: a path changed since stays as it is. */
  const cross = (e: Entry, from: Flat, to: Flat): boolean => {
    const now = flatOf(parts()), paths = new Set([...from.keys(), ...to.keys()]), out = new Map(now);
    let moved = false;
    for (const p of paths) {
      if (from.get(p) === to.get(p)) continue;
      if (now.get(p) !== from.get(p)) continue;
      const v = to.get(p); if (v === undefined) out.delete(p); else out.set(p, v);
      moved = true;
    }
    if (moved) set(unflat(parts(), out));
    void e;
    return moved;
  };
  return {
    ...start, undo: [], redo: [], gesture: null, toast: start.told ? { text: start.told } : null, said: null, focus: null, gone: {}, flash: null,

    commit(c, from) {
      const prev = parts(), desk = applyChange(prev.desk, c);
      record(prev, { ...prev, desk }, c.words, undefined, from, c.named);
      if (c.said) get().showToast({ text: c.said });
    },
    beginGesture(tag) { const g = get().gesture; if (!g || !g.startsWith(tag + '#')) set({ gesture: `${tag}#${++gestures}` }); },
    endGesture(tag) { const g = get().gesture; if (g && (!tag || g.startsWith(tag + '#'))) set({ gesture: null }); },
    setTier(t) { const s = get(); if (s.desk.tier !== t) set({ desk: { ...s.desk, tier: t } }); },

    switchEngine(to) {
      const prev = parts();
      if (prev.desk.engine === to) return;
      const r = switchEngine(ix, prev.desk, prev.book, to);
      /* a module the new engine does not have and that cannot wait parked (a source) leaves the desk with it */
      const mods = prev.mods.filter(id => r.desk.on.has(id)), board = prev.board.filter(id => r.desk.on.has(id));
      record(prev, { desk: { ...r.desk, on: onOf(mods, board) }, book: r.book, mods, board }, engineStep(to));
      const entry = get().undo[get().undo.length - 1]!, fresh = !seenEngines().includes(to);
      seeEngine(to);
      get().showToast({ text: (fresh && newOn(ix, get().desk, to, r.toast)) || r.toast, action: { label: { id: 'ui.undo' }, run: () => get().runUndo(entry) } });
    },

    add(id, before) {
      const s = get(), d = defOf(id); if (!d) return 'there';
      /* where it goes is the core's (addTo), and what a pedal or side unit arrives as (engaged, unless it would have
         nothing to run: arrivalChange) */
      const r = addTo(ix, s.desk, s.mods, s.board, d.id, before, s.gone);
      if (r.did === 'there') set({ flash: { id: d.id, ring: d.kind === 'module' ? 'flash' : 'here', n: (s.flash?.n ?? 0) + 1 } });
      if (r.did !== 'added') return r.did;
      const prev = parts(), desk = d.kind === 'module' ? prev.desk : applyChange(prev.desk, arrivalChange(ix, d, prev.desk));
      record(prev, { desk: { ...desk, on: onOf(r.mods, r.board) }, book: prev.book, mods: r.mods, board: r.board }, deskStep('add', d, prev.desk, ix));
      return 'added';
    },

    remove(id, focusAfter?: string | string[] | null) {
      const prev = parts(), d = defOf(id); if (!d || !prev.desk.on.has(d.id)) return;
      /* what leaves with it, and where it stood, are the core's (removeFrom) */
      const { mods, board, stood, pedals, last } = removeFrom(ix, prev.mods, prev.board, d.id);
      const gone = d.kind === 'module' ? { ...get().gone, [d.id]: stood } : get().gone;
      const words = deskStep('remove', d, prev.desk, ix);
      const back = d.kind === 'module' ? `.tile[data-m="${d.id}"] .grip` : d.kind === 'unit' ? `[data-xid="${d.id}"] .u-pow` : `[data-pid="${d.id}"] .foot, [data-pid="${d.id}"] .px`;
      record(prev, { ...prev, desk: { ...prev.desk, on: onOf(mods, board) }, mods, board }, words, back);
      const entry = get().undo[get().undo.length - 1]!;
      set({ gone, focus: focusAfter ? [focusAfter].flat() : null });
      get().showToast({ text: removedWords(d, prev.desk, pedals, last), action: { label: { id: 'ui.undo' }, run: () => get().runUndo(entry) } });
    },

    move(id, before, back) {
      const prev = parts(), d = defOf(id); if (!d) return;
      const mods = prev.mods.filter(x => x !== id), i = before ? mods.indexOf(before) : -1;
      if (i >= 0) mods.splice(i, 0, id); else mods.push(id);
      if (mods.join() === prev.mods.join()) return;
      /* a run of presses on one tile's grip, each within two seconds of the last, is one move (30-desk.js) */
      const s = get(), lastE = s.undo[s.undo.length - 1], tag = `move:${id}#`;
      set({ gesture: lastE?.gesture?.startsWith(tag) && Date.now() - lastE.at < 2000 ? lastE.gesture : `${tag}${++gestures}` });
      record(prev, { ...prev, desk: { ...prev.desk, on: onOf(mods, prev.board) }, mods }, deskStep('move', d, prev.desk, ix), back);
      set({ gesture: null });
    },

    runUndo(entry) {
      const s = get(), e = entry ?? s.undo[s.undo.length - 1]; if (!e) return;
      const i = s.undo.indexOf(e); if (i < 0) return;
      set({ undo: s.undo.filter(x => x !== e), gesture: null });
      const note = e.run ? e.run.undo() : (cross(e, e.after, e.before), undefined);
      set({ redo: [...get().redo, e], focus: e.back ? [e.back] : null });
      const said: Slot = { id: 'ui.undone', slots: { what: e.words.undone ?? e.words.label } };
      get().showToast({ text: note ? { list: [said, note], join: 'sentences' } : said, action: { label: { id: 'ui.redo' }, run: () => get().runRedo(e) } });
    },
    runRedo(entry) {
      const s = get(), e = entry ?? s.redo[s.redo.length - 1]; if (!e) return;
      const i = s.redo.indexOf(e); if (i < 0) return;
      set({ redo: s.redo.filter(x => x !== e), gesture: null });
      const note = e.run ? e.run.redo() : (cross(e, e.before, e.after), undefined);
      set({ undo: [...get().undo, e] });
      const said: Slot = { id: 'ui.redone', slots: { what: e.words.redone ?? e.words.label } };
      get().showToast({ text: note ? { list: [said, note], join: 'sentences' } : said, action: { label: { id: 'ui.undo' }, run: () => get().runUndo(e) } });
    },
    deskStep(words, fn) {
      const prev = parts(), before = flatOf(prev); fn();
      const s = get(), after = flatOf(parts());
      if ([...after].every(([k, v]) => before.get(k) === v)) return;
      set({ undo: [...s.undo, { words, first: prev.desk, before, after, gesture: null, at: Date.now() }].slice(-UNDO_MAX), redo: [] });
    },
    recordStep(words, run, back) {
      const s = get(), f = flatOf(parts()), e: Entry = { words, first: s.desk, before: f, after: f, gesture: null, at: Date.now(), run, ...(back ? { back } : {}) };
      set({ undo: [...s.undo, e].slice(-UNDO_MAX), redo: [] });
      return e;
    },
    showToast(t) { set({ toast: t }); if (t) get().say(t.text); },
    say(text) {
      const was = get().said;
      set({ said: { text: saying && was ? [...was.text, text] : [text], n: (was?.n ?? 0) + 1 } });
      if (!saying) { saying = true; queueMicrotask(() => { saying = false; }); }
    },
    setFocus(sel) { set({ focus: sel === null ? null : [sel].flat() }); },
    told(o) {
      const d = get().desk;
      set({ desk: { ...d, ...(o.lists ? { lists: { ...d.lists, ...o.lists } } : {}), ...(o.selection ? { selection: { ...d.selection, ...o.selection } } : {}), ...(o.vram ? { vram: o.vram } : {}), ...(o.songs ? { songs: { ...d.songs, ...o.songs } } : {}), ...(o.vstPlugins !== undefined ? { told: { ...d.told, vstPlugins: o.vstPlugins } } : {}) } });
      /* the words for another engine's adapters (package 3b: features/adapters.tsx, from the engines' book) */
      if (o.adaptersAway !== undefined && o.adaptersAway !== get().desk.told?.adaptersAway) set({ desk: { ...get().desk, told: { ...get().desk.told, adaptersAway: o.adaptersAway } } });
    },
  };
});

/* ---------------------------------------------------------------- the autosave
   Everything, on every change of the desk or the engines' book, as the core writes it (savedDeskOf), read back by
   readDesk() on the next visit. A toast, a focus request or the history moving writes nothing. */
let written = '';
useDesk.subscribe((s, prev) => {
  if (s.desk === prev.desk && s.book === prev.book) return;
  const rec = savedDeskOf(ix, s.desk, s.book), text = JSON.stringify(rec);
  if (text === written) return;
  /* a refused write (storage full or off) is reported (store/stored.ts): the read-out says Not saved, and the desk
     lives while the tab is open; it is written again with the next change */
  if (writeStored(DESK_KEY, text)) written = text;
});

/** The definitions of the modules on the desk, in the user's order. */
export const modsOf = (s: Pick<DeskStore, 'mods'>): Definition[] => s.mods.map(id => defOf(id)).filter((d): d is Definition => !!d);
