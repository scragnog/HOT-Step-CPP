// The desk store: what is on the desk (modules in the user's order, the board in chain order), the tier, the engine,
// the values, undo and redo, and the autosave. Every change goes through the core: a value through the Change a part's
// setter built (bindPart, src/derive/parts.ts), which the core has already cleaned and named; an engine switch through
// switchEngine(); a record through readDesk() and savedDeskOf(). The store decides nothing about a value: it applies
// what the core returns, records it as one undo step, and puts back only what a step changed.

import { create } from 'zustand';
import type { Definition, EngineId, Tier, Value } from '../../definitions/types.ts';
import {
  applyChange, arrivalChange, deskStep, emptyBook, engineStep, readDesk, removedWords, savedDeskOf, stepWords, switchEngine,
  type Change, type Desk, type EngineBook, type Msg, type Slot, type StepWords,
} from '../../derive.ts';
import { BOARD, BUILT_IN, boardOrder, defOf, ix } from '../model/index.ts';

/** Where the desk is kept between visits. */
export const DESK_KEY = 'hotstep.create.desk';

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
  return f;
}
function unflat(base: Parts, f: Flat): Parts {
  const vals: Desk['vals'] = {}, auto: Desk['auto'] = {}, state: Desk['state'] = {};
  for (const [p, j] of f) {
    if (p.startsWith('v:')) vals[p.slice(2)] = JSON.parse(j) as Value;
    else if (p.startsWith('a:')) auto[p.slice(2)] = JSON.parse(j) as boolean;
    else if (p.startsWith('s:')) state[p.slice(2)] = JSON.parse(j);
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
}

export interface Toast { text: Slot; action?: { label: Slot; run: () => void } }

export interface DeskStore extends Parts {
  undo: Entry[];
  redo: Entry[];
  /** The gesture in progress, whose changes merge into one step. */
  gesture: string | null;
  toast: Toast | null;
  /** Spoken in the live region (a move, a refused move). */
  said: { text: Slot; n: number } | null;
  /** Where focus should go once the desk has redrawn (a selector), set by a step that removed what had focus. */
  focus: string | null;
  /** Where each removed module stood (the module after it), so adding it back puts it there. */
  gone: { [id: string]: string | null };
  /** The modules (by id) the last add or move should ring. */
  flash: { id: string; n: number } | null;

  commit(c: Change): void;
  beginGesture(tag: string): void;
  endGesture(tag?: string): void;
  setTier(t: Tier): void;
  switchEngine(to: EngineId): void;
  add(id: string, before?: string | null): 'added' | 'there' | 'lab';
  remove(id: string, focusAfter?: string | null): void;
  move(id: string, before: string | null, back?: string): void;
  runUndo(entry?: Entry): void;
  runRedo(entry?: Entry): void;
  showToast(t: Toast | null): void;
  say(text: Slot): void;
  setFocus(sel: string | null): void;
  /** What the engine told the page: its lists, the selected models, the plugin host's count. */
  told(o: { lists?: Desk['lists']; selection?: Desk['selection']; vstPlugins?: number }): void;
}

const UNDO_MAX = 100;
let gestures = 0;

function initial(): Parts & { told: Msg | null } {
  const empty: Desk = { engine: 'ace', tier: 'quick', vals: {}, auto: {}, state: {}, on: onOf([], []) };
  let raw: unknown = null;
  try { const s = localStorage.getItem(DESK_KEY); raw = s ? JSON.parse(s) : null; } catch { raw = null; }
  const r = raw ? readDesk(ix, raw) : null;
  if (!r) return { desk: empty, book: emptyBook(), mods: [], board: [], told: null };
  const ids = [...r.desk.on];
  const mods = ids.filter(id => { const d = defOf(id); return d?.kind === 'module' && !d.builtIn; });
  const board = boardOrder(ids.filter(id => defOf(id)?.kind !== 'module'));
  return { desk: { ...r.desk, on: onOf(mods, board) }, book: r.book, mods, board, told: r.told };
}

export const useDesk = create<DeskStore>()((set, get) => {
  const start = initial();
  /** Records a step from the parts as they were to the parts as they are now. */
  const record = (prev: Parts, next: Parts, words: StepWords, back?: string) => {
    const s = get(), before = flatOf(prev), after = flatOf(next), last = s.undo[s.undo.length - 1];
    if (s.gesture && last && last.gesture === s.gesture) {
      const merged = stepWords(ix, last.first, next.desk) ?? words;
      const undo = [...s.undo.slice(0, -1), { ...last, after, words: merged, at: Date.now() }];
      set({ ...next, undo, redo: [] });
      return;
    }
    const entry: Entry = { words, first: prev.desk, before, after, gesture: s.gesture, at: Date.now(), ...(back ? { back } : {}) };
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

    commit(c) {
      const prev = parts(), desk = applyChange(prev.desk, c);
      record(prev, { ...prev, desk }, c.words);
      if (c.said) set({ toast: { text: c.said } });
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
      set({ toast: { text: r.toast } });
    },

    add(id, before) {
      const s = get(), d = defOf(id); if (!d) return 'there';
      if (d.kind === 'module' && d.labUnder && s.desk.tier !== 'lab') return 'lab';
      if (s.desk.on.has(d.id)) { set({ flash: { id: d.id, n: (s.flash?.n ?? 0) + 1 } }); return 'there'; }
      const prev = parts();
      let mods = [...prev.mods], board = prev.board, desk = prev.desk;
      const insert = (m: string, b: string | null | undefined) => { mods = mods.filter(x => x !== m); const i = b ? mods.indexOf(b) : -1; if (i >= 0) mods.splice(i, 0, m); else mods.push(m); };
      if (d.kind === 'module') {
        const back = s.gone[d.id];
        insert(d.id, before === undefined && back && mods.includes(back) ? back : before);
      } else {
        board = boardOrder([...board, d.id]);
        if (!mods.includes(BOARD.id)) insert(BOARD.id, before);
        /* what a pedal or side unit arrives as is the core's (it arrives engaged, unless it would have nothing to run) */
        desk = applyChange(desk, arrivalChange(ix, d, desk));
      }
      record(prev, { desk: { ...desk, on: onOf(mods, board) }, book: prev.book, mods, board }, deskStep('add', d, desk));
      set({ flash: { id: d.kind === 'module' ? d.id : BOARD.id, n: (s.flash?.n ?? 0) + 1 } });
      return 'added';
    },

    remove(id, focusAfter) {
      const prev = parts(), d = defOf(id); if (!d || !prev.desk.on.has(d.id)) return;
      let mods = prev.mods, board = prev.board, pedals = 0, last = false;
      const gone = { ...get().gone };
      if (d.kind === 'module') {
        const at = mods.indexOf(d.id); gone[d.id] = mods.slice(at + 1).find(x => !(defOf(x) as { source?: true } | undefined)?.source) ?? null;
        mods = mods.filter(x => x !== d.id);
        if (d.id === BOARD.id) { pedals = board.length; board = []; }
      } else {
        board = board.filter(x => x !== d.id);
        if (!board.length) { last = true; mods = mods.filter(x => x !== BOARD.id); }
      }
      const words = deskStep('remove', d, prev.desk);
      const back = d.kind === 'module' ? `.tile[data-m="${d.id}"] .grip` : d.kind === 'unit' ? `[data-xid="${d.id}"] .u-pow` : `[data-pid="${d.id}"] .foot, [data-pid="${d.id}"] .px`;
      record(prev, { ...prev, desk: { ...prev.desk, on: onOf(mods, board) }, mods, board }, words, back);
      const entry = get().undo[get().undo.length - 1]!;
      set({ gone, focus: focusAfter ?? null, toast: { text: removedWords(d, prev.desk, pedals, last), action: { label: { id: 'ui.undo' }, run: () => get().runUndo(entry) } } });
    },

    move(id, before, back) {
      const prev = parts(), d = defOf(id); if (!d) return;
      const mods = prev.mods.filter(x => x !== id), i = before ? mods.indexOf(before) : -1;
      if (i >= 0) mods.splice(i, 0, id); else mods.push(id);
      if (mods.join() === prev.mods.join()) return;
      /* a run of presses on one tile's grip, each within two seconds of the last, is one move (30-desk.js) */
      const s = get(), lastE = s.undo[s.undo.length - 1], tag = `move:${id}#`;
      set({ gesture: lastE?.gesture?.startsWith(tag) && Date.now() - lastE.at < 2000 ? lastE.gesture : `${tag}${++gestures}` });
      record(prev, { ...prev, desk: { ...prev.desk, on: onOf(mods, prev.board) }, mods }, deskStep('move', d, prev.desk), back);
      set({ gesture: null });
    },

    runUndo(entry) {
      const s = get(), e = entry ?? s.undo[s.undo.length - 1]; if (!e) return;
      const i = s.undo.indexOf(e); if (i < 0) return;
      set({ undo: s.undo.filter(x => x !== e), gesture: null });
      cross(e, e.after, e.before);
      set({ redo: [...get().redo, e], toast: { text: { id: 'ui.undone', slots: { what: e.words.undone ?? e.words.label } }, action: { label: { id: 'ui.redo' }, run: () => get().runRedo(e) } }, focus: e.back ?? null });
    },
    runRedo(entry) {
      const s = get(), e = entry ?? s.redo[s.redo.length - 1]; if (!e) return;
      const i = s.redo.indexOf(e); if (i < 0) return;
      set({ redo: s.redo.filter(x => x !== e), gesture: null });
      cross(e, e.before, e.after);
      set({ undo: [...get().undo, e], toast: { text: { id: 'ui.redone', slots: { what: e.words.redone ?? e.words.label } }, action: { label: { id: 'ui.undo' }, run: () => get().runUndo(e) } } });
    },
    showToast(t) { set({ toast: t }); },
    say(text) { set({ said: { text, n: (get().said?.n ?? 0) + 1 } }); },
    setFocus(sel) { set({ focus: sel }); },
    told(o) {
      const d = get().desk;
      set({ desk: { ...d, ...(o.lists ? { lists: o.lists } : {}), ...(o.selection ? { selection: o.selection } : {}), ...(o.vstPlugins !== undefined ? { told: { ...d.told, vstPlugins: o.vstPlugins } } : {}) } });
    },
  };
});

/* ---------------------------------------------------------------- the autosave
   Everything, on every change, as the core writes it (savedDeskOf), read back by readDesk() on the next visit. */
let written = '';
useDesk.subscribe(s => {
  const rec = savedDeskOf(ix, s.desk, s.book), text = JSON.stringify(rec);
  if (text === written) return;
  written = text;
  try { localStorage.setItem(DESK_KEY, text); } catch { /* storage full or off: the desk lives while the tab is open */ }
});

/** The definitions of the modules on the desk, in the user's order. */
export const modsOf = (s: Pick<DeskStore, 'mods'>): Definition[] => s.mods.map(id => defOf(id)).filter((d): d is Definition => !!d);
