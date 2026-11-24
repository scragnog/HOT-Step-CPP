// The Workflows panel's and the Drafts list's store (the concept's 32-workflows.js and 35-drafts.js): the user's saved
// workflows and revisions, the drafts, which workflow and revision are open, and what the panel shows open. It decides
// nothing about a record: names, cleaning, what a revision holds, "modified", the merge with another tab and every
// word are the core's (src/derive/workflows.ts, src/derive/records.ts). It keeps the lists, writes them (merged with
// what another tab saved meanwhile, C-09), and makes each change one undo step:
//
// - opening a workflow, loading or reverting a revision, loading a draft change the desk: a desk step (deskStep), with
//   which workflow and revision are open registered as undoable state ('workflow'), so Undo puts back only what the
//   step changed and nothing has changed since (C-03, C-07);
// - replacing, updating, renaming, deleting and importing a record change no desk: a record step (recordStep) that
//   undoes and redoes itself by name on the lists as they are then, so it never wipes later work, and says what it
//   left alone (C-02).

import { create } from 'zustand';
import type { EngineId, Revision, Workflow } from '../../definitions/types.ts';
import {
  applyDraft, draftLoaded, draftStep, draftTitle, dropRevision, dropWorkflow, emptySaved, findRevision, findWorkflow, importRevision, layoutOf, loadRevision, loadStep, loadedWords,
  mergeDrafts, mergeSaved, msg, newestDrafts, openStep, putRevision, putWords, putWorkflow, readDrafts, readSaved, recordStep, renameRevision, renameWorkflow, revNameClash, revisionIndex,
  revisionOf, sameName, savedFile, START_FRESH, storeSkipped, storeUnreadable, switchEngine, unloadedFiles, wfNameClash, whatRevision, whatWorkflow, workflowIndex, workflowModified, workflowOf,
  draftOf, DRAFT_MAX, onOf, type Desk, type Msg, type Open, type RevId, type Saved, type Slot, type StoredDraft,
} from '../../derive.ts';
import { engineName } from '../../definitions/engines.ts';
import { ix } from '../model/index.ts';
import { defineUndoable } from '../register/undoable.ts';
import { useDesk, type Entry } from './desk.ts';
import { readStored, keepAside, tellOnBoot, writeStored } from './stored.ts';

/** Where the lists are kept (deviation 5: the page's own keys). */
export const WF_KEY = 'hotstep.create.workflows';
export const DRAFT_KEY = 'hotstep.create.drafts';
/** Which workflow and revision are open, and which workflows show their revisions: kept with the desk. */
export const OPEN_KEY = 'hotstep.create.workflow';

export interface WfStore {
  saved: Saved;
  drafts: StoredDraft[];
  open: Open;
  /** Workflows whose revisions show (other than the open one), by name. */
  expanded: string[];
  /** The open workflow's revisions folded away. */
  fold: boolean;
  /** The revision whose details show: `<workflow>|<engine>|<name>|<user>`. */
  info: string | null;
  /** A row to wash green once drawn (a rename, an import: G-D-30): its selector. */
  wash: { sel: string; n: number } | null;
}

/* ---------------------------------------------------------------- reading, at boot */

const asOpen = (v: unknown): Open => {
  const o = (v && typeof v === 'object' ? v : {}) as Record<string, unknown>;
  return { workflow: typeof o.workflow === 'string' ? o.workflow : o.workflow === null ? null : START_FRESH, revision: typeof o.revision === 'string' ? o.revision : null, revFor: typeof o.revFor === 'string' ? o.revFor as EngineId : null };
};
let base: Saved = emptySaved(), draftBase: StoredDraft[] = [];
function boot(): Pick<WfStore, 'saved' | 'drafts' | 'open' | 'expanded' | 'fold'> {
  let saved = emptySaved(), drafts: StoredDraft[] = [];
  const w = readStored(WF_KEY);
  if (w.bad) tellOnBoot(storeUnreadable('wf'));
  else if (w.v != null) {
    const r = readSaved(ix, w.v);
    if (!r) { keepAside(WF_KEY, w.raw); tellOnBoot(storeUnreadable('wf')); }
    /* records it skipped: the whole stored value is copied aside first, since the next save writes back only the rest */
    else { saved = r.saved; if (r.bad) { keepAside(WF_KEY, w.raw); tellOnBoot(storeSkipped(r.bad, 'wf')); } }
  }
  const d = readStored(DRAFT_KEY);
  if (d.bad) tellOnBoot(storeUnreadable('drafts'));
  else if (d.v != null) {
    const r = readDrafts(ix, d.v);
    if (!r) { keepAside(DRAFT_KEY, d.raw); tellOnBoot(storeUnreadable('drafts')); }
    else { drafts = r.drafts; if (r.bad) { keepAside(DRAFT_KEY, d.raw); tellOnBoot(storeSkipped(r.bad, 'draft')); } }
  }
  base = structuredClone(saved); draftBase = structuredClone(drafts);
  const o = readStored(OPEN_KEY).v as { open?: unknown; expanded?: unknown; fold?: unknown } | null;
  const open = asOpen(o?.open ?? {}), known = open.workflow === null || !!findWorkflow(saved, open.workflow);
  return {
    saved, drafts,
    /* a workflow that is not here any more reads as Start fresh (the concept's deskState('workflow')) */
    open: known ? open : { workflow: START_FRESH, revision: null, revFor: null },
    expanded: Array.isArray(o?.expanded) ? o.expanded.filter((x): x is string => typeof x === 'string') : [],
    fold: o?.fold === true,
  };
}

export const useWf = create<WfStore>()(() => ({ ...boot(), info: null, wash: null }));

/** Reads the lists again (the test layer, after it put a scene's lists in storage). */
export function rebootWf(): void { useWf.setState({ ...boot(), info: null, wash: null }); }

/* ---------------------------------------------------------------- writing, merged with another tab's (C-09) */

/** Writes the lists: storage is read first and merged, so a workflow another tab saved meanwhile is kept, and appears
 *  here. Returns whether the browser took it. */
export function persistSaved(): boolean {
  const got = readStored(WF_KEY), theirs = !got.bad ? readSaved(ix, got.v)?.saved ?? null : null, ours = useWf.getState().saved;
  const m = theirs ? mergeSaved(base, ours, theirs) : ours;
  useWf.setState({ saved: m });
  const ok = writeStored(WF_KEY, JSON.stringify(savedFile(m)));
  base = structuredClone(m);
  return ok;
}
export function persistDrafts(): boolean {
  const got = readStored(DRAFT_KEY), theirs = !got.bad ? readDrafts(ix, got.v)?.drafts ?? null : null, ours = useWf.getState().drafts;
  const m = theirs ? mergeDrafts(draftBase, ours, theirs) : newestDrafts(ours);
  useWf.setState({ drafts: m });
  const ok = writeStored(DRAFT_KEY, JSON.stringify(m));
  draftBase = structuredClone(m);
  return ok;
}
/* which workflow is open, and which show their revisions, are kept as they change */
useWf.subscribe((s, p) => {
  if (s.open === p.open && s.expanded === p.expanded && s.fold === p.fold) return;
  writeStored(OPEN_KEY, JSON.stringify({ open: s.open, expanded: s.expanded, fold: s.fold }));
});

/** Another tab saved: take its changes in, keep ours. The workflow open here, deleted there, is no longer open, and
 *  the page says the desk is untouched (F-23); one renamed there stays open under its new name (F-17). */
export function theyWrote(key: string, text: string | null): void {
  if (text == null) return;
  let v: unknown; try { v = JSON.parse(text); } catch { return; }
  if (key === WF_KEY) {
    const r = readSaved(ix, v); if (!r) return;
    const s = useWf.getState(), was = s.saved, m = mergeSaved(base, was, r.saved);
    base = structuredClone(r.saved);
    let open = s.open, said: Msg | null = null;
    if (open.workflow && !findWorkflow(m, open.workflow)) {
      const gone = was.workflows.find(w => w.name === open.workflow), to = gone && m.workflows.find(w => !findWorkflow(was, w.name) && w.tier === gone.tier && w.modules.join() === gone.modules.join() && w.board.join() === gone.board.join());
      if (to) open = { ...open, workflow: to.name };
      else { said = msg('wf.gone.wf', { name: open.workflow }); open = { workflow: null, revision: null, revFor: null }; }
    } else if (open.workflow && open.revision && open.revFor && !findRevision(m, { workflow: open.workflow, engine: open.revFor, name: open.revision })) {
      said = msg('wf.gone.rev', { name: open.revision }); open = { ...open, revision: null, revFor: null };
    }
    useWf.setState({ saved: m, open });
    if (said) useDesk.getState().showToast({ text: said });
  }
  if (key === DRAFT_KEY) {
    const r = readDrafts(ix, v); if (!r) return;
    const m = mergeDrafts(draftBase, useWf.getState().drafts, r.drafts);
    draftBase = structuredClone(r.drafts);
    useWf.setState({ drafts: m });
  }
}
addEventListener('storage', e => { if (e.key === WF_KEY || e.key === DRAFT_KEY) theyWrote(e.key, e.newValue); });

/* ---------------------------------------------------------------- undo: which workflow and revision are open */

defineUndoable('workflow', {
  read: () => ({ ...useWf.getState().open }),
  write: v => useWf.setState({ open: asOpen(v) }),
});

/* ---------------------------------------------------------------- the desk's parts, for the steps below */

const desk = () => useDesk.getState();
const onSet = (mods: readonly string[], board: readonly string[]) => onOf(ix, mods, board);
/** What the desk becomes, worked out before it is applied: the parts a step sets at once. */
interface Plan { desk: Desk; book: ReturnType<typeof desk>['book']; mods: string[]; board: string[] }
const now = (): Plan => { const s = desk(); return { desk: s.desk, book: s.book, mods: s.mods, board: s.board }; };
/** A workflow's modules and board on a planned desk. */
function laid(p: Plan, w: Workflow): Plan { const { mods, board } = layoutOf(ix, w); return { ...p, mods, board, desk: { ...p.desk, on: onSet(mods, board) } }; }
/** A planned desk moved to another engine (the engine we leave keeps its own settings for when we come back: B-01). */
function onEngine(p: Plan, e: EngineId): Plan {
  if (p.desk.engine === e) return p;
  const sw = switchEngine(ix, p.desk, p.book, e), mods = p.mods.filter(m => sw.desk.on.has(m)), board = p.board.filter(b => sw.desk.on.has(b));
  return { desk: { ...sw.desk, on: onSet(mods, board) }, book: sw.book, mods, board };
}
const apply = (p: Plan) => desk().applyDesk(p);
const putOf = (from: Plan, to: Plan) => putWords(ix, from.desk, to.desk, from.mods.join() !== to.mods.join(), from.board.join() !== to.board.join());
const undoAction = (e: Entry | null) => (e ? { label: { id: 'ui.undo' }, run: () => useDesk.getState().runUndo(e) } : undefined);
const toast = (text: Slot, action?: { label: Slot; run: () => void }) => desk().showToast({ text, ...(action ? { action } : {}) });

/* ---------------------------------------------------------------- opening and loading */

/** Opens a workflow: its modules and pedals (and its tier when asked); settings stay as they are. Revert puts back the
 *  open workflow's modules and keeps its revision open. */
export function openWorkflow(name: string, o: { tier?: boolean; revert?: boolean } = {}): void {
  const f = findWorkflow(useWf.getState().saved, name); if (!f) return;
  const w = f.w, from = now(), keepRev = o.revert && useWf.getState().open.workflow === w.name;
  const to = laid(from, w);
  const e = desk().deskStep(openStep(w.name, !!o.revert, putOf(from, to)), () => {
    apply(o.tier ? { ...to, desk: { ...to.desk, tier: w.tier } } : to);
    const s = useWf.getState();
    useWf.setState({ open: keepRev ? { ...s.open, workflow: w.name } : { workflow: w.name, revision: null, revFor: null }, fold: false });
  });
  toast(msg(o.revert ? 'wf.reverted' : 'wf.opened', { name: w.name }), undoAction(e));
}

/** Loads a revision: its workflow's layout (not on Revert, which puts back only what a revision holds: C-03), its
 *  engine, every setting it keeps for that engine, its adapters and the content it holds. */
export function loadRev(id: RevId, user: boolean, revert = false): void {
  const st = useWf.getState(), r = findRevision(st.saved, id, user); if (!r) return;
  const w = findWorkflow(st.saved, id.workflow)?.w, keepDesk = revert && st.open.workflow === id.workflow;
  const from = now(), moved = onEngine(w && !keepDesk ? laid(from, w) : from, r.engine);
  const to: Plan = { ...moved, desk: loadRevision(ix, moved.desk, r).desk }, unloaded = unloadedFiles(ix, moved.desk, to.desk);
  const e = desk().deskStep(loadStep(r.name, revert, putOf(from, to)), () => {
    apply(to);
    useWf.setState({ open: { workflow: id.workflow, revision: r.name, revFor: r.engine }, expanded: [...new Set([...useWf.getState().expanded, id.workflow])], fold: false });
  });
  toast(loadedWords(ix, id.workflow, r, to.desk, from.desk.engine !== r.engine, unloaded, revert), undoAction(e));
}

/* ---------------------------------------------------------------- saving */

/** The desk's layout is not the open workflow's (what the save row's hint warns of). */
export function layoutChanged(): boolean {
  const s = useWf.getState(), w = findWorkflow(s.saved, s.open.workflow)?.w;
  return !!w && workflowModified(ix, w, desk().mods, desk().board);
}

/** Saves a revision of the open workflow under a name the core accepted; `replace` names the record it overwrites. */
export function saveRevision(name: string, withContent: boolean, replace: string | null): void {
  const s = useWf.getState(), wf = s.open.workflow, e = desk().desk.engine; if (!wf) return;
  const old = replace ? findRevision(s.saved, { workflow: wf, engine: e, name: replace }, true) : null;
  const at = old ? revisionIndex(s.saved, { workflow: wf, engine: e, name: old.name }) : 0;
  const nr = revisionOf(ix, desk().desk, wf, old ? old.name : name, withContent), marks = s.open;
  useWf.setState({ saved: putRevision(dropRevision(s.saved, nr), nr, at), open: { workflow: wf, revision: nr.name, revFor: e }, expanded: [...new Set([...s.expanded, wf])] });
  const ok = persistSaved();
  if (!old) { if (ok) toast(msg('wf.saved.rev', { wf, name: nr.name, engine: engineName(e) })); return; }
  const entry = desk().recordStep(recordStep('replace', old.name, whatRevision(wf, old.name)), {
    undo: () => { useWf.setState({ saved: putRevision(useWf.getState().saved, structuredClone(old)), open: marks }); persistSaved(); },
    redo: () => { useWf.setState({ saved: putRevision(useWf.getState().saved, structuredClone(nr)), open: { workflow: wf, revision: nr.name, revFor: e } }); persistSaved(); },
  });
  if (ok) toast(msg('wf.replaced.rev', { wf, name: old.name }), undoAction(entry));
}

/** Saves the desk as a workflow, and opens it; the toast offers a revision too (`more`). */
export function saveWorkflow(name: string, replace: string | null, more: () => void): void {
  const s = useWf.getState(), old = replace ? s.saved.workflows.find(w => w.name === replace) ?? null : null;
  const at = old ? workflowIndex(s.saved, old.name) : 0, nw = workflowOf(ix, desk().desk, old ? old.name : name), marks = s.open;
  useWf.setState({ saved: putWorkflow(old ? dropWorkflowOnly(s.saved, old.name) : s.saved, nw, at), open: { workflow: nw.name, revision: null, revFor: null } });
  const ok = persistSaved();
  if (!old) { if (ok) toast(msg('wf.saved.wf', { name: nw.name }), { label: msg('wf.saved.wf.more'), run: more }); return; }
  const entry = desk().recordStep(recordStep('replace', old.name, whatWorkflow(old.name)), {
    undo: () => { useWf.setState({ saved: putWorkflow(useWf.getState().saved, structuredClone(old)), open: marks }); persistSaved(); },
    redo: () => { useWf.setState({ saved: putWorkflow(useWf.getState().saved, structuredClone(nw)), open: { workflow: nw.name, revision: null, revFor: null } }); persistSaved(); },
  });
  if (ok) toast(msg('wf.replaced.wf', { name: old.name }), undoAction(entry));
}
/** A workflow taken out of the list without its revisions (a replaced layout keeps them). */
const dropWorkflowOnly = (s: Saved, name: string): Saved => ({ ...s, workflows: s.workflows.filter(w => !sameName(w.name, name)) });

/* ---------------------------------------------------------------- update, rename, delete (each undoable, C-23) */

export function updateWorkflow(name: string): void {
  const s = useWf.getState(), old = s.saved.workflows.find(w => w.name === name); if (!old) return;
  const nw = { ...workflowOf(ix, desk().desk, old.name), ...(old.note ? { note: old.note } : {}) };
  useWf.setState({ saved: putWorkflow(s.saved, nw) }); persistSaved();
  const entry = desk().recordStep(recordStep('update', old.name, whatWorkflow(old.name)), {
    undo: () => { useWf.setState({ saved: putWorkflow(useWf.getState().saved, structuredClone(old)) }); persistSaved(); },
    redo: () => { useWf.setState({ saved: putWorkflow(useWf.getState().saved, structuredClone(nw)) }); persistSaved(); },
  });
  toast(msg('wf.updated.wf', { name: old.name }), undoAction(entry));
}
export function updateRevision(id: RevId): void {
  const s = useWf.getState(), old = findRevision(s.saved, id, true); if (!old) return;
  const nr: Revision = { ...revisionOf(ix, desk().desk, old.workflow, old.name, !!old.withContent), engine: old.engine, ...(old.note ? { note: old.note } : {}) };
  useWf.setState({ saved: putRevision(s.saved, nr) }); persistSaved();
  const entry = desk().recordStep(recordStep('update', old.name, whatRevision(old.workflow, old.name)), {
    undo: () => { useWf.setState({ saved: putRevision(useWf.getState().saved, structuredClone(old)) }); persistSaved(); },
    redo: () => { useWf.setState({ saved: putRevision(useWf.getState().saved, structuredClone(nr)) }); persistSaved(); },
  });
  toast(msg('wf.updated.rev', { wf: old.workflow, name: old.name }), undoAction(entry));
}

/** Renames a workflow, its revisions, its open mark and its open list with it (F-02); refused (with the reason) when the
 *  name is another's. */
export function renameWf(from: string, to: string): Msg | null {
  const s = useWf.getState(), w = s.saved.workflows.find(x => x.name === from); if (!w) return msg('wf.gone.here');
  const err = wfNameClash(s.saved, to, from); if (err) return err;
  const move = (a: string, b: string) => {
    const t = useWf.getState();
    useWf.setState({ saved: renameWorkflow(t.saved, a, b), open: t.open.workflow === a ? { ...t.open, workflow: b } : t.open, expanded: t.expanded.map(x => (x === a ? b : x)), info: t.info?.startsWith(a + '|') ? b + t.info.slice(a.length) : t.info });
    persistSaved();
  };
  move(from, to);
  const entry = desk().recordStep(recordStep('rename', from, whatWorkflow(to)), {
    undo: () => { const t = useWf.getState(); if (wfNameClash(t.saved, from, to)) return msg('wf.taken.now', { old: from, name: to }); move(to, from); },
    redo: () => { const t = useWf.getState(); if (wfNameClash(t.saved, to, from)) return msg('wf.taken.again', { name: to }); move(from, to); },
  });
  toast(msg('wf.renamed.wf', { from, to }), undoAction(entry));
  useWf.setState({ wash: { sel: `.pre.wfrow[data-preset="${CSS.escape(to)}"]`, n: (useWf.getState().wash?.n ?? 0) + 1 } });
  return null;
}
export function renameRev(id: RevId, to: string): Msg | null {
  const s = useWf.getState(), r = findRevision(s.saved, id, true); if (!r) return msg('wf.gone.here');
  const err = revNameClash(s.saved, id.workflow, id.engine, to, id.name); if (err) return err;
  const move = (a: string, b: string) => {
    const t = useWf.getState(), o = t.open, k = (n: string) => `${id.workflow}|${id.engine}|${n}|1`;
    useWf.setState({ saved: renameRevision(t.saved, { ...id, name: a }, b), open: o.workflow === id.workflow && o.revision === a && o.revFor === id.engine ? { ...o, revision: b } : o, info: t.info === k(a) ? k(b) : t.info });
    persistSaved();
  };
  move(id.name, to);
  const entry = desk().recordStep(recordStep('rename', id.name, whatRevision(id.workflow, to)), {
    undo: () => { if (revNameClash(useWf.getState().saved, id.workflow, id.engine, id.name, to)) return msg('wf.taken.now', { old: id.name, name: to }); move(to, id.name); },
    redo: () => { if (revNameClash(useWf.getState().saved, id.workflow, id.engine, to, id.name)) return msg('wf.taken.again', { name: to }); move(id.name, to); },
  });
  toast(msg('wf.renamed.rev', { wf: id.workflow, from: id.name, to }), undoAction(entry));
  useWf.setState({ wash: { sel: `.pre.rev[data-rev="${CSS.escape(id.workflow)}"][data-rn="${CSS.escape(to)}"][data-rb="${id.engine}"]`, n: (useWf.getState().wash?.n ?? 0) + 1 } });
  return null;
}

export function deleteWf(name: string): void {
  const s = useWf.getState(), i = workflowIndex(s.saved, name), gone = s.saved.workflows[i]; if (!gone) return;
  const revs = s.saved.revisions.filter(r => sameName(r.workflow, name)), wasOn = s.open.workflow === gone.name, marks = s.open;
  const del = () => { const t = useWf.getState(); useWf.setState({ saved: dropWorkflow(t.saved, gone.name), open: t.open.workflow === gone.name ? { workflow: null, revision: null, revFor: null } : t.open }); persistSaved(); };
  del();
  const entry = desk().recordStep(recordStep('delete', gone.name, whatWorkflow(gone.name)), {
    undo: () => { let t = putWorkflow(useWf.getState().saved, structuredClone(gone), i); revs.forEach(r => { t = putRevision(t, structuredClone(r), Number.MAX_SAFE_INTEGER); }); useWf.setState({ saved: t, ...(wasOn ? { open: marks } : {}) }); persistSaved(); },
    redo: del,
  });
  toast(msg('wf.deleted.wf', { name: gone.name, n: revs.length }), undoAction(entry));
}
export function deleteRev(id: RevId): void {
  const s = useWf.getState(), gone = findRevision(s.saved, id, true); if (!gone) return;
  const i = revisionIndex(s.saved, id), wasOn = s.open.workflow === id.workflow && s.open.revision === gone.name && s.open.revFor === gone.engine;
  const del = () => { const t = useWf.getState(), o = t.open; useWf.setState({ saved: dropRevision(t.saved, id), open: o.workflow === id.workflow && o.revision === gone.name && o.revFor === gone.engine ? { ...o, revision: null, revFor: null } : o }); persistSaved(); };
  del();
  const entry = desk().recordStep(recordStep('delete', gone.name, whatRevision(id.workflow, gone.name)), {
    undo: () => { const t = useWf.getState(); useWf.setState({ saved: putRevision(t.saved, structuredClone(gone), i), ...(wasOn ? { open: { workflow: id.workflow, revision: gone.name, revFor: gone.engine } } : {}) }); persistSaved(); },
    redo: del,
  });
  toast(msg('wf.deleted.rev', { wf: id.workflow, name: gone.name }), undoAction(entry));
}

/* ---------------------------------------------------------------- import */

/** Takes a revision file in: cleaned, filed under its workflow, said with what did not come through; Undo takes it out
 *  (and puts back what it replaced). */
export function importFile(text: string): void {
  let v: unknown;
  try { v = JSON.parse(text); } catch { toast(msg('wf.import.notJson')); return; }
  try {
    const s = useWf.getState(), r = importRevision(ix, s.saved, v, s.open.workflow);
    if (!r.ok) { toast(r.says); return; }
    const sel = `.pre.rev[data-rev="${CSS.escape(r.wf)}"][data-rn="${CSS.escape(r.r.name)}"][data-rb="${r.r.engine}"]`;
    const put = () => {
      let t = useWf.getState().saved;
      if (r.newWf && workflowIndex(t, r.newWf.name) < 0) t = putWorkflow(t, structuredClone(r.newWf), 0);
      t = putRevision(t, structuredClone(r.r));
      const u = useWf.getState();
      useWf.setState({ saved: t, expanded: [...new Set([...u.expanded, r.wf])], wash: { sel, n: (u.wash?.n ?? 0) + 1 } }); persistSaved();
    };
    const unput = () => {
      let t = useWf.getState().saved;
      t = r.was ? putRevision(t, structuredClone(r.was)) : dropRevision(t, { workflow: r.wf, engine: r.r.engine, name: r.r.name });
      if (r.newWf && !t.revisions.some(x => sameName(x.workflow, r.newWf!.name))) t = dropWorkflow(t, r.newWf.name);
      useWf.setState({ saved: t }); persistSaved();
    };
    put();
    const entry = desk().recordStep(recordStep('import', r.r.name, whatRevision(r.wf, r.r.name)), { undo: unput, redo: put });
    toast(r.says, undoAction(entry));
  } catch { toast(msg('wf.import.failed')); }
}

/* ---------------------------------------------------------------- drafts */

const draftId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
/** Saves the desk's content as a draft; past thirty the oldest makes room, said with Undo. */
export function saveDraft(name: string): void {
  const d = draftOf(ix, desk().desk, name.trim()), dr: StoredDraft = { ...d, id: draftId(), at: Date.now() };
  const all = [dr, ...useWf.getState().drafts], gone = all.slice(DRAFT_MAX);
  useWf.setState({ drafts: all.slice(0, DRAFT_MAX) });
  const ok = persistDrafts();
  if (!gone.length) { if (ok) desk().say(msg('draft.saved')); return; }
  const entry = desk().recordStep(draftStep('draftSave'), {
    undo: () => { useWf.setState({ drafts: [...useWf.getState().drafts.filter(x => x.id !== dr.id), ...structuredClone(gone)] }); persistDrafts(); },
    redo: () => { useWf.setState({ drafts: [structuredClone(dr), ...useWf.getState().drafts.filter(x => !gone.some(g => g.id === x.id))] }); persistDrafts(); },
  });
  if (ok) toast(msg('draft.saved.room', { name: draftTitle(ix, gone[0]!) }), undoAction(entry));
}
/** Deletes a draft at once: it is undoable, so it does not ask first (C-23). */
export function deleteDraft(id: string): void {
  const s = useWf.getState(), i = s.drafts.findIndex(d => d.id === id), gone = s.drafts[i]; if (!gone) return;
  useWf.setState({ drafts: s.drafts.filter(d => d.id !== id) }); persistDrafts();
  const entry = desk().recordStep(draftStep('draftDelete'), {
    undo: () => {
      const t = useWf.getState().drafts; if (!t.some(x => x.id === id)) { const l = [...t]; l.splice(Math.min(i, l.length), 0, structuredClone(gone)); useWf.setState({ drafts: l }); persistDrafts(); }
    },
    redo: () => { useWf.setState({ drafts: useWf.getState().drafts.filter(x => x.id !== id) }); persistDrafts(); },
  /* focus goes back into the open list, not to Generate (C-25) */
  }, '#pop [data-draftdel], #draftName');
  toast(msg('draft.deleted'), undoAction(entry));
}
/** Loads a draft: on another engine, the engine switches first; its content replaces the desk's content. */
export function loadDraft(id: string): void {
  const d = useWf.getState().drafts.find(x => x.id === id); if (!d) return;
  const from = now(), moved = onEngine(from, d.engine), to: Plan = { ...moved, desk: applyDraft(ix, moved.desk, d) }, t = draftTitle(ix, d);
  const e = desk().deskStep(draftStep('draftLoad', typeof t === 'string' ? t.slice(0, 24) : t, putOf(from, to)), () => apply(to));
  toast(draftLoaded(d, from.desk.engine !== d.engine), undoAction(e));
}
