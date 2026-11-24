// What the scenarios on workflows, revisions, drafts and storage reach into the concept page for (testing/compat.ts):
// the lists in the concept's shape (S.workflows, S.revs, S.drafts), which workflow and revision are open (S.workflow,
// S.revision, S.revFor, S.wfDirty, S.revDirty, S.wfOpen, S.wfFold), the concept's functions over them (saveUser,
// renderPresets, loadPreset, applyRev, findPreset, saveDrafts, sameName, FACTORY), and its storage keys: the page keeps
// its records under keys of its own (deviation 5), so a scenario's seed under the concept's key is put where the page
// reads it before the lists are read again, a read of the concept's key reads the page's record in the concept's shape,
// and a write of the page's key goes through the concept's name (so a scenario that makes one key refuse writes reaches
// the page's). Only the test build contains this.

import { flushSync } from 'react-dom';
import type { EngineId, Revision, Value, Workflow } from '../../../definitions/types.ts';
import { FACTORY_WORKFLOWS } from '../../../definitions/factory.ts';
import { findRevision, findWorkflow, optionsOf, revisionModified, sameName, workflowModified, type StoredDraft } from '../../../derive.ts';
import { ix } from '../../model/index.ts';
import { DESK_KEY, useDesk } from '../../store/desk.ts';
import { DRAFT_KEY, WF_KEY, loadRev, openWorkflow, persistDrafts, persistSaved, rebootWf, useWf } from '../../store/workflows.ts';
import { CONCEPT_ID } from '../../shell/engines.ts';
import { sayNow } from '../../i18n/index.ts';
import { TO_ENGINE, asScene, expose, fromConcept, onMirror, onScene, toConcept, S } from '../compat.ts';

/* ---------------------------------------------------------------- the concept's shapes */

type CWf = { name: string; tier: string; mods: string[]; pedals: string[]; note?: string };
type CRev = { name: string; backend: string; vals: Record<string, unknown>; slots: unknown[]; note?: string; user?: boolean };
type CDraft = { id?: string; name: string; backend: string; at: number; prompt: string; vals: Record<string, unknown> };

const wfOut = (w: Workflow): CWf => ({ name: w.name, tier: w.tier, mods: [...w.modules], pedals: [...w.board], ...(w.note ? { note: w.note } : {}) });
const wfIn = (w: CWf): Workflow => ({ v: 1, name: w.name, tier: w.tier as Workflow['tier'], modules: [...(w.mods ?? [])], board: [...(w.pedals ?? [])], ...(w.note ? { note: w.note } : {}) });
const valsOut = (vals: { [k: string]: Value }, auto: { [k: string]: boolean } | undefined, e: EngineId) => {
  const o: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(vals)) o[k] = toConcept(k, v, e);
  for (const [k, a] of Object.entries(auto ?? {})) o[k + '.auto'] = a;
  return o;
};
const valsIn = (vals: Record<string, unknown>, e: EngineId) => {
  const v: { [k: string]: Value } = {}, auto: { [k: string]: boolean } = {};
  for (const [k, x] of Object.entries(vals ?? {})) { if (k.endsWith('.auto')) { if (typeof x === 'boolean') auto[k.slice(0, -5)] = x; } else { const y = fromConcept(k, x, e); if (y !== undefined) v[k] = y; } }
  return { vals: v, auto };
};
const revOut = (r: Revision): CRev => ({ name: r.name, backend: CONCEPT_ID[r.engine], vals: valsOut(r.vals, r.auto, r.engine), slots: Array.isArray(r.state?.slots) ? structuredClone(r.state.slots) : [], ...(r.note ? { note: r.note } : {}) });
const revIn = (wf: string, r: CRev): Revision | null => {
  const e = TO_ENGINE[r.backend]; if (!e) return null;
  const { vals, auto } = valsIn(r.vals, e);
  return { v: 1, workflow: wf, name: r.name, engine: e, vals, ...(Object.keys(auto).length ? { auto } : {}), ...(Array.isArray(r.slots) && r.slots.length ? { state: { slots: structuredClone(r.slots) as never } } : {}), ...(r.note ? { note: r.note } : {}) };
};
const draftOut = (d: StoredDraft): CDraft => { const { 'prompt.text': p, ...rest } = d.vals; return { id: d.id, name: d.name, backend: CONCEPT_ID[d.engine], at: d.at, prompt: typeof p === 'string' ? p : '', vals: valsOut(rest, d.auto, d.engine) }; };
const draftIn = (d: CDraft, i: number): StoredDraft | null => {
  const e = TO_ENGINE[d.backend]; if (!e) return null;
  const { vals, auto } = valsIn(d.vals, e);
  return { v: 1, id: d.id ?? `${d.at}-${i}`, name: d.name ?? '', engine: e, at: d.at, vals: { ...vals, ...(d.prompt ? { 'prompt.text': d.prompt } : {}) }, ...(Object.keys(auto).length ? { auto } : {}) };
};
const savedOut = () => {
  const s = useWf.getState().saved, revs: Record<string, CRev[]> = Object.create(null) as Record<string, CRev[]>;
  for (const r of s.revisions) (revs[r.workflow] ??= []).push(revOut(r));
  return { workflows: s.workflows.map(wfOut), revs };
};

/* ---------------------------------------------------------------- S, kept in step */

let lastWf = '', lastRevs = '', lastDrafts = '';
function mirror(): void {
  const w = useWf.getState(), d = useDesk.getState(), out = savedOut();
  S.workflows = out.workflows; S.revs = out.revs; S.drafts = w.drafts.map(draftOut);
  lastWf = JSON.stringify(S.workflows); lastRevs = JSON.stringify(S.revs); lastDrafts = JSON.stringify(S.drafts);
  S.workflow = w.open.workflow; S.revision = w.open.revision; S.revFor = w.open.revFor ? CONCEPT_ID[w.open.revFor] : null;
  const wf = findWorkflow(w.saved, w.open.workflow)?.w, r = w.open.workflow && w.open.revision && w.open.revFor ? findRevision(w.saved, { workflow: w.open.workflow, engine: w.open.revFor, name: w.open.revision }) : null;
  S.wfDirty = !!wf && workflowModified(ix, wf, d.mods, d.board);
  S.revDirty = !!r && revisionModified(ix, r, d.desk);
  S.wfOpen = new Set(w.expanded); S.wfFold = w.fold;
}

/** The lists as the scene in S has them, when the scene changed them. */
function takeScene(): boolean {
  const ws = JSON.stringify(S.workflows ?? []), rs = JSON.stringify(S.revs ?? {}), ds = JSON.stringify(S.drafts ?? []);
  const patch: Partial<ReturnType<typeof useWf.getState>> = {};
  if (ws !== lastWf || rs !== lastRevs) {
    const revs = Object.entries((S.revs ?? {}) as Record<string, CRev[]>).flatMap(([wf, l]) => (Array.isArray(l) ? l : []).map(r => revIn(wf, r)).filter((r): r is Revision => !!r));
    patch.saved = { workflows: ((S.workflows ?? []) as CWf[]).map(wfIn), revisions: revs };
  }
  if (ds !== lastDrafts) patch.drafts = ((S.drafts ?? []) as CDraft[]).map(draftIn).filter((d): d is StoredDraft => !!d);
  const open = { workflow: (S.workflow ?? null) as string | null, revision: (S.revision ?? null) as string | null, revFor: (S.revFor ? TO_ENGINE[S.revFor as string] ?? null : null) as EngineId | null };
  const now = useWf.getState().open;
  if (open.workflow !== now.workflow || open.revision !== now.revision || open.revFor !== now.revFor) patch.open = open;
  const exp = [...((S.wfOpen as Set<string> | undefined) ?? [])];
  if (exp.join('\u0000') !== useWf.getState().expanded.join('\u0000')) patch.expanded = exp;
  if (typeof S.wfFold === 'boolean' && S.wfFold !== useWf.getState().fold) patch.fold = S.wfFold;
  if (!Object.keys(patch).length) return false;
  asScene(() => useWf.setState(patch));
  return !!(patch.saved || patch.drafts);
}

/* ---------------------------------------------------------------- storage keys: the concept's and the page's */

const KEYS: Record<string, string> = { 'hs3-wf': WF_KEY, 'hs3-drafts': DRAFT_KEY, 'hs3-desk': DESK_KEY };
const BACK: Record<string, string> = Object.fromEntries(Object.entries(KEYS).map(([c, o]) => [o, c]));
function storageKeys(): void {
  const get = Storage.prototype.getItem, set = Storage.prototype.setItem;
  /* a seed under the concept's key is the page's to read (once, before the page wrote its own) */
  let seeded = false;
  for (const [c, o] of Object.entries(KEYS)) { const v = get.call(localStorage, c); if (v != null && get.call(localStorage, o) == null && c !== 'hs3-desk') { set.call(localStorage, o, v); seeded = true; } }
  if (seeded) rebootWf();
  Storage.prototype.getItem = function (k: string) {
    if (this === localStorage) {
      if (k === 'hs3-wf' && get.call(this, WF_KEY) != null) return JSON.stringify(savedOut());
      if (k === 'hs3-drafts' && get.call(this, DRAFT_KEY) != null) return JSON.stringify(useWf.getState().drafts.map(draftOut));
      if (k === 'hs3-desk') return get.call(this, DESK_KEY);
      const m = /^(hs3-(?:wf|drafts|desk))(-unreadable.*)$/.exec(k);
      if (m) return get.call(this, KEYS[m[1]!]! + m[2]);
    }
    return get.call(this, k);
  };
  let inner = false;
  Storage.prototype.setItem = function (k: string, v: string) {
    if (this === localStorage) {
      /* the page's key is written under the concept's name first, so a scenario's own setItem sees it */
      if (!inner && k in BACK) { inner = true; try { Storage.prototype.setItem.call(this, BACK[k]!, v); } finally { inner = false; } return; }
      if (k in KEYS) { set.call(this, KEYS[k]!, v); return; }
    }
    set.call(this, k, v);
  };
}

export function install(): void {
  storageKeys();
  mirror();
  useWf.subscribe(mirror);
  onMirror(() => mirror());
  /* a choice the server lists, as the concept keeps it: by its words (S.vals['samp.solver'] is "DPM++ 3M") */
  onMirror((S0, s) => {
    for (const k of Object.keys(S0.vals)) {
      if (ix.params.get(k)?.p.type !== 'choice' || !(k in s.desk.vals)) continue;
      const o = optionsOf(ix, k, s.desk).find(x => x.value === s.desk.vals[k]);
      if (o && typeof o.label !== 'string') continue;
      if (o) S0.vals[k] = sayNow(o.label);
    }
  });
  onScene(() => { takeScene(); });
  const asStep = (fn: () => void) => flushSync(fn);
  const revOf = (wf: string, r: CRev) => { const e = TO_ENGINE[r.backend]; if (!e) return null; const id = { workflow: wf, engine: e, name: r.name }; return { id, user: !!findRevision(useWf.getState().saved, id, true) }; };
  expose({
    booted: true,
    FACTORY: FACTORY_WORKFLOWS.map(wfOut),
    sameName,
    findPreset: (n: string) => { const f = findWorkflow(useWf.getState().saved, n); return f ? wfOut(f.w) : undefined; },
    /* the concept's saveUser writes S's lists; renderPresets draws them */
    saveUser() { takeScene(); flushSync(() => { persistSaved(); }); },
    renderPresets() { takeScene(); },
    saveDrafts() { takeScene(); flushSync(() => { persistDrafts(); }); },
    loadPreset(w: CWf, tier?: boolean, revert?: boolean) { takeScene(); asStep(() => openWorkflow(w.name, { tier: !!tier, revert: !!revert })); },
    applyRev(wf: string, r: CRev, revert?: boolean) { takeScene(); const x = revOf(wf, r); if (x) asStep(() => loadRev(x.id, x.user, !!revert)); },
    autosave() { /* the page writes the desk on every change */ },
  });
}
