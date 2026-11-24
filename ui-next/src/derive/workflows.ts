// The user's saved work (docs/05, "Saved data"; the concept's 32-workflows.js, 35-drafts.js, 13-stored.js): what the
// Workflows list and the Drafts list show and say, the one rule for names, reading stored lists record by record,
// merging what another tab saved, what "modified" means, importing and exporting a revision, and every word these
// say. Records themselves (what a revision and a workflow hold, migration, loading) are records.ts. Pure functions of
// the definitions and the desk; the page's store (src/app/store/workflows.ts) keeps the lists and writes them.

import type { Draft, EngineId, Json, Revision, Tier, Value, Workflow } from '../definitions/types.ts';
import { SAVED_VERSION } from '../definitions/types.ts';
import { ENGINE_IDS, engineName, engineShort } from '../definitions/engines.ts';
import { FACTORY_REVISIONS, FACTORY_WORKFLOWS } from '../definitions/factory.ts';
import { RETIRED } from '../definitions/retired.ts';
import { type Desk, type Index, has, resolve } from './core.ts';
import { cleanStates, cleanValues } from './values.ts';
import { EN, list, msg, type Msg, type Slot } from './messages.ts';
import { migrate } from './records.ts';
import { formatMsg, paramWords } from './strings.ts';
import { defName, stepName, type StepWords } from './steps.ts';

/* ---------------------------------------------------------------- the words */

/** The core's words for saved work, added to the core's table (messages.ts EN) as this file loads. */
const WORDS: { [id: string]: string } = {
  'wf.step.open': 'Open "{name}"',
  'wf.step.revert': 'Revert "{name}"',
  'wf.step.load': 'Load "{name}"',
  'wf.step.replace': 'Replace "{name}"',
  'wf.step.update': 'Update "{name}"',
  'wf.step.rename': 'Rename "{name}"',
  'wf.step.delete': 'Delete "{name}"',
  'wf.step.import': 'Import "{name}"',
  'wf.step.save': 'Save "{name}"',
  'wf.step.draftSave': 'Save draft',
  'wf.step.draftDelete': 'Delete draft',
  'wf.step.draftLoad': 'Load draft "{name}"',
  'wf.put.record': 'Puts back {what}',
  'wf.what.workflow': 'the "{name}" workflow',
  'wf.what.revision': '{wf} › {name}',
  'wf.what.draft': 'the draft',
  'wf.what.drafts': 'the drafts',
  'wf.what.settings': '{n, plural, one {# setting} other {# settings}}',
  'wf.opened': 'Opened {name}',
  'wf.reverted': 'Put back the modules of {name}',
  'wf.loaded': 'Loaded {wf} › {name}{on}{inc}{un}',
  'wf.loaded.on': ' on {engine}',
  'wf.rev.reverted': 'Put back the settings of {name}{inc}{un}',
  'wf.inc.lyrics': ', lyrics included',
  'wf.inc.box': ', {box} included',
  'wf.unloaded': ' · {files} unloaded',
  'wf.saved.rev': 'Saved {wf} › {name} for {engine}',
  'wf.saved.wf': 'Saved the "{name}" workflow',
  'wf.saved.wf.more': 'Save a revision too',
  'wf.replaced.rev': 'Replaced {wf} › {name}',
  'wf.replaced.wf': 'Replaced the "{name}" workflow\'s layout',
  'wf.updated.wf': 'Updated the "{name}" workflow\'s modules',
  'wf.updated.rev': 'Updated {wf} › {name}',
  'wf.renamed.wf': 'Renamed "{from}" to "{to}"',
  'wf.renamed.rev': 'Renamed {wf} › "{from}" to "{to}"',
  'wf.deleted.wf': 'Deleted the "{name}" workflow{n, plural, =0 {} one { and its # revision} other { and its # revisions}}',
  'wf.deleted.rev': 'Deleted {wf} › {name}',
  'wf.exported': 'Exported {wf} › {name}',
  'wf.imported': '{replaced, select, true {Replaced} other {Imported}} {wf} › {name} ({engine}){notes}',
  'wf.imported.notes': '. {notes}',
  'wf.import.dropped': '{n, plural, one {# value didn\'t fit its control and was} other {# values didn\'t fit their controls and were}} left out',
  'wf.import.limited': '{n, plural, one {# value was limited to its range} other {# values were limited to their range}}',
  'wf.import.unknown': '{n, plural, one {# module isn\'t} other {# modules aren\'t}} known here',
  'wf.import.cut': 'text too long for its field was shortened',
  'wf.import.beside': 'filed as "{name}" beside the factory one',
  'wf.import.notJson': 'That file isn\'t valid JSON',
  'wf.import.notRevision': 'That file isn\'t a HOT-Step revision',
  'wf.import.failed': 'That file couldn\'t be imported. Nothing was changed.',
  'wf.taken.now': '"{old}" is taken now, so it keeps the name "{name}"',
  'wf.taken.again': '"{name}" is taken now',
  'wf.gone.wf': '"{name}" was deleted in another tab. Your desk is unchanged.',
  'wf.gone.rev': '"{name}" was deleted in another tab. Your settings are unchanged.',
  'wf.gone.here': 'That workflow is no longer here.',
  'wf.name.empty': 'Give it a name first.',
  'wf.name.factory': 'That name belongs to a factory workflow. Pick another.',
  'wf.name.taken': 'The "{name}" workflow already exists. Pick another name.',
  'wf.name.replace': 'The "{name}" workflow already exists. Replace overwrites its layout, and Undo brings it back.',
  'rev.name.factory': 'That name belongs to a factory revision here. Pick another.',
  'rev.name.taken': '"{name}" already exists under {wf}. Pick another name.',
  'rev.name.replace': '"{name}" already exists under {wf}. Replace overwrites it, and Undo brings it back.',
  'wf.hint.rev': 'Every setting for {engine}, saved under {wf}.{repaint}{moved}',
  'wf.hint.repaint': ' Repaint’s region and settings stay with its song.',
  'wf.hint.moved': ' The desk changed since you opened it: a revision keeps only settings, so save a new workflow to keep the new modules.',
  'wf.hint.wf': 'Saves the desk layout. Settings go in a revision.',
  'wf.saveAs.rev.title': 'A revision of {name}',
  'wf.saveAs.rev.none': 'Open a workflow first',
  'wf.row.label': '{name}, {tier}{n, plural, =0 {} one {, # revision} other {, # revisions}}{mod, select, true {, modified} other {}}',
  'wf.row.tip': '{what} · {tier}',
  'wf.row.modules': '{n, plural, one {# module} other {# modules}}',
  'wf.row.count': '{n, plural, one {# revision} other {# revisions}}',
  'wf.row.chev': 'Show the revisions of {name}',
  'wf.row.openIn': 'Open in {tier}',
  'wf.row.openIn.label': 'Open {name} in {tier}',
  'rev.row.label': '{name}{mine, select, true {, yours} other {}}, {engine}{holds, select, lyrics {, holds lyrics} box {, holds a {box}} other {}}{mod, select, true {, modified} other {}}',
  'rev.row.tip.load': 'Load {wf}, then this revision\'s settings',
  'rev.row.tip.changed': '{n, plural, one {# setting} other {# settings}} changed since it was saved. The info button shows which.',
  'rev.row.tip.content': 'What it holds besides settings (description, lyrics or song info) changed since it was saved',
  'rev.row.tip.mod': 'Changed since it was saved',
  'rev.row.holds.lyrics': 'Holds lyrics too',
  'rev.row.holds.box': 'Holds a {box} too',
  'rev.row.info': 'What {name} holds',
  'rev.row.info.title': 'What it holds',
  'rev.row.info.title.mod': 'What it holds, and what changed',
  'wf.yours.empty': 'Save this desk as a workflow to see it here.',
  'wf.yours.empty.revs': 'Save this desk as a workflow to see it here. Revisions you saved under a factory workflow are marked “yours” below.',
  'wf.revert.wf.title': 'Put the workflow\'s modules back',
  'wf.update.wf.title': 'Save the new layout into this workflow',
  'wf.saveAs.wf.title': 'Save this desk as a workflow of your own',
  'rev.revert.title': 'Put back the saved settings of {name}; the desk stays as it is',
  'rev.update.title': 'Save the changes into this revision',
  'rev.saveAs.title': 'Save these settings as a revision of your own',
  'wf.saveAs.name': '{name} (mine)',
  'rev.detail.empty': 'Every setting at its default.',
  'rev.detail.content': 'Content',
  'rev.detail.post': 'Post-processing',
  'rev.detail.of': '{owner} · {label}',
  'rev.detail.slot': '{type} · {v}',
  'wf.box': '{e, select, minimax-m3 {caption} sa3 {prompt} heartmula {tags} other {description}}',
  'wf.box.cap': '{e, select, minimax-m3 {Caption} sa3 {Prompt} heartmula {Tags} other {Description}}',
  'rev.detail.lyrics': 'Lyrics',
  'rev.detail.lines': '{n, plural, one {# line} other {# lines}}',
  'rev.detail.quote': '“{text}”',
  'value.auto': 'auto',
  'store.skipped': '{n, plural, one {# {one} couldn\'t be read and was} other {# {many} couldn\'t be read and were}} skipped. A copy is kept in this browser; nothing was deleted',
  'store.skipped.wf.one': 'saved workflow or revision',
  'store.skipped.wf.many': 'saved workflows or revisions',
  'store.skipped.draft.one': 'draft',
  'store.skipped.draft.many': 'drafts',
  'store.unreadable': 'Your {what} couldn\'t be read. A copy is kept in this browser; nothing was deleted',
  'store.what.wf': 'saved workflows',
  'store.what.drafts': 'drafts',
  'store.desk': 'Your last desk couldn\'t be read, so this is a fresh desk. A copy is kept in this browser; nothing was deleted',
  'store.told': '{items}.',
  'store.aside': 'Download the copy',
  'store.aside.done': 'Downloaded the copies',
  'store.failed': '{why}, so {what, select, desk {your desk} other {that}} wasn\'t saved. It is safe while this tab stays open.',
  'store.why.full': 'This browser\'s storage is full',
  'store.why.blocked': 'This browser isn\'t letting the page store anything',
  'store.state.saved': 'Saved',
  'store.state.failed': 'Not saved',
  'store.state.tip': 'Your desk, songs and settings are kept in this browser as you work',
  'store.state.tip.failed': '{why}, so the last change was not saved. Your work is safe while this tab stays open. Export what you need{full, select, true {, or free some space,} other {}} and it saves again.',
  'draft.saved': 'Draft saved',
  'draft.saved.room': 'Draft saved. The oldest one, "{name}", made room',
  'draft.loaded': 'Loaded the draft{named, select, true { "{name}"} other {}}{on}',
  'draft.deleted': 'Deleted the draft',
  'draft.untitled': 'Untitled',
  'draft.empty': 'Empty',
  'draft.words': '{n, plural, one {# word} other {# words}}',
  'draft.line': '“{line}”',
  'draft.sub': '{when}{other, select, true { · {engine}} other {}} · {what}',
  'draft.yesterday': 'Yesterday {time}',
  'draft.same': 'Same as your last draft. Name it to keep a named copy.',
  'draft.none': 'No drafts yet.',
  'draft.load.title': 'Switches the engine to {engine} and loads the draft',
  'draft.delete.label': 'Delete draft {name}',
  'draft.about': '{box}{lyrics, select, true {, lyrics} other {}} and song info · last {n} kept',
  'draft.about.title': 'Saves the {box}, {lyrics, select, true {lyrics} other {any lyrics kept for other engines}}, song info, with the engine they belong to',
};
for (const [id, text] of Object.entries(WORDS)) {
  if (id in EN) throw new Error(`the core's word ${id} is defined twice`);
  EN[id] = text;
}

/* ---------------------------------------------------------------- names (F-02, F-10)
   One rule for every path that names a workflow or a revision: save, Save as, rename, import, the stored list and the
   merge between tabs. A name is taken in its normalised form (NFC, trimmed, a run of white space as one space) and
   compared in that form whatever its capitals, so two names that look the same are the same name, and no path can file
   two records under one key. An exact match is looked up first, so a list saved before the rule that holds
   "songwriter" beside the factory "Songwriter" still opens each. */

export const NAME_MAX = 28;
export const DRAFT_NAME_MAX = 40;
export const DRAFT_MAX = 30;
export const normName = (s: unknown): string => String(s ?? '').normalize('NFC').replace(/\s+/g, ' ').trim();
export const nameKey = (s: unknown): string => normName(s).toLowerCase();
export const sameName = (a: unknown, b: unknown): boolean => nameKey(a) === nameKey(b);

/** The user's saved workflows and revisions. A revision names its workflow (`workflow`), so a workflow may be called
 *  anything, "__proto__" included (F-20). */
export interface Saved { workflows: Workflow[]; revisions: Revision[] }
export const emptySaved = (): Saved => ({ workflows: [], revisions: [] });

/** A workflow by name, the user's or the factory's: an exact match first, then the same name. */
export function findWorkflow(saved: Saved, name: string | null | undefined): { w: Workflow; user: boolean } | null {
  if (name == null) return null;
  const u = saved.workflows.find(w => w.name === name), f = FACTORY_WORKFLOWS.find(w => w.name === name);
  if (u) return { w: u, user: true };
  if (f) return { w: f, user: false };
  const u2 = saved.workflows.find(w => sameName(w.name, name)), f2 = FACTORY_WORKFLOWS.find(w => sameName(w.name, name));
  return u2 ? { w: u2, user: true } : f2 ? { w: f2, user: false } : null;
}
export const isFactory = (name: string): boolean => FACTORY_WORKFLOWS.some(w => w.name === name);

/** A revision's identity: its workflow, its engine and its name. */
export interface RevId { workflow: string; engine: EngineId; name: string }
const sameRev = (r: RevId, id: RevId) => sameName(r.workflow, id.workflow) && r.engine === id.engine && sameName(r.name, id.name);
/** The revisions under a workflow: the user's first, then the factory's. */
export function revisionsOf(saved: Saved, wf: string): { r: Revision; user: boolean }[] {
  return [...saved.revisions.filter(r => sameName(r.workflow, wf)).map(r => ({ r, user: true })), ...FACTORY_REVISIONS.filter(r => r.workflow === wf).map(r => ({ r, user: false }))];
}
/** A revision by its identity, the user's (exactly named first) or the factory's. */
export function findRevision(saved: Saved, id: RevId, user?: boolean): Revision | null {
  const mine = () => saved.revisions.find(r => r.workflow === id.workflow && r.engine === id.engine && r.name === id.name) ?? saved.revisions.find(r => sameRev(r, id));
  const factory = () => FACTORY_REVISIONS.find(r => r.workflow === id.workflow && r.engine === id.engine && r.name === id.name);
  return (user === false ? factory() : user === true ? mine() : mine() ?? factory()) ?? null;
}

/** Why a workflow cannot take this name (null: it can); `self` is the workflow being renamed. */
export function wfNameClash(saved: Saved, name: string, self?: string): Msg | null {
  if (FACTORY_WORKFLOWS.some(w => sameName(w.name, name) && w.name !== self)) return msg('wf.name.factory');
  const c = saved.workflows.find(w => sameName(w.name, name) && w.name !== self);
  return c ? msg('wf.name.taken', { name: c.name }) : null;
}
/** Why a revision cannot take this name under its workflow on its engine (null: it can). */
export function revNameClash(saved: Saved, wf: string, engine: EngineId, name: string, self?: string): Msg | null {
  if (FACTORY_REVISIONS.some(r => r.workflow === wf && r.engine === engine && sameName(r.name, name))) return msg('rev.name.factory');
  const c = saved.revisions.find(r => sameName(r.workflow, wf) && r.engine === engine && sameName(r.name, name) && r.name !== self);
  return c ? msg('rev.name.taken', { name: c.name, wf }) : null;
}

/** What Save does with a name: refuses it (said beside the field), asks to replace a record of the same name (whose
 *  own name is kept, C-02), or saves. */
export type SaveCheck = { refuse: Msg } | { replace: string; says: Msg } | { ok: string };
export function saveCheck(saved: Saved, as: 'rev' | 'wf', wf: string | null, engine: EngineId, typed: string): SaveCheck {
  const n = normName(typed).slice(0, NAME_MAX).trim();
  if (!n) return { refuse: msg('wf.name.empty') };
  if (as === 'rev' && wf) {
    if (FACTORY_REVISIONS.some(r => r.workflow === wf && r.engine === engine && sameName(r.name, n))) return { refuse: msg('rev.name.factory') };
    const c = saved.revisions.find(r => sameName(r.workflow, wf) && r.engine === engine && sameName(r.name, n));
    return c ? { replace: c.name, says: msg('rev.name.replace', { name: c.name, wf }) } : { ok: n };
  }
  if (FACTORY_WORKFLOWS.some(w => sameName(w.name, n))) return { refuse: msg('wf.name.factory') };
  const c = saved.workflows.find(w => sameName(w.name, n));
  return c ? { replace: c.name, says: msg('wf.name.replace', { name: c.name }) } : { ok: n };
}

/* ---------------------------------------------------------------- the lists, as pure operations
   Every change to the saved lists goes through these, by name, so an undo finds its record after a merge with another
   tab replaced the objects (the concept's putRev, dropRev, putWf). */

const at = <T>(l: T[], i: number) => Math.max(0, Math.min(i, l.length));
export function putWorkflow(s: Saved, w: Workflow, i = 0): Saved {
  const k = s.workflows.findIndex(x => sameName(x.name, w.name)), workflows = [...s.workflows];
  if (k >= 0) workflows[k] = w; else workflows.splice(at(workflows, i), 0, w);
  return { ...s, workflows };
}
export function dropWorkflow(s: Saved, name: string): Saved {
  return { workflows: s.workflows.filter(w => !sameName(w.name, name)), revisions: s.revisions.filter(r => !sameName(r.workflow, name)) };
}
export function putRevision(s: Saved, r: Revision, i = 0): Saved {
  const k = s.revisions.findIndex(x => sameRev(x, r)), revisions = [...s.revisions];
  if (k >= 0) revisions[k] = r; else {
    /* placed among its workflow's revisions: before the i-th of them */
    const mine = revisions.map((x, j) => [x, j] as const).filter(([x]) => sameName(x.workflow, r.workflow));
    const j = mine[i]?.[1] ?? (mine.length ? mine[mine.length - 1]![1] + 1 : revisions.length);
    revisions.splice(j, 0, r);
  }
  return { ...s, revisions };
}
export function dropRevision(s: Saved, id: RevId): Saved { return { ...s, revisions: s.revisions.filter(r => !sameRev(r, id)) }; }
/** Where a revision stands among its workflow's revisions (-1: not there). */
export const revisionIndex = (s: Saved, id: RevId): number => s.revisions.filter(r => sameName(r.workflow, id.workflow)).findIndex(r => sameRev(r, id));
export const workflowIndex = (s: Saved, name: string): number => s.workflows.findIndex(w => sameName(w.name, name));
/** A rename takes the workflow's revisions with it (F-02). */
export function renameWorkflow(s: Saved, from: string, to: string): Saved {
  return { workflows: s.workflows.map(w => (w.name === from ? { ...w, name: to } : w)), revisions: s.revisions.map(r => (r.workflow === from ? { ...r, workflow: to } : r)) };
}
export function renameRevision(s: Saved, id: RevId, to: string): Saved {
  return { ...s, revisions: s.revisions.map(r => (r.workflow === id.workflow && r.engine === id.engine && r.name === id.name ? { ...r, name: to } : r)) };
}

/* ---------------------------------------------------------------- reading what was stored (C-08)
   Record by record: a workflow needs a name and a module list, a revision a name, a known engine and its values; what
   else is wrong is repaired (unknown modules and pedals dropped, the tier Studio, values cleaned). A record that cannot
   be repaired is skipped and counted, never trusted. Both the page's own list ({ v: 1, workflows, revisions }) and the
   concept's ({ workflows, revs: { <workflow>: [...] } }, version 0) are read. */

const isRec = (x: unknown): x is Record<string, unknown> => !!x && typeof x === 'object' && !Array.isArray(x);
const NAME_LIMIT = 200;
const goodName = (x: unknown): x is string => typeof x === 'string' && !!normName(x) && x.length <= NAME_LIMIT;
const TIERS: readonly Tier[] = ['quick', 'studio', 'lab'];

/** A workflow as it can be kept, or null. */
export function cleanWorkflow(ix: Index, raw: unknown): Workflow | null {
  if (!isRec(raw) || (typeof raw.v === 'number' && raw.v > SAVED_VERSION) || !goodName(raw.name)) return null;
  const mods = raw.modules ?? raw.mods, board = raw.board ?? raw.pedals;
  if (!Array.isArray(mods)) return null;
  const known = (x: unknown, kind: 'module' | 'board') => {
    const id = typeof x === 'string' ? ix.formerIds.get(x) ?? x : null, d = id ? ix.defs.find(y => y.id === id) : undefined;
    return d && (kind === 'module' ? d.kind === 'module' && !d.builtIn && !d.source : d.kind !== 'module') ? d.id : null;
  };
  const ids = (l: unknown[], kind: 'module' | 'board') => [...new Set(l.map(x => known(x, kind)).filter((x): x is string => !!x))];
  const w: Workflow = { v: SAVED_VERSION, name: normName(raw.name), tier: TIERS.includes(raw.tier as Tier) ? raw.tier as Tier : 'studio', modules: ids(mods, 'module'), board: Array.isArray(board) ? ids(board, 'board') : [] };
  if (typeof raw.note === 'string') w.note = raw.note.slice(0, NAME_LIMIT);
  return w;
}

/** A revision as it can be kept, or null; `dropped` counts the values (and list entries) it could not keep. */
export function cleanRevision(ix: Index, raw: unknown, wf?: string): { r: Revision; dropped: number; limited: number } | null {
  if (!isRec(raw)) return null;
  const r0 = wf !== undefined && !('workflow' in raw) && !('wf' in raw) ? { ...raw, wf } : raw;
  /* the concept's own shape (version 0) keeps the prompt and content values under `content`; migrate reads it */
  const m = migrate(r0);
  if (!m || m.kind !== 'revision') return null;
  const x = m.record as unknown as Record<string, unknown>;
  if (!goodName(x.name) || !ENGINE_IDS.includes(x.engine as EngineId) || !isRec(x.vals) || !goodName(x.workflow)) return null;
  const c = cleanValues(ix, x.vals, x.auto, RETIRED.keys), st = cleanStates(ix, x.state, RETIRED.states);
  /* a list entry a state could not keep (an adapter slot that is no slot) counts as a value left out (C-24) */
  const lost = Object.entries(isRec(x.state) ? x.state : {}).reduce((n, [k, v]) => n + (Array.isArray(v) && Array.isArray(st.state[k]) ? v.length - (st.state[k] as Json[]).length : 0), 0);
  const limited = Object.keys(c.vals).filter(k => typeof (x.vals as Record<string, unknown>)[k] === 'number' && c.vals[k] !== (x.vals as Record<string, unknown>)[k]).length;
  const r: Revision = { v: SAVED_VERSION, workflow: normName(x.workflow), name: normName(x.name), engine: x.engine as EngineId, vals: c.vals, ...(Object.keys(c.auto).length ? { auto: c.auto } : {}),
    ...(Object.keys(st.state).length ? { state: st.state } : {}), ...(x.withContent === true ? { withContent: true as const } : {}), ...(typeof x.note === 'string' ? { note: x.note.slice(0, NAME_LIMIT) } : {}) };
  return { r, dropped: c.dropped.length + st.dropped.length + lost, limited };
}

/** A stored list read record by record: what can be kept, and how many records could not (null: it is no list). */
export function readSaved(ix: Index, raw: unknown): { saved: Saved; bad: number } | null {
  if (!isRec(raw)) return null;
  let bad = 0;
  const workflows: Workflow[] = [], revisions: Revision[] = [];
  for (const x of Array.isArray(raw.workflows) ? raw.workflows : []) {
    const w = cleanWorkflow(ix, x);
    if (!w || workflows.some(y => sameName(y.name, w.name))) bad++; else workflows.push(w);
  }
  const take = (x: unknown, wf?: string) => {
    const c = cleanRevision(ix, x, wf);
    if (!c || revisions.some(y => sameRev(y, c.r))) bad++; else revisions.push(c.r);
  };
  if (Array.isArray(raw.revisions)) raw.revisions.forEach(x => take(x));
  if (isRec(raw.revs)) for (const [wf, l] of Object.entries(raw.revs)) { if (Array.isArray(l)) l.forEach(x => take(x, wf)); else bad++; }
  return { saved: { workflows, revisions }, bad };
}
/** The list as it is stored. */
export const savedFile = (s: Saved) => ({ v: SAVED_VERSION, workflows: s.workflows, revisions: s.revisions });

/* ---------------------------------------------------------------- two tabs (C-09)
   merge3(base, ours, theirs, key): records keyed by key(x). base is what this tab last read or wrote, ours what it holds
   now, theirs what storage holds now. A record only one side changed takes that side's version (an addition, an edit or
   a deletion); a record both changed takes ours. New records from the other tab come first, as new saves do. Records
   compare by content, whatever order their fields were written in. */

const canon = (x: unknown) => JSON.stringify(x, (_k, v: unknown) => (isRec(v) ? Object.fromEntries(Object.keys(v).sort().map(n => [n, v[n]])) : v));
export function merge3<T>(base: readonly T[], ours: readonly T[], theirs: readonly T[], key: (x: T) => string): T[] {
  const m = (l: readonly T[]) => new Map(l.map(x => [key(x), canon(x)])), B = m(base), O = m(ours), T0 = m(theirs), out: T[] = [], seen = new Set<string>();
  const pick = (k: string) => { const b = B.get(k), o = O.get(k), t = T0.get(k); return o === b ? t : t === b ? o : o ?? t; };
  const put = (k: string) => { if (seen.has(k)) return; seen.add(k); const v = pick(k); if (v !== undefined) out.push(JSON.parse(v) as T); };
  theirs.forEach(x => { const k = key(x); if (!B.has(k) && !O.has(k)) put(k); });
  ours.forEach(x => put(key(x))); theirs.forEach(x => put(key(x)));
  return out;
}
const wfKey = (w: Workflow) => nameKey(w.name);
const revKey = (r: Revision) => `${nameKey(r.workflow)}\u0000${r.engine}\u0000${nameKey(r.name)}`;
export const mergeSaved = (base: Saved, ours: Saved, theirs: Saved): Saved =>
  ({ workflows: merge3(base.workflows, ours.workflows, theirs.workflows, wfKey), revisions: merge3(base.revisions, ours.revisions, theirs.revisions, revKey) });

/* ---------------------------------------------------------------- a workflow on the desk */

/** A workflow's modules and board as the desk lays them out: known ids, no source; a board with pedals brings the
 *  Pedalboard, and without the Pedalboard there is no board. */
export function layoutOf(ix: Index, w: Pick<Workflow, 'modules' | 'board'>): { mods: string[]; board: string[] } {
  const id = (x: string) => ix.formerIds.get(x) ?? x, def = (x: string) => ix.defs.find(d => d.id === id(x));
  const mods = [...new Set(w.modules.map(id))].filter(x => { const d = def(x); return d?.kind === 'module' && !d.builtIn && !d.source; });
  const order = new Map(ix.defs.filter(d => d.kind !== 'module').sort((a, b) => (a.kind === b.kind ? a.at - b.at : a.kind === 'pedal' ? -1 : 1)).map((d, i) => [d.id, i]));
  let board = [...new Set(w.board.map(id))].filter(x => order.has(x)).sort((a, b) => order.get(a)! - order.get(b)!);
  const holder = ix.defs.find(d => d.kind === 'module' && !!d.board);
  if (holder && board.length && !mods.includes(holder.id)) mods.push(holder.id);
  if (!holder || !mods.includes(holder.id)) board = [];
  return { mods, board };
}
/** The open workflow is modified while the desk's modules or board differ from what Revert would put back (F-04). */
export function workflowModified(ix: Index, w: Workflow, mods: readonly string[], board: readonly string[]): boolean {
  const mine = layoutOf(ix, w), now = layoutOf(ix, { modules: [...mods], board: [...board] });
  return mine.mods.join() !== now.mods.join() || mine.board.join() !== now.board.join();
}

/* ---------------------------------------------------------------- a revision against the desk */

const storedOf = (ix: Index, key: string) => { const h = ix.params.get(key)!; return h.p.stored ?? h.d.stored; };
const isSource = (ix: Index, key: string) => { const d = ix.params.get(key)!.d; return d.kind === 'module' && !!d.source; };
/** The parameters a revision of an engine keeps: its settings (and its content with content), never a source's. */
export const revisionKeys = (ix: Index, e: EngineId, s: 'settings' | 'content'): string[] =>
  [...ix.params.keys()].filter(k => has(ix, k, e) && !isSource(ix, k) && storedOf(ix, k) === s);

/** A value as a revision holds it: "auto" on auto, else the value (a default where it holds none). */
function held(ix: Index, key: string, e: EngineId, vals: { [k: string]: Value }, auto: { [k: string]: boolean } | undefined): Value | 'auto' {
  const f = resolve(ix.params.get(key)!.p, e);
  if (f.auto && auto?.[key] !== false) return 'auto';
  return key in vals ? vals[key]! : f.default;
}
const onEngine = (desk: Desk, e: EngineId): Desk => ({ ...desk, engine: e });
const words = (ix: Index, key: string, desk: Desk, v: Value | 'auto'): Slot => (v === 'auto' ? msg('value.auto') : formatMsg(ix, key, desk, v));
const stateSlots = (st: Json | undefined) => (Array.isArray(st) ? st : []);

/** What Revert would put back and Update would save: each setting the revision keeps that the desk now differs from,
 *  with its value now. */
export function revisionDiff(ix: Index, r: Revision, desk: Desk): { key: string; now: Slot }[] {
  if (desk.engine !== r.engine) return [];
  return revisionKeys(ix, r.engine, 'settings').filter(k => ix.params.get(k)!.p.type !== 'text')
    .flatMap(k => { const a = held(ix, k, r.engine, r.vals, r.auto), b = held(ix, k, r.engine, desk.vals, desk.auto); return JSON.stringify(a) === JSON.stringify(b) ? [] : [{ key: k, now: words(ix, k, desk, b) }]; });
}
/** The state a revision keeps (the adapters) differs from the desk's. */
export function revisionStateDiffers(ix: Index, r: Revision, desk: Desk): boolean {
  if (desk.engine !== r.engine) return false;
  return [...ix.states.values()].some(({ s, d }) => s.stored === 'settings' && d.engines.includes(r.engine) && JSON.stringify(r.state?.[s.key] ?? stateSlots(undefined)) !== JSON.stringify(desk.state[s.key] ?? stateSlots(undefined)));
}
/** What a revision holds as content (lyrics, song info, a description) that the desk now differs from: only what it
 *  holds; an empty text it holds replaces nothing, so it is no difference. */
export function revisionContentDiffers(ix: Index, r: Revision, desk: Desk): boolean {
  if (!r.withContent || desk.engine !== r.engine) return false;
  return revisionKeys(ix, r.engine, 'content').some(k => k in r.vals && r.vals[k] !== '' && JSON.stringify(r.vals[k]) !== JSON.stringify(k in desk.vals ? desk.vals[k] : resolve(ix.params.get(k)!.p, r.engine).default));
}
/** A revision is modified while what Revert would put back differs from the desk (F-04): its own settings, its
 *  adapters or the content it holds. A change to something it does not hold is no change to it. */
export const revisionModified = (ix: Index, r: Revision, desk: Desk): boolean =>
  revisionDiff(ix, r, desk).length > 0 || revisionStateDiffers(ix, r, desk) || revisionContentDiffers(ix, r, desk);

/** What a revision holds besides settings: lyrics, or the main box's text. */
export function revisionHolds(r: Revision): 'lyrics' | 'box' | null {
  if (!r.withContent) return null;
  const t = (k: string) => typeof r.vals[k] === 'string' && !!(r.vals[k] as string).trim();
  return t('lyrics.text') ? 'lyrics' : t('prompt.text') ? 'box' : null;
}
/** What the main box is called on an engine, inside a sentence ("the description", "the tags"). */
export const boxWord = (_ix: Index, desk: Desk, e: EngineId = desk.engine): Msg => msg('wf.box', { e });

/** A revision's details (the info button): its settings that are not at their defaults, by what owns them, in the
 *  definitions' order; a modified revision also lists what changed since, with its value now (C-36); its adapters and
 *  the content it holds. */
export interface DetailRow { label: Slot; value: Slot; now?: Slot; wide?: true }
export interface Detail { groups: { name: Slot; rows: DetailRow[] }[]; empty: Msg | null }
export function revisionDetail(ix: Index, r: Revision, desk: Desk, modified: boolean): Detail {
  const d0 = onEngine(desk, r.engine), diff = new Map(modified ? revisionDiff(ix, r, desk).map(x => [x.key, x.now]) : []);
  const groups: Detail['groups'] = [], post: DetailRow[] = [];
  /* the modules in the desk's order, then the pedals and side units together, each named by its pedal (C-36) */
  const mods = ix.defs.filter(d => d.kind === 'module').sort((a, b) => a.at - b.at), board = ix.defs.filter(d => d.kind !== 'module').sort((a, b) => (a.kind === b.kind ? a.at - b.at : a.kind === 'pedal' ? -1 : 1));
  for (const def of [...mods, ...board]) {
    const rows: DetailRow[] = [];
    /* an adapter slot's strength stands on the slot's own line, "LoRA · 0.80" (the parameter whose name is the slot's) */
    const slotOf = (k: string) => /\{file:([\w.-]+)\.(\d+)\|/.exec(def.params.find(p => p.key === k)?.undoName ?? '');
    for (const s of def.state ?? []) {
      if (s.stored !== 'settings' || !def.engines.includes(r.engine)) continue;
      stateSlots(r.state?.[s.key]).forEach((x, i) => {
        if (!isRec(x) || typeof x.file !== 'string') return;
        const k = def.params.find(p => { const m = slotOf(p.key); return !!m && m[1] === s.key && +m[2]! === i; })?.key;
        const v = k && has(ix, k, r.engine) ? held(ix, k, r.engine, r.vals, r.auto) : null, type = typeof x.type === 'string' ? x.type : '';
        rows.push({ label: x.file, value: v === null ? type : msg('rev.detail.slot', { type, v: words(ix, k!, d0, v) }), ...(k && diff.has(k) ? { now: diff.get(k)! } : {}) });
      });
    }
    for (const p of def.params) {
      const k = p.key;
      if (!has(ix, k, r.engine) || isSource(ix, k) || storedOf(ix, k) !== 'settings' || p.type === 'text') continue;
      if (slotOf(k) && stateSlots(r.state?.[slotOf(k)![1]!])[+slotOf(k)![2]!]) continue;
      const v = held(ix, k, r.engine, r.vals, r.auto), f = resolve(p, r.engine), dflt = f.auto ? 'auto' : f.default;
      if (JSON.stringify(v) === JSON.stringify(dflt) && !diff.has(k)) continue;
      const now = diff.get(k), label = paramWords(ix, k, d0).label;
      const own = def.kind === 'module' ? label : def.on === k ? defName(def, d0) : msg('rev.detail.of', { owner: defName(def, d0), label });
      (def.kind === 'module' ? rows : post).push({ label: own, value: words(ix, k, d0, v), ...(now !== undefined ? { now } : {}) });
    }
    if (rows.length) groups.push({ name: defName(def, d0), rows });
  }
  if (post.length) groups.push({ name: msg('rev.detail.post'), rows: post });
  if (r.withContent) {
    const rows: DetailRow[] = [], text = (k: string) => (typeof r.vals[k] === 'string' ? (r.vals[k] as string) : '');
    const box = text('prompt.text');
    if (box.trim()) rows.push({ label: '', value: msg('rev.detail.quote', { text: box.length > 80 ? box.slice(0, 80) + '…' : box }), wide: true });
    for (const k of revisionKeys(ix, r.engine, 'content')) {
      if (k === 'prompt.text' || !(k in r.vals) || k === 'lyrics.text' || ix.params.get(k)!.p.type === 'text') continue;
      const v = r.vals[k]!, f = resolve(ix.params.get(k)!.p, r.engine);
      if (JSON.stringify(v) === JSON.stringify(f.default)) continue;
      rows.push({ label: paramWords(ix, k, d0).label, value: words(ix, k, d0, v) });
    }
    if (text('lyrics.text').trim()) rows.push({ label: msg('rev.detail.lyrics'), value: msg('rev.detail.lines', { n: text('lyrics.text').split('\n').length }) });
    if (rows.length) groups.push({ name: msg('rev.detail.content'), rows });
  }
  return { groups, empty: groups.length ? null : msg('rev.detail.empty') };
}

/* ---------------------------------------------------------------- the list, as the Workflows panel draws it */

/** Which workflow and revision are open (the concept's S.workflow, S.revision, S.revFor). */
export interface Open { workflow: string | null; revision: string | null; revFor: EngineId | null }
export const START_FRESH = FACTORY_WORKFLOWS[0]!.name;

export interface RevRow {
  id: RevId; user: boolean;
  /** A revision of the user's under a factory workflow: marked "yours" at rest (F-23). */
  mine: boolean;
  on: boolean; modified: boolean;
  holds: 'lyrics' | 'box' | null;
  engine: EngineId; label: Msg; tip: Slot; holdsTip: Msg | null;
}
export interface WfRow {
  name: string; user: boolean; tier: Tier; tierWord: Msg;
  on: boolean; modified: boolean;
  revisions: RevRow[];
  label: Msg; tip: Msg;
}
export interface ListModel { yours: WfRow[]; factory: WfRow[]; empty: Msg | null }

const tierWord = (t: Tier): Msg => msg(`tier.${t}`);
export function listModel(ix: Index, saved: Saved, open: Open, desk: Desk, mods: readonly string[], board: readonly string[]): ListModel {
  const box = boxWord(ix, desk);
  const revRow = (wf: string, r: Revision, user: boolean): RevRow => {
    const id: RevId = { workflow: wf, engine: r.engine, name: r.name };
    const on = open.workflow === wf && open.revision === r.name && open.revFor === r.engine, modified = on && revisionModified(ix, r, desk);
    const mine = user && FACTORY_WORKFLOWS.some(w => w.name === wf), holds = revisionHolds(r), n = modified ? revisionDiff(ix, r, desk).length : 0;
    const tip: Slot = modified ? (n ? msg('rev.row.tip.changed', { n }) : revisionContentDiffers(ix, r, desk) ? msg('rev.row.tip.content') : msg('rev.row.tip.mod')) : r.note ?? msg('rev.row.tip.load', { wf });
    return { id, user, mine, on, modified, holds, engine: r.engine, tip,
      label: msg('rev.row.label', { name: r.name, mine: String(mine), engine: engineName(r.engine), holds: holds ?? 'none', box: boxWord(ix, desk, r.engine), mod: String(modified) }),
      holdsTip: holds === 'lyrics' ? msg('rev.row.holds.lyrics') : holds === 'box' ? msg('rev.row.holds.box', { box }) : null };
  };
  const wfRow = (w: Workflow, user: boolean): WfRow => {
    const on = open.workflow === w.name, modified = on && workflowModified(ix, w, mods, board);
    const revisions = revisionsOf(saved, w.name).map(x => revRow(w.name, x.r, x.user)), board0 = ix.defs.find(d => d.kind === 'module' && !!d.board)?.id;
    const what: Slot = w.note ?? msg('wf.row.modules', { n: w.modules.filter(m => m !== board0).length + w.board.length });
    return { name: w.name, user, tier: w.tier, tierWord: tierWord(w.tier), on, modified, revisions,
      label: msg('wf.row.label', { name: w.name, tier: tierWord(w.tier), n: revisions.length, mod: String(modified) }), tip: msg('wf.row.tip', { what, tier: tierWord(w.tier) }) };
  };
  const yours = saved.workflows.map(w => wfRow(w, true)), factory = FACTORY_WORKFLOWS.map(w => wfRow(w, false));
  const empty = yours.length ? null : msg(FACTORY_WORKFLOWS.some(w => saved.revisions.some(r => sameName(r.workflow, w.name))) ? 'wf.yours.empty.revs' : 'wf.yours.empty');
  return { yours, factory, empty };
}

/** The save row's hint: what a revision or a workflow saves, and, for a revision of a changed desk, that the new
 *  modules would not be kept. */
export function saveHint(ix: Index, as: 'rev' | 'wf', wf: string | null, desk: Desk, layoutChanged: boolean): { says: Msg; warn: boolean } {
  if (as === 'wf' || !wf) return { says: msg('wf.hint.wf'), warn: false };
  const repaint = ix.defs.some(d => d.kind === 'module' && !!d.source && desk.on.has(d.id));
  return { says: msg('wf.hint.rev', { engine: engineName(desk.engine), wf, repaint: repaint ? msg('wf.hint.repaint') : '', moved: layoutChanged ? msg('wf.hint.moved') : '' }), warn: layoutChanged };
}

/* ---------------------------------------------------------------- steps and what they say */

const recordPut = (what: Slot): Msg => msg('wf.put.record', { what });
export const openStep = (name: string, revert: boolean, put: Msg): StepWords => ({ label: msg(revert ? 'wf.step.revert' : 'wf.step.open', { name }), put });
export const loadStep = (name: string, revert: boolean, put: Msg): StepWords => ({ label: msg(revert ? 'wf.step.revert' : 'wf.step.load', { name }), put });
export const recordStep = (kind: 'replace' | 'update' | 'rename' | 'delete' | 'import' | 'save', name: string, what: Slot): StepWords =>
  ({ label: msg(`wf.step.${kind}`, { name }), put: recordPut(what) });
export const draftStep = (kind: 'draftSave' | 'draftDelete' | 'draftLoad', name: Slot = '', put?: Msg): StepWords =>
  ({ label: msg(`wf.step.${kind}`, { name }), put: put ?? recordPut(msg(kind === 'draftSave' ? 'wf.what.drafts' : 'wf.what.draft')) });
export const whatWorkflow = (name: string): Msg => msg('wf.what.workflow', { name });
export const whatRevision = (wf: string, name: string): Msg => msg('wf.what.revision', { wf, name });

/** What an opening or a load puts back, said before it is pressed: the modules, the pedals, the engine, the settings. */
export function putWords(ix: Index, before: Desk, after: Desk, mods: boolean, board: boolean): Msg {
  const items: Slot[] = [];
  if (mods) items.push(msg('path.mods'));
  if (board) items.push(msg('path.pedals'));
  if (before.engine !== after.engine) items.push(msg('path.engine'));
  const keys = new Set([...Object.keys(before.vals), ...Object.keys(after.vals), ...Object.keys(before.auto), ...Object.keys(after.auto)]);
  const n = [...keys].filter(k => JSON.stringify(before.vals[k]) !== JSON.stringify(after.vals[k]) || before.auto[k] !== after.auto[k]).length;
  if (n === 1) { const k = [...keys].find(x => JSON.stringify(before.vals[x]) !== JSON.stringify(after.vals[x]) || before.auto[x] !== after.auto[x])!; if (ix.params.has(k)) items.push(stepName(ix, k, after)); }
  else if (n > 1) items.push(msg('wf.what.settings', { n }));
  return msg('step.put.desk', { what: list(items.length ? items : [msg('path.mods')]) });
}

/** A load's toast: where it loaded, what came with it, and the adapters it unloaded (S-09). */
export function loadedWords(ix: Index, wf: string, r: Revision, desk: Desk, switched: boolean, unloaded: string[], revert: boolean): Msg {
  const h = revisionHolds(r), inc = h === 'lyrics' ? msg('wf.inc.lyrics') : h === 'box' ? msg('wf.inc.box', { box: boxWord(ix, desk, r.engine) }) : '';
  const un = unloaded.length ? msg('wf.unloaded', { files: list(unloaded) }) : '';
  return revert ? msg('wf.rev.reverted', { name: r.name, inc, un }) : msg('wf.loaded', { wf, name: r.name, on: switched ? msg('wf.loaded.on', { engine: engineName(r.engine) }) : '', inc, un });
}
/** The files of the desk's adapter slots that a load took off. */
export function unloadedFiles(ix: Index, before: Desk, after: Desk): string[] {
  const files = (d: Desk) => [...ix.states.values()].filter(({ s }) => s.kind === 'slots').flatMap(({ s }) => stateSlots(d.state[s.key]).flatMap(x => (isRec(x) && typeof x.file === 'string' ? [x.file] : [])));
  const now = files(after);
  return files(before).filter(f => !now.includes(f));
}

/** Loading a revision's content: a text it holds empty replaces nothing ("no saved lyrics or caption means yours stay"). */
export function contentKeptOn(r: Revision): string[] {
  return Object.keys(r.vals).filter(k => r.vals[k] === '');
}

/* ---------------------------------------------------------------- export and import (C-24, F-20) */

export const REVISION_FORMAT = 'hotstep-revision';
/** A revision as a file: its workflow's layout beside it, so it can be imported where that workflow is not. */
export function exportFile(saved: Saved, wf: string, r: Revision, tier: Tier): { name: string; text: string } {
  const w = findWorkflow(saved, wf)?.w ?? { v: SAVED_VERSION, name: wf, tier, modules: [], board: [] };
  const { note: _n, ...rec } = r; void _n;
  return { name: `${wf}-${r.name}`.replace(/[^\w.-]+/g, '_') + `.${engineShort(r.engine)}.json`, text: JSON.stringify({ format: REVISION_FORMAT, v: SAVED_VERSION, workflow: { v: w.v, name: w.name, tier: w.tier, modules: w.modules, board: w.board }, revision: rec }, null, 2) };
}

export type Imported =
  | { ok: false; says: Msg }
  | { ok: true; wf: string; newWf: Workflow | null; r: Revision; was: Revision | null; says: Msg };
/** A revision read from a file, cleaned before it is stored; what did not come through as it was is counted by kind.
 *  A workflow named in the file is the one here with that name whatever its capitals; one not here is made from the
 *  file; with none, the revision goes under the open workflow (or Start fresh). A factory revision's name is not taken
 *  over: the import is filed beside it as "<name> (2)". A user revision of the same name is replaced (Undo brings it
 *  back). */
export function importRevision(ix: Index, saved: Saved, file: unknown, openWf: string | null): Imported {
  const no: Imported = { ok: false, says: msg('wf.import.notRevision') };
  if (!isRec(file)) return no;
  const rv = isRec(file.revision) ? file.revision : file, w = isRec(file.workflow) ? file.workflow : null;
  const wfIn = w && typeof w.name === 'string' && normName(w.name) ? normName(w.name).slice(0, NAME_MAX).trim() : '';
  const c = cleanRevision(ix, { ...rv, name: typeof rv.name === 'string' ? normName(rv.name).slice(0, NAME_MAX).trim() : rv.name }, wfIn || openWf || START_FRESH);
  if (!c) return no;
  let wf = wfIn, newWf: Workflow | null = null, unknown = 0;
  const here = wf ? findWorkflow(saved, wf) : null;
  if (here) wf = here.w.name;
  else if (wf && w && Array.isArray(w.modules ?? w.mods)) {
    newWf = cleanWorkflow(ix, { ...w, name: wf });
    const l = (x: unknown) => (Array.isArray(x) ? x : []);
    unknown = [...l(w.modules ?? w.mods), ...l(w.board ?? w.pedals)].filter(x => typeof x !== 'string' || !ix.defs.some(d => d.id === (ix.formerIds.get(x) ?? x))).length;
    if (!newWf) wf = openWf ?? START_FRESH;
  } else wf = openWf ?? START_FRESH;
  const long = (x: unknown) => typeof x === 'string' && normName(x).length > NAME_MAX;
  const texts = [rv.vals, isRec(rv.content) ? rv.content.vals : null, isRec(rv.content) ? { p: rv.content.prompt } : null];
  const cut = long(rv.name) || long(w?.name) || texts.some(o => isRec(o) && Object.values(o).some(x => typeof x === 'string' && x.length > 20000));
  const r: Revision = { ...c.r, workflow: wf };
  const notes: Slot[] = [];
  if (c.dropped) notes.push(msg('wf.import.dropped', { n: c.dropped }));
  if (c.limited) notes.push(msg('wf.import.limited', { n: c.limited }));
  if (unknown) notes.push(msg('wf.import.unknown', { n: unknown }));
  if (cut) notes.push(msg('wf.import.cut'));
  if (FACTORY_REVISIONS.some(x => x.workflow === wf && x.engine === r.engine && sameName(x.name, r.name))) {
    const base = r.name.slice(0, NAME_MAX - 4);
    let i = 2; while (revNameClash(saved, wf, r.engine, `${base} (${i})`)) i++;
    r.name = `${base} (${i})`;
    notes.push(msg('wf.import.beside', { name: r.name }));
  }
  const was = saved.revisions.find(x => sameName(x.workflow, wf) && x.engine === r.engine && sameName(x.name, r.name)) ?? null;
  if (was) r.name = was.name;
  return { ok: true, wf, newWf, r, was, says: msg('wf.imported', { replaced: String(!!was), wf, name: r.name, engine: engineName(r.engine), notes: notes.length ? msg('wf.imported.notes', { notes: list(notes, 'semicolon') }) : '' }) };
}

/* ---------------------------------------------------------------- drafts (35-drafts.js)
   A draft is the form's content for one engine; any engine can name one. Stored drafts are read one by one (C-08) and
   merged with what another tab saved before each write (C-09); the newest thirty are kept. */

export interface StoredDraft extends Draft { id: string; at: number }
/** A draft as it can be kept, or null: the page's shape, or the concept's ({ backend, at, prompt, vals }). */
export function cleanDraft(ix: Index, raw: unknown): StoredDraft | null {
  if (!isRec(raw) || typeof raw.at !== 'number' || !Number.isFinite(raw.at)) return null;
  const v0 = !('v' in raw);
  const r = v0 ? { ...raw, vals: { ...(isRec(raw.vals) ? raw.vals : {}), ...(typeof raw.prompt === 'string' ? { 'prompt.text': raw.prompt } : {}) } } : raw;
  const m = migrate(r);
  if (!m || m.kind !== 'draft') return null;
  const x = m.record as unknown as Record<string, unknown>;
  if (!ENGINE_IDS.includes(x.engine as EngineId)) return null;
  const c = cleanValues(ix, x.vals, x.auto, RETIRED.keys);
  return { v: SAVED_VERSION, id: typeof raw.id === 'string' ? raw.id.slice(0, 40) : `${raw.at}`, name: typeof x.name === 'string' ? x.name.slice(0, DRAFT_NAME_MAX) : '', engine: x.engine as EngineId, at: raw.at, vals: c.vals, ...(Object.keys(c.auto).length ? { auto: c.auto } : {}) };
}
/** A stored drafts list, newest first and at most thirty: what can be kept, and how many could not (null: no list). */
export function readDrafts(ix: Index, raw: unknown): { drafts: StoredDraft[]; bad: number } | null {
  if (!Array.isArray(raw)) return null;
  const out: StoredDraft[] = []; let bad = 0;
  for (const x of raw) { const d = cleanDraft(ix, x); if (!d || out.some(y => y.id === d.id)) bad++; else out.push(d); }
  return { drafts: newestDrafts(out), bad };
}
export const newestDrafts = (l: StoredDraft[]): StoredDraft[] => [...l].sort((a, b) => b.at - a.at).slice(0, DRAFT_MAX);
export const mergeDrafts = (base: StoredDraft[], ours: StoredDraft[], theirs: StoredDraft[]): StoredDraft[] => newestDrafts(merge3(base, ours, theirs, d => d.id));

/** Two drafts hold the same thing: the engine and every value. */
export const sameDraft = (a: Pick<Draft, 'engine' | 'vals'> | undefined, b: Pick<Draft, 'engine' | 'vals'>): boolean =>
  !!a && a.engine === b.engine && canon(a.vals) === canon(b.vals);

const textIn = (d: Pick<Draft, 'vals'>, k: string) => (typeof d.vals[k] === 'string' ? (d.vals[k] as string) : '');
/** A draft's name in the list: its own, or the start of its main box, or Untitled. */
export const draftTitle = (d: Pick<Draft, 'name' | 'vals'>): Slot => d.name || textIn(d, 'prompt.text').slice(0, 60) || msg('draft.untitled');
/** The second line tells drafts apart: when, the engine if it is not this one, and the first lyric line or the length. */
export function draftSub(d: StoredDraft, engine: EngineId, now: number, lang = 'en'): Msg {
  const ly = textIn(d, 'lyrics.text').split('\n').map(x => x.trim()).find(x => x && !/^\[.*\]$/.test(x));
  const n = textIn(d, 'prompt.text').trim().split(/\s+/).filter(Boolean).length;
  const what = ly ? msg('draft.line', { line: ly }) : n ? msg('draft.words', { n }) : msg('draft.empty');
  return msg('draft.sub', { when: draftWhen(d.at, now, lang), other: String(d.engine !== engine), engine: engineName(d.engine), what });
}
/** When a draft was saved: the time today, Yesterday and the time, then the date (F-18). */
export function draftWhen(t: number, now: number, lang = 'en'): Slot {
  const d = new Date(t), n = new Date(now), day = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime(), dd = Math.round((day(n) - day(d)) / 864e5);
  const tm = d.toLocaleTimeString(lang, { hour: 'numeric', minute: '2-digit' });
  return dd === 0 ? tm : dd === 1 ? msg('draft.yesterday', { time: tm }) : d.toLocaleDateString(lang, { month: 'short', day: 'numeric', ...(d.getFullYear() !== n.getFullYear() ? { year: 'numeric' as const } : {}) });
}
/** A draft's content on a desk: every content value of the draft's engine reset, then the draft's applied, cleaned. */
export function applyDraft(ix: Index, desk: Desk, d: Draft): Desk {
  const out: Desk = { ...desk, vals: { ...desk.vals }, auto: { ...desk.auto } };
  for (const k of revisionKeys(ix, d.engine, 'content')) { delete out.vals[k]; delete out.auto[k]; }
  const c = cleanValues(ix, d.vals, d.auto, RETIRED.keys);
  Object.assign(out.vals, c.vals); Object.assign(out.auto, c.auto);
  return out;
}
/** The words over the Drafts list: what a draft saves on this engine. */
export function draftsAbout(ix: Index, desk: Desk): { says: Msg; tip: Msg } {
  const lyrics = has(ix, 'lyrics.text', desk.engine), box = msg('wf.box.cap', { e: desk.engine });
  return { says: msg('draft.about', { box, lyrics: String(lyrics), n: DRAFT_MAX }), tip: msg('draft.about.title', { box: boxWord(ix, desk), lyrics: String(lyrics) }) };
}
export const draftLoaded = (d: Draft, switched: boolean): Msg =>
  msg('draft.loaded', { named: String(!!d.name), name: d.name, on: switched ? msg('wf.loaded.on', { engine: engineName(d.engine) }) : '' });

/* ---------------------------------------------------------------- storage words (F-15, F-16) */

export type StoreWhy = 'full' | 'blocked';
export const storeWhy = (fails: readonly StoreWhy[]): Msg => msg(fails.includes('full') ? 'store.why.full' : 'store.why.blocked');
/** What the page says when a write it was asked for failed: the desk's own autosave says "your desk". */
export const storeFailed = (fails: readonly StoreWhy[], desk: boolean): Msg => msg('store.failed', { why: storeWhy(fails), what: desk ? 'desk' : 'other' });
/** The read-out before Undo: Saved only once every store took its last write (F-15). */
export function saveState(fails: readonly StoreWhy[]): { text: Msg; tip: Msg; failed: boolean } {
  if (!fails.length) return { text: msg('store.state.saved'), tip: msg('store.state.tip'), failed: false };
  return { text: msg('store.state.failed'), tip: msg('store.state.tip.failed', { why: storeWhy(fails), full: String(fails.includes('full')) }), failed: true };
}
export const storeSkipped = (n: number, what: 'wf' | 'draft'): Msg => msg('store.skipped', { n, one: msg(`store.skipped.${what}.one`), many: msg(`store.skipped.${what}.many`) });
export const storeUnreadable = (what: 'wf' | 'drafts'): Msg => msg('store.unreadable', { what: msg(`store.what.${what}`) });
export const storeTold = (items: Msg[]): Msg => msg('store.told', { items: list(items, 'sentences') });
