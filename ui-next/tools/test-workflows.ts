// Tests of the core's rules for saved work (package 7, src/derive/workflows.ts and records.ts): each against what the
// concept does (32-workflows.js, 35-drafts.js, 13-stored.js) and the findings it fixed (C-02, C-08, C-09, C-24, F-02,
// F-04, F-10, F-15, F-17, F-20, F-23).
//
//   node tools/test-workflows.ts        exit 1 on a failure

import { DEFINITIONS } from '../src/definitions/index.ts';
import { FACTORY_REVISIONS, FACTORY_WORKFLOWS } from '../src/definitions/factory.ts';
import type { EngineId, Revision, Tier } from '../src/definitions/types.ts';
import {
  applyDraft, cleanDraft, draftOf, draftSub, draftTitle, emptySaved, exportFile, findRevision, findWorkflow, importRevision, indexOf, layoutOf, listModel, loadRevision, merge3, mergeSaved,
  normName, putRevision, putWorkflow, readDrafts, readSaved, renameWorkflow, revisionDetail, revisionModified, revisionOf, revNameClash, sameDraft, sameName, saveCheck, saveState, storeSkipped,
  storeTold, wfNameClash, workflowModified, workflowOf, dropWorkflow, openStep, putWords, type Desk, type Msg, type Saved,
} from '../src/derive.ts';
import { say } from '../src/derive/messages.ts';

const ix = indexOf(DEFINITIONS);
const J = (x: unknown) => JSON.stringify(x);
const fails: string[] = [];
let n = 0;
const ok = (what: string, cond: boolean, detail: unknown = '') => { n++; if (!cond) fails.push(`${what}${detail === '' ? '' : ': ' + (typeof detail === 'string' ? detail : J(detail))}`); };
const s = (m: unknown) => say(m as Msg);
const desk = (engine: EngineId, tier: Tier, on: string[], vals: Desk['vals'] = {}, auto: Desk['auto'] = {}, state: Desk['state'] = {}): Desk =>
  ({ engine, tier, vals, auto, state, on: new Set([...DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id), ...on]) });
const saved = (workflows: Saved['workflows'] = [], revisions: Saved['revisions'] = []): Saved => ({ workflows, revisions });
const wf = (name: string, modules: string[] = ['tempo'], board: string[] = [], tier: Tier = 'studio') => ({ v: 1 as const, name, tier, modules, board });

/* ---- names (F-02, F-10) */
ok('names are taken normalised: trimmed, a run of spaces as one, NFC', normName('  My \t polish ') === 'My polish' && normName('Café') === 'Café');
ok('and compared whatever their capitals', sameName('My  polish ', 'my polish') && !sameName('My polish', 'My polis'));
{
  const sv = saved([wf('abc'), wf('xyz')]);
  ok('a name that differs only in capitals is another workflow\'s', s(wfNameClash(sv, 'ABC', 'xyz')) === 'The "abc" workflow already exists. Pick another name.');
  ok('a factory name in other capitals is refused', s(wfNameClash(sv, ' songwriter ', 'xyz')) === 'That name belongs to a factory workflow. Pick another.');
  ok('a workflow may change its own capitals', wfNameClash(sv, 'ABC', 'abc') === null);
  ok('an exact match is found first', findWorkflow(saved([wf('songwriter')]), 'songwriter')!.user && !findWorkflow(saved([wf('songwriter')]), 'Songwriter')!.user);
  const r: Revision = { v: 1, workflow: 'Mine', name: 'My polish', engine: 'ace', vals: {} };
  const c = saveCheck(saved([wf('Mine')], [r]), 'rev', 'Mine', 'ace', 'my  POLISH');
  ok('Save asks to replace a revision of the same name, and keeps its casing (C-02)', 'replace' in c && c.replace === 'My polish' && /already exists under Mine/.test(s(c.says)), c);
  ok('the same name on another engine is another revision', 'ok' in saveCheck(saved([wf('Mine')], [r]), 'rev', 'Mine', 'sa3', 'My polish'));
  ok('a factory revision\'s name is refused under its workflow', 'refuse' in saveCheck(emptySaved(), 'rev', 'Deep sampler', 'ace', 'polishing'));
  ok('an empty name is refused', 'refuse' in saveCheck(emptySaved(), 'wf', null, 'ace', '   ') && s((saveCheck(emptySaved(), 'wf', null, 'ace', '') as { refuse: Msg }).refuse) === 'Give it a name first.');
  ok('a factory workflow\'s name is refused for a new workflow', 'refuse' in saveCheck(emptySaved(), 'wf', null, 'ace', 'SONGWRITER'));
  ok('a revision name in use under its workflow, for a rename', s(revNameClash(saved([wf('Mine')], [r, { ...r, name: 'Other' }]), 'Mine', 'ace', 'other', 'My polish')) === '"Other" already exists under Mine. Pick another name.');
}

/* ---- reading what was stored (C-08) */
{
  const concept = { workflows: [{ name: 'nopedals', mods: ['tempo'], tier: 'zzz' }, { name: 'Good', tier: 'lab', mods: ['seed', 'nope'], pedals: ['norm'] }, { mods: [] }, 7, null], revs: { Good: [{ name: 'r', backend: 'ace', vals: { 'samp.steps': 9 } }, { name: 'bad', backend: 'zzz', vals: {} }], Other: 'notalist' } };
  const r = readSaved(ix, concept)!;
  ok('the concept\'s stored list is read record by record, what can be repaired is (C-08)', J(r.saved.workflows.map(w => [w.name, w.tier, w.modules.join(), w.board.join()])) === J([['nopedals', 'studio', 'tempo', ''], ['Good', 'lab', 'seed', 'norm']]), r.saved.workflows);
  ok('revisions are checked one by one, five records skipped', J(r.saved.revisions.map(x => [x.workflow, x.name, x.engine, x.vals['samp.steps']])) === J([['Good', 'r', 'ace', 9]]) && r.bad === 5, r);
  ok('the page\'s own list reads back as written', J(readSaved(ix, { v: 1, workflows: r.saved.workflows, revisions: r.saved.revisions })!.saved) === J(r.saved));
  ok('a list whose workflows are no list reads empty', J(readSaved(ix, { workflows: 5 })) === J({ saved: emptySaved(), bad: 0 }));
  ok('what is no list is unreadable', readSaved(ix, 'x') === null && readSaved(ix, [1]) === null);
  ok('two records under one name: the second is skipped', readSaved(ix, { workflows: [wf('a'), wf('A ')] })!.bad === 1);
  ok('a workflow named "__proto__" is a workflow (F-20)', readSaved(ix, { workflows: [wf('__proto__')], revisions: [{ v: 1, workflow: '__proto__', name: 'odd', engine: 'ace', vals: {} }] })!.saved.revisions.length === 1);
  const d = readDrafts(ix, [{ backend: 'zzz' }, { name: 'ok', backend: 'ace', at: 1, prompt: 'p', vals: {} }, 'x'])!;
  ok('drafts are read one by one: the concept\'s draft keeps its prompt', d.drafts.length === 1 && d.bad === 2 && d.drafts[0]!.vals['prompt.text'] === 'p' && d.drafts[0]!.engine === 'ace', d);
  ok('drafts that are no list are unreadable', readDrafts(ix, { a: 1 }) === null);
  ok('what boot skipped is said in the page\'s words', s(storeTold([storeSkipped(5, 'wf'), storeSkipped(2, 'draft')])) === "5 saved workflows or revisions couldn't be read and were skipped. A copy is kept in this browser; nothing was deleted. 2 drafts couldn't be read and were skipped. A copy is kept in this browser; nothing was deleted.");
}

/* ---- two tabs (C-09) */
{
  const a = wf('A'), b = wf('B'), c = wf('C'), key = (w: { name: string }) => w.name;
  ok('a record the other tab added is kept, first', J(merge3([a], [a, b], [c, a], key).map(key)) === J(['C', 'A', 'B']));
  ok('a record the other tab deleted is gone, ours kept', J(merge3([a, b], [a, b, c], [a], key).map(key)) === J(['A', 'C']));
  ok('a record both changed takes ours', merge3([a], [{ ...a, tier: 'lab' as Tier }], [{ ...a, tier: 'quick' as Tier }], key)[0]!.tier === 'lab');
  ok('records compare by content, whatever order their fields were written in', merge3([a], [{ ...a }], [{ board: [], modules: ['tempo'], tier: 'studio' as Tier, name: 'A', v: 1 as const }], key).length === 1);
  const m = mergeSaved(saved([a]), saved([a]), saved([a, wf('a ')]));
  ok('names merge by the one name rule', m.workflows.length === 1, m);
}

/* ---- a workflow on the desk */
{
  const sw = FACTORY_WORKFLOWS.find(w => w.name === 'Songwriter')!;
  ok('a board with pedals brings the Pedalboard', J(layoutOf(ix, sw)) === J({ mods: ['tempo', 'avoid', 'takes', 'post'], board: ['lrc'] }), layoutOf(ix, sw));
  ok('without the Pedalboard there is no board', J(layoutOf(ix, { modules: ['tempo'], board: [] })) === J({ mods: ['tempo'], board: [] }));
  ok('the open workflow is not modified with its own layout', !workflowModified(ix, sw, ['tempo', 'avoid', 'takes', 'post'], ['lrc']));
  ok('a module added marks it (F-04)', workflowModified(ix, sw, ['tempo', 'avoid', 'takes', 'post', 'seed'], ['lrc']));
  ok('and a pedal taken off', workflowModified(ix, sw, ['tempo', 'avoid', 'takes', 'post'], []));
  const w = workflowOf(ix, desk('ace', 'studio', ['tempo', 'seed', 'repaint']), 'Mine');
  ok('a workflow keeps no source module', J(w.modules) === J(['tempo', 'seed']) && w.tier === 'studio');
}

/* ---- a revision against the desk (F-04, E-03, S-09) */
{
  const pol = FACTORY_REVISIONS.find(r => r.name === 'Polishing')!, d0 = desk('ace', 'lab', ['sampler', 'planner']);
  const loaded = loadRevision(ix, d0, pol).desk;
  ok('a revision just loaded is not modified', !revisionModified(ix, pol, loaded));
  ok('a setting it keeps, changed, marks it', revisionModified(ix, pol, { ...loaded, vals: { ...loaded.vals, 'samp.steps': 22 } }));
  ok('a change to content it does not hold does not (F-04)', !revisionModified(ix, pol, { ...loaded, vals: { ...loaded.vals, 'lyrics.text': 'la' } }));
  ok('on another engine it is not modified', !revisionModified(ix, pol, { ...loaded, engine: 'sa3' }));
  const mine = revisionOf(ix, desk('ace', 'studio', ['seed', 'sampler', 'takes'], { 'samp.steps': 31, 'takes.n': 2 }), 'Seed hunter', 'Night run');
  const det = revisionDetail(ix, mine, d0, false);
  ok('its details list the settings that are not at their defaults, in order', J(det.groups.flatMap(g => g.rows.map(r => s(r.label) + s(r.value)))) === J(['Steps31', 'Takes per run2']), det.groups.flatMap(g => g.rows.map(r => s(r.label) + s(r.value))));
  const mod = revisionDetail(ix, mine, { ...loadRevision(ix, d0, mine).desk, vals: { ...loadRevision(ix, d0, mine).desk.vals, 'samp.steps': 40 } }, true);
  ok('a modified one says what changed, with its value now (C-36)', mod.groups.flatMap(g => g.rows).some(r => s(r.label) === 'Steps' && s(r.value) === '31' && s(r.now!) === '40'));
  ok('one at every default says so', s(revisionDetail(ix, { v: 1, workflow: 'x', name: 'y', engine: 'ace', vals: {} }, d0, false).empty) === 'Every setting at its default.');
  const guard = revisionOf(ix, desk('yue2', 'lab', ['repetition'], { 'yc.rep': 1.3 }), 'Start fresh', 'guarded');
  ok('a revision holds the Repetition guard (E-03)', 'yc.rep' in guard.vals);
  const withLyrics = revisionOf(ix, desk('ace', 'studio', [], { 'lyrics.text': '' , 'prompt.text': 'folk' }), 'Start fresh', 'c', true);
  const ld = loadRevision(ix, desk('ace', 'studio', [], { 'lyrics.text': '[Verse]\nmine' }), withLyrics).desk;
  ok('content held as empty text replaces nothing: your lyrics stay', ld.vals['lyrics.text'] === '[Verse]\nmine' && ld.vals['prompt.text'] === 'folk', ld.vals);
  const slots = loadRevision(ix, desk('heartmula', 'studio', ['adapters'], {}, {}, { slots: [{ file: 'heart-folk-lm', type: 'LoRA' }] }), FACTORY_REVISIONS.find(r => r.name === 'Reference')!).desk;
  ok('a revision that holds no adapters unloads them (S-09)', !slots.state.slots || J(slots.state.slots) === '[]', slots.state);
}

/* ---- the list (F-23) */
{
  const sv = saved([], [{ v: 1, workflow: 'Deep sampler', name: 'Polishing mine', engine: 'ace', vals: {} }]);
  const m = listModel(ix, sv, { workflow: 'Start fresh', revision: null, revFor: null }, desk('ace', 'quick', []), [], []);
  const row = m.factory.find(w => w.name === 'Deep sampler')!.revisions[0]!;
  ok('a revision you saved under a factory workflow is yours, first among its revisions', row.user && row.mine && /, yours,/.test(s(row.label)), s(row.label));
  ok('the empty Yours group says where they are', /marked “yours”/.test(s(m.empty)));
  ok('the open workflow is the open row', m.factory.find(w => w.on)!.name === 'Start fresh' && /2 revisions/.test(s(m.factory[0]!.label)));
}

/* ---- import and export (C-24, F-20) */
{
  const file = { format: 'hotstep-revision', version: 1, workflow: { name: 'Imported wf', tier: 'lab', mods: ['sampler', 'nope'], pedals: ['norm', 'zz'] }, revision: { name: 'From file', backend: 'yue', vals: { 'samp.steps': 40, 'samp.cfg': 'x', 'bogus.key': 1 }, slots: [{ file: 'a', type: 'LoRA' }, 7] } };
  const r = importRevision(ix, emptySaved(), file, null);
  ok('a revision file imports into its workflow, cleaned', r.ok && r.newWf!.modules.join() === 'sampler' && r.newWf!.board.join() === 'norm' && J(r.r.vals) === '{"samp.steps":40}' && (r.r.state?.slots as unknown[]).length === 1 && r.r.engine === 'yue2', r);
  ok('and says what was left out, by kind', r.ok && s(r.says) === "Imported Imported wf › From file (YuE2). 3 values didn't fit their controls and were left out; 2 modules aren't known here", r.ok ? s(r.says) : r);
  const again = importRevision(ix, saved([r.ok ? r.newWf! : wf('x')], [r.ok ? r.r : (null as never)]), file, null);
  ok('the same again replaces it', again.ok && !!again.was && /^Replaced/.test(s(again.says)));
  const fac = importRevision(ix, emptySaved(), { workflow: { name: 'Songwriter' }, revision: { name: 'pop 3 TAKES', backend: 'ace', vals: {} } }, null);
  ok('an import does not take a factory revision\'s name', fac.ok && fac.r.name === 'pop 3 TAKES (2)' && fac.wf === 'Songwriter', fac);
  const caps = importRevision(ix, emptySaved(), { workflow: { name: 'SONGWRITER', tier: 'quick', mods: ['tempo'], pedals: [] }, revision: { name: 'From file', backend: 'ace', vals: {} } }, null);
  ok('a workflow named in the file is the one here whatever its capitals (F-02)', caps.ok && caps.wf === 'Songwriter' && !caps.newWf);
  ok('JSON that is not a revision is refused', !importRevision(ix, emptySaved(), { a: 1 }, null).ok);
  const fr = FACTORY_REVISIONS.find(x => x.name === 'Faithful render')!, ex = exportFile(emptySaved(), 'Deep sampler', fr, 'lab'), j = JSON.parse(ex.text);
  ok('export writes the revision with its workflow, named for both and the engine', ex.name === 'Deep_sampler-Faithful_render.mm3.json' && j.format === 'hotstep-revision' && j.revision.vals['samp.steps'] === 30 && j.workflow.name === 'Deep sampler', ex.name);
  const back = importRevision(ix, emptySaved(), j, null);
  ok('and an exported revision imports as it was', back.ok && J(back.r.vals) === J(fr.vals) && back.r.engine === fr.engine && back.wf === 'Deep sampler' && back.r.name === 'Faithful render (2)', back.ok ? back.r : back);
}

/* ---- the lists as operations (F-02) */
{
  let sv = saved([wf('Alpha')], [{ v: 1, workflow: 'Alpha', name: 'r1', engine: 'ace', vals: {} }]);
  sv = renameWorkflow(sv, 'Alpha', 'Gamma');
  ok('a rename takes the workflow\'s revisions with it', sv.workflows[0]!.name === 'Gamma' && sv.revisions[0]!.workflow === 'Gamma');
  ok('a revision is found by its identity', !!findRevision(sv, { workflow: 'gamma', engine: 'ace', name: 'R1' }));
  ok('putting a workflow of the same name replaces it in place', putWorkflow(saved([wf('a'), wf('b')]), wf('A', ['seed'])).workflows.map(w => w.modules.join()).join('|') === 'seed|tempo');
  ok('a revision put back goes where it stood among its workflow\'s', putRevision(saved([], [{ v: 1, workflow: 'A', name: 'x', engine: 'ace', vals: {} }, { v: 1, workflow: 'A', name: 'z', engine: 'ace', vals: {} }]), { v: 1, workflow: 'A', name: 'y', engine: 'ace', vals: {} }, 1).revisions.map(r => r.name).join() === 'x,y,z');
  ok('deleting a workflow deletes its revisions', dropWorkflow(sv, 'GAMMA').revisions.length === 0);
}

/* ---- steps and words */
{
  const d = desk('ace', 'quick', []), put = putWords(ix, d, d, true, true);
  ok('an opening is named for the workflow and says it puts back the modules and pedals', s(openStep('Songwriter', false, put).label) === 'Open "Songwriter"' && s(put) === 'Puts back the modules and the pedals. Anything else you did since stays as it is', s(put));
  ok('the read-out says Saved, and Not saved while a store refused (F-15)', s(saveState([]).text) === 'Saved' && s(saveState(['full']).text) === 'Not saved' && /storage is full/.test(s(saveState(['full']).tip)));
}

/* ---- drafts (35-drafts.js) */
{
  const d = desk('ace', 'studio', [], { 'prompt.text': 'first draft', 'lyrics.text': '[Verse]\nhello there' });
  const dr = { ...draftOf(ix, d, ''), id: 'a', at: Date.parse('2026-11-23T10:00:00') };
  ok('a draft holds the content', dr.vals['prompt.text'] === 'first draft' && dr.vals['lyrics.text'] === '[Verse]\nhello there');
  ok('the same content is the same draft', sameDraft(dr, draftOf(ix, d, 'x')) && !sameDraft(dr, draftOf(ix, { ...d, engine: 'yue2' }, '')));
  ok('its second line: when, and its first lyric line', /^.+ · “hello there”$/.test(s(draftSub(ix, dr, 'ace', Date.parse('2026-11-23T12:00:00')))), s(draftSub(ix, dr, 'ace', Date.parse('2026-11-23T12:00:00'))));
  ok('another engine\'s draft says the engine', /Yesterday .+ · ACE-Step 1\.5 · /.test(s(draftSub(ix, dr, 'yue2', Date.parse('2026-11-24T12:00:00')))));
  ok('a draft is named by its name, else its words, else Untitled', draftTitle(ix, dr) === 'first draft' && s(draftTitle(ix, { name: '', vals: {} })) === 'Untitled');
  const on = applyDraft(ix, desk('ace', 'studio', [], { 'prompt.text': 'other', 'song.title': 'T' }), dr);
  ok('loading a draft replaces the content: what it does not hold goes back to its default', on.vals['prompt.text'] === 'first draft' && !('song.title' in on.vals), on.vals);
  ok('a draft of a shape the page cannot read is skipped', cleanDraft(ix, { v: 9, at: 1, engine: 'ace', vals: {} }) === null);
}

if (fails.length) { console.log(`workflows: ${fails.length} of ${n} failed`); fails.forEach(f => console.log('  ' + f)); process.exit(1); }
console.log(`workflows: ${n} checks of the core's rules for saved work: ok`);
