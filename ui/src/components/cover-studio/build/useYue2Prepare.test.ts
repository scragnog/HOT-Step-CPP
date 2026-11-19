import assert from 'node:assert/strict';
import test from 'node:test';
import { parseHTML } from 'linkedom';
import { act, createElement, StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { structureHash } from '@/utils/lyrics/yue2Prepare';
import { prepareV2Problem } from '../../../../../server/src/contracts/yue2Prepare';
import { createYue2Prepare, PREPARE_DRAFT_KEY, useYue2Prepare, type PrepareInitial } from './useYue2Prepare';

const abc = ['X:1', 'M:4/4', 'L:1/8', 'K:C',
  '% verse', 'V:Vocal', 'C8|D8|', 'V:Ins', 'C8|D8|',
  '% chorus', 'V:Vocal', 'E8|F8|', 'V:Ins', 'E8|F8|',
  '% chorus', 'V:Vocal', 'G8|A8|', 'V:Ins', 'G8|A8|'].join('\n');
const lyrics = '[verse]\nfirst line\nsecond line\n[chorus]\nchorus words';
const initial = { sourceKey: 'recording-a', abc, lyrics };
const make = () => createYue2Prepare(initial);
function assigned() {
  const c = make();
  assert.ok(c.confirmStructure());
  const p = c.getSnapshot().doc;
  assert.ok(c.applySuggestions(c.captureGuard(), { [p.sectionIds[0]]: [p.blocks[0].id],
    [p.sectionIds[1]]: [p.blocks[1].id], [p.sectionIds[2]]: [p.blocks[1].id] }));
  return c;
}

test('step navigation keeps structure locked; one undo history restores the confirmed pair', () => {
  const c = assigned();
  assert.equal(c.getSnapshot().step, 'lyrics');
  assert.ok(c.confirmLyrics());
  const ready = c.getSnapshot().doc;
  assert.equal(c.getSnapshot().step, 'ready');
  assert.ok(c.showStep('structure'));
  assert.equal(c.applyStructuralOp({ kind: 'relabel', index: 0, label: 'intro' }), false);
  assert.equal(c.getSnapshot().doc, ready);
  assert.ok(c.beginStructureEdit());
  assert.ok(c.applyStructuralOp({ kind: 'relabel', index: 0, label: 'intro' }));
  const edited = c.getSnapshot().doc;
  assert.equal(edited.assignments[edited.sectionIds[0]], undefined);
  assert.deepEqual(edited.flagged, [edited.sectionIds[0]]);
  assert.deepEqual(edited.assignments[edited.sectionIds[1]], ready.assignments[ready.sectionIds[1]]);
  assert.deepEqual(edited.blocks, ready.blocks);
  assert.ok(c.undo()); assert.ok(c.undo());
  assert.deepEqual(c.getSnapshot().doc, ready);
  assert.equal(c.getSnapshot().editingStructure, false);
  assert.ok(c.redo()); assert.ok(c.redo());
  assert.deepEqual(c.getSnapshot().doc, edited);
  assert.ok(c.undo());
  assert.ok(c.applyStructuralOp({ kind: 'replace', abc: abc.replace('% verse', '% bridge') }));
  assert.equal(c.canRedo, false, 'a new operation drops the old redo branch');
});

test('re-paste retains unique exact text, leaves duplicates unassigned and undoes as one operation', () => {
  const c = assigned();
  const before = c.getSnapshot().doc;
  const guard = c.captureGuard();
  assert.ok(c.applyLyricOp({ kind: 'paste', paste: lyrics.replace('chorus words', 'new chorus words') }));
  const after = c.getSnapshot().doc;
  assert.equal(structureHash(after.abc), guard.structureHash);
  assert.deepEqual(after.confirmed.structure, before.confirmed.structure);
  assert.deepEqual(after.assignments[after.sectionIds[0]], before.assignments[before.sectionIds[0]]);
  assert.equal(after.assignments[after.sectionIds[1]], undefined);
  assert.equal(c.applySuggestions(guard, { [after.sectionIds[1]]: [before.blocks[1].id] }), false);
  assert.equal(c.getSnapshot().doc, after);
  assert.ok(c.undo());
  assert.deepEqual(c.getSnapshot().doc, before);
  // Accepted reuse made two identical blocks. A later paste cannot guess
  // which occurrence to retain, even though its text stayed unchanged.
  assert.ok(c.applyLyricOp({ kind: 'paste', paste: lyrics }));
  assert.equal(c.getSnapshot().doc.assignments[before.sectionIds[1]], undefined);
  assert.equal(c.getSnapshot().doc.assignments[before.sectionIds[2]], undefined);
});

test('all three captured guard fields reject stale suggestions without changing the document', () => {
  const c = assigned();
  const p = c.getSnapshot().doc;
  const guard = c.captureGuard();
  for (const changed of [{ sourceKey: 'recording-b' }, { structureHash: 'old' }, { lyricRevision: guard.lyricRevision + 1 }]) {
    assert.equal(c.applySuggestions({ ...guard, ...changed }, { [p.sectionIds[0]]: 'instrumental' }), false);
    assert.equal(c.getSnapshot().doc, p);
  }
  assert.ok(c.beginStructureEdit());
  const structureGuard = c.captureGuard();
  assert.ok(c.applyStructuralOp({ kind: 'relabel', index: 0, label: 'intro' }));
  assert.equal(c.applyStructuralOp({ kind: 'replace', abc }, structureGuard), false);
  assert.equal(c.matchesGuard(guard), false);
});

test('suggestions are atomic, ordered and repeat words as independent copies', () => {
  const c = make(); c.confirmStructure();
  const p = c.getSnapshot().doc;
  const guard = c.captureGuard();
  assert.equal(c.applySuggestions(guard, { [p.sectionIds[0]]: [p.blocks[0].id], missing: 'instrumental' }), false);
  assert.equal(c.getSnapshot().doc, p);
  assert.ok(c.applySuggestions(guard, { [p.sectionIds[0]]: [p.blocks[0].id, p.blocks[1].id],
    [p.sectionIds[1]]: [p.blocks[0].id, p.blocks[1].id], [p.sectionIds[2]]: 'instrumental' }));
  const next = c.getSnapshot().doc;
  assert.equal(prepareV2Problem(next), null);
  assert.equal(next.blocks.length, 4);
  const row = next.assignments[p.sectionIds[1]];
  assert.equal(row.kind, 'words');
  if (row.kind !== 'words') assert.fail();
  assert.deepEqual(row.blockIds.map(id => next.blocks.find(b => b.id === id)?.text), ['first line\nsecond line', 'chorus words']);
  assert.ok(c.confirmLyrics());
  const copyId = row.blockIds[0];
  assert.ok(c.applyLyricOp({ kind: 'text', blockId: copyId, text: 'independent words' }));
  assert.equal(c.getSnapshot().doc.blocks[0].text, p.blocks[0].text);
  assert.ok(c.undo()); assert.ok(c.undo()); assert.ok(c.undo());
  assert.deepEqual(c.getSnapshot().doc, p);
  assert.ok(c.redo());
  assert.deepEqual(c.getSnapshot().doc, next);
});

test('merge and boundary moves invalidate affected rows; renumbering preserves later assignments', () => {
  const c = assigned();
  const p = c.getSnapshot().doc;
  c.beginStructureEdit();
  assert.ok(c.applyStructuralOp({ kind: 'move', index: 1, bar: 2 }));
  let next = c.getSnapshot().doc;
  assert.equal(next.assignments[p.sectionIds[0]], undefined);
  assert.equal(next.assignments[p.sectionIds[1]], undefined);
  assert.deepEqual(next.assignments[p.sectionIds[2]], p.assignments[p.sectionIds[2]]);
  c.undo();
  assert.ok(c.applyStructuralOp({ kind: 'merge', index: 0, keep: 'first' }));
  next = c.getSnapshot().doc;
  assert.deepEqual(next.sectionIds, [p.sectionIds[0], p.sectionIds[2]]);
  assert.deepEqual(next.assignments[p.sectionIds[2]], p.assignments[p.sectionIds[2]]);
  assert.deepEqual(next.blocks, p.blocks);
  c.undo();
  assert.ok(c.applyStructuralOp({ kind: 'split', index: 0, bar: 2, label: 'intro' }));
  assert.equal(c.getSnapshot().doc.sectionIds.length, 4);
});

test('cut history travels with undo and draft restore, including legacy cuts', () => {
  const c = make();
  const cutAbc = abc.replace('C8|D8|', 'D8|').replace('C8|D8|', 'D8|');
  assert.ok(c.applyStructuralOp({ kind: 'cut', abc: cutAbc }));
  const cut = c.getSnapshot();
  assert.equal(cut.cuts.length, 1);
  c.confirmStructure();
  assert.equal(c.applySuggestions(c.captureGuard(), {}, true), false);
  assert.ok(c.undo()); assert.ok(c.undo());
  assert.equal(c.getSnapshot().cuts.length, 0);
  assert.equal(c.getSnapshot().doc.abc, abc);
  assert.ok(c.redo());
  const restored = make();
  assert.ok(restored.restoreDraft({ [PREPARE_DRAFT_KEY]: cut.doc, 'cover-studio-scoreCuts': cut.cuts }, initial));
  assert.equal(restored.getSnapshot().cuts.length, 1);
  assert.equal(restored.canUndo, false);
  assert.equal(restored.getSnapshot().doc.abc, cutAbc);
  assert.ok(restored.restoreDraft({ 'cover-studio-sheetAbc': abc, 'cover-studio-lyrics': lyrics,
    'cover-studio-scoreCuts': [{ abc, lyrics, after: cutAbc, afterLyrics: lyrics }] }, initial));
  assert.equal(restored.getSnapshot().cuts.length, 1);
  assert.equal(restored.getSnapshot().doc.confirmed.structure, null);
});

test('draft restore validates V2, migrates legacy saved ABC, and resets source history', () => {
  const c = assigned();
  c.confirmLyrics();
  const ready = structuredClone(c.getSnapshot().doc);
  const restored = make();
  assert.ok(restored.restoreDraft({ [PREPARE_DRAFT_KEY]: ready }, initial));
  assert.deepEqual(restored.getSnapshot().doc, ready);
  assert.equal(restored.getSnapshot().step, 'ready');
  assert.equal(restored.canUndo, false);
  assert.equal(restored.restoreDraft({ [PREPARE_DRAFT_KEY]: { version: 99 } }, initial), false);
  assert.deepEqual(restored.getSnapshot().doc, ready);
  const savedAbc = abc.replace('% verse', '% intro');
  assert.ok(restored.restoreDraft({ 'cover-studio-sheetAbc': savedAbc, 'cover-studio-lyrics': 'untagged words' },
    { sourceKey: 'recording-b', abc: '' }));
  assert.equal(restored.getSnapshot().doc.abc, savedAbc);
  assert.equal(restored.getSnapshot().doc.sourceKey, 'recording-b');
  assert.equal(restored.getSnapshot().doc.blocks[0].text, 'untagged words');
  assert.deepEqual(restored.getSnapshot().doc.assignments, {});
  assert.equal(restored.getSnapshot().step, 'structure');
  assert.equal(restored.undo(), false);
});

test('mounted hook persists mirror fields, restores on remount, and disables recording after cuts', async () => {
  const saved = { window: globalThis.window, document: globalThis.document, localStorage: globalThis.localStorage };
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  const storage = new Map<string, string>();
  Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true,
    localStorage: { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => storage.set(key, value) } });
  let hook!: ReturnType<typeof useYue2Prepare>;
  function Harness({ source }: { source: PrepareInitial }) { hook = useYue2Prepare({ ...source, hasRecording: true }); return null; }
  let root = createRoot(document.getElementById('root')!);
  const render = (source = initial) => act(async () => root.render(createElement(StrictMode, {}, createElement(Harness, { source }))));
  try {
    await render();
    assert.equal(hook.recordingSeekAvailable, true);
    await act(async () => { hook.applyStructuralOp({ kind: 'cut', abc: abc.replace('% verse', '% intro') }); });
    assert.equal(hook.recordingSeekAvailable, false);
    assert.equal(hook.recordingAlignmentAvailable, false);
    const fields = structuredClone(hook.draftFields);
    assert.deepEqual(JSON.parse(storage.get(PREPARE_DRAFT_KEY)!), fields[PREPARE_DRAFT_KEY]);
    await act(async () => { hook.undo(); });
    assert.equal(hook.recordingSeekAvailable, true);
    await act(async () => { hook.redo(); });
    await act(async () => root.unmount());
    root = createRoot(document.getElementById('root')!);
    await render();
    assert.equal(hook.recordingSeekAvailable, false);
    assert.deepEqual(hook.doc, fields[PREPARE_DRAFT_KEY]);
    assert.equal(hook.canUndo, false);
    await act(async () => root.unmount());
    root = createRoot(document.getElementById('root')!);
    // The browser may open a different source while the persisted document
    // and cuts still belong to the previous one.
    await render({ ...initial, sourceKey: 'recording-b' });
    assert.equal(hook.doc.sourceKey, 'recording-b');
    assert.equal(hook.doc.abc, abc);
    assert.equal(hook.recordingSeekAvailable, true);
    assert.equal(hook.canUndo, false);
    await render({ ...initial, sourceKey: 'recording-c' });
    assert.equal(hook.doc.sourceKey, 'recording-c');
    assert.equal(hook.recordingSeekAvailable, true);
  } finally {
    await act(async () => root.unmount());
    Object.assign(globalThis, saved);
  }
});
