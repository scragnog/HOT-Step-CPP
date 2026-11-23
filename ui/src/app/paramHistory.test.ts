import assert from 'node:assert/strict';
import { test } from 'node:test';
import { COALESCE_MS, HISTORY_LIMIT, createHistory } from './paramHistory';

function rig() {
  const store: Record<string, unknown> = { steps: 8, scale: 1 };
  let t = 1000;
  const h = createHistory((c, v) => { store[c.id] = v; }, () => t);
  const set = (id: string, v: unknown, label = id) => {
    const before = store[id]; store[id] = v;
    h.record(label, { id, backend: 'ace', before, after: v });
  };
  return { store, h, set, tick: (ms: number) => { t += ms; } };
}

test('undo and redo restore through the apply callback', () => {
  const { store, h, set, tick } = rig();
  set('steps', 12); tick(COALESCE_MS + 1);
  set('scale', 3);
  h.undo(); assert.equal(store.scale, 1);
  h.undo(); assert.equal(store.steps, 8);
  assert.equal(h.getSnapshot().canUndo, false);
  h.redo(); h.redo();
  assert.deepEqual([store.steps, store.scale], [12, 3]);
});

test('a drag on one control is one step; it starts from where the drag began', () => {
  const { store, h, set, tick } = rig();
  for (const v of [9, 10, 11, 12]) { set('steps', v); tick(50); }
  assert.equal(h.getSnapshot().steps.length, 1);
  h.undo(); assert.equal(store.steps, 8);
});

test('dragging back to the start leaves no step', () => {
  const { h, set, tick } = rig();
  set('steps', 9); tick(50); set('steps', 8);
  assert.equal(h.getSnapshot().steps.length, 0);
});

test('a transaction is one step, undone together', () => {
  const { store, h, set } = rig();
  h.transaction('Both', () => { set('steps', 20); set('scale', 5); set('steps', 21); });
  assert.equal(h.getSnapshot().steps.length, 1);
  assert.equal(h.getSnapshot().steps[0].label, 'Both');
  h.undo(); assert.deepEqual([store.steps, store.scale], [8, 1]);
});

test('a new edit after an undo drops the redo branch', () => {
  const { h, set, tick } = rig();
  set('steps', 12); tick(COALESCE_MS + 1);
  h.undo();
  set('scale', 2);
  assert.equal(h.getSnapshot().canRedo, false);
  assert.equal(h.getSnapshot().steps.length, 1);
});

test('a restore is not recorded as an edit', () => {
  const { h, set } = rig();
  set('steps', 12);
  h.undo(); h.redo();
  assert.equal(h.getSnapshot().steps.length, 1);
});

test('keeps the latest 100 steps', () => {
  const { h, set, tick } = rig();
  for (let i = 0; i < HISTORY_LIMIT + 5; i++) { set('steps', 100 + i); tick(COALESCE_MS + 1); }
  assert.equal(h.getSnapshot().steps.length, HISTORY_LIMIT);
});
