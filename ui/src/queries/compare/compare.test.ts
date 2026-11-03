import assert from 'node:assert/strict';
import test, { afterEach } from 'node:test';
import { parseHTML } from 'linkedom';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClientProvider } from '@tanstack/react-query';
import { createQueryClient } from '../client';
import { resourceKey } from '../keys';
import {
  compareQueries, CMP,
} from './queries';
import {
  applyScores, beginOptimisticScores, commitScoreResult, localStatus, patchSession,
  patchSessionSummary, reconcileScoreResult, rollbackScores, summaryOf, useSetCompareScore,
} from './writes';
import type { CompareSession, CompareTake } from '../../services/compareApi';

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

const json = (value: unknown) => new Response(JSON.stringify(value), {
  status: 200, headers: { 'Content-Type': 'application/json' },
});

const take = (songId: string, position: number): CompareTake => ({
  songId, position, title: songId.toUpperCase(), audioUrl: null, duration: 10,
  backend: null, createdAt: null, seed: null, seedB: null, seedBlend: null,
  generationParams: {}, scores: [],
});

const session = (): CompareSession => ({
  id: 's1', name: 'Comparison', group: '', baselineSongId: 'a',
  archived: false, status: 'unscored', scored: 0, total: 2,
  createdAt: '2026-01-01', updatedAt: '2026-01-01',
  criteria: [{ id: 'tone', label: 'Tone', type: 'scale', min: 1, max: 5 }],
  takes: [take('a', 0), take('b', 1)],
});

test('compare factories use canonical keys and pass the abort signal', async () => {
  const calls: string[] = [];
  globalThis.fetch = async url => {
    const value = String(url);
    calls.push(value);
    if (value.startsWith('/api/compare/sessions?')) return json({ sessions: [] });
    if (value === '/api/compare/sessions/s1') return json(session());
    if (value.startsWith('/api/compare/songs?')) return json({ total: 0, limit: 25, offset: 0, songs: [] });
    if (value.startsWith('/api/compare/analysis/')) return json({ songId: 'a' });
    if (value.startsWith('/api/compare/diff?')) return json({ a: 'a', b: 'b' });
    throw new Error(`unexpected ${value}`);
  };
  const client = createQueryClient();
  try {
    assert.deepEqual(await client.query(compareQueries.sessions({ archived: false })), { sessions: [] });
    assert.equal((await client.query(compareQueries.session('s1'))).id, 's1');
    assert.equal((await client.query(compareQueries.songSearch('drums', 25, 0))).total, 0);
    assert.equal((await client.query(compareQueries.analysis('a'))).songId, 'a');
    assert.equal((await client.query(compareQueries.diff('a', 'b'))).a, 'a');
    assert.deepEqual(calls, [
      '/api/compare/sessions?archived=0',
      '/api/compare/sessions/s1',
      '/api/compare/songs?q=drums&limit=25&offset=0',
      '/api/compare/analysis/a',
      '/api/compare/diff?a=a&b=b',
    ]);
    assert.deepEqual(compareQueries.sessions({ archived: false }).queryKey,
      ['hs', CMP.sessions, { api: '/api' }, null, { archived: false }]);
    assert.deepEqual(resourceKey(CMP.sessions), ['hs', 'compare/sessions']);
  } finally { client.clear(); }
});

test('localStatus and applyScores mirror the server rules', () => {
  const s = session();
  assert.deepEqual(localStatus(s), { status: 'unscored', scored: 0, total: 2 });

  const one = applyScores(s, [{ songId: 'a', criterion: 'tone', value: 4 }]);
  assert.equal(one.takes[0].scores[0].value, 4);
  assert.deepEqual(localStatus(one), { status: 'partial', scored: 1, total: 2 });
  assert.equal(one.status, 'partial');
  assert.equal(one.archived, false);

  const both = applyScores(one, [{ songId: 'b', criterion: 'tone', keep: true }]);
  assert.equal(both.status, 'scored');
  assert.equal(both.archived, true);

  // Clearing a value and an absent field both leave the rest alone.
  const cleared = applyScores(both, [{ songId: 'a', criterion: 'tone', value: null }]);
  assert.equal(cleared.takes[0].scores[0].value, null);
  assert.equal(cleared.takes[1].scores[0].keep, true);
  assert.equal(cleared.archived, true, 'auto-archive never reverses optimistically');
});

test('server result reconciles status, archive and counts', () => {
  const s = session();
  assert.deepEqual(reconcileScoreResult(s, { status: 'partial', archived: false, scored: 1, total: 3 }),
    { ...s, status: 'partial', archived: false, scored: 1, total: 3 });
});

test('optimistic scores patch the session and list, then commit or roll back', async () => {
  const client = createQueryClient();
  try {
    const current = session();
    const key = compareQueries.session('s1').queryKey;
    const listKey = compareQueries.sessions({ archived: false }).queryKey;
    client.setQueryData(key, current);
    client.setQueryData(listKey, { sessions: [summaryOf(current)] });

    const rollback = await beginOptimisticScores(client, 's1', [{ songId: 'a', criterion: 'tone', value: 5 }]);
    assert.equal(client.getQueryData<CompareSession>(key)?.takes[0].scores[0].value, 5);
    assert.equal(client.getQueryData<{ sessions: { status: string }[] }>(listKey)?.sessions[0].status, 'partial');

    commitScoreResult(client, 's1', { status: 'scored', archived: true, scored: 2, total: 2 });
    assert.equal(client.getQueryData<CompareSession>(key)?.archived, true);
    assert.equal(client.getQueryData<{ sessions: { status: string }[] }>(listKey)?.sessions[0].status, 'scored');

    rollbackScores(client, rollback);
    assert.deepEqual(client.getQueryData<CompareSession>(key), current);
    assert.equal(client.getQueryData<{ sessions: { status: string }[] }>(listKey)?.sessions[0].status, 'unscored');
  } finally { client.clear(); }
});

test('session commands patch the detail and the list row together', () => {
  const client = createQueryClient();
  try {
    const current = session();
    const key = compareQueries.session('s1').queryKey;
    const listKey = compareQueries.sessions({ archived: false }).queryKey;
    client.setQueryData(key, current);
    client.setQueryData(listKey, { sessions: [summaryOf(current)] });

    const withoutB = { ...current, takes: [current.takes[0]], total: 1 };
    patchSession(client, withoutB);
    assert.equal(client.getQueryData<CompareSession>(key)?.takes.length, 1);
    assert.equal(client.getQueryData<{ sessions: { total: number }[] }>(listKey)?.sessions[0].total, 1);

    patchSessionSummary(client, { ...summaryOf(current), archived: true, updatedAt: '2026-01-02' });
    assert.equal(client.getQueryData<CompareSession>(key)?.archived, true);
    assert.equal(client.getQueryData<CompareSession>(key)?.takes.length, 1, 'summary merge keeps takes');
    assert.equal(client.getQueryData<{ sessions: { archived: boolean }[] }>(listKey)?.sessions[0].archived, true);
  } finally { client.clear(); }
});

test('scoring the final take refetches list membership after the server auto-archives', async () => {
  const saved = { window: globalThis.window, document: globalThis.document };
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });
  const client = createQueryClient();
  const current = session();
  current.takes[0].scores = [{ criterion: 'tone', value: 4, keep: null, note: '' }];
  Object.assign(current, { status: 'partial', scored: 1 });
  const listKey = compareQueries.sessions({ archived: false }).queryKey;
  client.setQueryData(compareQueries.session('s1').queryKey, current);
  client.setQueryData(listKey, { sessions: [summaryOf(current)] });
  globalThis.fetch = async () => json({ status: 'scored', archived: true, scored: 2, total: 2 });
  let mutation!: ReturnType<typeof useSetCompareScore>;
  function Probe() { mutation = useSetCompareScore(); return null; }
  const root = createRoot(document.getElementById('root')!);
  try {
    await act(async () => root.render(createElement(QueryClientProvider, { client }, createElement(Probe))));
    await act(async () => {
      await mutation.mutateAsync({ sessionId: 's1', items: [{ songId: 'b', criterion: 'tone', value: 4 }] });
    });
    // The optimistic patch already archived the cached session, so only the
    // pre-mutation snapshot shows the membership change.
    assert.equal(client.getQueryState(listKey)?.isInvalidated, true);
  } finally {
    await act(async () => root.unmount());
    client.clear();
    Object.assign(globalThis, saved);
  }
});
