import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { WriteLyricsTab } from './WriteLyricsTab.tsx';
import { scratchRequest, followScratchLyrics } from './scratchLyricsClient.ts';

test('the prompt form is available for an album without profiles and restores its brief', () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
    getItem: (key: string) => key === 'lyric-scratch-24' ? JSON.stringify({ theme: 'A walk home' }) : null,
  } });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  try {
    const html = renderToStaticMarkup(React.createElement(QueryClientProvider, { client },
      React.createElement(WriteLyricsTab, { lyricsSetId: 24, onRefresh: () => {}, onViewLyrics: () => {} })));
    for (const label of ['Ollama model', 'Theme', 'Style', 'Structure', 'Starter lyrics', 'A walk home']) assert.ok(html.includes(label));
    assert.equal((html.match(/<textarea/g) || []).length, 4);
    assert.ok(html.includes('disabled=""'));
  } finally {
    client.clear();
    if (original) Object.defineProperty(globalThis, 'localStorage', original);
    else Reflect.deleteProperty(globalThis, 'localStorage');
  }
});

test('prompt request accepts a theme or starter lines without a profile and captures the album/model', () => {
  const brief = { theme: ' home ', style: ' folk ', structure: ' Verse, Chorus ', existingLyrics: ' ' };
  assert.deepEqual(scratchRequest(24, brief, 'local-model'), {
    lyricsSetId: 24, provider: 'ollama', model: 'local-model', theme: 'home', style: 'folk', structure: 'Verse, Chorus', existingLyrics: '',
  });
  assert.throws(() => scratchRequest(24, { ...brief, theme: '' }, ''), /theme or some starter/);
  assert.equal(scratchRequest(24, { ...brief, theme: '', existingLyrics: 'A first verse' }, '').existingLyrics, 'A first verse');
});

test('completion reconciles the durable job; failures show the server error and leaving the tab does not cancel the job', async () => {
  const controller = new AbortController();
  let reads = 0;
  const deps = {
    follow: (async () => 'succeeded') as any,
    get: (async (id: string) => { reads++; assert.equal(id, 'job-1'); return { job: { error: 'Ollama is offline' } }; }) as any,
  };
  assert.equal(await followScratchLyrics('job-1', controller.signal, deps), true);
  assert.equal(reads, 1);
  deps.follow = async () => 'failed';
  await assert.rejects(followScratchLyrics('job-1', controller.signal, deps), /Ollama is offline/);
  controller.abort();
  assert.equal(await followScratchLyrics('job-1', controller.signal, deps), false);
  assert.equal(reads, 2);
});
