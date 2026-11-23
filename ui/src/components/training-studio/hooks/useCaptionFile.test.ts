import assert from 'node:assert/strict';
import test from 'node:test';
import { parseHTML } from 'linkedom';
import { act, createElement, StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClientProvider } from '@tanstack/react-query';
import { createQueryClient } from '@/queries/client';
import { trainingQueries } from '@/queries/training-data/queries';
import { useCaptionFile } from './useCaptionFile';

test('a caption editor seeds again on return to a sample, after a visit that never loaded', async () => {
  const saved = { window: globalThis.window, document: globalThis.document, fetch: globalThis.fetch };
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });
  let readsOfA = 0;
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    if (String(input).includes('/samples/B/')) {
      // B never answers before the drawer leaves it.
      return new Promise<Response>((_, reject) => init?.signal?.addEventListener('abort', () =>
        reject(new DOMException('aborted', 'AbortError'))));
    }
    readsOfA++;
    return new Response(JSON.stringify({ text: `A${readsOfA}` }), { headers: { 'Content-Type': 'application/json' } });
  }) as typeof fetch;
  const client = createQueryClient();
  function Drawer({ sample }: { sample: string }) {
    const file = useCaptionFile(trainingQueries.sampleMm3(null, 'ds', sample), true);
    return createElement('span', {}, file.text ?? 'LOADING');
  }
  const root = createRoot(document.getElementById('root')!);
  const render = (sample: string) => act(async () => {
    root.render(createElement(StrictMode, {}, createElement(QueryClientProvider, { client }, createElement(Drawer, { sample }))));
  });
  const shown = () => document.querySelector('span')?.textContent;
  const settle = async () => { for (let i = 0; i < 5; i++) await act(async () => { await new Promise(r => setTimeout(r, 0)); }); };
  try {
    await render('A'); await settle();
    // StrictMode's remount may cancel and repeat the first read.
    const first = shown();
    assert.match(first ?? '', /^A\d$/);
    await render('B'); await settle();
    assert.equal(shown(), 'LOADING');
    await render('A'); await settle();
    assert.equal(shown(), `A${readsOfA}`, 'the return visit seeds from its own fresh read');
    assert.notEqual(shown(), first);
  } finally {
    await act(async () => root.unmount()); client.clear();
    Object.assign(globalThis, saved);
  }
});
