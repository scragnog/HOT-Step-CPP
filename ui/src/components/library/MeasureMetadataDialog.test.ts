import assert from 'node:assert/strict';
import test from 'node:test';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { parseHTML } from 'linkedom';

const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
Object.assign(window, { location: new URL('http://localhost:3000/') });
Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });
const { MeasureMetadataDialog } = await import('./MeasureMetadataDialog');
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

async function scenario(run: (ctx: {
  click: (label: string) => Promise<void>;
  calls: { url: string; method: string; body: any }[];
  song: { id: string; title: string; audio_url: string; bpm: number; key_scale: string };
  fail: () => void;
  closed: () => number;
}) => Promise<void>) {
  const originalFetch = globalThis.fetch;
  const calls: { url: string; method: string; body: any }[] = [];
  const song = { id: 'track', title: 'Test track', audio_url: '/audio/track.wav', bpm: 80, key_scale: 'D Minor' };
  let fail = false;
  let closed = 0;
  globalThis.fetch = (async (url: string, init?: RequestInit) => {
    const method = init?.method || 'GET';
    calls.push({ url: String(url), method, body: init?.body ? JSON.parse(String(init.body)) : undefined });
    if (String(url).endsWith('/analyze')) return fail
      ? json({ error: 'Essentia is not installed. No metadata was saved.' }, 503)
      : json({ bpm: 123, key: 'G', scale: 'major' });
    return json({ song });
  }) as typeof fetch;
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } });
  const root = createRoot(document.getElementById('root')!);
  const settle = () => new Promise(resolve => setTimeout(resolve, 30));
  try {
    await act(async () => { root.render(createElement(QueryClientProvider, { client },
      createElement(MeasureMetadataDialog, { songId: song.id, onClose: () => { closed++; } }))); await settle(); });
    await act(settle);
    const click = async (label: string) => {
      const button = Array.from(document.querySelectorAll('button')).find(b => b.textContent === label);
      assert.ok(button, `button ${label} exists`);
      assert.equal(button.hasAttribute('disabled'), false);
      await act(async () => { button.click(); await settle(); });
      await act(settle);
    };
    await run({ click, calls, song, fail: () => { fail = true; }, closed: () => closed });
  } finally {
    await act(async () => root.unmount());
    client.clear();
    globalThis.fetch = originalFetch;
  }
}

test('only confirmation writes BPM and key, leaving unrelated metadata alone', async () => {
  await scenario(async ({ click, calls, closed }) => {
    assert.equal(calls.some(c => c.url.endsWith('/analyze')), false, 'opening is not analysis');
    await click('Measure BPM and key');
    assert.deepEqual(calls.find(c => c.url.endsWith('/analyze'))?.body, { audioUrl: '/audio/track.wav' });
    assert.ok(document.body.textContent?.includes('80'));
    assert.ok(document.body.textContent?.includes('123'));
    assert.ok(document.body.textContent?.includes('D Minor'));
    assert.ok(document.body.textContent?.includes('G Major'));
    assert.equal(calls.some(c => c.method === 'PATCH'), false);
    await click('Replace saved values');
    assert.deepEqual(calls.filter(c => c.method === 'PATCH').map(c => c.body), [{ bpm: 123, key_scale: 'G Major' }]);
    assert.equal(closed(), 1);
  });
});

test('cancel and missing Essentia never save', async () => {
  await scenario(async ({ click, calls }) => {
    await click('Measure BPM and key');
    await click('Cancel');
    assert.equal(calls.some(c => c.method === 'PATCH'), false);
  });
  await scenario(async ({ click, calls, fail }) => {
    fail();
    await click('Measure BPM and key');
    assert.match(document.querySelector('[role="alert"]')?.textContent || '', /Essentia is not installed/);
    assert.equal(calls.some(c => c.method === 'PATCH'), false);
    assert.equal(Array.from(document.querySelectorAll('button')).some(b => b.textContent === 'Replace saved values'), false);
  });
});

test('a manual edit made after measuring is preserved', async () => {
  await scenario(async ({ click, calls, song }) => {
    await click('Measure BPM and key');
    song.bpm = 90;
    await click('Replace saved values');
    assert.match(document.querySelector('[role="alert"]')?.textContent || '', /changed during measurement/);
    assert.equal(calls.some(c => c.method === 'PATCH'), false);
  });
});
