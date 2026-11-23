import assert from 'node:assert/strict';
import test from 'node:test';
import { parseHTML } from 'linkedom';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// With nothing remembered, both folder dialogs must still ask the server for
// its default folder (an empty path) on first open, instead of sitting blank.
const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
// axios reads location.href at import time when it sees a window.
Object.assign(window, { location: { href: 'http://localhost/', origin: 'http://localhost' } });
const browsed: string[] = [];
Object.assign(globalThis, {
  window, document, IS_REACT_ACT_ENVIRONMENT: true,
  localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
  fetch: async (url: string) => {
    browsed.push(new URL(url, 'http://x').searchParams.get('path') ?? 'missing');
    return new Response(JSON.stringify({ current: 'C:\\server', parent: null, entries: [] }),
      { status: 200, headers: { 'Content-Type': 'application/json' } });
  },
});
const { FileBrowserModal } = await import('./FileBrowserModal');
const { FolderPicker } = await import('@/components/training-studio/panels/FolderPicker');

async function firstOpen(element: ReturnType<typeof createElement>) {
  browsed.length = 0;
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const root = createRoot(document.getElementById('root')!);
  await act(async () => { root.render(createElement(QueryClientProvider, { client }, element)); });
  for (let i = 0; i < 20 && browsed.length === 0; i++) {
    await act(async () => { await new Promise(r => setTimeout(r, 10)); });
  }
  await act(async () => { root.unmount(); });
  client.clear();
  return [...browsed];
}

const noop = () => {};

test('FileBrowserModal lists the server default folder on a fresh first open', async () => {
  assert.deepEqual(await firstOpen(createElement(FileBrowserModal, { open: true, onClose: noop, onSelect: noop, mode: 'folder' })), ['']);
});

test('FolderPicker lists the server default folder on a fresh first open', async () => {
  assert.deepEqual(await firstOpen(createElement(FolderPicker, { open: true, onClose: noop, onSelect: noop })), ['']);
});
