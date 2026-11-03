import assert from 'node:assert/strict';
import test from 'node:test';
import { parseHTML } from 'linkedom';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClientProvider } from '@tanstack/react-query';
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import { createQueryClient } from '../../queries/client';
import { envOptions, gpusOptions } from '../../queries/shell-queue/settings';
import { useEnvSettingsForm, type EnvSettingsForm } from './useEnvSettingsForm';

await i18n.use(initReactI18next).init({ lng: 'en', resources: {}, interpolation: { escapeValue: false } });

const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });

test('the env draft is seeded only while clean, and a dirty refetch surfaces external edits', async () => {
  const saved = { window: globalThis.window, document: globalThis.document, fetch: globalThis.fetch };
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });
  const posts: Array<Record<string, string>> = [];
  let failSave = false;
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith('/settings/env') && init?.method === 'POST') {
      const body = JSON.parse(String(init.body)) as { values: Record<string, string> };
      posts.push(body.values);
      return failSave ? json({ error: 'disk full' }, 500) : json({
        updated: Object.keys(body.values), restartRequired: false,
      });
    }
    if (url.includes('/settings/env')) return json({ values: { A: '1', B: '2' }, restartKeys: ['A'] });
    if (url.includes('/settings/gpus')) return json({ gpus: [] });
    throw new Error(`Unexpected request: ${url}`);
  }) as typeof fetch;

  const client = createQueryClient();
  let form!: EnvSettingsForm;
  function Probe() {
    form = useEnvSettingsForm();
    return createElement('span', {}, `${form.loading}|${form.dirty}|${form.values.A}|${form.values.B}`);
  }
  const root = createRoot(document.getElementById('root')!);
  const settle = async () => {
    for (let i = 0; i < 5; i++) await act(async () => { await new Promise(r => setTimeout(r, 0)); });
  };
  const shown = () => document.querySelector('span')?.textContent;
  try {
    await act(async () => {
      root.render(createElement(QueryClientProvider, { client }, createElement(Probe)));
    });
    await settle();
    assert.equal(shown(), 'false|false|1|2', 'the clean form seeds from the server');

    await act(async () => { form.change('A', '9'); });
    assert.equal(form.dirty, true);
    assert.equal(form.values.A, '9');

    // Another tab writes B=3 while this form has an unsaved A edit. The draft
    // must stay; the new server snapshot becomes the diff basis, so draft B=2
    // now differs from original B=3.
    await act(async () => {
      client.setQueryData(envOptions().queryKey, { values: { A: '1', B: '3' }, restartKeys: ['A'] });
    });
    await settle();
    assert.equal(form.values.A, '9', 'a dirty draft is not overwritten');
    assert.equal(form.values.B, '2', 'the draft keeps the value the user last saw');
    await act(async () => { form.change('A', '1'); });
    assert.equal(form.dirty, true, 'the external B edit is now a diff on its own');
    assert.equal(form.values.B, '2');

    await act(async () => { form.save(); });
    await settle();
    assert.deepEqual(posts, [{ B: '2' }], 'save sends only the changed keys');
    assert.equal(form.dirty, false, 'a successful save adopts the draft as the new original');
    assert.equal(form.status?.type, 'success');
    assert.equal(client.getQueryState(envOptions().queryKey)?.isInvalidated, false, 'the refetch ran through queryManager');

    failSave = true;
    await act(async () => { form.change('B', '7'); });
    await act(async () => { form.save(); });
    await settle();
    assert.equal(form.status?.type, 'error');
    assert.match(form.status?.text ?? '', /Failed to save/);
    assert.equal(form.dirty, true, 'a failed save leaves the draft dirty');

    // A clean refetch still replaces the draft (the form never went dirty).
    await act(async () => { form.change('B', '2'); });
    await act(async () => {
      client.setQueryData(envOptions().queryKey, { values: { A: '4', B: '2' }, restartKeys: [] });
    });
    await settle();
    assert.equal(form.values.A, '4');
    assert.equal(form.dirty, false);
  } finally {
    await act(async () => root.unmount());
    client.clear();
    Object.assign(globalThis, saved);
  }
});

test('a failed GPU read leaves the env form usable instead of blocking it', async () => {
  const saved = { window: globalThis.window, document: globalThis.document, fetch: globalThis.fetch };
  const { window, document } = parseHTML('<html><body><div id="root"></div></body></html>');
  Object.assign(globalThis, { window, document, IS_REACT_ACT_ENVIRONMENT: true });
  globalThis.fetch = (async (input: string | URL | Request) => {
    const url = String(input);
    if (url.includes('/settings/env')) return json({ values: { A: '1' }, restartKeys: [] });
    if (url.includes('/settings/gpus')) return json({ error: 'nvidia-smi missing' }, 500);
    throw new Error(`Unexpected request: ${url}`);
  }) as typeof fetch;

  const client = createQueryClient();
  let form!: EnvSettingsForm;
  function Probe() { form = useEnvSettingsForm(); return null; }
  const root = createRoot(document.getElementById('root')!);
  const settle = async () => {
    for (let i = 0; i < 8; i++) await act(async () => { await new Promise(r => setTimeout(r, 0)); });
  };
  try {
    await act(async () => {
      root.render(createElement(QueryClientProvider, { client }, createElement(Probe)));
    });
    await settle();
    assert.equal(form.loading, false);
    assert.equal(form.values.A, '1');
    assert.deepEqual(form.gpus, []);
    assert.equal(client.getQueryState(gpusOptions().queryKey)?.status, 'error');
  } finally {
    await act(async () => root.unmount());
    client.clear();
    Object.assign(globalThis, saved);
  }
});
