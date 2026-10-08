import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
import { config } from '../../../config.js';
import { AnthropicProvider } from './anthropic.js';
import { listProviders } from './registry.js';

const fallbackModels = ['claude-fable-5-1', 'claude-opus-5-5', 'claude-sonnet-5', 'claude-haiku-4-5-20251001'];

test('lists every active Anthropic model through the provider registry', async (t) => {
  const key = config.lireek.anthropicApiKey;
  config.lireek.anthropicApiKey = 'test-key';
  t.after(() => { config.lireek.anthropicApiKey = key; mock.restoreAll(); });

  const requests: Array<{ url: URL; headers: Headers }> = [];
  mock.method(globalThis, 'fetch', async (input: string | URL | Request, init?: RequestInit) => {
    const url = new URL(String(input));
    if (url.origin !== 'https://api.anthropic.com') return new Response('', { status: 503 });
    requests.push({ url, headers: new Headers(init?.headers) });
    return Response.json(requests.length === 1
      ? { data: [{ id: 'claude-fable-5-1' }], has_more: true, last_id: 'claude-fable-5-1' }
      : { data: [{ id: 'claude-opus-5-5' }, { id: 'claude-sonnet-5' }], has_more: false, last_id: 'claude-sonnet-5' });
  });

  const info = (await listProviders()).find(provider => provider.id === 'anthropic');
  assert.deepEqual(info?.models, ['claude-fable-5-1', 'claude-opus-5-5', 'claude-sonnet-5']);
  assert.equal(requests.length, 2);
  assert.equal(requests[0].url.pathname, '/v1/models');
  assert.equal(requests[0].url.searchParams.get('lifecycle'), 'active');
  assert.equal(requests[0].url.searchParams.get('limit'), '1000');
  assert.equal(requests[1].url.searchParams.get('after_id'), 'claude-fable-5-1');
  assert.equal(requests[0].headers.get('x-api-key'), 'test-key');
  assert.equal(requests[0].headers.get('anthropic-version'), '2023-06-01');
});

test('uses the static models without a key or when the model API fails', async (t) => {
  const key = config.lireek.anthropicApiKey;
  t.after(() => { config.lireek.anthropicApiKey = key; mock.restoreAll(); });
  const fetch = mock.method(globalThis, 'fetch', async () => new Response('', { status: 503 }));
  const provider = new AnthropicProvider();

  config.lireek.anthropicApiKey = '';
  assert.deepEqual((await provider.toInfoAsync()).models, fallbackModels);
  assert.equal(fetch.mock.callCount(), 0);

  config.lireek.anthropicApiKey = 'test-key';
  assert.deepEqual((await provider.toInfoAsync()).models, fallbackModels);
  assert.equal(fetch.mock.callCount(), 1);
});

test('sends a current Messages API request without rejected sampling fields', async (t) => {
  const key = config.lireek.anthropicApiKey;
  config.lireek.anthropicApiKey = 'test-key';
  t.after(() => { config.lireek.anthropicApiKey = key; mock.restoreAll(); });

  let payload: Record<string, unknown> | undefined;
  mock.method(globalThis, 'fetch', async (_input: string | URL | Request, init?: RequestInit) => {
    payload = JSON.parse(String(init?.body));
    return Response.json({ content: [{ type: 'text', text: 'reply' }] });
  });

  const reply = await new AnthropicProvider().call('system prompt', 'user prompt', 'claude-opus-5-5');
  assert.equal(reply, 'reply');
  assert.deepEqual(payload, {
    model: 'claude-opus-5-5', max_tokens: 4096, system: 'system prompt',
    messages: [{ role: 'user', content: 'user prompt' }], stream: false,
  });
});

test('returns text when a non-stream response begins with thinking blocks', async (t) => {
  t.after(() => mock.restoreAll());
  mock.method(globalThis, 'fetch', async () => Response.json({ content: [
    { type: 'thinking', thinking: '', signature: 'signed' },
    { type: 'text', text: 'expected ' },
    { type: 'thinking', thinking: '', signature: 'signed-again' },
    { type: 'text', text: 'reply' },
  ] }));

  assert.equal(await new AnthropicProvider().call('system', 'user', 'claude-opus-5-5'), 'expected reply');
});
