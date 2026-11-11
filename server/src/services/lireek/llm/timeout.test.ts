import test from 'node:test';
import assert from 'node:assert/strict';
import { config } from '../../../config.js';
import { LMStudioProvider } from './lmstudio.js';
import { OllamaProvider } from './ollama.js';
import { LlamaCppProvider } from './llamacpp.js';
import { OpenAICompatProvider } from './openai-compat.js';
import { OpenAIProvider } from './openai.js';
import { AnthropicProvider } from './anthropic.js';
import { GeminiProvider } from './gemini.js';
import { UnslothProvider } from './unsloth.js';

const providers = [LMStudioProvider, OllamaProvider, LlamaCppProvider, OpenAICompatProvider,
  OpenAIProvider, AnthropicProvider, GeminiProvider, UnslothProvider];

function frame(Provider: typeof providers[number], reasoning: boolean): string {
  const content = reasoning ? { reasoning_content: 'thinking' } : { content: 'reply' };
  const data = Provider === OllamaProvider ? { message: reasoning ? { thinking: 'thinking' } : content }
    : Provider === AnthropicProvider ? { type: 'content_block_delta', delta: reasoning ? { thinking: 'thinking' } : { text: 'reply' } }
    : Provider === GeminiProvider ? { candidates: [{ content: { parts: [{ text: 'reply' }] } }] }
    : { choices: [{ delta: content }] };
  return `${Provider === OllamaProvider ? '' : 'data: '}${JSON.stringify(data)}\n\n`;
}

test('active local streams outlive the timeout, including reasoning tokens', async t => {
  const timeout = config.lireek.llmTimeoutMs;
  const keyless = config.lireek.unslothKeyless;
  const username = config.lireek.unslothUsername;
  const password = config.lireek.unslothPassword;
  config.lireek.llmTimeoutMs = 180;
  Object.assign(config.lireek, { unslothKeyless: true, unslothUsername: '', unslothPassword: '' });
  t.after(() => { Object.assign(config.lireek, { llmTimeoutMs: timeout, unslothKeyless: keyless, unslothUsername: username, unslothPassword: password }); });
  for (const Provider of providers) {
    await t.test(Provider.name, async t => {
      t.mock.method(globalThis, 'fetch', async (_url: string | URL | Request, init?: RequestInit) => {
        const signal = init?.signal;
        return new Response(new ReadableStream({
          start(controller) {
            let index = 0;
            const timer = setInterval(() => {
              controller.enqueue(new TextEncoder().encode(frame(Provider, index < 5)));
              if (++index === 6) { clearInterval(timer); controller.close(); }
            }, 60);
            signal?.addEventListener('abort', () => { clearInterval(timer); controller.error(signal.reason); }, { once: true });
          },
        }));
      });
      const chunks: string[] = [];
      const expected = Provider === GeminiProvider ? 'reply'.repeat(6) : 'reply';
      assert.equal(await new Provider().call('system', 'user', 'fixture', text => chunks.push(text)), expected);
      assert.equal(chunks.length, 6);
    });
  }
});

test('a stream that stops producing tokens aborts and releases its reader', async t => {
  const timeout = config.lireek.llmTimeoutMs;
  config.lireek.llmTimeoutMs = 80;
  t.after(() => { config.lireek.llmTimeoutMs = timeout; });
  let body: ReadableStream<Uint8Array>;
  let signal: AbortSignal;
  t.mock.method(globalThis, 'fetch', async (_url: string | URL | Request, init?: RequestInit) => {
    signal = init!.signal!;
    body = new ReadableStream({ start(controller) {
      controller.enqueue(new TextEncoder().encode(frame(LMStudioProvider, false)));
      signal.addEventListener('abort', () => controller.error(signal.reason), { once: true });
    } });
    return new Response(body);
  });
  await assert.rejects(new LMStudioProvider().call('system', 'user', 'fixture', () => {}), /without response tokens.*LLM_TIMEOUT_MS/);
  assert.equal(body!.locked, false);
  assert.equal(signal!.aborted, true);
});

test('non-streaming calls retain the total time budget through JSON decoding', async t => {
  const timeout = config.lireek.llmTimeoutMs;
  config.lireek.llmTimeoutMs = 80;
  t.after(() => { config.lireek.llmTimeoutMs = timeout; });
  t.mock.method(globalThis, 'fetch', async (_url: string | URL | Request, init?: RequestInit) => new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode('{"choices":'));
      init!.signal!.addEventListener('abort', () => controller.error(init!.signal!.reason), { once: true });
    },
  })));
  await assert.rejects(new LMStudioProvider().call('system', 'user', 'fixture'), /LLM_TIMEOUT_MS.*80/);
});

test('SSE keepalives do not extend a silent model call', async t => {
  const timeout = config.lireek.llmTimeoutMs;
  config.lireek.llmTimeoutMs = 80;
  t.after(() => { config.lireek.llmTimeoutMs = timeout; });
  t.mock.method(globalThis, 'fetch', async (_url: string | URL | Request, init?: RequestInit) => new Response(new ReadableStream({
    start(controller) {
      const timer = setInterval(() => controller.enqueue(new TextEncoder().encode(': ping\n\n')), 20);
      init!.signal!.addEventListener('abort', () => { clearInterval(timer); controller.error(init!.signal!.reason); }, { once: true });
    },
  })));
  await assert.rejects(new LMStudioProvider().call('system', 'user', 'fixture', () => {}), /without response tokens/);
});

test('provider errors remain intact and clean up the timer', async t => {
  const timeout = config.lireek.llmTimeoutMs;
  config.lireek.llmTimeoutMs = 80;
  t.after(() => { config.lireek.llmTimeoutMs = timeout; });
  let signal: AbortSignal;
  t.mock.method(globalThis, 'fetch', async (_url: string | URL | Request, init?: RequestInit) => {
    signal = init!.signal!;
    return new Response('bad model', { status: 400 });
  });
  await assert.rejects(new LMStudioProvider().call('system', 'user', 'fixture'), /LM Studio error: 400 bad model/);
  assert.equal(signal!.aborted, true);
});

test('a successful call cleans up its timer and the next call reads the live setting', async t => {
  const timeout = config.lireek.llmTimeoutMs;
  config.lireek.llmTimeoutMs = 80;
  t.after(() => { config.lireek.llmTimeoutMs = timeout; });
  let signal: AbortSignal;
  t.mock.method(globalThis, 'fetch', async (_url: string | URL | Request, init?: RequestInit) => {
    signal = init!.signal!;
    return Response.json({ choices: [{ message: { content: 'reply' } }] });
  });
  assert.equal(await new LMStudioProvider().call('system', 'user', 'fixture'), 'reply');
  assert.equal(signal!.aborted, true);
  config.lireek.llmTimeoutMs = 120;
  t.mock.method(globalThis, 'fetch', async (_url: string | URL | Request, init?: RequestInit) => new Promise((_resolve, reject) => {
    init!.signal!.addEventListener('abort', () => reject(init!.signal!.reason), { once: true });
  }));
  await assert.rejects(new LMStudioProvider().call('system', 'user', 'fixture'), /LLM_TIMEOUT_MS.*120/);
});

test('LM Studio retries rejected thinking options with the same timeout guard', async t => {
  const timeout = config.lireek.llmTimeoutMs;
  config.lireek.llmTimeoutMs = 500;
  t.after(() => { config.lireek.llmTimeoutMs = timeout; });
  const requests: RequestInit[] = [];
  const payloads: Record<string, unknown>[] = [];
  let cancelled = false;
  t.mock.method(globalThis, 'fetch', async (_url: string | URL | Request, init?: RequestInit) => {
    requests.push(init!);
    payloads.push(JSON.parse(String(init!.body)));
    if (requests.length === 1) return new Response(new ReadableStream({ cancel() { cancelled = true; } }), { status: 400 });
    return Response.json({ choices: [{ message: { content: 'reply' } }] });
  });
  assert.equal(await new LMStudioProvider().call('system', 'user', 'fixture', undefined, { noThink: true }), 'reply');
  assert.equal(cancelled, true);
  assert.equal(payloads[0].reasoning_effort, 'none');
  assert.equal('reasoning_effort' in payloads[1], false);
  assert.equal(requests[0].signal, requests[1].signal);
});

test('silent completion timeout identifies the live setting', async t => {
  const timeout = config.lireek.llmTimeoutMs;
  config.lireek.llmTimeoutMs = 80;
  t.after(() => { config.lireek.llmTimeoutMs = timeout; });
  t.mock.method(globalThis, 'fetch', async (_url: string | URL | Request, init?: RequestInit) => new Promise((_resolve, reject) => {
    init?.signal?.addEventListener('abort', () => reject(init.signal!.reason), { once: true });
  }));
  await assert.rejects(new LMStudioProvider().call('system', 'user', 'fixture', () => {}),
    /LLM_TIMEOUT_MS.*80.*Settings > AI Services.*LLM call timeout/);
});
