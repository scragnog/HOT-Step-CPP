// sharedEventSource.test.ts — one connection per URL, replay for late joiners.
// No UI test runner is wired up for this project; run with the server's tsx:
//   (cd server && node --import tsx --test ../ui/src/services/sharedEventSource.test.ts)
import { test } from 'node:test';
import assert from 'node:assert/strict';

class FakeEventSource {
  static CONNECTING = 0; static OPEN = 1; static CLOSED = 2;
  static made: FakeEventSource[] = [];
  readyState = FakeEventSource.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(public url: string) { FakeEventSource.made.push(this); }
  close() { this.readyState = FakeEventSource.CLOSED; }
  open() { this.readyState = FakeEventSource.OPEN; this.onopen?.(); }
  send(data: string) { this.onmessage?.({ data }); }
}
Object.assign(globalThis, { EventSource: FakeEventSource, window: globalThis });
const { subscribeEventSource } = await import('./sharedEventSource');
const tick = () => new Promise(r => setTimeout(r, 0));

test('finished replays stop on every 4xx without EventSource or retry', async () => {
  const original = globalThis.fetch;
  try {
    for (const status of [400, 401, 403, 404, 410, 429]) {
      let requests = 0, failures = 0;
      const before = FakeEventSource.made.length;
      globalThis.fetch = async () => { requests++; return new Response(null, { status }); };
      const stop = subscribeEventSource('/finished', { error: () => { failures++; } }, { replayOnly: true, reconnect: true });
      await tick(); await tick();
      assert.equal(requests, 1, String(status));
      assert.equal(failures, 1);
      assert.equal(FakeEventSource.made.length, before);
      stop();
    }
  } finally { globalThis.fetch = original; }
});

test('finished replay parses split SSE frames and closes on terminal status', async () => {
  const original = globalThis.fetch;
  const seen: string[] = [];
  let cancelled = false;
  globalThis.fetch = async () => new Response(new ReadableStream({
    start(controller) {
      for (const chunk of ['data: step\r', '\n\r\ndata: done\n\ndata: ignored\n\n']) controller.enqueue(new TextEncoder().encode(chunk));
    },
    cancel() { cancelled = true; },
  }));
  const stop = subscribeEventSource('/finished-frames', { message: data => seen.push(data) }, { replayOnly: true, isTerminal: data => data === 'done' });
  try {
    await tick();
    assert.deepEqual(seen, ['step', 'done']);
    assert.equal(cancelled, true);
  } finally { stop(); globalThis.fetch = original; }
});

test('subscribers to one URL share a connection and late ones get the replay', async () => {
  const a: string[] = [], b: string[] = [];
  const stopA = subscribeEventSource('/s1', { message: d => a.push(d) });
  const es = FakeEventSource.made.at(-1)!;
  es.open(); es.send('1'); es.send('2');
  const stopB = subscribeEventSource('/s1', { message: d => b.push(d) });
  assert.equal(FakeEventSource.made.filter(e => e.url === '/s1').length, 1);
  await tick();
  es.send('3');
  assert.deepEqual(a, ['1', '2', '3']);
  assert.deepEqual(b, ['1', '2', '3']);
  stopA();
  assert.equal(es.readyState, FakeEventSource.OPEN);
  stopB();
  assert.equal(es.readyState, FakeEventSource.CLOSED);
});

test('a handler can unsubscribe itself during its replay', async () => {
  const seen: string[] = [];
  const keep = subscribeEventSource('/s2', {});
  const es = FakeEventSource.made.at(-1)!;
  es.open(); es.send('done'); es.send('after');
  const stop: () => void = subscribeEventSource('/s2', { message: d => { seen.push(d); stop(); } });
  await tick();
  assert.deepEqual(seen, ['done']);
  keep();
});

test('a reconnect clears the replay, since the server replays again', async () => {
  const stopA = subscribeEventSource('/s3', {});
  const es = FakeEventSource.made.at(-1)!;
  es.open(); es.send('old');
  es.open(); es.send('new');
  const b: string[] = [];
  const stopB = subscribeEventSource('/s3', { message: d => b.push(d) });
  await tick();
  assert.deepEqual(b, ['new']);
  stopA(); stopB();
});
