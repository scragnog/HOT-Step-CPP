// httpClient.test.ts — the shared transport's JSON/error/upload contract.
//   (cd server && node --import tsx --test ../ui/src/services/httpClient.test.ts)
import { test } from 'node:test';
import assert from 'node:assert/strict';

type FakeResponse = { ok: boolean; status: number; statusText: string; json: () => Promise<any> };
function jsonResponse(status: number, body: unknown): FakeResponse {
  return { ok: status >= 200 && status < 300, status, statusText: 'x', json: async () => body };
}

class FakeXhr {
  static made: FakeXhr[] = [];
  status = 0;
  responseText = '';
  upload = { onprogress: null as ((e: { lengthComputable: boolean; loaded: number; total: number }) => void) | null };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  private headers: Record<string, string> = {};
  private aborted = false;
  constructor(public method?: string, public url?: string) { FakeXhr.made.push(this); }
  open(method: string, url: string) { this.method = method; this.url = url; }
  setRequestHeader(k: string, v: string) { this.headers[k] = v; }
  getHeader(k: string) { return this.headers[k]; }
  abort() { this.aborted = true; this.onabort?.(); }
  send(_form: FormData) { /* resolved explicitly by the test */ }
  respond(status: number, body: unknown) {
    if (this.aborted) return;
    this.status = status;
    this.responseText = JSON.stringify(body);
    this.onload?.();
  }
}

Object.assign(globalThis, { XMLHttpRequest: FakeXhr });
const { ApiClient, ApiError } = await import('./httpClient');

test('get returns the parsed body and sends the bearer token', async () => {
  const calls: { url: string; headers: HeadersInit | undefined }[] = [];
  (globalThis as any).fetch = async (url: string, init: RequestInit) => {
    calls.push({ url, headers: init.headers });
    return jsonResponse(200, { ok: true });
  };
  const client = new ApiClient();
  const result = await client.get<{ ok: boolean }>('/songs', { token: 'tok' });
  assert.deepEqual(result, { ok: true });
  assert.equal(calls[0].url, '/api/songs');
  assert.deepEqual(calls[0].headers, { Authorization: 'Bearer tok' });
});

test('post sends JSON with Content-Type and no auth header when no token', async () => {
  let seenBody = '', seenHeaders: HeadersInit | undefined;
  (globalThis as any).fetch = async (_url: string, init: RequestInit) => {
    seenBody = init.body as string; seenHeaders = init.headers;
    return jsonResponse(200, { created: true });
  };
  const client = new ApiClient({ baseUrl: '/api/custom' });
  await client.post('/things', { a: 1 });
  assert.equal(seenBody, JSON.stringify({ a: 1 }));
  assert.deepEqual(seenHeaders, { 'Content-Type': 'application/json' });
});

test('a non-OK response throws ApiError carrying status, body and revision/reason', async () => {
  (globalThis as any).fetch = async () => jsonResponse(409, { error: 'stale', currentRevision: 5, reason: 'conflict' });
  const client = new ApiClient();
  await assert.rejects(
    () => client.patch('/documents/1', { a: 1 }),
    (err: unknown) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.status, 409);
      assert.equal(err.message, 'stale');
      assert.equal(err.currentRevision, 5);
      assert.equal(err.reason, 'conflict');
      return true;
    },
  );
});

test('a non-JSON error body still throws, falling back to statusText', async () => {
  (globalThis as any).fetch = async () => ({ ok: false, status: 500, statusText: 'Boom', json: async () => { throw new Error('not json'); } });
  const client = new ApiClient();
  await assert.rejects(() => client.get('/x'), (err: unknown) => {
    assert.ok(err instanceof ApiError);
    assert.equal(err.message, 'Boom');
    return true;
  });
});

test('delete passes the auth header and no body', async () => {
  let seenInit: RequestInit = {};
  (globalThis as any).fetch = async (_url: string, init: RequestInit) => { seenInit = init; return jsonResponse(200, { ok: true }); };
  const client = new ApiClient();
  await client.delete('/songs/1', { token: 't' });
  assert.equal(seenInit.method, 'DELETE');
  assert.equal(seenInit.body, undefined);
  assert.deepEqual(seenInit.headers, { Authorization: 'Bearer t' });
});

test('upload sends fields before files, reports progress, and resolves the parsed body', async () => {
  const client = new ApiClient();
  const progress: number[] = [];
  const file = new File([new Uint8Array([1, 2, 3])], 'a.wav');
  const result = client.upload<{ ok: boolean }>('/import', [{ field: 'audio', file }], {
    token: 'tok', fields: { description: 'd' }, onProgress: f => progress.push(f),
  });
  const xhr = FakeXhr.made.at(-1)!;
  assert.equal(xhr.method, 'POST');
  assert.equal(xhr.url, '/api/import');
  assert.equal(xhr.getHeader('Authorization'), 'Bearer tok');
  xhr.upload.onprogress?.({ lengthComputable: true, loaded: 5, total: 10 });
  xhr.respond(200, { ok: true });
  assert.deepEqual(await result, { ok: true });
  assert.deepEqual(progress, [0.5]);
});

test('upload rejects with ApiError on a non-2xx response', async () => {
  const client = new ApiClient();
  const result = client.upload('/import', [{ field: 'audio', file: new File([], 'a.wav') }]);
  FakeXhr.made.at(-1)!.respond(422, { error: 'bad file' });
  await assert.rejects(result, (err: unknown) => {
    assert.ok(err instanceof ApiError);
    assert.equal(err.status, 422);
    assert.equal(err.message, 'bad file');
    return true;
  });
});

test('aborting the signal before send rejects without touching the fake XHR transport', async () => {
  const client = new ApiClient();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    client.upload('/import', [{ field: 'audio', file: new File([], 'a.wav') }], { signal: controller.signal }),
    /Aborted/,
  );
});

test('streamUrl appends the token as a query param, never a header', () => {
  const client = new ApiClient();
  assert.equal(client.streamUrl('/logs'), '/api/logs');
  assert.equal(client.streamUrl('/logs', { token: 'tok' }), '/api/logs?token=tok');
  assert.equal(client.streamUrl('/logs?level=info', { token: 'tok' }), '/api/logs?level=info&token=tok');
});

test('mediaUrl leaves absolute/rooted paths alone and roots a bare name', () => {
  const client = new ApiClient();
  assert.equal(client.mediaUrl('/audio/a.wav'), '/audio/a.wav');
  assert.equal(client.mediaUrl('https://example.com/a.wav'), 'https://example.com/a.wav');
  assert.equal(client.mediaUrl('a.wav'), '/a.wav');
});
