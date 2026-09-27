import assert from 'node:assert/strict';
import { randomUUID, webcrypto } from 'node:crypto';
import { once } from 'node:events';
import { get as httpGet } from 'node:http';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { runInNewContext } from 'node:vm';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { DiscussionStore } from '../src/collaboration.js';
import { createDiscussionHttp } from '../src/discussion-http.js';

test('LAN browser can create request IDs without secure-context randomUUID', () => {
  const source = readFileSync(new URL('../viewer/viewer.js', import.meta.url), 'utf8');
  const helper = source.slice(source.indexOf('function requestId()'), source.indexOf('function saved('));
  const context = { crypto: { getRandomValues: webcrypto.getRandomValues.bind(webcrypto) } };
  const first = runInNewContext(helper + '; requestId()', context);
  assert.match(first, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  assert.notEqual(first, runInNewContext(helper + '; requestId()', context));
});

test('network clients share rooms, wait, agree and disconnect independently', { timeout: 60000 }, async () => {
  const temp = mkdtempSync(join(tmpdir(), 'hotstep-network-'));
  const dbPath = join(temp, 'discussion.db');
  const token = randomUUID();
  const { server, closeSessions } = createDiscussionHttp({ dbPath, token, allowedHostnames: ['127.0.0.1'] });
  const clients: Client[] = [];
  const transports: StreamableHTTPClientTransport[] = [];
  const store = new DiscussionStore(dbPath);
  try {
    server.listen(0, '127.0.0.1');
    await once(server, 'listening');
    const address = server.address();
    assert.ok(address && typeof address !== 'string');
    const base = `http://127.0.0.1:${address.port}`;
    const headers = { Authorization: `Bearer ${token}` };
    assert.equal((await fetch(base)).status, 401);
    assert.equal((await fetch(base + '/api/discussions/network-test/transcript.html')).status, 401);
    assert.equal((await fetch(base + '/mcp')).status, 401);
    assert.equal((await fetch(base + '/mcp', { headers: { Authorization: 'Bearer wrong' } })).status, 401);
    assert.equal((await fetch(base, { headers: { ...headers, Origin: 'http://evil.test' } })).status, 403);
    const badHost = await new Promise<number | undefined>((resolve, reject) => {
      httpGet(base, { headers: { ...headers, Host: `evil.test:${address.port}` } }, response => {
        response.resume(); resolve(response.statusCode);
      }).on('error', reject);
    });
    assert.equal(badHost, 403);
    const browserHeaders = { Authorization: `Basic ${Buffer.from(`discussion:${token}`).toString('base64')}` };
    assert.match(await (await fetch(base, { headers: browserHeaders })).text(), /Your message to both agents/);
    assert.equal((await fetch(base + '/mcp', { headers: { ...headers, 'mcp-session-id': 'unknown' } })).status, 404);
    assert.equal((await fetch(base + '/mcp', { headers })).status, 400);
    assert.equal((await fetch(base + '/mcp', { method: 'POST', headers: { ...headers, 'Content-Type': 'application/json' }, body: '{' })).status, 400);
    assert.equal((await fetch(base + '/mcp', { method: 'POST', headers: { ...headers, 'Content-Type': 'application/json' }, body: JSON.stringify({ junk: 'x'.repeat(130000) }) })).status, 400);
    const connect = async (name: string) => {
      const transport = new StreamableHTTPClientTransport(new URL(base + '/mcp'), { requestInit: { headers } });
      const client = new Client({ name, version: '1.0.0' });
      clients.push(client); transports.push(transport);
      await client.connect(transport);
      return client;
    };
    const call = async (client: Client, name: string, args: Record<string, unknown>) => {
      const result = await client.callTool({ name, arguments: args });
      assert.ok(!result.isError, JSON.stringify(result));
      return JSON.parse((result.content as { text: string }[])[0].text);
    };
    const a = await connect('remote-a');
    const b = await connect('remote-b');
    assert.notEqual(transports[0].sessionId, transports[1].sessionId);
    assert.equal((await a.listTools()).tools.filter(tool => tool.name.startsWith('collab_')).length, 13);
    assert.equal((await a.listTools()).tools.filter(tool => tool.name.startsWith('work_')).length, 6);
    assert.ok((await a.listTools()).tools.every(tool => /^(collab_|work_)/.test(tool.name)));
    const room = 'network-test';
    const pa = (await call(a, 'collab_join_discussion', { room, name: 'Remote A', brief: 'Transport test only.' })).participant_id;
    const pb = (await call(b, 'collab_join_discussion', { room, name: 'Remote B' })).participant_id;
    const exported = await fetch(base + '/api/discussions/network-test/transcript.html', { headers: browserHeaders });
    assert.equal(exported.status, 200);
    assert.match(await exported.text(), /Transport test only\./);
    assert.notEqual(pa, pb);
    // Cursor/presence defaults belong to each HTTP session, not the latest join globally.
    let page = await call(a, 'collab_read_discussion', { room, after_id: 0 });
    assert.equal(page.has_more, false);
    const pending = call(a, 'collab_wait_for_message', { room, after_id: page.next_after_id, timeout_ms: 10000 });
    await new Promise(resolve => setTimeout(resolve, 100));
    const local = store.join(room, 'Local agent').participant_id;
    const message = store.post(room, local, randomUUID(), 'proposal', 'Message from the shared local database.');
    page = await pending;
    assert.equal(page.messages.at(-1).id, message.id);
    const empty = await call(a, 'collab_wait_for_message', { room, after_id: page.next_after_id, timeout_ms: 25000 });
    assert.equal(empty.timed_out, true);
    assert.equal(empty.discussion.status, 'active');
    assert.equal((await fetch(base + '/api/discussions/network-test', { headers: browserHeaders }).then(r => r.json())).messages.at(-1).id, message.id);
    // Complete the same consensus protocol used by stdio clients.
    store.leave(room, local);
    const decision = await call(a, 'collab_record_decision', { room, participant_id: pa, request_id: randomUUID(), expected_revision: 0, read_after_id: page.next_after_id, plan: 'No implementation. Transport check complete.' });
    page = await call(a, 'collab_read_discussion', { room, after_id: page.next_after_id });
    await call(a, 'collab_agree_plan', { room, participant_id: pa, request_id: randomUUID(), revision: decision.revision, read_after_id: page.next_after_id });
    page = await call(b, 'collab_read_discussion', { room, after_id: 0 });
    const closed = await call(b, 'collab_agree_plan', { room, participant_id: pb, request_id: randomUUID(), revision: decision.revision, read_after_id: page.next_after_id });
    assert.equal(closed.discussion.status, 'closed');
    assert.equal(closed.consensus.reached, true);
    const p2 = (await call(a, 'collab_join_discussion', { room: 'disconnect', name: 'Remote A', brief: 'Presence test.' })).participant_id;
    await call(b, 'collab_join_discussion', { room: 'disconnect', name: 'Remote B' });
    const oldSession = transports[0].sessionId!;
    await transports[0].terminateSession();
    assert.equal(store.read('disconnect', 0, 100).participants.some(p => p.id === p2), false);
    assert.equal(store.read('disconnect', 0, 100).participants.some(p => p.name === 'Remote B'), true);
    assert.equal((await fetch(base + '/mcp', { headers: { ...headers, 'mcp-session-id': oldSession } })).status, 404);
    page = await call(b, 'collab_read_discussion', { room: 'disconnect', after_id: 0 });
    const closingWait = call(b, 'collab_wait_for_message', { room: 'disconnect', after_id: page.next_after_id, timeout_ms: 10000 });
    await new Promise(resolve => setTimeout(resolve, 100));
    const end = await fetch(base + '/api/discussions/disconnect/status', {
      method: 'POST', headers: { ...browserHeaders, Origin: base, 'Content-Type': 'application/json' },
      body: JSON.stringify({ participant_id: randomUUID(), request_id: randomUUID(), status: 'closed', body: 'End Discussion from remote browser.' }),
    });
    assert.equal(end.status, 200);
    assert.equal((await closingWait).discussion.status, 'closed');
    await transports[1].terminateSession();
  } finally {
    await Promise.all(clients.map(client => client.close()));
    await closeSessions();
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
    store.close();
    rmSync(temp, { recursive: true, force: true });
  }
});
