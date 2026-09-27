import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import { once } from 'node:events';
import { get as httpGet } from 'node:http';
import test from 'node:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport, getDefaultEnvironment } from '@modelcontextprotocol/sdk/client/stdio.js';
import { fileURLToPath } from 'node:url';
import { DiscussionStore } from '../src/collaboration.js';
import { createDiscussionViewer } from '../src/discussion-viewer.js';

test('group chat HTTP and MCP share the transcript without app access', { timeout: 15000 }, async t => {
  const temp = mkdtempSync(join(tmpdir(), 'hotstep-viewer-test-'));
  const dbPath = join(temp, 'collaboration.db');
  const server = createDiscussionViewer(dbPath);
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const base = `http://127.0.0.1:${address.port}`;
  let store: DiscussionStore | undefined;
  const client = new Client({ name: 'viewer-test-agent', version: '1.0.0' });
  const participant = randomUUID();
  const create = (value: Record<string, unknown>, origin = base) => fetch(base + '/api/discussions', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Origin: origin }, body: JSON.stringify(value),
  });
  const write = (endpoint: string, value: Record<string, unknown>, origin = base) => fetch(`${base}/api/discussions/review/${endpoint}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', Origin: origin },
    body: JSON.stringify({ participant_id: participant, request_id: randomUUID(), ...value }),
  });
  try {
    await t.test('an empty viewer serves its UI and does not create a database', async () => {
      const index = await fetch(base);
      assert.equal(index.status, 200);
      assert.match(await index.text(), /Your message to both agents/);
      assert.match(index.headers.get('content-security-policy')!, /default-src 'self'/);
      for (const asset of ['/viewer.js', '/viewer.css', '/transcript-print.js', '/transcript-print.css']) assert.equal((await fetch(base + asset)).status, 200);
      assert.equal((await fetch(base + '/api/discussions/missing/transcript.html')).status, 404);
      assert.deepEqual(await (await fetch(base + '/api/discussions')).json(), { discussions: [] });
      assert.equal(existsSync(dbPath), false);
    });
    await t.test('room creation validates input before creating a database', async () => {
      const input = { room: 'browser-room', brief: 'Review compatibility.', participant_id: randomUUID(), request_id: randomUUID() };
      assert.equal((await create({ ...input, room: 'bad room name' })).status, 400);
      assert.equal((await create({ ...input, brief: ' ' })).status, 400);
      assert.equal((await create(input, 'https://example.com')).status, 403);
      assert.equal(existsSync(dbPath), false);
    });
    await t.test('a human creates the first room with a brief, and retries preserve it', async () => {
      const input = { room: 'browser-room', brief: 'Review compatibility.', participant_id: randomUUID(), request_id: randomUUID() };
      const created = await create(input);
      assert.equal(created.status, 200);
      const result = await created.json() as any;
      assert.equal(result.discussion.id, input.room);
      assert.equal(result.message.author, 'You');
      assert.equal(result.message.body, input.brief);
      const retry = await create(input);
      assert.equal(retry.status, 200);
      assert.deepEqual(await retry.json(), result);
      assert.equal((await create({ ...input, brief: 'Overwrite the brief' })).status, 409);
      assert.equal((await create({ ...input, request_id: randomUUID() })).status, 409);
      const page = await (await fetch(base + '/api/discussions/browser-room')).json() as any;
      assert.equal(page.messages.length, 1);
      assert.equal(page.participants.length, 1);
      assert.equal(page.discussion.brief, input.brief);
      const exported = await fetch(base + '/api/discussions/browser-room/transcript.html');
      assert.equal(exported.status, 200); // No recorded plan is required.
      const html = await exported.text();
      assert.match(html, /Review compatibility\./);
      assert.match(html, /Unrevealed positions are sealed/);
      assert.match(html, /Save as PDF/);
    });
    await t.test('simultaneous creation cannot overwrite a room', async () => {
      const input = { room: 'same-name', brief: 'Original brief', participant_id: randomUUID(), request_id: randomUUID() };
      const responses = await Promise.all([
        create(input), create({ ...input, brief: 'Competing brief', participant_id: randomUUID(), request_id: randomUUID() }),
      ]);
      assert.deepEqual(responses.map(response => response.status).sort(), [200, 409]);
      const page = await (await fetch(base + '/api/discussions/same-name')).json() as any;
      assert.equal(page.participants.length, 1);
      assert.equal(page.messages.length, 1);
      assert.equal(page.messages[0].body, page.discussion.brief);
    });
    store = new DiscussionStore(dbPath);
    const agent = store.join('review', 'Codex', 'Review the design');
    await client.connect(new StdioClientTransport({
      command: process.execPath,
      args: [...process.execArgv.filter(arg => arg.startsWith('--preserve-symlinks')), '--import', 'tsx', 'src/collaboration-server.ts'],
      cwd: fileURLToPath(new URL('..', import.meta.url)),
      env: { ...getDefaultEnvironment(), HOTSTEP_COLLAB_DB: dbPath },
      stderr: 'inherit',
    }));

    await t.test('MCP agents can discover and join a room created in the browser', async () => {
      const listed = await client.callTool({ name: 'collab_list_discussions', arguments: {} });
      const rooms = JSON.parse((listed.content as { text: string }[])[0].text);
      assert.ok(rooms.some((room: { id: string }) => room.id === 'browser-room'));
      const joined = await client.callTool({ name: 'collab_join_discussion', arguments: { room: 'browser-room', name: 'Codex' } });
      assert.equal(joined.isError, undefined);
      const result = JSON.parse((joined.content as { text: string }[])[0].text);
      assert.equal(result.discussion.brief, 'Review compatibility.');
    });

    await t.test('a human post reaches a waiting MCP agent and retry does not duplicate it', async () => {
      const waiting = client.callTool({ name: 'collab_wait_for_message', arguments: { room: 'review', after_id: 0, timeout_ms: 2000 } });
      const request_id = randomUUID();
      const body = 'Can we keep this backwards compatible? <script>alert(1)</script>';
      const response = await write('messages', { body, request_id });
      assert.equal(response.status, 200);
      const sent = await response.json() as { id: number; body: string; author: string; kind: string };
      assert.equal(sent.author, 'You');
      assert.equal(sent.kind, 'user_direction');
      assert.equal(sent.body, body);
      const result = await waiting;
      const received = JSON.parse((result.content as { text: string }[])[0].text);
      assert.equal(received.messages[0].id, sent.id);
      const retry = await write('messages', { body, request_id });
      assert.equal((await retry.json() as { id: number }).id, sent.id);
      assert.equal((await write('messages', { body: 'different', request_id })).status, 409);
      assert.equal(store!.read('review', 0, 100).messages.length, 1);
    });
    await t.test('agent replies appear through HTTP, including pagination and proposed decisions', async () => {
      const first = store!.read('review', 0, 100).messages[0];
      const peer = store!.join('review', 'Claude');
      for (let i = 0; i < 105; i++) store!.post('review', i % 2 ? peer.participant_id : agent.participant_id, `reply-${i}`, 'reply', `Reply ${i}`, first.id);
      store!.decide('review', peer.participant_id, 'plan', 0, 'Keep compatibility', 'Check the old format');
      const page = await (await fetch(base + '/api/discussions/review')).json() as any;
      assert.equal(page.messages.length, 100);
      assert.equal(page.has_more, true);
      assert.equal(page.decision.plan, 'Keep compatibility');
      const rest = await (await fetch(`${base}/api/discussions/review?after_id=${page.next_after_id}`)).json() as any;
      assert.equal(rest.messages.length, 7);
      assert.equal(rest.has_more, false);
      const list = await (await fetch(base + '/api/discussions')).json() as any;
      assert.equal(list.discussions[0].id, 'review');
      const readonly = new DiscussionStore(dbPath, { readonly: true });
      try { assert.throws(() => readonly.post('review', agent.participant_id, 'readonly', 'reply', 'No write'), /readonly/i); }
      finally { readonly.close(); }
    });
    await t.test('Markdown download exports the latest plan and disagreements without writing storage', async () => {
      const before = store!.read('review', 0, 1000);
      const download = await fetch(base + '/api/discussions/review/plan.md');
      assert.equal(download.status, 200);
      assert.equal(download.headers.get('content-type'), 'text/markdown; charset=utf-8');
      assert.match(download.headers.get('content-disposition')!, /review-r1.md/);
      const markdown = await download.text();
      assert.match(markdown, /Revision: 1/);
      assert.match(markdown, /Keep compatibility/);
      assert.match(markdown, /Check the old format/);
      assert.match(markdown, /not user approval/);
      assert.doesNotMatch(markdown, /Reply 104/);
      assert.deepEqual(store!.read('review', 0, 1000), before);
      assert.equal((await fetch(base + '/api/discussions/browser-room/plan.md')).status, 404);
      assert.equal((await fetch(base + '/api/discussions/missing/plan.md')).status, 404);
      assert.equal((await write('plan.md', { body: 'Cannot mutate by export' })).status, 404);
      store!.decide('review', agent.participant_id, 'revised-plan', 1, 'Updated plan\\n\\n1. Preserve café vocals.\\n2. Keep timing.', 'None');
      const updated = await fetch(base + '/api/discussions/review/plan.md');
      assert.match(updated.headers.get('content-disposition')!, /review-r2.md/);
      const revised = await updated.text();
      assert.match(revised, /Updated plan\n\n1\. Preserve café vocals\.\n2\./);
      assert.doesNotMatch(revised, /Keep compatibility/);
    });
    await t.test('PDF print export includes the whole room and full plans without changing it', async () => {
      const before = store!.exportTranscript('review');
      const response = await fetch(base + '/api/discussions/review/transcript.html?after_id=999999');
      assert.equal(response.status, 200);
      assert.equal(response.headers.get('content-type'), 'text/html; charset=utf-8');
      assert.equal(response.headers.get('cache-control'), 'no-store');
      assert.match(response.headers.get('content-security-policy')!, /default-src 'self'/);
      const html = await response.text();
      assert.equal((html.match(/<article class="message"/g) || []).length, before.messages.length);
      assert.ok(before.messages.length > 100);
      assert.match(html, /Reply 0</);
      assert.match(html, /Reply 104</);
      assert.match(html, /Plan revision 1/);
      assert.match(html, /Keep compatibility/);
      assert.match(html, /Check the old format/);
      assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
      assert.doesNotMatch(html, /<script>alert/);
      assert.doesNotMatch(html, /browser-room/);
      assert.deepEqual(store!.exportTranscript('review').messages, before.messages);
      assert.deepEqual(store!.exportTranscript('review').discussion, before.discussion);
      assert.equal((await fetch(base + '/api/discussions/missing/transcript.html')).status, 404);
      assert.equal((await write('transcript.html', { body: 'Not a write endpoint' })).status, 404);
      assert.equal((await fetch(base + '/api/discussions/review/transcript.html', { headers: { Origin: 'https://example.com' } })).status, 403);
    });
    await t.test('human pause wakes agents, rejects further messages, and resume restores posting', async () => {
      const cursor = store!.read('review', 0, 1000).next_after_id;
      const waiting = client.callTool({ name: 'collab_wait_for_message', arguments: { room: 'review', after_id: cursor, timeout_ms: 2000 } });
      assert.equal((await write('status', { status: 'paused', body: 'Hold on, I want to rethink this.' })).status, 200);
      const received = JSON.parse(((await waiting).content as { text: string }[])[0].text);
      assert.equal(received.discussion.status, 'paused');
      assert.equal((await fetch(base + '/api/discussions/review/transcript.html')).status, 200);
      assert.equal((await write('messages', { body: 'Paused post' })).status, 409);
      assert.equal((await write('status', { status: 'active', body: 'Continue.' })).status, 200);
      assert.equal((await write('messages', { body: 'Consider the simpler option.' })).status, 200);
    });
    await t.test('invalid input, other origins, and agent impersonation are rejected', async () => {
      assert.equal((await write('messages', { body: 'Wrong origin' }, 'https://example.com')).status, 403);
      assert.equal((await write('messages', { body: ' ' })).status, 400);
      assert.equal((await write('messages', { body: 'x'.repeat(24001) })).status, 400);
      assert.equal((await write('messages', { participant_id: agent.participant_id, body: 'Pretend to be Codex' })).status, 409);
      assert.equal((await create({ room: 'rollback-test', brief: 'Must roll back', participant_id: agent.participant_id, request_id: randomUUID() })).status, 409);
      assert.equal((await fetch(base + '/api/discussions/rollback-test')).status, 404);
      assert.equal((await fetch(base + '/api/discussions/review?after_id=-1')).status, 400);
      assert.equal((await fetch(base + '/api/discussions/missing')).status, 404);
      assert.equal((await fetch(base + '/api/discussions', { method: 'DELETE' })).status, 405);
      const foreignHostStatus = await new Promise<number | undefined>((resolve, reject) => {
        httpGet(base + '/api/discussions', { headers: { Host: 'example.com' } }, response => {
          response.resume();
          resolve(response.statusCode);
        }).on('error', reject);
      });
      assert.equal(foreignHostStatus, 403);
    });
    await t.test('human pings and research controls share state with the agents', async () => {
      assert.equal((await write('messages', { body: '@codex check the current implementation.' })).status, 200);
      store!.activity('review', agent.participant_id, 'researching', 'Reading the relevant source.', 120);
      const page = await (await fetch(base + '/api/discussions/review')).json() as any;
      assert.equal(page.coordination.research.name, 'Codex');
      assert.equal(page.coordination.requests[0].participant_id, agent.participant_id);
      const released = await write('coordination', { body: 'Release', action: 'release_research' });
      assert.equal(released.status, 200);
      assert.equal(store!.read('review', 0, 1000).coordination.research, null);
      assert.equal((await write('coordination', { body: 'Clear pings', action: 'clear_requests' })).status, 200);
      assert.equal(store!.read('review', 0, 1000).coordination.requests.length, 0);
      assert.equal((await write('coordination', { body: 'Missing action' })).status, 400);
      const script = await (await fetch(base + '/viewer.js')).text();
      assert.match(script, /A Claude session started with --channels wakes on its own/);
    });
    await t.test('End Discussion closes through HTTP and releases an MCP waiter', async () => {
      const html = await (await fetch(base + '/')).text();
      assert.match(html, /id="end-discussion"[^>]*>End Discussion/);
      const script = await (await fetch(base + '/viewer.js')).text();
      assert.match(script, /byId\('end-discussion'\)\.addEventListener\('click'.*'closed'/);
      assert.match(script, /Consensus reached/);
      const cursor = store!.read('review', 0, 1000).next_after_id;
      const waiting = client.callTool({ name: 'collab_wait_for_message', arguments: { room: 'review', after_id: cursor, timeout_ms: 2000 } });
      assert.equal((await write('status', { status: 'closed', body: 'User ended the discussion.' })).status, 200);
      const received = JSON.parse(((await waiting).content as { text: string }[])[0].text);
      assert.equal(received.discussion.status, 'closed');
      assert.equal(received.timed_out, false);
      assert.equal((await write('messages', { body: 'After ending' })).status, 409);
      const ended = await (await fetch(base + '/api/discussions/review')).json() as any;
      assert.equal(ended.consensus.reached, false);
      assert.ok(ended.decision);
      const exported = await (await fetch(base + '/api/discussions/review/transcript.html')).text();
      assert.match(exported, /User ended the discussion/);
      assert.match(exported, /Codex/); // History survives expired/cleared presence.
    });
  } finally {
    await client.close();
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    store?.close();
    const target = resolve(temp);
    assert.ok(target.startsWith(resolve(tmpdir()) + sep) && target.includes('hotstep-viewer-test-'));
    rmSync(target, { recursive: true, force: true });
  }
});
