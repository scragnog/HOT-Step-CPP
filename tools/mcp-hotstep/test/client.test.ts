// client.test.ts — one real MCP round trip over the SDK's in-memory
// transport: tools list, a submit, a cancelled wait. The HTTP-level detail
// (login/retry, exact per-backend bodies, rejections) is test/http.test.ts;
// this file only checks that server.ts wires tools.ts up correctly end to end.

import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import type { AddressInfo } from 'node:net';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import type { CallToolResult } from '@modelcontextprotocol/sdk/types.js';

let fixtureServer: http.Server;
let client: Client;

before(async () => {
  // A minimal fixture: login, a generate submit that always accepts, and a
  // status endpoint that never reaches a terminal state (for the wait test).
  fixtureServer = http.createServer((req, res) => {
    if (req.url === '/api/auth/auto' && req.method === 'GET') {
      res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ user: { id: 'u1' }, token: 'tok-1' }));
      return;
    }
    if (req.url === '/api/generate' && req.method === 'POST') {
      res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ jobId: 'job-client-test', status: 'pending' }));
      return;
    }
    if (req.url?.startsWith('/api/generate/status/')) {
      res.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({ jobId: 'job-client-test', status: 'running' }));
      return;
    }
    res.writeHead(404).end('{}');
  });
  await new Promise<void>(resolve => fixtureServer.listen(0, '127.0.0.1', resolve));
  process.env.HOTSTEP_URL = `http://127.0.0.1:${(fixtureServer.address() as AddressInfo).port}`;

  const { createServer } = await import('../src/server.js');
  const server: McpServer = createServer();

  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  client = new Client({ name: 'mcp-hotstep-test-client', version: '1.0.0' });
  await Promise.all([client.connect(clientTransport), server.connect(serverTransport)]);
});

after(async () => {
  await client.close();
  await new Promise(resolve => fixtureServer.close(resolve));
});

function firstText(result: CallToolResult): string {
  const block = result.content[0];
  assert.equal(block?.type, 'text');
  return (block as { type: 'text'; text: string }).text;
}

test('lists all eight generation tools', async () => {
  const { tools } = await client.listTools();
  const names = tools.map(t => t.name).sort();
  assert.deepEqual(names, [
    'gen_backends', 'gen_cancel', 'gen_configure', 'gen_queue',
    'gen_song', 'gen_status', 'gen_submit', 'gen_wait',
  ]);
});

test('gen_submit over the wire returns the job the fixture server hands back', async () => {
  const result = await client.callTool({
    name: 'gen_submit',
    arguments: { backend: 'ace', caption: 'a test caption' },
  }) as CallToolResult;
  assert.equal(result.isError, undefined);
  const data = JSON.parse(firstText(result));
  assert.deepEqual(data, { jobId: 'job-client-test', status: 'pending' });
});

test('gen_wait, cancelled from the client side, does not ride out its 30s budget', { timeout: 5000 }, async () => {
  const ac = new AbortController();
  setTimeout(() => ac.abort(), 100);
  const start = Date.now();
  const result = await client.callTool(
    { name: 'gen_wait', arguments: { jobId: 'job-client-test', maxSeconds: 30 } },
    undefined,
    { signal: ac.signal },
  ).catch((err: Error) => {
    // The SDK surfaces a cancelled call as a rejected promise (it sends
    // notifications/cancelled to the server, then throws locally on its own
    // aborted signal) rather than a normal result — either way is "didn't
    // hang for 30s", which is what this test is actually checking.
    assert.match(err.message, /cancel|abort/i);
    return null;
  });
  assert.ok(Date.now() - start < 2000, 'should not wait anywhere near the 30s budget');
  if (result) {
    const data = JSON.parse(firstText(result as CallToolResult));
    assert.equal(data.status, 'running');
  }
});
