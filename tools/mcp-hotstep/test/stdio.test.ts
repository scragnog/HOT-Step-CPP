// stdio.test.ts — a REAL stdio transport, not the in-memory one client.test.ts
// uses. This is the only test in the package that can catch "something wrote
// a stray non-JSON byte to stdout before/during the handshake" — the in-memory
// transport never touches process.stdout at all, which is exactly why the
// config.ts bootstrap-logging bug (src/tools.ts no longer imports it) was
// invisible to every other test here.

import assert from 'node:assert/strict';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { getDefaultEnvironment, StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

const packageDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');

test('starts cleanly over a real stdio transport with a non-loopback HOTSTEP_URL', { timeout: 15000 }, async () => {
  const client = new Client({ name: 'mcp-hotstep-stdio-test', version: '1.0.0' });
  try {
    await client.connect(new StdioClientTransport({
      command: process.execPath,
      args: [...process.execArgv.filter(arg => arg.startsWith('--preserve-symlinks')), '--import', 'tsx', 'src/index.ts'],
      cwd: packageDir,
      // Non-loopback on purpose (192.0.2.1 is RFC 5737 TEST-NET-1 — reserved,
      // never routable, never dialled): exercises hotstepIsLoopback() === false
      // so gen_song's filesystem-path branch is live code here too, and proves
      // the handshake stays clean regardless of HOTSTEP_URL.
      env: { ...getDefaultEnvironment(), HOTSTEP_URL: 'http://192.0.2.1:3001' },
      stderr: 'inherit',
    }));
    const { tools } = await client.listTools();
    assert.equal(tools.length, 20, `expected all 20 tools, got: ${tools.map(t => t.name).join(', ')}`);
  } finally {
    await client.close();
  }
});
