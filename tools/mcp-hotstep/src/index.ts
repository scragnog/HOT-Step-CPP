// index.ts — stdio entrypoint. Tool registrations live in ./server.ts (kept
// separate so a test can create a server and connect it in-memory without
// also starting a real stdio transport on process stdin/stdout).

import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { createServer } from './server.js';
import { hotstepBaseUrl } from './http.js';

async function main() {
  const server = createServer();
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`[mcp-hotstep] Server started (HOTSTEP_URL=${hotstepBaseUrl()})`);
}

main().catch((err) => {
  console.error('[mcp-hotstep] Fatal error:', err);
  process.exit(1);
});
