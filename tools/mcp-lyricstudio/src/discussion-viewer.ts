import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { z } from 'zod';
import { DEFAULT_COLLAB_DB, DEFAULT_MAX_ROUNDS, DiscussionStore } from './collaboration.js';
import { handleWorkRequest } from './work-viewer.js';
import { renderDiscussionExport } from './discussion-export.js';

export async function readJson(request: IncomingMessage) {
  return new Promise<unknown>((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    request.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size <= 128000) chunks.push(chunk);
    });
    request.on('error', reject);
    request.on('aborted', () => reject(new Error('Request cancelled.')));
    request.on('end', () => {
      if (size > 128000) { reject(new Error('Message is too large.')); return; }
      try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); }
      catch { reject(new Error('Invalid JSON.')); }
    });
  });
}

const humanWrite = z.object({
  participant_id: z.string().uuid(),
  request_id: z.string().uuid(),
  body: z.string().trim().min(1).max(24000),
  status: z.enum(['active', 'paused', 'closed']).optional(),
  action: z.enum(['release_research', 'clear_requests', 'reveal_positions']).optional(),
  outcome: z.enum(['shipped', 'partial', 'abandoned', 'superseded']).optional(),
  commit: z.string().trim().max(100).optional(),
});
const humanCreate = humanWrite.pick({ participant_id: true, request_id: true }).extend({
  room: z.string().trim().min(1).max(100).regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]*$/),
  brief: z.string().trim().min(1).max(24000),
  blind_positions: z.boolean().default(true),
  max_rounds: z.number().int().min(1).max(6).default(DEFAULT_MAX_ROUNDS),
});

// Local group chat. Reads use read-only SQLite connections; explicit human
// posts and status changes write only to the separate collaboration database.
export function createDiscussionViewer(dbPath = process.env.HOTSTEP_COLLAB_DB ?? DEFAULT_COLLAB_DB, options: {
  allowedHostnames?: string[];
  handleRequest?: (request: IncomingMessage, response: ServerResponse) => Promise<boolean>;
} = {}) {
  const assetFiles: Record<string, { type: string; file: string }> = {
    '/': { type: 'text/html', file: 'index.html' },
    '/viewer.js': { type: 'text/javascript', file: 'viewer.js' },
    '/viewer.css': { type: 'text/css', file: 'viewer.css' },
    '/transcript-print.css': { type: 'text/css', file: 'transcript-print.css' },
    '/transcript-print.js': { type: 'text/javascript', file: 'transcript-print.js' },
  };
  const assets = new Map(Object.entries(assetFiles).map(([url, asset]) => [
    url, { type: asset.type, body: readFileSync(new URL(`../viewer/${asset.file}`, import.meta.url)) },
  ]));
  // Read-only page loads cannot migrate an older database; one writer open at
  // startup adds the sealed-position columns before the first GET needs them.
  if (existsSync(dbPath)) {
    try { new DiscussionStore(dbPath).close(); }
    catch { /* A locked or foreign database is reported per request below. */ }
  }
  const server = createServer(async (request, response) => {
    response.setHeader('Cache-Control', 'no-store');
    response.setHeader('X-Content-Type-Options', 'nosniff');
    response.setHeader('Content-Security-Policy', "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");
    const send = (status: number, value: unknown) => {
      response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
      response.end(JSON.stringify(value));
    };
    const address = server.address();
    const port = address && typeof address !== 'string' ? address.port : 0;
    const hosts = (options.allowedHostnames ?? ['127.0.0.1', 'localhost']).map(host => `${host}:${port}`);
    if (!hosts.includes(request.headers.host ?? '') || (request.headers.origin && request.headers.origin !== `http://${request.headers.host}`)) {
      send(403, { error: 'Use the configured discussion address and a same-origin page.' }); return;
    }
    if (options.handleRequest) {
      try { if (await options.handleRequest(request, response)) return; }
      catch {
        if (!response.headersSent) send(500, { error: 'Discussion request failed.' });
        else response.end();
        return;
      }
    }
    if (request.method !== 'GET' && request.method !== 'POST') {
      response.setHeader('Allow', 'GET, POST');
      send(405, { error: 'Method not allowed.' }); return;
    }
    if (request.method === 'POST' && (request.headers.origin !== `http://${request.headers.host}` || !request.headers['content-type']?.startsWith('application/json'))) {
      send(403, { error: 'Send messages from the discussion page.' }); return;
    }
    if (await handleWorkRequest(request, response, dbPath)) return;
    let store: DiscussionStore | undefined;
    try {
      const url = new URL(request.url ?? '/', `http://127.0.0.1:${port}`);
      const asset = assets.get(url.pathname);
      if (asset && request.method === 'GET') {
        response.writeHead(200, { 'Content-Type': `${asset.type}; charset=utf-8` });
        response.end(asset.body); return;
      }
      const roomMatch = /^\/api\/discussions\/([a-zA-Z0-9][a-zA-Z0-9._-]{0,99})(?:\/(messages|status|coordination|outcome|plan\.md|transcript\.html))?$/.exec(url.pathname);
      const isWrite = request.method === 'POST';
      const isCreate = isWrite && url.pathname === '/api/discussions';
      const isDownload = roomMatch?.[2] === 'plan.md' || roomMatch?.[2] === 'transcript.html';
      if ((!roomMatch && url.pathname !== '/api/discussions') || (roomMatch && isWrite !== Boolean(roomMatch[2] && !isDownload))) {
        send(404, { error: 'Not found.' }); return;
      }
      const after = url.searchParams.get('after_id') ?? '0';
      if (!/^\d+$/.test(after) || !Number.isSafeInteger(Number(after))) {
        send(400, { error: 'after_id must be a nonnegative safe integer.' }); return;
      }
      if (!existsSync(dbPath) && !isCreate) {
        send(roomMatch ? 404 : 200, roomMatch ? { error: 'Discussion not found. Create it from the discussion page first.' } : { discussions: [] }); return;
      }
      if (isCreate) {
        let input: z.infer<typeof humanCreate>;
        try { input = humanCreate.parse(await readJson(request)); }
        catch { send(400, { error: 'Enter a room name (letters, numbers, dots, underscores or hyphens; max 100 characters) and a brief (1 to 24,000 characters), with valid request and participant IDs.' }); return; }
        store = new DiscussionStore(dbPath);
        try { send(200, store.createViewerDiscussion(input.room, input.brief, input.participant_id, input.request_id, { blind_positions: input.blind_positions, max_rounds: input.max_rounds })); }
        catch (error) { send(409, { error: error instanceof Error ? error.message : 'Unable to create the discussion.' }); }
        return;
      }
      // Parse before opening a connection; a slow browser must not hold a DB handle.
      let input: z.infer<typeof humanWrite> | undefined;
      if (isWrite) {
        try { input = humanWrite.parse(await readJson(request)); }
        catch { send(400, { error: 'Enter a message between 1 and 24,000 characters with valid request and participant IDs.' }); return; }
      }
      store = new DiscussionStore(dbPath, { readonly: !isWrite });
      if (!roomMatch) { send(200, { discussions: store.list(100) }); return; }
      const room = roomMatch[1];
      try { store.room(room); }
      catch (error) {
        if (error instanceof Error && error.message.startsWith('Unknown discussion:')) {
          send(404, { error: 'Discussion not found.' }); return;
        }
        throw error;
      }
      if (input) {
        try {
          const participant = store.joinViewer(room, input.participant_id);
          if (roomMatch[2] === 'coordination') {
            if (!input.action) { send(400, { error: 'Choose a coordination action.' }); return; }
            send(200, store.clearCoordination(room, participant, input.request_id, input.action));
          } else if (roomMatch[2] === 'status') {
            if (!input.status) { send(400, { error: 'Choose a discussion status.' }); return; }
            send(200, store.status(room, participant, input.request_id, input.status, input.body));
          } else if (roomMatch[2] === 'outcome') {
            if (!input.outcome) { send(400, { error: 'Choose an outcome.' }); return; }
            send(200, store.recordOutcome(room, participant, input.request_id, input.outcome, input.body, input.commit));
          } else {
            // The browser cannot choose an agent identity, kind, or decision revision.
            send(200, store.post(room, participant, input.request_id, 'user_direction', input.body));
          }
        } catch (error) {
          send(409, { error: error instanceof Error ? error.message : 'Unable to post. Refresh the discussion and try again.' });
        }
        return;
      }
      if (roomMatch[2] === 'transcript.html') {
        const html = renderDiscussionExport(store.exportTranscript(room));
        response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        response.end(html); return;
      }
      if (roomMatch[2] === 'plan.md') {
        if (!store.latestDecision(room)) { send(404, { error: 'No proposed plan has been recorded for this discussion.' }); return; }
        const plan = store.exportPlan(room, { transcript: url.searchParams.get('transcript') === '1' });
        const filename = url.searchParams.get('transcript') === '1' ? plan.filename.replace(/\.md$/, '-transcript.md') : plan.filename;
        response.writeHead(200, {
          'Content-Type': 'text/markdown; charset=utf-8',
          'Content-Disposition': `attachment; filename="${filename}"`,
        });
        response.end(plan.markdown); return;
      }
      send(200, store.read(room, Number(after), 100));
    } catch {
      send(503, { error: 'Discussion storage is temporarily unavailable. The page will retry.' });
    } finally {
      store?.close();
    }
  });
  return server;
}
