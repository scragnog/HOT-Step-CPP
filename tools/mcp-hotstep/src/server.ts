// server.ts — the MCP server's tool registrations for HOT-Step CPP generation
//
// Everything here goes over the app's HTTP API (./http.ts) — the job queue
// and engine state live in the server process, not in a database this
// process could read directly. Env HOTSTEP_URL, default http://127.0.0.1:3001.
// The tool logic itself lives in ./tools.ts, kept separate so it can be
// tested directly against a loopback HTTP server.
//
// Split from index.ts (the stdio entrypoint) so a test can create a server
// and connect it over the SDK's in-memory transport without also starting a
// real stdio transport on process stdin/stdout.

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import type { CallToolResult } from '@modelcontextprotocol/sdk/types.js';
import {
  genBackends, genConfigure, genSubmit, genStatus, genCancel, genQueue, genWait, genSong,
  WAIT_DEFAULT_SECONDS, WAIT_MAX_SECONDS,
  type ToolOutcome,
} from './tools.js';

export function createServer(): McpServer {
  const server = new McpServer({ name: 'hotstep', version: '1.0.0' });

  /** The server's own response text, verbatim, on an HTTP failure — never
   *  paraphrased, so a tool failure shows exactly what the API said. */
  function toResult(outcome: ToolOutcome): CallToolResult {
    switch (outcome.kind) {
      case 'ok': return { content: [{ type: 'text', text: JSON.stringify(outcome.data, null, 2) }] };
      case 'rejected': return { content: [{ type: 'text', text: outcome.message }], isError: true };
      case 'http_error': return { content: [{ type: 'text', text: `HTTP ${outcome.status}: ${outcome.text}` }], isError: true };
    }
  }

  server.tool(
    'gen_backends',
    'List the registered generation backends (ace, minimax-m3, yue2), which one is active, the active backend\'s capability manifest, and its model catalogue.',
    {},
    async () => toResult(await genBackends()),
  );

  server.tool(
    'gen_configure',
    'Switch the active generation backend and/or select its models. Switching backend releases the OUTGOING backend\'s VRAM (fire-and-forget) before the new one loads anything. ' +
      'Model selection is engine STATE for MiniMax-Music3 (quant per role: lm, depth, cond, dit, voc) and YuE2 (lm, vae_variant) — posted here. ' +
      'ACE has no engine-state model selection: its model names travel per generate call instead (gen_submit\'s `model` arg), so this route answers 501 for it, which is reported as the reason, not an error to retry.',
    {
      backend: z.enum(['ace', 'minimax-m3', 'yue2']).optional().describe('Switch the active backend to this id. Omit to leave it as-is and only change model selection.'),
      model: z.record(z.string()).optional().describe('Model bucket selection posted to POST /api/backends/models, e.g. {"lm": "q8_0", "dit": "f16"} for minimax-m3. Applies to the backend named above, or the current active one if `backend` is omitted.'),
    },
    async ({ backend, model }) => toResult(await genConfigure({ backend, model })),
  );

  server.tool(
    'gen_submit',
    'Submit a generation job to POST /api/generate. Returns { jobId, status } immediately — poll with gen_status or gen_wait. ' +
      'Duration behaviour differs per backend: ACE (max 600s) treats it as a TARGET the LM aims for; MiniMax-Music3 (max 300s) and YuE2 treat it as a CEILING ONLY — the model usually ends earlier on its own stop token, and YuE2 ignores it entirely (always model-ended). ' +
      '`expectedBackend` is sent alongside `backend`: if the active backend has moved on since you checked gen_backends, the submit is refused with 409 instead of running on the wrong one.',
    {
      backend: z.enum(['ace', 'minimax-m3', 'yue2']).describe('The backend this request is for. Must match the currently active backend (gen_backends) or the submit is refused.'),
      operation: z.string().optional().describe('Generation operation (becomes `taskType`). Defaults to "text2music". Must be one the backend supports: ACE also has cover, repaint, lego, extract, complete, cover-nofsq; MiniMax-Music3 and YuE2 support text2music only.'),
      caption: z.string().describe('Audio style / caption describing the desired sound.'),
      lyrics: z.string().optional().describe('Lyrics with section tags. Omit or leave empty for an instrumental.'),
      instrumental: z.boolean().optional(),
      duration: z.number().optional().describe('Seconds. See the backend-specific behaviour above.'),
      seed: z.number().optional(),
      batchSize: z.number().optional().describe('ACE only (max 8); ignored by the other backends.'),
      title: z.string().optional(),
      model: z.string().optional().describe('ACE only: sets BOTH ditModel and lmModel to this name. No effect on MiniMax-Music3 or YuE2 — their model/adapter choice is engine state, set through gen_configure instead.'),
      options: z.record(z.union([z.string(), z.number(), z.boolean()])).optional()
        .describe('Backend-specific knobs sent as top-level HTTP fields. MiniMax-Music3 only reads fields prefixed "mm3" (e.g. mm3Steps), YuE2 only reads fields prefixed "yue2" (e.g. yue2Something) — anything else is silently ignored by those two. ACE reads plain unprefixed names (e.g. guidanceScale, inferenceSteps, negativePrompt). Any name that collides with one of this tool\'s own typed args is rejected before any HTTP call is made.'),
    },
    async (args) => toResult(await genSubmit(args)),
  );

  server.tool(
    'gen_status',
    'Get a generation job\'s current status (GET /api/generate/status/:id).',
    { jobId: z.string() },
    async ({ jobId }) => toResult(await genStatus(jobId)),
  );

  server.tool(
    'gen_cancel',
    'Cancel a running or queued generation job (POST /api/generate/cancel/:id).',
    { jobId: z.string() },
    async ({ jobId }) => toResult(await genCancel(jobId)),
  );

  server.tool(
    'gen_queue',
    'Inspect the generation queue: depth, whether the GPU lane is busy, and the currently running job if any (GET /api/generate/queue).',
    {},
    async () => toResult(await genQueue()),
  );

  server.tool(
    'gen_wait',
    `Poll a generation job's status until it reaches a terminal state (succeeded/failed/cancelled) or the time budget ends, whichever comes first. Default budget ${WAIT_DEFAULT_SECONDS}s, hard max ${WAIT_MAX_SECONDS}s. ` +
      'If this tool call itself is cancelled by the client, it returns the last known status and leaves the job running on the server — it never cancels the job for you (use gen_cancel for that). ' +
      `A budget above ${WAIT_DEFAULT_SECONDS}s needs the MCP client's own request timeout raised to match, or the client will give up on this call before the budget does.`,
    {
      jobId: z.string(),
      maxSeconds: z.number().optional().describe(`Seconds to wait before returning the latest status anyway. Default ${WAIT_DEFAULT_SECONDS}, clamped to ${WAIT_MAX_SECONDS}.`),
    },
    async ({ jobId, maxSeconds }, extra) => toResult(await genWait(jobId, maxSeconds, extra.signal)),
  );

  server.tool(
    'gen_song',
    'Look up a saved song by id (GET /api/songs/:id). Returns its metadata and audio_url resolved to an absolute URL under HOTSTEP_URL. ' +
      'Never returns audio bytes; a filesystem path is included only when HOTSTEP_URL points at this machine (loopback), since otherwise the path would not resolve for the caller.',
    { songId: z.string() },
    async ({ songId }) => toResult(await genSong(songId)),
  );

  return server;
}
