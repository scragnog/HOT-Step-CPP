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
//
// Descriptions are kept to a clause or two: every MCP client puts the whole
// tool list in context each session (test/client.test.ts caps it at 16 KB).
// The long-form reference is the README; the training field lists are
// served by train_fields rather than advertised in the schemas.

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import type { CallToolResult } from '@modelcontextprotocol/sdk/types.js';
import {
  genBackends, genConfigure, genSubmit, genStatus, genCancel, genQueue, genWait, genSong,
  trainCapabilities, trainDatasets, trainDataset, trainDatasetCreate, trainDatasetRescan,
  trainDatasetLabel, trainJobs, trainJob, trainWait, trainPrepare, trainStart, trainRuns, trainFields,
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

  const waitDescription = (what: string, terminal: string, cancelTool: string) =>
    `Poll a ${what} until ${terminal} or the budget ends (default ${WAIT_DEFAULT_SECONDS}s, max ${WAIT_MAX_SECONDS}s; the client's own timeout must allow it). ` +
    `Returns { jobId, outcome: done|budget|cancelled, status: last response or null }. Never cancels the job; use ${cancelTool}.`;
  const maxSeconds = z.number().optional();

  server.tool(
    'gen_backends',
    'Generation backends, the active one, its capabilities and model catalogue.',
    {},
    async () => toResult(await genBackends()),
  );

  server.tool(
    'gen_configure',
    'Switch the active backend (frees the old one\'s VRAM) and/or set its model selection. ACE has none and answers 501, which is not an error.',
    {
      backend: z.enum(['ace', 'minimax-m3', 'yue2']).optional(),
      model: z.record(z.string()).optional().describe('Role to quant, e.g. {"lm":"q8_0"}.'),
    },
    async ({ backend, model }) => toResult(await genConfigure({ backend, model })),
  );

  server.tool(
    'gen_submit',
    'Submit a generation; returns { jobId }. Refused with 409 if `backend` is not the active one. Duration is a target on ACE, a ceiling on MiniMax-Music3, ignored by YuE2.',
    {
      backend: z.enum(['ace', 'minimax-m3', 'yue2']),
      operation: z.string().optional().describe('Default text2music; ACE also cover, repaint, lego, extract, complete, cover-nofsq.'),
      caption: z.string(),
      lyrics: z.string().optional(),
      instrumental: z.boolean().optional(),
      duration: z.number().optional(),
      seed: z.number().optional(),
      batchSize: z.number().optional().describe('ACE only.'),
      title: z.string().optional(),
      ditModel: z.string().optional().describe('ACE only; DiT catalogue.'),
      lmModel: z.string().optional().describe('ACE only; LM catalogue.'),
      options: z.record(z.union([z.string(), z.number(), z.boolean()])).optional()
        .describe('Extra top-level fields: mm3* for MiniMax-Music3, yue2* for YuE2, plain names for ACE. Typed arg names are rejected.'),
    },
    async (args) => toResult(await genSubmit(args)),
  );

  server.tool('gen_status', 'A generation job\'s status.', { jobId: z.string() }, async ({ jobId }) => toResult(await genStatus(jobId)));
  server.tool('gen_cancel', 'Cancel a generation job.', { jobId: z.string() }, async ({ jobId }) => toResult(await genCancel(jobId)));
  server.tool('gen_queue', 'Queue depth, GPU lane state and the running job.', {}, async () => toResult(await genQueue()));

  server.tool(
    'gen_wait',
    waitDescription('generation job', 'succeeded/failed/cancelled', 'gen_cancel'),
    { jobId: z.string(), maxSeconds },
    async ({ jobId, maxSeconds }, extra) => toResult(await genWait(jobId, maxSeconds, extra.signal)),
  );

  server.tool(
    'gen_song',
    'A saved song\'s metadata with an absolute audio URL; a file path only when HOTSTEP_URL is loopback. No audio bytes.',
    { songId: z.string() },
    async ({ songId }) => toResult(await genSong(songId)),
  );

  server.tool('train_capabilities', 'Training capabilities and engine readiness.', {}, async () => toResult(await trainCapabilities()));
  server.tool('train_datasets', 'Training datasets: id, name, slug, sourceDir, sampleCount, asset flags.', {}, async () => toResult(await trainDatasets()));
  server.tool('train_dataset', 'One training dataset in full.', { id: z.string() }, async ({ id }) => toResult(await trainDataset(id)));

  server.tool(
    'train_dataset_create',
    'Create a training dataset from a source audio folder.',
    {
      name: z.string(),
      sourceDir: z.string().describe('Absolute path.'),
      recursive: z.boolean().optional(),
      customTag: z.string().optional().describe('Trigger word.'),
      tagPosition: z.enum(['prepend', 'append', 'replace']).optional(),
      genreRatio: z.number().optional().describe('0-100.'),
      defaultArtist: z.string().optional(),
      defaultAlbum: z.string().optional(),
      defaultGenre: z.string().optional(),
      defaultLanguage: z.string().optional().describe("Default 'english'."),
    },
    async (args) => toResult(await trainDatasetCreate(args)),
  );

  server.tool('train_dataset_rescan', 'Rescan a dataset\'s source folder. 409 while a job runs on it.', { id: z.string() }, async ({ id }) => toResult(await trainDatasetRescan(id)));

  server.tool(
    'train_dataset_label',
    'Start a labeling stage; returns { jobId }. label: Essentia/Genius/LLM. caption: re-caption. build: write dataset.json. 409 while a job runs, or when MOSS is asked for while training holds the engine.',
    {
      datasetId: z.string(),
      stage: z.enum(['label', 'caption', 'build']),
      options: z.record(z.unknown()).optional().describe('The stage\'s body, sent verbatim.'),
    },
    async (args) => toResult(await trainDatasetLabel(args)),
  );

  server.tool('train_jobs', 'Training jobs, optionally for one dataset.', { datasetId: z.string().optional() }, async ({ datasetId }) => toResult(await trainJobs(datasetId)));

  server.tool(
    'train_job',
    'A training job\'s status, or cancel it with cancel: true.',
    { jobId: z.string(), cancel: z.boolean().optional() },
    async ({ jobId, cancel }) => toResult(await trainJob(jobId, cancel)),
  );

  server.tool(
    'train_wait',
    waitDescription('training job', 'done/failed/cancelled', 'train_job cancel'),
    { jobId: z.string(), maxSeconds },
    async ({ jobId, maxSeconds }, extra) => toResult(await trainWait(jobId, maxSeconds, extra.signal)),
  );

  const startBackends = ['ace-lm', 'ace-dit', 'mm3-lm', 'yue2-joint'] as const;
  const fields = (backend: string) => z.record(z.unknown()).optional().describe(`Fields for ${backend}; see train_fields.`);
  const options = z.record(z.unknown()).optional().describe('Untyped extra fields; may not repeat a typed one.');

  server.tool(
    'train_fields',
    'The accepted fields for a train_prepare or train_start backend: name, type, required, route default, notes.',
    {
      backend: z.enum(['ace', 'mm3', 'yue2', ...startBackends]),
      stage: z.enum(['preprocess', 'joint-prepare']).optional().describe('yue2 prepare only.'),
    },
    async ({ backend, stage }) => toResult(trainFields(backend, stage)),
  );

  server.tool(
    'train_prepare',
    'Start the data stage training needs; returns the route\'s answer with jobId. ace: tensors; mm3: RVQ codes; yue2 needs stage (preprocess, then joint-prepare, whose manifest is train_start yue2-joint\'s dataset).',
    {
      datasetId: z.string(),
      backend: z.enum(['ace', 'mm3', 'yue2']),
      stage: z.enum(['preprocess', 'joint-prepare']).optional(),
      ace: fields('ace'),
      mm3: fields('mm3'),
      yue2Preprocess: fields('yue2 preprocess'),
      yue2JointPrepare: fields('yue2 joint-prepare'),
      options,
    },
    async (args) => toResult(await trainPrepare(args)),
  );

  server.tool(
    'train_start',
    'Start a training run; returns the route\'s answer with jobId. Send only the field object for `backend`. yue2-joint always uses trainingMethod aitk and needs `dataset` unless autoPrepare is true.',
    {
      datasetId: z.string(),
      backend: z.enum(startBackends),
      aceLm: fields('ace-lm'),
      aceDit: fields('ace-dit'),
      mm3Lm: fields('mm3-lm'),
      yue2Joint: fields('yue2-joint'),
      options,
    },
    async (args) => toResult(await trainStart(args)),
  );

  server.tool(
    'train_runs',
    'Previous runs for a dataset and backend, with the backend\'s readiness (preprocess variants, mm3 codes and bases, yue2 cache and joint-prepare state).',
    { datasetId: z.string(), backend: z.enum(startBackends) },
    async ({ datasetId, backend }) => toResult(await trainRuns(datasetId, backend)),
  );

  return server;
}
