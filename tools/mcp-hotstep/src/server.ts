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
  trainCapabilities, trainDatasets, trainDataset, trainDatasetCreate, trainDatasetRescan,
  trainDatasetLabel, trainJobs, trainJob, trainWait, trainPrepare, trainStart, trainRuns,
  WAIT_DEFAULT_SECONDS, WAIT_MAX_SECONDS,
  type ToolOutcome,
} from './tools.js';
import {
  acePreprocessSchema, mm3CodesSchema, yue2PreprocessSchema, yue2JointPrepareSchema,
  aceLmSchema, aceDitSchema, mm3LmSchema, yue2JointSchema,
} from './trainSchemas.js';

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
      ditModel: z.string().optional().describe('ACE only: DiT/synth catalogue entry (sent as ditModel). Independent of lmModel — ACE has separate DiT and LM catalogues, a name from one is not valid in the other. No effect on MiniMax-Music3 or YuE2 — their model/adapter choice is engine state, set through gen_configure instead.'),
      lmModel: z.string().optional().describe('ACE only: LM catalogue entry (sent as lmModel). Independent of ditModel — see above. No effect on MiniMax-Music3 or YuE2.'),
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
      'Returns { jobId, outcome, status }: `outcome` is "done" (reached a terminal state), "budget" (the time budget ran out first), or "cancelled" (this tool call itself was cancelled by the client). `status` is the last status response actually obtained, or null if none was — a slow or interrupted poll never fabricates one. ' +
      'If this tool call itself is cancelled by the client, it leaves the job running on the server — it never cancels the job for you (use gen_cancel for that). ' +
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

  server.tool(
    'train_capabilities',
    'Training capabilities and engine readiness for the dataset pipeline (GET /api/training/capabilities): essentia/genius/LLM/MOSS availability, preprocess/train-lm/train-dit readiness, engine model registries. Returned whole.',
    {},
    async () => toResult(await trainCapabilities()),
  );

  server.tool(
    'train_datasets',
    'List training datasets (GET /api/training/datasets), trimmed to id, name, slug, sourceDir, sampleCount and the on-disk asset flags (labeled/built/tensors/adapters per backend) — not the full rows, which also carry label/caption settings only the UI needs. Use train_dataset for one dataset in full.',
    {},
    async () => toResult(await trainDatasets()),
  );

  server.tool(
    'train_dataset',
    'Get one training dataset in full (GET /api/training/datasets/:id): settings, samples, asset state.',
    { id: z.string() },
    async ({ id }) => toResult(await trainDataset(id)),
  );

  server.tool(
    'train_dataset_create',
    'Create a training dataset from a source audio folder (POST /api/training/datasets). Returns 201 with the created dataset.',
    {
      name: z.string().describe('Dataset name.'),
      sourceDir: z.string().describe('Absolute path to the source audio folder.'),
      recursive: z.boolean().optional().describe('Scan subfolders too.'),
      customTag: z.string().optional().describe('Trigger word woven into captions.'),
      tagPosition: z.enum(['prepend', 'append', 'replace']).optional(),
      genreRatio: z.number().optional().describe('0-100.'),
      defaultArtist: z.string().optional(),
      defaultAlbum: z.string().optional(),
      defaultGenre: z.string().optional(),
      defaultLanguage: z.string().optional().describe("Default 'english'."),
    },
    async (args) => toResult(await trainDatasetCreate(args)),
  );

  server.tool(
    'train_dataset_rescan',
    'Rescan a dataset\'s source folder for new/removed/changed files (POST /api/training/datasets/:id/rescan). Refuses with 409 while a job is already running for this dataset.',
    { id: z.string() },
    async ({ id }) => toResult(await trainDatasetRescan(id)),
  );

  server.tool(
    'train_dataset_label',
    'Start one stage of the dataset labeling pipeline, async — each stage answers 202 with a jobId to poll with train_wait or train_job. ' +
      '`stage: "label"` posts `options` as LabelOptions to /datasets/:id/label (Essentia BPM/key, Genius lyrics, LLM caption — which steps run is in `options`). ' +
      '`stage: "caption"` posts `options` as CaptionOptions to /datasets/:id/enhance/caption (re-caption with a specific provider). ' +
      '`stage: "build"` posts `options` as { outputPath? } to /datasets/:id/build (write dataset.json). ' +
      'All three refuse with 409 "A job is already running for this dataset" while one is active. label additionally refuses with 409 "MOSS and the /understand step need the engine, which a training job owns. Wait for it, or caption with a cloud provider (Gemini) instead." when the engine is held and MOSS/understand was asked for; caption refuses with 409 "MOSS needs the engine, which a training job owns. Wait for it, or caption with a cloud provider (Gemini) instead." under the same condition.',
    {
      datasetId: z.string(),
      stage: z.enum(['label', 'caption', 'build']),
      options: z.record(z.unknown()).optional().describe('The stage\'s own body (LabelOptions / CaptionOptions / { outputPath? }), forwarded verbatim.'),
    },
    async (args) => toResult(await trainDatasetLabel(args)),
  );

  server.tool(
    'train_jobs',
    'List training jobs, optionally filtered to one dataset (GET /api/training/jobs?datasetId=).',
    { datasetId: z.string().optional() },
    async ({ datasetId }) => toResult(await trainJobs(datasetId)),
  );

  server.tool(
    'train_job',
    'Get one training job\'s status (GET /api/training/jobs/:jobId), or cancel it (DELETE, with cancel: true).',
    {
      jobId: z.string(),
      cancel: z.boolean().optional().describe('Cancel this job instead of reading its status.'),
    },
    async ({ jobId, cancel }) => toResult(await trainJob(jobId, cancel)),
  );

  server.tool(
    'train_wait',
    `Poll a training job's status until it reaches a terminal state (done/failed/cancelled) or the time budget ends, whichever comes first. Same contract as gen_wait: default budget ${WAIT_DEFAULT_SECONDS}s, hard max ${WAIT_MAX_SECONDS}s, returns { jobId, outcome, status } with outcome one of done/budget/cancelled and status the last response obtained or null. ` +
      'If this tool call itself is cancelled by the client, it leaves the job running on the server — it never cancels the job for you (use train_job with cancel: true for that). ' +
      `A budget above ${WAIT_DEFAULT_SECONDS}s needs the MCP client's own request timeout raised to match.`,
    {
      jobId: z.string(),
      maxSeconds: z.number().optional().describe(`Seconds to wait before returning the latest status anyway. Default ${WAIT_DEFAULT_SECONDS}, clamped to ${WAIT_MAX_SECONDS}.`),
    },
    async ({ jobId, maxSeconds }, extra) => toResult(await trainWait(jobId, maxSeconds, extra.signal)),
  );

  const options = z.record(z.unknown()).optional()
    .describe('Passthrough for fields the typed object does not name; merged under it into the POST body. A key the typed object already names is rejected before any HTTP call. The server validates the rest.');

  server.tool(
    'train_prepare',
    'Run the data-preparation stage that training needs, per backend. Async: returns the route\'s response, which carries the jobId — wait with train_wait. ace and mm3 need a built dataset (train_dataset_label stage build); yue2 preprocess reads the source folder flat. ' +
      'backend ace: ACE tensor preprocess (POST /datasets/:id/preprocess), fields in `ace`; returns 202 { jobId }. ' +
      'backend mm3: MiniMax-Music3 RVQ codes export (POST /datasets/:id/mm3-codes), fields in `mm3`; returns { jobId, kind }. Training also needs per-track <stem>.mm3.txt captions (train_dataset_label stage caption). ' +
      'backend yue2 needs `stage`: "preprocess" encodes the YuE2 latent cache (POST /datasets/:id/yue2-preprocess, fields in `yue2Preprocess`); "joint-prepare" imports the existing cache stages (latents, codec ids, lead sheets) into the joint-training dataset (POST /datasets/:id/yue2-joint-prepare, fields in `yue2JointPrepare`, CPU only) and returns the `manifest` path that train_start yue2-joint takes as `dataset`. ' +
      'Every route answers 409 "A job is already running for this dataset" while one is active, and 503 when ace-train is missing. Only the field object matching backend/stage may be sent.',
    {
      datasetId: z.string(),
      backend: z.enum(['ace', 'mm3', 'yue2']),
      stage: z.enum(['preprocess', 'joint-prepare']).optional().describe('yue2 only, and required there.'),
      ace: acePreprocessSchema.optional(),
      mm3: mm3CodesSchema.optional(),
      yue2Preprocess: yue2PreprocessSchema.optional().describe('Out-of-range numbers fall back to the default silently.'),
      yue2JointPrepare: yue2JointPrepareSchema.optional(),
      options,
    },
    async (args) => toResult(await trainPrepare(args)),
  );

  server.tool(
    'train_start',
    'Start a training run. Async: returns the route\'s response, which carries the jobId — wait with train_wait, cancel with train_job cancel: true. Never retried on 401 (a resend could start a second run). Each field\'s description gives the route\'s own default for an absent field. ' +
      'ace-lm: ACE planner-LM adapter (POST /datasets/:id/train-lm), fields in `aceLm`, needs train_prepare ace; 202 { jobId }. ' +
      'ace-dit: ACE DiT adapter (POST /datasets/:id/train-dit), fields in `aceDit`, needs train_prepare ace; 202 { jobId }. ' +
      'mm3-lm: MiniMax-Music3 LM adapter (POST /datasets/:id/mm3-train-lm), fields in `mm3Lm`, needs train_prepare mm3 and .mm3.txt captions; returns { jobId, kind, runName, outDir, attnBackend }. Out-of-range mm3 numbers fall back to the default SILENTLY rather than failing. ' +
      'yue2-joint: YuE2 joint AR+NAR adapter (POST /datasets/:id/yue2-joint-train), fields in `yue2Joint`; trainingMethod is always "aitk", set by this tool, and cannot be passed. It does NOT auto-prepare by default: a fresh run needs `dataset` (from train_prepare yue2 joint-prepare) unless `autoPrepare: true` is sent (the Training Studio form sends it); a resume never auto-prepares. steps is required, and saveEvery on a fresh run. GET /api/training/defaults has no YuE2 section, so the field descriptions are the only defaults. Returns the resolved recipe with the jobId. ' +
      'Every route answers 409 "A job is already running for this dataset" while one is active. Only the field object matching backend may be sent.',
    {
      datasetId: z.string(),
      backend: z.enum(['ace-lm', 'ace-dit', 'mm3-lm', 'yue2-joint']),
      aceLm: aceLmSchema.optional(),
      aceDit: aceDitSchema.optional(),
      mm3Lm: mm3LmSchema.optional(),
      yue2Joint: yue2JointSchema.optional(),
      options,
    },
    async (args) => toResult(await trainStart(args)),
  );

  server.tool(
    'train_runs',
    'Previous runs for one dataset and backend, with that backend\'s readiness folded in: { runs, readiness }. ' +
      'ace-lm: runs = GET /datasets/:id/train-lm (newest adapter state), readiness.preprocess = GET /preprocess (tensor variants). ' +
      'ace-dit: runs = GET /train-dit, readiness.preprocess = GET /preprocess. ' +
      'mm3-lm: runs = GET /mm3-runs, readiness.mm3 = GET /mm3 (codes counts, missing models, installed bases, regCandidates, defaults, presets). ' +
      'yue2-joint: runs = GET /yue2-joint-runs (each with its checkpoints and resumeError), readiness.yue2 = GET /yue2 (latent cache), readiness.jointPrepare = GET /yue2-joint-prepare (bases, defaultBase, defaultDevice, missing models). Its `ready` flag describes a fresh default output dir, so it reads false even after a prepare; the prepare job\'s own `manifest` is the path to train with.',
    {
      datasetId: z.string(),
      backend: z.enum(['ace-lm', 'ace-dit', 'mm3-lm', 'yue2-joint']),
    },
    async ({ datasetId, backend }) => toResult(await trainRuns(datasetId, backend)),
  );

  return server;
}
