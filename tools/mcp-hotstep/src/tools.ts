// tools.ts — the business logic behind every MCP tool, kept separate from
// index.ts's server.tool() wiring so it can be called (and tested) directly
// against a loopback HTTP server, the same way trainingWorkers.test.ts tests
// pullLinked against a fake worker.

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { z } from 'zod';
import { request, hotstepBaseUrl, hotstepIsLoopback, type HttpResult } from './http.js';
import {
  acePreprocessSchema, mm3CodesSchema, yue2PreprocessSchema, yue2JointPrepareSchema,
  aceLmSchema, aceDitSchema, mm3LmSchema, yue2JointSchema,
} from './trainSchemas.js';

// Mirrors server/src/config.ts's `data.dir`/`data.audioDir` resolution —
// NOT imported directly. That module bootstraps the app on import (creates
// .env from .env.example on first launch and logs to stdout), which would
// write non-JSON bytes onto this process's stdio JSON-RPC stream. ponytail:
// duplicated two-line resolution; update if config.ts's DATA_DIR handling
// changes. Only correct when this process sees the same DATA_DIR env the
// running server does — true for the default, unmodified setup.
const SERVER_SRC_DIR = fileURLToPath(new URL('../../../server/src', import.meta.url));
const AUDIO_DIR = path.join(path.resolve(SERVER_SRC_DIR, '..', process.env.DATA_DIR || './data'), 'audio');

export type ToolOutcome =
  | { kind: 'ok'; data: unknown }
  | { kind: 'rejected'; message: string }
  | { kind: 'http_error'; status: number; text: string };

function fromHttp(r: Pick<HttpResult, 'ok' | 'status' | 'text' | 'data'>): ToolOutcome {
  return r.ok ? { kind: 'ok', data: r.data } : { kind: 'http_error', status: r.status, text: r.text };
}

// ── gen_backends ─────────────────────────────────────────────────────────────

export async function genBackends(): Promise<ToolOutcome> {
  const [backends, capabilities, models] = await Promise.all([
    request('GET', '/api/backends'),
    request('GET', '/api/capabilities'),
    request('GET', '/api/backends/models'),
  ]);
  for (const r of [backends, capabilities, models]) if (!r.ok) return fromHttp(r);
  return { kind: 'ok', data: { backends: backends.data, capabilities: capabilities.data, models: models.data } };
}

// ── gen_configure ────────────────────────────────────────────────────────────

export interface GenConfigureArgs {
  backend?: 'ace' | 'minimax-m3' | 'yue2';
  model?: Record<string, string>;
}

export async function genConfigure({ backend, model }: GenConfigureArgs): Promise<ToolOutcome> {
  if (!backend && !model) return { kind: 'rejected', message: 'Nothing to do: pass `backend` and/or `model`.' };
  const out: Record<string, unknown> = {};
  if (backend) {
    const r = await request('POST', '/api/backends/active', { id: backend });
    if (!r.ok) return fromHttp(r);
    out.switch = r.data;
  }
  if (model) {
    const r = await request('POST', '/api/backends/models', { backend, selection: model });
    // 501 ("this backend has no engine-state model selection", i.e. ACE) is
    // the documented reason, not a failure to surface as an error.
    if (!r.ok && r.status !== 501) return fromHttp(r);
    out.modelSelection = r.ok ? r.data : { status: r.status, body: r.text };
  }
  return { kind: 'ok', data: out };
}

// ── gen_submit ───────────────────────────────────────────────────────────────

// ponytail: mirrors ACE_OPERATIONS/MM3_OPERATIONS/YUE2_OPERATIONS in
// server/src/services/backends/{ace,minimax,yue2}/index.ts. Nothing exposes a
// backend's operation list over HTTP (only its capabilities manifest, which
// has no such field), and generation stays HTTP-only per this slice's brief —
// so this is a client-side mirror. Update it if those arrays change.
export const BACKEND_OPERATIONS: Readonly<Record<string, readonly string[]>> = {
  ace: ['text2music', 'cover', 'repaint', 'lego', 'extract', 'complete', 'cover-nofsq'],
  'minimax-m3': ['text2music'],
  yue2: ['text2music'],
};

// Fields gen_submit itself writes onto the POST body — an `options` entry
// with any of these names would silently overwrite what the caller's own
// typed args asked for, so it is rejected before any HTTP call instead.
// Includes ACE's caption aliases (its resolveRequest checks `prompt` before
// `caption`) even though gen_submit never sets them itself.
export const RESERVED_FIELDS: ReadonlySet<string> = new Set([
  'backend', 'expectedBackend', 'taskType',
  'caption', 'prompt', 'songDescription', 'style',
  'lyrics', 'instrumental', 'duration', 'seed', 'randomSeed', 'batchSize', 'title',
  'ditModel', 'lmModel',
]);

export interface GenSubmitArgs {
  backend: 'ace' | 'minimax-m3' | 'yue2';
  operation?: string;
  caption: string;
  lyrics?: string;
  instrumental?: boolean;
  duration?: number;
  seed?: number;
  batchSize?: number;
  title?: string;
  /** ACE only: DiT/synth catalogue entry (translateParams.ts maps this to
   *  synth_model). Independent of `lmModel` — ACE has separate DiT and LM
   *  catalogues; a name from one bucket is not valid in the other. */
  ditModel?: string;
  /** ACE only: LM catalogue entry (maps to lm_model). Independent of
   *  `ditModel` — see above. */
  lmModel?: string;
  options?: Record<string, string | number | boolean>;
}

/** Pure: validates and builds the POST /api/generate body, or explains why
 *  not, without making any HTTP call. Exported so a test can check the exact
 *  body a given set of args produces. */
export function buildSubmitBody(args: GenSubmitArgs): { body: Record<string, unknown> } | { rejected: string } {
  const op = args.operation ?? 'text2music';
  const known = BACKEND_OPERATIONS[args.backend] ?? [];
  if (!known.includes(op)) {
    return { rejected: `Backend '${args.backend}' does not support operation '${op}'. Supported: ${known.join(', ')}.` };
  }
  for (const key of Object.keys(args.options ?? {})) {
    if (RESERVED_FIELDS.has(key)) {
      return { rejected: `options.${key} collides with a reserved top-level field — pass it as this tool's own \`${key}\` arg instead.` };
    }
  }
  const body: Record<string, unknown> = {
    backend: args.backend, expectedBackend: args.backend, taskType: op, caption: args.caption,
    ...(args.lyrics !== undefined ? { lyrics: args.lyrics } : {}),
    ...(args.instrumental !== undefined ? { instrumental: args.instrumental } : {}),
    ...(args.duration !== undefined ? { duration: args.duration } : {}),
    ...(args.seed !== undefined ? { seed: args.seed } : {}),
    ...(args.batchSize !== undefined ? { batchSize: args.batchSize } : {}),
    ...(args.title !== undefined ? { title: args.title } : {}),
    ...(args.ditModel !== undefined ? { ditModel: args.ditModel } : {}),
    ...(args.lmModel !== undefined ? { lmModel: args.lmModel } : {}),
    ...(args.options ?? {}),
  };
  return { body };
}

export async function genSubmit(args: GenSubmitArgs): Promise<ToolOutcome> {
  const built = buildSubmitBody(args);
  if ('rejected' in built) return { kind: 'rejected', message: built.rejected };
  // Never retried on 401: a silent relogin-and-resend here could double-submit
  // a GPU job. http.ts still clears the cached token so the NEXT call — this
  // one retried by hand, or any other tool — logs in fresh.
  const r = await request('POST', '/api/generate', built.body, { retryOn401: false });
  return fromHttp(r);
}

// ── gen_status / gen_cancel / gen_queue ─────────────────────────────────────

export async function genStatus(jobId: string): Promise<ToolOutcome> {
  return fromHttp(await request('GET', `/api/generate/status/${encodeURIComponent(jobId)}`));
}

export async function genCancel(jobId: string): Promise<ToolOutcome> {
  return fromHttp(await request('POST', `/api/generate/cancel/${encodeURIComponent(jobId)}`));
}

export async function genQueue(): Promise<ToolOutcome> {
  return fromHttp(await request('GET', '/api/generate/queue'));
}

// ── gen_wait ─────────────────────────────────────────────────────────────────

export const WAIT_POLL_MS = 2000;
export const WAIT_DEFAULT_SECONDS = 45;
export const WAIT_MAX_SECONDS = 900;

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise(resolve => {
    const t = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => { clearTimeout(t); resolve(); }, { once: true });
  });
}

export type WaitOutcome = 'done' | 'budget' | 'cancelled';

/** The polling loop behind gen_wait and train_wait: a cancelled `signal`
 *  returns the latest known status without acting on the job (no cancel
 *  call) — that is each caller's own job to make, not this loop's. Budget
 *  and signal are checked right before EVERY poll, including the first, and
 *  each poll's own request is bounded by both the signal and the remaining
 *  budget (http.ts's `signal`/`timeoutMs`): a request that is cut short by
 *  either comes back as `status: 0`, never a thrown error, ends the loop,
 *  and never overwrites the last status actually obtained — there is no
 *  unbounded fallback fetch. A real HTTP failure (>=400) is reported as
 *  `http_error` and ends the loop immediately, same as every other tool. */
export async function waitForJob(
  statusPath: string,
  isTerminal: (status: unknown) => boolean,
  maxSeconds: number | undefined,
  signal: AbortSignal,
): Promise<{ lastStatus: unknown; outcome: WaitOutcome } | { httpError: { status: number; text: string } }> {
  const budgetMs = Math.min(WAIT_MAX_SECONDS, Math.max(1, maxSeconds ?? WAIT_DEFAULT_SECONDS)) * 1000;
  const deadline = Date.now() + budgetMs;
  let lastStatus: unknown = null;
  let outcome: WaitOutcome = 'budget';
  for (;;) {
    if (signal.aborted) { outcome = 'cancelled'; break; }
    if (Date.now() >= deadline) { outcome = 'budget'; break; }
    const r = await request('GET', statusPath, undefined, { signal, timeoutMs: Math.max(0, deadline - Date.now()) });
    if (r.status === 0) { outcome = signal.aborted ? 'cancelled' : 'budget'; break; }
    if (!r.ok) return { httpError: { status: r.status, text: r.text } };
    lastStatus = r.data;
    if (isTerminal(r.data)) { outcome = 'done'; break; }
    if (signal.aborted) { outcome = 'cancelled'; break; }
    if (Date.now() >= deadline) { outcome = 'budget'; break; }
    await sleep(Math.min(WAIT_POLL_MS, deadline - Date.now()), signal);
  }
  return { lastStatus, outcome };
}

const isTerminalGenerationStatus = (data: unknown): boolean => {
  const status = (data as { status?: string } | undefined)?.status;
  return status === 'succeeded' || status === 'failed' || status === 'cancelled';
};

export async function genWait(jobId: string, maxSeconds: number | undefined, signal: AbortSignal): Promise<ToolOutcome> {
  const r = await waitForJob(`/api/generate/status/${encodeURIComponent(jobId)}`, isTerminalGenerationStatus, maxSeconds, signal);
  if ('httpError' in r) return { kind: 'http_error', status: r.httpError.status, text: r.httpError.text };
  return { kind: 'ok', data: { jobId, outcome: r.outcome, status: r.lastStatus } };
}

// ── gen_song ─────────────────────────────────────────────────────────────────

export async function genSong(songId: string): Promise<ToolOutcome> {
  const r = await request('GET', `/api/songs/${encodeURIComponent(songId)}`);
  if (!r.ok) return fromHttp(r);
  const song = (r.data as { song?: Record<string, unknown> } | undefined)?.song;
  if (!song) return { kind: 'ok', data: r.data };
  const audioUrl = typeof song.audio_url === 'string' ? song.audio_url : undefined;
  const absoluteAudioUrl = audioUrl ? `${hotstepBaseUrl()}${audioUrl}` : undefined;
  const audioFilePath = audioUrl && audioUrl.startsWith('/audio/') && hotstepIsLoopback()
    ? path.join(AUDIO_DIR, audioUrl.slice('/audio/'.length))
    : undefined;
  return {
    kind: 'ok',
    data: {
      ...song,
      ...(absoluteAudioUrl ? { absoluteAudioUrl } : {}),
      ...(audioFilePath ? { audioFilePath } : {}),
    },
  };
}

// ── train_capabilities ──────────────────────────────────────────────────────

export async function trainCapabilities(): Promise<ToolOutcome> {
  return fromHttp(await request('GET', '/api/training/capabilities'));
}

// ── train_datasets ───────────────────────────────────────────────────────────

/** Trimmed row: id/name/slug/sourceDir/sampleCount plus the on-disk asset
 *  flags (datasetAssets.ts) a caller needs to decide what stage comes next —
 *  not the full row (label/caption/path settings, counts the UI alone needs). */
export async function trainDatasets(): Promise<ToolOutcome> {
  const r = await request('GET', '/api/training/datasets');
  if (!r.ok) return fromHttp(r);
  const rows = (r.data as { datasets?: Array<Record<string, unknown>> } | undefined)?.datasets ?? [];
  const datasets = rows.map(d => ({
    id: d.id, name: d.name, slug: d.slug, sourceDir: d.sourceDir, sampleCount: d.sampleCount, assets: d.assets,
  }));
  return { kind: 'ok', data: { datasets } };
}

// ── train_dataset ────────────────────────────────────────────────────────────

export async function trainDataset(id: string): Promise<ToolOutcome> {
  return fromHttp(await request('GET', `/api/training/datasets/${encodeURIComponent(id)}`));
}

// ── train_dataset_create ─────────────────────────────────────────────────────

// Mirrors CreateDatasetInput (server/src/services/training/types.ts:600).
export interface TrainDatasetCreateArgs {
  name: string;
  sourceDir: string;
  recursive?: boolean;
  customTag?: string;
  tagPosition?: 'prepend' | 'append' | 'replace';
  genreRatio?: number;
  defaultArtist?: string;
  defaultAlbum?: string;
  defaultGenre?: string;
  defaultLanguage?: string;
}

export async function trainDatasetCreate(args: TrainDatasetCreateArgs): Promise<ToolOutcome> {
  return fromHttp(await request('POST', '/api/training/datasets', args));
}

// ── train_dataset_rescan ─────────────────────────────────────────────────────

export async function trainDatasetRescan(id: string): Promise<ToolOutcome> {
  return fromHttp(await request('POST', `/api/training/datasets/${encodeURIComponent(id)}/rescan`));
}

// ── train_dataset_label ──────────────────────────────────────────────────────

export type TrainDatasetLabelStage = 'label' | 'caption' | 'build';

export interface TrainDatasetLabelArgs {
  datasetId: string;
  stage: TrainDatasetLabelStage;
  /** label -> LabelOptions (types.ts:644), caption -> CaptionOptions
   *  (types.ts:672), build -> { outputPath? }. Forwarded as the POST body
   *  verbatim — each stage's shape is the server route's own, not re-typed
   *  here, so a field the route gains later needs no change on this side. */
  options?: Record<string, unknown>;
}

const LABEL_STAGE_PATH: Record<TrainDatasetLabelStage, string> = {
  label: 'label',
  caption: 'enhance/caption',
  build: 'build',
};

export async function trainDatasetLabel({ datasetId, stage, options }: TrainDatasetLabelArgs): Promise<ToolOutcome> {
  const path_ = `/api/training/datasets/${encodeURIComponent(datasetId)}/${LABEL_STAGE_PATH[stage]}`;
  return fromHttp(await request('POST', path_, options ?? {}));
}

// ── train_jobs / train_job ───────────────────────────────────────────────────

export async function trainJobs(datasetId?: string): Promise<ToolOutcome> {
  const qs = datasetId ? `?datasetId=${encodeURIComponent(datasetId)}` : '';
  return fromHttp(await request('GET', `/api/training/jobs${qs}`));
}

export async function trainJob(jobId: string, cancel?: boolean): Promise<ToolOutcome> {
  const path_ = `/api/training/jobs/${encodeURIComponent(jobId)}`;
  return fromHttp(await request(cancel ? 'DELETE' : 'GET', path_));
}

// ── train_wait ───────────────────────────────────────────────────────────────

// TrainingJobStatus (server/src/services/training/types.ts:50): 'queued' |
// 'running' | 'done' | 'failed' | 'cancelled' — the last three are terminal.
const isTerminalTrainingStatus = (data: unknown): boolean => {
  const status = (data as { status?: string } | undefined)?.status;
  return status === 'done' || status === 'failed' || status === 'cancelled';
};

export async function trainWait(jobId: string, maxSeconds: number | undefined, signal: AbortSignal): Promise<ToolOutcome> {
  const r = await waitForJob(`/api/training/jobs/${encodeURIComponent(jobId)}`, isTerminalTrainingStatus, maxSeconds, signal);
  if ('httpError' in r) return { kind: 'http_error', status: r.httpError.status, text: r.httpError.text };
  return { kind: 'ok', data: { jobId, outcome: r.outcome, status: r.lastStatus } };
}

// ── train_prepare / train_start: shared body building ───────────────────────

/** One route a train_prepare/train_start call can land on: the per-backend
 *  field object it reads (`key`), that object's schema (for the declared
 *  field names), and the route path under /datasets/:id/. */
interface TrainTarget { key: string; schema: z.AnyZodObject; route: string }

/** Pure: picks the caller's field object for `target`, refuses one meant for
 *  another backend, validates it against the target's strict schema (the
 *  schemas are not advertised in the MCP tool list, to keep it small, so
 *  this is where unknown keys and wrong types are caught; train_fields lists
 *  them), refuses an `options` key that the schema already names (it would
 *  silently override the typed field), and merges the rest. */
function buildTrainBody(
  target: TrainTarget,
  all: readonly TrainTarget[],
  args: Record<string, unknown>,
  options: Record<string, unknown> | undefined,
): { body: Record<string, unknown> } | { rejected: string } {
  for (const t of all) {
    if (t.key !== target.key && args[t.key] !== undefined) {
      return { rejected: `\`${t.key}\` does not apply here — this call posts to /${target.route}, which reads \`${target.key}\`.` };
    }
  }
  let fields: Record<string, unknown> = {};
  if (args[target.key] !== undefined) {
    const parsed = target.schema.safeParse(args[target.key]);
    if (!parsed.success) {
      const issues = parsed.error.issues.map(i => `${[target.key, ...i.path].join('.')}: ${i.message}`).join('; ');
      return { rejected: `${issues}. train_fields lists the accepted fields.` };
    }
    fields = parsed.data;
  }
  const declared = target.schema.shape as Record<string, unknown>;
  for (const key of Object.keys(options ?? {})) {
    if (Object.hasOwn(declared, key)) return { rejected: `options.${key} is a typed field — pass it inside \`${target.key}\` instead.` };
  }
  return { body: { ...(options ?? {}), ...fields } };
}

/** POST a training body. Never retried on 401, same reason as gen_submit: a
 *  silent relogin-and-resend could start the same GPU job twice. */
async function postTrain(datasetId: string, route: string, body: Record<string, unknown>): Promise<ToolOutcome> {
  return fromHttp(await request('POST', `/api/training/datasets/${encodeURIComponent(datasetId)}/${route}`, body, { retryOn401: false }));
}

// ── train_prepare ────────────────────────────────────────────────────────────

const PREPARE_TARGETS = {
  ace: { key: 'ace', schema: acePreprocessSchema, route: 'preprocess' },
  mm3: { key: 'mm3', schema: mm3CodesSchema, route: 'mm3-codes' },
  yue2Preprocess: { key: 'yue2Preprocess', schema: yue2PreprocessSchema, route: 'yue2-preprocess' },
  yue2JointPrepare: { key: 'yue2JointPrepare', schema: yue2JointPrepareSchema, route: 'yue2-joint-prepare' },
} as const satisfies Record<string, TrainTarget>;

export interface TrainPrepareArgs {
  datasetId: string;
  backend: 'ace' | 'mm3' | 'yue2';
  stage?: 'preprocess' | 'joint-prepare';
  ace?: Record<string, unknown>;
  mm3?: Record<string, unknown>;
  yue2Preprocess?: Record<string, unknown>;
  yue2JointPrepare?: Record<string, unknown>;
  options?: Record<string, unknown>;
}

function prepareTarget(backend: string, stage: string | undefined): TrainTarget | { rejected: string } {
  if (backend === 'yue2') {
    if (stage === 'preprocess') return PREPARE_TARGETS.yue2Preprocess;
    if (stage === 'joint-prepare') return PREPARE_TARGETS.yue2JointPrepare;
    return { rejected: 'backend yue2 needs stage: "preprocess" or "joint-prepare".' };
  }
  if (stage !== undefined) return { rejected: `stage applies to backend yue2 only, not ${backend}.` };
  if (backend === 'ace') return PREPARE_TARGETS.ace;
  if (backend === 'mm3') return PREPARE_TARGETS.mm3;
  return { rejected: `Unknown backend '${backend}'. Supported: ace, mm3, yue2.` };
}

export async function trainPrepare(args: TrainPrepareArgs): Promise<ToolOutcome> {
  const target = prepareTarget(args.backend, args.stage);
  if ('rejected' in target) return { kind: 'rejected', message: target.rejected };
  const built = buildTrainBody(target, Object.values(PREPARE_TARGETS), args as unknown as Record<string, unknown>, args.options);
  if ('rejected' in built) return { kind: 'rejected', message: built.rejected };
  return postTrain(args.datasetId, target.route, built.body);
}

// ── train_start ──────────────────────────────────────────────────────────────

const START_TARGETS = {
  'ace-lm': { key: 'aceLm', schema: aceLmSchema, route: 'train-lm' },
  'ace-dit': { key: 'aceDit', schema: aceDitSchema, route: 'train-dit' },
  'mm3-lm': { key: 'mm3Lm', schema: mm3LmSchema, route: 'mm3-train-lm' },
  'yue2-joint': { key: 'yue2Joint', schema: yue2JointSchema, route: 'yue2-joint-train' },
} as const satisfies Record<string, TrainTarget>;

export type TrainBackend = keyof typeof START_TARGETS;

export interface TrainStartArgs {
  datasetId: string;
  backend: TrainBackend;
  aceLm?: Record<string, unknown>;
  aceDit?: Record<string, unknown>;
  mm3Lm?: Record<string, unknown>;
  yue2Joint?: Record<string, unknown>;
  options?: Record<string, unknown>;
}

export async function trainStart(args: TrainStartArgs): Promise<ToolOutcome> {
  const target: TrainTarget | undefined = START_TARGETS[args.backend];
  if (!target) return { kind: 'rejected', message: `Unknown backend '${args.backend}'. Supported: ${Object.keys(START_TARGETS).join(', ')}.` };
  const built = buildTrainBody(target, Object.values(START_TARGETS), args as unknown as Record<string, unknown>, args.options);
  if ('rejected' in built) return { kind: 'rejected', message: built.rejected };
  const body = built.body;
  if (args.backend === 'yue2-joint') {
    // The route refuses anything but 'aitk' ("Legacy is never selected
    // implicitly"); the tool owns the value so a caller cannot pick Legacy.
    if ('trainingMethod' in body) return { kind: 'rejected', message: 'trainingMethod is set by this tool ("aitk") and cannot be passed.' };
    // The route resumes only on a NON-EMPTY resumeRunId (training.ts:3437);
    // resumeStep alone, or with an empty id, would start a FRESH run.
    if (body.resumeRunId !== undefined && (typeof body.resumeRunId !== 'string' || !body.resumeRunId.trim())) {
      return { kind: 'rejected', message: 'resumeRunId must be a non-empty run id; omit it (and resumeStep) for a fresh run.' };
    }
    if ((body.resumeRunId === undefined) !== (body.resumeStep === undefined)) {
      return { kind: 'rejected', message: 'resumeRunId and resumeStep go together: send both to resume, neither for a fresh run.' };
    }
    body.trainingMethod = 'aitk';
  }
  return postTrain(args.datasetId, target.route, body);
}

// ── train_fields ─────────────────────────────────────────────────────────────

function zodType(t: z.ZodTypeAny): string {
  const def = t._def as { typeName: string; innerType?: z.ZodTypeAny; values?: string[]; value?: unknown; options?: z.ZodTypeAny[]; type?: z.ZodTypeAny; checks?: Array<{ kind: string }> };
  switch (def.typeName) {
    case 'ZodOptional': return zodType(def.innerType!);
    case 'ZodEnum': return def.values!.map(v => JSON.stringify(v)).join(' | ');
    case 'ZodLiteral': return JSON.stringify(def.value);
    case 'ZodUnion': return def.options!.map(zodType).join(' | ');
    case 'ZodArray': return `${zodType(def.type!)}[]`;
    case 'ZodNumber': return def.checks?.some(c => c.kind === 'int') ? 'integer' : 'number';
    case 'ZodString': return 'string';
    case 'ZodBoolean': return 'boolean';
    case 'ZodObject': return 'object';
    default: return 'unknown';
  }
}

/** The field list for one schema, nested objects flattened as parent.child.
 *  `default` is lifted from the description's "Default ..." clause (null
 *  where the description states none); `notes` is the whole description. */
function describeFields(schema: z.AnyZodObject, prefix = ''): Array<Record<string, unknown>> {
  return Object.entries(schema.shape as Record<string, z.ZodTypeAny>).flatMap(([name, t]) => {
    const notes = t.description ?? '';
    const row = {
      name: prefix + name, type: zodType(t), required: !t.isOptional(),
      default: /\bDefault:?\s+((?:[^;.]|\.(?=\d))+)/.exec(notes)?.[1].trim() ?? null, notes,
    };
    const inner = t instanceof z.ZodOptional ? t.unwrap() : t;
    return inner instanceof z.ZodObject ? [row, ...describeFields(inner, `${prefix}${name}.`)] : [row];
  });
}

export function trainFields(backend: string, stage?: string): ToolOutcome {
  let target: TrainTarget | { rejected: string };
  if (Object.hasOwn(START_TARGETS, backend)) {
    target = stage === undefined ? START_TARGETS[backend as TrainBackend]
      : { rejected: `stage applies to train_prepare backend yue2 only, not ${backend}.` };
  } else {
    target = prepareTarget(backend, stage);
  }
  if ('rejected' in target) return { kind: 'rejected', message: target.rejected };
  return { kind: 'ok', data: { argument: target.key, route: `POST /api/training/datasets/:id/${target.route}`, fields: describeFields(target.schema) } };
}

// ── train_runs ───────────────────────────────────────────────────────────────

/** Per backend: the runs GET, then the readiness GETs folded in under
 *  `readiness`, keyed by name. */
const RUNS_ROUTES: Record<TrainBackend, { runs: string; readiness: Record<string, string> }> = {
  'ace-lm': { runs: 'train-lm', readiness: { preprocess: 'preprocess' } },
  'ace-dit': { runs: 'train-dit', readiness: { preprocess: 'preprocess' } },
  'mm3-lm': { runs: 'mm3-runs', readiness: { mm3: 'mm3' } },
  'yue2-joint': { runs: 'yue2-joint-runs', readiness: { yue2: 'yue2', jointPrepare: 'yue2-joint-prepare' } },
};

export async function trainRuns(datasetId: string, backend: TrainBackend): Promise<ToolOutcome> {
  const routes = RUNS_ROUTES[backend];
  if (!routes) return { kind: 'rejected', message: `Unknown backend '${backend}'. Supported: ${Object.keys(RUNS_ROUTES).join(', ')}.` };
  const base = `/api/training/datasets/${encodeURIComponent(datasetId)}`;
  const names = Object.keys(routes.readiness);
  const results = await Promise.all([routes.runs, ...Object.values(routes.readiness)].map(r => request('GET', `${base}/${r}`)));
  for (const r of results) if (!r.ok) return fromHttp(r);
  const readiness = Object.fromEntries(names.map((name, i) => [name, results[i + 1].data]));
  return { kind: 'ok', data: { runs: results[0].data, readiness } };
}
