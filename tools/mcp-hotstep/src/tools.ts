// tools.ts — the business logic behind every MCP tool, kept separate from
// index.ts's server.tool() wiring so it can be called (and tested) directly
// against a loopback HTTP server, the same way trainingWorkers.test.ts tests
// pullLinked against a fake worker.

import path from 'node:path';
import { request, hotstepBaseUrl, hotstepIsLoopback, type HttpResult } from './http.js';
// Pure path config, not live engine/job state — safe to read directly rather
// than over HTTP (no route exposes it). Only correct when this process sees
// the same DATA_DIR env the running server does; see the README caveat.
import { config as serverConfig } from '../../../server/src/config.js';

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
  model?: string;
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
    ...(args.model !== undefined ? { ditModel: args.model, lmModel: args.model } : {}),
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

/** A cancelled `signal` returns the latest known status and leaves the job
 *  running server-side — this never calls cancel on the caller's behalf. */
export async function genWait(jobId: string, maxSeconds: number | undefined, signal: AbortSignal): Promise<ToolOutcome> {
  const budgetMs = Math.min(WAIT_MAX_SECONDS, Math.max(1, maxSeconds ?? WAIT_DEFAULT_SECONDS)) * 1000;
  const deadline = Date.now() + budgetMs;
  const statusPath = `/api/generate/status/${encodeURIComponent(jobId)}`;
  let last: HttpResult;
  for (;;) {
    last = await request('GET', statusPath);
    if (!last.ok) return fromHttp(last);
    const status = (last.data as { status?: string } | undefined)?.status;
    if (status === 'succeeded' || status === 'failed' || status === 'cancelled') break;
    if (signal.aborted || Date.now() >= deadline) break;
    await sleep(Math.min(WAIT_POLL_MS, deadline - Date.now()), signal);
  }
  return { kind: 'ok', data: last.data };
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
    ? path.join(serverConfig.data.audioDir, audioUrl.slice('/audio/'.length))
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
