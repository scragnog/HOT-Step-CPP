// daw/dawFacade.ts - Glue for the /api/v1/daw facade.
//
// Talks to the existing /api routes over loopback HTTP (so the generation
// queue, envelope and translation stay the single source of truth), owns the
// facade job map, the per-job temp dir and the float32 WAV export.

import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import crypto from 'crypto';
import { v4 as uuidv4 } from 'uuid';
import { config, getFFmpegPath } from '../../config.js';
import { registerBusyCheck, trackBusy } from '../../dataroot/pointer.js';
import { DawError, DAW_API_VERSION, ditKind, isBaseModel, mapJobStatus, type DawGenerateRequest } from './dawSchema.js';

/** 127.0.0.0/8, ::1, and IPv4-mapped ::ffff:127.0.0.0/8. */
import { etaKey, estimate, recordRun, getFigures, getLoaded, setLoaded, etaByBackend, type EtaBackend } from './dawEta.js';

export function isLoopback(addr: string | undefined): boolean {
  if (!addr) return false;
  const a = addr.toLowerCase();
  if (a === '::1') return true;
  const v4 = a.startsWith('::ffff:') ? a.slice(7) : a;
  return /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(v4);
}

interface ForwardedHeaders { xff?: string; realIp?: string; forwarded?: string }

/** Client addresses a proxy reported: X-Forwarded-For, X-Real-IP and Forwarded `for=`. */
function forwardedHops(h: ForwardedHeaders): string[] {
  const hops = [...(h.xff?.split(',') ?? []), ...(h.realIp?.split(',') ?? [])];
  for (const m of (h.forwarded ?? '').matchAll(/for=("[^"]*"|[^;,\s]*)/gi)) {
    hops.push(m[1].replace(/"/g, '').replace(/^\[([^\]]*)\](:\d+)?$/, '$1').replace(/^(\d+\.\d+\.\d+\.\d+):\d+$/, '$1'));
  }
  return hops.map((x) => x.trim());
}

/** Socket and every forwarded hop are loopback (a LAN browser on the Vite proxy arrives over
 *  loopback but names its real address). A spoofed header can only add hops, so it only narrows this. */
export function isLocalCaller(remote: string | undefined, h: ForwardedHeaders = {}): boolean {
  return isLoopback(remote) && forwardedHops(h).every(isLoopback);
}

/** The facade's auth rule. A configured key is required from every caller, loopback and proxied
 *  included; with none configured only demonstrably local callers pass. Returns the 401 message, or null. */
export function dawAuthError(want: string, given: string | undefined, remote: string | undefined, h: ForwardedHeaders = {}): string | null {
  if (want) {
    const a = Buffer.from(given ?? ''), b = Buffer.from(want);
    return a.length === b.length && crypto.timingSafeEqual(a, b) ? null : 'X-DAW-Key is missing or wrong';
  }
  return isLocalCaller(remote, h) ? null : 'DAW facade is local-only; set HOTSTEP_DAW_KEY to allow other callers';
}

/** Detect audio container from magic bytes: wav, flac, aiff, or null. */
export function sniffAudio(head: Buffer): 'wav' | 'flac' | 'aiff' | null {
  if (head.length >= 12 && head.toString('latin1', 0, 4) === 'RIFF' && head.toString('latin1', 8, 12) === 'WAVE') return 'wav';
  if (head.length >= 4 && head.toString('latin1', 0, 4) === 'fLaC') return 'flac';
  if (head.length >= 12 && head.toString('latin1', 0, 4) === 'FORM') {
    const t = head.toString('latin1', 8, 12);
    if (t === 'AIFF' || t === 'AIFC') return 'aiff';
  }
  return null;
}

/** Read the first bytes of an uploaded file and check them against its extension. */
export function checkUploadMagic(filePath: string): void {
  const fd = fs.openSync(filePath, 'r');
  const head = Buffer.alloc(12);
  let n = 0;
  try { n = fs.readSync(fd, head, 0, 12, 0); } finally { fs.closeSync(fd); }
  const kind = sniffAudio(head.subarray(0, n));
  const ext = path.extname(filePath).toLowerCase();
  const want = ext === '.wav' ? 'wav' : ext === '.flac' ? 'flac' : (ext === '.aiff' || ext === '.aif') ? 'aiff' : null;
  if (!kind || kind !== want) {
    throw new DawError(400, 'bad_request', `uploaded file is not a valid ${want ?? 'audio'} file (content does not match)`);
  }
}

/** Shared secret for non-loopback callers; unset = loopback only. */
export const dawKey = () => process.env.HOTSTEP_DAW_KEY || '';

export const dawTmpRoot = () => path.join(config.data.dir, 'daw_tmp');

const base = () => `http://127.0.0.1:${config.server.port}`;

// ── Loopback client ───────────────────────────────────────────────────────

let token: string | null = null;

async function getToken(): Promise<string> {
  if (token) return token;
  const r = await fetch(`${base()}/api/auth/auto`);
  if (!r.ok) throw new DawError(502, 'upstream_error', `auth/auto returned ${r.status}`);
  token = ((await r.json()) as any).token;
  return token!;
}

type Caller = (method: string, route: string, body?: unknown) => Promise<{ status: number; data: any }>;
let caller: Caller = (method, route, body) => httpCall(method, route, body);
const call: Caller = (method, route, body) => caller(method, route, body);
/** Test hook: swap the loopback client. Returns the previous one. */
export function __setCaller(c: Caller): Caller { const prev = caller; caller = c; return prev; }

async function httpCall(method: string, route: string, body?: unknown, retry = true): Promise<{ status: number; data: any }> {
  const r = await fetch(`${base()}${route}`, {
    method,
    headers: { Authorization: `Bearer ${await getToken()}`, ...(body ? { 'Content-Type': 'application/json' } : {}) },
    body: body ? JSON.stringify(body) : undefined,
    // Metadata lookups must not hang a DAW request on a stuck server.
    ...(method === 'GET' && (route === '/api/models' || route === '/api/backends') ? { signal: AbortSignal.timeout(10_000) } : {}),
  });
  // Tokens live in memory upstream; mint a fresh one after a server restart.
  if (r.status === 401 && retry) { token = null; return httpCall(method, route, body, false); }
  let data: any = null;
  try { data = await r.json(); } catch { /* non-JSON */ }
  return { status: r.status, data };
}

function upstreamError(status: number, data: any): DawError {
  const msg = String(data?.error ?? `upstream returned ${status}`);
  if (status === 503) return new DawError(503, /paused/i.test(msg) ? 'engine_suspended' : 'engine_not_ready', msg);
  if (status === 409) return new DawError(409, 'backend_mismatch', msg);
  if (status === 404) return new DawError(404, 'not_found', msg);
  if (status === 400) return new DawError(400, 'bad_request', msg);
  return new DawError(502, 'upstream_error', msg);
}

// ── Status ─────────────────────────────────────────────────────────────────

export async function getStatus() {
  const [st, active] = await Promise.all([getStatusBase(), activeBackend().catch(() => undefined)]);
  const key = etaKey(active, st.loaded_model);
  const f = getFigures(key);
  return { ...st, eta: { backend: key, loaded: getLoaded(), ...f }, eta_by_backend: etaByBackend() };
}

async function getStatusBase() {
  const [health, models, caps] = await Promise.all([
    call('GET', '/api/health').catch(() => null),
    call('GET', '/api/models').catch(() => null),
    call('GET', '/api/capabilities').catch(() => null),
  ]);
  const dit: string[] = (models?.data?.models?.dit ?? []).map((m: any) => typeof m === 'string' ? m : (m?.name ?? String(m)));
  const ditModels = dit.map((name) => ({ name, kind: ditKind(name) }));
  const aceAlive = health?.data?.aceServer?.status === 'ok';
  return {
    api: DAW_API_VERSION,
    ace_server_alive: aceAlive,
    ready: aceAlive && health?.data?.engine?.ready === true,
    boot_status: health?.data?.engine?.bootStatus ?? null,
    version: health?.data?.version ?? null,
    loaded_model: models?.data?.defaults?.synth_model ?? models?.data?.config?.synth_model ?? null,
    dit_models: ditModels,
    non_turbo_available: ditModels.some((m) => m.kind !== 'turbo'),
    base_available: dit.some(isBaseModel),
    capabilities: caps?.data ?? null,
  };
}

/** Installed DiT names and the loaded one, for base-model gating. */
export async function getDitInfo(): Promise<{ dit: string[]; loaded: string | null }> {
  const { status, data } = await call('GET', '/api/models');
  if (status !== 200) throw upstreamError(status, data);
  const dit: string[] = (data?.models?.dit ?? []).map((m: any) => typeof m === 'string' ? m : (m?.name ?? String(m)));
  return { dit, loaded: data?.defaults?.synth_model ?? data?.config?.synth_model ?? null };
}

// ── Temp dirs ──────────────────────────────────────────────────────────────

export function newTmpDir(id: string): string {
  const dir = path.join(dawTmpRoot(), id);
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

/** Remove one entry, only if it is a direct child of daw_tmp. */
function removeTmp(id: string): void {
  const root = path.resolve(dawTmpRoot());
  const dir = path.resolve(root, id);
  if (path.dirname(dir) !== root) return;
  fs.rmSync(dir, { recursive: true, force: true });
}

const TTL_MS = 6 * 3600_000;
const FAILED_TTL_MS = 10 * 60_000;

type Aged = { createdAt: number; finishedAt?: number; finalState?: string };
function expired(j: Aged, now: number): boolean {
  if (now - j.createdAt > TTL_MS) return true;
  const bad = j.finalState === 'failed' || j.finalState === 'cancelled';
  return bad && now - (j.finishedAt ?? j.createdAt) > FAILED_TTL_MS;
}

/** Sweep the job maps directly, then orphaned daw_tmp dirs older than the TTL. */
export function sweepTmp(now = Date.now()): void {
  for (const map of [jobs, understands] as Map<string, Aged>[]) {
    for (const [id, j] of map) {
      if (expired(j, now)) { map.delete(id); try { removeTmp(id); } catch { /* next round */ } }
    }
  }
  const root = dawTmpRoot();
  if (!fs.existsSync(root)) return;
  for (const name of fs.readdirSync(root)) {
    if (jobs.has(name) || understands.has(name)) continue;
    let age: number;
    try { age = now - fs.statSync(path.join(root, name)).mtimeMs; } catch { continue; }
    if (age > TTL_MS) removeTmp(name);
  }
}

let sweeper: NodeJS.Timeout | null = null;
export function startSweeper(): void {
  if (sweeper) return;
  try { sweepTmp(); } catch (e) { console.warn('[daw] startup sweep failed:', e); }
  sweeper = setInterval(() => { try { sweepTmp(); } catch { /* next round */ } }, 15 * 60_000);
  sweeper.unref();
}

// ── Generate / jobs ────────────────────────────────────────────────────────

/**
 * One facade job. A plain job maps to one upstream job (upstreamId). A
 * sequential job (SA3 with takes > 1, since SA3 renders one take per job)
 * maps to N single-take upstream jobs run back to back; `seq` tracks them and
 * upstreamId always points at the current one.
 */
interface FacadeJob {
  id: string; upstreamId: string; seeds: number[]; finalState?: string; createdAt: number; finishedAt?: number;
  backend?: string;
  /** DiT model sent upstream (null = whatever is loaded). */
  ditModel?: string | null;
  /** ACE text2music only: requested length; takes are padded or cut to it exactly on export. */
  fitTo?: number;
  /** YuE2 only: requested length; longer takes are trimmed on export. */
  trimTo?: number;
  seq?: SeqState;
  eta?: { key: EtaBackend; takes: number; durationS: number; estS: number; cold: boolean; recorded?: boolean };
  /** Last time a poll saw the job still running (facade getJob or sequential runner). */
  lastPollAt?: number;
  /** Real completion time, when known to within MAX_DONE_GAP_MS. */
  doneAt?: number;
}
/** Only learn from a run whose completion time is known to within this. */
const MAX_DONE_GAP_MS = 3000;
interface SeqState {
  takes: number;
  upstreamIds: string[];
  /** Terminal status body per finished take. */
  results: any[];
  current: number;
  cancelled: boolean;
  failed?: string;
  /** A later take failed after at least one succeeded: done with fewer takes plus an error. */
  partial?: boolean;
  done?: Promise<void>;
}
const jobs = new Map<string, FacadeJob>();

let pollMs = 1000;
/** Test hook: poll interval of the sequential runner. Returns the previous one. */
export function __setPollMs(ms: number): number { const prev = pollMs; pollMs = ms; return prev; }
let retryDelaysMs = [500, 1000, 2000];
/** Test hook: backoff between status-poll retries. Returns the previous one. */
export function __setRetryDelays(ms: number[]): number[] { const prev = retryDelaysMs; retryDelaysMs = ms; return prev; }

async function activeBackend(): Promise<string | undefined> {
  try {
    const { status, data } = await call('GET', '/api/backends');
    return status === 200 && typeof data?.activeId === 'string' ? data.activeId : undefined;
  } catch { return undefined; }
}

/** The backend whose weights are resident now: active backend id plus loaded DiT. */
async function loadedNow(): Promise<{ activeId?: string; loadedModel: string | null; key: EtaBackend | null }> {
  const [activeId, models] = await Promise.all([
    activeBackend(),
    call('GET', '/api/models').catch(() => null),
  ]);
  const d = models?.status === 200 ? models.data : null;
  const loadedModel: string | null = d?.defaults?.synth_model ?? d?.config?.synth_model ?? null;
  return { activeId, loadedModel, key: activeId ? etaKey(activeId, loadedModel) : null };
}

const randomSeed = () => Math.floor(Math.random() * 0x7fffffff);

export async function submitGenerate(id: string, req: DawGenerateRequest, params: Record<string, unknown>): Promise<string> {
  const now = await loadedNow();
  const backend = now.activeId;
  const takes = req.takes ?? 1;
  const key = etaKey(backend, req.dit_model ?? now.loadedModel);
  const durationS = req.duration_s ?? 30;
  // Real resident backend wins; the last facade job is only a fallback.
  const resident = now.key ?? getLoaded();
  const cold = resident !== key;
  const eta = { key, takes, durationS, estS: estimate(key, takes, durationS, resident), cold };
  const trimTo = backend === 'yue2' && req.duration_s !== undefined ? req.duration_s : undefined;
  const fitTo = backend === 'ace' && req.mode === 'text2music' && req.duration_s !== undefined ? req.duration_s : undefined;
  // SA3 ignores batchSize; YuE2 caps songs per job at max_lm_batch. Either
  // way, run N single-take jobs with distinct seeds under one facade id.
  const sequential = takes > 1 && (backend === 'sa3' || (backend === 'yue2' && (await batchMax(backend)) < takes));
  if (sequential) {
    const seeds = Array.from({ length: takes }, (_, i) => req.seed !== undefined ? req.seed + i : randomSeed());
    const first = await submitOne(params, seeds[0]);
    const seq: SeqState = { takes, upstreamIds: [first], results: [], current: 0, cancelled: false };
    const j: FacadeJob = { id, upstreamId: first, seeds, backend, ditModel: req.dit_model ?? null, trimTo, seq, eta, createdAt: Date.now() };
    jobs.set(id, j);
    seq.done = trackBusy('DAW multi-take', runSequential(j, params)).catch((e) => { seq.failed = e?.message ?? String(e); });
    return id;
  }
  const { status, data } = await call('POST', '/api/generate', params);
  if (status !== 200 || !data?.jobId) throw upstreamError(status, data);
  // YuE2 seeds take i with seed + i (yue2SongEntries); report what it rendered.
  const seeds = req.seed !== undefined ? Array.from({ length: takes }, (_, i) => req.seed! + (backend === 'yue2' ? i : 0)) : [];
  jobs.set(id, { id, upstreamId: data.jobId, seeds, backend, ditModel: req.dit_model ?? null, trimTo, fitTo, eta, createdAt: Date.now() });
  return id;
}

/** Songs per job the backend reports (capabilities core.batch.max); 1 when unknown. */
async function batchMax(backend: string): Promise<number> {
  try {
    const { status, data } = await call('GET', `/api/capabilities?backend=${encodeURIComponent(backend)}`);
    const n = Number(data?.core?.batch?.max);
    return status === 200 && Number.isFinite(n) && n >= 1 ? n : 1;
  } catch { return 1; }
}

async function submitOne(params: Record<string, unknown>, seed: number): Promise<string> {
  const { status, data } = await call('POST', '/api/generate', { ...params, batchSize: 1, seed, randomSeed: false });
  if (status !== 200 || !data?.jobId) throw upstreamError(status, data);
  return data.jobId;
}

const TERMINAL = new Set(['succeeded', 'failed', 'cancelled']);

/** One status poll, retrying a non-200 (or a thrown call) with backoff before giving up. */
async function pollStatus(upstreamId: string): Promise<any> {
  for (let attempt = 0; ; attempt++) {
    let err: unknown;
    try {
      const { status, data } = await call('GET', `/api/generate/status/${upstreamId}`);
      if (status === 200) return data;
      err = upstreamError(status, data);
    } catch (e) { err = e; }
    if (attempt >= retryDelaysMs.length) throw err;
    await new Promise((r) => setTimeout(r, retryDelaysMs[attempt]));
  }
}

/**
 * Wait for each take in turn, then submit the next. Stops on cancel, on
 * failure, or when the job was swept from the map. A take that fails after
 * at least one succeeded leaves the job partial instead of failed.
 */
async function runSequential(j: FacadeJob, params: Record<string, unknown>): Promise<void> {
  const seq = j.seq!;
  // Swept from the map: stop, and cancel the in-flight take so the GPU stops.
  const swept = (): boolean => {
    if (jobs.get(j.id) === j) return false;
    const cur = seq.upstreamIds.length - 1;
    if (cur >= 0 && !seq.results[cur] && !seq.cancelled) {
      seq.cancelled = true;
      void call('POST', `/api/generate/cancel/${seq.upstreamIds[cur]}`).catch(() => null);
    }
    return true;
  };
  const fail = (i: number, msg: string): void => {
    if (seq.results.slice(0, i).some((r) => r?.status === 'succeeded')) { seq.partial = true; seq.failed = msg; return; }
    throw new Error(msg);
  };
  for (let i = 0; i < seq.takes; i++) {
    if (swept()) return;
    if (i > 0) {
      if (seq.cancelled) return;
      let next: string;
      try { next = await submitOne(params, j.seeds[i]); } catch (e: any) { return fail(i, e?.message ?? String(e)); }
      seq.upstreamIds.push(next);
      seq.current = i;
      j.upstreamId = next;
      // Cancel landed while this submit was in flight: cancelJob skipped the
      // finished take, so cancel the new one now.
      if (seq.cancelled) { await call('POST', `/api/generate/cancel/${next}`).catch(() => null); return; }
    }
    for (;;) {
      if (swept()) return;
      let data: any;
      try { data = await pollStatus(seq.upstreamIds[i]); } catch (e: any) { return fail(i, e?.message ?? String(e)); }
      const t = Date.now(), prev = j.lastPollAt ?? j.createdAt;
      j.lastPollAt = t;
      if (TERMINAL.has(data?.status)) {
        seq.results[i] = data;
        if (data.status === 'failed' && !seq.cancelled) return fail(i, data.error ?? `take ${i} failed`);
        if (data.status !== 'succeeded') return;
        if (i === seq.takes - 1 && t - prev < MAX_DONE_GAP_MS) j.doneAt = t;
        break;
      }
      if (seq.cancelled) return;
      await new Promise((r) => setTimeout(r, pollMs));
    }
  }
}

function requireJob(id: string): FacadeJob {
  const j = jobs.get(id);
  if (!j) throw new DawError(404, 'not_found', `unknown job ${id}`);
  return j;
}

async function upstreamStatus(j: FacadeJob, upstreamId = j.upstreamId): Promise<any> {
  const { status, data } = await call('GET', `/api/generate/status/${upstreamId}`);
  if (status !== 200) throw upstreamError(status, data);
  return data;
}

/** Fold a sequential job into one upstream-shaped status body. */
async function sequentialStatus(j: FacadeJob): Promise<any> {
  const seq = j.seq!;
  const done = seq.results.filter((r) => r?.status === 'succeeded');
  const bad = seq.partial ? undefined : seq.results.find((r) => r && r.status !== 'succeeded');
  const cur = seq.results[seq.current] || seq.partial ? null : await upstreamStatus(j, seq.upstreamIds[seq.current]).catch(() => null);
  let status: string;
  if (seq.cancelled || bad?.status === 'cancelled') status = 'cancelled';
  else if (seq.partial) status = 'partial';
  else if (seq.failed || bad) status = 'failed';
  else if (done.length === seq.takes) status = 'succeeded';
  else status = cur?.status === 'pending' && seq.current === 0 ? 'pending' : 'running';
  const curPct = cur ? Math.min(100, Math.max(0, Number(cur.progress ?? 0))) : 0;
  return {
    status,
    stage: cur?.stage ?? null,
    ace_phase_progress: cur?.ace_phase_progress,
    progress: ((done.length + curPct / 100) / seq.takes) * 100,
    result: {
      audioUrls: done.map((r) => r.result?.audioUrls?.[0]),
      durations: done.map((r) => r.result?.durations?.[0] ?? r.result?.duration ?? null),
    },
    error: seq.failed ?? bad?.error ?? (seq.partial ? seq.results.find((r) => r && r.status !== 'succeeded')?.error : null) ?? null,
  };
}

/** Remaining seconds for a job: estimate minus elapsed, floor 0. */
export function jobEtaS(jOrId: string | FacadeJob, now = Date.now()): number | null {
  const j = typeof jOrId === 'string' ? jobs.get(jOrId) : jOrId;
  if (!j?.eta) return null;
  if (j.finalState) return 0;
  return Math.max(0, Math.round((j.eta.estS - (now - j.createdAt) / 1000) * 10) / 10);
}

export async function getJob(id: string) {
  const j = requireJob(id);
  const out: ReturnType<typeof mapJobStatus> & { take_count?: number; takes_done?: number } =
    mapJobStatus(id, j.seq ? await sequentialStatus(j) : await upstreamStatus(j), j.seeds);
  if (j.seq) { out.take_count = j.seq.takes; out.takes_done = out.takes.length; }
  if (j.fitTo !== undefined) {
    for (const t of out.takes as Array<Record<string, unknown>>) t.duration_s = j.fitTo;
  }
  if (j.trimTo !== undefined) {
    for (const t of out.takes as Array<Record<string, unknown>>) {
      const from = typeof t.duration_s === 'number' && t.duration_s > j.trimTo ? t.duration_s : null;
      t.trimmed_from_s = from;
      if (from !== null) t.duration_s = j.trimTo;
    }
  }
  const seenAt = Date.now();
  if ((out.state === 'failed' || out.state === 'cancelled' || out.state === 'done' || out.state === 'partial') && !j.finalState) {
    j.finalState = out.state;
    j.finishedAt = seenAt;
    // Learn only from successful runs whose real end time is known: the
    // sequential runner stamps doneAt; otherwise this poll must land within
    // MAX_DONE_GAP_MS of the previous one (or of submit), so a slow poller
    // never teaches the EMA its poll gap.
    if (!j.seq && j.doneAt === undefined && seenAt - (j.lastPollAt ?? j.createdAt) < MAX_DONE_GAP_MS) j.doneAt = seenAt;
    if (out.state === 'done' && j.eta && !j.eta.recorded && j.doneAt !== undefined) {
      j.eta.recorded = true;
      const res = await loadedNow().catch(() => null);
      const still = res?.key === j.eta.key;
      recordRun(j.eta.key, j.eta.takes, j.eta.durationS, (j.doneAt - j.createdAt) / 1000, j.eta.cold, still);
      if (!still && res?.key) setLoaded(res.key);
    }
  } else if (!j.finalState) {
    j.lastPollAt = seenAt;
  }
  // While the engine loads weights the job reads 'running' with no render yet,
  // and elapsed time can pass the estimate. Report eta_s null (unknown) and an
  // additive phase field instead of a misleading 0. State stays unchanged.
  const loading = !j.finalState && out.state !== 'done' && /^loading\b/i.test(String(out.stage ?? ''));
  let etaS = loading ? null : jobEtaS(j);
  // SA3 renders in one synchronous call with no progress, so once its estimate
  // runs out the remaining time is unknown: report null, not a clamped 0.
  const sa3Overdue = !loading && !j.finalState && j.eta?.key === 'sa3' && etaS === 0;
  if (sa3Overdue) etaS = null;
  (out as Record<string, unknown>).phase = j.finalState ?? (loading ? 'loading_model' : sa3Overdue ? 'running' : out.state);
  (out as Record<string, unknown>).eta_s = etaS;
  (out as Record<string, unknown>).dit_model = j.ditModel ?? null;
  if (j.eta) (out as Record<string, unknown>).eta_backend = j.eta.key;
  return out;
}

export async function cancelJob(id: string) {
  const j = requireJob(id);
  // Finished jobs keep their state; cancel is a no-op on them.
  if (j.finalState === 'done' || j.finalState === 'partial' || j.finalState === 'failed' || j.finalState === 'cancelled') {
    return { api: DAW_API_VERSION, id, state: j.finalState };
  }
  // Sequential: flag first so the runner submits no further takes.
  let target: string | null = j.upstreamId;
  if (j.seq) {
    // Every take already rendered: nothing to stop, the job is done.
    if (j.seq.results.length === j.seq.takes && j.seq.results.every((r) => r?.status === 'succeeded')) {
      j.finalState = 'done';
      j.finishedAt = Date.now();
      return { api: DAW_API_VERSION, id, state: 'done' };
    }
    j.seq.cancelled = true;
    // Only cancel a take whose result is not recorded yet: upstream would
    // overwrite a finished take to cancelled. If the next submit is in
    // flight, the runner cancels its id as soon as it returns.
    const cur = j.seq.upstreamIds.length - 1;
    target = j.seq.results[cur] ? null : j.seq.upstreamIds[cur];
  }
  if (target) {
    const { status, data } = await call('POST', `/api/generate/cancel/${target}`);
    if (status !== 200) throw upstreamError(status, data);
  }
  j.finalState = 'cancelled';
  j.finishedAt = Date.now();
  return { api: DAW_API_VERSION, id, state: 'cancelled' };
}

function runFfmpeg(args: string[]): Promise<void> {
  const ff = getFFmpegPath();
  if (!ff) return Promise.reject(new DawError(500, 'upstream_error', 'ffmpeg not available'));
  // On timeout execFile kills the child with killSignal before rejecting.
  return new Promise((resolve, reject) => execFile(ff, args, { timeout: 120_000, killSignal: 'SIGKILL' }, (err) => err ? reject(err) : resolve()));
}

export interface ConvertOpts {
  /** Cut to this many seconds with a 10 ms fade-out at the end. Never pads. */
  trimS?: number;
  /** ACE only: pad with silence or cut to exactly this many seconds (48 kHz frames). */
  fitS?: number;
  /** Source length in seconds, if known; the fade-out goes at min(srcS, fitS). */
  srcS?: number;
}

/** Exact 48 kHz frame plan for fitting a take to the requested length. */
export function fitPlan(srcFrames: number, targetS: number, rate = 48000) {
  const frames = Math.round(targetS * rate);
  return { frames, pad: Math.max(0, frames - srcFrames), cut: Math.max(0, srcFrames - frames), fadeEndFrame: Math.min(srcFrames, frames) };
}
export const TRIM_FADE_S = 0.01;

/** ffmpeg arguments for the float32 48 kHz stereo export, with an optional trim. */
export function ffmpegArgs(src: string, dst: string, opts: ConvertOpts = {}): string[] {
  const args = ['-y', '-i', src];
  if (opts.fitS !== undefined) {
    const srcS = opts.srcS !== undefined && Number.isFinite(opts.srcS) ? opts.srcS : opts.fitS;
    const p = fitPlan(Math.round(srcS * 48000), opts.fitS);
    const st = Math.max(0, p.fadeEndFrame / 48000 - TRIM_FADE_S).toFixed(6);
    args.push('-af', `aresample=48000,afade=t=out:st=${st}:d=${TRIM_FADE_S},apad=whole_len=${p.frames},atrim=end_sample=${p.frames}`);
  } else if (opts.trimS !== undefined) {
    const st = Math.max(0, opts.trimS - TRIM_FADE_S).toFixed(6);
    args.push('-af', `afade=t=out:st=${st}:d=${TRIM_FADE_S}`, '-t', opts.trimS.toFixed(6));
  }
  return [...args, '-ar', '48000', '-ac', '2', '-c:a', 'pcm_f32le', dst];
}

type Converter = (src: string, dst: string, opts?: ConvertOpts) => Promise<void>;
let converter: Converter = (src, dst, opts) => runFfmpeg(ffmpegArgs(src, dst, opts));
/** Test hook: swap the WAV converter. Returns the previous one. */
export function __setConverter(c: Converter): Converter { const prev = converter; converter = c; return prev; }

const inflight = new Map<string, Promise<void>>();
registerBusyCheck('daw-convert', () => inflight.size ? [`${inflight.size} DAW audio conversion(s) running`] : []);

/** Convert src to out once; concurrent callers share one conversion. Unique .part, atomic rename. */
export function convertOnce(src: string, out: string, opts?: ConvertOpts): Promise<void> {
  if (fs.existsSync(out)) return Promise.resolve();
  const pending = inflight.get(out);
  if (pending) return pending;
  const part = `${out}.${crypto.randomBytes(6).toString('hex')}.part.wav`;
  const p = (async () => {
    try {
      await converter(src, part, opts);
      fs.renameSync(part, out);
    } catch (e) {
      try { fs.rmSync(part, { force: true }); } catch { /* ignore */ }
      throw e;
    } finally {
      inflight.delete(out);
    }
  })();
  inflight.set(out, p);
  return p;
}

/** True when p resolves to a path strictly inside dir (a sibling like <dir>X is outside). */
export function isInsideDir(dir: string, p: string): boolean {
  const rel = path.relative(path.resolve(dir), path.resolve(p));
  return rel !== '' && rel !== '..' && !rel.startsWith('..' + path.sep) && !path.isAbsolute(rel);
}

/** Path of a cached 48 kHz float32 stereo WAV for take n, converting on first use. */
export async function takeWavPath(id: string, n: number): Promise<string> {
  const j = requireJob(id);
  let up: any;
  let idx = n;
  if (j.seq) {
    // Take n of a sequential job is take 0 of its own upstream job.
    if (n >= j.seq.takes) throw new DawError(404, 'not_found', `no take ${n}`);
    up = j.seq.results[n];
    if (!up) throw new DawError(409, 'bad_request', `take ${n} is not finished`);
    idx = 0;
  } else {
    up = await upstreamStatus(j);
  }
  if (up.status !== 'succeeded') throw new DawError(409, 'bad_request', `job is ${up.status}, audio not ready`);
  const url: string | undefined = up.result?.audioUrls?.[idx];
  if (!url) throw new DawError(404, 'not_found', `no take ${n}`);
  const rel = url.replace(/^\/audio\//, '').split('?')[0];
  const src = path.resolve(config.data.audioDir, rel);
  if (!isInsideDir(config.data.audioDir, src) || !fs.existsSync(src)) {
    throw new DawError(404, 'not_found', `take ${n} audio file is missing`);
  }
  const out = path.join(newTmpDir(id), `take_${n}.wav`);
  // YuE2 ignores duration_s; trim longer takes to the requested length, never pad.
  const len = Number(up.result?.durations?.[idx] ?? up.result?.duration);
  const trim = j.trimTo !== undefined && (!Number.isFinite(len) || len > j.trimTo) ? { trimS: j.trimTo } : undefined;
  // ACE sizes the latent from the LM code count (about 9.44 s for 10 s); fit exactly.
  const fit = j.fitTo !== undefined ? { fitS: j.fitTo, srcS: Number.isFinite(len) ? len : undefined } : undefined;
  await convertOnce(src, out, fit ?? trim);
  return out;
}

// ── Understand ─────────────────────────────────────────────────────────────

interface UnderstandJob { id: string; finalState?: 'done' | 'failed'; result?: any; error?: string; createdAt: number; finishedAt?: number }
const understands = new Map<string, UnderstandJob>();

/** Start analysis; resolves with the result if it finishes within waitMs, else null (poll). */
export async function startUnderstand(id: string, filePath: string, waitMs = 5000): Promise<{ id: string; result: any | null }> {
  const u: UnderstandJob = { id, createdAt: Date.now() };
  understands.set(id, u);
  const work = call('POST', '/api/analyze', { audioUrl: filePath }).then(({ status, data }) => {
    if (status !== 200) throw upstreamError(status, data);
    u.result = {
      api: DAW_API_VERSION,
      id,
      analysis_id: id,
      state: 'done',
      bpm: data.bpm ?? null,
      key: [data.key, data.scale].filter(Boolean).join(' ') || null,
      timesig: data.timeSignature ?? data.time_signature ?? null,
      tags: data.tags ?? [],
    };
    u.finalState = 'done';
    u.finishedAt = Date.now();
  }).catch((e: any) => { u.finalState = 'failed'; u.finishedAt = Date.now(); u.error = e?.message ?? String(e); });
  await Promise.race([work, new Promise((r) => setTimeout(r, waitMs))]);
  return { id, result: u.finalState ? getUnderstand(id) : null };
}

export function getUnderstand(id: string) {
  const u = understands.get(id);
  if (!u) throw new DawError(404, 'not_found', `unknown analysis ${id}`);
  if (u.finalState === 'done') return u.result;
  if (u.finalState === 'failed') return { api: DAW_API_VERSION, id, analysis_id: id, state: 'failed', error: u.error };
  return { api: DAW_API_VERSION, id, analysis_id: id, state: 'running' };
}

export const newId = () => uuidv4();

/** Test hooks for the job map. */
export const __test = {
  addJob: (j: FacadeJob) => { jobs.set(j.id, j); },
  hasJob: (id: string) => jobs.has(id),
  removeJob: (id: string) => { jobs.delete(id); },
  getJob: (id: string) => jobs.get(id),
};
