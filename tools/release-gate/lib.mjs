// tools/release-gate/lib.mjs — shared helpers for the release-gate probes.
//
// Every tier file imports from here. Nothing in this file knows about tiers;
// it is HTTP against a running HOT-Step server (GATE_URL), a small state file
// the tiers share (GATE_RUN_DIR/state.json), WAV parsing, and a few numeric
// comparisons. Only node built-ins.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

export const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO = path.resolve(HERE, '..', '..');
export const BASE = (process.env.GATE_URL || 'http://localhost:3001').replace(/\/$/, '');
/** Root of the app under test: the extracted release, or the repo. */
export const APP_ROOT = process.env.GATE_APP_ROOT || REPO;
export const RUN_DIR = process.env.GATE_RUN_DIR || path.join(REPO, 'logs', 'release-gate', 'adhoc');
fs.mkdirSync(RUN_DIR, { recursive: true });

export const MIN = 60_000;
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── Shared state (one JSON file; each tier runs in its own process) ──────────
const STATE_PATH = path.join(RUN_DIR, 'state.json');
export const state = {
  read() { try { return JSON.parse(fs.readFileSync(STATE_PATH, 'utf8')); } catch { return {}; } },
  get(key, fallback) { const v = this.read()[key]; return v === undefined ? fallback : v; },
  set(key, value) { const s = this.read(); s[key] = value; fs.writeFileSync(STATE_PATH, JSON.stringify(s, null, 2)); },
  push(key, value) { const a = this.get(key, []); a.push(value); this.set(key, a); },
};

// ── Fixture text ─────────────────────────────────────────────────────────────
export const CAPTION = 'warm indie pop, acoustic guitar, soft brushed drums, female vocal, 100 bpm';
export const LYRICS = [
  '[Verse]',
  'Morning light on the kitchen floor',
  'Coffee steam and an open door',
  'Keys still hanging where they fell',
  'Nothing left to sell',
  '[Chorus]',
  'We keep on walking, we keep on walking',
  'Into the day, into the day',
].join('\n');
export const MM3_CAPTION = [
  'Global Metadata: Indie pop; medium tempo around 100 BPM; C major; warm and hopeful throughout; clean modern production with acoustic guitar and soft drums.',
  'Vocal Details: One female lead vocal, soft breathy timbre, mid register, gentle conversational delivery, light harmonies in the chorus, no vocal effects.',
  'Arrangement: Intro with fingerpicked acoustic guitar; verse adds soft brushed drums and bass; chorus opens up with strummed guitar and stacked harmonies; short outro fades on guitar.',
].join('\n');

// ── HTTP ─────────────────────────────────────────────────────────────────────
let token = null;
export async function auth() {
  if (token) return token;
  token = state.get('token') || null;
  if (token) return token;
  const r = await api('GET', '/api/auth/auto');
  token = r.token;
  state.set('token', token);
  return token;
}

/** Raw call: never throws on HTTP errors. Returns { status, body } (body parsed when JSON). */
export async function apiRaw(method, route, body, { auth: useAuth = false, timeoutMs = 60_000 } = {}) {
  const headers = {};
  const isForm = typeof FormData !== 'undefined' && body instanceof FormData;
  if (body !== undefined && !isForm) headers['content-type'] = 'application/json';
  if (useAuth) headers.authorization = `Bearer ${await auth()}`;
  const res = await fetch(BASE + route, {
    method,
    headers,
    body: body === undefined ? undefined : isForm ? body : JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  const text = await res.text();
  let parsed;
  try { parsed = JSON.parse(text); } catch { parsed = text; }
  return { status: res.status, body: parsed };
}

/** Call that throws on a non-2xx status with the server's error text. */
export async function api(method, route, body, opts) {
  const { status, body: out } = await apiRaw(method, route, body, opts);
  if (status < 200 || status >= 300) {
    const msg = typeof out === 'object' && out ? (out.error ?? JSON.stringify(out)) : String(out).slice(0, 300);
    const err = new Error(`${method} ${route} -> ${status}: ${msg}`);
    err.status = status;
    err.body = out;
    throw err;
  }
  return out;
}

export async function download(url) {
  const res = await fetch(url.startsWith('http') ? url : BASE + url, { signal: AbortSignal.timeout(2 * MIN) });
  if (!res.ok) throw new Error(`GET ${url} -> ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

// ── App state guards ─────────────────────────────────────────────────────────
export async function health() {
  return api('GET', '/api/health', undefined, { timeoutMs: 5000 });
}

export async function waitEngineReady(timeoutMs = 15 * MIN) {
  const started = Date.now();
  let last = null;
  for (;;) {
    try {
      last = await health();
      if (last?.engine?.ready) return last;
    } catch { last = null; }
    if (Date.now() - started > timeoutMs) {
      throw new Error(`engine not ready after ${Math.round(timeoutMs / 1000)} s: ${last?.engine?.bootStatus ?? 'server unreachable'}`);
    }
    await sleep(3000);
  }
}

/** Throws if anything else owns the GPU: a queued/running generation or training job. */
export async function assertIdle() {
  const q = await api('GET', '/api/generate/queue');
  if (q.running || q.pending) throw new Error(`generation queue is busy (running=${q.running}, pending=${q.pending})`);
  const { jobs = [] } = await api('GET', '/api/training/jobs');
  const live = jobs.find((j) => j.status === 'running' || j.status === 'queued');
  if (live) throw new Error(`training job ${live.kind} ${live.id} is ${live.status}`);
}

export async function activeBackend() {
  return (await api('GET', '/api/backends')).activeId;
}

export async function setBackend(id) {
  if ((await activeBackend()) !== id) await api('POST', '/api/backends/active', { id });
  await waitEngineReady();
}

/** Model catalogue for a backend; `ok` when it has something in every bucket named. */
export async function backendModels(id, need = []) {
  const m = await api('GET', `/api/backends/models?backend=${encodeURIComponent(id)}`);
  const buckets = m.buckets ?? {};
  const missing = need.filter((b) => !(Array.isArray(buckets[b]) && buckets[b].length));
  return { ...m, ok: missing.length === 0, missing };
}

// ── Generation ───────────────────────────────────────────────────────────────
/**
 * POST /api/generate, poll to a terminal state, download take 0 into
 * RUN_DIR/renders/<label>.wav. Throws on failed/cancelled/timeout with the
 * server's error text so the tier report says why.
 */
export async function generate(label, body, { timeoutMs = 15 * MIN } = {}) {
  const started = Date.now();
  const { jobId } = await api('POST', '/api/generate', body, { auth: true });
  if (!jobId) throw new Error(`${label}: no jobId in the response`);
  let last;
  for (;;) {
    last = await api('GET', `/api/generate/status/${jobId}`);
    if (last.status === 'succeeded') break;
    if (last.status === 'failed' || last.status === 'cancelled') {
      throw new Error(`${label}: ${last.status}: ${last.error ?? 'no error text'}`);
    }
    if (Date.now() - started > timeoutMs) {
      throw new Error(`${label}: timed out after ${Math.round(timeoutMs / 1000)} s (last stage: ${last.stage ?? '?'} ${last.progress ?? ''}%)`);
    }
    await sleep(2000);
  }
  const audioUrls = last.result?.audioUrls ?? [];
  if (!audioUrls.length) throw new Error(`${label}: succeeded but returned no audioUrls`);
  const file = await keep(label, audioUrls[0]);
  return {
    jobId,
    audioUrls,
    songIds: last.result?.songIds ?? [],
    result: last.result,
    seconds: Math.round((Date.now() - started) / 1000),
    file,
  };
}

/** Copy a served audio URL into the run folder and remember it for the ear test. */
export async function keep(label, url) {
  const dir = path.join(RUN_DIR, 'renders');
  fs.mkdirSync(dir, { recursive: true });
  const buf = await download(url);
  const file = path.join(dir, `${label}${path.extname(url.split('?')[0]) || '.wav'}`);
  fs.writeFileSync(file, buf);
  state.push('renders', { label, url, file });
  return file;
}

/** Newest per-generation log for a job, when the app's log folder is on this disk. */
export function generationLog(jobId) {
  const root = path.join(APP_ROOT, 'logs');
  if (!fs.existsSync(root)) return null;
  const sessions = fs.readdirSync(root).filter((d) => /^\d{4}-\d{2}-\d{2}_/.test(d)).sort().reverse();
  for (const s of sessions) {
    const gen = path.join(root, s, 'generations');
    if (!fs.existsSync(gen)) continue;
    const hit = fs.readdirSync(gen).find((f) => f.startsWith(`gen_${jobId}_`));
    if (hit) return path.join(gen, hit);
  }
  return null;
}

// ── Training jobs ────────────────────────────────────────────────────────────
export async function pollTrainingJob(jobId, label, { timeoutMs = 30 * MIN } = {}) {
  const started = Date.now();
  for (;;) {
    const r = await api('GET', `/api/training/jobs/${jobId}`);
    const j = r.job ?? r;
    if (j.status === 'done') return j;
    if (j.status === 'failed' || j.status === 'cancelled') {
      throw new Error(`${label}: ${j.status}: ${j.error ?? 'no error text'}`);
    }
    if (Date.now() - started > timeoutMs) {
      throw new Error(`${label}: timed out after ${Math.round(timeoutMs / 1000)} s (${j.phase ?? ''} ${j.done}/${j.total})`);
    }
    await sleep(3000);
  }
}

// ── WAV + numeric checks ─────────────────────────────────────────────────────
/** Parse a RIFF/WAVE buffer into mono float samples. PCM 8/16/24/32 and float 32/64. */
export function readWav(buf) {
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('not a RIFF/WAVE file');
  }
  let off = 12;
  let fmt = null;
  let data = null;
  while (off + 8 <= buf.length) {
    const id = buf.toString('ascii', off, off + 4);
    const size = buf.readUInt32LE(off + 4);
    const body = off + 8;
    if (id === 'fmt ') {
      fmt = {
        format: buf.readUInt16LE(body),
        channels: buf.readUInt16LE(body + 2),
        rate: buf.readUInt32LE(body + 4),
        bits: buf.readUInt16LE(body + 14),
        sub: size >= 26 ? buf.readUInt16LE(body + 24) : 0,
      };
    } else if (id === 'data') {
      data = buf.subarray(body, Math.min(body + size, buf.length));
    }
    off = body + size + (size & 1);
  }
  if (!fmt || !data) throw new Error('WAV is missing its fmt or data chunk');
  const format = fmt.format === 0xfffe ? fmt.sub : fmt.format;
  const { channels, bits, rate } = fmt;
  const bps = bits / 8;
  const frames = Math.floor(data.length / (bps * channels));
  const mono = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let acc = 0;
    for (let c = 0; c < channels; c++) {
      const p = (i * channels + c) * bps;
      let v;
      if (format === 3) v = bits === 32 ? data.readFloatLE(p) : data.readDoubleLE(p);
      else if (bits === 16) v = data.readInt16LE(p) / 32768;
      else if (bits === 24) v = (((data[p] | (data[p + 1] << 8) | (data[p + 2] << 16)) << 8) >> 8) / 8388608;
      else if (bits === 32) v = data.readInt32LE(p) / 2147483648;
      else if (bits === 8) v = (data[p] - 128) / 128;
      else throw new Error(`unsupported WAV: format ${format}, ${bits} bits`);
      acc += v;
    }
    mono[i] = acc / channels;
  }
  return { rate, channels, bits, mono, seconds: frames / rate };
}

export function wavFile(file) {
  return readWav(fs.readFileSync(file));
}

export function rms(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / Math.max(1, x.length));
}

/** Normalised zero-lag correlation over the common length. 1 = identical shape. */
export function correlation(a, b) {
  const n = Math.min(a.length, b.length);
  if (!n) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
}

/**
 * Mechanical audio check: parses, is at least `minSeconds` long, and is not
 * silence. This says nothing about quality. Only the ear does that.
 */
export function assertAudible(assert, file, label, minSeconds = 1) {
  const w = wavFile(file);
  assert.ok(w.seconds >= minSeconds, `${label}: only ${w.seconds.toFixed(2)} s of audio`);
  const level = rms(w.mono);
  assert.ok(level > 1e-4, `${label}: audio is silent (rms ${level.toExponential(2)})`);
  return w;
}

// ── Spectral fingerprint (tier 7 goldens) ────────────────────────────────────
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let j = 0; j < len / 2; j++) {
        const a = i + j;
        const b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/** Log-band energy per 100 ms frame: 48 log-spaced bands, 50 Hz to 16 kHz. */
export function fingerprint(mono, rate, { frame = 2048, hopSec = 0.1, bands = 48 } = {}) {
  const hop = Math.max(1, Math.round(rate * hopSec));
  const lo = 50;
  const hi = Math.min(16000, rate / 2);
  const binHz = rate / frame;
  const win = new Float64Array(frame);
  for (let i = 0; i < frame; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / frame);
  const rows = [];
  for (let start = 0; start + frame <= mono.length; start += hop) {
    const re = new Float64Array(frame);
    const im = new Float64Array(frame);
    for (let i = 0; i < frame; i++) re[i] = mono[start + i] * win[i];
    fft(re, im);
    const row = new Array(bands).fill(0);
    for (let k = 1; k < frame / 2; k++) {
      const hz = k * binHz;
      if (hz < lo || hz >= hi) continue;
      const b = Math.min(bands - 1, Math.floor((Math.log(hz / lo) / Math.log(hi / lo)) * bands));
      row[b] += re[k] * re[k] + im[k] * im[k];
    }
    rows.push(row.map((p) => +Math.log10(p + 1e-9).toFixed(3)));
  }
  return rows;
}

/** Mean per-frame Pearson correlation of two fingerprints. 1 = same spectrum shape. */
export function fingerprintSimilarity(a, b) {
  const n = Math.min(a.length, b.length);
  if (!n) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i];
    const y = b[i];
    const mx = x.reduce((s, v) => s + v, 0) / x.length;
    const my = y.reduce((s, v) => s + v, 0) / y.length;
    sum += correlation(x.map((v) => v - mx), y.map((v) => v - my));
  }
  return sum / n;
}

// ── Subprocess ───────────────────────────────────────────────────────────────
/**
 * Run a command to completion. `npx` (a .cmd on Windows) has to go through the
 * shell, so it is passed as one pre-quoted string; everything else is spawned
 * directly so paths with spaces survive.
 */
export function sh(cmd, args, { cwd = REPO, timeoutMs = 20 * MIN, env } = {}) {
  const viaShell = /^(npx|npm)$/.test(cmd) || /\.(cmd|bat)$/i.test(cmd);
  const q = (a) => (/[\s"]/.test(a) ? `"${a.replace(/"/g, '\\"')}"` : a);
  // This process is itself a node:test child. A nested `node --test` or
  // `tsx --test` that inherits NODE_TEST_CONTEXT sees "run() called
  // recursively", runs nothing and exits 0. Strip it.
  const childEnv = { ...process.env, ...(env ?? {}) };
  delete childEnv.NODE_TEST_CONTEXT;
  const common = { cwd, encoding: 'utf8', timeout: timeoutMs, maxBuffer: 64 << 20, windowsHide: true, env: childEnv };
  const r = viaShell
    ? spawnSync([cmd, ...args].map(q).join(' '), { ...common, shell: true })
    : spawnSync(cmd, args, common);
  return { status: r.status, out: `${r.stdout || ''}${r.stderr || ''}`, error: r.error };
}

export function walk(dir, match) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name !== 'node_modules') out.push(...walk(p, match));
    } else if (match.test(e.name)) out.push(p);
  }
  return out;
}

/** Last lines of a blob of output, for assertion messages. */
export function tail(text, lines = 40) {
  return text.trim().split(/\r?\n/).slice(-lines).join('\n');
}
