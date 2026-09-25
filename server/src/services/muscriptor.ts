// muscriptor.ts — MuScriptor (audio→MIDI) shared helpers
//
// MuScriptor is developed by Kyutai & Mirelo (Rouard, Krause, Roebel,
// Simon-Gabriel, Défossez — arXiv:2607.08168). Code MIT; model weights
// CC BY-NC 4.0 and GATED on Hugging Face (users request access + read token).
//
// NOTE (2026-07-16): the original integration ran the upstream Python CLI in
// a managed venv. That approach was removed — transcription is being ported
// to a native GGML binary (`ace-midi`). Design: docs/plans/muscriptor-cpp-port.md.
// What remains here is the part the C++ path also needs: the Hugging Face
// token store used to download the gated weights.
//
// Rob's local venv at data/muscriptor/venv is intentionally NOT deleted —
// it is the numerics-validation oracle for the port (see design doc §6).

import path from 'path';
import fs from 'fs';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { config } from '../config.js';

export const MUSCRIPTOR_DIR = path.join(config.data.dir, 'muscriptor');

export const MUSCRIPTOR_MODELS = ['small', 'medium', 'large'] as const;
export type MuscriptorModel = typeof MUSCRIPTOR_MODELS[number];

export const HF_MODEL_REPOS: Record<MuscriptorModel, string> = {
  small: 'MuScriptor/muscriptor-small',
  medium: 'MuScriptor/muscriptor-medium',
  large: 'MuScriptor/muscriptor-large',
};

// ── Hugging Face access token ────────────────────────────────────────────

const HF_TOKEN_PATH = path.join(MUSCRIPTOR_DIR, 'hf_token');

export function getHfToken(): string | null {
  try {
    const t = fs.readFileSync(HF_TOKEN_PATH, 'utf-8').trim();
    return t || null;
  } catch { return null; }
}

export function setHfToken(token: string): void {
  fs.mkdirSync(MUSCRIPTOR_DIR, { recursive: true });
  const t = (token || '').trim();
  if (!t) {
    fs.rmSync(HF_TOKEN_PATH, { force: true });
  } else {
    // A token that is not a plain hf_ string surfaces later as a gated-repo
    // 401 that looks like an access problem (#183). Refuse it here instead.
    if (!/^hf_[A-Za-z0-9]{10,}$/.test(t)) {
      const err = new Error('That does not look like a Hugging Face token: it should start with hf_ and contain only letters and digits.');
      (err as any).status = 400;
      throw err;
    }
    fs.writeFileSync(HF_TOKEN_PATH, t, { encoding: 'utf-8' });
  }
}

/** Heuristic: does this failure look like a gated-model / auth problem? */
export function looksLikeGatedError(text: string): boolean {
  return /gated|401|403|unauthorized|forbidden|restricted|access to model|awaiting a review|accept the conditions|not authenticated|invalid (user )?token|authentication/i.test(text);
}

// ── ace-midi engine binary ───────────────────────────────────────────────

/** Absolute path to ace-midi, or null if not built/shipped. Lives next to
 *  ace-server (same build output / portable layout). */
export function aceMidiExe(): string | null {
  const dir = path.dirname(config.aceServer.exe);
  const exe = path.join(dir, process.platform === 'win32' ? 'ace-midi.exe' : 'ace-midi');
  return fs.existsSync(exe) ? exe : null;
}

// ── Model weights (gated on HF — downloaded with the user's read token) ──

// MUSCRIPTOR_MODELS_DIR (Settings > Environment) mirrors ACESTEPCPP_MODELS so
// the gated weights can live outside the install (#86). Read at import, so a
// change needs a restart.
export const MODELS_DIR = process.env.MUSCRIPTOR_MODELS_DIR || path.join(config.data.dir, 'models', 'muscriptor');

export function modelDir(m: MuscriptorModel): string {
  return path.join(MODELS_DIR, m);
}

export function isModelDownloaded(m: MuscriptorModel): boolean {
  return fs.existsSync(path.join(modelDir(m), 'model.safetensors'))
      && fs.existsSync(path.join(modelDir(m), 'config.json'));
}

export interface ModelDownloadState {
  downloading: boolean;
  receivedBytes: number;
  totalBytes: number;
  error?: string;
  gated?: boolean;
}

const downloads = new Map<MuscriptorModel, ModelDownloadState>();

export function getModelStates(): Record<string, ModelDownloadState & { downloaded: boolean; sizeBytes: number }> {
  const out: Record<string, any> = {};
  for (const m of MUSCRIPTOR_MODELS) {
    const dl = downloads.get(m);
    let sizeBytes = 0;
    const downloaded = isModelDownloaded(m);
    if (downloaded) {
      try { sizeBytes = fs.statSync(path.join(modelDir(m), 'model.safetensors')).size; } catch { /* ignore */ }
    }
    out[m] = {
      downloaded,
      sizeBytes,
      downloading: dl?.downloading ?? false,
      receivedBytes: dl?.receivedBytes ?? 0,
      totalBytes: dl?.totalBytes ?? 0,
      error: dl?.error,
      gated: dl?.gated,
    };
  }
  return out;
}

async function fetchToFile(url: string, token: string | null, dest: string,
                           onProgress?: (received: number, total: number) => void): Promise<void> {
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  const res = await fetch(url, { headers, redirect: 'follow' });
  if (!res.ok || !res.body) {
    const body = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status} fetching ${url}: ${body.slice(0, 300)}`);
  }
  const total = parseInt(res.headers.get('content-length') || '0', 10);
  let received = 0;
  const src = Readable.fromWeb(res.body as any);
  src.on('data', (chunk: Buffer) => {
    received += chunk.length;
    onProgress?.(received, total);
  });
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const tmp = dest + '.part';
  await pipeline(src, fs.createWriteStream(tmp));
  fs.renameSync(tmp, dest);
}

/**
 * Begin downloading a model's config.json + model.safetensors from its gated
 * HF repo. Fire-and-poll: progress via getModelStates(). Single-flight.
 */
export function startModelDownload(m: MuscriptorModel): { started: boolean; error?: string } {
  if (downloads.get(m)?.downloading) return { started: false, error: 'Download already in progress' };
  if (isModelDownloaded(m)) return { started: false, error: 'Already downloaded' };

  const state: ModelDownloadState = { downloading: true, receivedBytes: 0, totalBytes: 0 };
  downloads.set(m, state);
  const token = getHfToken();
  const base = `https://huggingface.co/${HF_MODEL_REPOS[m]}/resolve/main`;

  (async () => {
    try {
      console.log(`[MidiStudio] Downloading ${m} weights from ${HF_MODEL_REPOS[m]}`);
      await fetchToFile(`${base}/config.json`, token, path.join(modelDir(m), 'config.json'));
      await fetchToFile(`${base}/model.safetensors`, token, path.join(modelDir(m), 'model.safetensors'),
        (received, total) => { state.receivedBytes = received; state.totalBytes = total; });
      console.log(`[MidiStudio] ${m} weights downloaded (${(state.receivedBytes / 1e9).toFixed(2)} GB)`);
    } catch (err: any) {
      state.error = err.message || 'Download failed';
      state.gated = looksLikeGatedError(state.error ?? '');
      console.error(`[MidiStudio] ${m} download FAILED: ${state.error}`);
    } finally {
      state.downloading = false;
    }
  })();

  return { started: true };
}
