// modelDownloadService.ts — Concurrent, resumable model downloads from HuggingFace
//
// Downloads GGUF files to the configured models directory with:
// - HTTP Range-based resumption (.part files)
// - Concurrent downloads (no artificial limit)
// - Progress tracking with speed + ETA
// - EventEmitter for SSE progress streaming

import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import https from 'https';
import http from 'http';
import { createHash, randomUUID } from 'crypto';
import { fileURLToPath } from 'url';
import { config, PORTABLE_MODE, PROJECT_ROOT, RUNTIME_LIB_DIR } from '../config.js';
import { getBusyBlockers, registerBusyCheck } from '../dataroot/pointer.js';
import type { RegistryFileEntry as RegistryFile, ModelRegistryResponse, DownloadStatus, DownloadJob } from '../contracts/modelManager.js';
import { MODEL_ROOTS, getHfToken, looksLikeGatedError, type DownloadableModel, type ModelDownloadState } from './muscriptor.js';
import { MODEL_FILES, resolveModelDir, targetRoot } from './muscriptorPaths.js';

// Load registry - resolve path based on mode:
// - Dev mode: relative to source file (../data/model-registry.json from services/)
// - Portable mode: PROJECT_ROOT/server/data/model-registry.json

const registryPath = PORTABLE_MODE
  ? path.join(PROJECT_ROOT, 'server', 'data', 'model-registry.json')
  : path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'data', 'model-registry.json');
const registry = JSON.parse(fs.readFileSync(registryPath, 'utf-8'));

// Engine variant detection — controls which registry items are visible.
// 'cuda' (default for legacy/pre-v1.1 builds) shows everything;
// 'vulkan' or 'cpu' hides CUDA-specific files, packs, and runtime.
function detectEngineVariant(): string {
  try {
    const variantFile = path.join(path.dirname(config.aceServer.exe), '.variant');
    if (fs.existsSync(variantFile)) {
      return fs.readFileSync(variantFile, 'utf-8').trim();
    }
  } catch {}
  return 'cuda'; // Assume CUDA if no marker
}
const ENGINE_VARIANT = detectEngineVariant();

/**
 * What actually went wrong, in a form a user can act on or paste into an issue.
 *
 * `err.message` alone is not enough, and #171 is why: when every address of a
 * host fails, Node rejects with an AggregateError whose message is the EMPTY
 * STRING and whose real causes sit in `.errors`. Three retries then logged
 * "Download failed after 3 attempts: yue2-lm-bf16.gguf — " with nothing after
 * the dash, and the reporter (and we) had no idea whether it was DNS, IPv6, a
 * proxy, a blocked CDN host or a full disk.
 *
 * So: unwrap the aggregate, name the errno and the address it could not reach,
 * and say which HOST the attempt was against — for a Hugging Face LFS file
 * that is a CDN redirect target, not huggingface.co, and the difference is the
 * whole diagnosis.
 */
export function describeDownloadError(err: any): string {
  if (!err) return 'unknown error';
  const message = typeof err.message === 'string' ? err.message.trim() : '';
  const parts: string[] = [message || err.name || String(err)];

  if (Array.isArray(err.errors) && err.errors.length) {
    const causes = err.errors.map((e: any) => {
      const where = e?.address ? `${e.address}${e.port ? `:${e.port}` : ''}` : '';
      return [e?.code, e?.syscall, where].filter(Boolean).join(' ') || e?.message || String(e);
    });
    parts.push(`(${[...new Set(causes)].join('; ')})`);
  } else if (err.code) {
    const where = err.address ? `${err.address}${err.port ? `:${err.port}` : ''}` : '';
    parts.push(`(${[err.code, err.syscall, where].filter(Boolean).join(' ')})`);
  }

  if (typeof err.url === 'string') {
    try { parts.push(`while fetching ${new URL(err.url).host}`); } catch { /* not a URL: say nothing */ }
  }
  return parts.join(' ');
}

/** The Authorization header for one hop, or nothing. The token goes only to
 *  huggingface.co over https; a redirect to a CDN or any other host gets none,
 *  so a pre-signed URL is never sent a credential it would reject or log. */
export function hfAuthHeader(url: URL, token?: string): Record<string, string> {
  return token && url.protocol === 'https:' && /(^|\.)huggingface\.co$/.test(url.hostname)
    ? { Authorization: `Bearer ${token}` }
    : {};
}

// Detect CUDA major version for runtime DLL selection
function detectCudaMajorVersion(): number {
  try {
    const versionFile = path.join(path.dirname(config.aceServer.exe), '.cuda-version');
    if (fs.existsSync(versionFile)) {
      return parseInt(fs.readFileSync(versionFile, 'utf-8').trim(), 10);
    }
  } catch {}
  return 13; // Default: assume CUDA 13 (latest)
}
const CUDA_MAJOR = detectCudaMajorVersion();

// IDs of CUDA-only file entries
const CUDA_ONLY_FILE_PREFIXES = ['cuda-rt-', 'supersep-rt-', 'trt-rt-', 'mm3-dit-trt-'];

/** True for catalogue entries that can only ever run on Windows.
 *
 *  Every cuda-rt-* and supersep-rt-* entry is a .dll. Until now the only
 *  gating was by engine variant (cuda / vulkan / cpu), which says nothing
 *  about the operating system, so a Linux box running the CUDA variant was
 *  offered cublas64_13.dll and friends, downloaded them, and dropped them
 *  next to a Linux binary where they sit inert (issue #81). There are no
 *  Linux or macOS equivalents in the catalogue: on those platforms the CUDA
 *  runtime comes from the distro or the toolkit, so the right behaviour is
 *  to hide these entries rather than substitute anything.
 *
 *  Keyed off the filename rather than a new registry field so an entry added
 *  later cannot forget to declare itself. */
const IS_WINDOWS = process.platform === 'win32';
function isWindowsOnlyFile(f: { filename?: string }): boolean {
  return typeof f.filename === 'string' && f.filename.toLowerCase().endsWith('.dll');
}
// IDs of CUDA-only packs (hidden entirely for non-CUDA)
const CUDA_ONLY_PACKS = ['cuda-runtime', 'cuda12-runtime', 'supersep-runtime', 'blackwell'];

// ── Types ───────────────────────────────────────────────────
// DownloadStatus and DownloadJob are shared with the client — imported above
// from contracts/modelManager.ts — so this service's job shape can't drift
// from what GET /api/model-manager/downloads actually streams.

interface InternalJob extends DownloadJob {
  abortController?: AbortController;
  speedSamples: { time: number; bytes: number }[];
  hfToken?: string;   // Optional Hugging Face token for gated repos
}

// RegistryFile (the catalogue entry, before installed-status enrichment) and
// RegistryCompanion are imported from contracts/modelManager.ts — that file
// also documents the companion-file behavior this interface used to.

// ── Service ─────────────────────────────────────────────────

/** A required companion fetched to `tmp`, waiting to be renamed to `dest`. */
interface Staged { tmp: string; dest: string }

class ModelDownloadService extends EventEmitter {
  private jobs = new Map<string, InternalJob>();

  /** Get the models directory path */
  get modelsDir(): string {
    return config.aceServer.models;
  }

  /** Where runtime DLLs go: the data folder's tools/runtime in a release
   *  (on the engine's PATH via trtLibs), beside ace-server in dev. */
  get engineDir(): string {
    return RUNTIME_LIB_DIR ?? path.dirname(config.aceServer.exe);
  }

  /** Resolve the target directory for a registry file */
  private getTargetDir(file: RegistryFile): string {
    if (file.muscriptorModel) return path.join(targetRoot(this._muscriptorRoots()), file.muscriptorModel);
    if (file.role === 'runtime') {
      // Runtime DLLs go alongside ace-server.exe
      return this.engineDir;
    }
    return file.subdir
      ? path.join(this.modelsDir, file.subdir)
      : this.modelsDir;
  }

  /** MuScriptor folders as the runtime reads them (a seam for tests). */
  private _muscriptorRoots() { return MODEL_ROOTS; }

  /** Where an installed copy of this entry actually is. Differs from the
   *  download target for a MuScriptor model still in its pre-#228 folder. */
  private _installedDir(file: RegistryFile): string {
    return file.muscriptorModel
      ? resolveModelDir(this._muscriptorRoots(), file.muscriptorModel).dir
      : this.getTargetDir(file);
  }

  /** Installed means this entry's own files, by exact path. Matching on the
   *  basename would show every MuScriptor variant installed as soon as one
   *  held a model.safetensors. Weights and required companions are one group. */
  private _isInstalled(file: RegistryFile, byName: Set<string>): boolean {
    if (!file.muscriptorModel) return byName.has(file.filename);
    return resolveModelDir(this._muscriptorRoots(), file.muscriptorModel).complete;
  }

  /** Get all files in the registry, enriched with installed status.
   *  Filters out CUDA-specific entries for non-CUDA engine variants. */
  getRegistry(): ModelRegistryResponse {
    const installed = this.getInstalledFiles();
    const isCuda = ENGINE_VARIANT === 'cuda';
    const wrongCudaTag = CUDA_MAJOR <= 12 ? 'cuda13' : 'cuda12';

    // Filter files: hide CUDA-only entries for non-CUDA builds,
    // and hide wrong-CUDA-version entries
    const filteredFiles = registry.files
      .filter((f: RegistryFile) => isCuda || !CUDA_ONLY_FILE_PREFIXES.some(p => f.id.startsWith(p)))
      .filter((f: RegistryFile) => IS_WINDOWS || !isWindowsOnlyFile(f))
      .filter((f: RegistryFile) => !f.tags?.includes(wrongCudaTag))
      .map((f: RegistryFile) => ({
        ...f,
        installed: this._isInstalled(f, installed),
        outdated: this._isInstalled(f, installed) && this._isKnownStale(f),
      }));

    /** Ids of the files a client on THIS platform can actually see. */
    const visibleFileIds = new Set<string>(filteredFiles.map((f: RegistryFile) => f.id));

    // Filter packs: hide CUDA-only packs, strip CUDA file IDs from remaining
    // Also hide the wrong-version CUDA runtime pack and remap IDs in model packs
    const filteredPacks = registry.packs
      .filter((p: any) => isCuda || !CUDA_ONLY_PACKS.includes(p.id))
      // A pack whose every file is a Windows DLL has nothing to offer on
      // Linux or macOS. Without this the pack survives the isCuda gate on a
      // Linux CUDA install and renders with all of its files filtered away.
      .filter((p: any) => {
        if (IS_WINDOWS) return true;
        const ids: string[] = Array.isArray(p.fileIds) ? p.fileIds : [];
        if (ids.length === 0) return true;
        return !ids.every((id: string) => {
          const f = registry.files.find((x: RegistryFile) => x.id === id);
          return f ? isWindowsOnlyFile(f) : false;
        });
      })
      .filter((p: any) => {
        // Show only the correct CUDA runtime pack
        if (p.id === 'cuda-runtime') return CUDA_MAJOR >= 13;
        if (p.id === 'cuda12-runtime') return CUDA_MAJOR <= 12;
        return true;
      })
      .map((p: any) => {
        // fileIds comes from model-registry.json, so a hand-edited catalogue
        // can ship a pack without it. Unguarded, that threw here and took the
        // whole /registry response down on every non-CUDA install, and threw
        // again in the UI's getPackFiles on CUDA (issue #120).
        const packFileIds: string[] = Array.isArray(p.fileIds) ? p.fileIds : [];
        if (!Array.isArray(p.fileIds)) {
          console.warn(`[Models] Pack "${p.id}" has no fileIds array — listing it empty.`);
        }
        if (!isCuda) {
          return { ...p, fileIds: packFileIds.filter((id: string) => !CUDA_ONLY_FILE_PREFIXES.some(pfx => id.startsWith(pfx))) };
        }
        // Remap cuda-rt-* IDs in packs to version-specific ones for CUDA 12
        if (CUDA_MAJOR <= 12) {
          return {
            ...p,
            fileIds: packFileIds.map((id: string) => {
              if (['cuda-rt-cublas', 'cuda-rt-cublaslt', 'cuda-rt-cudart'].includes(id)) {
                return `${id}-12`;
              }
              return id;
            }),
          };
        }
        return { ...p, fileIds: packFileIds };
      })
      // Final pass: no pack may reference a file that did not survive the
      // filters above. The per-branch stripping only covers the non-CUDA case,
      // so a Linux CUDA install kept cuda-rt-* ids in mixed packs like
      // quick-start while the files themselves had just been hidden by the
      // Windows-only rule — 12 dangling ids, which the UI renders as a pack
      // that can never reach "installed". Reconciling against the filtered
      // list here means a new platform rule cannot reintroduce the same skew.
      .map((p: any) => {
        const ids: string[] = Array.isArray(p.fileIds) ? p.fileIds : [];
        const kept = ids.filter((id: string) => visibleFileIds.has(id));
        if (kept.length === ids.length) return p;
        return { ...p, fileIds: kept };
      })
      // A pack left with nothing is not a pack.
      .filter((p: any) => !Array.isArray(p.fileIds) || p.fileIds.length > 0);

    return {
      packs: filteredPacks,
      files: filteredFiles,
      modelsDir: this.modelsDir,
      variant: ENGINE_VARIANT,
      cudaMajor: CUDA_MAJOR,
    };
  }

  /** Scan models directory for installed model files (.gguf, .onnx, .safetensors),
   *  and engine directory for runtime DLLs */
  getInstalledFiles(): Set<string> {
    const dir = this.modelsDir;
    const files = new Set<string>();

    // Scan models root directory
    if (fs.existsSync(dir)) {
      for (const f of fs.readdirSync(dir)) {
        if (f.endsWith('.gguf') || f.endsWith('.onnx') || f.endsWith('.safetensors') || f.endsWith('.bin')) files.add(f);
      }
      // Scan subdirectories (e.g. supersep/)
      for (const sub of fs.readdirSync(dir)) {
        const subPath = path.join(dir, sub);
        try {
          if (fs.statSync(subPath).isDirectory()) {
            for (const f of fs.readdirSync(subPath)) {
              if (f.endsWith('.gguf') || f.endsWith('.onnx') || f.endsWith('.safetensors') || f.endsWith('.bin')) files.add(f);
            }
          }
        } catch {}
      }
    }

    // Scan the StableStep (SA3) directory — lives two levels deep
    // (<modelsDir>/onnx/sa3) and contains tokenizer JSON files that the
    // generic scan above ignores.
    const sa3Dir = path.join(dir, 'onnx', 'sa3');
    if (fs.existsSync(sa3Dir)) {
      for (const f of fs.readdirSync(sa3Dir)) {
        if (!f.endsWith('.part')) files.add(f);
      }
    }

    // Scan mm3/ for the TensorRT DiT manifest companion (mm3-dit-trt.engine.json)
    // — the generic subdirectory scan above stops at .gguf/.onnx/.safetensors/.bin
    // and never matches .json. Deliberately scoped to mm3/ (not a generic
    // ".json anywhere" rule) so an unrelated stray .json in another subdir
    // doesn't start reading as "installed". This does NOT reach
    // mm3/mm3-trt-cache/*.engine — that's a nested sub-subdirectory this loop
    // never descends into, and .engine isn't a matched extension here or in
    // deleteFile()'s whitelist below: a built TensorRT engine is GPU- and
    // TensorRT-version-locked, so it can never be a catalogue download/delete
    // target, only a same-machine build artifact.
    const mm3Dir = path.join(dir, 'mm3');
    if (fs.existsSync(mm3Dir)) {
      for (const f of fs.readdirSync(mm3Dir)) {
        if (f.endsWith('.json')) files.add(f);
      }
    }

    // yue2/minted_{manifest.json,codes.i32} — the AR trainer's regulariser pack
    // is data, not weights, so neither extension is in the generic scan above
    // and both would otherwise read as never-installed forever in the Model
    // Manager even once downloaded.
    const yue2Dir = path.join(dir, 'yue2');
    if (fs.existsSync(yue2Dir)) {
      for (const f of fs.readdirSync(yue2Dir)) {
        if (f.endsWith('.json') || f.endsWith('.i32')) files.add(f);
      }
    }

    // Scan for runtime DLLs: the download folder, and beside ace-server, where
    // a release ships some and older versions downloaded them
    for (const engDir of new Set([this.engineDir, path.dirname(config.aceServer.exe)])) {
      if (!fs.existsSync(engDir)) continue;
      for (const f of fs.readdirSync(engDir)) {
        if (f.endsWith('.dll')) files.add(f);
      }
    }

    return files;
  }

  /** Get all active/recent download jobs */
  getJobs(): DownloadJob[] {
    return Array.from(this.jobs.values()).map(j => ({
      jobId: j.jobId,
      fileId: j.fileId,
      filename: j.filename,
      status: j.status,
      bytesDownloaded: j.bytesDownloaded,
      totalBytes: j.totalBytes,
      speed: j.speed,
      error: j.error,
    }));
  }

  /** Start downloading a file by registry ID.
   *  Optional hfToken is forwarded as `Authorization: Bearer <token>` on
   *  huggingface.co requests (needed for gated repos; empty = anonymous). */
  startDownload(fileId: string, hfToken?: string): string {
    const file = registry.files.find((f: RegistryFile) => f.id === fileId);
    if (!file) throw new Error(`Unknown file ID: ${fileId}`);

    // Check if already downloading
    for (const job of this.jobs.values()) {
      if (job.fileId === fileId && (job.status === 'downloading' || job.status === 'queued')) {
        return job.jobId; // Return existing job
      }
    }

    const jobId = randomUUID().slice(0, 8);
    const job: InternalJob = {
      jobId,
      fileId,
      filename: file.filename,
      status: 'queued',
      bytesDownloaded: 0,
      totalBytes: file.sizeBytes,
      speed: 0,
      speedSamples: [],
      // MIDI Studio keeps its own token; use it when the request carries none.
      hfToken: hfToken?.trim() || (file.muscriptorModel ? getHfToken() ?? undefined : undefined),
    };

    this.jobs.set(jobId, job);
    this._executeDownload(job, file);
    return jobId;
  }

  private _muscriptorEntry(m: DownloadableModel): RegistryFile | undefined {
    return registry.files.find((f: RegistryFile) => f.muscriptorModel === m);
  }

  /** MIDI Studio's download button: the same job, token and busy gate as the
   *  Model Manager's, so the two can never download one model twice. */
  startMuscriptorDownload(m: DownloadableModel): { started: boolean; error?: string } {
    const file = this._muscriptorEntry(m);
    if (!file) return { started: false, error: `Unknown model '${m}'` };
    if (this._activeJob(file.id)) return { started: false, error: 'Download already in progress' };
    if (resolveModelDir(this._muscriptorRoots(), m).complete) return { started: false, error: 'Already downloaded' };
    this.startDownload(file.id);
    return { started: true };
  }

  /** The newest job for a MuScriptor model, in MIDI Studio's progress shape. */
  muscriptorState(m: DownloadableModel): ModelDownloadState | undefined {
    const file = this._muscriptorEntry(m);
    const job = file && [...this.jobs.values()].reverse().find(j => j.fileId === file.id);
    if (!job || job.status === 'completed' || job.status === 'cancelled') return undefined;
    const active = job.status === 'downloading' || job.status === 'queued';
    return {
      downloading: active,
      receivedBytes: job.bytesDownloaded,
      totalBytes: job.totalBytes,
      error: active ? undefined : job.error,
      gated: !active && !!job.error && looksLikeGatedError(job.error),
    };
  }

  private _activeJob(fileId: string): InternalJob | undefined {
    for (const job of this.jobs.values()) {
      if (job.fileId === fileId && (job.status === 'downloading' || job.status === 'queued')) return job;
    }
    return undefined;
  }

  /** Cancel an active download */
  cancelDownload(jobId: string): boolean {
    const job = this.jobs.get(jobId);
    if (!job) return false;
    if (job.status !== 'downloading' && job.status !== 'queued') return false;

    job.status = 'cancelled';
    job.abortController?.abort();

    // Clean up .part file — resolve correct directory
    const file = registry.files.find((f: RegistryFile) => f.id === job.fileId);
    const targetDir = file ? this.getTargetDir(file) : this.modelsDir;
    const partPath = path.join(targetDir, `${job.filename}.part`);
    try { fs.unlinkSync(partPath); } catch {}
    for (const c of file?.companions ?? []) {
      if (c.required) try { fs.unlinkSync(path.join(targetDir, `${c.filename}.part`)); } catch {}
    }

    this.emit('progress');
    return true;
  }

  /** Resume a paused/failed download */
  resumeDownload(jobId: string): string {
    const job = this.jobs.get(jobId);
    if (!job) throw new Error(`Unknown job: ${jobId}`);
    if (job.status !== 'paused' && job.status !== 'failed') {
      throw new Error(`Job ${jobId} is ${job.status}, cannot resume`);
    }

    const file = registry.files.find((f: RegistryFile) => f.id === job.fileId);
    if (!file) throw new Error(`Registry entry gone for ${job.fileId}`);

    // Check .part file for resume offset
    const targetDir = this.getTargetDir(file);
    const partPath = path.join(targetDir, `${job.filename}.part`);
    if (fs.existsSync(partPath)) {
      job.bytesDownloaded = fs.statSync(partPath).size;
    } else {
      job.bytesDownloaded = 0;
    }

    job.status = 'queued';
    job.error = undefined;
    job.speedSamples = [];
    this._executeDownload(job, file);
    return jobId;
  }

  /** Delete a model/runtime file from disk */
  deleteFile(filename: string): boolean {
    // A MuScriptor entry is addressed by its id: several of them share the
    // name model.safetensors, so a filename cannot say which one to remove.
    const managed = registry.files.find((f: RegistryFile) => f.id === filename && f.muscriptorModel);
    if (managed) return this._deleteMuscriptor(managed);

    // Safety: only known model/runtime extensions.
    // .json covers StableStep tokenizer files under onnx/sa3; .i32 is the YuE2 minted
    // regulariser's codes blob under yue2/, which is a catalogue download and
    // so has to be deletable like any other.
    if (!filename.endsWith('.gguf') && !filename.endsWith('.onnx') && !filename.endsWith('.safetensors') && !filename.endsWith('.dll') && !filename.endsWith('.bin') && !filename.endsWith('.json') && !filename.endsWith('.i32')) {
      throw new Error('Can only delete .gguf, .onnx, .safetensors, .bin, .dll, .json, or .i32 files');
    }

    // For DLLs, check engine directory
    if (filename.endsWith('.dll')) {
      const filePath = path.join(this.engineDir, filename);
      const resolved = path.resolve(filePath);
      if (!resolved.startsWith(path.resolve(this.engineDir))) throw new Error('Path traversal denied');
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
        return true;
      }
      return false;
    }

    // Check root dir and subdirectories for models
    const candidates = [path.join(this.modelsDir, filename)];
    try {
      for (const sub of fs.readdirSync(this.modelsDir)) {
        const subPath = path.join(this.modelsDir, sub);
        if (fs.statSync(subPath).isDirectory()) {
          candidates.push(path.join(subPath, filename));
        }
      }
    } catch {}
    // StableStep (SA3) files live two levels deep: <modelsDir>/onnx/sa3
    candidates.push(path.join(this.modelsDir, 'onnx', 'sa3', filename));

    const modelsResolved = path.resolve(this.modelsDir);
    for (const filePath of candidates) {
      if (fs.existsSync(filePath)) {
        const resolved = path.resolve(filePath);
        if (!resolved.startsWith(modelsResolved)) throw new Error('Path traversal denied');
        fs.unlinkSync(filePath);
        return true;
      }
    }
    return false;
  }

  /** Remove one MuScriptor model: its weights, config and leftovers, in the
   *  folders the runtime would read it from (shared and pre-#228 legacy, or
   *  just the override). Nothing else in those folders is touched, and the
   *  folder itself goes only if that left it empty. */
  private _deleteMuscriptor(file: RegistryFile): boolean {
    const variant = file.muscriptorModel!;
    for (const j of this.jobs.values()) {
      if (j.fileId === file.id && (j.status === 'downloading' || j.status === 'queued')) {
        throw new Error(`${file.displayName} is downloading; cancel it first`);
      }
    }
    const r = this._muscriptorRoots();
    // A folder the user pointed MUSCRIPTOR_MODELS_DIR at is theirs: never deleted from.
    if (r.override) return false;
    const dirs = [r.shared, r.legacy].map(root => path.join(root, variant));
    let removed = false;
    for (const dir of dirs) {
      for (const name of MODEL_FILES) {
        for (const suffix of ['', '.part', '.sha256']) {
          try { fs.unlinkSync(path.join(dir, name + suffix)); if (!suffix) removed = true; } catch { /* not there */ }
        }
      }
      try { fs.rmdirSync(dir); } catch { /* holds other files, or gone */ }
    }
    return removed;
  }

  /** Clean up completed/cancelled/failed jobs older than 60s */
  cleanupJobs(): void {
    for (const [id, job] of this.jobs) {
      if (job.status === 'completed' || job.status === 'cancelled' || job.status === 'failed') {
        this.jobs.delete(id);
      }
    }
  }

  // ── Internal ────────────────────────────────────────────────

  /** Retry delays in ms (3 attempts: immediate, 2s, 5s) */
  private static readonly RETRY_DELAYS = [0, 2000, 5000];

  private async _executeDownload(job: InternalJob, file: RegistryFile): Promise<void> {
    // A complete pre-#228 MuScriptor copy is already installed; downloading
    // the same gigabytes into the shared folder would just duplicate it.
    if (file.muscriptorModel) {
      const found = resolveModelDir(this._muscriptorRoots(), file.muscriptorModel);
      if (found.complete && found.source === 'legacy' && await this._matchesRegistry(path.join(found.dir, file.filename), file)) {
        job.status = 'completed';
        job.bytesDownloaded = job.totalBytes;
        this.emit('progress');
        return;
      }
    }

    // Determine target directory based on file role
    const targetDir = this.getTargetDir(file);
    if (!fs.existsSync(targetDir)) {
      fs.mkdirSync(targetDir, { recursive: true });
    }

    const partPath = path.join(targetDir, `${file.filename}.part`);
    const finalPath = path.join(targetDir, file.filename);

    // Check if already fully downloaded -- and, for entries that declare a
    // sha256, that what is on disk is still what the registry describes.
    if (fs.existsSync(finalPath)) {
      if (await this._matchesRegistry(finalPath, file)) {
        // A required companion missing beside good weights is fetched now.
        const staged: Staged[] = [];
        try {
          await this._fetchRequired(file, targetDir, job, staged);
          // A cancel during the config fetch stands; it must not become "completed".
          if ((job.status as DownloadStatus) === 'cancelled') { this._discardStaged(staged); return; }
          this._publishStaged(staged);
        } catch (err: any) {
          this._discardStaged(staged);
          if ((job.status as DownloadStatus) !== 'cancelled') this._failJob(job, file, err);
          return;
        }
        job.status = 'completed';
        job.bytesDownloaded = job.totalBytes;
        this.emit('progress');
        // Installs that predate a companion (or got the weights placed by
        // hand) would otherwise never receive it.
        this._fetchCompanions(file, targetDir).catch(() => {});
        return;
      }
      console.log(`[ModelManager] ${file.filename} on disk does not match the registry sha256 — re-downloading`);
      try { fs.unlinkSync(finalPath); } catch {}
      try { fs.unlinkSync(`${finalPath}.sha256`); } catch {}
    }

    for (let attempt = 0; attempt < ModelDownloadService.RETRY_DELAYS.length; attempt++) {
      const delay = ModelDownloadService.RETRY_DELAYS[attempt];
      if (delay > 0) {
        console.log(`[ModelManager] Retry ${attempt + 1}/${ModelDownloadService.RETRY_DELAYS.length} for ${file.filename} in ${delay / 1000}s...`);
        await new Promise(r => setTimeout(r, delay));
      }

      // Check existing .part file for resume
      let startByte = 0;
      if (fs.existsSync(partPath)) {
        startByte = fs.statSync(partPath).size;
        job.bytesDownloaded = startByte;
      }

      job.status = 'downloading';
      job.abortController = new AbortController();
      job.error = undefined;
      this.emit('progress');

      const repoPath = file.repoPath || file.filename;
      // A catalogue entry with no repo builds the URL
      // huggingface.co/undefined/resolve/main/<file>. Hugging Face answers a
      // request for a repo that does not exist with 401 rather than 404, so
      // this surfaced to users as "Unauthorised" and sent them hunting for a
      // token that could never have helped (issues #87, #74). Say what is
      // actually wrong instead.
      if (!file.repo) {
        throw new Error(
          `${file.filename} has no download source in the model catalogue — ` +
          `the weights are not published yet. This is not an authentication problem.`
        );
      }
      const url = `https://huggingface.co/${file.repo}/resolve/main/${repoPath}`;

      // Required files are staged as .part first (a denied or gated repo
      // fails here, in a second, not after the weights have come down) but
      // published only once the weights are in place: a cancelled or failed
      // job must not leave a config beside missing weights.
      const staged: Staged[] = [];
      try {
        await this._fetchRequired(file, targetDir, job, staged);
        if ((job.status as DownloadStatus) === 'cancelled') { this._discardStaged(staged); return; }

        await this._downloadWithRedirects(url, partPath, job, startByte);

        if ((job.status as DownloadStatus) === 'cancelled') { this._discardStaged(staged); return; }

        // Validate downloaded file before finalising
        this._validateDownload(partPath, file);
        if (file.sha256) {
          const got = await this._hashFile(partPath);
          if (got !== file.sha256) {
            try { fs.unlinkSync(partPath); } catch {}
            throw new Error(`sha256 mismatch for ${file.filename}: expected ${file.sha256}, got ${got}`);
          }
        }

        // Rename .part to final
        fs.renameSync(partPath, finalPath);
        if (file.sha256) this._writeSidecar(finalPath, file.sha256);
        this._publishStaged(staged);
        job.status = 'completed';
        job.speed = 0;
        this.emit('progress');
        console.log(`[ModelManager] Download complete: ${file.filename}`);

        // Best-effort: fetch any companion files (e.g. a LICENSE that must
        // travel with the weights) into the same directory. Never blocks or
        // fails the parent job — errors are logged only.
        this._fetchCompanions(file, targetDir).catch(() => {});

        return; // Success — exit retry loop
      } catch (err: any) {
        this._discardStaged(staged);
        if ((job.status as DownloadStatus) === 'cancelled') return;

        const isLastAttempt = attempt === ModelDownloadService.RETRY_DELAYS.length - 1;
        const why = this._describeFailure(err, file);
        if (isLastAttempt) {
          job.status = 'failed';
          job.error = why;
          job.speed = 0;
          this.emit('progress');
          console.error(`[ModelManager] Download failed after ${attempt + 1} attempts: ${file.filename} — ${why}`);
        } else {
          console.warn(`[ModelManager] Download attempt ${attempt + 1} failed for ${file.filename}: ${why}`);
          job.speedSamples = [];
        }
      }
    }
  }

  /** Paths queued for or undergoing a background sha256 check, so a
   *  burst of registry reads schedules each file once. */
  private readonly _hashing = new Set<string>();
  private readonly _hashQueue = new Map<string, { file: RegistryFile; size: number }>();
  private _hashWorkerRunning = false;

  /** True when a sha256-declaring entry is installed and ALREADY KNOWN to
   *  hold the wrong bytes -- a republished model still sitting under its old
   *  filename. This never hashes: it answers from the <file>.sha256 sidecar
   *  and, when there is no usable one, kicks off a background hash and says
   *  "not stale" for now. The next registry read has the answer, and the UI
   *  refreshes on the 'progress' event this emits when it lands. */
  private _isKnownStale(file: RegistryFile): boolean {
    if (!file.sha256) return false;
    const filePath = path.join(this._installedDir(file), file.filename);
    if (!fs.existsSync(filePath)) return false;
    const cached = this._cachedSha(filePath);
    if (cached) return cached !== file.sha256;
    if (!this._hashing.has(filePath)) {
      this._hashing.add(filePath);
      this._hashQueue.set(filePath, { file, size: file.sizeBytes });
      if (!this._hashWorkerRunning) {
        this._hashWorkerRunning = true;
        // Collect this registry read's entries before choosing the smallest.
        setImmediate(() => { void this._drainHashQueue(); });
      }
    }
    return false;
  }

  /** One background read at a time, smallest first. Completed sidecars survive
   *  restarts; _matchesRegistry checks them again when a queued file starts.
   *  Let generation, training and other registered work finish between files. */
  private async _drainHashQueue(): Promise<void> {
    try {
      while (this._hashQueue.size) {
        if (getBusyBlockers().length) {
          await new Promise<void>(resolve => { setTimeout(resolve, 2000).unref(); });
          continue;
        }
        const [filePath, { file }] = [...this._hashQueue.entries()]
          .sort((a, b) => a[1].size - b[1].size)[0];
        this._hashQueue.delete(filePath);
        try { await this._matchesRegistry(filePath, file); }
        catch { /* A missing or unreadable file can be retried on a later read. */ }
        finally {
          this._hashing.delete(filePath);
          this.emit('progress');
        }
      }
    } finally {
      this._hashWorkerRunning = false;
    }
  }

  /** describeDownloadError, plus what to do about it for a gated MuScriptor repo. */
  private _describeFailure(err: any, file: RegistryFile): string {
    const why = describeDownloadError(err);
    return file.muscriptorModel && looksLikeGatedError(why)
      ? `${why}. Request access on the model's Hugging Face page, then save a read token (MIDI Studio, or the token field on the StableStep tab).`
      : why;
  }

  private _failJob(job: InternalJob, file: RegistryFile, err: any): void {
    job.status = 'failed';
    job.error = this._describeFailure(err, file);
    job.speed = 0;
    this.emit('progress');
    console.error(`[ModelManager] ${file.filename} failed — ${job.error}`);
  }

  /** sha256 of a file, streamed so a multi-GB model never lands in memory. */
  private _hashFile(filePath: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const h = createHash('sha256');
      const rs = fs.createReadStream(filePath);
      rs.on('error', reject);
      rs.on('data', chunk => h.update(chunk));
      rs.on('end', () => resolve(h.digest('hex')));
    });
  }

  /** The <file>.sha256 sidecar's hash, but only if it was computed from the
   *  file as it is now. Returns null when there is no sidecar, it is
   *  unreadable, or the file has changed since it was written. Never hashes. */
  private _cachedSha(filePath: string): string | null {
    try {
      const stat = fs.statSync(filePath);
      const [sha, stamp] = fs.readFileSync(`${filePath}.sha256`, 'utf8').trim().split(/\s+/);
      return stamp === `${stat.size}:${Math.floor(stat.mtimeMs)}` ? sha : null;
    } catch {
      return null;
    }
  }

  /** True when the file on disk is the one the registry entry describes.
   *
   *  Entries without a sha256 keep the old behaviour: present means current.
   *  For the rest this is the only thing that can catch a republished model
   *  sitting under its old filename -- the YuE2 tokenizer head went v4 -> v9
   *  at the same name and 832 bytes apart, well inside the 5% tolerance
   *  _validateDownload allows, so nothing else would ever have noticed.
   *
   *  Hashing gigabytes on every check would be absurd, so a verified hash is
   *  remembered in a <file>.sha256 sidecar stamped with the size and mtime it
   *  came from; only a file that has changed, or has never been verified,
   *  is actually read. */
  private async _matchesRegistry(filePath: string, file: RegistryFile): Promise<boolean> {
    if (!file.sha256) return true;
    const cached = this._cachedSha(filePath);
    if (cached) return cached === file.sha256;
    const sha = await this._hashFile(filePath);
    this._writeSidecar(filePath, sha);
    return sha === file.sha256;
  }

  private _writeSidecar(filePath: string, sha: string): void {
    try {
      const stat = fs.statSync(filePath);
      fs.writeFileSync(`${filePath}.sha256`, `${sha} ${stat.size}:${Math.floor(stat.mtimeMs)}\n`);
    } catch { /* the cache is an optimisation, not a requirement */ }
  }

  /** Validate a downloaded file is a real binary (not an HTML error page) */
  private _validateDownload(filePath: string, file: RegistryFile): void {
    const stat = fs.statSync(filePath);

    // Size check: must be within 5% of expected size
    if (file.sizeBytes > 0) {
      const tolerance = file.sizeBytes * 0.05;
      if (Math.abs(stat.size - file.sizeBytes) > tolerance) {
        try { fs.unlinkSync(filePath); } catch {}
        throw new Error(
          `Size mismatch: expected ${(file.sizeBytes / 1024 / 1024).toFixed(1)} MB, ` +
          `got ${(stat.size / 1024 / 1024).toFixed(1)} MB — file may be corrupt or an error page`
        );
      }
    }

    // PE header check for DLLs
    if (file.filename.endsWith('.dll')) {
      const fd = fs.openSync(filePath, 'r');
      const header = Buffer.alloc(2);
      fs.readSync(fd, header, 0, 2, 0);
      fs.closeSync(fd);
      if (header.toString('ascii') !== 'MZ') {
        try { fs.unlinkSync(filePath); } catch {}
        throw new Error(
          `Invalid PE header — downloaded file is not a valid DLL ` +
          `(got "${header.toString('ascii')}" instead of "MZ")`
        );
      }
    }

    // GGUF magic check for model files
    if (file.filename.endsWith('.gguf')) {
      const fd = fs.openSync(filePath, 'r');
      const header = Buffer.alloc(4);
      fs.readSync(fd, header, 0, 4, 0);
      fs.closeSync(fd);
      if (header.toString('ascii') !== 'GGUF') {
        try { fs.unlinkSync(filePath); } catch {}
        throw new Error(
          `Invalid GGUF header — downloaded file is not a valid model ` +
          `(got "${header.toString('ascii')}" instead of "GGUF")`
        );
      }
    }
  }

  /** Fetch each of a file's companion files into `targetDir`, skipping ones
   *  that already exist. No progress tracking, no resume — these are small
   *  (license/readme-sized) side files, not models. Generic: any registry
   *  entry can declare `companions`, not just MM3. */
  private async _fetchCompanions(file: RegistryFile, targetDir: string): Promise<void> {
    if (!file.companions?.length) return;
    for (const companion of file.companions) {
      const dest = path.join(targetDir, companion.filename);
      if (fs.existsSync(dest)) continue;
      const repoPath = companion.repoPath || companion.filename;
      const url = `https://huggingface.co/${file.repo}/resolve/main/${repoPath}`;
      try {
        await this._downloadSimple(url, dest);
        console.log(`[ModelManager] Companion file fetched: ${companion.filename}`);
      } catch (err: any) {
        console.warn(`[ModelManager] Companion fetch failed for ${companion.filename}: ${describeDownloadError(err)}`);
      }
    }
  }

  /** Fetch the companions marked `required`. Unlike _fetchCompanions this
   *  sends the HF token, validates what arrived, installs it atomically and
   *  throws on any failure, so the job fails instead of reporting a model
   *  that cannot load. A copy already on disk is kept if it still validates.
   *  Fetched files stay as validated `.part` files in `staged`; the caller
   *  publishes them (_publishStaged) or drops them (_discardStaged). */
  private async _fetchRequired(file: RegistryFile, targetDir: string, job: InternalJob, staged: Staged[]): Promise<void> {
    // cancelDownload aborts job.abortController; every request below carries
    // its signal, and nothing is published once it has fired.
    const signal = (job.abortController ??= new AbortController()).signal;
    const cancelled = () => signal.aborted || (job.status as DownloadStatus) === 'cancelled';
    for (const c of file.companions ?? []) {
      if (!c.required) continue;
      if (cancelled()) return;
      const dest = path.join(targetDir, c.filename);
      if (this._validRequired(dest)) continue;
      const tmp = `${dest}.part`;
      try {
        await this._downloadSimple(`https://huggingface.co/${file.repo}/resolve/main/${c.repoPath || c.filename}`, tmp, job.hfToken, signal);
        if (cancelled()) throw new Error('Download cancelled');
        if (!this._validRequired(tmp)) throw new Error(`${c.filename} from ${file.repo} is not valid`);
        staged.push({ tmp, dest });
      } catch (err) {
        try { fs.unlinkSync(tmp); } catch {}
        throw err;
      }
    }
  }

  private _publishStaged(staged: Staged[]): void {
    while (staged.length) {
      fs.renameSync(staged[0].tmp, staged[0].dest);
      staged.shift(); // an entry leaves the list only once published, so a throw leaves it for _discardStaged
    }
  }

  private _discardStaged(staged: Staged[]): void {
    for (const { tmp } of staged.splice(0)) try { fs.unlinkSync(tmp); } catch {}
  }

  /** A required companion is usable: present, non-empty, and for JSON, an object. */
  private _validRequired(p: string): boolean {
    try {
      const text = fs.readFileSync(p, 'utf8');
      if (!text.trim()) return false;
      if (!p.replace(/\.part$/, '').endsWith('.json')) return true;
      const v = JSON.parse(text);
      return !!v && typeof v === 'object' && !Array.isArray(v);
    } catch {
      return false;
    }
  }

  /** Minimal one-shot download (redirects, no Range/resume, no progress
   *  events) for small companion files. The token, if any, follows
   *  hfAuthHeader's rule on every hop. */
  private _downloadSimple(url: string, destPath: string, token?: string, signal?: AbortSignal, redirectCount = 0): Promise<void> {
    if (redirectCount > 5) return Promise.reject(new Error('Too many redirects'));
    return new Promise((resolve, reject) => {
      const parsedUrl = new URL(url);
      const transport = parsedUrl.protocol === 'https:' ? https : http;
      const req = transport.get(parsedUrl, { headers: { 'User-Agent': 'HOT-Step-CPP/1.0', ...hfAuthHeader(parsedUrl, token) }, signal }, (res) => {
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          const redirectUrl = new URL(res.headers.location, parsedUrl).href;
          this._downloadSimple(redirectUrl, destPath, token, signal, redirectCount + 1).then(resolve).catch(reject);
          return;
        }
        if (res.statusCode && res.statusCode >= 400) {
          res.resume();
          reject(Object.assign(new Error(`HTTP ${res.statusCode}: ${res.statusMessage}`), { url }));
          return;
        }
        const writeStream = fs.createWriteStream(destPath);
        res.pipe(writeStream);
        writeStream.on('finish', () => writeStream.close(() => resolve()));
        writeStream.on('error', reject);
        // An aborted response never ends the pipe: close the file and settle.
        res.on('error', err => { writeStream.destroy(); reject(err); });
        res.on('close', () => { if (!res.complete) { writeStream.destroy(); reject(new Error('Download interrupted')); } });
      });
      req.on('error', reject);
    });
  }

  private _downloadWithRedirects(url: string, partPath: string, job: InternalJob, startByte: number, redirectCount = 0): Promise<void> {
    if (redirectCount > 5) return Promise.reject(new Error('Too many redirects'));

    return new Promise((resolve, reject) => {
      const parsedUrl = new URL(url);
      const transport = parsedUrl.protocol === 'https:' ? https : http;

      const headers: Record<string, string> = {
        'User-Agent': 'HOT-Step-CPP/1.0',
      };
      if (startByte > 0) {
        headers['Range'] = `bytes=${startByte}-`;
      }
      // Hugging Face token for gated repos — only sent to huggingface.co
      // itself. CDN redirect targets (cdn-lfs / xethub) use pre-signed URLs
      // and reject requests carrying an extra Authorization header.
      Object.assign(headers, hfAuthHeader(parsedUrl, job.hfToken));

      // Which host the failure was actually against. After a redirect this is
      // the CDN, not huggingface.co — see describeDownloadError.
      const fail = (err: any) => { if (err && typeof err === 'object' && !err.url) err.url = url; return err; };

      const req = transport.get(parsedUrl, { headers }, (res) => {
        // Handle redirects
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume(); // Drain response
          // Resolve relative redirects (e.g. /api/resolve-cache/...) against the original URL
          const redirectUrl = new URL(res.headers.location, parsedUrl).href;
          this._downloadWithRedirects(redirectUrl, partPath, job, startByte, redirectCount + 1)
            .then(resolve).catch(reject);
          return;
        }

        if (res.statusCode === 416) {
          // Range not satisfiable — file might be complete
          res.resume();
          resolve();
          return;
        }

        if (res.statusCode && res.statusCode >= 400) {
          res.resume();
          reject(fail(new Error(`HTTP ${res.statusCode}: ${res.statusMessage}`)));
          return;
        }

        // Parse total size from Content-Range or Content-Length
        const contentRange = res.headers['content-range'];
        if (contentRange) {
          const match = contentRange.match(/bytes \d+-\d+\/(\d+)/);
          if (match) job.totalBytes = parseInt(match[1], 10);
        } else if (res.headers['content-length'] && startByte === 0) {
          job.totalBytes = parseInt(res.headers['content-length'], 10);
        }

        const writeStream = fs.createWriteStream(partPath, {
          flags: startByte > 0 ? 'a' : 'w',
        });

        // A write failure (no space, no permission on the models dir, a
        // read-only volume) was an unhandled 'error' event: pipe() does not
        // forward it, so it reached the process as an uncaught exception
        // instead of failing this one download.
        writeStream.on('error', (err) => {
          res.destroy();
          reject(fail(err));
        });

        res.on('data', (chunk: Buffer) => {
          job.bytesDownloaded += chunk.length;

          // Speed tracking
          const now = Date.now();
          job.speedSamples.push({ time: now, bytes: chunk.length });
          // Keep only last 3 seconds of samples
          const cutoff = now - 3000;
          job.speedSamples = job.speedSamples.filter(s => s.time > cutoff);
          // Calculate speed
          if (job.speedSamples.length > 1) {
            const totalSampleBytes = job.speedSamples.reduce((a, s) => a + s.bytes, 0);
            const elapsed = (now - job.speedSamples[0].time) / 1000;
            job.speed = elapsed > 0 ? totalSampleBytes / elapsed : 0;
          }

          this.emit('progress');
        });

        res.on('end', () => {
          writeStream.end(() => resolve());
        });

        res.on('error', (err) => {
          writeStream.end();
          if (job.status !== 'cancelled') {
            job.status = 'paused';
            job.error = describeDownloadError(fail(err));
          }
          reject(fail(err));
        });

        res.pipe(writeStream, { end: false });

        // Handle abort
        if (job.abortController) {
          job.abortController.signal.addEventListener('abort', () => {
            res.destroy();
            writeStream.end();
          });
        }
      });

      req.on('error', (err) => {
        if (job.status !== 'cancelled') {
          job.status = 'paused';
          job.error = describeDownloadError(fail(err));
        }
        reject(fail(err));
      });
    });
  }
}

export const modelDownloadService = new ModelDownloadService();

// A data-folder move must not start under a download writing into models/.
registerBusyCheck('model-downloads', () => modelDownloadService.getJobs()
  .filter(j => j.status === 'downloading' || j.status === 'queued')
  .map(j => `downloading ${j.filename}`));
