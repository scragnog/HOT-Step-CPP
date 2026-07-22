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
import { randomUUID } from 'crypto';
import { fileURLToPath } from 'url';
import { config, PORTABLE_MODE, PROJECT_ROOT } from '../config.js';

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
const CUDA_ONLY_FILE_PREFIXES = ['cuda-rt-', 'supersep-rt-'];
// IDs of CUDA-only packs (hidden entirely for non-CUDA)
const CUDA_ONLY_PACKS = ['cuda-runtime', 'cuda12-runtime', 'supersep-runtime', 'blackwell'];

// ── Types ───────────────────────────────────────────────────

export type DownloadStatus = 'queued' | 'downloading' | 'paused' | 'completed' | 'failed' | 'cancelled';

export interface DownloadJob {
  jobId: string;
  fileId: string;
  filename: string;
  status: DownloadStatus;
  bytesDownloaded: number;
  totalBytes: number;
  speed: number;        // bytes/sec rolling average
  error?: string;
}

interface InternalJob extends DownloadJob {
  abortController?: AbortController;
  speedSamples: { time: number; bytes: number }[];
  hfToken?: string;   // Optional Hugging Face token for gated repos
}

interface RegistryFile {
  id: string;
  filename: string;
  role: string;
  subdir?: string;
  repoPath?: string;     // Path within the HuggingFace repo (e.g. "runtime/cublas64_13.dll")
  displayName: string;
  scale: string | null;
  variant: string | null;
  quant: string;
  sizeBytes: number;
  repo: string;
  description: string;
  tags: string[];
}

// ── Service ─────────────────────────────────────────────────

class ModelDownloadService extends EventEmitter {
  private jobs = new Map<string, InternalJob>();

  /** Get the models directory path */
  get modelsDir(): string {
    return config.aceServer.models;
  }

  /** Get the engine directory (where ace-server.exe lives) for runtime DLLs */
  get engineDir(): string {
    return path.dirname(config.aceServer.exe);
  }

  /** Resolve the target directory for a registry file */
  private getTargetDir(file: RegistryFile): string {
    if (file.role === 'runtime') {
      // Runtime DLLs go alongside ace-server.exe
      return this.engineDir;
    }
    return file.subdir
      ? path.join(this.modelsDir, file.subdir)
      : this.modelsDir;
  }

  /** Get all files in the registry, enriched with installed status.
   *  Filters out CUDA-specific entries for non-CUDA engine variants. */
  getRegistry(): { packs: any[]; files: (RegistryFile & { installed: boolean })[]; modelsDir: string; variant: string; cudaMajor: number } {
    const installed = this.getInstalledFiles();
    const isCuda = ENGINE_VARIANT === 'cuda';
    const wrongCudaTag = CUDA_MAJOR <= 12 ? 'cuda13' : 'cuda12';

    // Filter files: hide CUDA-only entries for non-CUDA builds,
    // and hide wrong-CUDA-version entries
    const filteredFiles = registry.files
      .filter((f: RegistryFile) => isCuda || !CUDA_ONLY_FILE_PREFIXES.some(p => f.id.startsWith(p)))
      .filter((f: RegistryFile) => !f.tags?.includes(wrongCudaTag))
      .map((f: RegistryFile) => ({
        ...f,
        installed: installed.has(f.filename),
      }));

    // Filter packs: hide CUDA-only packs, strip CUDA file IDs from remaining
    // Also hide the wrong-version CUDA runtime pack and remap IDs in model packs
    const filteredPacks = registry.packs
      .filter((p: any) => isCuda || !CUDA_ONLY_PACKS.includes(p.id))
      .filter((p: any) => {
        // Show only the correct CUDA runtime pack
        if (p.id === 'cuda-runtime') return CUDA_MAJOR >= 13;
        if (p.id === 'cuda12-runtime') return CUDA_MAJOR <= 12;
        return true;
      })
      .map((p: any) => {
        if (!isCuda) {
          return { ...p, fileIds: p.fileIds.filter((id: string) => !CUDA_ONLY_FILE_PREFIXES.some(pfx => id.startsWith(pfx))) };
        }
        // Remap cuda-rt-* IDs in packs to version-specific ones for CUDA 12
        if (CUDA_MAJOR <= 12) {
          return {
            ...p,
            fileIds: p.fileIds.map((id: string) => {
              if (['cuda-rt-cublas', 'cuda-rt-cublaslt', 'cuda-rt-cudart'].includes(id)) {
                return `${id}-12`;
              }
              return id;
            }),
          };
        }
        return p;
      });

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
    // (<modelsDir>/onnx/sa3) and contains non-model extensions
    // (.onnx.data, .json) that the generic scan above ignores.
    const sa3Dir = path.join(dir, 'onnx', 'sa3');
    if (fs.existsSync(sa3Dir)) {
      for (const f of fs.readdirSync(sa3Dir)) {
        if (!f.endsWith('.part')) files.add(f);
      }
    }

    // Scan engine directory for runtime DLLs
    const engDir = this.engineDir;
    if (fs.existsSync(engDir)) {
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
      hfToken: hfToken?.trim() || undefined,
    };

    this.jobs.set(jobId, job);
    this._executeDownload(job, file);
    return jobId;
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
    // Safety: only known model/runtime extensions.
    // .data / .json are StableStep (SA3) companions (sa3-dit.onnx.data,
    // tokenizer.json etc.) living under onnx/sa3.
    if (!filename.endsWith('.gguf') && !filename.endsWith('.onnx') && !filename.endsWith('.safetensors') && !filename.endsWith('.dll') && !filename.endsWith('.bin') && !filename.endsWith('.data') && !filename.endsWith('.json')) {
      throw new Error('Can only delete .gguf, .onnx, .safetensors, .bin, .dll, .data, or .json files');
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
    // Determine target directory based on file role
    const targetDir = this.getTargetDir(file);
    if (!fs.existsSync(targetDir)) {
      fs.mkdirSync(targetDir, { recursive: true });
    }

    const partPath = path.join(targetDir, `${file.filename}.part`);
    const finalPath = path.join(targetDir, file.filename);

    // Check if already fully downloaded
    if (fs.existsSync(finalPath)) {
      job.status = 'completed';
      job.bytesDownloaded = job.totalBytes;
      this.emit('progress');
      return;
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
      const url = `https://huggingface.co/${file.repo}/resolve/main/${repoPath}`;

      try {
        await this._downloadWithRedirects(url, partPath, job, startByte);

        if ((job.status as DownloadStatus) === 'cancelled') return;

        // Validate downloaded file before finalising
        this._validateDownload(partPath, file);

        // Rename .part to final
        fs.renameSync(partPath, finalPath);
        job.status = 'completed';
        job.speed = 0;
        this.emit('progress');
        console.log(`[ModelManager] Download complete: ${file.filename}`);
        return; // Success — exit retry loop
      } catch (err: any) {
        if ((job.status as DownloadStatus) === 'cancelled') return;

        const isLastAttempt = attempt === ModelDownloadService.RETRY_DELAYS.length - 1;
        if (isLastAttempt) {
          job.status = 'failed';
          job.error = err.message;
          job.speed = 0;
          this.emit('progress');
          console.error(`[ModelManager] Download failed after ${attempt + 1} attempts: ${file.filename} — ${err.message}`);
        } else {
          console.warn(`[ModelManager] Download attempt ${attempt + 1} failed for ${file.filename}: ${err.message}`);
          job.speedSamples = [];
        }
      }
    }
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
      if (job.hfToken && /(^|\.)huggingface\.co$/.test(parsedUrl.hostname)) {
        headers['Authorization'] = `Bearer ${job.hfToken}`;
      }

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
          reject(new Error(`HTTP ${res.statusCode}: ${res.statusMessage}`));
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
            job.error = err.message;
          }
          reject(err);
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
          job.error = err.message;
        }
        reject(err);
      });
    });
  }
}

export const modelDownloadService = new ModelDownloadService();
