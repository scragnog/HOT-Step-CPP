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
import { config } from '../config.js';
import { createRequire } from 'module';

// Load registry
const require = createRequire(import.meta.url);
const registry = require('../data/model-registry.json');

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
}

interface RegistryFile {
  id: string;
  filename: string;
  role: string;
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

  /** Get all files in the registry, enriched with installed status */
  getRegistry(): { packs: typeof registry.packs; files: (RegistryFile & { installed: boolean })[]; modelsDir: string } {
    const installed = this.getInstalledFiles();
    return {
      packs: registry.packs,
      files: registry.files.map((f: RegistryFile) => ({
        ...f,
        installed: installed.has(f.filename),
      })),
      modelsDir: this.modelsDir,
    };
  }

  /** Scan models directory for installed .gguf files */
  getInstalledFiles(): Set<string> {
    const dir = this.modelsDir;
    if (!fs.existsSync(dir)) return new Set();
    return new Set(
      fs.readdirSync(dir).filter(f => f.endsWith('.gguf'))
    );
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

  /** Start downloading a file by registry ID */
  startDownload(fileId: string): string {
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

    // Clean up .part file
    const partPath = path.join(this.modelsDir, `${job.filename}.part`);
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
    const partPath = path.join(this.modelsDir, `${job.filename}.part`);
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

  /** Delete a model file from disk */
  deleteFile(filename: string): boolean {
    // Safety: only .gguf files
    if (!filename.endsWith('.gguf')) throw new Error('Can only delete .gguf files');

    const filePath = path.join(this.modelsDir, filename);
    if (!fs.existsSync(filePath)) return false;

    // Ensure path is within models dir (prevent traversal)
    const resolved = path.resolve(filePath);
    const modelsResolved = path.resolve(this.modelsDir);
    if (!resolved.startsWith(modelsResolved)) throw new Error('Path traversal denied');

    fs.unlinkSync(filePath);
    return true;
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

  private async _executeDownload(job: InternalJob, file: RegistryFile): Promise<void> {
    const modelsDir = this.modelsDir;
    if (!fs.existsSync(modelsDir)) {
      fs.mkdirSync(modelsDir, { recursive: true });
    }

    const partPath = path.join(modelsDir, `${file.filename}.part`);
    const finalPath = path.join(modelsDir, file.filename);

    // Check if already fully downloaded
    if (fs.existsSync(finalPath)) {
      job.status = 'completed';
      job.bytesDownloaded = job.totalBytes;
      this.emit('progress');
      return;
    }

    // Check existing .part file for resume
    let startByte = 0;
    if (fs.existsSync(partPath)) {
      startByte = fs.statSync(partPath).size;
      job.bytesDownloaded = startByte;
    }

    job.status = 'downloading';
    job.abortController = new AbortController();
    this.emit('progress');

    const url = `https://huggingface.co/${file.repo}/resolve/main/${file.filename}`;

    try {
      await this._downloadWithRedirects(url, partPath, job, startByte);

      if ((job.status as DownloadStatus) === 'cancelled') return;

      // Rename .part to final
      fs.renameSync(partPath, finalPath);
      job.status = 'completed';
      job.speed = 0;
      this.emit('progress');
      console.log(`[ModelManager] Download complete: ${file.filename}`);
    } catch (err: any) {
      if ((job.status as DownloadStatus) === 'cancelled') return;
      job.status = 'failed';
      job.error = err.message;
      job.speed = 0;
      this.emit('progress');
      console.error(`[ModelManager] Download failed: ${file.filename} — ${err.message}`);
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

      const req = transport.get(parsedUrl, { headers }, (res) => {
        // Handle redirects
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume(); // Drain response
          this._downloadWithRedirects(res.headers.location, partPath, job, startByte, redirectCount + 1)
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
