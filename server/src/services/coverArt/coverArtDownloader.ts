// coverArtDownloader.ts — First-use download manager for cover art assets
//
// Downloads sd-cli.exe + FLUX.2-klein-4B GGUF + VAE + Qwen3 LLM from
// HuggingFace on first use. Supports progress tracking and cancellation.
//
// Reference: server/src/services/modelDownloadService.ts

import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import https from 'https';
import http from 'http';
import { getCoverArtDir, REQUIRED_FILES } from './coverArtService.js';

// ── Download manifest ───────────────────────────────────────────────────
// Each entry defines a file to download, its source URL, and expected size.

export interface ManifestEntry {
  filename: string;
  url: string;
  sizeBytes: number;
  description: string;
}

/**
 * Download manifest for FLUX.2-klein-4B cover art pipeline.
 *
 * File sources:
 * - Diffusion model: leejet/FLUX.2-klein-4B-GGUF (Q4_0)
 * - VAE: black-forest-labs/FLUX.2-dev (flux2_ae.safetensors)
 * - LLM: unsloth/Qwen3-4B-GGUF (Q4_K_M)
 * - sd-cli: leejet/stable-diffusion.cpp GitHub releases
 */
const MANIFEST: ManifestEntry[] = [
  {
    filename: REQUIRED_FILES.diffusionModel,
    url: 'https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_0.gguf',
    sizeBytes: 2_460_378_560,
    description: 'FLUX.2-klein-4B diffusion model (Q4)',
  },
  {
    filename: REQUIRED_FILES.vae,
    url: 'https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/main/ae.safetensors',
    sizeBytes: 335_304_388, // ~320 MB
    description: 'FLUX.2 VAE decoder',
  },
  {
    filename: REQUIRED_FILES.llm,
    url: 'https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf',
    sizeBytes: 2_800_000_000, // ~2.6 GB (approximate)
    description: 'Qwen3-4B text encoder (Q4_K_M)',
  },
  // Note: sd-cli.exe must be downloaded separately from GitHub releases
  // or built from source. The URL depends on the platform and CUDA version.
  // For now, the user must place sd-cli.exe manually or we'll add
  // auto-download from GitHub releases in a follow-up.
];

// ── Types ───────────────────────────────────────────────────────────────

export type DownloadPhase = 'idle' | 'downloading' | 'completed' | 'failed' | 'cancelled';

export interface FileProgress {
  filename: string;
  description: string;
  status: DownloadPhase;
  bytesDownloaded: number;
  totalBytes: number;
  speed: number; // bytes/sec
  error?: string;
}

export interface OverallStatus {
  phase: DownloadPhase;
  installed: boolean;
  files: FileProgress[];
  totalBytes: number;
  downloadedBytes: number;
  overallProgress: number; // 0-100
  sdCliMissing: boolean; // true if sd-cli.exe needs manual placement
}

// ── Download Service ────────────────────────────────────────────────────

class CoverArtDownloader extends EventEmitter {
  private fileProgress: Map<string, FileProgress> = new Map();
  private abortControllers: Map<string, AbortController> = new Map();
  private speedSamples: Map<string, { time: number; bytes: number }[]> = new Map();
  private _downloading = false;

  /** Get overall download/installation status */
  getStatus(): OverallStatus {
    const dir = getCoverArtDir();
    const files: FileProgress[] = [];
    let totalBytes = 0;
    let downloadedBytes = 0;

    for (const entry of MANIFEST) {
      const filePath = path.join(dir, entry.filename);
      const existing = this.fileProgress.get(entry.filename);

      if (existing) {
        files.push({ ...existing });
        totalBytes += existing.totalBytes;
        downloadedBytes += existing.bytesDownloaded;
      } else if (fs.existsSync(filePath)) {
        const size = fs.statSync(filePath).size;
        files.push({
          filename: entry.filename,
          description: entry.description,
          status: 'completed',
          bytesDownloaded: size,
          totalBytes: size,
          speed: 0,
        });
        totalBytes += size;
        downloadedBytes += size;
      } else {
        files.push({
          filename: entry.filename,
          description: entry.description,
          status: 'idle',
          bytesDownloaded: 0,
          totalBytes: entry.sizeBytes,
          speed: 0,
        });
        totalBytes += entry.sizeBytes;
      }
    }

    // Check sd-cli.exe separately (not in manifest — requires manual placement or GitHub release)
    const sdCliPath = path.join(dir, REQUIRED_FILES.sdCli);
    const sdCliExists = fs.existsSync(sdCliPath);

    const allFilesComplete = files.every(f => f.status === 'completed');
    const anyFailed = files.some(f => f.status === 'failed');
    const anyDownloading = files.some(f => f.status === 'downloading');

    let phase: DownloadPhase = 'idle';
    if (anyDownloading) phase = 'downloading';
    else if (anyFailed) phase = 'failed';
    else if (allFilesComplete && sdCliExists) phase = 'completed';

    return {
      phase,
      installed: allFilesComplete && sdCliExists,
      files,
      totalBytes,
      downloadedBytes,
      overallProgress: totalBytes > 0 ? Math.round((downloadedBytes / totalBytes) * 100) : 0,
      sdCliMissing: !sdCliExists,
    };
  }

  /** Start downloading all missing model files */
  async startDownload(): Promise<void> {
    if (this._downloading) return;
    this._downloading = true;

    const dir = getCoverArtDir();
    fs.mkdirSync(dir, { recursive: true });

    try {
      // Download files sequentially to avoid bandwidth contention
      for (const entry of MANIFEST) {
        const filePath = path.join(dir, entry.filename);
        if (fs.existsSync(filePath)) {
          console.log(`[CoverArt Download] ${entry.filename}: already exists, skipping`);
          continue;
        }

        await this._downloadFile(entry, dir);

        // Check if cancelled
        if (this.fileProgress.get(entry.filename)?.status === 'cancelled') {
          break;
        }
      }
    } finally {
      this._downloading = false;
    }
  }

  /** Cancel all active downloads */
  cancelDownload(): void {
    for (const [filename, controller] of this.abortControllers) {
      controller.abort();
      const progress = this.fileProgress.get(filename);
      if (progress && progress.status === 'downloading') {
        progress.status = 'cancelled';
      }
    }
    this.abortControllers.clear();
    this._downloading = false;
    this.emit('progress');
  }

  // ── Internal ─────────────────────────────────────────────────────────

  private async _downloadFile(entry: ManifestEntry, dir: string): Promise<void> {
    const partPath = path.join(dir, `${entry.filename}.part`);
    const finalPath = path.join(dir, entry.filename);

    // Check for partial download to resume
    let startByte = 0;
    if (fs.existsSync(partPath)) {
      startByte = fs.statSync(partPath).size;
    }

    const progress: FileProgress = {
      filename: entry.filename,
      description: entry.description,
      status: 'downloading',
      bytesDownloaded: startByte,
      totalBytes: entry.sizeBytes,
      speed: 0,
    };
    this.fileProgress.set(entry.filename, progress);
    this.speedSamples.set(entry.filename, []);

    const abortController = new AbortController();
    this.abortControllers.set(entry.filename, abortController);

    console.log(`[CoverArt Download] Starting: ${entry.filename} (${(entry.sizeBytes / 1024 / 1024 / 1024).toFixed(1)} GB)`);
    if (startByte > 0) {
      console.log(`[CoverArt Download] Resuming from ${(startByte / 1024 / 1024).toFixed(0)} MB`);
    }

    this.emit('progress');

    try {
      await this._httpDownload(entry.url, partPath, progress, startByte);

      if (progress.status === 'cancelled') return;

      // Rename .part → final
      fs.renameSync(partPath, finalPath);
      progress.status = 'completed';
      progress.speed = 0;
      console.log(`[CoverArt Download] Complete: ${entry.filename}`);
    } catch (err: any) {
      if (progress.status === 'cancelled') return;
      progress.status = 'failed';
      progress.error = err.message;
      progress.speed = 0;
      console.error(`[CoverArt Download] Failed: ${entry.filename} — ${err.message}`);
    } finally {
      this.abortControllers.delete(entry.filename);
      this.emit('progress');
    }
  }

  private _httpDownload(url: string, partPath: string, progress: FileProgress, startByte: number, redirectCount = 0): Promise<void> {
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
          res.resume();
          this._httpDownload(res.headers.location, partPath, progress, startByte, redirectCount + 1)
            .then(resolve).catch(reject);
          return;
        }

        if (res.statusCode === 416) {
          // Range not satisfiable — file is complete
          res.resume();
          resolve();
          return;
        }

        if (res.statusCode && res.statusCode >= 400) {
          res.resume();
          reject(new Error(`HTTP ${res.statusCode}: ${res.statusMessage}`));
          return;
        }

        // Parse total size
        const contentRange = res.headers['content-range'];
        if (contentRange) {
          const match = contentRange.match(/bytes \d+-\d+\/(\d+)/);
          if (match) progress.totalBytes = parseInt(match[1], 10);
        } else if (res.headers['content-length'] && startByte === 0) {
          progress.totalBytes = parseInt(res.headers['content-length'], 10);
        }

        const writeStream = fs.createWriteStream(partPath, {
          flags: startByte > 0 ? 'a' : 'w',
        });

        const samples = this.speedSamples.get(progress.filename) || [];

        res.on('data', (chunk: Buffer) => {
          progress.bytesDownloaded += chunk.length;

          // Speed tracking (3-second rolling window)
          const now = Date.now();
          samples.push({ time: now, bytes: chunk.length });
          const cutoff = now - 3000;
          while (samples.length > 0 && samples[0].time < cutoff) samples.shift();

          if (samples.length > 1) {
            const totalSampleBytes = samples.reduce((a, s) => a + s.bytes, 0);
            const elapsed = (now - samples[0].time) / 1000;
            progress.speed = elapsed > 0 ? totalSampleBytes / elapsed : 0;
          }

          this.emit('progress');
        });

        res.on('end', () => {
          writeStream.end(() => resolve());
        });

        res.on('error', (err) => {
          writeStream.end();
          reject(err);
        });

        res.pipe(writeStream, { end: false });

        // Handle abort
        const abortCtrl = this.abortControllers.get(progress.filename);
        if (abortCtrl) {
          abortCtrl.signal.addEventListener('abort', () => {
            res.destroy();
            writeStream.end();
          });
        }
      });

      req.on('error', reject);
    });
  }
}

export const coverArtDownloader = new CoverArtDownloader();
