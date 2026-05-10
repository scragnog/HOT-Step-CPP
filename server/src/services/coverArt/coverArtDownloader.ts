// coverArtDownloader.ts — First-use download manager for cover art assets
//
// Downloads sd.exe (from GitHub releases) + FLUX.2-klein-4B GGUF +
// VAE + Qwen3 LLM from HuggingFace on first use.
// Supports progress tracking, resume, and cancellation.
//
// sd.exe download flow:
//   1. Query GitHub API for latest stable-diffusion.cpp release
//   2. Pick the right asset (CUDA on Windows, Metal on macOS)
//   3. Download the ZIP, extract sd.exe + DLLs to cover-art directory
//
// Reference: server/src/services/modelDownloadService.ts

import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import https from 'https';
import http from 'http';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { getCoverArtDir, REQUIRED_FILES } from './coverArtService.js';

const execFileAsync = promisify(execFile);

// ── Download manifest ───────────────────────────────────────────────────

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
 * - sd.exe: leejet/stable-diffusion.cpp GitHub releases (auto-detected)
 */
const MODEL_MANIFEST: ManifestEntry[] = [
  {
    filename: REQUIRED_FILES.diffusionModel,
    url: 'https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_0.gguf',
    sizeBytes: 2_460_378_560,
    description: 'FLUX.2-klein-4B diffusion model (Q4)',
  },
  {
    filename: REQUIRED_FILES.vae,
    url: 'https://huggingface.co/black-forest-labs/FLUX.2-dev/resolve/main/ae.safetensors',
    sizeBytes: 335_304_388,
    description: 'FLUX.2 VAE decoder',
  },
  {
    filename: REQUIRED_FILES.llm,
    url: 'https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf',
    sizeBytes: 2_800_000_000,
    description: 'Qwen3-4B text encoder (Q4_K_M)',
  },
];

// ── GitHub Release Asset Selection ──────────────────────────────────────

/**
 * Platform-specific patterns for selecting the right sd.exe release asset.
 * Ordered by preference — first match wins.
 */
const SD_ASSET_PATTERNS: Record<string, string[]> = {
  win32: [
    '-bin-win-cuda12-x64.zip',   // NVIDIA CUDA (most HOT-Step users)
    '-bin-win-avx2-x64.zip',     // CPU fallback (AVX2)
    '-bin-win-avx-x64.zip',      // CPU fallback (AVX)
  ],
  darwin: [
    '-bin-Darwin-',              // macOS (Metal/ARM)
  ],
  linux: [
    '-bin-Linux-',               // Linux (CPU)
  ],
};

const GITHUB_RELEASES_API = 'https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest';

/** The binary name inside the release ZIP */
const SD_BINARY_NAME = process.platform === 'win32' ? 'sd.exe' : 'sd';

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
    const allEntries = this._getAllEntries();
    const files: FileProgress[] = [];
    let totalBytes = 0;
    let downloadedBytes = 0;

    for (const entry of allEntries) {
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

    const allFilesComplete = files.every(f => f.status === 'completed');
    const anyFailed = files.some(f => f.status === 'failed');
    const anyDownloading = files.some(f => f.status === 'downloading');

    let phase: DownloadPhase = 'idle';
    if (anyDownloading) phase = 'downloading';
    else if (anyFailed) phase = 'failed';
    else if (allFilesComplete) phase = 'completed';

    return {
      phase,
      installed: allFilesComplete,
      files,
      totalBytes,
      downloadedBytes,
      overallProgress: totalBytes > 0 ? Math.round((downloadedBytes / totalBytes) * 100) : 0,
    };
  }

  /** Start downloading all missing files (models + sd.exe) */
  async startDownload(): Promise<void> {
    if (this._downloading) return;
    this._downloading = true;

    const dir = getCoverArtDir();
    fs.mkdirSync(dir, { recursive: true });

    try {
      // 1. Download model files from HuggingFace
      for (const entry of MODEL_MANIFEST) {
        const filePath = path.join(dir, entry.filename);
        if (fs.existsSync(filePath)) {
          console.log(`[CoverArt Download] ${entry.filename}: already exists, skipping`);
          continue;
        }

        await this._downloadFile(entry, dir);

        if (this.fileProgress.get(entry.filename)?.status === 'cancelled') {
          return;
        }
      }

      // 2. Download sd.exe from GitHub releases (if not present)
      const sdPath = path.join(dir, REQUIRED_FILES.sdCli);
      if (!fs.existsSync(sdPath)) {
        await this._downloadSdBinary(dir);
      } else {
        console.log(`[CoverArt Download] ${REQUIRED_FILES.sdCli}: already exists, skipping`);
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

  // ── sd.exe download from GitHub ──────────────────────────────────────

  /**
   * Download sd.exe from the latest GitHub release.
   * 1. Query GitHub API for the latest release
   * 2. Find the right ZIP asset for the current platform
   * 3. Download and extract
   */
  private async _downloadSdBinary(dir: string): Promise<void> {
    const progressKey = REQUIRED_FILES.sdCli;

    // Set progress to show we're working on it
    const progress: FileProgress = {
      filename: progressKey,
      description: `stable-diffusion.cpp (${process.platform === 'win32' ? 'CUDA' : 'native'})`,
      status: 'downloading',
      bytesDownloaded: 0,
      totalBytes: 0,
      speed: 0,
    };
    this.fileProgress.set(progressKey, progress);
    this.emit('progress');

    try {
      // Step 1: Query GitHub API for latest release
      console.log('[CoverArt Download] Querying GitHub for latest stable-diffusion.cpp release...');
      const releaseData = await this._fetchJson(GITHUB_RELEASES_API);

      if (!releaseData || !Array.isArray(releaseData.assets)) {
        throw new Error('Failed to fetch release data from GitHub');
      }

      // Step 2: Find the right asset for this platform
      const patterns = SD_ASSET_PATTERNS[process.platform] || SD_ASSET_PATTERNS.linux;
      let assetUrl: string | null = null;
      let assetName: string = '';
      let assetSize = 0;

      for (const pattern of patterns) {
        const asset = releaseData.assets.find((a: any) => a.name.includes(pattern));
        if (asset) {
          assetUrl = asset.browser_download_url;
          assetName = asset.name;
          assetSize = asset.size || 0;
          break;
        }
      }

      if (!assetUrl) {
        throw new Error(`No compatible sd binary found for platform: ${process.platform}`);
      }

      console.log(`[CoverArt Download] Found: ${assetName} (${(assetSize / 1024 / 1024).toFixed(0)} MB)`);
      progress.totalBytes = assetSize;
      progress.description = `sd.exe engine (${assetName.includes('cuda') ? 'CUDA' : 'CPU'})`;
      this.emit('progress');

      // Step 3: Download the ZIP
      const zipPath = path.join(dir, assetName);
      await this._httpDownload(assetUrl, zipPath, progress, 0);

      if (progress.status === 'cancelled') return;

      // Step 4: Extract the ZIP
      console.log(`[CoverArt Download] Extracting ${assetName}...`);
      progress.description = `Extracting sd.exe...`;
      this.emit('progress');

      await this._extractZip(zipPath, dir);

      // Step 5: Verify sd.exe exists after extraction
      const sdPath = path.join(dir, SD_BINARY_NAME);
      if (!fs.existsSync(sdPath)) {
        // The ZIP might have a subdirectory — search for it
        const found = this._findFile(dir, SD_BINARY_NAME);
        if (found) {
          // Move it to the root of the cover-art directory
          fs.renameSync(found, sdPath);
          console.log(`[CoverArt Download] Moved ${SD_BINARY_NAME} from ${path.dirname(found)} to ${dir}`);

          // Also move any DLLs from that subdirectory
          const subDir = path.dirname(found);
          if (subDir !== dir) {
            const files = fs.readdirSync(subDir);
            for (const f of files) {
              const ext = path.extname(f).toLowerCase();
              if (ext === '.dll' || ext === '.so' || ext === '.dylib') {
                const src = path.join(subDir, f);
                const dst = path.join(dir, f);
                if (!fs.existsSync(dst)) {
                  fs.renameSync(src, dst);
                }
              }
            }
            // Clean up empty subdirectory
            try { fs.rmdirSync(subDir, { recursive: true } as any); } catch {}
          }
        } else {
          throw new Error(`${SD_BINARY_NAME} not found after extraction`);
        }
      }

      // Step 6: Clean up ZIP
      try { fs.unlinkSync(zipPath); } catch {}
      console.log(`[CoverArt Download] sd.exe ready`);

      progress.status = 'completed';
      progress.speed = 0;
    } catch (err: any) {
      if (progress.status === 'cancelled') return;
      progress.status = 'failed';
      progress.error = err.message;
      progress.speed = 0;
      console.error(`[CoverArt Download] sd.exe download failed: ${err.message}`);
    } finally {
      this.emit('progress');
    }
  }

  /** Fetch JSON from a URL (for GitHub API) */
  private _fetchJson(url: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const parsedUrl = new URL(url);
      const transport = parsedUrl.protocol === 'https:' ? https : http;

      transport.get(parsedUrl, {
        headers: {
          'User-Agent': 'HOT-Step-CPP/1.0',
          'Accept': 'application/vnd.github.v3+json',
        },
      }, (res) => {
        // Handle redirects
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          this._fetchJson(res.headers.location).then(resolve).catch(reject);
          return;
        }

        if (res.statusCode && res.statusCode >= 400) {
          res.resume();
          reject(new Error(`GitHub API error: HTTP ${res.statusCode}`));
          return;
        }

        let body = '';
        res.on('data', (chunk: Buffer) => body += chunk.toString());
        res.on('end', () => {
          try { resolve(JSON.parse(body)); }
          catch { reject(new Error('Invalid JSON from GitHub API')); }
        });
        res.on('error', reject);
      }).on('error', reject);
    });
  }

  /** Extract a ZIP archive using platform tools */
  private async _extractZip(zipPath: string, targetDir: string): Promise<void> {
    if (process.platform === 'win32') {
      // PowerShell Expand-Archive
      await execFileAsync('powershell.exe', [
        '-NoProfile', '-Command',
        `Expand-Archive -Path '${zipPath}' -DestinationPath '${targetDir}' -Force`,
      ], { timeout: 120_000 });
    } else {
      // macOS/Linux: unzip
      await execFileAsync('unzip', ['-o', zipPath, '-d', targetDir], {
        timeout: 120_000,
      });
    }
  }

  /** Recursively search for a file by name in a directory */
  private _findFile(dir: string, filename: string): string | null {
    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isFile() && entry.name === filename) return fullPath;
        if (entry.isDirectory()) {
          const found = this._findFile(fullPath, filename);
          if (found) return found;
        }
      }
    } catch {}
    return null;
  }

  /** Build the complete list of entries including sd.exe for status display */
  private _getAllEntries(): ManifestEntry[] {
    const sdEntry: ManifestEntry = {
      filename: REQUIRED_FILES.sdCli,
      url: '', // resolved at download time
      sizeBytes: process.platform === 'win32' ? 336_000_000 : 21_000_000, // estimate
      description: `stable-diffusion.cpp (${process.platform === 'win32' ? 'CUDA' : 'native'})`,
    };
    return [...MODEL_MANIFEST, sdEntry];
  }

  // ── File download (shared) ───────────────────────────────────────────

  private async _downloadFile(entry: ManifestEntry, dir: string): Promise<void> {
    const partPath = path.join(dir, `${entry.filename}.part`);
    const finalPath = path.join(dir, entry.filename);

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
        if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          this._httpDownload(res.headers.location, partPath, progress, startByte, redirectCount + 1)
            .then(resolve).catch(reject);
          return;
        }

        if (res.statusCode === 416) {
          res.resume();
          resolve();
          return;
        }

        if (res.statusCode && res.statusCode >= 400) {
          res.resume();
          reject(new Error(`HTTP ${res.statusCode}: ${res.statusMessage}`));
          return;
        }

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
