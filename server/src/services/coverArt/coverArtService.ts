// coverArtService.ts — Cover art generation via stable-diffusion.cpp (sd-cli)
//
// Spawns sd-cli.exe as a subprocess to generate album cover art using
// FLUX.2-klein-4B. Same integration pattern as mastering.exe.
//
// Output: 1024×1024 PNG saved as WebP alongside the song's audio file.

import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { randomInt } from 'crypto';
import { v4 as uuidv4 } from 'uuid';
import { config } from '../../config.js';
import { getDb } from '../../db/database.js';
import { buildCoverArtPrompt, type CoverArtPromptOpts } from './promptBuilder.js';

const execFileAsync = promisify(execFile);

// ── Constants ───────────────────────────────────────────────────────────

/** Directory name within the models folder for cover art assets */
const COVER_ART_DIR = 'cover-art';

/** Expected filenames within the cover-art directory */
export const REQUIRED_FILES = {
  sdCli: process.platform === 'win32' ? 'sd.exe' : 'sd',
  diffusionModel: 'flux-2-klein-4b-Q4_0.gguf',
  vae: 'flux2_vae.safetensors',
  llm: 'Qwen3-4B-Q4_K_M.gguf',
} as const;

/** Generation parameters */
const GEN_WIDTH = 1024;
const GEN_HEIGHT = 1024;
const GEN_STEPS = 4;
const GEN_CFG_SCALE = 1.0;
const GEN_TIMEOUT_MS = 180_000; // 3 minutes

// ── Path resolution ─────────────────────────────────────────────────────

/** Get the cover-art assets directory */
export function getCoverArtDir(): string {
  return path.join(config.aceServer.models, COVER_ART_DIR);
}

/** Resolve path to a file in the cover-art directory */
function getFilePath(filename: string): string {
  return path.join(getCoverArtDir(), filename);
}

// ── Readiness check ─────────────────────────────────────────────────────

export interface CoverArtStatus {
  installed: boolean;
  missingFiles: string[];
  dir: string;
}

/** Check if all required files for cover art generation are present. */
export function getCoverArtReadiness(): CoverArtStatus {
  const dir = getCoverArtDir();
  const missing: string[] = [];

  for (const [, filename] of Object.entries(REQUIRED_FILES)) {
    const filePath = path.join(dir, filename);
    if (!fs.existsSync(filePath)) {
      missing.push(filename);
    }
  }

  return {
    installed: missing.length === 0,
    missingFiles: missing,
    dir,
  };
}

// ── Generation ──────────────────────────────────────────────────────────

export interface GenerateCoverArtOpts extends CoverArtPromptOpts {
  songId: string;
}

/** Options for the image-only phase (no songId needed) */
export interface GenerateCoverImageOpts extends CoverArtPromptOpts {}

export interface CoverArtResult {
  coverUrl: string;
  prompt: string;
  durationMs: number;
}

/**
 * Phase 1: Generate the cover art image (GPU-heavy, no DB writes).
 *
 * 1. Builds a prompt from song metadata
 * 2. Spawns sd-cli.exe with FLUX.2-klein-4B
 * 3. Saves the output as PNG in the audio directory
 *
 * Returns the coverUrl and prompt. Does NOT touch the database.
 * Use linkCoverToSong() afterwards to associate with a song.
 */
export async function generateCoverImage(opts: GenerateCoverImageOpts): Promise<CoverArtResult> {
  const startTime = Date.now();

  // Verify readiness
  const status = getCoverArtReadiness();
  if (!status.installed) {
    throw new Error(`Cover art not ready — missing: ${status.missingFiles.join(', ')}`);
  }

  // Build prompt
  const prompt = buildCoverArtPrompt(opts);
  console.log(`[CoverArt] Prompt: "${prompt}"`);

  // Resolve paths
  const sdCli = getFilePath(REQUIRED_FILES.sdCli);
  const diffusionModel = getFilePath(REQUIRED_FILES.diffusionModel);
  const vae = getFilePath(REQUIRED_FILES.vae);
  const llm = getFilePath(REQUIRED_FILES.llm);

  // Output to audio directory as PNG (sd-cli determines format from extension)
  const outputFilename = `cover_${uuidv4()}.png`;
  const outputPath = path.join(config.data.audioDir, outputFilename);

  // Random seed — each cover should be unique
  const seed = randomInt(0, 2 ** 32);

  // Build sd-cli command
  const args = [
    '--diffusion-model', diffusionModel,
    '--vae', vae,
    '--llm', llm,
    '-p', prompt,
    '-n', 'text, lettering, words, typography, watermark, signature, logo, title, font, writing, caption, label, stamp, banner',
    '--seed', String(seed),
    '--cfg-scale', String(GEN_CFG_SCALE),
    '--steps', String(GEN_STEPS),
    '--width', String(GEN_WIDTH),
    '--height', String(GEN_HEIGHT),
    '--sampling-method', 'euler',
    '--diffusion-fa',
    '-o', outputPath,
  ];

  console.log(`[CoverArt] Running: ${path.basename(sdCli)} (${GEN_WIDTH}×${GEN_HEIGHT}, ${GEN_STEPS} steps, seed=${seed})`);

  try {
    const { stdout, stderr } = await execFileAsync(sdCli, args, {
      timeout: GEN_TIMEOUT_MS,
      maxBuffer: 10 * 1024 * 1024, // 10MB buffer for verbose output
    });

    // Log sd-cli output
    if (stdout) {
      for (const line of stdout.split('\n')) {
        if (line.trim()) console.log(`[CoverArt] ${line.trim()}`);
      }
    }
    if (stderr) {
      for (const line of stderr.split('\n')) {
        if (line.trim()) console.log(`[CoverArt] ${line.trim()}`);
      }
    }
  } catch (err: any) {
    // Clean up partial output
    try { if (fs.existsSync(outputPath)) fs.unlinkSync(outputPath); } catch {}
    // execFile's `message` is only "Command failed: <the whole command line>",
    // so re-echoing it told a reporter nothing they did not already know and
    // left the actual failure invisible (#161). sd-cli writes its real
    // diagnosis — the missing file, the CUDA/Metal error, the OOM — to stderr,
    // and the exit code distinguishes a crash from our own timeout.
    const tail = String(err.stderr || err.stdout || '')
      .split('\n').map((l: string) => l.trim()).filter(Boolean).slice(-15);
    for (const line of tail) console.error(`[CoverArt] sd-cli: ${line}`);
    const why = err.killed && err.signal
      ? `killed by ${err.signal} after ${Math.round(GEN_TIMEOUT_MS / 1000)}s`
      : typeof err.code === 'number' ? `exit ${err.code}`
      : err.code ? String(err.code) : 'no exit code';
    throw new Error(`sd-cli failed (${why})${tail.length ? `: ${tail[tail.length - 1]}` : ''}`);
  }

  // Verify output was created
  if (!fs.existsSync(outputPath)) {
    throw new Error('sd-cli completed but no output file was generated');
  }

  const coverUrl = `/audio/${outputFilename}`;
  const durationMs = Date.now() - startTime;

  console.log(`[CoverArt] Image generated: ${outputFilename} (${(durationMs / 1000).toFixed(1)}s)`);

  return {
    coverUrl,
    prompt,
    durationMs,
  };
}

/**
 * Phase 2: Link a generated cover image to a song in the database.
 * This is a lightweight DB UPDATE — no GPU work.
 */
export function linkCoverToSong(coverUrl: string, songId: string): void {
  try {
    getDb().prepare('UPDATE songs SET cover_url = ? WHERE id = ?')
      .run(coverUrl, songId);
    console.log(`[CoverArt] Linked cover to song ${songId}: ${coverUrl}`);
  } catch (dbErr: any) {
    console.error(`[CoverArt] DB update failed for ${songId}: ${dbErr.message}`);
  }
}

/**
 * Convenience wrapper: generate image + link to song in one call.
 * Used by the sequential (non-parallel) path and the cover art API endpoint.
 */
export async function generateCoverArt(opts: GenerateCoverArtOpts): Promise<CoverArtResult> {
  const result = await generateCoverImage(opts);
  linkCoverToSong(result.coverUrl, opts.songId);
  return result;
}
