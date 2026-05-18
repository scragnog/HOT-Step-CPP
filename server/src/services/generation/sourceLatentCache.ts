/**
 * File-based latent cache for VAE-encoded source/timbre audio.
 *
 * Stores raw f32 latent files alongside audio in a `.latent-cache/` directory.
 * Cache key is a hash of (audioPath + tempoScale + pitchShift), so the same
 * source with different processing params gets separate cache entries.
 *
 * Files are raw f32 [T*64] — matching the upstream acestep.cpp wire format.
 * No HSLAT wrapper: these are internal cache files, not user-facing exports.
 */

import { createHash } from 'crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync, statSync } from 'fs';
import { dirname, join } from 'path';

const CACHE_DIR_NAME = '.latent-cache';

/**
 * Build a deterministic cache key from audio path + processing params.
 * Returns a hex SHA-256 hash suitable for use as a filename.
 */
function cacheKey(audioPath: string, tempo?: number, pitch?: number, vaeModel?: string): string {
  const parts = [audioPath];
  if (tempo !== undefined && tempo !== 1.0) parts.push(`tempo=${tempo}`);
  if (pitch !== undefined && pitch !== 0)   parts.push(`pitch=${pitch}`);
  if (vaeModel) parts.push(`vae=${vaeModel}`);
  return createHash('sha256').update(parts.join('|')).digest('hex');
}

/**
 * Resolve the cache directory for a given audio file.
 * Creates `.latent-cache/` next to the audio file if it doesn't exist.
 */
function cacheDir(audioPath: string): string {
  const dir = join(dirname(audioPath), CACHE_DIR_NAME);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  return dir;
}

/**
 * Look up a cached latent for the given audio + processing params.
 * Returns the raw f32 Buffer if found, or undefined on miss.
 */
export function getCachedLatent(audioPath: string, tempo?: number, pitch?: number, vaeModel?: string): Buffer | undefined {
  try {
    const key = cacheKey(audioPath, tempo, pitch, vaeModel);
    const filePath = join(cacheDir(audioPath), `${key}.raw`);
    if (!existsSync(filePath)) return undefined;

    const buf = readFileSync(filePath);
    // Validate: must be a multiple of 256 bytes (64 floats per frame)
    if (buf.length === 0 || buf.length % 256 !== 0) {
      console.warn(`[Latent Cache] Corrupt cache file (${buf.length} bytes), ignoring: ${filePath}`);
      return undefined;
    }

    const stat = statSync(filePath);
    console.log(`[Latent Cache] HIT — ${(buf.length / 1024).toFixed(0)} KB, cached ${stat.mtime.toISOString()}`);
    return buf;
  } catch (err) {
    console.warn(`[Latent Cache] Read error: ${err}`);
    return undefined;
  }
}

/**
 * Save a latent buffer to the file cache.
 */
export function saveCachedLatent(audioPath: string, latent: Buffer, tempo?: number, pitch?: number, vaeModel?: string): void {
  try {
    const key = cacheKey(audioPath, tempo, pitch, vaeModel);
    const filePath = join(cacheDir(audioPath), `${key}.raw`);
    writeFileSync(filePath, latent);
    console.log(`[Latent Cache] STORED — ${(latent.length / 1024).toFixed(0)} KB → ${filePath}`);
  } catch (err) {
    console.warn(`[Latent Cache] Write error: ${err}`);
  }
}
