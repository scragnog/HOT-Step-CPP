// audio.ts — audio metadata the player needs before it can draw anything.
//
// Right now that is waveform peaks. The player asks for them by the same URL it
// is about to hand its <audio> element, so nothing on the UI side has to know
// where files live on disk.

import { Router } from 'express';
import path from 'path';
import { config } from '../config.js';
import { getPeaks } from '../services/audio/peaks.js';

const router = Router();

/** The URL prefixes the player is allowed to ask about, and where they live.
 *  Anything else is refused — this endpoint takes a path from the browser, so
 *  the allowlist is the security boundary. */
function rootsFor(prefix: string): string | null {
  if (prefix === 'audio') return config.data.audioDir;
  if (prefix === 'references') return path.join(config.data.dir, 'references');
  return null;
}

/** Resolve a player URL to a file on disk, or null if it points anywhere we are
 *  not prepared to serve. Traversal is caught by comparing resolved paths
 *  rather than by scanning for '..', which decodes away. */
function resolveAudioUrl(url: string): string | null {
  let clean: string;
  try {
    clean = decodeURIComponent(url.split('?')[0].split('#')[0]);
  } catch {
    return null;
  }
  if (!clean.startsWith('/')) return null;

  const segments = clean.split('/').filter(Boolean);
  if (segments.length < 2) return null;

  const root = rootsFor(segments[0]);
  if (!root) return null;

  const resolved = path.resolve(root, ...segments.slice(1));
  const rootResolved = path.resolve(root);
  if (resolved !== rootResolved && !resolved.startsWith(rootResolved + path.sep)) return null;

  return resolved;
}

/**
 * GET /api/audio/peaks?url=/audio/<file>.wav
 *
 * Returns the waveform envelope for one file: 2000 min/max buckets plus the
 * true duration. Computed once per file and cached on disk, so this is a few
 * milliseconds after the first call.
 *
 * A file we cannot decode still answers 200, with `unsupported` set and empty
 * buckets. The player draws a flat strip and plays the track anyway — a
 * missing waveform must never be a reason audio does not come out.
 */
router.get('/peaks', (req, res) => {
  const url = String(req.query.url || '');
  if (!url) return res.status(400).json({ error: 'url query parameter is required' });

  const filePath = resolveAudioUrl(url);
  if (!filePath) return res.status(400).json({ error: 'url is not a servable audio path' });

  const peaks = getPeaks(filePath);
  // Peaks are immutable for a given file version and the cache key already
  // covers changes, so let the browser keep them for the session.
  res.setHeader('Cache-Control', 'private, max-age=3600');
  res.json(peaks);
});

export default router;
