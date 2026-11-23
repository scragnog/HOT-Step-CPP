// artwork.ts — GET /api/artwork?url=<https url>
// Serves remote artwork (Lyric Studio artist and album art) from a disk cache so the
// UI can load images from 'self' only (CSP img-src). The URL must already be stored
// on an artist, album, song or recording, or be on a Genius host (search previews).

import { Router } from 'express';
import { getDb } from '../db/database.js';
import { ArtworkError, getArtwork, isPreviewHost, parseArtworkUrl } from '../services/artwork/artworkCache.js';

const router = Router();

function isStoredUrl(url: string): boolean {
  return !!getDb().prepare(
    'SELECT 1 FROM artists WHERE image_url = ? UNION ALL SELECT 1 FROM lyrics_sets WHERE image_url = ?'
    + ' UNION ALL SELECT 1 FROM songs WHERE cover_url = ? UNION ALL SELECT 1 FROM audio_generations WHERE cover_url = ? LIMIT 1',
  ).get(url, url, url, url);
}

router.get('/', async (req, res) => {
  try {
    const url = typeof req.query.url === 'string' ? req.query.url : '';
    const u = parseArtworkUrl(url);
    if (!isPreviewHost(u) && !isStoredUrl(url)) throw new ArtworkError(403, 'URL is not a stored artwork URL');
    const { file, type } = await getArtwork(url);
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Cache-Control', 'private, max-age=86400');
    res.type(type).sendFile(file);
  } catch (e) {
    const status = e instanceof ArtworkError ? e.status : 500;
    if (!res.headersSent) res.status(status).json({ error: e instanceof Error ? e.message : 'Artwork failed' });
  }
});

export default router;
