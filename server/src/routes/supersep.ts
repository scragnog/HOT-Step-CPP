// supersep.ts — SuperSep stem separation route (proxy to ace-server)
//
// Routes:
//   POST /api/supersep/separate — start separation (reads file from disk)
//   GET  /api/supersep/:jobId/progress — poll job progress
//   GET  /api/supersep/:jobId/result — get stem list metadata
//   GET  /api/supersep/:jobId/stem/:index — download individual stem WAV
//   POST /api/supersep/recombine — remix stems with volume/mute controls

import { Router } from 'express';
import { config } from '../config.js';
import { ensureEngineFormat } from '../services/audioConvert.js';
import path from 'path';

const router = Router();

const ACE_URL = config.aceServer.url;

// POST /api/supersep/separate
// Body (JSON): { audioUrl: "/references/uuid.flac" }
// Query: level=0..5 (BASIC/VOCAL_SPLIT/FULL/MAXIMUM/VOCALS_ONLY/STABLESTEP)
//        4 = 2-stem via the 6-stem BS-RoFormer (instrumental = mix − vocals)
//        5 = 2-stem via the dual BS-Roformer-Leap Xe models (both stems neural)
//        See SuperSepLevel in engine/src/supersep.h.
// Reads the file from disk, converts non-WAV/MP3 to WAV, forwards to ace-server.
router.post('/separate', async (req, res) => {
  try {
    const level = parseInt(String(req.query.level ?? '0'), 10);
    const { audioUrl } = req.body || {};

    if (!audioUrl || typeof audioUrl !== 'string') {
      return res.status(400).json({ error: 'audioUrl required in request body' });
    }

    // Resolve server-side file path from URL
    // audioUrl is like "/references/uuid.flac" → data/references/uuid.flac
    const basename = path.basename(audioUrl);
    let filePath: string;
    if (audioUrl.startsWith('/references/')) {
      filePath = path.join(config.data.dir, 'references', basename);
    } else if (audioUrl.startsWith('/audio/')) {
      filePath = path.join(config.data.audioDir, basename);
    } else {
      // Fallback: try in references
      filePath = path.join(config.data.dir, 'references', basename);
    }

    console.log(`[SuperSep] separate: level=${level}, file=${filePath}`);

    // Read and convert to engine-compatible format (WAV/MP3)
    let audioBody: Buffer;
    try {
      audioBody = ensureEngineFormat(filePath);
    } catch (err: any) {
      console.error(`[SuperSep] Format conversion failed:`, err.message);
      return res.status(400).json({ error: `Audio conversion failed: ${err.message}` });
    }

    console.log(`[SuperSep] Forwarding ${audioBody.length} bytes to ace-server`);

    // Forward WAV/MP3 body to ace-server
    const aceRes = await fetch(
      `${ACE_URL}/supersep/separate?level=${level}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: audioBody,
      }
    );

    if (!aceRes.ok) {
      const err = await aceRes.text();
      console.error(`[SuperSep] ace-server returned ${aceRes.status}: ${err}`);
      return res.status(aceRes.status).json({ error: err });
    }

    const data = await aceRes.json();
    res.json(data);
  } catch (err: any) {
    console.error('[SuperSep] separate error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// GET /api/supersep/:jobId/progress
router.get('/:jobId/progress', async (req, res) => {
  try {
    const aceRes = await fetch(
      `${ACE_URL}/supersep/progress?id=${req.params.jobId}`
    );
    const data = await aceRes.json();
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/supersep/:jobId/release — let the engine drop a finished job.
// The engine keeps every job resident (its pool has no eviction); Cover
// Studio calls this when a split is cleared or replaced (#132).
router.post('/:jobId/release', async (req, res) => {
  try {
    const aceRes = await fetch(`${ACE_URL}/supersep/release?id=${req.params.jobId}`, { method: 'POST' });
    res.status(aceRes.ok ? 200 : aceRes.status).json({ ok: aceRes.ok });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/supersep/:jobId/result
router.get('/:jobId/result', async (req, res) => {
  try {
    const aceRes = await fetch(
      `${ACE_URL}/supersep/result?id=${req.params.jobId}`
    );
    if (!aceRes.ok) {
      return res.status(aceRes.status).json(await aceRes.json());
    }
    const data = await aceRes.json();
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/supersep/:jobId/stem/:index — proxy WAV download
router.get('/:jobId/stem/:index', async (req, res) => {
  try {
    const aceRes = await fetch(
      `${ACE_URL}/supersep/serve?id=${req.params.jobId}&stem=${req.params.index}`
    );
    if (!aceRes.ok) {
      return res.status(aceRes.status).json({ error: 'Failed to fetch stem' });
    }
    const buf = Buffer.from(await aceRes.arrayBuffer());
    res.set('Content-Type', 'audio/wav');
    res.set('Content-Disposition', `attachment; filename="stem_${req.params.index}.wav"`);
    res.send(buf);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/supersep/recombine — remix stems and return WAV
router.post('/recombine', async (req, res) => {
  try {
    const aceRes = await fetch(`${ACE_URL}/supersep/recombine`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
    });
    if (!aceRes.ok) {
      return res.status(aceRes.status).json(await aceRes.json());
    }
    const buf = Buffer.from(await aceRes.arrayBuffer());
    res.set('Content-Type', 'audio/wav');
    res.send(buf);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
