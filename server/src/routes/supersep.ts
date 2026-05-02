// supersep.ts — SuperSep stem separation route (proxy to ace-server)
//
// Routes:
//   POST /api/supersep/separate — upload audio, start separation job
//   GET  /api/supersep/:jobId/progress — poll job progress
//   GET  /api/supersep/:jobId/result — get stem list metadata
//   GET  /api/supersep/:jobId/stem/:index — download individual stem WAV
//   POST /api/supersep/recombine — remix stems with volume/mute controls

import { Router } from 'express';
import { aceClient } from '../services/aceClient.js';
import { config } from '../config.js';

const router = Router();

const ACE_URL = config.aceServer.url;

// POST /api/supersep/separate
// Accepts: multipart/form-data with 'audio' field, or raw audio body
// Query: level=0..3 (BASIC/VOCAL_SPLIT/FULL/MAXIMUM)
router.post('/separate', async (req, res) => {
  try {
    const level = parseInt(String(req.query.level ?? '0'), 10);

    // Forward raw body to ace-server
    const aceRes = await fetch(
      `${ACE_URL}/supersep/separate?level=${level}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: req.body,
      }
    );

    if (!aceRes.ok) {
      const err = await aceRes.text();
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
