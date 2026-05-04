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
// Accepts: raw audio body (WAV, MP3, FLAC, OGG, M4A, etc.)
// Query: level=0..3 (BASIC/VOCAL_SPLIT/FULL/MAXIMUM)
// Non-WAV/MP3 is auto-converted to WAV via ffmpeg before forwarding to ace-server.
router.post('/separate', async (req, res) => {
  try {
    const level = parseInt(String(req.query.level ?? '0'), 10);
    let audioBody: Buffer = Buffer.isBuffer(req.body) ? req.body : Buffer.from(req.body || []);
    console.log(`[SuperSep] separate: level=${level}, body=${audioBody.length} bytes, type=${req.headers['content-type']}`);

    if (!audioBody.length) {
      return res.status(400).json({ error: 'No audio body received by Node server' });
    }

    // Detect format from magic bytes — ace-server only decodes WAV and MP3
    const isWav = audioBody.length >= 4 && audioBody.slice(0, 4).toString() === 'RIFF';
    const isMp3 = audioBody.length >= 3 && (
      (audioBody[0] === 0xFF && (audioBody[1] & 0xE0) === 0xE0) || // MP3 sync word
      audioBody.slice(0, 3).toString() === 'ID3'                    // ID3 header
    );

    if (!isWav && !isMp3) {
      // Convert to WAV via ffmpeg (handles FLAC, OGG, M4A, AAC, OPUS, etc.)
      console.log(`[SuperSep] Non-WAV/MP3 detected, converting via ffmpeg...`);
      const fs = await import('fs');
      const path = await import('path');
      const { execFileSync } = await import('child_process');
      // @ts-ignore
      const ffmpegPathImport = (await import('ffmpeg-static')).default;
      const ffmpegPath = ffmpegPathImport as unknown as string | null;

      if (!ffmpegPath) {
        return res.status(400).json({ error: 'ffmpeg not available — cannot convert this audio format. Upload WAV or MP3.' });
      }

      const tmpDir = path.default.join(process.cwd(), 'data', 'tmp');
      if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });
      const id = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
      const tmpIn = path.default.join(tmpDir, `supersep_in_${id}.dat`);
      const tmpOut = path.default.join(tmpDir, `supersep_out_${id}.wav`);

      try {
        fs.writeFileSync(tmpIn, audioBody);
        execFileSync(ffmpegPath, [
          '-y', '-i', tmpIn,
          '-ar', '44100', '-ac', '2',
          '-c:a', 'pcm_s16le', '-f', 'wav',
          tmpOut,
        ], { timeout: 120_000, stdio: ['pipe', 'pipe', 'pipe'] });
        audioBody = fs.readFileSync(tmpOut);
        console.log(`[SuperSep] Converted to WAV: ${audioBody.length} bytes`);
      } catch (convErr: any) {
        const stderr = convErr.stderr?.toString()?.slice(-300) || '';
        console.error(`[SuperSep] ffmpeg conversion failed:`, stderr || convErr.message);
        return res.status(400).json({ error: `Audio format conversion failed: ${stderr || convErr.message}` });
      } finally {
        try { fs.unlinkSync(tmpIn); } catch {}
        try { fs.unlinkSync(tmpOut); } catch {}
      }
    }

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
