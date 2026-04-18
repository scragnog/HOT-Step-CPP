// mastering.ts — Mastering routes for reference track management and mastering execution
//
// Endpoints:
//   POST   /api/mastering/upload-reference  — Upload a reference audio file
//   GET    /api/mastering/references         — List uploaded reference tracks
//   DELETE /api/mastering/references/:name   — Delete a reference track
//   POST   /api/mastering/run               — Run mastering on an existing song

import { Router } from 'express';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import multer from 'multer';
import { config } from '../config.js';
import { getUserId } from './auth.js';
import { getDb } from '../db/database.js';

const execFileAsync = promisify(execFile);
const router = Router();

// Reference tracks directory
const refsDir = path.join(config.data.dir, 'references');
fs.mkdirSync(refsDir, { recursive: true });

// Multer for reference file uploads
const upload = multer({
  dest: refsDir,
  limits: { fileSize: 100 * 1024 * 1024 }, // 100MB max
  fileFilter: (_req, file, cb) => {
    const allowed = ['audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/mp3', 'audio/flac',
                     'audio/ogg', 'audio/aac', 'audio/mp4', 'application/octet-stream'];
    cb(null, true); // Accept all — we'll check extension
  },
});

/** Resolve the mastering.exe path (same directory as ace-server exe) */
function getMasteringExePath(): string {
  const aceExe = config.aceServer.exe;
  if (aceExe) {
    return path.join(path.dirname(aceExe), 'mastering.exe');
  }
  // Fallback: look relative to project root
  return path.resolve(process.cwd(), '..', 'engine', 'build', 'Release', 'mastering.exe');
}

/** Run mastering.exe on a target WAV, given a reference WAV */
export async function runMastering(targetWavPath: string, referenceWavPath: string, outputWavPath: string): Promise<void> {
  const exe = getMasteringExePath();

  if (!fs.existsSync(exe)) {
    throw new Error(`mastering.exe not found at ${exe}`);
  }
  if (!fs.existsSync(targetWavPath)) {
    throw new Error(`Target file not found: ${targetWavPath}`);
  }
  if (!fs.existsSync(referenceWavPath)) {
    throw new Error(`Reference file not found: ${referenceWavPath}`);
  }

  console.log(`[Mastering] Running: ${exe}`);
  console.log(`[Mastering]   target:    ${targetWavPath}`);
  console.log(`[Mastering]   reference: ${referenceWavPath}`);
  console.log(`[Mastering]   output:    ${outputWavPath}`);

  const { stderr } = await execFileAsync(exe, [
    '--target', targetWavPath,
    '--reference', referenceWavPath,
    '--output', outputWavPath,
  ], { timeout: 120_000 }); // 2 minute timeout

  // Log mastering output (it goes to stderr)
  if (stderr) {
    for (const line of stderr.split('\n')) {
      if (line.trim()) console.log(`[Mastering] ${line.trim()}`);
    }
  }
}

// ── POST /upload-reference ──────────────────────────────────
router.post('/upload-reference', upload.single('file'), (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const file = req.file;
  if (!file) { res.status(400).json({ error: 'No file uploaded' }); return; }

  // Rename to original filename (sanitized)
  const safeName = file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_');
  const newPath = path.join(refsDir, safeName);

  // Avoid overwriting
  let finalPath = newPath;
  let finalName = safeName;
  if (fs.existsSync(newPath)) {
    const ext = path.extname(safeName);
    const base = path.basename(safeName, ext);
    finalName = `${base}_${Date.now()}${ext}`;
    finalPath = path.join(refsDir, finalName);
  }

  fs.renameSync(file.path, finalPath);

  console.log(`[Mastering] Reference uploaded: ${finalName}`);
  res.json({
    name: finalName,
    path: finalPath,
    url: `/references/${finalName}`,
  });
});

// ── GET /references ─────────────────────────────────────────
router.get('/references', (_req, res) => {
  try {
    const files = fs.readdirSync(refsDir)
      .filter(f => !f.startsWith('.'))
      .map(f => ({
        name: f,
        path: path.join(refsDir, f),
        size: fs.statSync(path.join(refsDir, f)).size,
        url: `/references/${f}`,
      }));
    res.json({ references: files });
  } catch {
    res.json({ references: [] });
  }
});

// ── DELETE /references/:name ────────────────────────────────
router.delete('/references/:name', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const filePath = path.join(refsDir, req.params.name);
  if (!fs.existsSync(filePath)) {
    res.status(404).json({ error: 'Reference not found' });
    return;
  }

  // Security: ensure the path is within refsDir
  if (!filePath.startsWith(refsDir)) {
    res.status(400).json({ error: 'Invalid path' });
    return;
  }

  fs.unlinkSync(filePath);
  console.log(`[Mastering] Reference deleted: ${req.params.name}`);
  res.json({ ok: true });
});

// ── POST /run — Run mastering on existing song ──────────────
router.post('/run', async (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const { songId, referenceName } = req.body;
  if (!songId || !referenceName) {
    res.status(400).json({ error: 'songId and referenceName are required' });
    return;
  }

  try {
    // Get song from DB
    const song = getDb().prepare('SELECT * FROM songs WHERE id = ?').get(songId) as any;
    if (!song) { res.status(404).json({ error: 'Song not found' }); return; }

    // Resolve file paths
    const audioUrl = song.audio_url; // e.g. /audio/uuid.wav
    const audioFilename = path.basename(audioUrl);
    const targetPath = path.join(config.data.audioDir, audioFilename);
    const referencePath = path.join(refsDir, referenceName);

    if (!fs.existsSync(targetPath)) {
      res.status(404).json({ error: `Audio file not found: ${audioFilename}` });
      return;
    }
    if (!fs.existsSync(referencePath)) {
      res.status(404).json({ error: `Reference not found: ${referenceName}` });
      return;
    }

    // Output path: same name with _mastered suffix
    const ext = path.extname(audioFilename);
    const base = path.basename(audioFilename, ext);
    const masteredFilename = `${base}_mastered${ext}`;
    const masteredPath = path.join(config.data.audioDir, masteredFilename);
    const masteredUrl = `/audio/${masteredFilename}`;

    // Run mastering
    await runMastering(targetPath, referencePath, masteredPath);

    // Update DB with mastered URL
    getDb().prepare('UPDATE songs SET mastered_audio_url = ? WHERE id = ?')
      .run(masteredUrl, songId);

    console.log(`[Mastering] Song ${songId} mastered → ${masteredUrl}`);
    res.json({
      ok: true,
      masteredUrl,
      songId,
    });
  } catch (err: any) {
    console.error(`[Mastering] Failed:`, err.message);
    res.status(500).json({ error: err.message });
  }
});

export default router;
