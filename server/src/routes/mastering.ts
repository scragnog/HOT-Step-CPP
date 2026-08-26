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
import { config, getFFmpegPath } from '../config.js';
import { getUserId } from './auth.js';
import { precomputePeaks } from '../services/audio/peaks.js';
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

/** Resolve tool paths — both live in the same build directory as ace-server */
function getToolPath(name: string): string {
  // Strip .exe on non-Windows (macOS/Linux binaries have no extension)
  const binaryName = process.platform === 'win32' ? name : name.replace(/\.exe$/, '');
  const aceExe = config.aceServer.exe;
  if (aceExe) {
    return path.join(path.dirname(aceExe), binaryName);
  }
  return path.resolve(process.cwd(), '..', 'engine', 'build', 'Release', binaryName);
}

/** Convert any audio format to WAV using mp3-codec (for MP3) or ffmpeg (for everything else) */
export async function convertToWav(inputPath: string, outputWavPath: string): Promise<void> {
  const ext = path.extname(inputPath).toLowerCase();

  if (ext === '.wav') {
    // Already WAV — just copy
    fs.copyFileSync(inputPath, outputWavPath);
    return;
  }

  if (ext === '.mp3') {
    // Use mp3-codec.exe (no ffmpeg dependency needed)
    const codec = getToolPath('mp3-codec.exe');
    if (fs.existsSync(codec)) {
      console.log(`[Mastering] Converting MP3 → WAV via mp3-codec`);
      await execFileAsync(codec, ['-i', inputPath, '-o', outputWavPath], { timeout: 60_000 });
      return;
    }
  }

  // For FLAC, OGG, AAC, etc. — use ffmpeg
  const ffmpegPath = getFFmpegPath();
  if (!ffmpegPath) {
    throw new Error(
      `Cannot convert ${ext} to WAV. ffmpeg not available — provide a WAV/MP3 file.`
    );
  }
  console.log(`[Mastering] Converting ${ext} → WAV via ffmpeg`);
  try {
    await execFileAsync(ffmpegPath, [
      '-y', '-i', inputPath,
      '-ac', '2', '-ar', '48000', '-c:a', 'pcm_f32le',
      outputWavPath,
    ], { timeout: 120_000 });
  } catch {
    throw new Error(
      `ffmpeg conversion failed for ${ext}. Provide a WAV/MP3 file.`
    );
  }
}

/** Convert WAV back to MP3 using mp3-codec */
async function convertWavToMp3(wavPath: string, mp3Path: string, bitrate = 192): Promise<void> {
  const codec = getToolPath('mp3-codec.exe');
  if (!fs.existsSync(codec)) {
    throw new Error(`mp3-codec.exe not found at ${codec}`);
  }
  console.log(`[Mastering] Encoding WAV → MP3 (${bitrate} kbps)`);
  await execFileAsync(codec, ['-i', wavPath, '-o', mp3Path, '-b', String(bitrate)], { timeout: 60_000 });
}

/**
 * Run mastering on any supported audio format.
 *
 * Pipeline:
 *   1. Convert target + reference to temp WAV (if not already WAV)
 *   2. Run mastering.exe (WAV → WAV)
 *   3. If original target was MP3, re-encode mastered WAV to MP3
 *   4. Clean up temp files
 */
export async function runMastering(targetPath: string, referencePath: string, outputPath: string): Promise<void> {
  const exe = getToolPath('mastering.exe');

  if (!fs.existsSync(exe)) {
    throw new Error(`mastering.exe not found at ${exe}`);
  }
  if (!fs.existsSync(targetPath)) {
    throw new Error(`Target file not found: ${targetPath}`);
  }
  if (!fs.existsSync(referencePath)) {
    throw new Error(`Reference file not found: ${referencePath}`);
  }

  const targetExt = path.extname(targetPath).toLowerCase();
  const outputExt = path.extname(outputPath).toLowerCase();
  const tempDir = path.join(config.data.dir, 'mastering_temp');
  fs.mkdirSync(tempDir, { recursive: true });

  const tempId = Date.now().toString(36);
  const tempTargetWav = path.join(tempDir, `target_${tempId}.wav`);
  const tempRefWav = path.join(tempDir, `ref_${tempId}.wav`);
  const tempOutputWav = path.join(tempDir, `mastered_${tempId}.wav`);

  const tempFiles = [tempTargetWav, tempRefWav, tempOutputWav];

  try {
    // Step 1: Convert inputs to WAV
    console.log(`[Mastering] Preparing inputs...`);
    await convertToWav(targetPath, tempTargetWav);
    await convertToWav(referencePath, tempRefWav);

    // Step 2: Run mastering.exe on WAV files
    console.log(`[Mastering] Running mastering.exe`);
    console.log(`[Mastering]   target:    ${targetPath} (${targetExt})`);
    console.log(`[Mastering]   reference: ${referencePath}`);
    console.log(`[Mastering]   output:    ${outputPath} (${outputExt})`);

    const { stderr } = await execFileAsync(exe, [
      '--target', tempTargetWav,
      '--reference', tempRefWav,
      '--output', tempOutputWav,
      '--pcm32f',
    ], { timeout: 120_000 });

    if (stderr) {
      for (const line of stderr.split('\n')) {
        if (line.trim()) console.log(`[Mastering] ${line.trim()}`);
      }
    }

    // Step 3: Convert output to final format
    if (outputExt === '.mp3') {
      await convertWavToMp3(tempOutputWav, outputPath);
    } else {
      // WAV output — just move
      fs.copyFileSync(tempOutputWav, outputPath);
    }

    console.log(`[Mastering] Done → ${outputPath}`);
  } finally {
    // Step 4: Clean up temp files
    for (const f of tempFiles) {
      try { if (fs.existsSync(f)) fs.unlinkSync(f); } catch {}
    }
    try { fs.rmdirSync(tempDir); } catch {} // Remove if empty
  }
}

// ── POST /upload-reference ──────────────────────────────────
router.post('/upload-reference', upload.single('file'), async (req, res) => {
  try {
    const userId = getUserId(req);
    if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

    const file = req.file;
    if (!file) { res.status(400).json({ error: 'No file uploaded' }); return; }

    // Sanitize original filename
    const safeName = file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_');
    const ext = path.extname(safeName).toLowerCase();

    // Engine only supports WAV and MP3 — convert anything else to WAV on upload
    const needsConvert = ext !== '.wav' && ext !== '.mp3';
    const baseName = path.basename(safeName, ext);
    const targetName = needsConvert ? `${baseName}.wav` : safeName;

    // Avoid overwriting existing files
    let finalName = targetName;
    let finalPath = path.join(refsDir, finalName);
    if (fs.existsSync(finalPath)) {
      const targetExt = path.extname(targetName);
      const targetBase = path.basename(targetName, targetExt);
      finalName = `${targetBase}_${Date.now()}${targetExt}`;
      finalPath = path.join(refsDir, finalName);
    }

    if (needsConvert) {
      console.log(`[Mastering] Converting ${ext} → WAV: ${safeName}`);
      await convertToWav(file.path, finalPath);
      // Clean up the original temp file
      try { fs.unlinkSync(file.path); } catch {}
    } else {
      fs.renameSync(file.path, finalPath);
    }

    console.log(`[Mastering] Reference uploaded: ${finalName}`);
    res.json({
      name: finalName,
      path: finalPath,
      url: `/references/${finalName}`,
    });
  } catch (err: any) {
    console.error(`[Mastering] Upload failed:`, err.message);
    // Clean up temp file on error
    if (req.file?.path) {
      try { fs.unlinkSync(req.file.path); } catch {}
    }
    res.status(500).json({ error: err.message || 'Upload failed' });
  }
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

    // Build the waveform for the new file now, so switching to the mastered
    // variant in the player does not wait on it.
    precomputePeaks(masteredPath);

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
