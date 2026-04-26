/**
 * analyze.ts — Audio analysis routes
 *
 * POST /api/analyze         — Essentia CLI: BPM + key detection
 * POST /api/analyze/metadata — music-metadata: ID3/Vorbis/FLAC tag extraction
 */
import { Router, Request, Response } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { execFile } from 'child_process';
import { promises as fs } from 'fs';
import os from 'os';
import multer from 'multer';
import { parseBuffer } from 'music-metadata';
import { config } from '../config.js';

const router = Router();

const AUDIO_DIR = path.join(config.data.dir, 'audio');

/** Resolve a frontend audio URL (e.g. `/audio/xxx.mp3`) to an absolute file path. */
const resolveAudioPath = (audioUrl: string): string => {
    if (audioUrl.startsWith('/audio/')) {
        return path.join(AUDIO_DIR, audioUrl.replace('/audio/', ''));
    }
    if (audioUrl.startsWith('/references/')) {
        return path.join(config.data.dir, 'references', audioUrl.replace('/references/', ''));
    }
    if (audioUrl.startsWith('http')) {
        try {
            const parsed = new URL(audioUrl);
            if (parsed.pathname.startsWith('/audio/')) {
                return path.join(AUDIO_DIR, parsed.pathname.replace('/audio/', ''));
            }
            if (parsed.pathname.startsWith('/references/')) {
                return path.join(config.data.dir, 'references', parsed.pathname.replace('/references/', ''));
            }
        } catch {
            // fall through
        }
    }
    return audioUrl;
};

/**
 * POST /api/analyze
 * Body: { audioUrl: string }
 * Returns: { bpm: number, key: string, scale: string } or error
 */
router.post('/', async (req: Request, res: Response) => {
    const { audioUrl } = req.body;
    if (!audioUrl) {
        res.status(400).json({ error: 'audioUrl is required' });
        return;
    }

    const audioPath = resolveAudioPath(audioUrl);

    // Verify the file exists
    try {
        await fs.access(audioPath);
    } catch {
        res.status(404).json({ error: `Audio file not found: ${audioPath}` });
        return;
    }

    // Create temp output file for Essentia
    const tmpFile = path.join(os.tmpdir(), `essentia_${Date.now()}.json`);

    // Essentia supports wav, mp3, flac natively. Convert anything else via ffmpeg.
    const SUPPORTED_EXTS = ['.wav', '.mp3', '.flac', '.aiff', '.aif'];
    const ext = path.extname(audioPath).toLowerCase();
    let inputPath = audioPath;
    let tmpWav: string | null = null;

    if (!SUPPORTED_EXTS.includes(ext)) {
        tmpWav = path.join(os.tmpdir(), `essentia_input_${Date.now()}.wav`);
        console.log(`[analyze] Converting ${ext} to WAV via ffmpeg...`);
        try {
            await new Promise<void>((resolve, reject) => {
                execFile('ffmpeg', ['-y', '-i', audioPath, '-ar', '44100', '-ac', '2', tmpWav!],
                    { timeout: 60_000 },
                    (error) => error ? reject(error) : resolve()
                );
            });
            inputPath = tmpWav;
        } catch (ffErr: any) {
            console.error('[analyze] ffmpeg conversion failed:', ffErr.message);
            res.status(500).json({ error: `Format conversion failed: ${ffErr.message}` });
            return;
        }
    }

    try {
        // Run Essentia CLI
        await new Promise<string>((resolve, reject) => {
            execFile(
                config.essentia.bin,
                [inputPath, tmpFile],
                { timeout: 120_000, maxBuffer: 10 * 1024 * 1024 },
                (error, _stdout, _stderr) => {
                    // Essentia writes info to stderr even on success — check if output file exists
                    if (error && !error.killed) {
                        fs.access(tmpFile).then(() => resolve('ok')).catch(() => reject(error));
                    } else {
                        resolve('ok');
                    }
                }
            );
        });
        const raw = await fs.readFile(tmpFile, 'utf-8');
        const data = JSON.parse(raw);

        const bpm = Math.round(data?.rhythm?.bpm ?? 0);
        const keyData = data?.tonal?.key_edma ?? {};
        const key = keyData.key ?? '';
        const scale = keyData.scale ?? '';

        // Cleanup temp files
        fs.unlink(tmpFile).catch(() => { });
        if (tmpWav) fs.unlink(tmpWav).catch(() => { });

        console.log(`[analyze] BPM: ${bpm}, Key: ${key} ${scale} (from ${path.basename(audioPath)})`);

        res.json({ bpm, key, scale });
    } catch (err: any) {
        // Cleanup temp files on error
        fs.unlink(tmpFile).catch(() => { });
        if (tmpWav) fs.unlink(tmpWav).catch(() => { });
        console.error('[analyze] Essentia failed:', err.message || err);
        res.status(500).json({ error: `Analysis failed: ${err.message || 'Unknown error'}` });
    }
});

// ── Metadata extraction (music-metadata) ─────────────────────────────────────

const metadataUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 200 * 1024 * 1024 }, // 200MB
});

/**
 * POST /api/analyze/metadata — extract ID3/Vorbis/FLAC tags from uploaded audio.
 * Returns { artist, title, album, duration, sampleRate, bitrate }.
 */
router.post('/metadata', metadataUpload.single('audio'), async (req: Request, res: Response) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: 'No file uploaded' });
      return;
    }
    const metadata = await parseBuffer(
      req.file.buffer,
      { mimeType: req.file.mimetype as any },
      { duration: true, skipCovers: true },
    );
    console.log(
      `[analyze/metadata] Extracted: artist="${metadata.common.artist || ''}", ` +
      `title="${metadata.common.title || ''}", album="${metadata.common.album || ''}"`,
    );
    res.json({
      artist: metadata.common.artist || '',
      title: metadata.common.title || '',
      album: metadata.common.album || '',
      duration: metadata.format.duration || null,
      sampleRate: metadata.format.sampleRate || null,
      bitrate: metadata.format.bitrate || null,
    });
  } catch (err: any) {
    console.error('[analyze/metadata] Failed:', err.message);
    res.status(500).json({ error: 'Failed to extract metadata', details: err.message });
  }
});

export default router;
