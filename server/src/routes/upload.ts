/**
 * upload.ts — Audio file upload route for Cover Studio
 *
 * POST /api/upload/audio — Upload audio file, save to data/references/
 */
import { Router, Request, Response } from 'express';
import path from 'path';
import fs from 'fs';
import { randomUUID } from 'crypto';
import multer from 'multer';
import { config } from '../config.js';

const router = Router();

const ALLOWED_EXTENSIONS = ['.mp3', '.wav', '.flac', '.m4a', '.mp4', '.ogg', '.opus', '.webm', '.aac'];

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 200 * 1024 * 1024 }, // 200MB
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (ALLOWED_EXTENSIONS.includes(ext) || file.mimetype.startsWith('audio/') || file.mimetype === 'application/octet-stream') {
      cb(null, true);
    } else {
      cb(new Error(`Invalid file type "${file.originalname}" (${file.mimetype}). Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`));
    }
  },
});

/**
 * POST /api/upload/audio
 * Multipart form: field "audio" with audio file
 * Returns: { audio_url: "/references/<uuid>.<ext>", filename: "original.mp3" }
 */
router.post('/audio', upload.single('audio'), (req: Request, res: Response) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: 'No file uploaded' });
      return;
    }

    const ext = path.extname(req.file.originalname).toLowerCase() || '.mp3';
    const filename = `${randomUUID()}${ext}`;
    const refsDir = path.join(config.data.dir, 'references');

    // Ensure references dir exists
    fs.mkdirSync(refsDir, { recursive: true });

    const filePath = path.join(refsDir, filename);
    fs.writeFileSync(filePath, req.file.buffer);

    console.log(`[upload] Saved ${req.file.originalname} (${(req.file.size / 1024 / 1024).toFixed(1)} MB) → ${filename}`);

    res.json({
      audio_url: `/references/${filename}`,
      filename: req.file.originalname,
    });
  } catch (err: any) {
    console.error('[upload] Failed:', err.message);
    res.status(500).json({ error: 'Upload failed', details: err.message });
  }
});

/**
 * POST /api/upload/latent
 * Multipart form: field "latent" with .latent file
 * Returns: { latent_url: "/references/<uuid>.latent", metadata: { ... } }
 *
 * Accepts both HSLAT-headerered files and raw float32 files.
 * If HSLAT, embedded metadata is returned for UI pre-population.
 */
const latentUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB (latents are small — typically <1MB)
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (ext === '.latent' || file.mimetype === 'application/octet-stream') {
      cb(null, true);
    } else {
      cb(new Error(`Invalid file type "${file.originalname}". Expected .latent file.`));
    }
  },
});

router.post('/latent', latentUpload.single('latent'), (req: Request, res: Response) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: 'No file uploaded' });
      return;
    }

    // Import readHslat dynamically to avoid circular deps at module load
    // (latentFormat is a pure utility, no circular risk, but keeping import local for clarity)
    const { readHslat } = require('../services/latentFormat.js');

    const buf = req.file.buffer;
    let metadata: Record<string, unknown> = {};

    try {
      const parsed = readHslat(buf);
      metadata = parsed.metadata;
      // Validate raw latent portion
      if (parsed.rawLatent.length > 0 && parsed.rawLatent.length % 256 !== 0) {
        res.status(400).json({
          error: `Latent data size ${parsed.rawLatent.length} is not a multiple of 256 bytes (64 × float32)`,
        });
        return;
      }
    } catch (parseErr: any) {
      res.status(400).json({ error: `Invalid latent file: ${parseErr.message}` });
      return;
    }

    const filename = `${randomUUID()}.latent`;
    const refsDir = path.join(config.data.dir, 'references');
    fs.mkdirSync(refsDir, { recursive: true });
    const filePath = path.join(refsDir, filename);
    fs.writeFileSync(filePath, buf);

    console.log(`[upload] Latent saved: ${req.file.originalname} (${(buf.length / 1024).toFixed(0)} KB) → ${filename}`);

    res.json({
      latent_url: `/references/${filename}`,
      filename: req.file.originalname,
      metadata,
    });
  } catch (err: any) {
    console.error('[upload] Latent upload failed:', err.message);
    res.status(500).json({ error: 'Latent upload failed', details: err.message });
  }
});

export default router;
