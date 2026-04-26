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

export default router;
