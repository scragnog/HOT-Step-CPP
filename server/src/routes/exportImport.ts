// Mount at /api/export-import. Clients send IDs and options, never local paths.
import { Router } from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import multer from 'multer';
import { config } from '../config.js';
import { getDb } from '../db/database.js';
import { getUserId } from './auth.js';
import { recordAudioAsset } from '../services/assets/audioAssets.js';
import { IMPORT_EXTENSIONS } from '../services/library/importTrack.js';
import { exportRequest, importRequest, validateProfileImport } from '../services/exportImport/contracts.js';
import { importAssets, resolveExports } from '../services/exportImport/operations.js';

const router = Router();
const upload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, callback) => {
      const dir = path.join(config.data.dir, 'references');
      fs.mkdirSync(dir, { recursive: true });
      callback(null, dir);
    },
    filename: (_req, file, callback) => callback(null, `${randomUUID()}${path.extname(file.originalname).toLowerCase()}`),
  }),
  limits: { fileSize: 500 * 1024 * 1024, files: 1 },
  fileFilter: (_req, file, callback) => callback(null, IMPORT_EXTENSIONS.includes(path.extname(file.originalname).toLowerCase())),
});

router.post('/assets', (req, res, next) => {
  if (!getUserId(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
  next();
}, upload.single('audio'), async (req, res) => {
  const userId = getUserId(req)!;
  if (!req.file) { res.status(400).json({ error: 'Unsupported or missing audio file' }); return; }
  try {
    const file = req.file;
    const hash = createHash('sha256');
    for await (const chunk of fs.createReadStream(file.path)) hash.update(chunk);
    const asset = recordAudioAsset(getDb(), {
      userId, url: `/references/${file.filename}`, filename: file.originalname,
      size: file.size, sha256: hash.digest('hex'),
    });
    res.json({ assetId: asset.id });
  } catch (error) {
    try { fs.unlinkSync(req.file.path); } catch { /* already gone */ }
    res.status(500).json({ error: (error as Error).message });
  }
});

router.post('/exports/resolve', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }
  const parsed = exportRequest.safeParse(req.body);
  if (!parsed.success) { res.status(400).json({ error: 'Invalid export request', issues: parsed.error.issues }); return; }
  res.json({ items: resolveExports(getDb(), userId, parsed.data) });
});

router.post('/imports', async (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }
  const parsed = importRequest.safeParse(req.body);
  if (!parsed.success) { res.status(400).json({ error: 'Invalid import request', issues: parsed.error.issues }); return; }
  res.json({ items: await importAssets(getDb(), userId, parsed.data) });
});

router.post('/profiles/validate', (req, res) => {
  if (!getUserId(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
  try { res.json(validateProfileImport(req.body)); }
  catch (error) { res.status(400).json({ error: 'Invalid profile', details: (error as Error).message }); }
});

export default router;
