// Mount at /api/midi-studio/soundfonts.
import { randomUUID } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { Router, type Request, type RequestHandler } from 'express';
import multer from 'multer';
import { config } from '../config.js';
import { SoundFontLibrary, SoundFontLibraryError, soundFontFormat } from '../services/soundFontLibrary.js';
import { getUserId } from './auth.js';

export const SOUND_FONT_SIZE_LIMIT = 1024 ** 3;

export function createSoundFontLibraryRouter(
  library = new SoundFontLibrary(path.join(config.data.dir, 'soundfonts')),
  authorized = getUserId,
  sizeLimit = SOUND_FONT_SIZE_LIMIT,
) {
  const router = Router();
  const upload = multer({
    storage: multer.diskStorage({
      destination: (_req, _file, done) => {
        try { done(null, library.uploadFolder()); }
        catch (error) { done(error as Error, ''); }
      },
      filename: (_req, _file, done) => done(null, `.upload-${randomUUID()}.tmp`),
    }),
    limits: { fileSize: sizeLimit + 1, files: 1, fields: 1, fieldSize: 1024 },
    fileFilter: (_req, file, done) => {
      if (soundFontFormat(file.originalname)) done(null, true);
      else done(new SoundFontLibraryError(400, 'Choose an .sf2 or .sf3 SoundFont.'));
    },
  });

  const canWrite: RequestHandler = (req, res, next) => {
    if (!authorized(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
    next();
  };

  router.get('/', (_req, res) => {
    res.set('Cache-Control', 'no-store').json(library.list());
  });
  router.post('/', canWrite, upload.single('file'), async (req, res) => {
    const file = req.file;
    if (!file) throw new SoundFontLibraryError(400, 'Choose an .sf2 or .sf3 SoundFont.');
    try {
      if (file.size > sizeLimit) throw new SoundFontLibraryError(413, 'The SoundFont is too large.');
      const { entry, created } = await library.add(file.path, file.originalname, req.body?.name, file.size);
      res.status(created ? 201 : 200).json(entry);
    } finally {
      await fs.rm(file.path, { force: true }).catch(error => console.warn('[SoundFontLibrary] Could not clean an upload', error));
    }
  });
  router.get('/:id', (req, res, next) => {
    const { path: bankPath } = library.file(req.params.id);
    res.set('Cache-Control', 'no-store').type('application/octet-stream');
    res.sendFile(path.basename(bankPath), { root: path.dirname(bankPath), cacheControl: false }, error => { if (error) next(error); });
  });
  router.patch('/:id', canWrite, (req: Request<{ id: string }>, res) => {
    res.json(library.rename(req.params.id, req.body?.name));
  });
  router.delete('/:id', canWrite, (req: Request<{ id: string }>, res) => {
    library.remove(req.params.id);
    res.status(204).end();
  });
  return router;
}

export default createSoundFontLibraryRouter();
