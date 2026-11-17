/**
 * Last-resort error handler for /api. Without it an error passed to next()
 * (a multer limit, a body-parser failure) gets Express's default HTML page,
 * and the UI can only say "The server ran into a problem".
 */
import type { NextFunction, Request, Response } from 'express';
import multer from 'multer';

export function apiErrorHandler(err: any, _req: Request, res: Response, next: NextFunction): void {
  if (res.headersSent) { next(err); return; }
  if (err instanceof multer.MulterError) {
    const tooLarge = err.code === 'LIMIT_FILE_SIZE';
    res.status(tooLarge ? 413 : 400).json({ error: tooLarge ? 'The file is too large to upload' : err.message });
    return;
  }
  const status = Number.isInteger(err?.status) && err.status >= 400 && err.status < 500 ? err.status : 500;
  if (status === 500) console.error('[api] Unhandled error:', err);
  res.status(status).json({ error: String(err?.message || 'The server ran into a problem') });
}
