// assets/audioAssets.ts — identity and ownership for uploaded audio.
//
// /api/upload/audio writes the file under data/references/<uuid>.<ext> and
// records it here. A workflow captures the asset id, never a client-supplied
// path, and resolves it when it runs: the asset must belong to the user and
// its file must still be there. An upload is never overwritten, so a new
// source is a new asset id and `sha256` identifies the bytes.

import { randomUUID } from 'node:crypto';
import path from 'node:path';
import fs from 'node:fs';
import type Database from 'better-sqlite3';
import { WorkflowError } from '../workflows/workflowJobs.js';

export interface AudioAsset {
  id: string;
  /** Served path, e.g. /references/<uuid>.mp3 (what generation requests take). */
  url: string;
  filename: string;
  size: number;
  sha256: string;
  createdAt: number;
}

interface Row { id: string; user_id: string | null; url: string; filename: string; size: number; sha256: string; created_at: number }

export function ensureAudioAssetSchema(db: Database.Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS audio_assets (
      id TEXT PRIMARY KEY,
      user_id TEXT,
      url TEXT NOT NULL,
      filename TEXT NOT NULL,
      size INTEGER NOT NULL,
      sha256 TEXT NOT NULL,
      created_at INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS audio_assets_owner ON audio_assets(user_id, created_at);
  `);
}

const toAsset = (r: Row): AudioAsset => ({ id: r.id, url: r.url, filename: r.filename, size: r.size, sha256: r.sha256, createdAt: r.created_at });

/** Record an upload. `userId` is null for an upload made without a token;
 *  such an asset exists but resolves for nobody. */
export function recordAudioAsset(db: Database.Database, a: { userId: string | null; url: string; filename: string; size: number; sha256: string }, now = Date.now()): AudioAsset {
  ensureAudioAssetSchema(db);
  const id = randomUUID();
  db.prepare('INSERT INTO audio_assets (id, user_id, url, filename, size, sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)')
    .run(id, a.userId, a.url, a.filename, a.size, a.sha256, now);
  return toAsset({ id, user_id: a.userId, url: a.url, filename: a.filename, size: a.size, sha256: a.sha256, created_at: now });
}

/** The user's asset with its file on disk, or a 404 (unknown, someone
 *  else's, or its file is gone). `dataDir` holds references/. */
export function resolveAudioAsset(db: Database.Database, id: string, userId: string, dataDir: string): AudioAsset & { path: string } {
  ensureAudioAssetSchema(db);
  const r = db.prepare('SELECT * FROM audio_assets WHERE id = ? AND user_id = ?').get(id, userId) as Row | undefined;
  if (!r) throw new WorkflowError(404, `Audio asset ${id} not found`);
  const file = path.join(dataDir, 'references', path.basename(r.url));
  if (!fs.existsSync(file)) throw new WorkflowError(404, `Audio asset ${id} has no file any more`);
  return { ...toAsset(r), path: file };
}
