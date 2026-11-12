// backup.ts — a verified copy of hotstep.db, safe while the app runs.
//
// Copying the .db file while the app runs can miss whatever still sits in
// the -wal file, or catch a page mid-write. SQLite's online backup API copies
// a consistent snapshot through a connection instead, whoever else is
// writing (dev and prod may share this database, services/instance.ts). The
// copy is then checked with integrity_check and compared table by table
// against the source's schema; a copy that fails is deleted, never kept.
//
//   cd server; npx tsx src/db/backup.ts [destination.db]
//
// Default destination: <data dir>/backups/hotstep-<timestamp>.db. An existing
// file is never overwritten.

import { removeSync } from '../services/fsRemove.js';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import Database from 'better-sqlite3';
import { BUSY_TIMEOUT_MS } from './database.js';

export interface BackupResult { destination: string; pages: number; tables: number; bytes: number }

const tablesOf = (db: Database.Database): string[] =>
  (db.prepare(`SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name`).all() as Array<{ name: string }>).map(r => r.name);

export async function backupDatabase(source: string, destination: string): Promise<BackupResult> {
  if (fs.existsSync(destination)) throw new Error(`Refusing to overwrite ${destination}`);
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  const src = new Database(source, { readonly: true, fileMustExist: true });
  try {
    src.pragma(`busy_timeout = ${BUSY_TIMEOUT_MS}`);
    const { totalPages } = await src.backup(destination);
    const copy = new Database(destination, { readonly: true });
    try {
      const integrity = copy.pragma('integrity_check') as Array<{ integrity_check: string }>;
      if (integrity.length !== 1 || integrity[0].integrity_check !== 'ok') {
        throw new Error(`integrity_check failed: ${integrity.map(r => r.integrity_check).join('; ')}`);
      }
      const want = tablesOf(src).join(',');
      const got = tablesOf(copy).join(',');
      if (want !== got) throw new Error(`table list differs: source [${want}], copy [${got}]`);
      return { destination, pages: totalPages, tables: tablesOf(copy).length, bytes: fs.statSync(destination).size };
    } finally {
      copy.close();
    }
  } catch (err) {
    removeSync(destination, { force: true });
    throw err;
  } finally {
    src.close();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const { config } = await import('../config.js');
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const destination = path.resolve(process.argv[2] ?? path.join(config.data.dir, 'backups', `hotstep-${stamp}.db`));
  try {
    const r = await backupDatabase(config.data.dbPath, destination);
    console.log(`[DB] Backup verified: ${r.destination} (${r.tables} tables, ${r.pages} pages, ${r.bytes} bytes, integrity ok)`);
  } catch (err) {
    console.error(`[DB] Backup failed: ${(err as Error).message}`);
    process.exitCode = 1;
  }
}
