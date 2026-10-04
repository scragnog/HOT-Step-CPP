#!/usr/bin/env npx tsx
/**
 * Move YuE2 ladders pulled from a training worker out of their old
 * `remote-<worker>-<uuid>` folders (2026-10-04). Finished runs get the name a
 * local run gets, `<trigger>_<YYYY-MM-DD_HH-MM-SS>` from the worker's start
 * time; unfinished ones go to `_remote/<worker>/` under that name. The run
 * index, yue2-linked.json and album presets follow each move. Nothing is
 * deleted. Run from server/ with the app's training queue idle:
 *
 *   npx tsx scripts/migrate-remote-ladder-folders.ts           (dry run)
 *   npx tsx scripts/migrate-remote-ladder-folders.ts --apply
 */
import { initDb } from '../src/db/database.js';
import { migrateYue2RemoteFolders } from '../src/services/training/yue2Cleanup.js';

initDb();
const apply = process.argv.includes('--apply');
const rows = migrateYue2RemoteFolders(apply);
for (const r of rows) {
  console.log(`${r.error ? 'FAILED ' : ''}${r.finished ? 'finished' : 'staged  '}  ${r.from}\n          -> ${r.to}${r.error ? `\n          ${r.error}` : ''}`);
}
const failed = rows.filter(r => r.error).length;
console.log(`${rows.length} run(s) ${apply ? 'processed' : 'would move'}${apply ? `, ${failed} failed` : ' (dry run; --apply to move)'}`);
if (failed) process.exitCode = 1;
