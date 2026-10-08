import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const SERVER_ROOT = fileURLToPath(new URL('../../../', import.meta.url));

// updated_at is the revision accepted operations are checked against
// (operations.ts datasetRevision), so a read-only dataset view that re-syncs
// unchanged counters or caches a detected album must not move it. Runs
// against a scratch database.
test('a read-only re-sync (same counters, newly cached album) keeps the dataset revision; a real change moves it', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'datasets-repo-'));
  const script = [
    "import assert from 'node:assert/strict';",
    "import { initDb } from './src/db/database.js';",
    "import * as repo from './src/services/training/datasetsRepo.js';",
    "import { syncCounters } from './src/services/training/datasetDetail.js';",
    'initDb();',
    "const at = '2026-01-01T00:00:00.000Z';",
    "repo.insertDataset({ id: 'd1', slug: 'd1', name: 'd1', sourceDir: 'x', recursive: false, customTag: '', tagPosition: 'prepend',",
    "  genreRatio: 0, defaultArtist: '', defaultAlbum: '', defaultGenre: '', defaultLanguage: 'en', sampleCount: 2, labeledCount: 1,",
    "  excludedCount: 0, status: 'labeled', builtAt: '', datasetJsonPath: '', albumName: '', createdAt: at, updatedAt: at } as any);",
    "const sample = (caption: string) => ({ caption, excluded: false, fileMissing: false, tagAlbum: 'Album A' }) as any;",
    "const ds = () => repo.getDataset('d1')!;",
    // A read-only view: same counters, album detected for the first time.
    "assert.equal(syncCounters(ds(), [sample('x'), sample('')]), at);",
    "assert.equal(ds().updatedAt, at);",
    "assert.equal(ds().albumName, 'Album A');",
    // A real change moves the revision, and the caller gets the new one.
    "const next = syncCounters(ds(), [sample('x'), sample('y')]);",
    "assert.notEqual(next, at);",
    "assert.equal(ds().updatedAt, next);",
    "assert.equal(ds().labeledCount, 2);",
  ].join(String.fromCharCode(10));
  try {
    execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], { cwd: SERVER_ROOT, stdio: 'pipe',
      env: { ...process.env, DATA_DIR: path.join(root, 'data'), TRAINING_DIR: path.join(root, 'training') } });
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});
