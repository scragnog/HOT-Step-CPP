import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import Database from 'better-sqlite3';

test('blind labels survive checkpoint changes, scores capture rating mode, and cleanup records the pick', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-blind-rungs-'));
  process.env.DATA_DIR = root;
  process.env.TRAINING_DIR = path.join(root, 'training');
  const { initDb, getDb, closeDb } = await import('../../db/database.js');
  const { recordYue2AitkRun, listYue2AitkRuns, yue2RunFinished } = await import('./yue2AitkRuns.js');
  const { scoreYue2Rung, listYue2RungScores, yue2RungScoresCsv } = await import('./yue2RungScores.js');
  const { runYue2Cleanup } = await import('./yue2Cleanup.js');
  const out = path.join(root, 'yue2-joint-adapters', 'run');
  const add = (step: number) => {
    const dir = path.join(out, `checkpoint-step${step}`);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'native-ar.safetensors'), 'x');
    fs.writeFileSync(path.join(dir, 'native-nar.safetensors'), 'x');
  };
  try {
    const legacyDb = new Database(path.join(root, 'hotstep.db'));
    legacyDb.exec(`CREATE TABLE yue2_rung_scores (
      id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id TEXT NOT NULL, dataset_slug TEXT NOT NULL,
      source_run TEXT NOT NULL DEFAULT '', refine_run TEXT NOT NULL, checkpoint_dir TEXT NOT NULL UNIQUE,
      step INTEGER NOT NULL, kl REAL, recon REAL, drift REAL, rung INTEGER NOT NULL DEFAULT 0,
      frozen INTEGER NOT NULL DEFAULT 0, settings TEXT NOT NULL DEFAULT '{}', previews TEXT NOT NULL DEFAULT '[]',
      metrics TEXT NOT NULL DEFAULT '{}', likeness INTEGER, corruption INTEGER, notes TEXT NOT NULL DEFAULT '',
      created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')));
      INSERT INTO yue2_rung_scores (dataset_id, dataset_slug, refine_run, checkpoint_dir, step)
      VALUES ('legacy-ds', 'legacy', 'legacy-run', 'legacy-checkpoint', 5);`);
    legacyDb.close();
    initDb();
    assert.deepEqual(getDb().prepare('SELECT blind, blind_label FROM yue2_rung_scores WHERE dataset_id = ?').get('legacy-ds'),
      { blind: 0, blind_label: '' });
    for (const step of [10, 20, 30]) add(step);
    recordYue2AitkRun({ version: 1, jobId: 'blind-job', datasetId: 'blind-ds', datasetSlug: 'blind-slug', method: 'aitk',
      output: out, options: { method: 'base-matched' }, status: 'done', createdAt: 1, updatedAt: 2, checkpoints: [] });
    const first = listYue2AitkRuns('blind-ds')[0].blindLabels!;
    assert.deepEqual(Object.values(first).sort(), ['A', 'B', 'C']);
    assert.notDeepEqual([10, 20, 30].map(step => first[step]), ['A', 'B', 'C']);
    assert.deepEqual(listYue2AitkRuns('blind-ds')[0].blindLabels, first);
    fs.rmSync(path.join(out, 'checkpoint-step20'), { recursive: true });
    add(40);
    const labels = listYue2AitkRuns('blind-ds')[0].blindLabels!;
    assert.equal(labels[20], first[20]);
    assert.equal(labels[40], 'D');
    const pickedStep = [10, 30, 40].find(step => labels[step] !== 'A')!;
    const pickedLabel = labels[pickedStep];
    const ds = { id: 'blind-ds', slug: 'blind-slug', sourceDir: path.join(root, 'source') };
    fs.mkdirSync(ds.sourceDir);
    const scored = scoreYue2Rung(ds, { refineRun: 'blind-job', step: pickedStep, likeness: 4, corruption: 2, blind: true, blindLabel: pickedLabel });
    assert.equal(scored.blind, true);
    assert.equal(scored.blindLabel, pickedLabel);
    const notes = scoreYue2Rung(ds, { refineRun: 'blind-job', step: pickedStep, notes: 'second listen' });
    assert.equal(notes.blind, true);
    assert.equal(notes.blindLabel, pickedLabel);
    const plainStep = [10, 30, 40].find(step => step !== pickedStep)!;
    const plain = scoreYue2Rung(ds, { refineRun: 'blind-job', step: plainStep, likeness: 3, blind: false });
    assert.equal(plain.blind, false);
    assert.equal(plain.blindLabel, '');
    assert.equal(listYue2RungScores(ds.id).length, 2);
    assert.match(yue2RungScoresCsv([notes]), /blind,blind_label/);
    assert.equal(JSON.parse(JSON.stringify(notes)).blindLabel, pickedLabel);
    const result = runYue2Cleanup(ds, 'blind-job', pickedStep,
      { caches: false, otherCheckpoints: false, otherRuns: false, resume: false, otherPreviews: false },
      { blind: true, blindLabel: pickedLabel });
    assert.equal(result.finishError, undefined);
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(out, 'finished'), 'utf8')),
      { pickedStep, pickedBlind: true, pickedLabel, at: JSON.parse(fs.readFileSync(path.join(out, 'finished'), 'utf8')).at });
    assert.equal(yue2RunFinished(out), true);
    fs.writeFileSync(path.join(out, 'finished'), '');
    assert.equal(yue2RunFinished(out), true);
    const write = fs.writeFileSync;
    fs.writeFileSync = ((file: fs.PathOrFileDescriptor, data: string | NodeJS.ArrayBufferView, options?: fs.WriteFileOptions) => {
      if (file === path.join(out, 'finished')) throw new Error('marker disk failure');
      return write(file, data, options);
    }) as typeof fs.writeFileSync;
    try {
      const failed = runYue2Cleanup(ds, 'blind-job', pickedStep,
        { caches: false, otherCheckpoints: false, otherRuns: false, resume: false, otherPreviews: false },
        { blind: true, blindLabel: pickedLabel });
      assert.match(failed.finishError ?? '', /marker disk failure/);
    } finally { fs.writeFileSync = write; }
    const old = getDb().prepare('SELECT blind, blind_label FROM yue2_rung_scores WHERE step = ?').get(plainStep) as { blind: number; blind_label: string };
    assert.deepEqual(old, { blind: 0, blind_label: '' });
  } finally {
    closeDb();
    fs.rmSync(root, { recursive: true, force: true });
  }
});
