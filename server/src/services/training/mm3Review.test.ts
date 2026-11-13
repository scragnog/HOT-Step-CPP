// mm3Review.test.ts — the MM3 ladder review: blind letters kept in the run's
// manifest, rung and album scores round-tripping, and cleanup that keeps the
// chosen step, the base takes and the MM3 codes while it stays in the run.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

test('blind letters stick, scores round-trip, cleanup keeps the chosen step and prunes the rest', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mm3-review-'));
  process.env.DATA_DIR = root;
  process.env.TRAINING_DIR = path.join(root, 'training');
  const { initDb, closeDb } = await import('../../db/database.js');
  const { datasetDir } = await import('./paths.js');
  const m = await import('./mm3Review.js');
  const runDir = path.join(root, 'runs', 'albumx-2026-11-07_13-11-56');
  const other = path.join(root, 'runs', 'albumx-2026-11-01_00-00-00');
  const write = (f: string, body = 'x') => { fs.mkdirSync(path.dirname(f), { recursive: true }); fs.writeFileSync(f, body); };
  const ckpt = (step: number) => ({ step, name: `ckpt-${step}`, dir: path.join(runDir, `ckpt-${step}`), loss: step / 100 });
  try {
    initDb();
    // Cleanup deletes caches: never let it reach a real data folder.
    assert.ok(path.resolve(datasetDir('albumx')).startsWith(path.resolve(root)), datasetDir('albumx'));
    for (const s of [100, 200, 300]) write(path.join(runDir, `ckpt-${s}`, 'adapter_model.safetensors'), 'w'.repeat(s));
    write(path.join(other, 'ckpt-100', 'adapter_model.safetensors'));
    write(path.join(runDir, 'hotstep-run.json'), JSON.stringify({ version: 1, runName: path.basename(runDir), options: {} }));
    write(path.join(runDir, 'resume-state.bin'), 'r'.repeat(50));
    const prev = (f: string) => path.join(runDir, 'previews', f);
    for (const f of ['base-artist.wav', 'step-000100-artist.wav', 'step-000200-artist.wav', 'step-000200-artist-s5.wav', 'step-000300-artist.wav']) write(prev(f), 'p'.repeat(10));
    write(datasetDir('albumx') + '/mm3-codes/a.bin', 'c');
    write(datasetDir('albumx') + '/yue2-stems/a.wav', 's'.repeat(7));
    const ds = { id: 'ds1', slug: 'albumx', sourceDir: path.join(root, 'src') };
    let run = { runName: path.basename(runDir), dir: runDir, checkpoints: [ckpt(100), ckpt(200)] };

    // Letters are given once, saved, and a new checkpoint never moves them.
    const first = m.mm3BlindLabels(run);
    assert.deepEqual(Object.values(first).sort(), ['A', 'B']);
    run = { ...run, checkpoints: [ckpt(100), ckpt(200), ckpt(300)] };
    const second = m.mm3BlindLabels(run);
    assert.equal(second[100], first[100]); assert.equal(second[200], first[200]); assert.equal(second[300], 'C');
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(runDir, 'hotstep-run.json'), 'utf-8')).blindLabels, second);

    // Scores: a notes edit keeps the rating; a blind rating must name its own letter.
    m.scoreMm3Rung(ds, run, { step: 200, likeness: 4, corruption: 2, blind: true, blindLabel: second[200] });
    m.scoreMm3Rung(ds, run, { step: 200, notes: 'chorus drifts at 1:10' });
    assert.throws(() => m.scoreMm3Rung(ds, run, { step: 100, likeness: 3, blind: true, blindLabel: second[200] }), /Blind label/);
    assert.throws(() => m.scoreMm3Rung(ds, run, { step: 150, likeness: 3 }), /No checkpoint/);
    const [row] = m.listMm3RungScores('ds1', run.runName);
    assert.deepEqual([row.step, row.likeness, row.corruption, row.notes, row.blind, row.blindLabel], [200, 4, 2, 'chorus drifts at 1:10', true, second[200]]);
    m.scoreMm3Album(ds, run.runName, { score: 4, vocals: 'over' });
    m.scoreMm3Album(ds, run.runName, { notes: 'voice arrived' });
    assert.deepEqual((({ score, instruments, vocals, notes }) => ({ score, instruments, vocals, notes }))(m.getMm3AlbumScore(run.runName)!),
      { score: 4, instruments: null, vocals: 'over', notes: 'voice arrived' });

    // The plan: the other two checkpoints, their three takes (the extra take
    // included, the base kept), the resume state, and every cache but the codes.
    const plan = m.planMm3Cleanup(ds, run, 200);
    assert.deepEqual([plan.otherCheckpoints.count, plan.otherCheckpoints.bytes], [2, 400]);
    assert.deepEqual([plan.otherPreviews.count, plan.resume.count, plan.resume.bytes], [2, 1, 50]);
    assert.deepEqual(plan.caches.detail, ['yue2-stems']);
    assert.equal('otherRuns' in plan, false);

    const done = m.runMm3Cleanup(ds, run, 200, { otherCheckpoints: true, otherPreviews: true, resume: true, caches: true });
    assert.equal(done.freedBytes, plan.otherCheckpoints.bytes + plan.otherPreviews.bytes + plan.resume.bytes + plan.caches.bytes);
    assert.deepEqual(fs.readdirSync(runDir).sort(), ['ckpt-200', 'hotstep-run.json', 'previews']);
    assert.deepEqual(fs.readdirSync(path.join(runDir, 'previews')).filter(f => f.endsWith('.wav')).sort(),
      ['base-artist.wav', 'step-000200-artist-s5.wav', 'step-000200-artist.wav']);
    assert.deepEqual(m.prunedMm3Previews(path.join(runDir, 'previews')).map(p => p.step).sort(), [100, 300]);
    assert.equal(fs.existsSync(path.join(other, 'ckpt-100', 'adapter_model.safetensors')), true);
    assert.equal(fs.existsSync(datasetDir('albumx') + '/mm3-codes/a.bin'), true);
    assert.equal(fs.existsSync(datasetDir('albumx') + '/yue2-stems'), false);
    // Scores outlive the checkpoints they judged.
    assert.equal(m.listMm3RungScores('ds1', run.runName).length, 1);
  } finally {
    closeDb();
    fs.rmSync(root, { recursive: true, force: true });
  }
});

// A `previews` folder or a checkpoint that is a junction into another run must
// stop the plan before cleanup writes or deletes anything there.
test('cleanup refuses junctions out of the run, before any write', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mm3-review-link-'));
  process.env.DATA_DIR = root;
  process.env.TRAINING_DIR = path.join(root, 'training');
  const m = await import('./mm3Review.js');
  const runDir = path.join(root, 'runs', 'albumx-2026-11-07_13-11-56');
  const other = path.join(root, 'runs', 'albumx-2026-11-01_00-00-00');
  const write = (f: string) => { fs.mkdirSync(path.dirname(f), { recursive: true }); fs.writeFileSync(f, 'x'); };
  const ckpt = (step: number) => ({ step, name: `ckpt-${step}`, dir: path.join(runDir, `ckpt-${step}`) });
  const ds = { id: 'ds1', slug: 'albumx', sourceDir: path.join(root, 'src') };
  try {
    for (const s of [10, 20]) write(path.join(runDir, `ckpt-${s}`, 'adapter_model.safetensors'));
    write(path.join(other, 'previews', 'step-000010-artist.wav'));
    write(path.join(other, 'ckpt-30', 'adapter_model.safetensors'));
    fs.symlinkSync(path.join(other, 'previews'), path.join(runDir, 'previews'), 'junction');
    const run = { runName: path.basename(runDir), dir: runDir, checkpoints: [ckpt(10), ckpt(20)] };
    assert.throws(() => m.planMm3Cleanup(ds, run, 20), /outside the run folder/);
    assert.throws(() => m.runMm3Cleanup(ds, run, 20, { otherPreviews: true, otherCheckpoints: true }), /outside the run folder/);
    assert.deepEqual(fs.readdirSync(path.join(other, 'previews')), ['step-000010-artist.wav']);
    assert.equal(fs.existsSync(path.join(runDir, 'ckpt-10')), true);

    // A checkpoint folder linked to another run's checkpoint: refused too.
    fs.rmdirSync(path.join(runDir, 'previews')); // the junction only
    fs.symlinkSync(path.join(other, 'ckpt-30'), path.join(runDir, 'ckpt-30'), 'junction');
    assert.throws(() => m.runMm3Cleanup(ds, { ...run, checkpoints: [ckpt(10), ckpt(20), ckpt(30)] }, 20, { otherCheckpoints: true }), /outside the run folder/);
    assert.equal(fs.existsSync(path.join(other, 'ckpt-30', 'adapter_model.safetensors')), true);
    assert.equal(fs.existsSync(path.join(runDir, 'ckpt-10')), true);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
