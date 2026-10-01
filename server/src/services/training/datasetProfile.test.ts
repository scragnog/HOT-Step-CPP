import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { calibrateYue2Length, listYue2TrainLogs } from './datasetProfile.js';

test('calibrateYue2Length keeps the preset at the reference album length', () => {
  const c = calibrateYue2Length(45, 100, 10);
  assert.equal(c.steps, 100);
  assert.equal(c.saveEvery, 10);
});

test('calibrateYue2Length scales steps and saves together, keeping the rung count', () => {
  const long = calibrateYue2Length(53, 100, 10);
  assert.deepEqual([long.steps, long.saveEvery], [120, 12]);
  const short = calibrateYue2Length(32, 100, 10);
  assert.deepEqual([short.steps, short.saveEvery], [70, 7]);
});

test('calibrateYue2Length clamps the factor to 0.6..2', () => {
  assert.equal(calibrateYue2Length(5, 100, 10).factor, 0.6);
  assert.equal(calibrateYue2Length(500, 100, 10).factor, 2);
  assert.equal(calibrateYue2Length(500, 100, 10).steps, 200);
});

test('listYue2TrainLogs enumerates a root log and every segment log that exists', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-train-logs-'));
  try {
    fs.writeFileSync(path.join(root, 'train.jsonl'), 'root');
    const seg1 = path.join(root, 'segments', 'segment-000001');
    const seg2 = path.join(root, 'segments', 'segment-000002');
    fs.mkdirSync(seg1, { recursive: true });
    fs.mkdirSync(seg2, { recursive: true }); // still training: no train.jsonl yet, must not appear
    fs.writeFileSync(path.join(seg1, 'train.jsonl'), 'one');
    const logs = listYue2TrainLogs(root);
    assert.deepEqual(logs.map(l => l.seg), [path.basename(root), 'segment-000001']);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

const TRAINING_SRC_ROOT = fileURLToPath(new URL('../../../', import.meta.url));
function runInIsolatedTrainingDir(trainingDir: string, script: string): void {
  execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
    cwd: TRAINING_SRC_ROOT, env: { ...process.env, TRAINING_DIR: trainingDir }, stdio: 'pipe',
  });
}

test('archiveYue2TrainLogs throws when its destination cannot be created, leaving the source log in place', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-archive-fail-'));
  try {
    const script = [
      "import fs from 'node:fs';",
      "import path from 'node:path';",
      "import { archiveYue2TrainLogs } from './src/services/training/datasetProfile.js';",
      "const output = path.join(process.env.TRAINING_DIR, 'run'); fs.mkdirSync(output, { recursive: true });",
      "fs.writeFileSync(path.join(output, 'train.jsonl'), 'x');",
      // A plain file where the dataset's folder should be makes mkdirSync(trainLogArchiveDir) fail.
      "const datasets = path.join(process.env.TRAINING_DIR, 'datasets'); fs.mkdirSync(datasets, { recursive: true });",
      "fs.writeFileSync(path.join(datasets, 'blocked'), 'x');",
      "let threw = false; try { archiveYue2TrainLogs('blocked', 'job1', output); } catch { threw = true; }",
      "if (!threw) throw new Error('expected archiveYue2TrainLogs to throw');",
      "if (!fs.existsSync(path.join(output, 'train.jsonl'))) throw new Error('source log missing after a failed archive');",
    ].join('');
    runInIsolatedTrainingDir(root, script);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('noteYue2TrainLog shallow-merges: a patch without keptStep keeps the one already stored', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-note-'));
  try {
    const script = [
      "import fs from 'node:fs';",
      "import path from 'node:path';",
      "import { noteYue2TrainLog, trainLogArchiveDir } from './src/services/training/datasetProfile.js';",
      "noteYue2TrainLog('album', 'job1', { keptStep: 120, status: 'done' });",
      "noteYue2TrainLog('album', 'job1', { status: 'finished' });",
      "const saved = JSON.parse(fs.readFileSync(path.join(trainLogArchiveDir('album'), 'job1.json'), 'utf8'));",
      "if (saved.keptStep !== 120) throw new Error('keptStep was dropped: ' + saved.keptStep);",
      "if (saved.status !== 'finished') throw new Error('status was not updated: ' + saved.status);",
    ].join('');
    runInIsolatedTrainingDir(root, script);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('archiveYue2TrainLogs propagates a non-ENOENT segment-directory enumeration failure instead of treating it as no logs', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-archive-enoent-'));
  try {
    const script = [
      "import fs from 'node:fs';",
      "import path from 'node:path';",
      "import { archiveYue2TrainLogs } from './src/services/training/datasetProfile.js';",
      "const output = path.join(process.env.TRAINING_DIR, 'run');",
      "fs.mkdirSync(path.join(output, 'segments', 'segment-000001'), { recursive: true });",
      "fs.writeFileSync(path.join(output, 'segments', 'segment-000001', 'train.jsonl'), 'x');",
      "const real = fs.readdirSync;",
      "fs.readdirSync = (p, ...rest) => { if (String(p) === path.join(output, 'segments')) { const e = new Error('simulated'); e.code = 'EIO'; throw e; } return real(p, ...rest); };",
      "let threw = false; try { archiveYue2TrainLogs('album', 'job1', output); } catch (err) { threw = err?.code === 'EIO'; } finally { fs.readdirSync = real; }",
      "if (!threw) throw new Error('expected archiveYue2TrainLogs to propagate the EIO, not swallow it as no logs');",
    ].join('');
    runInIsolatedTrainingDir(root, script);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('noteYue2TrainLog propagates a non-ENOENT sidecar read failure and leaves the existing sidecar untouched', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-note-enoent-'));
  try {
    const script = [
      "import fs from 'node:fs';",
      "import path from 'node:path';",
      "import { noteYue2TrainLog, trainLogArchiveDir } from './src/services/training/datasetProfile.js';",
      "noteYue2TrainLog('album', 'job1', { keptStep: 120, status: 'done' });",
      "const file = path.join(trainLogArchiveDir('album'), 'job1.json');",
      "const real = fs.readFileSync;",
      "fs.readFileSync = (p, ...rest) => { if (p === file) { const e = new Error('simulated'); e.code = 'EIO'; throw e; } return real(p, ...rest); };",
      "let threw = false; try { noteYue2TrainLog('album', 'job1', { status: 'finished' }); } catch { threw = true; } finally { fs.readFileSync = real; }",
      "if (!threw) throw new Error('expected noteYue2TrainLog to propagate the EIO');",
      "const saved = JSON.parse(fs.readFileSync(file, 'utf8'));",
      "if (saved.keptStep !== 120 || saved.status !== 'done') throw new Error('sidecar was modified despite the propagated read failure: ' + JSON.stringify(saved));",
    ].join('');
    runInIsolatedTrainingDir(root, script);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
