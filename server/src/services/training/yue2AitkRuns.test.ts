import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import test from 'node:test';
import { checkpointRecords } from './yue2AitkRuns.js';

test('AITK checkpoint discovery exposes combined and native AR/NAR outputs', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-aitk-runs-'));
  try {
    const step2 = path.join(root, 'checkpoint-step2');
    const step10 = path.join(root, 'checkpoint-step10');
    fs.mkdirSync(step2); fs.mkdirSync(step10);
    for (const name of ['adapter.safetensors', 'optimizer.resume', 'native-ar.safetensors', 'native-nar.safetensors']) {
      fs.writeFileSync(path.join(step10, name), 'x');
    }
    fs.writeFileSync(path.join(step2, 'native-ar.safetensors'), 'x');
    const rows = checkpointRecords(root);
    assert.deepEqual(rows.map(r => r.step), [10, 2]);
    assert.equal(rows[0].arPath, path.join(step10, 'native-ar.safetensors'));
    assert.equal(rows[0].narPath, path.join(step10, 'native-nar.safetensors'));
    assert.equal(rows[1].adapterPath, undefined);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('AITK run catalogue writes and rereads atomically in an isolated training root', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-aitk-index-'));
  try {
    const script = [
      "import fs from 'node:fs';",
      "import path from 'node:path';",
      "import { recordYue2AitkRun, listYue2AitkRuns, listAllYue2AitkRuns, aitkRunIndexPath } from './src/services/training/yue2AitkRuns.js';",
      "const out = path.join(process.env.TRAINING_DIR, 'run'); fs.mkdirSync(path.join(out, 'checkpoint-step4'), { recursive: true });",
      "fs.writeFileSync(path.join(out, 'checkpoint-step4', 'native-ar.safetensors'), 'x');",
      "recordYue2AitkRun({ version: 1, jobId: 'job-test', datasetId: 'ds-test', datasetSlug: 'slug-test', method: 'aitk', output: out, options: {}, status: 'running', createdAt: 1, updatedAt: 2, checkpoints: [] });",
      "const rows = listYue2AitkRuns('ds-test'); if (rows.length !== 1 || rows[0].checkpoints[0].arPath === undefined) throw new Error('catalogue reread failed');",
      "const all = listAllYue2AitkRuns(); if (all.length !== 1 || all[0].jobId !== 'job-test') throw new Error('all-runs catalogue failed');",
      "if (!fs.existsSync(aitkRunIndexPath())) throw new Error('catalogue missing');",
    ].join('');
    execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
      cwd: fileURLToPath(new URL('../../../', import.meta.url)), env: { ...process.env, TRAINING_DIR: root }, stdio: 'pipe',
    });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
