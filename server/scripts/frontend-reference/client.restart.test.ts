// client.restart.test.ts — restart interruption over HTTP. One harness
// process starts a training job and a workflow job that are still running
// when it is killed; a second process on the same data dir reads them back.
// The only process this test spawns is the harness's own entry point
// (restartChild.ts), run by this same node binary; each child installs the
// safety guards before its own imports and reports its violations.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CHILD = path.join(HERE, 'restartChild.ts');

function harness(args: string[]) {
  const child = spawn(process.execPath, ['--import', 'tsx', CHILD, ...args], { cwd: path.join(HERE, '..', '..'), stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  child.stderr.on('data', d => { stderr += d; });
  const firstLine = new Promise<Record<string, any>>((resolve, reject) => {
    let out = '';
    child.stdout.on('data', d => {
      out += d;
      const line = out.split('\n').find(l => l.startsWith('{'));
      if (line) resolve(JSON.parse(line));
    });
    child.on('exit', code => reject(new Error(`harness ${args[0]} exited ${code} before reporting:\n${stderr.slice(-2000)}`)));
  });
  return { child, firstLine };
}

test('a restart reconciles running work: workflow job and pipeline interrupted, training job failed as restarted', { timeout: 120_000 }, async () => {
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'restart-data-'));
  const sourceDir = fs.mkdtempSync(path.join(os.tmpdir(), 'restart-source-'));
  const children: ReturnType<typeof harness>['child'][] = [];
  try {
    const first = harness(['start', dataDir, sourceDir]);
    children.push(first.child);
    const started = await first.firstLine;
    assert.deepEqual(started.violations, [], 'first life: no guard violations');
    first.child.kill('SIGKILL');
    await new Promise(resolve => first.child.once('exit', resolve));

    const stateFile = path.join(dataDir, 'restart-state.json');
    fs.writeFileSync(stateFile, JSON.stringify(started));
    const second = harness(['check', dataDir, stateFile]);
    children.push(second.child);
    const after = await second.firstLine;
    await new Promise(resolve => second.child.once('exit', resolve));
    assert.deepEqual(after.violations, [], 'second life: no guard violations');

    // Workflow jobs are durable: a job caught running is interrupted, never rerun.
    assert.equal(after.workflow.status, 'interrupted');
    // Preparation pipelines are durable too: interrupted, its running stage with it.
    assert.equal(after.pipeline.status, 'interrupted');
    assert.match(after.pipeline.error, /Server restarted during preparation/);
    assert.deepEqual(after.pipeline.stages.map((s: { stage: string; status: string }) => [s.stage, s.status]), [['latents', 'interrupted']]);
    // Training jobs are kept in memory, with their record on disk from
    // creation: the startup reconcile marks one caught running as failed.
    // Nothing is resumed.
    assert.equal(after.training.status, 200);
    assert.deepEqual([after.training.body.status, after.training.body.error], ['failed', 'Server restarted']);
    assert.deepEqual((after.listed as Array<{ id: string; status: string }>).map(j => [j.id, j.status]), [[started.trainingJobId, 'failed']]);
  } finally {
    // A failed assertion must not leave a harness process running.
    for (const child of children) {
      if (child.exitCode === null && child.signalCode === null) {
        child.kill('SIGKILL');
        await new Promise(resolve => child.once('exit', resolve));
      }
    }
    fs.rmSync(dataDir, { recursive: true, force: true });
    fs.rmSync(sourceDir, { recursive: true, force: true });
    fs.rmSync(`${sourceDir}-prep`, { recursive: true, force: true });
  }
});
