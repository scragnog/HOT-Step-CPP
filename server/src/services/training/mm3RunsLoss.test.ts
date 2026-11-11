// Checkpoint loss in a run summary is a 5-pass trailing mean, and a resume
// drops the passes the interrupted launch logged past its saved state.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { readMm3Run } from './mm3Runs.js';

test('checkpoint loss is the trailing 5-pass mean and survives a resume', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mm3-loss-'));
  try {
    for (const step of [16, 24]) {
      fs.mkdirSync(path.join(dir, `ckpt-${step}`));
      fs.writeFileSync(path.join(dir, `ckpt-${step}`, 'adapter_model.safetensors'), '');
    }
    const ev = (o: object) => JSON.stringify(o);
    const lines = [
      ...[1, 2, 3].map(e => ev({ type: 'epoch', epoch: e, loss: e, step: e * 4 })),
      // The interrupted launch logged passes 4 and 5, then died; the resume
      // restarts from pass 3 and replays only pass 4 before the checkpoint.
      ev({ type: 'epoch', epoch: 4, loss: 1000, step: 16 }),
      ev({ type: 'epoch', epoch: 5, loss: 1000, step: 20 }),
      ev({ type: 'resumed', step: 12, epoch: 3 }),
      ev({ type: 'epoch', epoch: 4, loss: 4, step: 16 }),
      ev({ type: 'milestone', step: 16, loss: 99 }),
      ev({ type: 'epoch', epoch: 5, loss: 5, step: 20 }),
      ev({ type: 'epoch', epoch: 6, loss: 6, step: 24 }),
      ev({ type: 'milestone', step: 24, loss: 99 }),
    ];
    fs.writeFileSync(path.join(dir, 'train-log.jsonl'), lines.join('\n') + '\n');
    const loss = Object.fromEntries(readMm3Run(dir)!.checkpoints.map(c => [c.step, c.loss]));
    assert.equal(loss[16], 2.5);      // mean of passes 1-4
    assert.equal(loss[24], 4);        // mean of passes 2-6, replayed pass 5 = 5
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
