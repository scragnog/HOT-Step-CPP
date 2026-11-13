import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { mm3BatchTrainBody, mm3CodesComplete, pickedCheckpointError, skipNarFurther } from './yue2BatchRunner.js';
import type { Yue2AitkRunRecord } from './yue2AitkRuns.js';

function run(over: Partial<Yue2AitkRunRecord>): Pick<Yue2AitkRunRecord, 'options'> {
  return { options: {}, ...over };
}

test('a local run with no decoder freeze gets the NAR follow-up', () => {
  assert.equal(skipNarFurther(run({})), false);
});

test('a decoder-only follow-up never gets a second one', () => {
  assert.equal(skipNarFurther(run({ options: { freezePlannerNow: true } })), true);
});

test('undefined (no run found) skips rather than throwing', () => {
  assert.equal(skipNarFurther(undefined), false);
});

// A remote-origin run is rejected before skipNarFurther is even consulted —
// see stageRequest's 'nar' branch, which throws directly (Reviewer/Lead,
// round 3 #4: no silent substitute). Not unit-testable here since
// stageRequest is not exported; RefinePanel.tsx's equivalent decision
// (narOnUseOutcome) is directly tested in RefinePanel.skipNarFurther.test.ts.

test('a captured pick finishes only from a complete checkpoint of that run', () => {
  const runs = [
    { jobId: 'ladder', checkpoints: [
      { step: 40, dir: 'd40', arPath: 'a40', narPath: 'n40' },
      { step: 60, dir: 'd60', arPath: 'a60' },
    ] },
    { jobId: 'other', checkpoints: [{ step: 40, dir: 'o40', arPath: 'oa', narPath: 'on' }] },
  ] as Parameters<typeof pickedCheckpointError>[0];
  assert.equal(pickedCheckpointError(runs, 'ladder', 40), null);
  assert.match(pickedCheckpointError(runs, 'ladder', 60)!, /No complete checkpoint at step 60/);
  assert.match(pickedCheckpointError(runs, 'ladder', 80)!, /No checkpoint at step 80/);
  assert.match(pickedCheckpointError(runs, 'gone', 40)!, /not one of this dataset's runs/);
});

test('MM3 codes are complete only when every current manifest id has its file', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mm3-codes-'));
  try {
    const manifest = path.join(dir, 'dataset.json');
    const inner = path.join(dir, 'codes');
    fs.mkdirSync(inner);
    fs.writeFileSync(manifest, JSON.stringify({ samples: [{ id: 'kept' }, { id: 'added' }] }));
    for (const id of ['kept', 'removed']) fs.writeFileSync(path.join(inner, `${id}.codes`), 'x');
    assert.equal(mm3CodesComplete(manifest, inner), false);   // same count as the manifest, wrong ids
    fs.writeFileSync(path.join(inner, 'added.codes'), 'x');
    assert.equal(mm3CodesComplete(manifest, inner), true);
    assert.equal(mm3CodesComplete(path.join(dir, 'missing.json'), inner), false);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('an MM3 batch never sends the first dataset trigger to every run', () => {
  const recipe = { steps: 900, trigger: 'first_album', preview: { seconds: 40 } };
  assert.deepEqual(mm3BatchTrainBody(recipe), { steps: 900, preview: { seconds: 40 } });
  assert.equal(recipe.trigger, 'first_album');   // the stored batch recipe is left alone
});
