import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import {
  ACE_DIT_FORM_DEFAULTS, ACE_LM_FORM_DEFAULTS, YUE2_JOINT_FORM_DEFAULTS,
  aceDitExecutionBody, aceLmExecutionBody,
} from '../../../contracts/trainingRecipes.js';
import { mm3ExecutionBody, resolveRecipe, yue2ArExecutionBody, yue2NarExecutionBody } from './operations.js';
import { TrainingOperationFailure } from '../operations.js';
import { resolveTrainingDefaultLayers } from '../trainingDefaults.js';

type Case = { family: 'ace-lm' | 'ace-dit'; form: Record<string, unknown>; variantKey: string; formerBody: Record<string, unknown> };
const cases = JSON.parse(readFileSync(new URL('./ace-body-golden.json', import.meta.url), 'utf8')) as Case[];

for (const [index, fixture] of cases.entries()) {
  test(`ACE recipe preserves former complete body ${index + 1} (${fixture.family})`, () => {
    const actual = fixture.family === 'ace-lm'
      ? aceLmExecutionBody(fixture.form as typeof ACE_LM_FORM_DEFAULTS, fixture.variantKey || undefined)
      : aceDitExecutionBody(fixture.form as typeof ACE_DIT_FORM_DEFAULTS, fixture.variantKey || undefined);
    assert.deepEqual(actual, fixture.formerBody);
  });
}

type OtherCase = { family: 'mm3-lm' | 'yue2-nar' | 'yue2-ar'; form: Record<string, unknown>; formerBody: Record<string, unknown>;
  worker: string; activePreset?: string; trainLaunder?: boolean; flashSupported?: boolean; overtrain?: boolean; mintedMissing?: boolean };
const otherCases = JSON.parse(readFileSync(new URL('./other-body-golden.json', import.meta.url), 'utf8')) as OtherCase[];
for (const [index, fixture] of otherCases.entries()) {
  test(`recipe preserves former complete body ${index + 1} (${fixture.family}, ${fixture.worker})`, () => {
    const actual = fixture.family === 'mm3-lm'
      ? mm3ExecutionBody(fixture.form, fixture.activePreset!, fixture.trainLaunder!, fixture.flashSupported!)
      : fixture.family === 'yue2-nar'
        ? yue2NarExecutionBody(fixture.form, fixture.activePreset!)
        : yue2ArExecutionBody(fixture.form, fixture.overtrain!, fixture.mintedMissing!);
    assert.deepEqual(JSON.parse(JSON.stringify(actual)), fixture.formerBody);
  });
}

test('joint recipe keeps the former default and a saved tuned form across method switches', () => {
  const fixture = JSON.parse(readFileSync(new URL('./joint-form-golden.json', import.meta.url), 'utf8')) as
    { former: Record<string, unknown>; legacy: Record<string, unknown> };
  assert.deepEqual(YUE2_JOINT_FORM_DEFAULTS, fixture.former);
  const baseMatched = resolveRecipe('yue2-joint', { ...fixture.former, steps: 0, cautious: false });
  assert.deepEqual(JSON.parse(JSON.stringify(baseMatched.execution)), { ...fixture.former, steps: 0, cautious: false });
  assert.ok('effective' in baseMatched);
  const tuned = resolveRecipe('yue2-joint', fixture.legacy);
  assert.deepEqual(JSON.parse(JSON.stringify(tuned.execution)), fixture.legacy);
  assert.equal('effective' in tuned, false);
});

test('ACE interactive projection does not leak stored pipeline fields under default form values', () => {
  const stored = { batch: 4, dora: true };
  const lm = resolveRecipe('ace-lm', ACE_LM_FORM_DEFAULTS, 'lora',
    { builtin: ACE_LM_FORM_DEFAULTS, stored, deferred: [] });
  assert.deepEqual(lm.execution, aceLmExecutionBody(ACE_LM_FORM_DEFAULTS));
  assert.equal(lm.resolved.batch, 1);
  assert.equal(lm.provenance.batch, 'override');
  const dit = resolveRecipe('ace-dit', ACE_DIT_FORM_DEFAULTS, 'lora',
    { builtin: ACE_DIT_FORM_DEFAULTS, stored, deferred: [] });
  assert.deepEqual(dit.execution, aceDitExecutionBody(ACE_DIT_FORM_DEFAULTS));
  assert.equal(dit.resolved.batch, 1);
  assert.equal(dit.provenance.batch, 'override');
});
test('partial forms resolve; invalid string controls get a boundary error', () => {
  for (const family of ['mm3-lm', 'yue2-nar', 'yue2-ar'] as const) {
    assert.ok(resolveRecipe(family, {}).execution);
  }
  assert.throws(() => resolveRecipe('mm3-lm', { trigger: 0 }), error =>
    error instanceof TrainingOperationFailure && error.status === 400
      && error.body.issues?.[0]?.path === 'payload.overrides.trigger');
});

test('built-in, stored and explicit default layers retain false, zero and provenance', () => {
  const value = resolveTrainingDefaultLayers({ steps: 500, enabled: true }, { steps: 250, enabled: false }, { steps: 0 });
  assert.deepEqual(value.resolved, { steps: 0, enabled: false });
  assert.deepEqual(value.provenance, { steps: 'override', enabled: 'stored' });
  assert.deepEqual(value.stored, { steps: 250, enabled: false });
});
