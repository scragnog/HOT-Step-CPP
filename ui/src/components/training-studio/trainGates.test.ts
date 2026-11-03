import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ACE_LM_FORM_DEFAULTS } from '../../../../server/src/contracts/trainingRecipes';
import type { TrainingJobKind } from '../../services/trainingApi';
import type { TrainingSection, TrainingTargetId } from './trainingPluginTypes';
import { aceDitRecipeInput, aceLmRecipeInput, resolveTrainGate, type TrainGate, type TrainGateInput } from './trainGates';
import { resolveTrainingTarget } from './trainingTargets';

// ── "Before": TrainPanel's gate chain as it stood before the plug-in shell,
// transcribed with the backend-id branches it used.
function legacyGate(backendId: string, section: TrainingSection, i: Omit<TrainGateInput, 'target' | 'section'>): TrainGate {
  const sa3Mode = backendId === 'sa3', mm3Mode = backendId === 'minimax-m3', yue2Mode = backendId === 'yue2';
  if (!i.hasSelectedDataset) return 'pick-dataset';
  if (!i.loaded) return 'loading';
  if (sa3Mode && section === 'train') return 'sa3';
  if (!i.binaryAvailable) return 'no-binary';
  if (yue2Mode) return 'yue2';
  if (!i.built) return 'needs-build';
  if (mm3Mode) return 'mm3';
  if (i.statusPending) return 'status-loading';
  if (i.noVariant) return 'needs-preprocess';
  if (i.jobActive && i.jobKind !== 'train-lm' && i.jobKind !== 'train-dit' && i.jobKind !== 'audition') return 'other-job';
  return 'ace';
}

const bools = [false, true];
const kinds: Array<TrainingJobKind | undefined> = [undefined, 'train-lm', 'train-dit', 'audition', 'label', 'build', 'mm3-codes'];
const sections: TrainingSection[] = ['prepare', 'optimise', 'train'];

test('gate outcome matches the pre-shell chain for ace / yue2 / minimax-m3 across every input', () => {
  let n = 0;
  for (const backend of ['ace', 'yue2', 'minimax-m3'] as TrainingTargetId[]) {
    for (const section of sections) for (const hasSelectedDataset of bools) for (const loaded of bools)
      for (const binaryAvailable of bools) for (const built of bools) for (const statusPending of bools)
        for (const noVariant of bools) for (const jobActive of bools) for (const jobKind of kinds) {
          const rest = { hasSelectedDataset, loaded, binaryAvailable, built, statusPending, noVariant, jobActive, jobKind };
          const target = resolveTrainingTarget({ activeBackendId: backend });
          assert.equal(resolveTrainGate({ target, section, ...rest }), legacyGate(backend, section, rest), JSON.stringify({ backend, section, rest }));
          n++;
        }
  }
  assert.ok(n > 5000);
});

test('SA3 skips the binary, built and preprocess gates on the Train page only', () => {
  const ok = { hasSelectedDataset: true, loaded: true, binaryAvailable: false, built: false, statusPending: false, noVariant: true, jobActive: false };
  assert.equal(resolveTrainGate({ target: 'sa3', section: 'train', ...ok }), 'sa3');
  assert.equal(resolveTrainGate({ target: 'sa3', section: 'prepare', ...ok }), 'no-binary');
  assert.equal(resolveTrainGate({ target: 'sa3', section: 'train', ...ok, loaded: false }), 'loading');
});

// ── Defaults and requests ──
test('ACE request inputs carry the form unchanged, plus the variant only when the status named one', () => {
  const form = { adapterName: 'x', epochs: 3, ...{ rank: 8 } };
  assert.deepEqual(aceLmRecipeInput(form, 'v1'), { ...form, variantKey: 'v1' });
  assert.deepEqual(aceLmRecipeInput(form, ''), form);
  assert.deepEqual(aceDitRecipeInput(form, undefined), form);
  assert.equal(Object.keys(aceLmRecipeInput(form, undefined)).includes('variantKey'), false);
  // The caller's form is not mutated.
  assert.deepEqual(form, { adapterName: 'x', epochs: 3, rank: 8 });
});

test('ACE LM form seeds still come from the shared recipe defaults', async () => {
  const mem = new Map<string, string>();
  Object.assign(globalThis, { localStorage: { getItem: (k: string) => mem.get(k) ?? null, setItem: () => {}, removeItem: () => {} } });
  const { TRAIN_LM_DEFAULTS } = await import('./TrainLmForm');
  assert.deepEqual(TRAIN_LM_DEFAULTS, ACE_LM_FORM_DEFAULTS);
});

// ── Mismatch regressions: the job's target, not the generation backend ──
test('MM3 job running, backend switched to ACE: consumers still resolve MM3 and the target stays pinned', () => {
  const input = { activeBackendId: 'ace', runningJobKind: 'mm3-codes' as TrainingJobKind };
  assert.equal(resolveTrainingTarget(input), 'minimax-m3'); // PreprocessPanel mm3Mode, stepper, Train gate
});

test('YuE2 job running after switching to SA3: DatasetList keeps YuE2; SA3 does not leak', () => {
  assert.equal(resolveTrainingTarget({ activeBackendId: 'sa3', runningJobKind: 'yue2-nar-train' }), 'yue2');
  assert.equal(resolveTrainingTarget({ activeBackendId: 'sa3' }), 'sa3');
});
