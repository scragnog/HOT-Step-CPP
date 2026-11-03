import assert from 'node:assert/strict';
import { test } from 'node:test';
import { yue2PreparationStageSchema } from '../../../../server/src/contracts/trainingPreparation';
import { trainingRecipeFamilySchema } from '../../../../server/src/contracts/trainingRecipes';
import type { TrainingPhase } from '../../stores/trainingStore';
import {
  TRAINING_TARGETS, resolvePhase, resolveTrainingTarget, targetForBackend, targetForJobKind,
} from './trainingTargets';

const ALL_PHASES: TrainingPhase[] = ['dataset', 'preprocess', 'optimise', 'train', 'refine', 'review', 'monitor'];

// ── The "before" matrix for ace / yue2 / minimax-m3 with target = active ──
// Frozen from PhaseStepper.tsx and TrainingStudio.tsx as they stood before the
// plug-in shell (the backend-id branches and the two phase-fallback lines).
const LEGACY_VISIBLE: Record<string, TrainingPhase[]> = {
  ace: ['dataset', 'preprocess', 'train', 'monitor'],
  'minimax-m3': ['dataset', 'preprocess', 'train', 'monitor'],
  yue2: ['dataset', 'preprocess', 'optimise', 'train', 'review'],
};
const LEGACY_LABEL: Record<string, string | undefined> = {
  ace: undefined, 'minimax-m3': 'trainingStudio.mm3.phaseCodes', yue2: 'trainingStudio.phase.prepare',
};
const legacyFallback = (backendId: string, phase: TrainingPhase): TrainingPhase =>
  (backendId === 'yue2' && phase === 'monitor') || (backendId !== 'yue2' && phase === 'optimise') ? 'train' : phase;

for (const backend of ['ace', 'minimax-m3', 'yue2']) {
  test(`${backend}, target=active: visible phases, labels and fallbacks match the pre-shell behaviour`, () => {
    const target = resolveTrainingTarget({ activeBackendId: backend });
    assert.equal(target, backend);
    const spec = TRAINING_TARGETS[target];
    assert.deepEqual(spec.phases, LEGACY_VISIBLE[backend]);
    assert.equal(spec.phaseLabels.preprocess?.key, LEGACY_LABEL[backend]);
    for (const p of ALL_PHASES) assert.equal(resolvePhase(target, p), legacyFallback(backend, p), `${backend}/${p}`);
  });
}

test('stage ids and recipe families are stable and match the server contracts', () => {
  const yue2 = TRAINING_TARGETS.yue2;
  assert.deepEqual(yue2.prepareStages.map(s => s.id), ['latents', 'codes', 'sheet', 'stems', 'align']);
  assert.deepEqual(yue2.trainStages.map(s => s.id), ['nar', 'ar', 'joint']);
  const serverIds = yue2PreparationStageSchema.options as readonly string[];
  for (const s of [...yue2.prepareStages, ...yue2.trainStages]) assert.ok(serverIds.includes(s.id), s.id);
  assert.deepEqual(TRAINING_TARGETS.ace.trainStages.map(s => s.id), ['lm', 'dit']);
  assert.deepEqual(TRAINING_TARGETS['minimax-m3'].prepareStages.map(s => s.id), ['codes']);
  for (const spec of Object.values(TRAINING_TARGETS)) {
    for (const r of spec.recipeFields) {
      if (r.family !== 'sa3') assert.ok(trainingRecipeFamilySchema.options.includes(r.family), r.family);
      assert.ok([...spec.prepareStages, ...spec.trainStages].some(s => s.id === r.stage), `${spec.id}/${r.stage}`);
    }
  }
});

// ── The target rule ──
test('training follows the active generation backend', () => {
  assert.equal(resolveTrainingTarget({ activeBackendId: 'ace' }), 'ace');
  assert.equal(resolveTrainingTarget({ activeBackendId: 'sa3' }), 'sa3');
  assert.equal(resolveTrainingTarget({ activeBackendId: 'yue2' }), 'yue2');
  assert.equal(targetForBackend('some-future-engine'), 'ace');
  assert.equal(targetForBackend(undefined), 'ace');
});

test('a running job keeps the target it was submitted with', () => {
  assert.equal(resolveTrainingTarget({ activeBackendId: 'ace', runningJobKind: 'sa3-train' }), 'sa3');
  assert.equal(resolveTrainingTarget({ activeBackendId: 'ace', runningJobKind: 'train-dit' }), 'ace');
  assert.equal(resolveTrainingTarget({ activeBackendId: 'yue2', runningJobKind: 'yue2-joint-train' }), 'yue2');
  assert.equal(resolveTrainingTarget({ activeBackendId: 'ace', runningJobKind: 'mm3-train-lm' }), 'minimax-m3');
  // Kinds that are not target-bound never pin it.
  for (const kind of ['label', 'build', 'audition', 'enhance-caption'] as const) {
    assert.equal(targetForJobKind(kind), null);
    assert.equal(resolveTrainingTarget({ activeBackendId: 'ace', runningJobKind: kind }), 'ace');
  }
});

// ── SA3 backend ──
test('SA3 has no Preprocess, Optimise or Monitor tab and shows Dataset and Train', () => {
  const sa3 = TRAINING_TARGETS.sa3;
  assert.deepEqual(sa3.phases, ['dataset', 'train']);
  assert.deepEqual(sa3.prepareStages, []);
  assert.equal(sa3.previewSource, 'sa3-previews');
});

test('switching tabs: every phase SA3 cannot stand on falls back to Train, the rest stay', () => {
  for (const p of ALL_PHASES) {
    assert.equal(resolvePhase('sa3', p), p === 'dataset' || p === 'train' ? p : 'train', p);
  }
});

test('switching the backend from ACE to SA3 moves a user off Preprocess and Monitor', () => {
  const before = resolveTrainingTarget({ activeBackendId: 'ace' });
  const after = resolveTrainingTarget({ activeBackendId: 'sa3' });
  assert.equal(resolvePhase(before, 'preprocess'), 'preprocess');
  assert.equal(resolvePhase(after, 'preprocess'), 'train');
  assert.equal(resolvePhase(after, 'monitor'), 'train');
  // And back: Train stays Train, Dataset stays Dataset.
  assert.equal(resolvePhase(before, 'train'), 'train');
  assert.equal(resolvePhase(before, 'dataset'), 'dataset');
});

test('HeartMuLa trains under its own target: Prepare, Train, Review, no ACE stages', () => {
  const hm = TRAINING_TARGETS.heartmula;
  assert.deepEqual(hm.phases, ['dataset', 'preprocess', 'train', 'review']);
  assert.equal(hm.phaseLabels.preprocess?.fallback, 'Prepare');
  assert.deepEqual(hm.prepareStages.map(s => s.id), ['codes']);
  assert.equal(hm.previewSource, 'heartmula-previews');
  assert.equal(targetForJobKind('heartmula-prepare'), 'heartmula');
  assert.equal(targetForJobKind('heartmula-train'), 'heartmula');
  assert.equal(targetForBackend('heartmula'), 'heartmula');
  assert.equal(resolveTrainingTarget({ activeBackendId: 'ace', runningJobKind: 'heartmula-train' }), 'heartmula');
  assert.equal(resolvePhase('heartmula', 'optimise'), 'train');
});
