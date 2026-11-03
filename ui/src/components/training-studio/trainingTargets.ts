// trainingTargets.ts — which backend Training Studio trains for, and what each one shows.
//
// Pure data and logic, no React, so the phase/stage/gate matrix is unit-tested
// (trainingTargets.test.ts). The effective target is resolved ONCE above the
// tabs (useTrainingTarget) and drives the chips, preparation, train and
// previews. Rendering lives in the plug-ins; this file is what they agree on.

import { TRAINING_TARGET_IDS, type TrainingTargetId } from '../../../../server/src/contracts/training';
import type { TrainingJobKind } from '../../services/trainingApi';
import type { TrainingPhase, TrainingStageSpec, TrainingTargetSpec } from './trainingPluginTypes';

const isTargetId = (v: unknown): v is TrainingTargetId =>
  typeof v === 'string' && (TRAINING_TARGET_IDS as readonly string[]).includes(v);

/** An unregistered backend trains the way the studio always did for it: as ACE. */
export function targetForBackend(backendId: string | null | undefined): TrainingTargetId {
  return isTargetId(backendId) ? backendId : 'ace';
}

/**
 * The target a job was submitted under, from its kind. Null for kinds that are
 * not target-bound (label, build, enhance, audition, calibrate).
 */
export function targetForJobKind(kind: TrainingJobKind | null | undefined): TrainingTargetId | null {
  if (!kind) return null;
  if (kind === 'sa3-train') return 'sa3';
  if (kind === 'preprocess' || kind === 'train-lm' || kind === 'train-dit') return 'ace';
  if (kind.startsWith('mm3-')) return 'minimax-m3';
  if (kind.startsWith('yue2-')) return 'yue2';
  if (kind.startsWith('heartmula-')) return 'heartmula';
  return null;
}

/**
 * Follow the active backend. A queued or running job keeps its own target
 * when the generation backend changes.
 */
export function resolveTrainingTarget(input: {
  activeBackendId: string | null | undefined;
  runningJobKind?: TrainingJobKind | null;
}): TrainingTargetId {
  const locked = targetForJobKind(input.runningJobKind);
  if (locked) return locked;
  return targetForBackend(input.activeBackendId);
}

// ── Static per-target facts ──────────────────────────────────────────────

const stage = (id: string, labelKey: string, label: string): TrainingStageSpec => ({ id, labelKey, label });
const yue2Key = (n: number) => `trainingStudio.yue2.runAllStageName${n}`;

const PREPARE_LABEL = { key: 'trainingStudio.phase.prepare', fallback: 'Prepare' };
const CODES_LABEL = { key: 'trainingStudio.mm3.phaseCodes', fallback: 'Codes' };

const ACE_PHASES: readonly TrainingPhase[] = ['dataset', 'preprocess', 'train', 'monitor'];
const HEARTMULA_PHASES: readonly TrainingPhase[] = ['dataset', 'preprocess', 'train', 'review'];

export const TRAINING_TARGETS: Record<TrainingTargetId, TrainingTargetSpec> = {
  ace: {
    id: 'ace',
    phases: ACE_PHASES,
    reachablePhases: ['dataset', 'preprocess', 'train', 'refine', 'review', 'monitor'],
    phaseLabels: {},
    prepareStages: [stage('tensors', 'trainingStudio.assets.tensors', 'Tensors')],
    trainStages: [stage('lm', 'trainingStudio.assets.lm', 'LM'), stage('dit', 'trainingStudio.assets.dit', 'DiT')],
    recipeFields: [{ stage: 'lm', family: 'ace-lm' }, { stage: 'dit', family: 'ace-dit' }],
    previewSource: 'none',
  },
  'minimax-m3': {
    id: 'minimax-m3',
    phases: ACE_PHASES,
    reachablePhases: ['dataset', 'preprocess', 'train', 'refine', 'review', 'monitor'],
    phaseLabels: { preprocess: CODES_LABEL },
    prepareStages: [stage('codes', 'trainingStudio.assets.codes', 'Codes')],
    trainStages: [stage('lm', 'trainingStudio.assets.lmAdapter', 'LM adapter')],
    recipeFields: [{ stage: 'lm', family: 'mm3-lm' }],
    previewSource: 'mm3-records',
  },
  yue2: {
    id: 'yue2',
    phases: ['dataset', 'preprocess', 'optimise', 'train', 'review'],
    reachablePhases: ['dataset', 'preprocess', 'optimise', 'train', 'refine', 'review'],
    phaseLabels: { preprocess: PREPARE_LABEL },
    // Same ids as the server's Yue2PreparationStage.
    prepareStages: [
      stage('latents', yue2Key(1), 'latent cache'),
      stage('codes', yue2Key(2), 'codes'),
      stage('sheet', yue2Key(3), 'lead sheets'),
      stage('stems', yue2Key(4), 'vocal stems'),
      stage('align', yue2Key(5), 'lyric cursor spans'),
    ],
    trainStages: [
      stage('nar', yue2Key(6), 'NAR LoRA training'),
      stage('ar', yue2Key(7), 'AR LoRA training'),
      stage('joint', 'trainingStudio.yue2.runAllJointStageName', 'joint training'),
    ],
    recipeFields: [{ stage: 'nar', family: 'yue2-nar' }, { stage: 'ar', family: 'yue2-ar' }, { stage: 'joint', family: 'yue2-joint' }],
    previewSource: 'yue2-ladder',
  },
  // HeartMuLa: Prepare encodes audio to HeartCodec codes (no ACE tensors, latents or FSQ); Review is the checkpoint previews.
  heartmula: {
    id: 'heartmula',
    phases: HEARTMULA_PHASES,
    reachablePhases: HEARTMULA_PHASES,
    phaseLabels: { preprocess: PREPARE_LABEL },
    prepareStages: [stage('codes', 'trainingStudio.heartmula.stageCodes', 'HeartCodec codes')],
    trainStages: [stage('lora', 'trainingStudio.heartmula.stageTrain', 'Backbone LoRA')],
    // The recipe is the status route's `defaults`, as SA3's is.
    recipeFields: [],
    previewSource: 'heartmula-previews',
  },
  // No Preprocess tab: the trainer prepares audio and encodes inside the Train page.
  sa3: {
    id: 'sa3',
    phases: ['dataset', 'train'],
    reachablePhases: ['dataset', 'train'],
    phaseLabels: {},
    prepareStages: [],
    trainStages: [
      stage('audio', 'trainingStudio.sa3.stageAudio', 'Prepare audio'),
      stage('train', 'trainingStudio.sa3.stageTrain', 'Train'),
    ],
    recipeFields: [{ stage: 'train', family: 'sa3' }],
    previewSource: 'sa3-previews',
  },
};

/** A phase the target cannot stand on lands on Train. */
export function resolvePhase(target: TrainingTargetId, phase: TrainingPhase): TrainingPhase {
  return TRAINING_TARGETS[target].reachablePhases.includes(phase) ? phase : 'train';
}
