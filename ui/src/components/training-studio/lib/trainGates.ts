// trainGates.ts — which page the Train panel shows, as a pure function.
//
// TrainPanel evaluates these gates top to bottom and renders the first that
// holds. The ORDER is behaviour: each backend meets a different subset (SA3
// and YuE2 skip the built gate, MM3 skips the preprocess gate), so it is
// pinned by trainGates.test.ts against the pre-plug-in branches.

import type { TrainingJobKind } from '@/services/trainingApi';
import type { TrainingSection, TrainingTargetId } from './trainingPluginTypes';

export type TrainGate =
  | 'pick-dataset' | 'loading' | 'sa3' | 'no-binary' | 'yue2' | 'needs-build'
  | 'mm3' | 'status-loading' | 'needs-preprocess' | 'other-job' | 'ace';

export interface TrainGateInput {
  target: TrainingTargetId;
  section: TrainingSection;
  hasSelectedDataset: boolean;
  /** Dataset detail and capabilities have both loaded. */
  loaded: boolean;
  binaryAvailable: boolean;
  built: boolean;
  /** ACE LM status has not arrived yet (request in flight or inside the debounce). */
  statusPending: boolean;
  /** The ACE LM status arrived and names no preprocessed variant. */
  noVariant: boolean;
  jobActive: boolean;
  jobKind?: TrainingJobKind;
}

export function resolveTrainGate(i: TrainGateInput): TrainGate {
  if (!i.hasSelectedDataset) return 'pick-dataset';
  if (!i.loaded) return 'loading';
  if (i.target === 'sa3' && i.section === 'train') return 'sa3';
  if (!i.binaryAvailable) return 'no-binary';
  if (i.target === 'yue2') return 'yue2';
  if (!i.built) return 'needs-build';
  if (i.target === 'minimax-m3') return 'mm3';
  if (i.statusPending) return 'status-loading';
  if (i.noVariant) return 'needs-preprocess';
  if (i.jobActive && i.jobKind !== 'train-lm' && i.jobKind !== 'train-dit' && i.jobKind !== 'audition') return 'other-job';
  return 'ace';
}

// ── ACE recipe inputs ────────────────────────────────────────────────────
// What handleStart / handleStartDit hand to resolveTrainingRecipe: the form
// plus the variant the status reported.

export const aceLmRecipeInput = (form: object, variantKey: string | undefined): Record<string, unknown> =>
  ({ ...form, ...(variantKey ? { variantKey } : {}) });

export const aceDitRecipeInput = aceLmRecipeInput;
