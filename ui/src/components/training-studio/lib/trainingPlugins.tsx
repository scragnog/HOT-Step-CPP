// trainingPlugins.tsx — the registry of training plug-ins, keyed by target.
//
// One entry per backend; TrainingStudio, PhaseStepper, DatasetAssetChips and
// TrainPanel read everything target-specific from here instead of branching on
// backend ids themselves.

import type { TrainingPlugin, TrainingTargetId } from './trainingPluginTypes';
import { acePlugin } from '@/components/training-studio/plugins/ace';
import { mm3Plugin } from '@/components/training-studio/plugins/mm3';
import { sa3Plugin } from '@/components/training-studio/plugins/sa3';
import { createHeartmulaPlugin } from '@/components/training-studio/plugins/heartmula';
import { heartmulaChips, heartmulaReadiness } from '@/components/training-studio/plugins/chips';
import { TRAINING_TARGETS } from './trainingTargets';
import { useHeartmulaController } from '@/components/training-studio/hooks/useHeartmulaController';
import { yue2Plugin } from '@/components/training-studio/plugins/yue2';

const heartmulaPlugin = createHeartmulaPlugin({
  target: { ...TRAINING_TARGETS.heartmula, chips: heartmulaChips, readiness: heartmulaReadiness },
  useController: useHeartmulaController,
});

export const TRAINING_PLUGINS: Record<TrainingTargetId, TrainingPlugin> = {
  ace: acePlugin,
  yue2: yue2Plugin,
  'minimax-m3': mm3Plugin,
  sa3: sa3Plugin,
  heartmula: heartmulaPlugin,
};

export const getTrainingPlugin = (target: TrainingTargetId): TrainingPlugin => TRAINING_PLUGINS[target];
