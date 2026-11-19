// yue2.tsx — the YuE2 plug-in: latents, codes, cursor spans and the two adapters.
//
// Its Prepare, Optimise and Train pages all come from the stage shell
// (Yue2TrainStages, Yue2OptimisePanel); Prepare and Train differ only by section.

import { Yue2OptimisePanel } from '@/components/training-studio/panels/Yue2OptimisePanel';
import { Yue2TrainStages } from '@/components/training-studio/panels/Yue2TrainStages';
import { yue2Chips, yue2Readiness } from './chips';
import { TRAINING_TARGETS } from '@/components/training-studio/lib/trainingTargets';
import type { TrainingPlugin } from '@/components/training-studio/lib/trainingPluginTypes';

export const yue2Plugin: TrainingPlugin = {
  ...TRAINING_TARGETS.yue2,
  chips: yue2Chips,
  readiness: yue2Readiness,
  renderPrepare: ({ datasetId, trigger }) => <Yue2TrainStages datasetId={datasetId} trigger={trigger} section="prepare" />,
  renderOptimise: ({ datasetId }) => <Yue2OptimisePanel datasetId={datasetId} />,
  renderTrain: ({ datasetId, trigger }) => <Yue2TrainStages datasetId={datasetId} trigger={trigger} section="train" />,
};
