// sa3.tsx — the Stable Audio 3 plug-in.
//
// No Preprocess tab: the Train page prepares the instrumentals and the trainer
// encodes them itself. The chips report what is verified on disk (assets.sa3,
// server/src/contracts/training.ts), never what a finished run implies.

import { Sa3TrainCard } from '@/components/training-studio/cards/Sa3TrainCard';
import { sa3Chips, sa3Readiness } from './chips';
import { TRAINING_TARGETS } from '@/components/training-studio/lib/trainingTargets';
import type { TrainingPlugin } from '@/components/training-studio/lib/trainingPluginTypes';

export const sa3Plugin: TrainingPlugin = {
  ...TRAINING_TARGETS.sa3,
  chips: sa3Chips,
  readiness: sa3Readiness,
  renderTrain: ({ datasetId, slug }) => <Sa3TrainCard datasetId={datasetId} slug={slug} />,
};
