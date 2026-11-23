// mm3.tsx — the MiniMax-Music3 plug-in: RVQ codes and the LM adapter.

import { Mm3TrainCard } from '@/components/training-studio/cards/Mm3TrainCard';
import { mm3Chips, mm3Readiness } from './chips';
import { TRAINING_TARGETS } from '@/components/training-studio/lib/trainingTargets';
import type { TrainingPlugin } from '@/components/training-studio/lib/trainingPluginTypes';

export const mm3Plugin: TrainingPlugin = {
  ...TRAINING_TARGETS['minimax-m3'],
  chips: mm3Chips,
  readiness: mm3Readiness,
  renderTrain: ({ datasetId, trigger }) => <Mm3TrainCard key={datasetId} datasetId={datasetId} trigger={trigger} />,
};
