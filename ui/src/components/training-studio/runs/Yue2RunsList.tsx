// Yue2RunsList.tsx — the NAR checkpoint ladder for a YuE2 dataset.
// Read-only on purpose: there is no resume route for YuE2 yet, so the ladder
// is for picking which rung to load into the adapter field.

import { useQuery } from '@tanstack/react-query';
import React from 'react';
import { useTranslation } from 'react-i18next';
import { useReload } from '@/queries/training-runs/hooks';
import { runQueries } from '@/queries/training-runs/queries';
import { formatLoss } from '@/utils/format/format';
import { RunLadderList } from '@/components/training-studio/runs/RunLadderList';

export const Yue2RunsList: React.FC<{ datasetId: string; reloadKey: unknown }> = ({ datasetId, reloadKey }) => {
  const { t } = useTranslation();
  const query = useQuery(runQueries.narRuns(datasetId));
  useReload(runQueries.narRuns(datasetId).queryKey, reloadKey);

  return (
    <RunLadderList
      runs={query.data?.runs ?? null}
      error={query.error?.message ?? null}
      title={t('trainingStudio.yue2.runsTitle', 'Previous runs')}
      blurb={t('trainingStudio.yue2.runsBlurb',
        'Every NAR LoRA this dataset has produced. Each snapshot is a plain safetensors file, so the '
        + 'whole ladder is loadable — copy a path into the YuE2 adapter field to hear that rung.')}
      labels={{
        running: t('trainingStudio.yue2.runsRunning', 'Running now'),
        halted: t('trainingStudio.yue2.runsHalted',
          'This run still holds its optimizer state, which means it stopped before its clean '
          + 'finish — the engine deletes that file on export. Continuing a run is not wired '
          + 'up yet; start a fresh one with the same recipe.'),
        show: t('trainingStudio.yue2.runsShow', 'Ladder'),
        hide: t('trainingStudio.yue2.runsHide', 'Hide ladder'),
        final: t('trainingStudio.yue2.runsFinal', 'final'),
        copy: t('trainingStudio.yue2.runsCopy', 'copy path'),
        copied: t('trainingStudio.yue2.runsCopied', 'copied'),
        steps: (done, cap) => t('trainingStudio.yue2.runsSteps', 'step {{done}} of {{cap}}', { done, cap }),
        ckpts: n => t('trainingStudio.yue2.runsCkpts', '{{n}} checkpoints', { n }),
        trigger: w => t('trainingStudio.yue2.runsTrigger', 'trigger "{{w}}"', { w }),
        shape: (rank, target) => t('trainingStudio.yue2.runsShape', 'rank {{rank}}, {{target}}', { rank, target }),
        best: (loss, step) => t('trainingStudio.yue2.runsBest', 'best loss {{loss}} at {{step}}', { loss, step }),
        checkpointLoss: loss => formatLoss(loss),
      }}
    />
  );
};
