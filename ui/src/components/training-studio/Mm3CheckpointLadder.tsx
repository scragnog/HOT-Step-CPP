// Mm3CheckpointLadder.tsx — an MM3 run's checkpoint ladder, drawn with the same
// rung and actions as YuE2's (TrainingPreviewRung, TrainingLadderActions): one
// rung per saved checkpoint with its takes, "Render more", and "Use this
// checkpoint", which points the dataset's album presets and the global LM
// adapter at that step. Read from the run folder, so it survives a reload.
// MM3 has no rung scores, so there is no score row.
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useReload } from '../../queries/training-runs/hooks';
import { runQueries } from '../../queries/training-runs/queries';
import { queryManager } from '../../queries/invalidation';
import { getMm3Ladder, renderMm3LadderTake } from '../../services/trainingApi';
import { selectMm3Checkpoint } from '../../services/trainingReviewApi';
import { setBackendParamFor } from '../../stores/globalParamsStore';
import { Mm3LadderRows } from './Mm3LadderRows';
import { TrainingCheckpointLadder } from './TrainingPreviewPanel';

export const Mm3CheckpointLadder: React.FC<{
  datasetId: string;
  run: string;
  /** Changes when the live run reports a new preview, to re-read the folder. */
  reloadKey?: unknown;
}> = ({ datasetId, run, reloadKey }) => {
  const { t } = useTranslation();
  const client = useQueryClient();
  const options = runQueries.mm3Ladder(datasetId, run);
  const query = useQuery({ ...options, enabled: !!datasetId && !!run });
  useReload(options.queryKey, reloadKey);
  const [note, setNote] = useState('');
  const [error, setError] = useState('');
  const fail = (err: unknown) => setError(err instanceof Error ? err.message : String(err));
  const refresh = () => void queryManager.invalidate(client, options.queryKey);

  const renderMut = useMutation({
    mutationFn: (step: number) => renderMm3LadderTake(datasetId, run, step),
    onMutate: () => setError(''),
    onSuccess: refresh,
    onError: fail,
  });
  // The ladder is re-read first so the dataset revision the command carries
  // is current and a checkpoint deleted since can't be picked.
  const selectMut = useMutation({
    mutationFn: async (step: number) => {
      const fresh = await getMm3Ladder(datasetId, run);
      if (!fresh.checkpoints.some(c => c.step === step)) throw new Error('That checkpoint is no longer on disk. Reload the ladder.');
      return selectMm3Checkpoint(fresh.dataset, run, step);
    },
    onMutate: () => { setError(''); setNote(''); },
    onSuccess: (r, step) => {
      setBackendParamFor('minimax-m3', 'mm3LmAdapter', r.ref);
      setNote(r.presets
        ? t('trainingStudio.mm3.usePresets', 'Step {{step}} is now the LM adapter, and {{count}} album preset(s) use it.', { step, count: r.presets })
        : t('trainingStudio.mm3.useNoPreset', 'Step {{step}} is now the LM adapter. No album preset was linked to this dataset to update.', { step }));
      refresh();
    },
    onError: fail,
  });

  const ladder = query.data;
  if (!ladder) return query.error ? <div className="mt-2 text-xs text-rose-500">{query.error.message}</div> : null;
  if (!ladder.checkpoints.length && !ladder.previews.length) return null;
  return <TrainingCheckpointLadder>
    <p className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">{t('trainingStudio.yue2.method.ladderTitle', 'Checkpoint ladder')} · {run}{ladder.running ? ` · ${t('trainingStudio.yue2.method.ladderLive', 'training')}` : ''}</p>
    <p className="mt-1 text-[11px] text-zinc-500">{t('trainingStudio.mm3.ladderHint', "Every saved checkpoint is a rung. Its first take uses the run's preview seed and caption, so rungs compare like with like; Render more adds takes on a new random seed. Use this checkpoint makes that step the LM adapter in the global bar and in this dataset's album presets; pick another step to move it. Rendering and picking open once the run has finished.")}</p>
    {note && <div className="mt-2 text-[12px] text-emerald-700 dark:text-emerald-300">{note}</div>}
    {error && <div className="mt-2 text-[12px] text-rose-500">{error}</div>}
    <Mm3LadderRows ladder={ladder} rendering={renderMut.isPending ? renderMut.variables ?? null : null}
      picking={selectMut.isPending} onRender={step => renderMut.mutate(step)} onUse={step => selectMut.mutate(step)} />
  </TrainingCheckpointLadder>;
};

export default Mm3CheckpointLadder;
