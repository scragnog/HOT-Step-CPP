// Mm3PreviewStrip.tsx — the mid-run listening strip for MM3 LM training.
//
// The loss curve says "it is fitting". It cannot say whether it is becoming the
// artist or eating the base model, and those two look identical on a chart. So
// the run renders audio at checkpoints and this puts it in front of the ear
// while the run is still abortable.
//
// LAYOUT IS THE ARGUMENT. Rows are grouped by caption (artist / control), and
// within a group they run left to right by step with the no-adapter BASE render
// first. Read a row across and you hear a trajectory; read the control row and
// you hear what the adapter is doing to prompts it was never trained on, which
// is where "the backing got simpler and cheaper" shows up.

import React from 'react';
import { Music4, Volume2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { useTrainingStore, type TrainingPreviewRow } from '../../stores/trainingStore';

function previewUrl(row: TrainingPreviewRow): string {
  return `/api/training/mm3/preview?run=${encodeURIComponent(row.run)}`
       + `&file=${encodeURIComponent(row.file)}`;
}

const KIND_LABEL: Record<TrainingPreviewRow['kind'], string> = {
  artist: 'Artist caption',
  control: 'Control caption',
};

const KIND_BLURB: Record<TrainingPreviewRow['kind'], string> = {
  artist: 'The held-out song\'s own caption with the trigger — is the identity arriving?',
  control: 'A neutral prompt the adapter never trained on, rendered WITH the adapter — is the '
         + 'base model still intact?',
};

const PreviewCard: React.FC<{ row: TrainingPreviewRow }> = ({ row }) => (
  <div className={`flex-shrink-0 w-52 rounded-lg border px-2.5 py-2 ${
    row.base
      ? 'border-sky-500/30 bg-sky-500/5'
      : 'border-zinc-200 dark:border-white/10 bg-zinc-50 dark:bg-white/[0.02]'
  }`}>
    <div className="flex items-baseline justify-between gap-2">
      <span className={`text-xs font-semibold tabular-nums ${
        row.base ? 'text-sky-500' : 'text-zinc-800 dark:text-zinc-200'
      }`}>
        {row.base ? 'base' : `step ${row.step}`}
      </span>
      <span className="text-[10px] text-zinc-500 tabular-nums">
        {row.loss !== undefined ? row.loss.toFixed(3) : ''}
      </span>
    </div>
    <audio controls preload="none" src={previewUrl(row)} className="w-full mt-1.5 h-8" />
    <div className="text-[10px] text-zinc-500 mt-1 tabular-nums">
      {row.seconds}s · seed {row.seed} · rendered in {(row.ms / 1000).toFixed(0)}s
    </div>
  </div>
);

export const Mm3PreviewStrip: React.FC = () => {
  const { t } = useTranslation();
  const previews = useTrainingStore(s => s.trainPreviews);
  if (!previews.length) return null;

  const kinds: Array<TrainingPreviewRow['kind']> = ['artist', 'control'];

  return (
    <div className="mt-4 rounded-xl border border-zinc-200 dark:border-white/5 p-3">
      <div className="flex items-center gap-2 mb-1">
        <Volume2 size={14} className="text-amber-500" />
        <h4 className="text-xs font-semibold text-zinc-900 dark:text-white">
          {t('trainingStudio.mm3.previews', 'Previews')}
        </h4>
        <span className="text-[10px] text-zinc-500">
          {t('trainingStudio.mm3.previewsHint',
            'Same caption, same seed, one variable: the checkpoint.')}
        </span>
      </div>

      {kinds.map(kind => {
        const rows = previews.filter(p => p.kind === kind);
        if (!rows.length) return null;
        return (
          <div key={kind} className="mt-2">
            <div className="flex items-center gap-1.5">
              <Music4 size={11} className="text-zinc-500" />
              <span className="text-[11px] font-medium text-zinc-600 dark:text-zinc-300">
                {KIND_LABEL[kind]}
              </span>
              <span className="text-[10px] text-zinc-500">{KIND_BLURB[kind]}</span>
            </div>
            {/* Horizontal scroll rather than a wrap: the row IS the trajectory,
                and wrapping breaks the left-to-right reading it depends on. */}
            <div className="flex gap-2 mt-1.5 overflow-x-auto pb-1">
              {rows.map(row => <PreviewCard key={`${row.run}-${row.id}`} row={row} />)}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default Mm3PreviewStrip;
