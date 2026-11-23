// DatasetAssetChips.tsx — the pipeline-progress pill row for one dataset
//
// Label → Build → (backend-specific pipeline stages), in stage order, lit
// when the artefact is actually on disk and dimmed when it is not. The server
// reads every flag fresh (services/training/datasetAssets.ts), so a deleted
// tensors folder or adapter goes dark on the next refresh.
//
// The stage chips after Labeled/Built depend on the effective training target
// (persisted override or active backend) — ACE's tensors/DiT/LM chips are
// meaningless for MM3, YuE2 or SA3, where the pipeline artefacts are entirely
// different. Each target's plug-in (plugins/*.tsx) builds its own chips. `assets`
// carries every backend's block, so this reads the target itself rather than
// asking both call sites to pass one.
//
// Shared by the dataset cards and the batch-import wizard so both windows onto
// the same corpus never disagree about what is done.

import React from 'react';
import { useTranslation } from 'react-i18next';
import type { DatasetAssets } from '@/services/trainingApi';
import { getTrainingPlugin } from '@/components/training-studio/lib/trainingPlugins';
import type { TrainingTargetId } from '@/components/training-studio/lib/trainingPluginTypes';
import { useTrainingTarget } from '@/components/training-studio/hooks/useTrainingTarget';
import { formatDayYear } from '@/utils/format/dateUtils';

const ON = 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/25';
const OFF = 'text-zinc-400 dark:text-zinc-600 bg-transparent border-zinc-300/60 dark:border-white/10';

function formatWhen(iso: string): string {
  if (!iso) return '';
  const d = new Date(iso.includes('T') || iso.includes('Z') ? iso : iso.replace(' ', 'T') + 'Z');
  if (Number.isNaN(d.getTime())) return iso;
  return formatDayYear(d);
}

interface ChipProps {
  on: boolean;
  label: string;
  title: string;
  suffix?: string;
}

const Chip: React.FC<ChipProps> = ({ on, label, title, suffix }) => (
  <span
    title={title}
    className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full border whitespace-nowrap ${on ? ON : OFF}`}
  >
    {label}{on && suffix ? <span className="font-normal opacity-80"> {suffix}</span> : null}
  </span>
);

interface DatasetAssetChipsProps {
  assets: DatasetAssets | undefined;
  /** Sample counts for the Labeled chip's tooltip — the row's own cached ones. */
  labeledCount?: number;
  sampleCount?: number;
  className?: string;
  /** Defaults to the studio's effective training target. */
  target?: TrainingTargetId;
}

export const DatasetAssetChips: React.FC<DatasetAssetChipsProps> = ({
  assets, labeledCount = 0, sampleCount = 0, className = '', target: targetProp,
}) => {
  const { t } = useTranslation();
  const studioTarget = useTrainingTarget().target;
  const target = targetProp ?? studioTarget;
  if (!assets) return null;

  // Chips whose data block is missing (an old cached row, a response from
  // before this block existed) render OFF rather than disappearing — the row
  // still shows the full stage list for the target.
  const stageChips = getTrainingPlugin(target).chips(assets, { t, formatWhen }).map(c => (
    <Chip key={c.id} on={c.on} label={c.label} title={c.title} suffix={c.suffix} />
  ));

  return (
    <div className={`flex items-center gap-1 flex-wrap ${className}`}>
      <Chip
        on={assets.labeled}
        label={t('trainingStudio.assets.labeled')}
        title={assets.labeled
          ? t('trainingStudio.assets.labeledOn', { done: labeledCount, total: sampleCount })
          : t('trainingStudio.assets.labeledOff')}
      />
      <Chip
        on={assets.built}
        label={t('trainingStudio.assets.built')}
        title={assets.built ? t('trainingStudio.assets.builtOn') : t('trainingStudio.assets.builtOff')}
      />
      {stageChips}
    </div>
  );
};

export default DatasetAssetChips;
