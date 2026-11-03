// DatasetDetail.tsx — sub-step router for one dataset: Label → Review → Build

import React from 'react';
import { ArrowLeft, Database } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useTrainingStore } from '../../stores/trainingStore';
import { BuildPanel } from './BuildPanel';
import { JobProgress } from './JobProgress';
import { LabelPanel } from './LabelPanel';
import { SampleGrid } from './SampleGrid';

const STEPS: Array<{ id: 'label' | 'review' | 'build'; labelKey: string }> = [
  { id: 'label',  labelKey: 'trainingStudio.step.label' },
  { id: 'review', labelKey: 'trainingStudio.step.review' },
  { id: 'build',  labelKey: 'trainingStudio.step.build' },
];

export const DatasetDetail: React.FC = () => {
  const { t } = useTranslation();
  const detail = useTrainingStore(s => s.detail);
  const step = useTrainingStore(s => s.step);
  const setStep = useTrainingStore(s => s.setStep);
  const closeDataset = useTrainingStore(s => s.closeDataset);
  const error = useTrainingStore(s => s.error);

  if (!detail) return null;

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <button
          onClick={closeDataset}
          className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-800 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
          title={t('trainingStudio.list.title')}
        >
          <ArrowLeft size={16} />
        </button>
        <Database size={16} className="text-amber-500 flex-shrink-0" />
        <div className="min-w-0">
          <div className="text-sm font-bold text-zinc-900 dark:text-white truncate">{detail.name}</div>
          <div className="text-[11px] font-mono text-zinc-500 truncate" title={detail.sourceDir}>{detail.sourceDir}</div>
        </div>
        <div className="ml-auto text-[11px] text-zinc-500 whitespace-nowrap">
          {t('trainingStudio.list.labeled', { done: detail.labeledCount, total: detail.sampleCount })}
        </div>
      </div>

      {/* Warnings from the server scan */}
      {detail.warnings.length > 0 && (
        <div className="flex flex-col gap-1.5">
          {detail.warnings.map((w, i) => (
            <div key={i} className="px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">{w}</div>
          ))}
        </div>
      )}

      {/* Action failures (rescan, label start, dataset patch, cancel…). Rendered
          here, not inside a step panel, so a failure is never silent. */}
      {error && (
        <div className="px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">{error}</div>
      )}

      {/* Step tabs */}
      <div className="flex items-center gap-1 border-b border-zinc-200 dark:border-white/5">
        {STEPS.map(s => (
          <button
            key={s.id}
            onClick={() => setStep(s.id)}
            className={`px-3 py-2 text-xs font-semibold border-b-2 -mb-px transition-colors ${
              step === s.id
                ? 'text-amber-500 border-amber-500'
                : 'text-zinc-500 border-transparent hover:text-zinc-700 dark:hover:text-zinc-300'
            }`}
          >
            {t(s.labelKey)}
          </button>
        ))}
      </div>

      {/* Job progress must be visible from EVERY step — bulk actions on the
          Review grid start jobs too, and invisible work reads as "nothing
          happened". LabelPanel embeds its own copy, so skip the label step. */}
      {step !== 'label' && <JobProgress />}

      {step === 'label' && <LabelPanel />}
      {step === 'review' && <SampleGrid />}
      {step === 'build' && <BuildPanel />}
    </div>
  );
};

export default DatasetDetail;
