// BuildPanel.tsx — dataset.json build step
//
// Writes dataset.json next to the audio (where Side-Step's build_dataset puts
// it), so both tools read the same file. Missing audio blocks the build;
// everything else is an advisory warning.

import React, { useEffect, useRef, useState } from 'react';
import { AlertTriangle, CheckCircle2, Copy, Hammer, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { PatchDatasetInput, TagPosition } from '../../services/trainingApi';
import { StyledSelect } from '../shared/StyledSelect';
import { useTrainingStore } from '../../stores/trainingStore';
import { JobProgress } from './JobProgress';
import { SendToLyricStudio } from './SendToLyricStudio';
import { useActiveTrainingJob } from '../../queries/training-runs/hooks';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
/** One PATCH per keystroke / slider tick would also rewind the control from the
 *  response, so both inputs render from a local draft and settle after this. */
const PATCH_DEBOUNCE_MS = 400;

const TAG_POSITIONS: Array<{ id: TagPosition; labelKey: string }> = [
  { id: 'prepend', labelKey: 'trainingStudio.build.tagPrepend' },
  { id: 'append',  labelKey: 'trainingStudio.build.tagAppend' },
  { id: 'replace', labelKey: 'trainingStudio.build.tagReplace' },
];

export const BuildPanel: React.FC = () => {
  const { t } = useTranslation();
  const detail = useTrainingStore(s => s.detail);
  const samplesById = useTrainingStore(s => s.samplesById);
  const sampleOrder = useTrainingStore(s => s.sampleOrder);
  const patchDataset = useTrainingStore(s => s.patchDataset);
  const startBuild = useTrainingStore(s => s.startBuild);
  const activeJob = useActiveTrainingJob();

  const [building, setBuilding] = useState(false);
  const [copied, setCopied] = useState(false);
  const [tagDraft, setTagDraft] = useState<string | null>(null);
  const [ratioDraft, setRatioDraft] = useState<number | null>(null);
  const patchTimers = useRef<Record<string, number>>({});

  const datasetId = detail?.id;
  useEffect(() => {
    setTagDraft(null);
    setRatioDraft(null);
  }, [datasetId]);

  // Clear pending timers on unmount so a queued PATCH cannot fire after the
  // user has navigated away.
  const timers = patchTimers.current;
  useEffect(() => () => { for (const h of Object.values(timers)) window.clearTimeout(h); }, [timers]);

  const schedulePatch = (key: string, patch: PatchDatasetInput) => {
    window.clearTimeout(patchTimers.current[key]);
    patchTimers.current[key] = window.setTimeout(() => void patchDataset(patch), PATCH_DEBOUNCE_MS);
  };

  if (!detail) return null;

  const included = sampleOrder.map(id => samplesById[id]).filter(s => s && !s.excluded);
  const totalSeconds = included.reduce((sum, s) => sum + (s.duration || 0), 0);
  const instrumental = included.filter(s => s.isInstrumental).length;
  const unlabeled = included.filter(s => !s.caption.trim()).length;
  const missing = included.filter(s => s.fileMissing).length;

  const jobRunning = !!activeJob && (activeJob.status === 'queued' || activeJob.status === 'running');
  const built = !!detail.datasetJsonPath && !!detail.builtAt;
  const canBuild = included.length > 0 && missing === 0 && !jobRunning && !building;

  const handleBuild = async () => {
    setBuilding(true);
    try { await startBuild(); } finally { setBuilding(false); }
  };

  const copyPath = () => {
    try {
      void navigator.clipboard.writeText(detail.datasetJsonPath);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch { /* ignore */ }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className={`${CARD} flex flex-col gap-4`}>
        <div>
          <h2 className="text-sm font-bold text-zinc-900 dark:text-white">{t('trainingStudio.build.title')}</h2>
          <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-1">{t('trainingStudio.build.subtitle')}</p>
        </div>

        {/* Summary */}
        <div className="flex items-center gap-4 flex-wrap text-xs text-zinc-600 dark:text-zinc-400">
          <span>{t('trainingStudio.build.summaryFiles', { count: included.length })}</span>
          <span>{t('trainingStudio.build.summaryDuration', { minutes: Math.round(totalSeconds / 60) })}</span>
          <span>{t('trainingStudio.build.summaryInstrumental', { count: instrumental })}</span>
        </div>

        {/* Trigger word + position */}
        <div className="grid grid-cols-2 gap-3">
          <label className="flex flex-col gap-1.5">
            <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.build.trigger')}</span>
            <input
              type="text"
              value={tagDraft ?? detail.customTag}
              onChange={(e) => { setTagDraft(e.target.value); schedulePatch('customTag', { customTag: e.target.value }); }}
              className="rounded-lg px-3 py-2 text-sm bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-amber-500"
            />
          </label>
          <div className="flex flex-col gap-1.5">
            <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.build.tagPosition')}</span>
            <StyledSelect
              accent="amber"
              value={detail.tagPosition}
              onChange={(v) => void patchDataset({ tagPosition: v as TagPosition })}
              options={TAG_POSITIONS.map(p => ({ value: p.id, label: t(p.labelKey) }))}
            />
          </div>
        </div>

        {/* Genre ratio */}
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">{t('trainingStudio.build.genreRatio')}</span>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={ratioDraft ?? detail.genreRatio}
            onChange={(e) => { setRatioDraft(Number(e.target.value)); schedulePatch('genreRatio', { genreRatio: Number(e.target.value) }); }}
            className="w-full accent-amber-500"
          />
          <span className="text-[11px] text-zinc-500">{t('trainingStudio.build.genreRatioHint', { value: ratioDraft ?? detail.genreRatio })}</span>
        </label>

        {/* Warnings */}
        {included.length > 0 && included.length < 10 && (
          <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
            <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.build.warnSmall', { count: included.length })}
          </div>
        )}
        {unlabeled > 0 && (
          <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
            <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.build.warnUnlabeled', { count: unlabeled })}
          </div>
        )}
        {missing > 0 && (
          <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">
            <AlertTriangle size={13} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.build.warnMissing', { count: missing })}
          </div>
        )}

        <button
          onClick={() => void handleBuild()}
          disabled={!canBuild}
          className="self-start flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {building ? <Loader2 size={15} className="animate-spin" /> : <Hammer size={15} />}
          {building
            ? t('trainingStudio.build.building')
            : built ? t('trainingStudio.build.rebuild') : t('trainingStudio.build.go')}
        </button>
      </div>

      <JobProgress />

      {built && (
        <div className={`${CARD} flex items-start gap-3 border-emerald-500/30`}>
          <CheckCircle2 size={16} className="text-emerald-500 mt-0.5 flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <div className="text-sm font-semibold text-zinc-900 dark:text-white">{t('trainingStudio.build.doneTitle')}</div>
            <div className="text-[11px] font-mono text-zinc-500 truncate" title={detail.datasetJsonPath}>
              {t('trainingStudio.build.donePath', { path: detail.datasetJsonPath })}
            </div>
          </div>
          <button
            onClick={copyPath}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-semibold bg-zinc-100 dark:bg-white/5 border border-zinc-200 dark:border-white/10 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors flex-shrink-0"
          >
            <Copy size={12} /> {copied ? '✓' : t('trainingStudio.build.copyPath')}
          </button>
        </div>
      )}

      <SendToLyricStudio />
    </div>
  );
};

export default BuildPanel;
