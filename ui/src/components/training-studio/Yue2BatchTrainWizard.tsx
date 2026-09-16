// Yue2BatchTrainWizard.tsx — "Train multiple…" for the YuE2 backend
//
// The sibling of BatchImportWizard, and deliberately NOT the same thing:
// BatchImportWizard turns a folder of folders into datasets, this one takes
// datasets that already exist and queues the seven-stage YuE2 chain over them,
// one after another. It imports nothing.
//
// Two lists in one modal. The STAGES list is the seven stages of
// Yue2TrainStages, all ticked by default, and it applies to every dataset in
// the queue — the point of unticking is "these albums already have latents
// and codes, just train the two LoRAs". The DATASETS list is the store's own
// dataset grid with a checkbox per row, annotated with what that dataset has
// on disk already (`assets.yue2`, the same source the grid's chips read) so a
// queue is picked with eyes open.
//
// WHAT THIS DOES NOT DO: reorder stages, or tick a stage back on because a
// later one needs it. Stage 5 reads the vocal stems stage 4 writes, and stage
// 7 reads both; unticking a stage whose output is missing is a real way to
// make a later stage fail. The dependency line under the stage list says so
// rather than the modal quietly overriding the choice — a silently re-added
// stage is a 40-minute separation run the user explicitly said no to.

import React, { useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { AlertTriangle, ListChecks, Loader2, Search, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import type { TrainingDatasetSummary } from '../../services/trainingApi';
import {
  YUE2_ALL_STAGES, YUE2_STAGES,
  useTrainingStore, type Yue2StageKey, type Yue2StageSet,
} from '../../stores/trainingStore';

const STAGE_LABELS: Record<Yue2StageKey, [key: string, fallback: string]> = {
  latents: ['trainingStudio.yue2.runAllStageName1', 'latent cache'],
  codes:   ['trainingStudio.yue2.runAllStageName2', 'codes'],
  sheet:   ['trainingStudio.yue2.runAllStageName3', 'lead sheets'],
  stems:   ['trainingStudio.yue2.runAllStageName4', 'vocal stems'],
  align:   ['trainingStudio.yue2.runAllStageName5', 'lyric cursor spans'],
  nar:     ['trainingStudio.yue2.runAllStageName6', 'NAR LoRA training'],
  ar:      ['trainingStudio.yue2.runAllStageName7', 'AR LoRA training'],
};

const CHIP_ON = 'text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
const CHIP_OFF = 'text-zinc-500 bg-zinc-500/10 border-zinc-500/20';

/** A dataset is "fully trained" when both LoRA halves are on disk. Only a
 *  default-selection hint — the chain re-checks every skip predicate against
 *  the server when it actually reaches the stage. */
function fullyTrained(ds: TrainingDatasetSummary): boolean {
  return !!ds.assets?.yue2?.narAdapter && !!ds.assets?.yue2?.arAdapter;
}

interface Yue2BatchTrainWizardProps {
  open: boolean;
  onClose: () => void;
}

/** Mount gate. The body seeds its ticks from the dataset list ONCE, in a
 *  useState initialiser, so opening the modal always starts from a fresh
 *  default selection without an effect that re-seeds — and, more importantly,
 *  without one that could fire again mid-session and throw away the user's
 *  ticks when `loadDatasets` refreshes the list underneath them. */
export const Yue2BatchTrainWizard: React.FC<Yue2BatchTrainWizardProps> = ({ open, onClose }) =>
  (open ? <Body onClose={onClose} /> : null);

const Body: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const { t } = useTranslation();
  const datasets = useTrainingStore(s => s.datasets);
  const runYue2Queue = useTrainingStore(s => s.runYue2Queue);
  const queueActive = useTrainingStore(s => s.yue2QueueActive);
  const runAllActive = useTrainingStore(s => s.yue2RunAllActive);

  const [stages, setStages] = useState<Yue2StageSet>(() => ({ ...YUE2_ALL_STAGES }));
  // Default selection: every BUILT dataset that is not already trained end to
  // end. Unbuilt datasets have no dataset.json, so their manifests carry no
  // lyrics and the cursor-span stage would align nothing — selectable, but
  // never the default.
  const [checked, setChecked] = useState<Record<string, boolean>>(() => {
    const seed: Record<string, boolean> = {};
    for (const ds of datasets) seed[ds.id] = !!ds.assets?.built && !fullyTrained(ds);
    return seed;
  });
  const [filter, setFilter] = useState('');
  /** Index of the last row clicked — the anchor a shift-click extends from. */
  const anchorRef = useRef<number | null>(null);

  const visible = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return datasets;
    return datasets.filter(d =>
      d.name.toLowerCase().includes(q)
      || d.albumName.toLowerCase().includes(q)
      || d.customTag.toLowerCase().includes(q)
      || d.sourceDir.toLowerCase().includes(q));
  }, [datasets, filter]);

  const selected = datasets.filter(d => checked[d.id]);
  const allVisibleSelected = visible.length > 0 && visible.every(d => checked[d.id]);
  const someVisibleSelected = visible.some(d => checked[d.id]);
  const anyStage = YUE2_STAGES.some(s => stages[s]);
  const busy = queueActive || runAllActive;
  const canSubmit = selected.length > 0 && anyStage && !busy;

  const toggleRow = (index: number, shift: boolean): void => {
    const row = visible[index];
    if (!row) return;
    const want = !checked[row.id];
    const anchor = anchorRef.current;
    if (shift && anchor !== null && anchor !== index) {
      const [lo, hi] = anchor < index ? [anchor, index] : [index, anchor];
      const next = { ...checked };
      for (let i = lo; i <= hi; i++) next[visible[i].id] = want;
      setChecked(next);
    } else {
      setChecked({ ...checked, [row.id]: want });
    }
    anchorRef.current = index;
  };

  const setAllVisible = (value: boolean): void => {
    const next = { ...checked };
    for (const d of visible) next[d.id] = value;
    setChecked(next);
    anchorRef.current = null;
  };

  const handleSubmit = (): void => {
    if (!canSubmit) return;
    void runYue2Queue(
      selected.map(d => ({ datasetId: d.id, name: d.name, trigger: d.customTag || '' })),
      stages,
    );
    onClose();
  };

  // Which selected datasets can't produce cursor spans — the one preflight
  // worth surfacing, because the stage exits 0 having aligned nothing rather
  // than failing, and the AR LoRA behind it trains on an empty cursor column.
  const unbuiltSelected = selected.filter(d => !d.assets?.built);

  return createPortal(
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="w-[680px] max-h-[85vh] flex flex-col overflow-hidden rounded-xl border border-zinc-200 dark:border-white/10 bg-white dark:bg-suno-card shadow-2xl">
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-200 dark:border-white/5">
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
            {t('trainingStudio.yue2.batch.title', 'Queue YuE2 training')}
          </h3>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-800 dark:hover:text-white transition-colors">
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 flex flex-col gap-4">
          <p className="text-[11px] text-zinc-500 leading-relaxed">
            {t('trainingStudio.yue2.batch.intro',
              'Runs the seven-stage YuE2 chain over each dataset you tick, one dataset at a time, in the '
              + 'order shown. Each one uses its own trigger word. Stages already complete for a dataset '
              + 'are skipped; a dataset that fails does not stop the ones behind it.')}
          </p>

          {/* Stages */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between gap-2">
              <label className="text-xs font-semibold text-zinc-600 dark:text-zinc-400">
                {t('trainingStudio.yue2.batch.stages', 'Stages to perform')}
              </label>
              <div className="flex items-center gap-3 text-[11px] font-semibold">
                <button
                  onClick={() => setStages({ ...YUE2_ALL_STAGES })}
                  className="text-amber-600 dark:text-amber-400 hover:underline"
                >
                  {t('trainingStudio.yue2.batch.stagesAll', 'All')}
                </button>
                <button
                  onClick={() => setStages({
                    latents: false, codes: false, sheet: false, stems: false, align: false, nar: true, ar: true,
                  })}
                  className="text-amber-600 dark:text-amber-400 hover:underline"
                >
                  {t('trainingStudio.yue2.batch.stagesTrainOnly', 'Training only')}
                </button>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              {YUE2_STAGES.map((stage, i) => (
                <label
                  key={stage}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-zinc-200 dark:border-white/10 text-xs text-zinc-700 dark:text-zinc-300 cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={stages[stage]}
                    onChange={() => setStages({ ...stages, [stage]: !stages[stage] })}
                    className="accent-amber-500"
                  />
                  <span className="tabular-nums text-zinc-500">{i + 1}</span>
                  {t(STAGE_LABELS[stage][0], STAGE_LABELS[stage][1])}
                </label>
              ))}
            </div>
            <div className="flex items-start gap-2 text-[11px] text-zinc-500">
              <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.yue2.batch.stageDeps',
                'Later stages read what earlier ones write: cursor spans need the vocal stems, and both '
                + 'LoRAs need the latent cache. Unticking a stage whose output is missing will make the '
                + 'stages after it fail.')}
            </div>
            {!anyStage && (
              <div className="px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-xs text-red-500 dark:text-red-400">
                {t('trainingStudio.yue2.batch.noStages', 'Pick at least one stage.')}
              </div>
            )}
          </div>

          {/* Datasets */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between gap-2 flex-wrap">
              <label className="flex items-center gap-2 text-xs font-semibold text-zinc-600 dark:text-zinc-400 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={allVisibleSelected}
                  ref={(el) => { if (el) el.indeterminate = someVisibleSelected && !allVisibleSelected; }}
                  onChange={() => setAllVisible(!allVisibleSelected)}
                  disabled={visible.length === 0}
                  className="accent-amber-500"
                />
                {t('trainingStudio.yue2.batch.datasets', 'Datasets')}
                <span className="font-normal text-zinc-500">
                  ({t('trainingStudio.yue2.batch.selectedCount', '{{count}} selected', { count: selected.length })})
                </span>
              </label>
              <span className="text-[11px] text-zinc-500 hidden sm:inline">
                {t('trainingStudio.batch.shiftHint')}
              </span>
            </div>

            <div className="relative">
              <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-zinc-400 pointer-events-none" />
              <input
                type="text"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder={t('trainingStudio.yue2.batch.filter', 'Filter by name, album or trigger…') as string}
                className="w-full pl-8 pr-3 py-2 rounded-lg text-xs bg-zinc-100 dark:bg-black/20 border border-zinc-200 dark:border-white/10 text-zinc-700 dark:text-zinc-300 focus:outline-none focus:border-amber-500"
              />
            </div>

            <div className="rounded-xl border border-zinc-200 dark:border-white/5 divide-y divide-zinc-200 dark:divide-white/5 max-h-72 overflow-y-auto">
              {visible.length === 0 && (
                <div className="px-3 py-4 text-xs text-zinc-500 text-center">
                  {t('trainingStudio.yue2.batch.noMatches', 'No datasets match.')}
                </div>
              )}
              {visible.map((ds, i) => {
                const y = ds.assets?.yue2;
                const chips: Array<[string, boolean]> = [
                  [t('trainingStudio.yue2.batch.chipLatents', 'latents'), !!y?.latentsReady],
                  [t('trainingStudio.yue2.batch.chipCodes', 'codes'), !!y?.codesReady],
                  [t('trainingStudio.yue2.batch.chipCursor', 'cursor'), !!y?.cursorReady],
                  [t('trainingStudio.yue2.batch.chipNar', 'NAR'), !!y?.narAdapter],
                  [t('trainingStudio.yue2.batch.chipAr', 'AR'), !!y?.arAdapter],
                ];
                return (
                  <label
                    key={ds.id}
                    className="flex flex-col gap-1 px-3 py-2 text-xs select-none cursor-pointer hover:bg-black/5 dark:hover:bg-white/5"
                  >
                    <div className="flex items-center gap-2.5">
                      {/* onClick rather than onChange: only the mouse event
                          carries shiftKey, and a click forwarded by the
                          wrapping <label> preserves it. */}
                      <input
                        type="checkbox"
                        checked={!!checked[ds.id]}
                        onClick={(e) => toggleRow(i, e.shiftKey)}
                        onChange={() => { /* handled in onClick */ }}
                        className="accent-amber-500 flex-shrink-0"
                      />
                      <span
                        className="flex-1 min-w-0 truncate text-zinc-700 dark:text-zinc-300 font-medium"
                        title={ds.sourceDir}
                      >
                        {ds.name}
                      </span>
                      {ds.customTag ? (
                        <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-amber-600 dark:text-amber-400 bg-amber-500/10 border-amber-500/20 flex-shrink-0">
                          {ds.customTag}
                        </span>
                      ) : (
                        <span
                          className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-zinc-500 bg-zinc-500/10 border-zinc-500/20 flex-shrink-0"
                          title={t('trainingStudio.yue2.batch.noTriggerHint',
                            'No trigger word on this dataset — it trains without one.') as string}
                        >
                          {t('trainingStudio.yue2.batch.noTrigger', 'no trigger')}
                        </span>
                      )}
                      {!ds.assets?.built && (
                        <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded-full border text-red-500 bg-red-500/10 border-red-500/20 flex-shrink-0">
                          {t('trainingStudio.yue2.batch.notBuilt', 'not built')}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-1.5 flex-wrap pl-[26px]">
                      {ds.albumName && (
                        <span className="text-[11px] text-zinc-500 truncate max-w-[200px]">{ds.albumName}</span>
                      )}
                      <span className="text-[11px] text-zinc-500 tabular-nums">
                        {t('trainingStudio.list.samples', { count: ds.sampleCount })}
                      </span>
                      {chips.map(([label, on]) => (
                        <span
                          key={label}
                          className={`text-[10px] font-semibold px-1.5 py-0.5 rounded-full border ${on ? CHIP_ON : CHIP_OFF}`}
                        >
                          {label}
                        </span>
                      ))}
                    </div>
                  </label>
                );
              })}
            </div>
          </div>

          {unbuiltSelected.length > 0 && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-[11px] text-amber-700 dark:text-amber-300">
              <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.yue2.batch.unbuiltWarning',
                '{{count}} selected dataset(s) have not been built. Their manifests carry no lyrics, so '
                + 'the cursor-span stage will align nothing and the AR LoRA will train without it. Build '
                + 'them in the Dataset phase first.', { count: unbuiltSelected.length })}
            </div>
          )}

          {busy && (
            <div className="px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-700 dark:text-amber-300">
              {t('trainingStudio.yue2.batch.busy',
                'A YuE2 run is already going — wait for it to finish before queueing another.')}
            </div>
          )}

          <p className="text-[10px] text-zinc-500 leading-snug">
            {t('trainingStudio.yue2.runAllReloadWarning',
              'This chain runs in this browser tab, not on the server — it does not survive a page reload. '
              + 'Switching between Training Studio phases is fine; closing or reloading the tab stops it.')}
          </p>
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-zinc-200 dark:border-white/5 bg-zinc-50 dark:bg-black/20">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-zinc-200 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-300 dark:hover:bg-zinc-600 transition-colors"
          >
            {t('trainingStudio.wizard.cancel')}
          </button>
          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {busy ? <Loader2 size={13} className="animate-spin" /> : <ListChecks size={13} />}
            {t('trainingStudio.yue2.batch.start', 'Queue {{count}} dataset(s)', { count: selected.length })}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
};

export default Yue2BatchTrainWizard;
