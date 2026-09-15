// Yue2TrainStages.tsx — Training Studio phase 3, YuE2 branch: the six-stage
// page.
//
// Six stages, fixed top-to-bottom order: 1 latent cache, 2 codes, 3 vocal
// stems, 4 lyric cursor spans, 5 NAR LoRA training, 6 AR LoRA training. Each
// stage is its own card — exported from Yue2TrainCard.tsx (1
// Yue2PreprocessCard, 5 Yue2NarTrainCard) or Yue2ArTrainCard.tsx (2
// Yue2TokenizeCard, 3 Yue2StemsCard, 4 Yue2AlignCard, 6
// Yue2ArTrainStageCard) — and this file owns nothing about
// any one stage's form. What it DOES own: the ONE useYue2Status() and ONE
// useYue2ArStatus() call every stage reads from, so six cards never
// independently re-fetch the same two payloads; the licence banner and the
// status-fetch error banner, both rendered once, above every stage; and the
// "Perform all stages" control — identical top and bottom — that drives the
// store's runYue2AllStages chain.
//
// THE RELOAD CALLBACK REFRESHES BOTH HOOKS. Stage 1 (latents) and stage 4
// (NAR training) change what useYue2ArStatus considers blocked for stages 2,
// 3 and 5 — its own `stages.preprocess.done` mirrors the same latent cache —
// and stage 2 or 3 finishing changes the manifest useYue2Status would be
// reading if anything here still read it. Two stale copies of "is the cache
// ready" is how a card starts lying, so both hooks reload together.
//
// THE PREFLIGHT LINE ABOVE EACH RUN-ALL BUTTON is its own small poll of
// listYue2Runs / listYue2ArRuns (useYue2LatestOutcomes below) — advisory only.
// It can go stale by a few seconds after a stage finishes elsewhere; the chain
// itself (trainingStore.runYue2AllStages) re-checks every skip predicate
// fresh, immediately before starting each stage, so a stale preview here never
// produces a stale skip decision, only a preview line that catches up a beat
// late.

import React, { useState } from 'react';
import { ListChecks, Loader2, Scale, XCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { listYue2ArRuns, listYue2Runs } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { Yue2StemsCard, Yue2AlignCard, Yue2ArTrainStageCard, Yue2TokenizeCard } from './Yue2ArTrainCard';
import { Yue2BatchTrainWizard } from './Yue2BatchTrainWizard';
import { Yue2NarTrainCard, Yue2PreprocessCard } from './Yue2TrainCard';
import { useYue2ArStatus } from './useYue2ArStatus';
import { useYue2Status } from './useYue2Status';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const BTN_RUNALL = 'w-full px-4 py-2.5 rounded-lg text-sm font-semibold bg-amber-500 text-black '
                  + 'hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors '
                  + 'flex items-center justify-center gap-2';

/** The newest NAR and AR run's outcome, for the preflight line only — the
 *  store's own chain does not read this, it asks the server fresh. Its own
 *  hook (not useYue2Status/useYue2ArStatus) because neither of those payloads
 *  carries run history; Yue2RunsList/Yue2ArRunsList already fetch the same
 *  two endpoints for the ladder display, so this is a second read of small,
 *  cheap, disk-backed lists, not a second source of truth. */
function useYue2LatestOutcomes(datasetId: string, reloadKey: unknown): { narDone: boolean; arDone: boolean } {
  const [narDone, setNarDone] = useState(false);
  const [arDone, setArDone] = useState(false);

  React.useEffect(() => {
    let cancelled = false;
    void listYue2Runs(datasetId)
      .then(r => { if (!cancelled) setNarDone(r.runs[0]?.outcome === 'completed'); })
      .catch(() => { if (!cancelled) setNarDone(false); });
    void listYue2ArRuns(datasetId)
      .then(r => { if (!cancelled) setArDone(r.runs[0]?.outcome === 'completed'); })
      .catch(() => { if (!cancelled) setArDone(false); });
    return () => { cancelled = true; };
  }, [datasetId, reloadKey]);

  return { narDone, arDone };
}

const RunAllControl: React.FC<{
  datasetId: string;
  trigger: string;
  skipLabels: string[];
  disabled: boolean;
  jobBusyElsewhere: boolean;
  runAllActive: boolean;
  runAllStage: number | null;
  onQueueMultiple: () => void;
}> = ({ datasetId, trigger, skipLabels, disabled, jobBusyElsewhere, runAllActive, runAllStage, onQueueMultiple }) => {
  const { t } = useTranslation();
  const runYue2AllStages = useTrainingStore(s => s.runYue2AllStages);

  const stageName = (n: number): string => {
    switch (n) {
      case 1: return t('trainingStudio.yue2.runAllStageName1', 'latent cache');
      case 2: return t('trainingStudio.yue2.runAllStageName2', 'codes');
      case 3: return t('trainingStudio.yue2.runAllStageName3', 'vocal stems');
      case 4: return t('trainingStudio.yue2.runAllStageName4', 'lyric cursor spans');
      case 5: return t('trainingStudio.yue2.runAllStageName5', 'NAR LoRA training');
      case 6: return t('trainingStudio.yue2.runAllStageName6', 'AR LoRA training');
      default: return '';
    }
  };

  return (
    <div className={CARD}>
      <p className="text-[11px] text-zinc-500 leading-relaxed mb-1">
        {skipLabels.length > 0
          ? t('trainingStudio.yue2.runAllPreflightSkip',
              'Runs every stage below that is not already complete, in order. Already done, so skipped: '
              + '{{skip}}.', { skip: skipLabels.join(', ') })
          : t('trainingStudio.yue2.runAllPreflight',
              'Runs all six stages below in order, from the latent cache through the AR LoRA.')}
      </p>
      <p className="text-[10px] text-zinc-500 leading-snug mb-3">
        {t('trainingStudio.yue2.runAllReloadWarning',
          'This chain runs in this browser tab, not on the server — it does not survive a page reload. '
          + 'Switching between Training Studio phases is fine; closing or reloading the tab stops it.')}
      </p>
      <button
        onClick={() => void runYue2AllStages(datasetId, trigger)}
        disabled={disabled}
        className={BTN_RUNALL}
      >
        {runAllActive ? <Loader2 size={15} className="animate-spin" /> : <ListChecks size={15} />}
        {runAllActive && runAllStage
          ? t('trainingStudio.yue2.runAllRunning', 'Running stage {{n}} of 6: {{name}}',
              { n: runAllStage, name: stageName(runAllStage) })
          : t('trainingStudio.yue2.runAllStart', 'Perform all stages')}
      </button>
      {!runAllActive && jobBusyElsewhere && (
        <p className="text-[10px] text-amber-600 dark:text-amber-400 mt-2">
          {t('trainingStudio.yue2.runAllBusyElsewhere',
            'A stage is already running from its own Start button — wait for it to finish.')}
        </p>
      )}
      {/* The same chain over SEVERAL datasets. Offered here as well as on the
          dataset grid because this page is where someone stands when they
          realise they have five more albums to get through. */}
      <button
        onClick={onQueueMultiple}
        className="mt-2 flex items-center gap-1.5 text-[11px] font-semibold text-amber-600 dark:text-amber-400 hover:underline"
      >
        <ListChecks size={12} /> {t('trainingStudio.yue2.batch.ctaInline', 'Queue several datasets…')}
      </button>
    </div>
  );
};

export const Yue2TrainStages: React.FC<{ datasetId: string; trigger?: string }> = ({ datasetId, trigger }) => {
  const { t } = useTranslation();
  const storeError = useTrainingStore(s => s.error);
  const activeJob = useTrainingStore(s => s.activeJob);
  const yue2RunAllActive = useTrainingStore(s => s.yue2RunAllActive);
  const yue2RunAllStage = useTrainingStore(s => s.yue2RunAllStage);

  const { status: yue2Status, error: yue2StatusError, reload: reloadYue2Status } = useYue2Status(datasetId);
  const { status: arStatus, error: arStatusError, reload: reloadArStatus } = useYue2ArStatus(datasetId);

  // Bumped on every explicit reload, on top of the two hooks' own
  // job-finished auto-refresh — the outcomes poll below has no `activeJob`
  // awareness of its own, so it rides this one nonce instead of duplicating
  // that logic a third time.
  const [reloadNonce, setReloadNonce] = useState(0);
  const [batchOpen, setBatchOpen] = useState(false);
  const reload = () => { reloadYue2Status(); reloadArStatus(); setReloadNonce(n => n + 1); };

  const jobStatus = activeJob?.status;
  const jobBusy = jobStatus === 'queued' || jobStatus === 'running';
  const outcomesKey = `${reloadNonce}:${activeJob?.id ?? ''}:${jobBusy ? 'busy' : jobStatus ?? ''}`;
  const { narDone, arDone } = useYue2LatestOutcomes(datasetId, outcomesKey);

  const effectiveTrigger = trigger ?? '';

  if ((!yue2Status && !yue2StatusError) || (!arStatus && !arStatusError)) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  const skipLabels: string[] = [];
  if (arStatus?.stages.preprocess.done) skipLabels.push(t('trainingStudio.yue2.runAllStageName1', 'latent cache'));
  if (arStatus?.stages.tokenize.done) skipLabels.push(t('trainingStudio.yue2.runAllStageName2', 'codes'));
  // Stems are counted, not flagged: the stage is "done" once anything has been
  // separated, which is the same test the chain itself applies.
  if ((arStatus?.stages.align.stemsReady ?? 0) > 0) {
    skipLabels.push(t('trainingStudio.yue2.runAllStageName3', 'vocal stems'));
  }
  if (arStatus?.stages.align.done) {
    skipLabels.push(t('trainingStudio.yue2.runAllStageName4', 'lyric cursor spans'));
  }
  if (narDone) skipLabels.push(t('trainingStudio.yue2.runAllStageName5', 'NAR LoRA training'));
  if (arDone) skipLabels.push(t('trainingStudio.yue2.runAllStageName6', 'AR LoRA training'));

  const runAllControl = (
    <RunAllControl
      datasetId={datasetId}
      trigger={effectiveTrigger}
      skipLabels={skipLabels}
      disabled={yue2RunAllActive || jobBusy}
      jobBusyElsewhere={jobBusy}
      runAllActive={yue2RunAllActive}
      runAllStage={yue2RunAllStage}
      onQueueMultiple={() => setBatchOpen(true)}
    />
  );

  return (
    <div className="flex flex-col gap-4">
      {(yue2StatusError || arStatusError || storeError) && (
        <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-3 flex items-start gap-2 text-sm text-red-500">
          <XCircle size={16} className="mt-0.5 flex-shrink-0" />
          <span className="min-w-0 break-words">{yue2StatusError || arStatusError || storeError}</span>
        </div>
      )}

      {/* The licence, verbatim as the server sends it, rendered exactly once
          for all six stages. A trained adapter is a derivative of CC BY-NC
          weights and carries the same terms, so this belongs above every
          button that makes one, not in a footnote. */}
      {yue2Status?.license && (
        <div className="rounded-xl border border-amber-500/25 bg-amber-500/10 p-3 flex items-start gap-2 text-[11px] text-amber-700 dark:text-amber-300">
          <Scale size={14} className="mt-0.5 flex-shrink-0" />
          <span className="min-w-0">
            {yue2Status.license}
            {' '}
            {t('trainingStudio.yue2.licenseDerivative',
              'An adapter trained on them is a derivative and carries the same terms.')}
          </span>
        </div>
      )}

      {runAllControl}

      {yue2Status && <Yue2PreprocessCard status={yue2Status} onDone={reload} />}
      {arStatus && <Yue2TokenizeCard status={arStatus} onDone={reload} />}
      {arStatus && <Yue2StemsCard status={arStatus} onDone={reload} />}
      {arStatus && <Yue2AlignCard status={arStatus} onDone={reload} />}
      <Yue2NarTrainCard datasetId={datasetId} trigger={trigger} status={yue2Status} reload={reload} />
      <Yue2ArTrainStageCard datasetId={datasetId} trigger={trigger} status={arStatus} reload={reload} />

      {runAllControl}

      <Yue2BatchTrainWizard open={batchOpen} onClose={() => setBatchOpen(false)} />
    </div>
  );
};

export default Yue2TrainStages;
