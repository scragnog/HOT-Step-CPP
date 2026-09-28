// Yue2TrainStages.tsx — Training Studio phase 3, YuE2 branch: the seven-stage
// page.
//
// Seven stages, fixed top-to-bottom order: 1 latent cache, 2 codes, 3 lead
// sheets, 4 vocal stems, 5 lyric cursor spans, 6 NAR LoRA training, 7 AR LoRA
// training. Each stage is its own card — exported from Yue2TrainCard.tsx (1
// Yue2PreprocessCard, 6 Yue2NarTrainCard) or Yue2ArTrainCard.tsx (2
// Yue2TokenizeCard, 3 Yue2SheetCard, 4 Yue2StemsCard, 5 Yue2AlignCard, 7
// Yue2ArTrainStageCard) — and this file owns nothing about
// any one stage's form. What it DOES own: the ONE useYue2Status() and ONE
// useYue2ArStatus() call every stage reads from, so seven cards never
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

import React, { useEffect, useState } from 'react';
import { Check, ChevronDown, ListChecks, Loader2, XCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { useTrainingStore } from '../../stores/trainingStore';
import {
  Yue2StemsCard, Yue2AlignCard, Yue2SheetCard, Yue2TokenizeCard,
} from './Yue2ArTrainCard';
import { Yue2PreprocessCard } from './Yue2TrainCard';
import { Yue2AitkTrainCard } from './Yue2AitkTrainCard';
import { Yue2AitkBatchWizard } from './Yue2AitkBatchWizard';
import { useYue2ArStatus } from './useYue2ArStatus';
import { useYue2Status } from './useYue2Status';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';

/** A preparation stage that is already complete folds to one line; click to
 *  open the full card. Decided when the page opens, so a stage that finishes
 *  while you watch stays open, and one that stops being complete (caches
 *  cleared) opens again. */
const DoneFold: React.FC<{ done: boolean; title: string; children: React.ReactNode }> = ({ done, title, children }) => {
  const [open, setOpen] = useState(!done);
  useEffect(() => { if (!done) setOpen(true); }, [done]);
  if (!done) return <>{children}</>;
  return (
    <div className="flex flex-col gap-2">
      <button type="button" onClick={() => setOpen(!open)}
        className={`${CARD} !py-2.5 flex items-center gap-2 text-left hover:bg-zinc-50 dark:hover:bg-white/[0.03] transition-colors`}>
        <ChevronDown size={14} className={`text-zinc-400 transition-transform duration-200 ${open ? 'rotate-180' : ''}`} />
        <Check size={14} className="text-emerald-500" />
        <span className="text-sm font-semibold text-zinc-900 dark:text-white">{title}</span>
        <span className="text-[11px] text-zinc-500">done</span>
      </button>
      {open && children}
    </div>
  );
};
const BTN_RUNALL = 'w-full px-4 py-2.5 rounded-lg text-sm font-semibold bg-amber-500 text-black '
                  + 'hover:bg-amber-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors '
                  + 'flex items-center justify-center gap-2';
const LYRIC_TIMING_KEY = 'hs-yue2-aitk-lyric-timing:';
const RunAllControl: React.FC<{
  datasetId: string;
  trigger: string;
  skipLabels: string[];
  disabled: boolean;
  jobBusyElsewhere: boolean;
  runAllActive: boolean;
  runAllStage: number | null;
  onQueueMultiple: () => void;
  /** Joint Training chain: the button calls this instead of the legacy
   *  runYue2AllStages, whose final stage is the legacy NAR/AR trainers —
   *  the joint chain's final stage is the training card's own start. */
  onRun?: () => void;
  /** Joint Training's shorter stage list; overrides the legacy seven-name map. */
  stageNames?: Record<number, string>;
  /** Pre-flight line for the nothing-is-skipped case. */
  preflightAll?: string;
}> = ({ datasetId, trigger, skipLabels, disabled, jobBusyElsewhere, runAllActive, runAllStage, onQueueMultiple, onRun, stageNames, preflightAll }) => {
  const { t } = useTranslation();
  const runYue2AllStages = useTrainingStore(s => s.runYue2AllStages);

  const total = stageNames ? Object.keys(stageNames).length : 7;
  const stageName = (n: number): string => {
    if (stageNames && stageNames[n]) return stageNames[n];
    switch (n) {
      case 1: return t('trainingStudio.yue2.runAllStageName1', 'latent cache');
      case 2: return t('trainingStudio.yue2.runAllStageName2', 'codes');
      case 3: return t('trainingStudio.yue2.runAllStageName3', 'lead sheets');
      case 4: return t('trainingStudio.yue2.runAllStageName4', 'vocal stems');
      case 5: return t('trainingStudio.yue2.runAllStageName5', 'lyric cursor spans');
      case 6: return t('trainingStudio.yue2.runAllStageName6', 'NAR LoRA training');
      case 7: return t('trainingStudio.yue2.runAllStageName7', 'AR LoRA training');
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
          : preflightAll ?? t('trainingStudio.yue2.runAllPreflight',
              'Runs all seven stages below in order, from the latent cache through the AR LoRA.')}
      </p>
      <p className="text-[10px] text-zinc-500 leading-snug mb-3">
        {t('trainingStudio.yue2.runAllReloadWarning',
          'This chain runs in this browser tab, not on the server — it does not survive a page reload. '
          + 'Switching between Training Studio phases is fine; closing or reloading the tab stops it.')}
      </p>
      <button
        onClick={() => { if (onRun) onRun(); else void runYue2AllStages(datasetId, trigger); }}
        disabled={disabled}
        className={BTN_RUNALL}
      >
        {runAllActive ? <Loader2 size={15} className="animate-spin" /> : <ListChecks size={15} />}
        {runAllActive && runAllStage
          ? t('trainingStudio.yue2.runAllRunning', 'Running stage {{n}} of {{total}}: {{name}}',
              { n: runAllStage, total, name: stageName(runAllStage) })
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
  const [lyricTiming, setLyricTiming] = useState(() => {
    if (typeof window === 'undefined') return false;
    const saved = window.localStorage.getItem(`${LYRIC_TIMING_KEY}${datasetId}`);
    return saved === 'true';
  });
  const storeError = useTrainingStore(s => s.error);
  const activeJob = useTrainingStore(s => s.activeJob);
  const yue2RunAllActive = useTrainingStore(s => s.yue2RunAllActive);
  const yue2RunAllStage = useTrainingStore(s => s.yue2RunAllStage);
  const runYue2JointStages = useTrainingStore(s => s.runYue2JointStages);

  // The Joint Training card hands its own start function out through this
  // ref (see Yue2AitkTrainCard's exposeStart), so the joint run-all chain
  // below can train with the card's current form instead of duplicating the
  // request builder on this page.
  const jointStartRef = React.useRef<(() => Promise<string | null>) | null>(null);
  const exposeJointStart = React.useCallback((fn: () => Promise<string | null>) => { jointStartRef.current = fn; }, []);

  const { status: yue2Status, error: yue2StatusError, reload: reloadYue2Status } = useYue2Status(datasetId);
  const { status: arStatus, error: arStatusError, reload: reloadArStatus } = useYue2ArStatus(datasetId);

  // Bumped on every explicit reload, on top of the two hooks' own
  // job-finished auto-refresh — the outcomes poll below has no `activeJob`
  // awareness of its own, so it rides this one nonce instead of duplicating
  // that logic a third time.
  const [aitkBatchOpen, setAitkBatchOpen] = useState(false);
  const reload = () => { reloadYue2Status(); reloadArStatus(); };

  const jobStatus = activeJob?.status;
  const jobBusy = jobStatus === 'queued' || jobStatus === 'running';

  const effectiveTrigger = trigger ?? '';

  if ((!yue2Status && !yue2StatusError) || (!arStatus && !arStatusError)) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }


  const setAitkLyricTiming = (value: boolean) => {
    setLyricTiming(value);
    window.localStorage.setItem(`${LYRIC_TIMING_KEY}${datasetId}`, String(value));
  };

  {
    // Preparation skip preview for the joint chain: stems and alignment only
    // matter while the lyric-timing objective is on.
    const jointSkipLabels: string[] = [];
    if (arStatus?.stages.preprocess.done) jointSkipLabels.push(t('trainingStudio.yue2.runAllStageName1', 'latent cache'));
    if (arStatus?.stages.tokenize.done) jointSkipLabels.push(t('trainingStudio.yue2.runAllStageName2', 'codes'));
    if (arStatus?.stages.sheet.done) jointSkipLabels.push(t('trainingStudio.yue2.runAllStageName3', 'lead sheets'));
    if (lyricTiming) {
      if ((arStatus?.stages.align.stemsReady ?? 0) > 0) jointSkipLabels.push(t('trainingStudio.yue2.runAllStageName4', 'vocal stems'));
      if (arStatus?.stages.align.done) jointSkipLabels.push(t('trainingStudio.yue2.runAllStageName5', 'lyric cursor spans'));
    }
    // With timing off the chain's final stage lands on the slot-4 number, so
    // the name map agrees with the chain's own numbering.
    const jointTrainingName = t('trainingStudio.yue2.runAllJointStageName', 'joint training');
    const jointStageNames: Record<number, string> = lyricTiming
      ? {
        1: t('trainingStudio.yue2.runAllStageName1', 'latent cache'),
        2: t('trainingStudio.yue2.runAllStageName2', 'codes'),
        3: t('trainingStudio.yue2.runAllStageName3', 'lead sheets'),
        4: t('trainingStudio.yue2.runAllStageName4', 'vocal stems'),
        5: t('trainingStudio.yue2.runAllStageName5', 'lyric cursor spans'),
        6: jointTrainingName,
      }
      : {
        1: t('trainingStudio.yue2.runAllStageName1', 'latent cache'),
        2: t('trainingStudio.yue2.runAllStageName2', 'codes'),
        3: t('trainingStudio.yue2.runAllStageName3', 'lead sheets'),
        4: jointTrainingName,
      };
    const jointRunAllControl = (
      <RunAllControl
        datasetId={datasetId}
        trigger={effectiveTrigger}
        skipLabels={jointSkipLabels}
        disabled={yue2RunAllActive || jobBusy}
        jobBusyElsewhere={jobBusy}
        runAllActive={yue2RunAllActive}
        runAllStage={yue2RunAllStage}
        onQueueMultiple={() => setAitkBatchOpen(true)}
        onRun={() => void runYue2JointStages(datasetId, lyricTiming,
          () => (jointStartRef.current ? jointStartRef.current() : Promise.resolve(null)))}
        stageNames={jointStageNames}
        preflightAll={t('trainingStudio.yue2.runAllJointPreflight',
          'Runs every stage below in order, ending with joint training.')}
      />
    );
    return (
      <div className="flex flex-col gap-4">
        {storeError && (
          <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-3 flex items-start gap-2 text-sm text-red-500">
            <XCircle size={16} className="mt-0.5 flex-shrink-0" />
            <span className="min-w-0 break-words">{storeError}</span>
          </div>
        )}
        <div className={CARD}>
          <div className="flex items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">{t('trainingStudio.yue2.aitkBatch.title', 'Train several datasets')}</h3>
              <p className="text-[11px] text-zinc-500 mt-1">{t('trainingStudio.yue2.aitkBatch.short', 'Run preparation and joint AR + NAR training for multiple datasets in sequence.')}</p>
            </div>
            <button type="button" onClick={() => setAitkBatchOpen(true)} className="shrink-0 rounded-lg bg-amber-500 px-3 py-2 text-xs font-semibold text-black hover:bg-amber-400">
              <ListChecks size={14} className="inline mr-1" />{t('trainingStudio.yue2.aitkBatch.open', 'Train multiple')}
            </button>
          </div>
        </div>
        {jointRunAllControl}
        {yue2Status && <DoneFold key={`pp-${datasetId}`} done={!!arStatus?.stages.preprocess.done} title={t('trainingStudio.yue2.ppTitle', 'Latent cache')}>
          <Yue2PreprocessCard status={yue2Status} onDone={reload} /></DoneFold>}
        {arStatus && <DoneFold key={`tok-${datasetId}`} done={arStatus.stages.tokenize.done} title={t('trainingStudio.yue2ar.tokTitle', 'Codes')}>
          <Yue2TokenizeCard status={arStatus} onDone={reload} /></DoneFold>}
        {arStatus && <DoneFold key={`sheet-${datasetId}`} done={arStatus.stages.sheet.done} title={t('trainingStudio.yue2ar.sheetTitle', 'Lead sheets')}>
          <Yue2SheetCard datasetId={datasetId} status={arStatus} onDone={reload} /></DoneFold>}
        {lyricTiming && arStatus && <DoneFold key={`stems-${datasetId}`} title={t('trainingStudio.yue2ar.stemsTitle', 'Vocal stems')}
          done={arStatus.stages.align.stemsReady > 0 && (arStatus.stages.align.stemsNeeded === 0 || arStatus.stages.align.stemsReady >= arStatus.stages.align.stemsNeeded)}>
          <Yue2StemsCard status={arStatus} onDone={reload} /></DoneFold>}
        {lyricTiming && arStatus && <DoneFold key={`align-${datasetId}`} done={arStatus.stages.align.done} title={t('trainingStudio.yue2ar.alignTitle', 'Lyric cursor spans')}>
          <Yue2AlignCard status={arStatus} onDone={reload} /></DoneFold>}
        <Yue2AitkTrainCard key={datasetId} datasetId={datasetId} legacyManifest={arStatus?.manifestPath || yue2Status?.manifestPath}
          cursorReady={!!arStatus?.stages.align.done} lyricTiming={lyricTiming} onLyricTimingChange={setAitkLyricTiming}
          exposeStart={exposeJointStart} />
        {jointRunAllControl}
        <Yue2AitkBatchWizard open={aitkBatchOpen} onClose={() => setAitkBatchOpen(false)} />
      </div>
    );
  }
};

export default Yue2TrainStages;
