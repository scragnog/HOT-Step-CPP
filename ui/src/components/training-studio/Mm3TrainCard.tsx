// Mm3TrainCard.tsx — Training Studio phase 3, MiniMax-Music3 branch.
//
// Rendered instead of the ACE LM/DiT cards when the active backend is
// MiniMax-Music3: codes + the MM3-native captions -> an LM LoRA.
//
// The CODES export is phase 2, not here (Mm3CodesCard) — for MM3, codes are
// what preprocessing means. This card only GATES on them, exactly as the ACE
// train panel gates on a preprocess variant, so there is one place to run the
// export and one place to run training.
//
// Everything below the form is the SHARED job machinery: the same SSE stream,
// the same JobProgress, the same loss chart.
//
// THE DEFAULTS ARE NOT DUPLICATED HERE. They arrive in the `mm3` status
// payload from services/training/mm3Train.ts, which is the single place the
// validated recipe lives. This form seeds itself from that response, so a
// change on the server reaches the UI without an edit here.

import React, { useState } from 'react';
import {
  AlertTriangle, ChevronDown, ChevronRight, Cpu, Loader2, Play, ShieldCheck, Volume2, XCircle,
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { estimateMm3PeakMb } from '../../services/trainingApi';
import type { Mm3TrainLmRequest } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { JobProgress } from './JobProgress';
import { Mm3PreviewStrip } from './Mm3PreviewStrip';
import { useMm3Status } from './useMm3Status';
import { TrainingChart } from './TrainingChart';

const CARD = 'rounded-xl border border-zinc-200 dark:border-white/5 bg-white dark:bg-suno-card p-4';
const INPUT = 'w-full px-2.5 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 '
            + 'dark:border-white/10 text-sm text-zinc-800 dark:text-zinc-200 outline-none '
            + 'focus:border-amber-500/50';

interface FormState {
  steps: number;
  saveEvery: number;
  rank: number;
  alpha: number;
  lr: number;
  maxFrames: number;
  cropMode: 'random' | 'beginning' | 'structured';
  cropStartFrac: number;
  cropEndFrac: number;
  cropStartTiles: number;
  depthLossWeight: number;
  depthLossFrames: number;
  optimizer: 'muon' | 'adamw' | 'prodigy';
  muonLrScale: number;
  adapterType: 'lora' | 'lokr';
  lokrFactor: number;
  sharedCaption: string;
  gradAccum: number;
  seed: number;
  trigger: string;
  triggerPrepend: boolean;
  basePrecision: string;
  holdout: number;
  evalEvery: number;
  cropAnchor: 'song' | 'zero';
  previewEverySteps: number;
  previewEveryMinutes: number;
  previewSeconds: number;
  previewSeed: number;
  previewCaption: string;
  previewControl: boolean;
  previewBaseline: boolean;
  previewScaleMlp: number;
  regDatasetId: string;
  regEvery: number;
  regTopK: number;
}

const NumField: React.FC<{
  label: string; value: number; onChange: (v: number) => void; step?: number; hint?: string;
}> = ({ label, value, onChange, step = 1, hint }) => (
  <label className="flex flex-col gap-1">
    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">{label}</span>
    <input
      type="number" className={INPUT} value={value} step={step}
      onChange={e => onChange(Number(e.target.value))}
    />
    {hint && <span className="text-[10px] text-zinc-500 leading-snug">{hint}</span>}
  </label>
);

export const Mm3TrainCard: React.FC<{ datasetId: string; trigger?: string }> = ({ datasetId, trigger }) => {
  const { t } = useTranslation();
  const activeJob = useTrainingStore(s => s.activeJob);
  const startMm3TrainLm = useTrainingStore(s => s.startMm3TrainLm);
  const mm3Live = useTrainingStore(s => s.mm3Live);
  const storeError = useTrainingStore(s => s.error);
  const setPhase = useTrainingStore(s => s.setPhase);
  const trainStepSeries = useTrainingStore(s => s.trainStepSeries);
  const trainMilestones = useTrainingStore(s => s.trainMilestones);
  const trainLmEpochs = useTrainingStore(s => s.trainLmEpochs);
  const trainEvalSeries = useTrainingStore(s => s.trainEvalSeries);
  const trainMaxEpochs = useTrainingStore(s => s.trainMaxEpochs);

  const { status, error: statusError } = useMm3Status(datasetId);
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  // The form is DERIVED, not seeded: server defaults underneath, the user's
  // edits on top. No effect copies one into the other, so there is no window
  // where the form holds stale numbers and no cascading render — and the
  // validated recipe still has exactly one home (mm3Train.ts).
  const [edits, setEdits] = useState<Partial<FormState>>({});

  const jobKind = activeJob?.kind;
  const jobStatus = activeJob?.status;
  const jobRunning = jobStatus === 'queued' || jobStatus === 'running';
  const mine = jobKind === 'mm3-train-lm';

  // Muon normalises its update, so the schedule LR alone understates what the
  // optimizer applied by the scale factor.
  const lrScale = status?.defaults.optimizer === 'muon' ? (status.defaults.muonLrScale ?? 1) : 1;

  const form: FormState | null = status ? {
    steps: status.defaults.steps ?? 800,
    saveEvery: status.defaults.saveEvery ?? 100,
    // Rank follows the recommendation for the same reason as the base: at the
    // default 256 nothing fits below ~24 GB, so a 16 GB card would open on a
    // red 'will not fit' form with the fix two fields away and unstated.
    rank: status.recommended?.rank ?? status.defaults.rank ?? 256,
    alpha: status.defaults.alpha ?? 256,
    lr: status.defaults.lr ?? 8e-5,
    maxFrames: status.defaults.maxFrames ?? 1500,
    cropMode: (status.defaults.cropMode as 'random' | 'beginning' | 'structured') ?? 'structured',
    cropStartFrac: status.defaults.cropStartFrac ?? 0.55,
    cropEndFrac: status.defaults.cropEndFrac ?? 0.15,
    cropStartTiles: status.defaults.cropStartTiles ?? 3,
    depthLossWeight: status.defaults.depthLossWeight ?? 1.0,
    depthLossFrames: status.defaults.depthLossFrames ?? 128,
    optimizer: status.defaults.optimizer ?? 'prodigy',
    muonLrScale: status.defaults.muonLrScale ?? 64,
    adapterType: status.defaults.adapterType ?? 'lora',
    lokrFactor: status.defaults.lokrFactor ?? 6,
    sharedCaption: status.sharedCaption ?? '',
    gradAccum: status.defaults.gradAccum ?? 1,
    seed: status.defaults.seed ?? 42,
    trigger: trigger ?? '',
    triggerPrepend: status.defaults.triggerPrepend !== false,
    // The RECOMMENDED base, not the global default: the default was chosen on a
    // 32 GB card and on a 12 GB one it is simply wrong. The server picks the
    // highest-fidelity base that fits THIS GPU, and falls back to the default
    // when it cannot read the card.
    // The configured default WINS over the VRAM recommender's pick. The
    // recommender walks bases by fidelity and falls through to a smaller one
    // whenever the best does not fit its budget, which silently downgraded
    // f16 to q8_0. It still sets `rank` and still raises overBudget, so the
    // user is warned rather than quietly given a different base.
    basePrecision: status.defaults.basePrecision || status.recommended?.base || 'f16',
    holdout: status.defaults.holdout ?? 0.15,
    evalEvery: status.defaults.evalEvery ?? 50,
    cropAnchor: (status.defaults.cropAnchor as 'song' | 'zero') ?? 'song',
    // Previews default OFF. They are the fastest way to learn whether a run is
    // worth finishing, but each one costs about a minute, so opting in is the
    // user's call rather than a surprise on the clock.
    // Cadence follows checkpoints (Rob, 2026-08-25): a preview per checkpoint,
    // no minutes clock. `||` on purpose — a server default of 0 ("off") falls
    // back to the checkpoint cadence rather than disabling previews.
    previewEverySteps: status.defaults.previewEverySteps || (status.defaults.saveEvery ?? 50),
    previewEveryMinutes: 0,
    previewSeconds: 40,
    previewSeed: 424242,
    previewCaption: '',
    // Off by default (Rob, 2026-08-25): with a preview at every checkpoint the
    // control and baseline renders would double the pause cost for takes that
    // rarely get listened to. Both remain a checkbox away.
    previewControl: false,
    previewBaseline: false,
    previewScaleMlp: 1.0,
    // Prior preservation is off until a corpus is chosen: it needs a second
    // dataset the user has to nominate, and defaulting it on would silently
    // train a different objective than the form otherwise describes.
    regDatasetId: '',
    regEvery: status.defaults.regularisation?.every ?? 3,
    regTopK: status.defaults.regularisation?.topK ?? 64,
    ...edits,
  } : null;

  // Peak VRAM for the CURRENT form, not for the defaults — rank is the biggest
  // single term after the base itself, so an estimate pinned to the defaults
  // would be wrong for exactly the users who need it most.
  const chosen = status?.bases?.find(b => b.id === form?.basePrecision);
  const peak = (() => {
    if (!form || !status?.vramModel || !chosen) return null;
    const mb    = estimateMm3PeakMb(chosen.bytes, form.rank, form.maxFrames, status.vramModel,
                                    form.optimizer)
                + (chosen.extraMb || 0);
    const total = status.gpuTotalMb || 0;
    const gb    = (mb / 1024).toFixed(1);
    // 0 means the engine could not be read, NOT a card with no memory. Show the
    // estimate without a verdict rather than inventing a scary one.
    if (total <= 0) {
      return { text: t('trainingStudio.mm3.peakUnknown', { gb }), tone: 'text-zinc-500' };
    }
    const totalGb = (total / 1024).toFixed(1);
    if (mb + 1536 <= total) {
      return { text: t('trainingStudio.mm3.peakFits', { gb, totalGb }), tone: 'text-emerald-500' };
    }
    if (mb <= total) {
      // Fits on paper, with nothing left for the desktop. This is the state that
      // produced 12-14 s/step instead of 3.7 in the f16 A/B, so it is a warning
      // rather than an error.
      return { text: t('trainingStudio.mm3.peakTight', { gb, totalGb }), tone: 'text-amber-500' };
    }
    // "Pick a smaller base" is bad advice when the server already established
    // that nothing in the catalogue fits at any rank on the ladder.
    if (status.recommended?.overBudget) {
      return { text: t('trainingStudio.mm3.peakNoFit', { gb, totalGb }), tone: 'text-rose-500' };
    }
    return { text: t('trainingStudio.mm3.peakOver', { gb, totalGb }), tone: 'text-rose-500' };
  })();

  const startTrain = async () => {
    if (!form) return;
    setBusy(true);
    try {
      const body: Mm3TrainLmRequest = {
        steps: form.steps, saveEvery: form.saveEvery, rank: form.rank, alpha: form.alpha,
        lr: form.lr, maxFrames: form.maxFrames, cropMode: form.cropMode,
        cropStartFrac: form.cropStartFrac, cropEndFrac: form.cropEndFrac,
        cropStartTiles: form.cropStartTiles,
        depthLossWeight: form.depthLossWeight, depthLossFrames: form.depthLossFrames,
        optimizer: form.optimizer, muonLrScale: form.muonLrScale,
        adapterType: form.adapterType, lokrFactor: form.lokrFactor,
        sharedCaption: form.sharedCaption,
        gradAccum: form.gradAccum, seed: form.seed,
        basePrecision: form.basePrecision, holdout: form.holdout, evalEvery: form.evalEvery,
        cropAnchor: form.cropAnchor,
        ...(form.trigger.trim()
          ? { trigger: form.trigger.trim(), triggerPrepend: form.triggerPrepend }
          : {}),
        ...(form.regDatasetId ? {
          regularisation: {
            datasetId: form.regDatasetId,
            every: form.regEvery,
            topK: form.regTopK,
          },
        } : {}),
        ...(form.previewEverySteps > 0 || form.previewEveryMinutes > 0 ? {
          preview: {
            everySteps: form.previewEverySteps,
            everyMinutes: form.previewEveryMinutes,
            seconds: form.previewSeconds,
            seed: form.previewSeed,
            control: form.previewControl,
            baseline: form.previewBaseline,
            scaleMlp: form.previewScaleMlp,
            ...(form.previewCaption.trim() ? { caption: form.previewCaption.trim() } : {}),
          },
        } : {}),
      };
      await startMm3TrainLm(body);
    } finally {
      setBusy(false);
    }
  };

  if (!status && !statusError) {
    return (
      <div className="flex items-center justify-center py-20 text-zinc-500 text-sm">
        <Loader2 size={18} className="animate-spin mr-2" /> …
      </div>
    );
  }

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) =>
    setEdits(e => ({ ...e, [k]: v }));

  const hasCodes = (status?.codes ?? 0) > 0;
  const trainBlocked = (status?.missingForTrain.length ?? 0) > 0;

  return (
    <div className="flex flex-col gap-4">
      {(statusError || storeError) && (
        <div className="rounded-xl border border-red-500/25 bg-red-500/10 p-3 flex items-start gap-2 text-sm text-red-500">
          <XCircle size={16} className="mt-0.5 flex-shrink-0" />
          <span className="min-w-0 break-words">{statusError || storeError}</span>
        </div>
      )}

      {/* ── LM LoRA ── */}
      <div className={CARD}>
        <div className="flex items-center gap-2 mb-2">
          <Cpu size={15} className="text-amber-500" />
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
            {t('trainingStudio.mm3.trainTitle', 'LM LoRA training')}
          </h3>
        </div>
        <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
          {t('trainingStudio.mm3.trainBlurb',
            'Trains the planner LM on this dataset. Checkpoints are written straight to the MiniMax-Music3 '
            + 'adapter folder, so they appear in the generation panel\'s adapter picker as soon as they are '
            + 'saved — there is no install step. The engine is paused for the run.')}
        </p>

        {trainBlocked ? (
          <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
            <span>
              {t('trainingStudio.mm3.missing', 'Missing model files')}: {status?.missingForTrain.join(', ')}
            </span>
          </div>
        ) : !hasCodes ? (
          <div className="flex flex-col items-start gap-3">
            <div className="flex items-start gap-2 text-xs text-zinc-500">
              <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
              {t('trainingStudio.mm3.needsCodes',
                'Export the RVQ codes first — training reads them, not the audio.')}
            </div>
            <button
              onClick={() => setPhase('preprocess')}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-500/15 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/25 transition-colors"
            >
              {t('trainingStudio.mm3.goToCodes', 'Go to Codes')}
            </button>
          </div>
        ) : form && (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <NumField label={t('trainingStudio.mm3.steps', 'Steps')} value={form.steps}
                onChange={v => set('steps', v)} />
              <NumField label={t('trainingStudio.mm3.saveEvery', 'Checkpoint every')} value={form.saveEvery}
                onChange={v => set('saveEvery', v)} />
              <NumField label={t('trainingStudio.mm3.rank', 'Rank')} value={form.rank}
                onChange={v => set('rank', v)} />
              <NumField label={t('trainingStudio.mm3.maxFrames', 'Crop (frames)')} value={form.maxFrames}
                onChange={v => set('maxFrames', v)} step={50} />
              <NumField label={t('trainingStudio.mm3.cropStartFrac', 'Crops at song start')}
                value={form.cropStartFrac} onChange={v => set('cropStartFrac', v)} step={0.05}
                hint={t('trainingStudio.mm3.cropStartFracHint',
                  'Share anchored at frame 0 — what teaches songs to OPEN like songs. '
                  + 'Too low and renders jump in mid-flow.') as string} />
              <NumField label={t('trainingStudio.mm3.cropStartTiles', 'Start tiles')}
                value={form.cropStartTiles} onChange={v => set('cropStartTiles', v)} step={1}
                hint={t('trainingStudio.mm3.cropStartTilesHint',
                  'Half the start share stays at frame 0; the rest lands on aligned tiles '
                  + 'after it, teaching the intro→build→verse arc under short crops. '
                  + '1 = frame 0 only.') as string} />
              <NumField label={t('trainingStudio.mm3.cropEndFrac', 'Crops at song end')}
                value={form.cropEndFrac} onChange={v => set('cropEndFrac', v)} step={0.05}
                hint={t('trainingStudio.mm3.cropEndFracHint',
                  'Share flush to the track end — the only place EOS is taught. The '
                  + 'REMAINDER of these two is the random share (currently '
                  + `${Math.max(0, Math.round((1 - form.cropStartFrac - form.cropEndFrac) * 100))}% mid-song crops).`) as string} />
              <NumField label={t('trainingStudio.mm3.depthLossWeight', 'Acoustic loss weight')}
                value={form.depthLossWeight} onChange={v => set('depthLossWeight', v)} step={0.1}
                hint={t('trainingStudio.mm3.depthLossWeightHint',
                  'Trains the adapter to keep vocal timbre intact: acoustic codebooks are '
                  + 'supervised through the frozen depth decoder. 0 disables — renders then '
                  + 'drift into chipmunk/goblin voices. Leave at 1.') as string} />
              <NumField label={t('trainingStudio.mm3.depthLossFrames', 'Acoustic frames/step')}
                value={form.depthLossFrames} onChange={v => set('depthLossFrames', v)} step={16}
                hint={t('trainingStudio.mm3.depthLossFramesHint',
                  'Frames sampled per step for the acoustic loss.') as string} />
            </div>
            <label className="flex flex-col gap-1 mt-3">
              <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                {t('trainingStudio.mm3.trigger', 'Trigger word')}
              </span>
              <input className={INPUT} value={form.trigger} onChange={e => set('trigger', e.target.value)} />
            </label>
            {/* A sibling, not a child: a label nested inside a label is invalid
                and clicking the checkbox would focus the text input instead. */}
            <label className="flex items-start gap-2 mt-2 text-[11px] text-zinc-600 dark:text-zinc-300">
              <input type="checkbox" className="mt-0.5" checked={form.triggerPrepend}
                onChange={e => set('triggerPrepend', e.target.checked)}
                disabled={!form.trigger.trim()} />
              <span>
                {t('trainingStudio.mm3.triggerPrepend', 'Train the trigger')}
                <span className="block text-[10px] text-zinc-500">
                  {t('trainingStudio.mm3.triggerPrependHint',
                    'Puts "trigger, " at the front of every training caption, in memory — your files '
                    + 'are not touched. Leave this ON. With it off the word is only recorded in the '
                    + 'adapter sidecar and never learned, so typing it at render time bolts an unseen '
                    + 'token sequence onto your prompt and makes the result WORSE, not better.')}
                </span>
              </span>
            </label>

            {/* -- Previews ------------------------------------------------
                Above Advanced on purpose: this is the control that decides
                whether the user finds out at step 200 or at step 800 that the
                run is going the wrong way. */}
            <div className="mt-3 rounded-lg border border-zinc-200 dark:border-white/10 p-3">
              <div className="flex items-center gap-2">
                <Volume2 size={13} className="text-amber-500" />
                <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                  {t('trainingStudio.mm3.previewTitle', 'Audio previews during training')}
                </span>
              </div>
              <p className="text-[10px] text-zinc-500 leading-snug mt-1">
                {t('trainingStudio.mm3.previewBlurb',
                  'Renders a sample from the checkpoint while the run is still going, so you can hear '
                  + 'whether it is heading the right way and abort if not. Each preview point pauses '
                  + 'training for about a minute: the trainer has to hand the whole card to the render, '
                  + 'so it saves its optimizer state, exits, renders, and resumes exactly where it was. '
                  + 'Zero in both fields turns previews off.')}
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-2">
                <NumField label={t('trainingStudio.mm3.previewEverySteps', 'Every N steps')}
                  value={form.previewEverySteps} onChange={v => set('previewEverySteps', v)}
                  step={50} hint={t('trainingStudio.mm3.previewStepsHint', '0 = off') as string} />
                <NumField label={t('trainingStudio.mm3.previewEveryMinutes', 'Or every N minutes')}
                  value={form.previewEveryMinutes} onChange={v => set('previewEveryMinutes', v)}
                  step={5} hint={t('trainingStudio.mm3.previewMinutesHint',
                    'Whichever comes first') as string} />
                <NumField label={t('trainingStudio.mm3.previewSeconds', 'Length (s)')}
                  value={form.previewSeconds} onChange={v => set('previewSeconds', v)}
                  step={4} hint={t('trainingStudio.mm3.previewSecondsHint',
                    '24 s costs about 16 s of GPU') as string} />
                <NumField label={t('trainingStudio.mm3.previewSeed', 'Preview seed')}
                  value={form.previewSeed} onChange={v => set('previewSeed', v)}
                  hint={t('trainingStudio.mm3.previewSeedHint',
                    'Fixed across the run') as string} />
                <NumField label={t('trainingStudio.mm3.previewScaleMlp', 'Preview MLP scale')}
                  value={form.previewScaleMlp} onChange={v => set('previewScaleMlp', v)}
                  step={0.05} hint={t('trainingStudio.mm3.previewScaleMlpHint',
                    'How hard the adapter’s MLP delta is applied in previews only. '
                    + '1 = full, 0 = attention only.') as string} />
              </div>
              <label className="flex flex-col gap-1 mt-3">
                <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                  {t('trainingStudio.mm3.previewCaption', 'Preview caption')}
                </span>
                <textarea
                  className={`${INPUT} font-mono text-[11px] leading-snug`} rows={3}
                  placeholder={t('trainingStudio.mm3.previewCaptionPlaceholder',
                    'Blank = the first held-out song’s own caption, with the trigger prepended') as string}
                  value={form.previewCaption}
                  onChange={e => set('previewCaption', e.target.value)}
                />
                <span className="text-[10px] text-zinc-500 leading-snug">
                  {t('trainingStudio.mm3.previewCaptionHint',
                    'Leave blank unless you have a reason not to: a held-out caption carries the '
                    + 'artist’s true BPM and tuning, and a wrong one becomes audible as soon as '
                    + 'identity bakes in. The trigger is prepended as "trigger, " on the caption’s '
                    + 'first line, which is the exact shape the training rows use.')}
                </span>
              </label>
              <div className="flex flex-col gap-1.5 mt-3">
                <label className="flex items-start gap-2 text-[11px] text-zinc-600 dark:text-zinc-300">
                  <input type="checkbox" className="mt-0.5" checked={form.previewControl}
                    onChange={e => set('previewControl', e.target.checked)} />
                  <span>
                    {t('trainingStudio.mm3.previewControl', 'Also render a neutral control caption')}
                    <span className="block text-[10px] text-zinc-500">
                      {t('trainingStudio.mm3.previewControlHint',
                        'An off-genre prompt rendered WITH the adapter. This is the one that catches '
                        + 'the adapter damaging the base planner - on the artist caption a damaged '
                        + 'model and a good one both sound roughly like the artist.')}
                    </span>
                  </span>
                </label>
                <label className="flex items-start gap-2 text-[11px] text-zinc-600 dark:text-zinc-300">
                  <input type="checkbox" className="mt-0.5" checked={form.previewBaseline}
                    onChange={e => set('previewBaseline', e.target.checked)} />
                  <span>
                    {t('trainingStudio.mm3.previewBaseline', 'Render a no-adapter reference first')}
                    <span className="block text-[10px] text-zinc-500">
                      {t('trainingStudio.mm3.previewBaselineHint',
                        'Free - the engine is still up before the run starts. Without it there is '
                        + 'nothing to judge "worse than base" against.')}
                    </span>
                  </span>
                </label>
              </div>
            </div>

            {/* -- Prior preservation -------------------------------------- */}
            <div className="mt-3 rounded-lg border border-zinc-200 dark:border-white/10 p-3">
              <div className="flex items-center gap-2">
                <ShieldCheck size={13} className="text-amber-500" />
                <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">
                  {t('trainingStudio.mm3.regTitle', 'Prior preservation')}
                </span>
              </div>
              <p className="text-[10px] text-zinc-500 leading-snug mt-1">
                {t('trainingStudio.mm3.regBlurb',
                  'Spends some steps on an UNRELATED dataset, scored against what the base model '
                  + 'itself predicted there rather than against that music. The adapter is then '
                  + 'penalised for changing its mind about material that has nothing to do with the '
                  + 'artist, which is the only thing in the objective that separates "learned the '
                  + 'voice" from "rewrote the planner". Leave the corpus unset to turn it off.')}
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-2">
                <label className="flex flex-col gap-1 md:col-span-1">
                  <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                    {t('trainingStudio.mm3.regDataset', 'Corpus')}
                  </span>
                  <select className={INPUT} value={form.regDatasetId}
                    onChange={e => set('regDatasetId', e.target.value)}>
                    <option value="">{t('trainingStudio.mm3.regOff', 'Off')}</option>
                    {(status?.regCandidates ?? []).map(d => (
                      <option key={d.id} value={d.id}>{d.name} ({d.songs})</option>
                    ))}
                  </select>
                  <span className="text-[10px] text-zinc-500 leading-snug">
                    {(status?.regCandidates?.length ?? 0) === 0
                      ? t('trainingStudio.mm3.regNone',
                          'No other dataset has RVQ codes yet — a regularisation corpus needs exactly '
                          + 'what a training corpus needs. Run the codes export on one.')
                      : t('trainingStudio.mm3.regDatasetHint',
                          'Pick something with nothing in common with this artist. Its lyrics may be '
                          + 'empty; only its captions and codes are used.')}
                  </span>
                </label>
                <NumField label={t('trainingStudio.mm3.regEvery', 'Every N steps')}
                  value={form.regEvery} onChange={v => set('regEvery', v)}
                  hint={t('trainingStudio.mm3.regEveryHint',
                    '3 = one prior step per two style steps') as string} />
                <NumField label={t('trainingStudio.mm3.regTopK', 'Classes kept')}
                  value={form.regTopK} onChange={v => set('regTopK', v)} step={64}
                  hint={t('trainingStudio.mm3.regTopKHint',
                    'Coverage: 64 = 90%, 128 = 94%, 256 = 97%') as string} />
              </div>
              {form.regDatasetId && (
                <div className="mt-2 text-[10px] text-amber-600/90 dark:text-amber-400/90 leading-snug">
                  {t('trainingStudio.mm3.regDilution',
                    'This dilutes style exposure: at every {{n}} steps only {{s}} of your {{total}} '
                    + 'steps train on the artist. Raise Steps to about {{want}} to keep the same '
                    + 'exposure as running without it.',
                    {
                      n: form.regEvery,
                      s: form.steps - Math.floor(form.steps / Math.max(2, form.regEvery)),
                      total: form.steps,
                      want: Math.round(form.steps / (1 - 1 / Math.max(2, form.regEvery))),
                    })}
                </div>
              )}
            </div>

            <button
              onClick={() => setAdvanced(v => !v)}
              className="flex items-center gap-1 mt-3 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
            >
              {advanced ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              {t('trainingStudio.mm3.advanced', 'Advanced')}
            </button>

            {advanced && (
              <div className="mt-3 pl-3 border-l-2 border-zinc-200 dark:border-white/10 flex flex-col gap-3">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <NumField label={t('trainingStudio.mm3.alpha', 'Alpha')} value={form.alpha}
                    onChange={v => set('alpha', v)} />
                  <NumField label={t('trainingStudio.mm3.lr', 'Learning rate')} value={form.lr}
                    onChange={v => set('lr', v)} step={1e-5} />
                  <NumField label={t('trainingStudio.mm3.gradAccum', 'Grad accum')} value={form.gradAccum}
                    onChange={v => set('gradAccum', v)} />
                  <NumField label={t('trainingStudio.mm3.seed', 'Seed')} value={form.seed}
                    onChange={v => set('seed', v)} />
                </div>
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                    {t('trainingStudio.mm3.sharedCaption', 'Dataset-wide caption')}
                  </span>
                  <textarea className={INPUT} rows={4} value={form.sharedCaption}
                    placeholder={'artist name, album name, genre, guitar character, vocal character, '
                      + 'rhythm section, production character, tempo, structure'}
                    onChange={e => set('sharedCaption', e.target.value)} />
                  <span className="text-[10px] text-zinc-500 leading-snug">
                    {t('trainingStudio.mm3.sharedCaptionHint',
                      'ONE caption used for every track, and the single biggest quality lever there '
                      + 'is. With the caption held constant across rows the adapter has nowhere to '
                      + 'put the style except into itself, and the caption becomes the handle that '
                      + 'summons the album. Start with the artist name so it doubles as the trigger. '
                      + 'Aim for 60-80 tokens of comma-separated descriptors. Saved to '
                      + '_shared-caption.txt beside the dataset. LEAVE BLANK to use per-song '
                      + '.mm3.txt captions instead — but then every track needs one, and tracks '
                      + 'without are skipped.')}
                  </span>
                </label>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                      {t('trainingStudio.mm3.optimizer', 'Optimizer')}
                    </span>
                    <select className={INPUT} value={form.optimizer}
                      onChange={e => set('optimizer', e.target.value as 'muon' | 'adamw' | 'prodigy')}>
                      <option value="prodigy">Prodigy (sets its own LR)</option>
                      <option value="adamw">AdamW</option>
                      <option value="muon">Muon</option>
                    </select>
                    <span className="text-[10px] text-zinc-500 leading-snug">
                      {t('trainingStudio.mm3.optimizerHint',
                        'Prodigy estimates its own step size, so the learning rate below becomes a '
                        + 'schedule multiplier only. On Green Day it converged to 8.19e-5 against the '
                        + '8e-5 tuned by hand. It costs two extra state buffers (about +2.7 GB at rank '
                        + '128) and CANNOT resume, so it is unavailable when mid-training previews are '
                        + 'on. AdamW matches the published SimpleTuner recipe. Muon is NOT recommended: '
                        + 'at the default LR scale of 64 it produced an adapter that rendered digital '
                        + 'silence, because that value was tuned on a different model.')}
                    </span>
                  </label>
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                      {t('trainingStudio.mm3.adapterType', 'Adapter type')}
                    </span>
                    <select className={INPUT} value={form.adapterType}
                      onChange={e => set('adapterType', e.target.value as 'lora' | 'lokr')}>
                      <option value="lokr">LoKr</option>
                      <option value="lora">LoRA</option>
                    </select>
                    <span className="text-[10px] text-zinc-500 leading-snug">
                      {t('trainingStudio.mm3.adapterTypeHint',
                        'LoKr trains a Kronecker pair per slot instead of a low-rank pair, for a '
                        + 'smaller file: about 528 MB at factor 6 against a rank-128 LoRA\'s 1.4 GB. '
                        + 'NOT YET VALIDATED BY EAR — no LoKr adapter has been auditioned, so treat a '
                        + 'LoKr run as an experiment. FACTOR is the size knob, not dim: factor 16 gives '
                        + 'only 27M parameters, fewer than rank 64, and rank 64 was the setting that '
                        + 'turned lyrics to gibberish.')}
                    </span>
                  </label>
                  {form.adapterType === 'lokr' && (
                    <NumField label={t('trainingStudio.mm3.lokrFactor', 'LoKr factor')}
                      value={form.lokrFactor} onChange={v => set('lokrFactor', v)}
                      hint={t('trainingStudio.mm3.lokrFactorHint',
                        '6 gives ~528 MB and 264M parameters. Higher is smaller and less capable: '
                        + '8 -> 274 MB, 16 -> 109 MB.') as string} />
                  )}
                  {form.optimizer === 'muon' && (
                    <NumField label={t('trainingStudio.mm3.muonLrScale', 'Muon LR scale')}
                      value={form.muonLrScale} onChange={v => set('muonLrScale', v)}
                      hint={t('trainingStudio.mm3.muonLrScaleHint',
                        '64 is the best of the values measured so far, not a tuned optimum.') as string} />
                  )}
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="flex flex-col gap-1">
                    <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                      {t('trainingStudio.mm3.base', 'Base precision')}
                    </span>
                    <select className={INPUT} value={form.basePrecision}
                      onChange={e => set('basePrecision', e.target.value)}>
                      {(status?.bases ?? []).map(b => (
                        <option key={b.id} value={b.id}>
                          {b.id} — {(b.bytes / 1073741824).toFixed(1)} GB
                          {b.quality === 'poor' ? ' (not recommended)' : ''}
                          {b.id === status?.recommended?.base ? ' \u2713' : ''}
                        </option>
                      ))}
                    </select>
                    {peak && (
                      <span className={`text-[10px] leading-snug ${peak.tone}`}>{peak.text}</span>
                    )}
                    <span className="text-[10px] text-zinc-500 leading-snug">
                      {chosen?.quality === 'poor'
                        ? t('trainingStudio.mm3.basePoor',
                            'This base is too lossy to train against: measured +14% on the first-step loss '
                            + 'and roughly double the gradient norm, i.e. the quantizer injects more error '
                            + 'than the adapter is being asked to learn. Prefer the smallest base marked '
                            + 'good or better that fits.')
                        : t('trainingStudio.mm3.baseHint',
                            'Every base trains the same way — the frozen weights are dequantized in-graph, '
                            + 'so only VRAM and fidelity change, and step time barely moves (within ~5% '
                            + 'across the whole range). Pick the smallest one that comfortably fits your '
                            + 'card, then spend what is left on LoRA rank.')}
                    </span>
                  </label>
                  <NumField label={t('trainingStudio.mm3.holdout', 'Hold-out fraction')}
                    value={form.holdout} onChange={v => set('holdout', v)} step={0.05}
                    hint={t('trainingStudio.mm3.holdoutHint',
                      '0 disables evaluation — the training loss then cannot tell learning from '
                      + 'memorising. Ignored below 6 songs.') as string} />
                </div>
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                    {t('trainingStudio.mm3.cropMode', 'Crop mode')}
                  </span>
                  <select className={INPUT} value={form.cropMode}
                    onChange={e => set('cropMode', e.target.value as 'random' | 'beginning' | 'structured')}>
                    <option value="structured">structured</option>
                    <option value="random">random</option>
                    <option value="beginning">beginning</option>
                  </select>
                  <span className={`text-[10px] leading-snug ${
                    form.cropMode !== 'structured'
                      ? 'text-amber-600/80 dark:text-amber-400/80' : 'text-zinc-500'
                  }`}>
                    {form.cropMode === 'random'
                      ? t('trainingStudio.mm3.cropModeRandom',
                          '`random` hands the model a prompt followed straight by mid-song audio with '
                          + 'no history in front of it, and supervises it — so it learns that a song may '
                          + 'begin at any position. It also leaves the ending to chance: EOS is only '
                          + 'supervised by a crop that reaches the track end, which on a 3-minute song '
                          + 'is under 3% of steps. Renders begin mid-flow and never resolve.')
                      : form.cropMode === 'beginning'
                        ? t('trainingStudio.mm3.cropModeIntros',
                            '`beginning` takes every track from frame 0, so every supervised position '
                            + 'has the song’s real history in front of it. What it can never teach is an '
                            + 'ending: EOS is only supervised by a crop that reaches the track end, and '
                            + 'it never does unless the whole song fits in the window.')
                        : t('trainingStudio.mm3.cropModeStructured',
                            '`structured` anchors most crops at frame 0, so every supervised position '
                            + 'carries the song’s real history exactly as it will at generation time, '
                            + 'and pins the rest flush to the track end — the only place EOS is '
                            + 'supervised. This is the default and the two others are each broken at '
                            + 'one end.')}
                  </span>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                    {t('trainingStudio.mm3.cropAnchor', 'Crop position')}
                  </span>
                  <select className={INPUT} value={form.cropAnchor}
                    onChange={e => set('cropAnchor', e.target.value as 'song' | 'zero')}>
                    <option value="song">song (true position)</option>
                    <option value="zero">zero (legacy)</option>
                  </select>
                  <span className="text-[10px] text-zinc-500 leading-snug">
                    {t('trainingStudio.mm3.cropAnchorHint',
                      'Under "zero" every crop was presented to the model as if it were the opening of '
                      + 'the song, whatever part of the track it came from - while generation always '
                      + 'starts at frame 0. That mismatch teaches the model that a song can begin '
                      + 'anywhere, and shows up as sound arriving instantly at 0:00 and as tempo '
                      + 'drifting mid-track. "song" labels each crop with where it actually is. Runs '
                      + 'trained under the two are not comparable.')}
                  </span>
                </label>
              </div>
            )}

            <button
              onClick={() => void startTrain()}
              disabled={busy || jobRunning}
              className="mt-4 px-4 py-2 rounded-lg text-sm font-semibold bg-amber-500 text-black hover:bg-amber-400 disabled:opacity-40 transition-colors flex items-center gap-2"
            >
              <Play size={14} />
              {t('trainingStudio.mm3.start', 'Start training')}
            </button>
          </>
        )}
      </div>

      {/* ── Live run: the shared job machinery, unchanged ── */}
      {mine && activeJob && (
        <div className={CARD}>
          <JobProgress />
          {jobKind === 'mm3-train-lm' && mm3Live && (
            // The run-stats row. MM3 has no epochs, so none of the ACE tiles
            // apply — these are the numbers that actually say what the run is
            // doing, and STEP TIME is the one that exposes a VRAM spill (about
            // 4 s when it fits, ~10x that when it pages to host memory).
            <div className="mt-3 grid grid-cols-2 md:grid-cols-5 gap-2">
              {[
                { k: 'loss', v: mm3Live.loss ? mm3Live.loss.toFixed(4) : '—' },
                { k: 'gradNorm', v: mm3Live.gradNorm ? mm3Live.gradNorm.toFixed(3) : '—' },
                // The SCHEDULE value is not what Muon applies: its update is
                // normalised, so the effective rate is lr x muon-lr-scale.
                // Showing 3.4e-5 while the optimizer used 2.2e-3 was a quietly
                // misleading tile.
                {
                  k: 'lr',
                  v: mm3Live.lr
                    ? (lrScale > 1
                      ? `${(mm3Live.lr * lrScale).toExponential(2)}`
                      : mm3Live.lr.toExponential(2))
                    : '—',
                },
                { k: 'stepTime', v: mm3Live.stepMs ? `${(mm3Live.stepMs / 1000).toFixed(1)}s` : '—' },
                {
                  k: 'vram',
                  v: mm3Live.totalMb
                    ? `${Math.round(mm3Live.usedMb / 1024)}/${Math.round(mm3Live.totalMb / 1024)} GB`
                    : '—',
                  warn: mm3Live.totalMb > 0 && mm3Live.usedMb > mm3Live.totalMb - 512,
                },
              ].map(tile => (
                <div key={tile.k}
                  className="rounded-lg border border-zinc-200 dark:border-white/5 px-2.5 py-1.5">
                  <div className="text-[10px] uppercase tracking-wider text-zinc-500">
                    {t(`trainingStudio.mm3.stat.${tile.k}`, tile.k)}
                  </div>
                  <div className={`text-sm font-semibold tabular-nums ${
                    (tile as { warn?: boolean }).warn ? 'text-amber-500' : 'text-zinc-800 dark:text-zinc-200'
                  }`}>{tile.v}</div>
                </div>
              ))}
            </div>
          )}
          {jobKind === 'mm3-train-lm' && <Mm3PreviewStrip />}
          {jobKind === 'mm3-train-lm' && trainStepSeries.length > 1 && (
            <div className="mt-3">
              {/* Four series now, on one fractional-epoch axis: per-step noise,
                  the epoch mean, its 5-epoch average, and — the one that
                  matters — held-out loss. No target line: MM3 has no
                  target-loss auto-stop. */}
              <TrainingChart
                epochs={trainLmEpochs}
                steps={trainStepSeries}
                milestones={trainMilestones}
                evals={trainEvalSeries}
                target={0}
                maxEpochs={trainMaxEpochs}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Mm3TrainCard;
