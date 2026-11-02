// Yue2NarTrainCard.tsx — YuE2 stage 4: a LoRA on the frozen NAR half.
// Reads `status` as a prop from the one useYue2Status() call in
// Yue2TrainStages; the server payload is the only home of the measured recipe.
//
// The card is chrome only: the recipe form lives in useYue2NarForm, the
// override grid in Yue2NarAdvancedPanel, the recipe chips in
// Yue2NarPresetPicker and the running-job view in Yue2NarLiveRun.

import { AlertTriangle, ChevronDown, ChevronRight, Cpu, Play } from 'lucide-react';
import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useActiveTrainingJob } from '../../queries/training-runs/hooks';
import type { Yue2Status, Yue2TrainRequest } from '../../services/trainingApi';
import { assertRecipeWorker, resolveTrainingRecipe } from '../../services/trainingRecipesApi';
import { useTrainingStore } from '../../stores/trainingStore';
import { formatDurationMs } from '../../utils/trainingEta';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { gb, NumField } from './TrainingNumberField';
import { Yue2NarAdvancedPanel } from './Yue2NarAdvancedPanel';
import { Yue2NarLiveRun } from './Yue2NarLiveRun';
import { Yue2NarPresetPicker } from './Yue2NarPresetPicker';
import { Yue2RunsList } from './Yue2RunsList';
import { useYue2NarForm } from './useYue2NarForm';
import { BTN_GO, CARD, INPUT } from './yue2CardUi';

/** Stage 4: the NAR LoRA. `status` arrives as a prop from ONE useYue2Status()
 *  call in Yue2TrainStages, which also owns the loading gate, the licence
 *  banner and the status-fetch error banner — sharing the fetch across the
 *  five stages, rather than each stage re-requesting the same payload, is the
 *  whole reason this card no longer calls the hook itself. */
export const Yue2NarTrainCard: React.FC<{
  datasetId: string; trigger?: string; status: Yue2Status | null; reload: () => void;
}> = ({ datasetId, trigger, status, reload }) => {
  const { t } = useTranslation();
  const activeJob = useActiveTrainingJob();
  const startYue2Train = useTrainingStore(s => s.startYue2Train);
  const yue2Live = useTrainingStore(s => s.yue2Live);
  const trainStepSeries = useTrainingStore(s => s.trainStepSeries);
  const trainMilestones = useTrainingStore(s => s.trainMilestones);

  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);

  const { base, form, set, reset, setEdits, presets, activePreset, applyPreset, peak } =
    useYue2NarForm(status, trigger);

  const jobStatus = activeJob?.status;
  const jobRunning = jobStatus === 'queued' || jobStatus === 'running';

  const startTrain = async () => {
    if (!form) return;
    setBusy(true);
    try {
      const recipe = await resolveTrainingRecipe<Record<string, unknown>>('yue2-nar', { ...form, activePreset });
      const body = recipe.execution as Yue2TrainRequest;
      assertRecipeWorker(recipe);
      await startYue2Train(body);
    } catch (err) {
      useTrainingStore.setState({ error: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusy(false);
      reload();
    }
  };

  const trainBlocked = (status?.missingForTrain.length ?? 0) > 0;
  const clips = status?.cache?.clips ?? 0;
  const tensors = form && status ? status.targetTensors[form.target] : 0;

  return (
    <div className="flex flex-col gap-4">
      {/* ── NAR LoRA ── */}
      <div className={CARD}>
        <div className="flex items-center gap-2 mb-2">
          <Cpu size={15} className="text-amber-500" />
          <h3 className="text-sm font-semibold text-zinc-900 dark:text-white">
            {t('trainingStudio.yue2.trainTitle', 'NAR LoRA training')}
          </h3>
        </div>
        <p className="text-[11px] text-zinc-500 leading-relaxed mb-3">
          {t('trainingStudio.yue2.trainBlurb',
            'Trains a LoRA on the frozen NAR half of the YuE2 LM from the cached latents. Snapshots are '
            + 'written as plain safetensors files you can load straight into the YuE2 adapter field — '
            + 'there is no install step. The engine is paused for the run.')}
        </p>

        {trainBlocked ? (
          <div className="flex items-start gap-2 text-xs text-amber-600 dark:text-amber-400">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
            <span>
              {t('trainingStudio.yue2.missing', 'Missing model files')}: {status?.missingForTrain.join(', ')}
            </span>
          </div>
        ) : clips <= 0 ? (
          <div className="flex items-start gap-2 text-xs text-zinc-500">
            <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
            {t('trainingStudio.yue2.needsLatents',
              'Encode the latents first — training reads the cache, not the audio.')}
          </div>
        ) : form && base && status && (
          <>
            {presets && (
              <Yue2NarPresetPicker presets={presets} activePreset={activePreset}
                onPick={applyPreset} disabled={busy} />
            )}

            <label className="flex flex-col gap-1 mb-3">
              <ParamLabel
                label={t('trainingStudio.yue2.trigger', 'Trigger word')}
                className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                info={t('trainingStudio.yue2.triggerInfo',
                  'Put in front of every training caption, and the only handle the trained style has at '
                  + 'generation time. Required: without one the adapter has no word to address it by, and '
                  + 'the run is refused rather than warned about twelve minutes in.') as string}
              />
              <input className={INPUT} value={form.trigger}
                onChange={e => set('trigger', e.target.value)} />
            </label>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <NumField label={t('trainingStudio.yue2.steps', 'Steps')} value={form.steps}
                onChange={v => set('steps', v)} step={1000}
                defaultValue={base.steps} onReset={() => reset('steps')}
                hint={formatDurationMs(form.steps * status.vramModel.secondsPerStep * 1000)}
                info={t('trainingStudio.yue2.stepsInfo',
                  'How many training steps to run. Step count is the only lever measured for this trainer, '
                  + 'so it is the only thing the Recipe presets move. More steps mean more training time '
                  + 'and, past a point, a risk of overcooking the adapter; fewer steps finish faster but '
                  + 'may undertrain.') as string} />
              <NumField label={t('trainingStudio.yue2.rank', 'Rank')} value={form.rank}
                onChange={v => set('rank', v)} step={64}
                defaultValue={base.rank} onReset={() => reset('rank')}
                info={t('trainingStudio.yue2.rankHint',
                  'How much capacity the LoRA has. 256 is what was measured; 128 saves about 2.6 GB of '
                  + 'VRAM at the cost of some capacity, and the ladder between them, plus upstream\'s '
                  + 'default of 16, have not been heard here. Higher rank can capture more detail but '
                  + 'costs more VRAM and a larger adapter file.') as string} />
              <NumField label={t('trainingStudio.yue2.saveEvery', 'Snapshot every')}
                value={form.saveEvery} onChange={v => set('saveEvery', v)} step={500}
                defaultValue={base.saveEvery} onReset={() => reset('saveEvery')}
                info={t('trainingStudio.yue2.saveEveryHint',
                  'How often, in steps, a checkpoint snapshot is written to disk. These are the rungs on '
                  + 'the ladder — and where a killed run comes back from. Lower means more frequent '
                  + 'snapshots to pick from, at the cost of more disk writes.') as string} />
              <label className="flex flex-col gap-1">
                <ParamLabel
                  label={t('trainingStudio.yue2.target', 'Trained sites')}
                  className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                  meta={t('trainingStudio.yue2.targetMeta', '{{n}} exported tensors', { n: tensors }) as string}
                  info={t('trainingStudio.yue2.targetHint',
                    'Which weights inside the NAR half the LoRA touches. nar_attn trains only attention; '
                    + 'nar_attn_mlp adds the feed-forward layers on all 28 NAR blocks — 196 sites — and is '
                    + 'what the measured runs used; nar_attn_mlp_proj adds the projection layers on top of '
                    + 'that. Training more sites gives the adapter more places to change the sound, at the '
                    + 'cost of a larger file and more VRAM.') as string}
                  onReset={form.target !== base.target ? () => reset('target') : undefined}
                />
                <StyledSelect
                  accent="amber"
                  value={form.target}
                  onChange={v => set('target', v)}
                  className="w-full"
                  options={[
                    { value: 'nar_attn', label: 'nar_attn', hint: t('trainingStudio.yue2.targetAttnHint', 'Attention only.') as string },
                    { value: 'nar_attn_mlp', label: 'nar_attn_mlp', hint: t('trainingStudio.yue2.targetAttnMlpHint', 'Attention plus the FFN on all 28 NAR blocks — what the measured runs used.') as string },
                    { value: 'nar_attn_mlp_proj', label: 'nar_attn_mlp_proj', hint: t('trainingStudio.yue2.targetAttnMlpProjHint', 'nar_attn_mlp plus the projection layers.') as string },
                  ]}
                />
              </label>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
              <label className="flex flex-col gap-1">
                <ParamLabel
                  label={t('trainingStudio.yue2.base', 'Base model')}
                  className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
                  info={t('trainingStudio.yue2.baseHint',
                    'Which YuE2 LM checkpoint the LoRA trains against; the same base must be loaded at '
                    + 'generation time for the adapter to apply. Only bf16 has been trained on here. '
                    + 'Whether a quantized base trains usefully at all is not established, so the others '
                    + 'are offered without a ranking.') as string}
                />
                <StyledSelect
                  accent="amber"
                  value={form.lmType}
                  onChange={v => set('lmType', v)}
                  className="w-full"
                  options={status.bases.length > 0
                    ? status.bases.map(b => ({
                        value: b.id,
                        label: `${b.id}${b.proven ? ` — ${t('trainingStudio.yue2.baseProven', 'measured')}` : ''} · ${gb(b.bytes)}`,
                      }))
                    : [{ value: form.lmType, label: form.lmType }]}
                />
              </label>
              <div className="flex flex-col gap-1">
                <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">
                  {t('trainingStudio.yue2.cost', 'What this costs')}
                </span>
                {peak && <span className={`text-sm font-semibold ${peak.tone}`}>{peak.text}</span>}
                <span className="text-[10px] text-zinc-500 leading-snug">
                  {t('trainingStudio.yue2.costHint',
                    'Measured on one card: 17.0 GB at rank 128 and 19.6 GB at rank 256, the roughly 7 GB '
                    + 'bf16 base included in both. {{clips}} clips in the cache; 10,000 steps at rank 256 '
                    + 'took about 12 minutes.', { clips })}
                </span>
              </div>
            </div>

            <button
              onClick={() => setAdvanced(v => !v)}
              className="flex items-center gap-1 mt-3 text-[11px] text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors"
            >
              {advanced ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              {t('trainingStudio.yue2.advanced', 'Advanced')}
            </button>

            {advanced && (
              <Yue2NarAdvancedPanel form={form} base={base} status={status}
                set={set} reset={reset}
                onOptimizerPatch={patch => setEdits(p => ({ ...p, ...patch }))} />
            )}

            <button
              onClick={() => void startTrain()}
              disabled={busy || jobRunning || !form.trigger.trim()}
              className={`mt-4 ${BTN_GO}`}
            >
              <Play size={14} />
              {t('trainingStudio.yue2.start', 'Start training')}
            </button>
          </>
        )}
      </div>

      {/* ── Previous runs and their ladders ── */}
      <Yue2RunsList datasetId={datasetId} reloadKey={`${activeJob?.id ?? ''}:${jobStatus ?? ''}`} />

      {/* ── Live run: the shared job machinery, unchanged ── */}
      <Yue2NarLiveRun job={activeJob} live={yue2Live}
        stepSeries={trainStepSeries} milestones={trainMilestones} />
    </div>
  );
};
