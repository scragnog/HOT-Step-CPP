// Yue2NarAdvancedPanel.tsx — the advanced half of the YuE2 NAR recipe form.
// Every control here is an override of a measured default; the panel is split
// out so the card itself reads as name, cost and the one Start button.

import React from 'react';
import { useTranslation } from 'react-i18next';
import type { Yue2Status } from '../../services/trainingApi';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { NumField } from './TrainingNumberField';
import { Yue2OptimizerFields } from './Yue2OptimizerFields';
import type { TrainForm } from './useYue2NarForm';

export const Yue2NarAdvancedPanel: React.FC<{
  form: TrainForm;
  base: TrainForm;
  status: Yue2Status;
  set: <K extends keyof TrainForm>(key: K, value: TrainForm[K]) => void;
  reset: (key: keyof TrainForm) => void;
  onOptimizerPatch: (patch: Partial<TrainForm>) => void;
}> = ({ form, base, status, set, reset, onOptimizerPatch }) => {
  const { t } = useTranslation();

  return (
    <div className="mt-3 pl-3 border-l-2 border-zinc-200 dark:border-white/10 flex flex-col gap-3">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <NumField label={t('trainingStudio.yue2.alpha', 'Alpha')} value={form.alpha}
          onChange={v => set('alpha', v)} step={64}
          defaultValue={base.alpha} onReset={() => reset('alpha')}
          info={t('trainingStudio.yue2.alphaInfo',
            'Scales how strongly the LoRA weights apply relative to the rank. Raising it makes '
            + 'the adapter\'s effect stronger at the same rank; lowering it weakens it.') as string} />
        <Yue2OptimizerFields value={form} onChange={onOptimizerPatch}
          defaults={base} onReset={reset} />
        {form.optimizer !== 'prodigy' && (<NumField label={t('trainingStudio.yue2.lr', 'Learning rate')} value={form.lr}
          onChange={v => set('lr', v)} step={1e-5}
          defaultValue={base.lr} onReset={() => reset('lr')}
          info={t('trainingStudio.yue2.lrInfo',
            'How big a step the optimizer takes on each update. Higher trains faster but risks '
            + 'instability; lower is steadier but slower to converge. Only used when the '
            + 'optimizer above is not Prodigy, which sets its own effective rate automatically.') as string} />)}
        <label className="flex flex-col gap-1">
          <ParamLabel
            label={t('trainingStudio.yue2.lrScheduler', 'Schedule')}
            className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
            info={t('trainingStudio.yue2.lrSchedulerInfo',
              'How the learning rate moves over the run. cosine decays it from the starting '
              + 'rate down toward zero as training progresses; constant holds it steady for the '
              + 'whole run.') as string}
            onReset={form.lrScheduler !== base.lrScheduler ? () => reset('lrScheduler') : undefined}
          />
          <StyledSelect
            accent="amber"
            value={form.lrScheduler}
            onChange={v => set('lrScheduler', v)}
            className="w-full"
            options={[
              { value: 'cosine', label: 'cosine' },
              { value: 'constant', label: 'constant' },
            ]}
          />
        </label>
        <NumField label={t('trainingStudio.yue2.warmup', 'Warmup steps')} value={form.warmup}
          onChange={v => set('warmup', v)} step={10}
          defaultValue={base.warmup} onReset={() => reset('warmup')}
          info={t('trainingStudio.yue2.warmupInfo',
            'How many steps the learning rate ramps up from zero at the start of the run, before '
            + 'the schedule above takes over. More warmup steps ease into training more gently; '
            + 'fewer reach the full rate sooner but risk an unstable start.') as string} />
        <NumField label={t('trainingStudio.yue2.gradAccum', 'Grad accum')} value={form.gradAccum}
          onChange={v => set('gradAccum', v)}
          defaultValue={base.gradAccum} onReset={() => reset('gradAccum')}
          info={t('trainingStudio.yue2.gradAccumInfo',
            'How many steps of gradients get summed before the optimizer applies an update, '
            + 'which behaves like a larger batch without the extra VRAM. Raising it smooths '
            + 'updates but makes each optimizer step rarer; 1 updates every step.') as string} />
        <NumField label={t('trainingStudio.yue2.maxGradNorm', 'Clip grad norm')}
          value={form.maxGradNorm} onChange={v => set('maxGradNorm', v)} step={0.1}
          defaultValue={base.maxGradNorm} onReset={() => reset('maxGradNorm')}
          info={t('trainingStudio.yue2.maxGradNormInfo',
            'Caps how large the gradient is allowed to get on any one step, cutting it back to '
            + 'this length if it is bigger. Guards against a spike destabilising the run; too '
            + 'low a cap can slow learning by blunting every real update as well.') as string} />
        <NumField label={t('trainingStudio.yue2.weightDecay', 'Weight decay')}
          value={form.weightDecay} onChange={v => set('weightDecay', v)} step={0.01}
          defaultValue={base.weightDecay} onReset={() => reset('weightDecay')}
          info={t('trainingStudio.yue2.weightDecayInfo',
            'Shrinks the adapter\'s weights a little on every step, independent of the loss. '
            + 'Higher discourages large weights, which can reduce overfitting but also blunt '
            + 'the adapter\'s learned effect; 0 turns it off.') as string} />
        <NumField label={t('trainingStudio.yue2.seed', 'Seed')} value={form.seed}
          onChange={v => set('seed', v)}
          defaultValue={base.seed} onReset={() => reset('seed')}
          info={t('trainingStudio.yue2.seedInfo',
            'Seeds the random draws that choose clip order and dropout during training. Changing '
            + 'it changes which clips and dropout draws land on which step, not the training '
            + 'recipe itself; the same seed with the same recipe reproduces the same run.') as string} />
        <NumField label={t('trainingStudio.yue2.captionDropout', 'Caption dropout')}
          value={form.captionDropout} onChange={v => set('captionDropout', v)} step={0.05}
          defaultValue={base.captionDropout} onReset={() => reset('captionDropout')}
          info={t('trainingStudio.yue2.captionDropoutHint',
            'Chance of swapping in the EMPTY style prefix instead of the real caption on a given '
            + 'step. 0 is not "more likeness": it is the trigger word ceasing to mean anything '
            + 'relative to no trigger word. Raising it trains the trigger to matter more; too '
            + 'high leaves too little signal from the real captions.') as string} />
        <NumField label={t('trainingStudio.yue2.abcDropout', 'ABC dropout')}
          value={form.abcDropout} onChange={v => set('abcDropout', v)} step={0.05}
          defaultValue={base.abcDropout} onReset={() => reset('abcDropout')}
          info={t('trainingStudio.yue2.abcDropoutHint',
            'Chance a clip whose source has a lead sheet trains cot=off instead of cot=full this '
            + 'draw. A source with no lead sheet always trains cot=off. 0.5 is upstream\'s own '
            + 'split; raising it trains more like a source with no lead sheet, lowering it leans '
            + 'more on cot=full.') as string} />
        <NumField label={t('trainingStudio.yue2.kvCache', 'K/V canvases')} value={form.kvCache}
          onChange={v => set('kvCache', v)}
          defaultValue={base.kvCache} onReset={() => reset('kvCache')}
          info={t('trainingStudio.yue2.kvCacheHint',
            'How many AR-prefix caches are held at once, about 104 MB each at 10 s clips. A real '
            + 'VRAM knob: raising it uses more VRAM but can reuse more cached prefixes; 8 is what '
            + 'was measured.') as string} />
        <NumField label={t('trainingStudio.yue2.clipBlock', 'Clip block')} value={form.clipBlock}
          onChange={v => set('clipBlock', v)}
          defaultValue={base.clipBlock} onReset={() => reset('clipBlock')}
          info={t('trainingStudio.yue2.clipBlockHint',
            'How many steps one working set of clips is held for before the next block is drawn. '
            + 'With per-clip codec ids every clip is its own conditioning, so drawing from the '
            + 'whole album misses the K/V cache almost every step and re-runs the AR prefix. '
            + 'Raising it reuses the cache longer per block; 0 draws from everything.') as string} />
        <NumField label={t('trainingStudio.yue2.logEvery', 'Log every')} value={form.logEvery}
          onChange={v => set('logEvery', v)}
          defaultValue={base.logEvery} onReset={() => reset('logEvery')}
          info={t('trainingStudio.yue2.logEveryInfo',
            'How often, in steps, the trainer writes a log line with its metrics. Only affects '
            + 'how closely you can watch progress, not the training itself.') as string} />
        <label className="flex flex-col gap-1">
          <ParamLabel
            label={t('trainingStudio.yue2.tSampling', 'Timestep sampling')}
            className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider"
            info={t('trainingStudio.yue2.tSamplingInfo',
              'Which denoising timesteps get sampled during training. logit-normal biases '
              + 'sampling toward the middle of the noise range; uniform samples every timestep '
              + 'equally.') as string}
            onReset={form.tSampling !== base.tSampling ? () => reset('tSampling') : undefined}
          />
          <StyledSelect
            accent="amber"
            value={form.tSampling}
            onChange={v => set('tSampling', v)}
            className="w-full"
            options={[
              { value: 'logit-normal', label: 'logit-normal' },
              { value: 'uniform', label: 'uniform' },
            ]}
          />
        </label>
      </div>
      <p className="text-[11px] text-zinc-500 leading-snug">
        {status.defaults.deletesResumeStateOnFinish
          ? t('trainingStudio.yue2.resumeStateNote',
              'The optimizer state is deleted when a run exports cleanly — it is the run\'s '
              + 'middle, not its result. A run that stops early keeps it, and the engine will '
              + 'only continue from it if the rank, alpha, trained sites, trigger, seed and '
              + 'grad-accum all match, on this same machine and build.')
          : ''}
      </p>
    </div>
  );
};
