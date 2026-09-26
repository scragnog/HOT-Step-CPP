import { useTranslation } from 'react-i18next';
import type { Yue2OptimOptions } from '../../services/trainingApi';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { ParamLabel } from '../shared/ParamLabel';

const input = 'w-full rounded border border-zinc-300 dark:border-white/10 bg-white dark:bg-zinc-900 px-2 py-1 text-sm text-zinc-800 dark:text-zinc-200';

export function Yue2OptimizerFields({ value, onChange, joint = false }: {
  value: Yue2OptimOptions;
  onChange: (patch: Partial<Yue2OptimOptions>) => void;
  /** Joint trainer only: exposes AdamW on the LmOptim path and the cautious
   *  update mask. The legacy trainers reject both, so they never see them. */
  joint?: boolean;
}) {
  const { t } = useTranslation();
  const lmOptim = value.optimizer !== 'adamw';
  return <div className="flex flex-col gap-2 text-xs text-zinc-600 dark:text-zinc-400">
    <label className="flex flex-col gap-1">
      <ParamLabel
        label={t('trainingStudio.yue2Optim.optimizer', 'Optimizer')}
        info={t('trainingStudio.yue2Optim.optimizerInfo', "Which algorithm updates the adapter weights each step. Prodigy learns its own step size as it trains, so the learning-rate field is ignored, and is the tested default. AdamW uses a fixed learning rate and, in the joint trainer, a native 8-bit kernel. Muon orthogonalizes weight updates for matrices with a short side of 16 or more; smaller matrices fall back to AdamW.")}
      />
      <StyledSelect
        accent="amber"
        value={value.optimizer}
        onChange={(optimizer) => onChange(optimizer === 'adamw' ? { optimizer, cautious: false } : { optimizer })}
        options={[
          { value: 'prodigy' as const, label: 'Prodigy' },
          { value: 'adamw' as const, label: 'AdamW' },
          ...(joint ? [{ value: 'adamw-lm' as const, label: t('trainingStudio.yue2Optim.adamwLm', 'AdamW (graph, experimental)') }] : []),
          { value: 'muon' as const, label: 'Muon' },
        ]}
        className="w-full"
      />
    </label>
    {joint && value.optimizer === 'adamw-lm' && <span>{t('trainingStudio.yue2Optim.adamwLmInfo', "Runs AdamW on the same fp32, warmup-then-cosine graph optimizer Prodigy and Muon use, instead of the 8-bit kernel the plain AdamW option uses, so its numbers differ from plain AdamW. It is what lets cautious updates apply.")}</span>}
    {joint && lmOptim && <Toggle
      accent="amber"
      checked={value.cautious === true}
      onChange={(cautious) => onChange({ cautious })}
      label={t('trainingStudio.yue2Optim.cautious', 'Cautious updates (experimental)')}
      info={t('trainingStudio.yue2Optim.cautiousInfo', "Zeroes each weight update where it disagrees in sign with the gradient and rescales the rest to keep the step size. Off: every update applies as computed.")}
    />}
    {joint && lmOptim && value.cautious && <span>{t('trainingStudio.yue2Optim.cautiousActiveHint', 'On: fewer contradictory updates, at the cost of some step size. Untested for quality here; the shipped Prodigy preset trains with this on.')}</span>}
    {value.optimizer === 'prodigy' && <>
      <span>{t('trainingStudio.yue2Optim.prodigyHint', 'Learns the effective learning rate. Warmup and schedule still apply; the base multiplier is 1.0. Uses two extra optimizer buffers compared with AdamW.')}</span>
      <label className="flex flex-col gap-1">
        <ParamLabel
          label={t('trainingStudio.yue2Optim.d0', 'Initial step estimate (d0)')}
          info={t('trainingStudio.yue2Optim.d0Info', "Prodigy's starting guess for its own step size, before it adapts. Raising it lets Prodigy start with larger updates and reach its adapted rate sooner; lowering it starts more cautiously. Prodigy corrects the estimate as training runs either way, so this mostly affects the first steps.")}
        />
        <input className={input} type="number" min={1e-12} step={1e-6} value={value.prodigyD0}
          onChange={e => onChange({ prodigyD0: Number(e.target.value) })} />
      </label>
    </>}
    {value.optimizer === 'muon' && <>
      <label className="flex flex-col gap-1">
        <ParamLabel
          label={t('trainingStudio.yue2Optim.muonScale', 'Muon learning-rate scale')}
          info={t('trainingStudio.yue2Optim.muonScaleInfo', "Multiplier on the base learning rate for the tensors Muon updates. Raising it makes those updates bigger and training move faster, at greater risk of instability; lowering it trains those tensors more slowly and cautiously. Only affects tensors large enough for Muon; smaller ones use AdamW at the unscaled rate.")}
        />
        <input className={input} type="number" min={0.001} step={1} value={value.muonLrScale}
          onChange={e => onChange({ muonLrScale: Number(e.target.value) })} />
      </label>
      <label className="flex flex-col gap-1">
        <ParamLabel
          label={t('trainingStudio.yue2Optim.muonSteps', 'Orthogonalization iterations')}
          info={t('trainingStudio.yue2Optim.muonStepsInfo', "How many Newton-Schulz iterations Muon runs to orthogonalize each weight update. Raising it gets closer to a fully orthogonal update at the cost of more compute per step; lowering it is cheaper but a coarser approximation.")}
        />
        <input className={input} type="number" min={1} max={20} step={1} value={value.muonNsSteps}
          onChange={e => onChange({ muonNsSteps: Number(e.target.value) })} />
      </label>
      <span>{t('trainingStudio.yue2Optim.muonHint', 'Matrices with a short side below 16 use AdamW. The training log reports how many tensors use Muon.')}</span>
    </>}
  </div>;
}
