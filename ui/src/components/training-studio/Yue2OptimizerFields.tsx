import { useTranslation } from 'react-i18next';
import type { Yue2OptimOptions } from '../../services/trainingApi';

const input = 'w-full rounded border border-zinc-300 dark:border-white/10 bg-white dark:bg-zinc-900 px-2 py-1 text-sm text-zinc-800 dark:text-zinc-200';

export function Yue2OptimizerFields({ value, onChange }: {
  value: Yue2OptimOptions;
  onChange: (patch: Partial<Yue2OptimOptions>) => void;
}) {
  const { t } = useTranslation();
  return <div className="flex flex-col gap-2 text-xs text-zinc-600 dark:text-zinc-400">
    <label>{t('trainingStudio.yue2Optim.optimizer', 'Optimizer')}
      <select className={input} value={value.optimizer}
        onChange={e => onChange({ optimizer: e.target.value as Yue2OptimOptions['optimizer'] })}>
        <option value="prodigy">Prodigy</option>
        <option value="adamw">AdamW</option>
        <option value="muon">Muon</option>
      </select>
    </label>
    {value.optimizer === 'prodigy' && <>
      <span>{t('trainingStudio.yue2Optim.prodigyHint', 'Learns the effective learning rate. Warmup and schedule still apply; the base multiplier is 1.0. Uses two extra optimizer buffers compared with AdamW.')}</span>
      <label>{t('trainingStudio.yue2Optim.d0', 'Initial step estimate (d0)')}
        <input className={input} type="number" min={1e-12} step={1e-6} value={value.prodigyD0}
          onChange={e => onChange({ prodigyD0: Number(e.target.value) })} />
      </label>
    </>}
    {value.optimizer === 'muon' && <>
      <label>{t('trainingStudio.yue2Optim.muonScale', 'Muon learning-rate scale')}
        <input className={input} type="number" min={0.001} step={1} value={value.muonLrScale}
          onChange={e => onChange({ muonLrScale: Number(e.target.value) })} />
      </label>
      <label>{t('trainingStudio.yue2Optim.muonSteps', 'Orthogonalization iterations')}
        <input className={input} type="number" min={1} max={20} step={1} value={value.muonNsSteps}
          onChange={e => onChange({ muonNsSteps: Number(e.target.value) })} />
      </label>
      <span>{t('trainingStudio.yue2Optim.muonHint', 'Matrices with a short side below 16 use AdamW. The training log reports how many tensors use Muon.')}</span>
    </>}
  </div>;
}
