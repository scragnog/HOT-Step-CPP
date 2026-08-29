// PreprocessOptionsForm.tsx — base-model pickers + the advanced drawer
//
// The four always-visible controls are the ones that change the tensors'
// identity (which base they belong to, and how much audio is encoded); the
// rest live behind the Advanced drawer because their defaults are the ones
// HOT-Step inference itself uses.

import React from 'react';
import { Info } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { StyledSelect } from '../shared/StyledSelect';
import type {
  PreprocessCompat,
  PreprocessDtype,
  PreprocessNormalize,
  TrainingCapabilities,
} from '../../services/trainingApi';

/** Everything the panel needs to POST, with every default already resolved. */
export interface PreprocessFormState {
  ditModel: string;
  vaeModel: string;
  textEncoder: string;
  maxDuration: number;
  normalize: PreprocessNormalize;
  targetDb: number;
  dtype: PreprocessDtype;
  compat: PreprocessCompat;
  maxCaptionTokens: number;
  maxLyricTokens: number;
  vaeChunk: number;
  vaeOverlap: number;
  overwrite: boolean;
  stopEngine: boolean;
}

export const PREPROCESS_DEFAULTS: PreprocessFormState = {
  ditModel: '',
  vaeModel: '',
  textEncoder: '',
  maxDuration: 600,
  normalize: 'peak',
  targetDb: -1.0,
  dtype: 'f32',
  compat: 'hotstep',
  // Raised from Side-Step's 256/512 (2026-07-30) — every caption in every
  // dataset was being truncated. See the train-preprocess route for the
  // reasoning and docs/plans/2026-07-30-conditioning-token-caps.md for the
  // measurements and how to roll back.
  maxCaptionTokens: 512,
  maxLyricTokens: 2048,
  vaeChunk: 384,
  vaeOverlap: 48,
  overwrite: false,
  stopEngine: true,
};

/** BF16 bases are the only valid training targets — quantized ones are not. */
export const BF16_RE = /bf16/i;

const FIELD =
  'rounded-lg px-3 py-2 text-sm bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 focus:outline-none focus:border-amber-500 disabled:opacity-50 disabled:cursor-not-allowed';
const LABEL = 'text-xs font-semibold text-zinc-600 dark:text-zinc-400';

/** BF16 names first — they are what the picker should land on. */
function orderForTraining(names: string[]): string[] {
  return [...names.filter(n => BF16_RE.test(n)), ...names.filter(n => !BF16_RE.test(n))];
}

interface Props {
  capabilities: TrainingCapabilities | null;
  value: PreprocessFormState;
  onChange: (patch: Partial<PreprocessFormState>) => void;
  disabled?: boolean;
}

export const PreprocessOptionsForm: React.FC<Props> = ({ capabilities, value, onChange, disabled }) => {
  const { t } = useTranslation();

  const pp = capabilities?.preprocess;
  const ditModels = orderForTraining(pp?.ditModels ?? []);
  const vaeModels = orderForTraining(pp?.vaeModels ?? []);
  const textEncoders = pp?.textEncoders ?? [];

  // The snapshot survives the engine being stopped (the server caches it), so
  // the pickers only go dead when it has never been probed at all.
  const noModelSource = capabilities?.engine.up === false && (pp?.modelsCachedAt ?? 0) === 0;
  const lock = !!disabled || noModelSource;

  const num = (raw: string, fallback: number): number => {
    const n = Number(raw);
    return Number.isFinite(n) ? n : fallback;
  };

  /** BF16 bases carry the "training ready" note; everything else is bare. */
  const modelOptions = (names: string[], current: string) => [
    ...(current === '' ? [{ value: '', label: '—' }] : []),
    ...names.map(name => ({
      value: name,
      label: name,
      hint: BF16_RE.test(name) ? t('trainingStudio.preprocess.trainingReady') : undefined,
    })),
  ];

  return (
    <div className="flex flex-col gap-4">
      {noModelSource && (
        <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-amber-500/25 bg-amber-500/10 text-xs text-amber-600 dark:text-amber-400">
          <Info size={13} className="mt-0.5 flex-shrink-0" />
          {t('trainingStudio.preprocess.engineDownForModels')}
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div className="flex flex-col gap-1.5">
          <span className={LABEL}>{t('trainingStudio.preprocess.baseModel')}</span>
          <StyledSelect
            accent="amber"
            value={value.ditModel}
            disabled={lock}
            onChange={(v) => onChange({ ditModel: v })}
            options={modelOptions(ditModels, value.ditModel)}
            searchPlaceholder="Filter models…"
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <span className={LABEL}>{t('trainingStudio.preprocess.vae')}</span>
          <StyledSelect
            accent="amber"
            value={value.vaeModel}
            disabled={lock}
            onChange={(v) => onChange({ vaeModel: v })}
            options={modelOptions(vaeModels, value.vaeModel)}
            searchPlaceholder="Filter models…"
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <span className={LABEL}>{t('trainingStudio.preprocess.textEncoder')}</span>
          <StyledSelect
            accent="amber"
            value={value.textEncoder}
            disabled={lock}
            onChange={(v) => onChange({ textEncoder: v })}
            options={modelOptions(textEncoders, value.textEncoder)}
            searchPlaceholder="Filter models…"
          />
        </div>

        <label className="flex flex-col gap-1.5">
          <span className={LABEL}>{t('trainingStudio.preprocess.maxDuration')}</span>
          <input
            type="number"
            min={0}
            step={10}
            value={value.maxDuration}
            disabled={!!disabled}
            onChange={(e) => onChange({ maxDuration: num(e.target.value, 600) })}
            className={FIELD}
          />
        </label>
      </div>

      <details className="rounded-lg border border-zinc-200 dark:border-white/5 px-3 py-2">
        <summary className="cursor-pointer text-xs font-semibold text-zinc-600 dark:text-zinc-400 select-none">
          {t('trainingStudio.preprocess.advanced')}
        </summary>

        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.preprocess.normalize')}</span>
            <StyledSelect
              accent="amber"
              value={value.normalize}
              disabled={!!disabled}
              onChange={(v) => onChange({ normalize: v as PreprocessNormalize })}
              options={[
                { value: 'peak', label: 'peak' },
                { value: 'none', label: 'none' },
              ]}
            />
          </div>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.preprocess.targetDb')}</span>
            <input
              type="number"
              min={-60}
              max={0}
              step={0.5}
              value={value.targetDb}
              disabled={!!disabled || value.normalize === 'none'}
              onChange={(e) => onChange({ targetDb: num(e.target.value, -1) })}
              className={FIELD}
            />
          </label>

          <div className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.preprocess.dtype')}</span>
            <StyledSelect
              accent="amber"
              value={value.dtype}
              disabled={!!disabled}
              onChange={(v) => onChange({ dtype: v as PreprocessDtype })}
              options={[
                { value: 'f32', label: 'f32' },
                { value: 'bf16', label: 'bf16' },
              ]}
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.preprocess.compat')}</span>
            <StyledSelect
              accent="amber"
              value={value.compat}
              disabled={!!disabled}
              onChange={(v) => onChange({ compat: v as PreprocessCompat })}
              options={[
                { value: 'hotstep', label: 'HOT-Step' },
                { value: 'sidestep', label: 'Side-Step parity (debug)' },
              ]}
            />
          </div>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.preprocess.maxCaptionTokens')}</span>
            <input
              type="number"
              min={16}
              max={4096}
              step={16}
              value={value.maxCaptionTokens}
              disabled={!!disabled}
              onChange={(e) => onChange({ maxCaptionTokens: num(e.target.value, 512) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.preprocess.maxLyricTokens')}</span>
            <input
              type="number"
              min={16}
              max={4096}
              step={16}
              value={value.maxLyricTokens}
              disabled={!!disabled}
              onChange={(e) => onChange({ maxLyricTokens: num(e.target.value, 2048) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.preprocess.vaeChunk')}</span>
            <input
              type="number"
              min={64}
              step={64}
              value={value.vaeChunk}
              disabled={!!disabled}
              onChange={(e) => onChange({ vaeChunk: num(e.target.value, 384) })}
              className={FIELD}
            />
          </label>

          <label className="flex flex-col gap-1.5">
            <span className={LABEL}>{t('trainingStudio.preprocess.vaeOverlap')}</span>
            <input
              type="number"
              min={0}
              step={8}
              value={value.vaeOverlap}
              disabled={!!disabled}
              onChange={(e) => onChange({ vaeOverlap: num(e.target.value, 64) })}
              className={FIELD}
            />
          </label>
        </div>

        <div className="mt-3 flex flex-col gap-2">
          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <input
              type="checkbox"
              checked={value.overwrite}
              disabled={!!disabled}
              onChange={(e) => onChange({ overwrite: e.target.checked })}
              className="accent-amber-500"
            />
            {t('trainingStudio.preprocess.overwrite')}
          </label>

          <label className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
            <input
              type="checkbox"
              checked={value.stopEngine}
              disabled={!!disabled}
              onChange={(e) => onChange({ stopEngine: e.target.checked })}
              className="accent-amber-500"
            />
            {t('trainingStudio.preprocess.stopEngine')}
          </label>
          <span className="text-[11px] text-zinc-500 pl-6">{t('trainingStudio.preprocess.stopEngineHelp')}</span>
        </div>
      </details>
    </div>
  );
};

export default PreprocessOptionsForm;
