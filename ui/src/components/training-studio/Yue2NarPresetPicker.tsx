// Yue2NarPresetPicker.tsx — the YuE2 NAR recipe chips.
// Presets are derived from the step/snapshot fields, so a hand edit falls back
// to Custom instead of misreporting a preset.

import React from 'react';
import { useTranslation } from 'react-i18next';
import type { Yue2PresetName } from '../../services/trainingApi';
import { PRESET_ORDER } from './useYue2NarForm';
import { BTN_SM } from './yue2CardUi';

const ACTIVE = ' !border-amber-500 !text-amber-600 dark:!text-amber-400 !bg-amber-500/10';

export const Yue2NarPresetPicker: React.FC<{
  presets: Record<Yue2PresetName, { steps: number; saveEvery: number }>;
  activePreset: Yue2PresetName | 'custom';
  onPick: (preset: Yue2PresetName) => void;
  disabled?: boolean;
}> = ({ presets, activePreset, onPick, disabled }) => {
  const { t } = useTranslation();

  return (
    <div className="mb-3">
      <div className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider mb-1">
        {t('trainingStudio.yue2.preset', 'Recipe')}
      </div>
      <div className="flex flex-wrap gap-2">
        {PRESET_ORDER.map(p => (
          <button key={p} type="button" onClick={() => onPick(p)} disabled={disabled}
            className={BTN_SM + (activePreset === p ? ACTIVE : '')}>
            {t(`trainingStudio.yue2.preset.${p}`, p)}
            <span className="ml-1 text-zinc-500">{presets[p].steps}</span>
          </button>
        ))}
        {activePreset === 'custom' && (
          <span className={BTN_SM + ACTIVE + ' cursor-default'}>
            {t('trainingStudio.yue2.preset.custom', 'Custom')}
          </span>
        )}
      </div>
      <p className="text-[11px] text-zinc-500 mt-1">
        {t('trainingStudio.yue2.presetInfo',
          'Step count is the only lever measured here, so it is the only thing the presets '
          + 'move. At the measured 0.065 s per step the wall clock is close to linear: 10,000 '
          + 'steps is about 12 minutes and 20,000 about 23.')}
      </p>
    </div>
  );
};
