import { YUE2_JOINT_LORA_STOP } from '../../contracts/trainingRecipes.js';
import type { ResolveYue2PresetRequest, ResolveYue2PresetResult } from '../../contracts/preferences.js';

/** Apply a raw browser or document preset without changing its stored body. */
export function resolveYue2Preset(input: ResolveYue2PresetRequest): ResolveYue2PresetResult {
  const { preset, currentForm, lyricTiming } = input;
  return {
    effectiveForm: { ...currentForm, adapterType: 'lora', ...YUE2_JOINT_LORA_STOP, cautious: false, ...preset.settings },
    lyricTiming: preset.version === 2 && typeof preset.settings.lyricTiming === 'boolean'
      ? preset.settings.lyricTiming : lyricTiming,
  };
}
