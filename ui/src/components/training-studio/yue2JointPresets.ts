import type { Yue2JointTrainRequest } from '../../services/trainingApi';

export const YUE2_JOINT_PRESETS_KEY = 'hs-yue2-joint-presets';

export type Yue2JointPreset = {
  version?: 2;
  name: string;
  settings: Partial<Yue2JointTrainRequest>;
};

export function readYue2JointPresets(): Yue2JointPreset[] {
  try {
    const value = JSON.parse(window.localStorage.getItem(YUE2_JOINT_PRESETS_KEY) || '[]') as unknown;
    return Array.isArray(value) ? value.filter((item): item is Yue2JointPreset =>
      !!item && typeof item === 'object' && typeof item.name === 'string'
      && !!item.settings && typeof item.settings === 'object') : [];
  } catch { return []; }
}
