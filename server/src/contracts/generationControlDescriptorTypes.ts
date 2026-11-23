import type { GENERATION_DEFAULTS } from './generationControls.js';

type Defaults = typeof GENERATION_DEFAULTS;
export type CoreGenerationParamKey = keyof Defaults;

/** Wire metadata for the core controls. Arrays and objects use custom editors. */
export interface CoreGenerationParamDescriptor {
  key: CoreGenerationParamKey;
  type: 'slider' | 'select' | 'toggle' | 'text' | 'array' | 'object';
  label: string;
  default: Defaults[CoreGenerationParamKey];
  hint?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string; label: string }[];
}
