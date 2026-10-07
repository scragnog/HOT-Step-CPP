import type { BackendExtensionParam } from '../backends/types.js';
import { generationIntentSchema } from '../../contracts/generation.js';
import { resolveGenerationParams } from './defaults.js';

export class GenerationIntentError extends Error {}

function validExtension(value: unknown, extension: BackendExtensionParam): boolean {
  if (extension.type === 'toggle') return typeof value === 'boolean';
  if (extension.type === 'text') return typeof value === 'string';
  if (extension.type === 'select') return typeof value === 'string'
    && (extension.options ?? []).some(option => option.value === value);
  return typeof value === 'number' && Number.isFinite(value)
    && (extension.min === undefined || value >= extension.min)
    && (extension.max === undefined || value <= extension.max);
}

/** Resolve compact UI state with the active engine's declared extension policy. */
export function resolveGenerationIntent(
  intent: unknown,
  backendId: string,
  extensions: readonly BackendExtensionParam[],
): Record<string, unknown> {
  const parsed = generationIntentSchema.safeParse(intent);
  if (!parsed.success) throw new GenerationIntentError(`Invalid generation intent: ${parsed.error.message}`);
  const state = { ...parsed.data.params };
  const backendParams = state.backendParams;
  if (backendParams !== undefined && (backendParams === null || typeof backendParams !== 'object' || Array.isArray(backendParams))) {
    throw new GenerationIntentError('backendParams must be an object');
  }
  const declared = new Map(extensions.map(extension => [extension.key, extension]));
  const values: Record<string, unknown> = {};
  for (const [key, value] of Object.entries((backendParams ?? {}) as Record<string, unknown>)) {
    const extension = declared.get(key);
    if (!extension) throw new GenerationIntentError(`Unknown backend parameter: ${key}`);
    if (!validExtension(value, extension)) throw new GenerationIntentError(`Invalid backend parameter: ${key}`);
    values[key] = value;
  }
  for (const extension of extensions) {
    if (values[extension.key] === undefined && extension.default !== undefined) {
      values[extension.key] = extension.default;
    }
  }
  state.backendParams = values;
  return { ...resolveGenerationParams(state, { backendId, settings: parsed.data.settings }), ...(parsed.data.input ?? {}) };
}
