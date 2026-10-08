import type { TrainingRecipeFamily } from '../../../server/src/contracts/trainingRecipes';
import type { TrainingWorkerRef } from '../../../server/src/contracts/trainingOperation';
import { currentWorkerRef, snapshotFor, trainingOperation } from './trainingOperations';

export interface TrainingRecipe<T = Record<string, unknown>> {
  recipeVersion: number;
  family: TrainingRecipeFamily;
  builtin: T;
  stored: Record<string, unknown>;
  resolved?: T;
  execution?: Record<string, unknown>;
  deferred: string[];
  provenance?: Record<string, 'builtin' | 'stored' | 'override'>;
  worker?: TrainingWorkerRef;
}

export function getTrainingRecipe<T>(family: TrainingRecipeFamily, preset?: string): Promise<TrainingRecipe<T>> {
  const params = new URLSearchParams();
  if (preset) params.set('preset', preset);
  const worker = currentWorkerRef();
  if (worker.kind === 'remote') params.set('worker', worker.name);
  const suffix = params.size ? `?${params}` : '';
  return trainingOperation('recipes', `/${family}${suffix}`);
}

/** Resolve a form at command time. The worker is captured in the operation
 *  envelope, so a later Train on switch cannot change the accepted recipe. */
export function resolveTrainingRecipe<T>(family: TrainingRecipeFamily, overrides: Record<string, unknown>, preset?: string): Promise<TrainingRecipe<T>> {
  return trainingOperation('recipes', '/resolve', {
    body: snapshotFor({ kind: `recipe:${family}`, idempotencyKey: crypto.randomUUID(),
      payload: { family, recipeVersion: 1, overrides, ...(preset ? { preset } : {}) } }),
  });
}

/** Legacy start functions still read Train on. Refuse a switch after the
 *  resolver captured its worker, before handing the body to that function. */
export function assertRecipeWorker(recipe: Pick<TrainingRecipe, 'worker'>): void {
  const current = currentWorkerRef();
  if (!recipe.worker || current.kind !== recipe.worker.kind
      || (current.kind === 'remote' && (recipe.worker.kind !== 'remote' || current.name !== recipe.worker.name))) {
    throw new Error('Training worker changed while the recipe was resolved. Review the form and start again.');
  }
}
