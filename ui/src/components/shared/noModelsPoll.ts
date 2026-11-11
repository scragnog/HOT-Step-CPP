// The engine's ready flag is set right after spawn, before the model scan, so a
// health read can say ready=true and the next one noModels=true. Never stop
// polling on ready: the banner has to catch that later transition.
import { pollWhile } from '../../queries/contract';
import type { HealthResponse } from '../../../../server/src/contracts/health';

export const NO_MODELS_POLL_MS = 5000;
export const noModelsRefetchInterval = pollWhile<HealthResponse>(() => true, NO_MODELS_POLL_MS);

/** Failure takes priority even if the model scan also reported an empty folder. */
export function engineNotice(engine?: Pick<HealthResponse['engine'], 'failed' | 'noModels'>): 'failed' | 'no-models' | null {
  if (engine?.failed === true) return 'failed';
  if (engine?.noModels === true) return 'no-models';
  return null;
}
