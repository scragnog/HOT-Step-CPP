// The studio follows the active backend until a submitted job pins its target.

import { useBackendStore } from '../../stores/backendStore';
import { useTrainingStore } from '../../stores/trainingStore';
import { resolveTrainingTarget } from './trainingTargets';
import type { TrainingTargetId } from './trainingPluginTypes';

export function useTrainingTarget(): { target: TrainingTargetId } {
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const jobKind = useTrainingStore(s => s.activeJob?.kind);
  const jobStatus = useTrainingStore(s => s.activeJob?.status);
  const running = jobStatus === 'queued' || jobStatus === 'running';
  return { target: resolveTrainingTarget({ activeBackendId, runningJobKind: running ? jobKind : null }) };
}
