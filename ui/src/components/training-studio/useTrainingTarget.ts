// The studio follows the active backend until a submitted job pins its target.

import { useBackendStore } from '../../stores/backendStore';
import { useActiveTrainingJob } from '../../queries/training-runs/hooks';
import { resolveTrainingTarget } from './trainingTargets';
import type { TrainingTargetId } from './trainingPluginTypes';

export function useTrainingTarget(): { target: TrainingTargetId } {
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const job = useActiveTrainingJob();
  const running = job?.status === 'queued' || job?.status === 'running';
  return { target: resolveTrainingTarget({ activeBackendId, runningJobKind: running ? job?.kind : null }) };
}
