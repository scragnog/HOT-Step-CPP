import { useQuery } from '@tanstack/react-query';
import { runQueries } from '../../queries/training-runs/queries';
import { useTrainingStore } from '../../stores/trainingStore';

export function useYue2Status(datasetId: string) {
  const worker = useTrainingStore(s => s.trainingWorker);
  const query = useQuery({ ...runQueries.yue2Status(worker, datasetId), enabled: !!datasetId });
  return { status: query.data ?? null, error: query.error?.message ?? null,
    reload: () => { void query.refetch(); } };
}
