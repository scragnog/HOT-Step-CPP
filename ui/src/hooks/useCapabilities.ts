import { useQuery } from '@tanstack/react-query';
import { capabilitiesOptions } from '@/queries/global-models/options';
import { useBackendStore, type BackendCapabilities } from '@/stores/backendStore';

export interface UseCapabilitiesResult {
  capabilities: BackendCapabilities | null;
  loading: boolean;
}
export function useCapabilities(): UseCapabilitiesResult {
  const backend = useBackendStore(s => s.activeBackendId);
  const query = useQuery({ ...capabilitiesOptions(backend), enabled: !!backend });
  // The scoped compatibility projection retains the last usable manifest while
  // the raw down response in Query keeps recovery polling active.
  const projected = useBackendStore(s => s.capabilities[backend]);
  return { capabilities: projected ?? query.data ?? null, loading: query.isPending };
}
