import { queryOptions } from '@tanstack/react-query';
import type { CoreGenerationParamDescriptor } from '../../../server/src/contracts/generationControlDescriptorTypes';
import type { BackendExtensionParam } from '../stores/backendStore';
import { ApiClient } from '../services/httpClient';
import { capabilitiesOptions } from './global-models/options';
import { scopeOf, serverKey } from './keys';

export function generationControlDescriptorsOptions() {
  return queryOptions({
    queryKey: serverKey('global-models/generation-control-descriptors', { auth: true }),
    queryFn: ({ queryKey, signal }) => new ApiClient({ baseUrl: scopeOf(queryKey)!.api })
      .get<CoreGenerationParamDescriptor[]>('/generation-controls/descriptors', { signal }),
  });
}

type BackendControlKey = `backendParams.${string}`;
export type BackendGenerationParamDescriptor = Omit<BackendExtensionParam, 'key'> & { key: BackendControlKey };

const backendControlKey = (key: string): BackendControlKey => `backendParams.${key}`;

export function backendControlDescriptorsOptions(backend: string) {
  return queryOptions({
    ...capabilitiesOptions(backend),
    select: ({ extensions }): BackendGenerationParamDescriptor[] => extensions.map(({ key, visible_when, ...descriptor }) => ({
      ...descriptor,
      key: backendControlKey(key),
      ...(visible_when && { visible_when: { ...visible_when, key: backendControlKey(visible_when.key) } }),
    })),
  });
}
