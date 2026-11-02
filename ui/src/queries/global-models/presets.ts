import { queryOptions } from '@tanstack/react-query';
import { queryClient } from '../client';
import { queryManager } from '../invalidation';
import { localOptions } from './options';
import { preferencesApi, type PresetFamily, type ImportPreferenceItem } from '../../services/preferencesApi';
import type { TypedDocument } from '../../services/workflowApi';

const options = <T>(family: PresetFamily) => queryOptions({
  ...localOptions<{ documents: TypedDocument<T>[] }>('vst-presets', `/preferences/presets/${family}`, { family }),
  // refresh() also supplies conflict read-back: it must see the current revision.
  staleTime: 0,
});
async function write<T>(family: PresetFamily, command: () => Promise<T>) {
  await queryClient.cancelQueries({ queryKey: options(family).queryKey });
  try { return await command(); }
  finally { await queryManager.invalidate(queryClient, options(family).queryKey); }
}

// The collection retains unsaved edits, import decisions and revision conflicts.
// Its API adapter gives Query sole ownership of authoritative server reads.
export const vstPresetsApi: typeof preferencesApi.presets = {
  ...preferencesApi.presets,
  list: <T>(family: PresetFamily) => queryClient.query(options<T>(family)),
  create: <T>(family: PresetFamily, body: T) => write(family, () => preferencesApi.presets.create(family, body)),
  update: <T>(family: PresetFamily, id: string, revision: number, body: T) =>
    write(family, () => preferencesApi.presets.update(family, id, revision, body)),
  remove: (family: PresetFamily, id: string, revision: number) =>
    write(family, () => preferencesApi.presets.remove(family, id, revision)),
  import: (family: PresetFamily, items: ImportPreferenceItem[]) =>
    write(family, () => preferencesApi.presets.import(family, items)),
};
