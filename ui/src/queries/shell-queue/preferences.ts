// preferences.ts — Query-backed defaults for the shared preference services.
//
// PresetCollection and SettingDocument keep unsaved edits, legacy imports
// and revision conflicts; these adapters give Query their authoritative
// reads. Reads are always fresh (staleTime 0): a refresh after a conflict
// must see the current revision, so an entry is dropped as soon as nothing
// reads it (gcTime 0): it only exists to join concurrent reads of one family
// into one request. Every write invalidates the family it changed. Writes
// run once, never retried.

import { queryOptions } from '@tanstack/react-query';
import {
  preferencesApi, type ImportPreferenceItem, type PresetFamily, type SettingsFamily,
} from '../../services/preferencesApi';
import type { TypedDocument } from '../../services/workflowApi';
import { queryClient } from '../client';
import { serverKey } from '../keys';
import { queryManager } from '../invalidation';

const presetsOptions = <T>(family: PresetFamily) => queryOptions({
  queryKey: serverKey('shell-queue/presets', { filters: { family } }),
  queryFn: ({ signal }) => preferencesApi.presets.list<T>(family, { signal }),
  staleTime: 0,
  gcTime: 0,
});

export const settingOptions = <T>(family: SettingsFamily) => queryOptions({
  queryKey: serverKey('shell-queue/setting', { filters: { family } }),
  queryFn: ({ signal }) => preferencesApi.settings.get<T>(family, { signal }),
  staleTime: 0,
  gcTime: 0,
});

/** Run a write with no read of `key` in flight across it, then refetch. */
async function write<R>(key: readonly unknown[], command: () => Promise<R>): Promise<R> {
  await queryClient.cancelQueries({ queryKey: key });
  try { return await command(); }
  finally { await queryManager.invalidate(queryClient, key); }
}

export const queriedPresetsApi: typeof preferencesApi.presets = {
  ...preferencesApi.presets,
  list: <T>(family: PresetFamily) =>
    queryClient.query(presetsOptions<T>(family)) as Promise<{ documents: TypedDocument<T>[] }>,
  create: <T>(family: PresetFamily, body: T) =>
    write(presetsOptions(family).queryKey, () => preferencesApi.presets.create(family, body)),
  update: <T>(family: PresetFamily, id: string, revision: number, body: T) =>
    write(presetsOptions(family).queryKey, () => preferencesApi.presets.update(family, id, revision, body)),
  remove: (family: PresetFamily, id: string, revision: number) =>
    write(presetsOptions(family).queryKey, () => preferencesApi.presets.remove(family, id, revision)),
  import: (family: PresetFamily, items: ImportPreferenceItem[]) =>
    write(presetsOptions(family).queryKey, () => preferencesApi.presets.import(family, items)),
};

export const queriedSettingsApi: typeof preferencesApi.settings = {
  ...preferencesApi.settings,
  get: <T>(family: SettingsFamily) =>
    queryClient.query(settingOptions<T>(family)) as Promise<{ document: TypedDocument<T> | null }>,
  upsert: <T>(family: SettingsFamily, expectedRevision: number | undefined, body: T) =>
    write(settingOptions(family).queryKey, () => preferencesApi.settings.upsert(family, expectedRevision, body)),
  import: (family: SettingsFamily, item: ImportPreferenceItem) =>
    write(settingOptions(family).queryKey, () => preferencesApi.settings.import(family, item)),
};
