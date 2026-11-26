import { queryOptions } from '@tanstack/react-query';
import { queryClient } from '@/queries/client';
import { queryManager } from '@/queries/invalidation';
import { serverKey } from '@/queries/keys';
import { apiClient } from '@/services/httpClient';

function openPreferences(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('hs-midi-preview', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('preferences');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export const DEFAULT_SOUND_FONT = 'default';
export const CUSTOM_SOUND_FONT = 'custom';
export const DEFAULT_SOUND_FONT_ID = 'midi-generaluser-gs';
const LIBRARY_PREFIX = 'library:';
const LIBRARY_PATH = '/midi-studio/soundfonts';

export interface SoundFontPreference {
  selection: string;
  custom?: File;
}

export interface SoundFontEntry {
  id: string;
  name: string;
  size: number;
  format: 'sf2' | 'sf3';
  addedAt: string;
}

export function librarySelection(id: string): string {
  return LIBRARY_PREFIX + id;
}

export function libraryBankUrl(selection: string): string | null {
  return selection.startsWith(LIBRARY_PREFIX)
    ? `${LIBRARY_PATH}/${encodeURIComponent(selection.slice(LIBRARY_PREFIX.length))}`
    : null;
}

export const soundFontLibraryOptions = () => queryOptions({
  queryKey: serverKey('studios/soundfonts'),
  queryFn: ({ signal }) => apiClient.get<SoundFontEntry[]>(LIBRARY_PATH, { signal }),
  refetchInterval: 15_000,
});

const FILE_KEY = 'soundfont';
const changes = new EventTarget();
const tabChanges = typeof window !== 'undefined' && 'BroadcastChannel' in window
  ? new BroadcastChannel('hs-midi-soundfont')
  : null;
import.meta.hot?.dispose(() => tabChanges?.close());

function refreshLibrary() {
  return queryManager.invalidate(queryClient, soundFontLibraryOptions().queryKey);
}

tabChanges?.addEventListener('message', event => {
  if (event.data === 'library') void refreshLibrary();
});

function normalize(value: SoundFontPreference | File | undefined): SoundFontPreference {
  if (value instanceof File) return { selection: CUSTOM_SOUND_FONT, custom: value };
  return value ?? { selection: DEFAULT_SOUND_FONT };
}

async function withPreference(update?: (current: SoundFontPreference) => SoundFontPreference): Promise<SoundFontPreference> {
  const db = await openPreferences();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = db.transaction('preferences', update ? 'readwrite' : 'readonly');
      const store = transaction.objectStore('preferences');
      const request = store.get(FILE_KEY);
      let preference: SoundFontPreference;
      request.onsuccess = () => {
        preference = normalize(request.result);
        if (update) {
          const updated = update(preference);
          if (updated !== preference) store.put(updated, FILE_KEY);
          preference = updated;
        }
      };
      transaction.oncomplete = () => resolve(preference);
      transaction.onabort = () => reject(transaction.error ?? request.error);
    });
  } finally {
    db.close();
  }
}

let migration: Promise<SoundFontPreference> | undefined;

export async function readSoundFontPreference(): Promise<SoundFontPreference> {
  const preference = await withPreference();
  if (!preference.custom) return preference;
  const file = preference.custom;
  migration ??= (async () => {
    const entry = await uploadBank(file);
    return writePreference(current => current.custom ? ({
      selection: current.selection === CUSTOM_SOUND_FONT ? librarySelection(entry.id) : current.selection,
    }) : current);
  })().finally(() => { migration = undefined; });
  // Preserve the old bank for playback and retry the upload on a later visit.
  return migration.catch(() => preference);
}

export function subscribeToSoundFontChanges(listener: () => void): () => void {
  const onMessage = (event: MessageEvent) => {
    if (event.data !== 'library') listener();
  };
  changes.addEventListener('change', listener);
  tabChanges?.addEventListener('message', onMessage);
  return () => {
    changes.removeEventListener('change', listener);
    tabChanges?.removeEventListener('message', onMessage);
  };
}

export function notifySoundFontAvailability(): void {
  changes.dispatchEvent(new Event('change'));
}

async function writePreference(update: (current: SoundFontPreference) => SoundFontPreference): Promise<SoundFontPreference> {
  let changed = false;
  const result = await withPreference(current => {
    const updated = update(current);
    changed = updated !== current;
    return updated;
  });
  if (changed) {
    notifySoundFontAvailability();
    tabChanges?.postMessage('selection');
  }
  return result;
}

export async function selectSoundFont(selection: string): Promise<void> {
  await writePreference(current => current.selection === selection ? current : { ...current, selection });
}

async function libraryChanged() {
  tabChanges?.postMessage('library');
  await refreshLibrary();
}

async function uploadBank(file: File): Promise<SoundFontEntry> {
  const entry = await apiClient.upload<SoundFontEntry>(LIBRARY_PATH, [{ field: 'file', file }], {
    fields: { name: file.name },
    silent: true,
  });
  await libraryChanged();
  return entry;
}

export async function saveCustomSoundFont(file: File): Promise<void> {
  const entry = await uploadBank(file);
  await selectSoundFont(librarySelection(entry.id));
}

export async function renameLibrarySoundFont(id: string, name: string): Promise<void> {
  await apiClient.patch(`${LIBRARY_PATH}/${encodeURIComponent(id)}`, { name }, { silent: true });
  await libraryChanged();
}

export async function resetMissingSoundFont(selection: string): Promise<void> {
  await writePreference(current => current.selection === selection
    ? { ...current, selection: DEFAULT_SOUND_FONT }
    : current);
}

export async function removeLibrarySoundFont(id: string): Promise<void> {
  await apiClient.delete(`${LIBRARY_PATH}/${encodeURIComponent(id)}`, { silent: true });
  try {
    await resetMissingSoundFont(librarySelection(id));
  } finally {
    await libraryChanged();
  }
}
