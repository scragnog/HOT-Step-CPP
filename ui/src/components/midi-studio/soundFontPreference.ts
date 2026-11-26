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

export interface SoundFontPreference {
  selection: string;
  custom?: File;
}

const FILE_KEY = 'soundfont';
const changes = new EventTarget();
const tabChanges = typeof window !== 'undefined' && 'BroadcastChannel' in window
  ? new BroadcastChannel('hs-midi-soundfont')
  : null;
import.meta.hot?.dispose(() => tabChanges?.close());

function migrate(value: SoundFontPreference | File | undefined): SoundFontPreference {
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
        preference = migrate(request.result);
        if (update) {
          preference = update(preference);
          store.put(preference, FILE_KEY);
        }
      };
      transaction.oncomplete = () => resolve(preference);
      transaction.onabort = () => reject(transaction.error ?? request.error);
    });
  } finally {
    db.close();
  }
}

export function readSoundFontPreference(): Promise<SoundFontPreference> {
  return withPreference();
}

export function subscribeToSoundFontChanges(listener: () => void): () => void {
  changes.addEventListener('change', listener);
  tabChanges?.addEventListener('message', listener);
  return () => {
    changes.removeEventListener('change', listener);
    tabChanges?.removeEventListener('message', listener);
  };
}

export function notifySoundFontAvailability(): void {
  changes.dispatchEvent(new Event('change'));
}

async function writePreference(update: (current: SoundFontPreference) => SoundFontPreference): Promise<void> {
  await withPreference(update);
  changes.dispatchEvent(new Event('change'));
  tabChanges?.postMessage(null);
}

export function selectSoundFont(selection: string): Promise<void> {
  return writePreference(current => ({ ...current, selection }));
}

export async function saveCustomSoundFont(file: File): Promise<void> {
  if (!/\.(sf2|sf3)$/i.test(file.name)) throw new Error('Expected an SF2 or SF3 file');
  const { SoundBankLoader } = await import('spessasynth_core');
  SoundBankLoader.fromArrayBuffer(await file.arrayBuffer(), { sampleData: false });
  await writePreference(() => ({ selection: CUSTOM_SOUND_FONT, custom: file }));
}

export function clearCustomSoundFont(): Promise<void> {
  return writePreference(({ selection }) => ({
    selection: selection === CUSTOM_SOUND_FONT ? DEFAULT_SOUND_FONT : selection,
  }));
}
