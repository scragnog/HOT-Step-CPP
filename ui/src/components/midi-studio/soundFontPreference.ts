function openPreferences(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('hs-midi-preview', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('preferences');
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function withPreference<T>(
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const db = await openPreferences();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = db.transaction('preferences', mode);
      const request = operation(transaction.objectStore('preferences'));
      transaction.oncomplete = () => resolve(request.result);
      transaction.onabort = () => reject(transaction.error ?? request.error);
    });
  } finally {
    db.close();
  }
}

const FILE_KEY = 'soundfont';

export function readCustomSoundFont(): Promise<File | undefined> {
  return withPreference('readonly', store => store.get(FILE_KEY));
}

export async function saveCustomSoundFont(file: File): Promise<void> {
  if (!/\.(sf2|sf3)$/i.test(file.name)) throw new Error('Expected an SF2 or SF3 file');
  const { SoundBankLoader } = await import('spessasynth_core');
  SoundBankLoader.fromArrayBuffer(await file.arrayBuffer(), { sampleData: false });
  await withPreference('readwrite', store => store.put(file, FILE_KEY));
}

export async function clearCustomSoundFont(): Promise<void> {
  await withPreference('readwrite', store => store.delete(FILE_KEY));
}
