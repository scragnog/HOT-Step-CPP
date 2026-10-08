// trainingCreateDraft.ts — open a Training Studio audition as a Create setup.
//
// "Send to Custom-Gen" stores a draft on the server (review operations,
// server/src/services/training/review/auditionDraft.ts) and opens Create at
// /?trainingDraft=<id>. Create applies it only when the user chooses to, so an
// unrelated unsaved form is never overwritten. Applying sets the form content,
// every generation parameter of the mirrored recipe and the LM codes cache
// setting; it never starts a generation.
import { trainingOperation } from '../../services/trainingOperations';

export const TRAINING_DRAFT_QUERY = 'trainingDraft';

export interface TrainingCreateDraftData {
  /** Create's persisted form keys (hs-*). */
  content: Record<string, string | number | boolean>;
  /** Global parameters, applied through the store's set<Key> setters. */
  params: Record<string, unknown>;
  settings?: { cacheLmCodes?: boolean };
}

export function trainingDraftIdFromUrl(search: string = window.location.search): string | null {
  return new URLSearchParams(search).get(TRAINING_DRAFT_QUERY);
}

/** Drop ?trainingDraft from the address, keeping the rest. */
export function clearTrainingDraftQuery(): void {
  const url = new URL(window.location.href);
  url.searchParams.delete(TRAINING_DRAFT_QUERY);
  window.history.replaceState(window.history.state, '', url.pathname + url.search + url.hash);
}

export async function loadTrainingCreateDraft(id: string, token: string): Promise<TrainingCreateDraftData> {
  const { draft } = await trainingOperation<{ draft: { data: TrainingCreateDraftData } }>(
    'review', `/draft/${encodeURIComponent(id)}`, { token });
  return draft.data;
}

export interface ApplyDeps {
  store: { getState(): Record<string, unknown>; setState(partial: Record<string, unknown>): void };
  write(key: string, value: unknown): void;
  storage: Pick<Storage, 'getItem' | 'setItem'>;
  /** The active backend's key for a per-backend setting. */
  scopedKey(base: string): string;
}

const setterFor = (key: string) => `set${key[0].toUpperCase()}${key.slice(1)}`;

/** Apply a draft to Create. Everything is checked first, so a draft this
 *  build cannot apply in full changes nothing. */
export function applyTrainingCreateDraft(data: TrainingCreateDraftData, deps: ApplyDeps): void {
  const state = deps.store.getState();
  const bad = Object.keys(data.content).filter(k => !k.startsWith('hs-'))
    .concat(Object.keys(data.params).filter(k => k !== 'pluginParams' && typeof state[setterFor(k)] !== 'function'));
  if (bad.length) throw new Error(`This Create cannot apply the draft's ${bad.join(', ')}`);

  for (const [key, value] of Object.entries(data.content)) deps.write(key, value);
  for (const [key, value] of Object.entries(data.params)) {
    if (key === 'pluginParams') {
      // Per backend: the active backend's copy, as the store keeps it.
      deps.store.setState({ pluginParams: value });
      try { deps.storage.setItem(deps.scopedKey('hs-pluginParams'), JSON.stringify(value)); } catch { /* storage full */ }
      continue;
    }
    (state[setterFor(key)] as (v: unknown) => void)(value);
  }
  if (data.settings?.cacheLmCodes !== undefined) {
    try {
      const raw = deps.storage.getItem('ace-settings');
      const settings = raw ? JSON.parse(raw) : {};
      settings.cacheLmCodes = data.settings.cacheLmCodes;
      // Through write(), so App's live usePersistedState copy updates too.
      deps.write('ace-settings', settings);
    } catch { /* unreadable settings: the cache stays as it was */ }
  }
}
