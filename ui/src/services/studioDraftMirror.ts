// studioDraftMirror.ts — mirror a studio's autosaved fields into its own
// revisioned server draft (/api/studio-drafts).
//
// Local storage stays the editing copy, saved on the same schedule as before;
// the server draft follows it after a short debounce. The selected draft
// (id, revision) is a per-browser pointer, so two clients never write each
// other's draft. A revision conflict never overwrites the other writer: the
// local edit is saved as a new draft and the pointer moves to it. A failed
// save keeps the edit local and retries on the next change.
import { useEffect, useMemo, useState } from 'react';
import type { StudioDraftBody } from '../../../server/src/contracts/studioDrafts';
import { studioDraftsApi } from './studioDraftsApi';

export interface DraftPointer { id: string; revision: number; saved: string }

export interface DraftMirrorDeps {
  api: Pick<typeof studioDraftsApi, 'create' | 'update'>;
  pointer: { read(): DraftPointer | null; write(value: DraftPointer): void };
  onError(message: string): void;
}

const statusOf = (err: unknown) => (err as { status?: number } | null)?.status;
const messageOf = (err: unknown) => err instanceof Error ? err.message : String(err);

export function createDraftMirror(deps: DraftMirrorDeps) {
  let pending: { token: string; body: StudioDraftBody } | null = null;
  let running: Promise<void> | null = null;

  async function drain(): Promise<void> {
    while (pending) {
      const { token, body } = pending;
      pending = null;
      const saved = JSON.stringify(body);
      const pointer = deps.pointer.read();
      if (pointer?.saved === saved) continue;
      try {
        let document;
        let notice = '';
        if (!pointer) document = (await deps.api.create(token, body)).document;
        else {
          try { document = (await deps.api.update(token, pointer.id, pointer.revision, body)).document; }
          catch (err) {
            if (statusOf(err) !== 409 && statusOf(err) !== 404) throw err;
            document = (await deps.api.create(token, body)).document;
            if (statusOf(err) === 409) notice = 'The saved draft changed elsewhere; your edits were saved as a new draft.';
          }
        }
        deps.pointer.write({ id: document.id, revision: document.revision, saved });
        deps.onError(notice);
      } catch (err) {
        pending ??= { token, body };
        deps.onError(`Draft not saved: ${messageOf(err)}. Your edits stay in this browser.`);
        return;
      }
    }
  }

  return {
    /** Queue the latest body; returns when everything queued so far is written or has failed. */
    save(token: string, body: StudioDraftBody): Promise<void> {
      pending = { token, body };
      running ??= drain().finally(() => { running = null; });
      return running;
    },
  };
}

const pointerKey = (studio: string) => `hs-studioDraft:${studio}`;

function localPointer(studio: StudioDraftBody['studio']): DraftMirrorDeps['pointer'] {
  return {
    read() {
      try {
        const value = JSON.parse(localStorage.getItem(pointerKey(studio)) ?? 'null');
        return value && typeof value.id === 'string' && Number.isInteger(value.revision) ? value : null;
      } catch { return null; }
    },
    write(value) { try { localStorage.setItem(pointerKey(studio), JSON.stringify(value)); } catch { /* storage full */ } },
  };
}

/** Mirror `fields` into this browser's draft for `studio`. Returns the last
 *  save problem ('' when none) for the studio to show. */
export function useStudioDraftMirror(studio: StudioDraftBody['studio'], token: string | null | undefined,
  fields: Record<string, unknown>, identity: Omit<StudioDraftBody, 'studio' | 'fields'> = {}): string {
  const [error, setError] = useState('');
  const mirror = useMemo(() => createDraftMirror({ api: studioDraftsApi, pointer: localPointer(studio), onError: setError }), [studio]);
  const body = JSON.stringify({ studio, fields, ...identity });
  useEffect(() => {
    if (!token) return;
    const timer = window.setTimeout(() => { void mirror.save(token, JSON.parse(body)); }, 1000);
    return () => window.clearTimeout(timer);
  }, [mirror, token, body]);
  return error;
}
