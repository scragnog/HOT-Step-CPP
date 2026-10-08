// studioDraftMirror.ts — mirror a studio's autosaved fields into its own
// revisioned server draft (/api/studio-drafts), and load a saved draft back.
//
// Local storage stays the editing copy, saved on the same schedule as before;
// the server draft follows it after a short debounce. Each editor (one studio
// in one tab) keeps its own draft id and expected revision, so a second tab or
// another client can never be overwritten silently: a stale write gets 409 and
// the local edit is saved as a new draft instead. The per-browser pointer only
// says which draft a newly opened editor starts from. A failed save keeps the
// edit local and retries on the next change. Loading a draft never starts a job.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { StudioDraftBody } from '../../../server/src/contracts/studioDrafts';
import { studioDraftsApi, type StudioDraftDocument } from './studioDraftsApi';

export interface DraftPointer { id: string; revision: number; saved: string }

export interface DraftMirrorDeps {
  api: Pick<typeof studioDraftsApi, 'create' | 'update'>;
  pointer: { read(): DraftPointer | null; write(value: DraftPointer): void };
  onError(message: string): void;
}

const statusOf = (err: unknown) => (err as { status?: number } | null)?.status;
const messageOf = (err: unknown) => err instanceof Error ? err.message : String(err);

export function createDraftMirror(deps: DraftMirrorDeps) {
  // This editor's draft. Read once: another editor's later writes to the
  // shared pointer must not lend this one their revision.
  let current = deps.pointer.read();
  let pending: { token: string; body: StudioDraftBody } | null = null;
  let running: Promise<void> | null = null;
  let epoch = 0;
  let loadSeq = 0;

  async function drain(): Promise<void> {
    while (pending) {
      const { token, body } = pending;
      pending = null;
      const saved = JSON.stringify(body);
      if (current?.saved === saved) continue;
      const started = epoch;
      try {
        let document;
        let notice = '';
        if (!current) document = (await deps.api.create(token, body)).document;
        else {
          try { document = (await deps.api.update(token, current.id, current.revision, body)).document; }
          catch (err) {
            if (statusOf(err) !== 409 && statusOf(err) !== 404) throw err;
            document = (await deps.api.create(token, body)).document;
            if (statusOf(err) === 409) notice = 'The saved draft changed elsewhere; your edits were saved as a new draft.';
          }
        }
        if (started !== epoch) continue; // a draft was loaded meanwhile; it now owns this editor
        current = { id: document.id, revision: document.revision, saved };
        deps.pointer.write(current);
        deps.onError(notice);
      } catch (err) {
        if (started !== epoch) continue;
        pending ??= { token, body };
        deps.onError(`Draft not saved: ${messageOf(err)}. Your edits stay in this browser.`);
        return;
      }
    }
  }

  return {
    /** Queue the latest body; resolves when everything queued so far is written or has failed. */
    save(token: string, body: StudioDraftBody): Promise<void> {
      pending = { token, body };
      running ??= drain().finally(() => { running = null; });
      return running;
    },
    /** Whether `body` differs from what this editor last saved (or nothing was saved yet). */
    unsaved(body: StudioDraftBody): boolean { return current?.saved !== JSON.stringify(body); },
    /** Make a loaded draft this editor's draft. Queued and in-flight saves of the old form are dropped. */
    adopt(document: StudioDraftDocument): void {
      epoch++;
      pending = null;
      current = { id: document.id, revision: document.revision, saved: JSON.stringify(document.body) };
      deps.pointer.write(current);
      deps.onError('');
    },
    /** Load a draft into this editor. Everything is decided after the fetch:
     *  a later load wins, a refusal (e.g. another backend) changes nothing, and
     *  the live form (not the form at selection time) decides whether to ask
     *  before replacing unsaved edits. Resolves true when the draft was applied. */
    async load(token: string, id: string, h: LoadHooks): Promise<boolean> {
      const seq = ++loadSeq;
      try {
        const { document, sourceError } = await h.get(token, id);
        if (seq !== loadSeq) return false;
        const refusal = h.refuse?.(document.body);
        if (refusal) { deps.onError(refusal); return false; }
        if (this.unsaved(h.current()) && !h.confirm(REPLACE_PROMPT)) return false;
        this.adopt(document);
        h.apply(document.body);
        if (sourceError) deps.onError(sourceError);
        return true;
      } catch (err) {
        if (seq === loadSeq) deps.onError(`Draft not loaded: ${messageOf(err)}`);
        return false;
      }
    },
  };
}

const REPLACE_PROMPT = 'Replace the current form with the saved draft? Edits not yet saved to a draft will be lost.';

export interface LoadHooks {
  get(token: string, id: string): Promise<{ document: StudioDraftDocument; sourceError: string | null }>;
  /** The form as it is now. */
  current(): StudioDraftBody;
  confirm(message: string): boolean;
  /** A reason this editor cannot take the draft, or '' to accept it. */
  refuse?(body: StudioDraftBody): string;
  apply(body: StudioDraftBody): void;
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

export interface StudioDraftControl {
  /** The last save or load problem, '' when none. */
  error: string;
  list(): Promise<StudioDraftDocument[]>;
  /** Fetch a draft and hand its fields to `apply`, unless `refuse` names a
   *  reason or the user keeps unsaved edits. Resolves false when nothing was applied. */
  load(id: string, apply: (body: StudioDraftBody) => void, refuse?: (body: StudioDraftBody) => string): Promise<boolean>;
}

/** Mirror `fields` into this editor's draft for `studio`. */
export function useStudioDraftMirror(studio: StudioDraftBody['studio'], token: string | null | undefined,
  fields: Record<string, unknown>, identity: Omit<StudioDraftBody, 'studio' | 'fields'> = {}): StudioDraftControl {
  const [error, setError] = useState('');
  const mirror = useMemo(() => createDraftMirror({ api: studioDraftsApi, pointer: localPointer(studio), onError: setError }), [studio]);
  const body = JSON.stringify({ studio, fields, ...identity });
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (!token) return;
    timer.current = window.setTimeout(() => { void mirror.save(token, JSON.parse(body)); }, 1000);
    return () => window.clearTimeout(timer.current);
  }, [mirror, token, body]);
  const list = useCallback(async () => token ? (await studioDraftsApi.list(token, studio)).documents : [], [token, studio]);
  const bodyRef = useRef(body);
  bodyRef.current = body;
  const load = useCallback(async (id: string, apply: (draft: StudioDraftBody) => void, refuse?: (draft: StudioDraftBody) => string) => {
    if (!token) return false;
    return mirror.load(token, id, {
      get: studioDraftsApi.get, current: () => JSON.parse(bodyRef.current), refuse,
      confirm: message => window.confirm(message),
      apply: draft => { window.clearTimeout(timer.current); apply(draft); },
    });
  }, [token, mirror]);
  return { error, list, load };
}
