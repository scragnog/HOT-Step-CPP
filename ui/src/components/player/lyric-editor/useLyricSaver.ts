// useLyricSaver.ts — writing the timings, and every way that can go wrong.
//
// The rules live here rather than in the modal: a write carries the revision it
// read, a mismatch is stale and nothing is written, a failure is not retried on
// a timer, and a successful write re-arms autosave. The editor only renders the
// resulting badge and offers Save or Reload.
//
// The return is `{ data, ...handlers }`, the shape every editor hook shares:
// `data` is the state the badge and the buttons read, the rest is what they may
// call. The exported types are inferred from the function, never retyped.

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';
import { queryClient } from '../../../queries/client';
import { sharedKeys } from '../../../queries/keys';
import { saveLyricTimings } from '../../../services/api';
import { ApiError } from '../../../services/httpClient';
import { sidecarUrl, type LyricsJson } from '../../../utils/wordLrcUtils';
import { AUTOSAVE_MS, cloneLines, type EditLine } from './timingModel';

const jsonKey = (audioUrl: string) => sharedKeys.lyricsJson(sidecarUrl(audioUrl, '.lyrics.json'));
const lrcKey = (audioUrl: string) => sharedKeys.lrc(sidecarUrl(audioUrl, '.lrc'));

/** Both sidecars are rewritten by every write, so both reads are refreshed. */
async function refreshSidecars(audioUrl: string) {
  await queryClient.refetchQueries({ queryKey: jsonKey(audioUrl) });
  await queryClient.refetchQueries({ queryKey: lrcKey(audioUrl) });
}

export interface LyricSaverOptions {
  audioUrl: string;
  /** Current lines, for dirty checks and the autosave timer. */
  lines: EditLine[];
  /** The newest lines, read at save time so a pending drag is included. */
  getLines: () => EditLine[];
  autoSave: boolean;
  /** False until the sidecar has arrived. */
  enabled: boolean;
  /** The revision on disk when the editor opened. */
  revision?: number;
}

export function useLyricSaver({ audioUrl, lines, getLines, autoSave, enabled, revision }: LyricSaverOptions) {
  const { t } = useTranslation();
  const [baseline, setBaseline] = useState<string>(() => JSON.stringify(lines));
  /** The last saved array itself, so Revert does not have to reparse a string. */
  const [savedLines, setSavedLines] = useState<EditLine[]>(lines);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(false);
  const [stale, setStale] = useState(false);
  const [tick, setTick] = useState(0);

  const baselineRef = useRef(baseline);
  const staleRef = useRef(stale);
  useLayoutEffect(() => { staleRef.current = stale; }, [stale]);
  const savingRef = useRef(false);
  const revisionRef = useRef<number>(Number(revision) || 0);

  useEffect(() => {
    if (typeof revision === 'number') revisionRef.current = revision;
  }, [revision]);

  const dirty = useMemo(() => JSON.stringify(lines) !== baseline, [lines, baseline]);

  /** Take the file as it now is, without writing anything. */
  const adopt = useCallback((next: EditLine[], nextRevision?: number) => {
    const serialized = JSON.stringify(next);
    baselineRef.current = serialized;
    setBaseline(serialized);
    setSavedLines(next);
    if (typeof nextRevision === 'number') revisionRef.current = nextRevision;
    setStale(false);
    setSaveError(false);
    setTick(n => n + 1);
  }, []);

  const save = useCallback(async (opts: { silent?: boolean } = {}) => {
    if (savingRef.current || staleRef.current) return;
    const snapshot = getLines();
    const serialized = JSON.stringify(snapshot);
    if (serialized === baselineRef.current) return;
    savingRef.current = true;
    setSaving(true);
    try {
      const result = await saveLyricTimings(audioUrl, snapshot, revisionRef.current);
      await refreshSidecars(audioUrl);
      baselineRef.current = serialized;
      revisionRef.current = Number(result.revision) || revisionRef.current;
      setBaseline(serialized);
      setSavedLines(snapshot);
      setSaveError(false);
      if (!opts.silent) toast.success(t('lyrics.editor.saved', 'Timings saved'));
      // Autosave is re-armed here, on the success path, rather than after the
      // try with a flag. A failure therefore leaves it disarmed and waits for
      // the user's Save instead of retrying every 1.2 s with a toast each time.
      setTick(n => n + 1);
    } catch (err: unknown) {
      if (err instanceof ApiError && err.status === 409) {
        setStale(true);
        toast.error(t('lyrics.editor.staleToast', 'The timings changed on disk. Reload before saving.'));
      } else {
        setSaveError(true);
        toast.error((err instanceof Error && err.message) || t('lyrics.editor.saveFailed', 'Could not save the timings'));
      }
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  }, [audioUrl, getLines, t]);

  const saveRef = useRef(save);
  useLayoutEffect(() => { saveRef.current = save; }, [save]);

  useEffect(() => {
    if (!autoSave || !enabled || stale) return;
    if (JSON.stringify(lines) === baselineRef.current) return;
    const timer = setTimeout(() => { void saveRef.current({ silent: true }); }, AUTOSAVE_MS);
    return () => clearTimeout(timer);
  }, [lines, tick, enabled, autoSave, stale]);

  /** Throw away local edits and take the file as it is now. */
  const reload = useCallback(async (): Promise<{ lines: EditLine[]; revision: number }> => {
    await refreshSidecars(audioUrl);
    const fresh = queryClient.getQueryData<LyricsJson>(jsonKey(audioUrl));
    const next = cloneLines(fresh?.lines);
    const rev = Number(fresh?.revision) || 0;
    adopt(next, rev);
    toast.info(t('lyrics.editor.reloaded', 'Reloaded the timings from disk'));
    return { lines: next, revision: rev };
  }, [audioUrl, adopt, t]);

  return {
    data: { dirty, saving, saveError, stale, savedLines },
    save, adopt, reload,
  };
}

/** The hook's whole return, so its shape is never written twice. */
export type LyricSaver = ReturnType<typeof useLyricSaver>;
/** What a component may call: the handlers, without the state. */
export type LyricSaverHandlers = Omit<LyricSaver, 'data'>;
