// useEnvSettingsForm.ts — the one owner of the editable .env form.
//
// The form is server state with a draft on top: envOptions() is the source of
// truth, and the draft is seeded from it only while the user has no unsaved
// edits. A refetch while dirty updates `original` (not the draft), so an edit
// made in another tab — or by a restart that rewrote defaults — shows up as a
// new diff instead of being silently overwritten.
//
// Save and restart are mutations because each owns a pending/error lifecycle.
// Their only notifier is the status line below the form; both service calls are
// silent at the transport, so a failure never also toasts. Invalidation goes
// through queryManager.env.values, the one owner (docs/dev/tanstack-query.md P5).

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { GpuInfo } from '../../../../server/src/contracts/settings';
import { envOptions, gpusOptions } from '../../queries/shell-queue/settings';
import { queryManager } from '../../queries/invalidation';
import { settingsApi, shutdownApi } from '../../services/api';
import { ApiError } from '../../services/apiError';

export interface EnvFormStatus {
  type: 'success' | 'error' | 'warning';
  text: string;
}

interface EnvFormState {
  /** What the inputs show. */
  draft: Record<string, string>;
  /** The last server snapshot the diff is measured against. */
  original: Record<string, string>;
}

const sameValues = (a: Record<string, string>, b: Record<string, string>) => {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const key of keys) if (a[key] !== b[key]) return false;
  return true;
};

export interface EnvSettingsForm {
  values: Record<string, string>;
  loading: boolean;
  status: EnvFormStatus | null;
  dirty: boolean;
  saving: boolean;
  restarting: boolean;
  /** Keys already saved that only a restart puts into effect (issue #99). */
  pendingRestartKeys: string[];
  gpus: GpuInfo[];
  change: (key: string, value: string) => void;
  save: () => void;
  restartNow: () => void;
}

export function useEnvSettingsForm(): EnvSettingsForm {
  const { t } = useTranslation();
  const client = useQueryClient();
  const envQuery = useQuery(envOptions());
  // GPUs are a separate read: a failed detection leaves the text input in
  // place instead of blocking the form. No retries — the old one-shot load
  // did not retry either, and a missing nvidia-smi will not start working.
  const gpuQuery = useQuery({ ...gpusOptions(), retry: false });
  // Seed from a cached snapshot: reopening Settings marks it seen below, and an
  // unchanged refetch keeps its identity, so it would never seed the form.
  const [form, setForm] = useState<EnvFormState>(() => {
    const values = envQuery.data?.values ?? {};
    return { draft: values, original: values };
  });
  const [status, setStatus] = useState<EnvFormStatus | null>(null);
  const [pendingRestartKeys, setPendingRestartKeys] = useState<string[]>([]);

  // Seed only while clean. Once the user has edited a field the draft is
  // theirs; the newest server values still replace `original`, so external
  // edits surface as diffs on top of the user's draft. This is React's
  // "adjust state while rendering" pattern: the guard makes it a one-shot
  // update when a new snapshot arrives, not a render loop.
  const [seenData, setSeenData] = useState(envQuery.data);
  if (envQuery.data !== seenData) {
    setSeenData(envQuery.data);
    const data = envQuery.data;
    if (data) setForm(prev => ({
      draft: sameValues(prev.draft, prev.original) ? data.values : prev.draft,
      original: data.values,
    }));
  }

  const dirty = !sameValues(form.draft, form.original);
  const restartKeys = envQuery.data?.restartKeys ?? [];

  const change = (key: string, value: string) => {
    setForm(prev => ({ draft: { ...prev.draft, [key]: value }, original: prev.original }));
    setStatus(null);
  };

  const saveMut = useMutation({
    mutationFn: (diff: Record<string, string>) => settingsApi.updateEnv(diff),
    onMutate: () => {
      setStatus(null);
      // Capture the draft at submit time so later keystrokes don't get
      // silently committed when the response lands.
      return { submittedDraft: { ...form.draft } };
    },
    onSuccess: (res, _diff, ctx) => {
      // Only the submitted values become the new baseline; edits made after
      // clicking Save remain as a visible diff against this snapshot.
      setForm(prev => ({ draft: prev.draft, original: ctx?.submittedDraft ?? prev.draft }));
      void queryManager.env.values(client);
      if (res.restartRequired) {
        const stale = res.updated.filter(k => restartKeys.includes(k));
        setPendingRestartKeys(stale);
        setStatus({
          type: 'warning',
          text: stale.length
            ? `Saved. The engine is still running the old ${stale.join(', ')} until you restart.`
            : `Saved ${res.updated.length} setting(s). Some changes require a restart.`,
        });
      } else {
        setPendingRestartKeys([]);
        setStatus({ type: 'success', text: `Saved ${res.updated.length} setting(s). Changes are live.` });
      }
    },
    onError: e => setStatus({ type: 'error', text: `Failed to save: ${e instanceof Error ? e.message : String(e)}` }),
  });

  const save = () => {
    if (!dirty) return;
    const diff: Record<string, string> = {};
    for (const [key, value] of Object.entries(form.draft)) {
      if (value !== form.original[key]) diff[key] = value;
    }
    saveMut.mutate(diff);
  };

  /** Apply restart-required settings by restarting, from where they were saved.
   *  The restart control already existed in the sidebar, but nothing in the
   *  Save Environment flow pointed at it, so a saved value could sit on screen
   *  looking active while the engine ran on the old one indefinitely (#99).
   *
   *  The app-level restarting overlay has to be raised too. Without it the
   *  Settings page stayed on screen against a server that was going away: the
   *  button read "Restarting…" forever, nothing reloaded when the server came
   *  back, and every panel that probes the engine (post-processing among them)
   *  quietly degraded to "unavailable" (issue #153). */
  const restartMut = useMutation({
    mutationFn: () => shutdownApi.restart(),
    onMutate: () => {
      setStatus({ type: 'warning', text: 'Restarting…' });
      window.dispatchEvent(new CustomEvent('hotstep:restarting'));
    },
    onError: e => {
      // A refused restart (an HTTP error) means the server is still up — drop
      // the overlay and surface the refusal. A dropped connection (a plain
      // Error, no status) is the expected outcome of a successful one: the
      // server tears the socket down as it goes, so keep waiting for the reload.
      if (e instanceof ApiError && e.status > 0) {
        setStatus({ type: 'error', text: t('api.error.refused') });
        window.dispatchEvent(new CustomEvent('hotstep:restart-cancelled'));
      }
    },
  });

  const loadError = envQuery.isError
    ? `Failed to load: ${envQuery.error instanceof Error ? envQuery.error.message : String(envQuery.error)}`
    : null;

  return {
    values: form.draft,
    loading: envQuery.isPending || gpuQuery.isPending,
    status: status ?? (loadError ? { type: 'error', text: loadError } : null),
    dirty,
    saving: saveMut.isPending,
    restarting: restartMut.isPending,
    pendingRestartKeys,
    gpus: gpuQuery.data?.gpus ?? [],
    change,
    save,
    restartNow: () => restartMut.mutate(),
  };
}
