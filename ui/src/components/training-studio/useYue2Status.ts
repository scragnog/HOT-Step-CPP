// useYue2Status.ts — the YuE2 dataset status the training card reads.
//
// The YuE2 analogue of useMm3Status, and its own file for the same two reasons:
// the card then exports nothing but a component (fast refresh), and the latent
// cache, the model readiness and the measured defaults arrive as ONE shape from
// ONE fetch, so no two parts of the card can disagree about what is on disk.
//
// The endpoint is cheap and never throws server-side, so this polls on the one
// event that can actually change its answer: a YuE2 job of ours reaching a
// terminal state. Preprocess rewrites the latent cache; training does not, but
// it does leave a run the checkpoint list has to notice, and re-reading both
// after either is simpler than two subscriptions that differ by one field.

import { useEffect, useState } from 'react';

import * as trainingApi from '../../services/trainingApi';
import type { Yue2Status } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';

export function useYue2Status(datasetId: string): {
  status: Yue2Status | null; error: string | null; reload: () => void;
} {
  const activeJob = useTrainingStore(s => s.activeJob);
  // Tagged with the dataset it was fetched for. On a dataset switch the old
  // answer must not be handed out while the new fetch is in flight: a card that
  // mounts in that gap takes the previous dataset's manifest path as its own.
  const [fetched, setFetched] = useState<{ datasetId: string; status: Yue2Status } | null>(null);
  const status = fetched?.datasetId === datasetId ? fetched.status : null;
  const [error, setError] = useState<string | null>(null);
  /** Bumped by `reload`. A counter rather than a boolean so two reloads in a
   *  row are two fetches. */
  const [nonce, setNonce] = useState(0);

  const kind = activeJob?.kind;
  const jobStatus = activeJob?.status;
  const running = jobStatus === 'queued' || jobStatus === 'running';
  const mine = kind === 'yue2-preprocess' || kind === 'yue2-nar-train';
  const finishedKey = mine && !running ? `${activeJob?.id ?? ''}:${jobStatus ?? ''}` : '';

  useEffect(() => {
    let cancelled = false;
    trainingApi.getYue2Status(datasetId)
      .then(s => { if (!cancelled) { setFetched({ datasetId, status: s }); setError(null); } })
      .catch(err => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)); });
    return () => { cancelled = true; };
  }, [datasetId, finishedKey, nonce]);

  return { status, error, reload: () => setNonce(n => n + 1) };
}
