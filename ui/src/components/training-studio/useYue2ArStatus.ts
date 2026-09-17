// useYue2ArStatus.ts — the YuE2 AR dataset status the composer card reads.
//
// The AR sibling of useYue2Status, and its own file for the same two reasons:
// the card then exports nothing but a component (fast refresh), and all three
// cache stages, their model readiness and the recipe arrive as ONE shape from
// ONE fetch, so no two stages of the card can disagree about what is on disk.
//
// Cache jobs move this answer where the NAR hook watches two. Stage 1 is the
// latent cache the NAR card writes, so a preprocess finishing while this card is
// on screen changes what it may offer — and tokenize and align each rewrite the
// same manifest every other stage is read out of.

import { useEffect, useState } from 'react';

import * as trainingApi from '../../services/trainingApi';
import type { Yue2ArStatus } from '../../services/trainingApi';
import { useTrainingStore } from '../../stores/trainingStore';

export function useYue2ArStatus(datasetId: string): {
  status: Yue2ArStatus | null; error: string | null; reload: () => void;
} {
  const activeJob = useTrainingStore(s => s.activeJob);
  const [status, setStatus] = useState<Yue2ArStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  /** Bumped by `reload`. A counter rather than a boolean so two reloads in a
   *  row are two fetches. */
  const [nonce, setNonce] = useState(0);

  const kind = activeJob?.kind;
  const jobStatus = activeJob?.status;
  const running = jobStatus === 'queued' || jobStatus === 'running';
  const mine = kind === 'yue2-preprocess' || kind === 'yue2-tokenize'
    || kind === 'yue2-stems' || kind === 'yue2-align' || kind === 'yue2-sheet' || kind === 'yue2-ar-train';
  const finishedKey = mine && !running ? `${activeJob?.id ?? ''}:${jobStatus ?? ''}` : '';

  useEffect(() => {
    let cancelled = false;
    trainingApi.getYue2ArStatus(datasetId)
      .then(s => { if (!cancelled) { setStatus(s); setError(null); } })
      .catch(err => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)); });
    return () => { cancelled = true; };
  }, [datasetId, finishedKey, nonce]);

  return { status, error, reload: () => setNonce(n => n + 1) };
}
