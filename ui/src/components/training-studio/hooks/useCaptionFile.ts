// useCaptionFile.ts — one sample caption file (MM3 or YuE2) as an editor seed.

import { useEffect, useRef, useState } from 'react';
import { hashKey, useQuery, useQueryClient } from '@tanstack/react-query';
import type { trainingQueries } from '@/queries/training-data/queries';

/**
 * Query owns the read (and cancels it when the drawer moves to another sample
 * or worker), but the editor is seeded only by the first answer fetched on
 * each visit to a key: never a cached copy, and never a later refetch that
 * could write over the user's typing. Returning to a sample is a new visit,
 * so it seeds again from that visit's fresh answer.
 */
export function useCaptionFile(options: ReturnType<typeof trainingQueries.sampleMm3>, enabled: boolean) {
  const client = useQueryClient();
  const query = useQuery({ ...options, enabled, refetchOnReconnect: false });
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const savedRef = useRef('');
  const seeded = useRef(false);
  const hash = hashKey(options.queryKey);
  // A new visit. StrictMode's repeated mount effect only clears an unseeded
  // editor again, before any answer has arrived.
  useEffect(() => { seeded.current = false; setText(null); setError(null); }, [hash]);
  useEffect(() => {
    if (seeded.current || !query.isFetchedAfterMount || query.isFetching) return;
    seeded.current = true;
    if (query.data) { savedRef.current = query.data.text; setText(query.data.text); }
    else { setText(''); setError(query.error?.message ?? null); }
  }, [hash, query.isFetchedAfterMount, query.isFetching, query.data, query.error]);
  /** A confirmed write: the cache takes the server's answer. */
  const saved = (value: string) => {
    savedRef.current = value;
    client.setQueryData(options.queryKey, { text: value });
  };
  return { text, setText, error, setError, savedRef, saved };
}
