// projection.ts — one Query-owned cadence for the server audio queue.
//
// The executor still sequences submissions, applies the engine-park rules
// and follows its own 2.5 s job read. This hook only owns the server queue
// snapshot: it polls at ACTIVE_POLL_MS while anything is waiting or running
// (including a studio row with no server item yet) and at an idle cadence
// otherwise, so another tab's work is still noticed. Every successful read
// goes through the store's single reducer.

import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  applyServerItems, hasActiveQueueWork, useAudioGenQueueSelector, useAudioQueueOwner,
} from '../../stores/audioGenQueueStore';
import { ACTIVE_POLL_MS } from '../contract';
import { audioQueueItemsOptions } from './queue';

/** The old seven-tick idle backoff, kept as one value. */
const IDLE_POLL_MS = ACTIVE_POLL_MS * 7;

const ACTIVE_SERVER_STATUS = new Set(['held', 'pending', 'submitting', 'submitted']);

/** Mount once, beside the app shell. Owner and activity are reactive, so the
 *  interval switches without remounting the query. */
export function useAudioQueueProjection(token?: string): void {
  const owner = useAudioQueueOwner();
  const localActive = useAudioGenQueueSelector(hasActiveQueueWork);
  const server = owner === 'server';
  const enabled = server && !!token;

  const { data } = useQuery({
    ...audioQueueItemsOptions(),
    enabled,
    refetchInterval: query => {
      if (!enabled) return false;
      const active = localActive
        || !query.state.data
        || query.state.data.items.some(item => ACTIVE_SERVER_STATUS.has(item.status));
      return active ? ACTIVE_POLL_MS : IDLE_POLL_MS;
    },
  });

  useEffect(() => {
    if (enabled && data) void applyServerItems(data.items);
  }, [enabled, data]);
}
