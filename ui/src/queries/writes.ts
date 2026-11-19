// writes.ts — the shared write layer (P2.3, docs/dev/tanstack-query.md).
//
// Two helpers replace the hand-rolled write lifecycles scattered across stores
// and components:
//
// - writeAndInvalidate: run a command, then invalidate what it dirtied. The
//   plan argument is required — a write that forgets invalidation cannot be
//   expressed here (the F4#11 class). Named events resolve through the
//   manager's event map; parameterized writes pass a closure over its groups.
// - useUpload: one mutation over apiClient.upload — progress, abort and the
//   shared failure handling every upload used to hand-roll.

import { useCallback, useRef, useState } from 'react';
import { useMutation, useQueryClient, type QueryClient, type QueryKey } from '@tanstack/react-query';
import { apiClient } from '@/services/httpClient';
import { invalidateEvents, queryManager, type EventRef } from './invalidation';

/** What a write dirtied: named events, or a direct plan for parameterized keys. */
export type WritePlan = (client: QueryClient) => Promise<unknown> | unknown;

/** Compose a plan from named events (queryManager.events). */
export function events(...refs: EventRef[]): WritePlan {
  return client => invalidateEvents(client, refs);
}

/** A parameterized plan: the write knows its own keys (a worker scope, an id)
 *  and names them here; the manager still owns the invalidateQueries call. */
export function keysPlan(...keys: QueryKey[]): WritePlan {
  return client => queryManager.invalidate(client, ...keys);
}

/** Run a command and invalidate what it dirtied. The plan is required: a
 *  write that forgets invalidation cannot be expressed here. */
export async function writeAndInvalidate<T>(client: QueryClient, fn: () => Promise<T>, plan: WritePlan): Promise<T> {
  const result = await fn();
  await plan(client);
  return result;
}

export interface UploadInput<T = unknown> {
  /** Path under the client base, e.g. '/upload/latent'. */
  path: string;
  /** `name` names a plain Blob (a File uses its own name). */
  files: { field: string; file: File | Blob; name?: string }[];
  fields?: Record<string, string>;
  /** Invalidated after a successful upload (see WritePlan). */
  plan?: WritePlan;
  /** Extra progress sink beside the hook's own `progress` state. */
  onProgress?: (fraction: number) => void;
  /** Per-call success handler — contextual UI only, never toasts (the global
   *  handler owns error notification). */
  onSuccess?: (data: T) => void;
  /** Per-call failure handler for inline error text. Without `silent`,
   *  transport failures are also toasted globally — pass `silent: true` when
   *  this handler renders the failure itself, so it is the only notifier. */
  onError?: (err: unknown) => void;
  /** Skip the global failure toast (the caller's `onError` renders it). */
  silent?: boolean;
}

/** The one upload mutation: progress + abort over apiClient.upload, with the
 *  same failure handling as every other request (the XHR path routes through
 *  handleFailure). Aborts reject with an AbortError and never toast. */
export function useUpload<T = unknown>() {
  const queryClient = useQueryClient();
  const controllerRef = useRef<AbortController | null>(null);
  const [progress, setProgress] = useState(0);

  const mutation = useMutation({
    mutationFn: async (input: UploadInput<T>): Promise<T> => {
      const controller = new AbortController();
      controllerRef.current = controller;
      setProgress(0);
      return apiClient.upload<T>(input.path, input.files, {
        fields: input.fields,
        signal: controller.signal,
        silent: input.silent,
        onProgress: fraction => { setProgress(fraction); input.onProgress?.(fraction); },
      });
    },
    onSuccess: (data, input) => { input.onSuccess?.(data); if (input.plan) void input.plan(queryClient); },
    onError: (err, input) => { input.onError?.(err); },
  });

  const abort = useCallback(() => { controllerRef.current?.abort(); }, []);

  return { ...mutation, progress, abort };
}
