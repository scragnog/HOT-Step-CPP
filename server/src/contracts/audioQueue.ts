// contracts/audioQueue.ts — wire schemas for the durable audio intent queue
// (/api/audio-queue). zod 4; clients import the inferred types.

import { z } from 'zod/v4';

/** POST /api/audio-queue/items */
export const enqueueAudioIntentSchema = z.object({
  /** Caller-chosen, unique per item. Repeating it returns the existing item
   *  instead of queuing a second one, so two clients (or a retried request)
   *  cannot submit the same item twice. */
  idempotencyKey: z.string().min(1).max(200),
  /** The exact body to submit to /api/generate, normally a resolve preview's
   *  `request`. Captured as is; it is never re-resolved. */
  request: z.record(z.string(), z.unknown()),
  /** Free-form client metadata kept with the item (group, title, playlist). */
  meta: z.record(z.string(), z.unknown()).optional(),
});
export type EnqueueAudioIntent = z.infer<typeof enqueueAudioIntentSchema>;

export type AudioIntentStatus =
  | 'pending'      // waiting to be submitted
  | 'submitting'   // claimed; the submit call is in flight
  | 'submitted'    // accepted by /api/generate; job running or queued
  | 'succeeded' | 'failed' | 'cancelled'
  | 'interrupted'; // the server restarted with this item submitting or submitted; outcome unknown

export interface AudioIntentItem {
  id: string;
  idempotencyKey: string;
  status: AudioIntentStatus;
  /** The engine the request was resolved for (its expectedBackend). */
  engine: string;
  request: Record<string, unknown>;
  meta: Record<string, unknown> | null;
  /** The generation job of the current attempt. */
  jobId: string | null;
  attempt: number;
  /** Job ids of earlier attempts, oldest first. */
  previousJobIds: string[];
  /** Why a pending item is not being submitted right now. */
  waiting: string | null;
  cancelRequested: boolean;
  error: string | null;
  result: Record<string, unknown> | null;
  createdAt: number;
  updatedAt: number;
}

export interface AudioQueueState {
  paused: boolean;
  /** Items submitted at once, at most (lets YuE2 coalesce a batch). */
  maxInFlight: number;
  counts: Partial<Record<AudioIntentStatus, number>>;
}
