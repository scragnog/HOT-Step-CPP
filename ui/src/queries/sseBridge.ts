// sseBridge.ts — server-sent events drive the query cache.
//
// Streams stay the push channel; they never become a second fetch owner.
// An area maps each event to cache effects and the bridge applies them:
//
// - invalidate: keys to refetch. Bursts coalesce into one invalidation per
//   key per `coalesceMs` window.
// - update: validated in-place edits. Applied only when the entry exists and
//   the updater returns a value; returning undefined rejects the event.
// - once: a side effect keyed by id (a toast, a "job finished" refresh). Runs
//   at most once per QueryClient, so the server's replay on reconnect or on a
//   late join never repeats it.
//
// The bridge rides sharedEventSource (one connection per URL per tab), so it
// sits beside the existing log/text/audio stream reducers, not in place of
// them. The caller captures its scope when it starts the bridge: the URL and
// every key `map` builds come from that capture, and after teardown nothing
// the old stream delivers can touch the cache.

import type { QueryClient, QueryKey } from '@tanstack/react-query';
import { hashKey } from '@tanstack/react-query';
import { subscribeEventSource } from '../services/sharedEventSource';
import { invalidateKeys } from './invalidation';

export interface BridgeEffect {
  invalidate?: QueryKey[];
  update?: { key: QueryKey; apply: (old: unknown) => unknown }[];
  once?: { id: string; run: () => void };
}

export interface BridgeOptions<E> {
  url: string;
  map: (event: E) => BridgeEffect | undefined | void;
  /** Defaults to JSON.parse; a frame that fails to parse is skipped. */
  parse?: (data: string) => E | undefined;
  /** Authoritative snapshots to refetch after the stream drops and reopens. */
  reconnectInvalidate?: QueryKey[];
  coalesceMs?: number;
  reconnect?: boolean;
  /** Injection point for tests; defaults to the shared connection manager. */
  subscribe?: typeof subscribeEventSource;
  /** Every parsed frame, before cache effects. The training projection uses
   *  this to keep its store-owned samples/logs/previews in step. */
  onMessage?: (event: E) => void;
  /** The stream dropped. EventSource reconnects; a caller may probe once. */
  onError?: () => void;
}

const ONCE_CAP = 2000;
const onceLedger = new WeakMap<QueryClient, Set<string>>();

function claimOnce(client: QueryClient, id: string): boolean {
  let seen = onceLedger.get(client);
  if (!seen) onceLedger.set(client, seen = new Set());
  if (seen.has(id)) return false;
  seen.add(id);
  // ponytail: FIFO cap; a terminal event older than 2000 claims could replay.
  if (seen.size > ONCE_CAP) seen.delete(seen.values().next().value!);
  return true;
}

function defaultParse<E>(data: string): E | undefined {
  try { return JSON.parse(data) as E; } catch { return undefined; }
}

/** Start bridging a stream into `client`. Returns the teardown. */
export function bridgeEvents<E>(client: QueryClient, opts: BridgeOptions<E>): () => void {
  const parse = opts.parse ?? defaultParse<E>;
  const coalesceMs = opts.coalesceMs ?? 50;
  const pending = new Map<string, QueryKey>();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let closed = false;
  let dropped = false;

  const flush = () => {
    timer = undefined;
    if (closed) return;
    const keys = [...pending.values()];
    pending.clear();
    void invalidateKeys(client, keys);
  };
  const queue = (keys: QueryKey[] | undefined) => {
    if (!keys?.length) return;
    for (const k of keys) pending.set(hashKey(k), k);
    timer ??= setTimeout(flush, coalesceMs);
  };

  const unsubscribe = (opts.subscribe ?? subscribeEventSource)(opts.url, {
    open: () => {
      if (closed || !dropped) return;
      dropped = false;
      queue(opts.reconnectInvalidate);
    },
    error: () => { dropped = true; opts.onError?.(); },
    message: (data) => {
      if (closed) return;
      const event = parse(data);
      if (event === undefined) return;
      opts.onMessage?.(event);
      const effect = opts.map(event);
      if (!effect) return;
      for (const { key, apply } of effect.update ?? []) {
        const old = client.getQueryData(key);
        if (old === undefined) continue;
        const next = apply(old);
        if (next !== undefined) client.setQueryData(key, next);
      }
      queue(effect.invalidate);
      if (effect.once && claimOnce(client, effect.once.id)) effect.once.run();
    },
  }, { reconnect: opts.reconnect });

  return () => {
    if (closed) return;
    closed = true;
    clearTimeout(timer);
    pending.clear();
    unsubscribe();
  };
}
