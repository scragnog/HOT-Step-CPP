// params.ts — read, write and undo generation parameters. Writes go through the same store
// setters classic uses (so the hs-* keys stay right); the undo history is the host's.

import { useCallback, useMemo, useSyncExternalStore } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useGlobalParamsStore } from '../stores/globalParamsStore';
import { useBackendStore } from '../stores/backendStore';
import { generationControlDescriptorsOptions, backendControlDescriptorsOptions } from '../queries/generationControlDescriptors';
import { defaultOf, snapshot, isBackendParam, isEditableParam, paramHistory, readParam, resetParam, setParam } from '../app/paramStore';
import type { ParamHistoryStep, ParamId, ParamMeta } from './types';

const sameValue = (a: unknown, b: unknown) => a === b || JSON.stringify(a) === JSON.stringify(b);

/** A parameter's value and its writers. `changed` is true when it differs from the default. */
export function useParam<T = unknown>(id: ParamId): { value: T; set: (v: T) => void; reset: () => void; changed: boolean } {
  const raw = useGlobalParamsStore((s: Record<string, unknown>) => readParam(s, id));
  // The store's own object stays private: the skin gets a copy that changes only when the value does.
  const value = useMemo(() => snapshot(raw), [raw]) as T;
  // Re-render when the active backend changes: a backend knob's default can differ.
  useBackendStore(s => s.activeBackendId);
  const set = useCallback((v: T) => setParam(id, v), [id]);
  const reset = useCallback(() => resetParam(id), [id]);
  return { value, set, reset, changed: !sameValue(value, defaultOf(id)) };
}

/** What a control is: label, type, range, default. Undefined while loading or for an unknown id. */
export function useParamMeta(id: ParamId): ParamMeta | undefined {
  const backend = useBackendStore(s => s.activeBackendId) || 'ace';
  const core = useQuery({ ...generationControlDescriptorsOptions(), enabled: !isBackendParam(id) });
  const ext = useQuery({ ...backendControlDescriptorsOptions(backend), enabled: isBackendParam(id) });
  return useMemo(() => {
    const d = isBackendParam(id) ? ext.data?.find(x => x.key === id) : core.data?.find(x => x.key === id);
    if (!d) return undefined;
    return {
      id, type: d.type, label: d.label, hint: d.hint, default: d.default,
      min: d.min, max: d.max, step: d.step, options: d.options?.map(o => ({ value: o.value, label: o.label })),
      editable: isEditableParam(id),
    };
  }, [id, core.data, ext.data]);
}

/** Write helpers. Every set is one undo step; a drag on one control collapses into one. */
export const params = {
  set: (id: ParamId, value: unknown): void => setParam(id, value),
  reset: (id: ParamId): void => resetParam(id),
  /** Everything set inside `fn` is one undo step called `label`. */
  transaction: <T>(label: string, fn: () => T): T => paramHistory.transaction(label, fn),
};

/** Undo and redo for params.set. Host-owned, in memory, 100 steps. `steps` is oldest first;
 *  those at index >= `cursor` are undone. */
export function useParamHistory(): {
  undo: () => void; redo: () => void; canUndo: boolean; canRedo: boolean;
  steps: readonly ParamHistoryStep[]; cursor: number;
} {
  const s = useSyncExternalStore(paramHistory.subscribe, paramHistory.getSnapshot);
  return { undo: paramHistory.undo, redo: paramHistory.redo, canUndo: s.canUndo, canRedo: s.canRedo, steps: s.steps, cursor: s.cursor };
}
