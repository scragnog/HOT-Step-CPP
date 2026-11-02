// useTrainingTarget.ts — the one effective training target of the studio.
//
// Call it once, in TrainingStudio, and pass the result down; chips, preparation,
// train and previews all follow it. The override persists under the key the old
// TrainPanel picker used, so a saved choice survives. usePersistedState would
// not do here: its instances only sync across browser tabs, and the picker and
// the stepper live in the same one.

import { useCallback, useSyncExternalStore } from 'react';
import { useBackendStore } from '../../stores/backendStore';
import { useActiveTrainingJob } from '../../queries/training-runs/hooks';
import {
  TRAINING_TARGET_STORAGE_KEY, isTargetPinned, parseTargetOverride, resolveTrainingTarget,
  type TrainingTargetOverride,
} from './trainingTargets';
import type { TrainingTargetId } from './trainingPluginTypes';

function readOverride(): TrainingTargetOverride {
  try {
    const raw = localStorage.getItem(TRAINING_TARGET_STORAGE_KEY);
    return parseTargetOverride(raw === null ? null : JSON.parse(raw));
  } catch {
    return 'active';
  }
}

let current: TrainingTargetOverride = readOverride();
const listeners = new Set<() => void>();

function setOverride(next: TrainingTargetOverride): void {
  if (next === current) return;
  current = next;
  try { localStorage.setItem(TRAINING_TARGET_STORAGE_KEY, JSON.stringify(next)); } catch { /* per-viewer convenience only */ }
  listeners.forEach(l => l());
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  const onStorage = (e: StorageEvent) => {
    if (e.key !== TRAINING_TARGET_STORAGE_KEY) return;
    const next = readOverride();
    if (next !== current) { current = next; listeners.forEach(l => l()); }
  };
  window.addEventListener('storage', onStorage);
  return () => { listeners.delete(listener); window.removeEventListener('storage', onStorage); };
}

export interface TrainingTargetState {
  /** The target every view follows. */
  target: TrainingTargetId;
  override: TrainingTargetOverride;
  setOverride: (next: TrainingTargetOverride) => void;
  /** True while any queued/running target-bound job pins the target; the picker is read-only then. */
  locked: boolean;
}

export function useTrainingTarget(): TrainingTargetState {
  const override = useSyncExternalStore(subscribe, () => current);
  const activeBackendId = useBackendStore(s => s.activeBackendId);
  const job = useActiveTrainingJob();
  const running = job?.status === 'queued' || job?.status === 'running';
  const runningJobKind = running ? job?.kind : null;
  const target = resolveTrainingTarget({ override, activeBackendId, runningJobKind });
  const set = useCallback((next: TrainingTargetOverride) => setOverride(next), []);
  return { target, override, setOverride: set, locked: isTargetPinned(runningJobKind) };
}
