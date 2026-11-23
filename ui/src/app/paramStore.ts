// paramStore.ts — generation parameters by id, for shells (skin-api/params.ts) and the host's
// undo history. An id is a GENERATION_DEFAULTS key (`inferenceSteps`) or `backendParams.<key>`
// for a knob the active backend declares. Reads and writes go through globalParamsStore's own
// setters, so every hs-* key stays right and classic sees the same values.
//
// Only controls the global store owns can be set here. Content fields (caption, lyrics, bpm,
// keyScale...) belong to the Create request and go through generate().

import { GENERATION_DEFAULTS } from '../../../server/src/contracts/generationControls';
import { useGlobalParamsStore, setBackendParamFor } from '../stores/globalParamsStore';
import { useBackendStore } from '../stores/backendStore';
import { queryClient } from '../queries/client';
import { capabilitiesOptions } from '../queries/global-models/options';
import { createHistory, type ParamChange } from './paramHistory';

export const BACKEND_PREFIX = 'backendParams.';
type Store = Record<string, unknown> & { backendParams?: Record<string, unknown> };

const state = (): Store => useGlobalParamsStore.getState() as Store;
const activeBackend = () => useBackendStore.getState().activeBackendId || 'ace';
const setterName = (id: string) => `set${id[0].toUpperCase()}${id.slice(1)}`;

export const isBackendParam = (id: string) => id.startsWith(BACKEND_PREFIX);

/** True when the global store has a setter for the id (a content field has none). */
export function isEditableParam(id: string): boolean {
  if (isBackendParam(id)) return id.length > BACKEND_PREFIX.length;
  return id in GENERATION_DEFAULTS && id !== 'backendParams' && typeof state()[setterName(id)] === 'function';
}

export function readParam(s: Store, id: string): unknown {
  return isBackendParam(id) ? s.backendParams?.[id.slice(BACKEND_PREFIX.length)] : s[id];
}

export function defaultOf(id: string): unknown {
  if (!isBackendParam(id)) return (GENERATION_DEFAULTS as Record<string, unknown>)[id];
  const key = id.slice(BACKEND_PREFIX.length);
  return queryClient.getQueryData(capabilitiesOptions(activeBackend()).queryKey)?.extensions.find(e => e.key === key)?.default;
}

function write(id: string, value: unknown, backend = activeBackend()): void {
  if (isBackendParam(id)) setBackendParamFor(backend, id.slice(BACKEND_PREFIX.length), value);
  else (state()[setterName(id)] as (v: unknown) => void)(value);
}

export const paramHistory = createHistory((c: ParamChange, value) => write(c.id, value, c.backend));

export function setParam(id: string, value: unknown): void {
  if (!isEditableParam(id)) throw new Error(`"${id}" is not an editable parameter (content fields go through generate())`);
  const before = readParam(state(), id), backend = activeBackend();
  write(id, value);
  paramHistory.record(id, { id, backend, before, after: value });
}

export const resetParam = (id: string): void => setParam(id, defaultOf(id));
