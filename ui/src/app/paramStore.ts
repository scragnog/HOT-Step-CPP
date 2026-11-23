// paramStore.ts — generation parameters by id, for shells (skin-api/params.ts) and the host's
// undo history. An id is a GENERATION_DEFAULTS key (`inferenceSteps`) or `backendParams.<key>`
// for a knob the active backend declares. Reads and writes go through globalParamsStore's own
// setters, so every hs-* key stays right and classic sees the same values.
//
// Only controls the global store owns can be set here. Content fields (caption, lyrics, bpm,
// keyScale...) belong to the Create request and go through generate().

import { GENERATION_DEFAULTS } from '../../../server/src/contracts/generationControls';
import { useGlobalParamsStore, setBackendParamFor, scopedKey } from '../stores/globalParamsStore';
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

/** Declared ids only: a GENERATION_DEFAULTS key or a backend knob. Anything else (the store's
 *  setters and other actions included) reads as undefined. */
export function readParam(s: Store, id: string): unknown {
  if (isBackendParam(id)) {
    const bag = s.backendParams, key = id.slice(BACKEND_PREFIX.length);
    return bag && Object.hasOwn(bag, key) ? bag[key] : undefined;
  }
  return Object.hasOwn(GENERATION_DEFAULTS, id) && id !== 'backendParams' && typeof s[id] !== 'function' ? s[id] : undefined;
}

/** A copy of a structured value, so a caller can never mutate the store's own object. */
export const snapshot = <T>(v: T): T => (v !== null && typeof v === 'object' ? structuredClone(v) : v);

export function defaultOf(id: string): unknown {
  if (!isBackendParam(id)) return (GENERATION_DEFAULTS as Record<string, unknown>)[id];
  const key = id.slice(BACKEND_PREFIX.length);
  return queryClient.getQueryData(capabilitiesOptions(activeBackend()).queryKey)?.extensions.find(e => e.key === key)?.default;
}

// Core controls the store remembers per backend (hs-*@<backend>); see BACKEND_SCOPED_FIELDS there.
const SCOPED = new Set(['inferMethod', 'scheduler', 'guidanceMode']);

/** Write through the store setter when `backend` is the active one. A scoped control of another
 *  backend goes straight to that backend's slot (the live store holds only the active backend's). */
function write(id: string, value: unknown, backend: string): void {
  if (isBackendParam(id)) setBackendParamFor(backend, id.slice(BACKEND_PREFIX.length), value);
  else if (SCOPED.has(id) && backend !== activeBackend()) {
    try { localStorage.setItem(scopedKey(`hs-${id}`, backend), JSON.stringify(value)); } catch { /* full */ }
  } else (state()[setterName(id)] as (v: unknown) => void)(value);
}

export const paramHistory = createHistory((c: ParamChange, value) => write(c.id, snapshot(value), c.backend));

export function setParam(id: string, value: unknown): void {
  if (!isEditableParam(id)) throw new Error(`"${id}" is not an editable parameter (content fields go through generate())`);
  const before = snapshot(readParam(state(), id)), backend = activeBackend(), after = snapshot(value);
  write(id, after, backend);
  paramHistory.record(id, { id, backend, before, after });
}

export const resetParam = (id: string): void => setParam(id, defaultOf(id));
