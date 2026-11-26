// Stand-ins for the modules hooks.mjs replaces. One file, chosen by the ?m= query.
//   zustand          a store with getState, setState and subscribe, enough for globalParamsStore.ts
//   zod/v4           a chainable no-op, enough for the schema declarations in generationControls.ts
//   SettingsPanel    DEFAULT_SETTINGS, read from the real file's text
//   backendStore     the active backend and its capabilities, set through globalThis.__realApp
//   httpClient       an apiClient whose calls resolve to nothing
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const which = new URL(import.meta.url).searchParams.get('m');

function makeStore(init) {
  let state; const subs = [];
  const set = p => { state = { ...state, ...(typeof p === 'function' ? p(state) : p) }; subs.forEach(f => f(state)); };
  const get = () => state;
  state = init(set, get);
  const hook = sel => (sel ? sel(state) : state);
  hook.getState = get; hook.setState = set; hook.subscribe = f => { subs.push(f); return () => {}; };
  return hook;
}
const chain = () => new Proxy(function () {}, { get: () => chain(), apply: () => chain() });

const UI = process.env.HOTSTEP_APP ? `${process.env.HOTSTEP_APP}/ui/src/` : fileURLToPath(new URL('../../../ui/src/', import.meta.url));
function defaultSettings() {
  const src = fs.readFileSync(UI + 'components/settings/SettingsPanel.tsx', 'utf8');
  const m = src.match(/export const DEFAULT_SETTINGS: AppSettings = (\{[\s\S]*?\n\});/);
  return Function(`return (${m[1]});`)();
}

/* create(init) and create()(init) */
export const create = init => (typeof init === 'function' ? makeStore(init) : i => makeStore(i));
export const z = chain();
export const DEFAULT_SETTINGS = which === 'SettingsPanel' ? defaultSettings() : undefined;
export const useBackendStore = Object.assign(sel => (sel ? sel(globalThis.__realApp.backend) : globalThis.__realApp.backend), { getState: () => globalThis.__realApp.backend, setState: () => {}, subscribe: () => () => {} });
/* backendStore: a model choice not yet confirmed by the server: none */
export const pendingSelection = () => ({});
export const apiClient = { put: async () => undefined, get: async () => undefined, post: async () => undefined };
