// The existing Create page, run in Node: its stores, the bar's panels and CreatePanel (tools/real-app/page.ts), and
// the request it submits when Generate is pressed. tools/compare-request.ts hands it what a user did on the page
// (controls used, files chosen) and gets back the request, finished by the server's resolveCreateIntent() as both
// sides are.
//
// One line of the existing page cannot be run here, because it is in App.tsx, which draws the whole application:
// handleGenerate puts the bar's params under CreatePanel's and the app settings over both. It is restated below, and
// checked against its line's text, so a change there fails the comparison until this is read again.

import fs from 'node:fs';
import path from 'node:path';
import type { EngineId } from '../../src/definitions/engines.ts';
import type { Json } from '../../src/definitions/types.ts';
import { APP, FIX, SELECTION, SETTINGS, url } from './real.ts';

const RESTATED: [string, string][] = [
  ['ui/src/App.tsx', 'contentIntent ? { ...contentIntent, params: { ...engineParams, ...contentIntent.params,'],
];
for (const [file, line] of RESTATED) {
  if (!fs.readFileSync(path.join(APP, file), 'utf8').includes(line)) {
    console.log(`tools/real-app/app.ts restates "${line}" from ${file}, and that line has changed: read it again and update the restatement`);
    process.exit(1);
  }
}

/* the browser the existing code expects: its storage, and the backend store it reads */
const store = new Map<string, string>();
(globalThis as Record<string, unknown>).localStorage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => store.set(k, v), removeItem: (k: string) => store.delete(k) };
const real = { backend: {} as Record<string, unknown> };
(globalThis as Record<string, unknown>).__realApp = real;
const setBackend = (e: EngineId, selection: Record<string, string>) => {
  const hm = Object.fromEntries(Object.entries(selection).filter(([k]) => k.startsWith('heartmula.')).map(([k, v]) => [k.slice(10), v]));
  real.backend = {
    activeBackendId: e,
    backends: [{ id: e, displayName: e, ready: true }],
    capabilities: { [e]: { backend: e, core: { batch: { max: FIX.core[e]?.batchMax ?? 1 }, duration: { max: FIX.core[e]?.durationMax ?? 600 } }, features: FIX.features[e] ?? {}, extensions: FIX.extensions[e] ?? [] } },
    models: { heartmula: { defaults: hm } },
    fetchModels: async () => undefined, selectModels: async () => undefined, switchBackend: async () => undefined,
  };
};
setBackend('ace', SELECTION);

const { useGlobalParamsStore } = await import(url('ui/src/stores/globalParamsStore.ts'));
const INIT = useGlobalParamsStore.getState();
const page = await import('./page.ts');

type State = Record<string, unknown>;
/** Sets a store field as the page does when a file, a model or a seed is chosen in a dialog: through its setter
 *  (setBackendParam for a backend knob), a path into a field by setting the whole field again. */
function setStore(field: string, value: Json): void {
  const st = useGlobalParamsStore.getState() as State;
  const parts = field.split(/\.|\[(\d+)\]\.?/).filter(x => x !== undefined && x !== '');
  if (parts[0] === 'backendParams' && parts.length === 2) { (st.setBackendParam as (k: string, v: Json) => void)(parts[1]!, value); return; }
  const root = parts[0]!;
  const setter = st['set' + root[0]!.toUpperCase() + root.slice(1)];
  if (typeof setter === 'function') (setter as (v: unknown) => void)(value);
  else useGlobalParamsStore.setState({ [root]: value });
}

/** A fresh install of the existing page on an engine: empty storage but for every folded section of declared knobs
 *  opened (as a user opens one to reach a knob), the store at its defaults, the server's selection. */
export function resetStore(e: EngineId, selection: Record<string, string> = SELECTION): void {
  setBackend(e, selection);
  store.clear();
  const groups = new Map<string, Record<string, boolean>>();
  for (const x of (FIX.extensions[e] ?? []) as { group?: string; section?: string }[]) if (x.section) groups.set(x.group ?? 'generation', { ...groups.get(x.group ?? 'generation'), [x.section]: true });
  for (const [g, open] of groups) store.set(`hs-ext-sections:${e}:${g}`, JSON.stringify(open));
  useGlobalParamsStore.setState({ ...INIT, backendParams: {} });
}

/** What a user did on the existing page, in the page's terms: models the server has selected, store fields set by a
 *  dialog (a file, a seed), CreatePanel state the page keeps in its storage, the controls used in order, and the
 *  caption source CreatePanel would put in the intent. */
export interface Existing {
  selection: Record<string, string>;
  store: [field: string, value: Json][];
  storage: [key: string, value: Json][];
  use: [control: string, value: Json, when?: 'last'][];
  captionSource?: Record<string, Json>;
}

const clean = (o: unknown) => JSON.parse(JSON.stringify(o)) as Record<string, Json>;
/** The intent the existing page submits for that state: Generate pressed on CreatePanel (its handleGenerate), then
 *  App.tsx's handleGenerate (restated above). `missing`: the controls the page did not show to be used. */
export function existingIntent(e: EngineId, x: Existing): { intent: { kind: 'create'; params: Record<string, Json>; compose: Record<string, Json>; captionSource?: Record<string, Json> } | null; missing: string[]; why?: string } {
  page.fresh(() => resetStore(e, x.selection));
  for (const [k, v] of x.storage) store.set(k, JSON.stringify(v));
  for (const [field, value] of x.store) setStore(field, value);
  const { missing } = page.drive(e, x.use);
  const g = page.pressGenerate(e);
  if (typeof g === 'string') return { intent: null, missing, why: g };
  const engineParams = useGlobalParamsStore.getState().getGlobalParams();
  const intent = g.intent as { params: Record<string, unknown>; compose: Record<string, Json>; captionSource?: Record<string, Json> };
  return {
    missing,
    intent: clean({
      ...intent, kind: 'create',
      params: { ...engineParams, ...intent.params, ...SETTINGS.request },
      ...(x.captionSource ? { captionSource: x.captionSource } : {}),
    }) as never,
  };
}

/** Every control the existing page can show on an engine (page.ts explore). */
export const explore = (e: EngineId) => page.explore(e, () => resetStore(e));
