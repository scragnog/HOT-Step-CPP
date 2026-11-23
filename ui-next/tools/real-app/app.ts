// The real application's request code, loaded in Node (through hooks.mjs, registered by load.mjs), and the
// existing Create page's request built from it for any state of its stores. Both sides of tools/compare-request.ts
// end in the server's own resolveCreateIntent(), which is what /api/resolve/preview runs on what Create submits.
//
// What the existing page does in React components, and so cannot be imported, is restated here with the line it
// comes from, and checked against that line's text so a change there fails the comparison until this is read again:
//   CreatePanel.tsx  the `compose` block (YuE2 and HeartMuLa blank the trigger, auto-expand and Beat I/O), the
//                    content it hands createContentParams(), and the intent's params and captionSource
//   App.tsx          handleGenerate: the global params, then the intent's params, then the app settings

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { EngineId } from '../../src/definitions/engines.ts';
import type { Json } from '../../src/definitions/types.ts';
import type { RealApp } from '../../src/derive.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
/* the repository ui-next sits in; HOTSTEP_APP names it when ui-next is a copy somewhere else */
export const APP = process.env.HOTSTEP_APP ?? path.resolve(ROOT, '..');
const url = (p: string) => 'file:///' + path.join(APP, p).replace(/\\/g, '/');
export const FIX = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools', 'fixtures', 'real-app.json'), 'utf8'));

/* the lines restated below; each must still be in its file */
const RESTATED: [string, string][] = [
  ['ui/src/components/create/CreatePanel.tsx', "const compose = yue2Ui || activeBackendId === 'heartmula'"],
  ['ui/src/components/create/CreatePanel.tsx', '? { autoExpand: false, loraTrigger: \'\', beatIntro: false, introBars }'],
  ['ui/src/components/create/CreatePanel.tsx', 'caption, lyrics, negativePrompt, instrumental, loraTrigger: compose.loraTrigger, beatIntro: compose.beatIntro, introBars,'],
  ['ui/src/components/create/CreatePanel.tsx', 'params: { ...legacyParams, caption, lyrics },'],
  ['ui/src/App.tsx', 'contentIntent ? { ...contentIntent, params: { ...engineParams, ...contentIntent.params,'],
];
for (const [file, line] of RESTATED) {
  if (!fs.readFileSync(path.join(APP, file), 'utf8').includes(line)) {
    console.log(`tools/real-app/app.ts restates "${line}" from ${file}, and that line has changed: read it again and update the restatement`);
    process.exit(1);
  }
}

/* the browser the existing code expects: empty storage (a fresh install), and the backend store it reads */
const store = new Map<string, string>();
(globalThis as Record<string, unknown>).localStorage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => store.set(k, v), removeItem: (k: string) => store.delete(k) };
const real = { backend: {} as Record<string, unknown> };
(globalThis as Record<string, unknown>).__realApp = real;
/** The models the stand-in server reports as selected: real names, so a frozen model choice means something. */
export const SELECTION: Record<string, string> = {
  'heartmula.variant': 'heartmula-3b', 'heartmula.global': 'HeartMuLa-oss-3B-Q8_0.gguf', 'heartmula.local': 'HeartMuLa-local-Q8_0.gguf',
  'heartmula.flow': 'HeartCodec-flow-F16.gguf', 'heartmula.scalar_decoder': 'HeartCodec-scalar-F16.gguf', 'heartmula.lmAdapter': '',
};
const setBackend = (e: EngineId, selection: Record<string, string>) => {
  const hm = Object.fromEntries(Object.entries(selection).filter(([k]) => k.startsWith('heartmula.')).map(([k, v]) => [k.slice(10), v]));
  real.backend = {
    activeBackendId: e,
    capabilities: { [e]: { core: { batch: { max: FIX.core[e]?.batchMax ?? 1 }, duration: { max: FIX.core[e]?.durationMax ?? 600 } } } },
    models: { heartmula: { defaults: hm } },
  };
};
setBackend('ace', SELECTION);

const { useGlobalParamsStore } = await import(url('ui/src/stores/globalParamsStore.ts'));
const { createContentParams, createContentFromDraft } = await import(url('ui/src/components/create/createContent.ts'));
const { resolveGenerationParams } = await import(url('server/src/services/generation/defaults.ts'));
const { resolveCreateIntent: resolveIntent } = await import(url('server/src/services/generation/resolve/resolveIntent.ts'));
/* DEFAULT_SETTINGS lives in a React component; the stand-in reads it from that file's text */
const { DEFAULT_SETTINGS } = await import(new URL('./stubs.mjs?m=SettingsPanel', import.meta.url).href);
const INIT = useGlobalParamsStore.getState();

const SETTINGS = {
  trigger: { triggerUseFilename: DEFAULT_SETTINGS.triggerUseFilename, triggerPlacement: DEFAULT_SETTINGS.triggerPlacement },
  request: { source: 'create', coResident: DEFAULT_SETTINGS.coResident, cacheLmCodes: DEFAULT_SETTINGS.cacheLmCodes, parallelWhisper: DEFAULT_SETTINGS.parallelWhisper, parallelQualityEval: DEFAULT_SETTINGS.parallelQualityEval, parallelCoverArt: DEFAULT_SETTINGS.parallelCoverArt },
};

/** What the page is handed to finish a request: the real functions and facts. */
export const REAL: RealApp = {
  resolveGenerationParams, createContentParams,
  contentDefaults: createContentFromDraft({}), generationDefaults: FIX.generationDefaults,
  extensions: FIX.extensions, durationMax: Object.fromEntries(Object.entries(FIX.core as Record<string, { durationMax: number | null }>).map(([e, c]) => [e, c.durationMax])),
  settings: SETTINGS,
};

export const resolveCreateIntent = resolveIntent as (intent: unknown, engine: string, data?: unknown) => { request: Record<string, unknown> };

/** One state of the existing Create page: what is set in its global params store (through the store's own setters,
 *  in order), the CreatePanel content and compose switches, the caption source it holds, and the models the server has
 *  selected. */
export interface Existing {
  store: [field: string, value: Json][];
  content: Record<string, Json>;
  panel: { autoExpand?: boolean; loraTrigger?: string; beatIntro?: boolean; introBars?: number };
  captionSource?: Record<string, Json>;
  selection: Record<string, string>;
}

const clean = (o: unknown) => JSON.parse(JSON.stringify(o)) as Record<string, Json>;
type State = Record<string, unknown>;
/** Sets a store field the way its control does: through its setter (setBackendParam for a backend knob), a path into
 *  a field by setting the whole field again. */
function setStore(field: string, value: Json): void {
  const st = useGlobalParamsStore.getState() as State;
  const parts = field.split(/\.|\[(\d+)\]\.?/).filter(x => x !== undefined && x !== '');
  if (parts[0] === 'backendParams' && parts.length === 2) { (st.setBackendParam as (k: string, v: Json) => void)(parts[1]!, value); return; }
  const root = parts[0]!;
  let next: unknown = value;
  if (parts.length > 1) {
    next = structuredClone(st[root] ?? (/^\d+$/.test(parts[1]!) ? [] : {}));
    let at = next as Record<string, unknown>;
    for (let i = 1; i < parts.length - 1; i++) { if (at[parts[i]!] === undefined) return; at = at[parts[i]!] as Record<string, unknown>; }
    if (Array.isArray(at) && at[Number(parts[parts.length - 1])] === undefined) return;
    at[parts[parts.length - 1]!] = value;
  }
  const setter = st['set' + root[0]!.toUpperCase() + root.slice(1)];
  if (typeof setter === 'function') (setter as (v: unknown) => void)(next);
  else useGlobalParamsStore.setState({ [root]: next });
}

/** The intent the existing Create page submits for that state (CreatePanel.tsx createIntent, App.tsx handleGenerate). */
export function existingIntent(e: EngineId, x: Existing): { kind: 'create'; params: Record<string, Json>; compose: Record<string, Json>; captionSource?: Record<string, Json> } {
  setBackend(e, x.selection);
  store.clear();
  useGlobalParamsStore.setState({ ...INIT, backendParams: {} });
  try {
    for (const [field, value] of x.store) setStore(field, value);
    const p = x.panel, introBars = p.introBars ?? 2;
    const compose = e === 'yue2' || e === 'heartmula'
      ? { autoExpand: false, loraTrigger: '', beatIntro: false, introBars }
      : { autoExpand: p.autoExpand ?? false, loraTrigger: p.loraTrigger ?? '', beatIntro: p.beatIntro ?? false, introBars };
    const c = { ...createContentFromDraft({}), ...x.content };
    const legacyParams = createContentParams({ ...c, loraTrigger: compose.loraTrigger, beatIntro: compose.beatIntro, introBars },
      { mm3Mode: e === 'minimax-m3', yue2Active: e === 'yue2', durationMax: FIX.core[e]?.durationMax ?? undefined });
    const engineParams = useGlobalParamsStore.getState().getGlobalParams();
    return clean({
      kind: 'create',
      params: { ...engineParams, ...legacyParams, caption: c.caption, lyrics: c.lyrics, ...SETTINGS.request },
      compose,
      ...(x.captionSource ? { captionSource: x.captionSource } : {}),
    }) as ReturnType<typeof existingIntent>;
  } finally {
    useGlobalParamsStore.setState({ ...INIT, backendParams: {} });
    store.clear();
  }
}

/** The store's own value for a field before anything is set (GLOBAL_PARAM_DEFAULTS, as a fresh install reads it). */
export const storeDefault = (field: string): unknown => (INIT as Record<string, unknown>)[field];
