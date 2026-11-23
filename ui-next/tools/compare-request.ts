// The request built from the definitions, compared field by field with what the existing frontend sends, for every
// engine at default values. The existing code is the ground truth: getGlobalParams() (ui/src/stores/
// globalParamsStore.ts) and createContentParams() (ui/src/components/create/createContent.ts), joined as App.tsx
// joins them for Create. The definitions' side runs derive.ts realState(), then the real application's own
// resolveGenerationParams() (server/src/services/generation/defaults.ts) and createContentParams().
//
//   node --import ./tools/real-app/load.mjs tools/compare-request.ts            fails on a difference not in
//                                                                               tools/fixtures/request-differences.json
//   node --import ./tools/real-app/load.mjs tools/compare-request.ts --print    prints every difference
//
// (npm run request). Each accepted difference in the fixture says why it is acceptable; a new one, or one that has
// gone away, fails the run until the fixture says so.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFINITIONS } from '../src/definitions/index.ts';
import { ENGINES, type EngineId } from '../src/definitions/engines.ts';
import { indexOf, realState, realDefaultsFrom, type Desk } from '../src/derive.ts';
import type { Json } from '../src/definitions/types.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
/* the repository ui-next sits in; HOTSTEP_APP names it when ui-next is a copy somewhere else */
const APP = process.env.HOTSTEP_APP ?? path.resolve(ROOT, '..');
const url = (p: string) => 'file:///' + path.join(APP, p).replace(/\\/g, '/');
const FIX = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools', 'fixtures', 'real-app.json'), 'utf8'));
const EXPECT_FILE = path.join(ROOT, 'tools', 'fixtures', 'request-differences.json');

/* the browser the existing code expects: empty storage (a fresh install), and the backend store it reads */
const store = new Map<string, string>();
(globalThis as Record<string, unknown>).localStorage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => store.set(k, v), removeItem: (k: string) => store.delete(k) };
const real = { backend: {} as Record<string, unknown> };
(globalThis as Record<string, unknown>).__realApp = real;
const BATCH: Record<string, number> = { ace: 8 };
const setEngine = (e: EngineId) => {
  real.backend = {
    activeBackendId: e,
    capabilities: { [e]: { core: { batch: { max: BATCH[e] ?? 1 } } } },
    /* HeartMuLa's frozen model choices come from the server's selection; at defaults it reports none */
    models: { heartmula: { defaults: { variant: '', global: '', local: '', flow: '', scalar_decoder: '', lmAdapter: '' } } },
  };
};
setEngine('ace');

const { useGlobalParamsStore } = await import(url('ui/src/stores/globalParamsStore.ts'));
const { createContentParams, createContentFromDraft } = await import(url('ui/src/components/create/createContent.ts'));
const { resolveGenerationParams } = await import(url('server/src/services/generation/defaults.ts'));
/* DEFAULT_SETTINGS lives in a React component; the stand-in reads it from that file's text */
const { DEFAULT_SETTINGS } = await import(new URL('./real-app/stubs.mjs?m=SettingsPanel', import.meta.url).href);
const GD = FIX.generationDefaults as Record<string, Json>;

/* what App.tsx adds to every Create request from the app's settings: the same on both sides, not the desk's */
const enrich = { source: 'create', coResident: DEFAULT_SETTINGS.coResident, cacheLmCodes: DEFAULT_SETTINGS.cacheLmCodes, parallelWhisper: DEFAULT_SETTINGS.parallelWhisper, parallelQualityEval: DEFAULT_SETTINGS.parallelQualityEval, parallelCoverArt: DEFAULT_SETTINGS.parallelCoverArt };
const contentOpts = (e: EngineId) => ({ mm3Mode: e === 'minimax-m3', yue2Active: e === 'yue2' });
const clean = (o: Record<string, unknown>) => JSON.parse(JSON.stringify(o)) as Record<string, Json>;

function legacy(e: EngineId): Record<string, Json> {
  setEngine(e);
  return clean({ ...useGlobalParamsStore.getState().getGlobalParams(), ...createContentParams(createContentFromDraft({}), contentOpts(e)), ...enrich });
}

const ix = indexOf(DEFINITIONS);
function ours(e: EngineId): { body: Record<string, Json>; ext: Record<string, Json> } {
  const desk: Desk = { engine: e, tier: 'lab', vals: {}, auto: {}, state: {}, facts: {}, on: new Set(DEFINITIONS.map(d => d.id)) };
  const rs = realState(ix, desk, f => GD[f], realDefaultsFrom(GD, FIX.extensions[e] ?? []));
  const resolved = resolveGenerationParams({ ...rs.params, backendParams: rs.backendParams }, { backendId: e, settings: { triggerUseFilename: DEFAULT_SETTINGS.triggerUseFilename, triggerPlacement: DEFAULT_SETTINGS.triggerPlacement } });
  const override = ENGINES.find(x => x.id === e)!;
  const content = createContentParams({ ...createContentFromDraft({}), ...rs.content }, contentOpts(e));
  return { body: clean({ ...('jobOverride' in override ? { backendOverride: e } : {}), ...resolved, ...rs.body, ...content, ...enrich }), ext: rs.backendParams };
}

interface Diff { engine: string; field: string; existing?: Json; definitions?: Json; kind: string }
const all: Diff[] = [];
const J = (x: unknown) => JSON.stringify(x);
for (const { id: e } of ENGINES) {
  const a = legacy(e), { body: b, ext } = ours(e);
  for (const f of [...new Set([...Object.keys(a), ...Object.keys(b)])].sort()) {
    if (J(a[f]) === J(b[f])) continue;
    let kind = 'value differs';
    if (!(f in b)) kind = 'only the existing UI sends it';
    else if (!(f in a)) {
      const x = (FIX.extensions[e] ?? []).find((y: { key: string }) => y.key === f);
      kind = f in ext ? (x && J(x.default) === J(b[f]) ? 'only the definitions send it: the server declares this default' : x ? 'only the definitions send it: the server declares another default' : 'only the definitions send it: not declared') : 'only the definitions send it';
    }
    all.push({ engine: e, field: f, ...(f in a ? { existing: a[f] } : {}), ...(f in b ? { definitions: b[f] } : {}), kind });
  }
}

const expected: (Diff & { why: string })[] = fs.existsSync(EXPECT_FILE) ? JSON.parse(fs.readFileSync(EXPECT_FILE, 'utf8')).differences : [];
const key = (d: Diff) => `${d.engine}|${d.field}|${J(d.existing)}|${J(d.definitions)}`;
const known = new Map(expected.map(d => [key(d), d]));
const fresh = all.filter(d => !known.has(key(d))), gone = expected.filter(d => !all.some(x => key(x) === key(d)));

if (process.argv.includes('--print')) {
  for (const { id: e } of ENGINES) {
    const mine = all.filter(d => d.engine === e);
    const byKind = new Map<string, Diff[]>(); mine.forEach(d => byKind.set(d.kind, [...(byKind.get(d.kind) || []), d]));
    console.log(`\n${e}: ${Object.keys(legacy(e)).length} fields from the existing UI, ${Object.keys(ours(e).body).length} from the definitions, ${mine.length} differences`);
    for (const [k, list] of byKind) { console.log(`  ${k} (${list.length}):`); list.forEach(d => console.log(`    ${d.field}: ${'existing' in d ? J(d.existing) : '-'} | ${'definitions' in d ? J(d.definitions) : '-'}${known.get(key(d))?.why ? `   ${known.get(key(d))!.why}` : ''}`)); }
  }
}
if (process.argv.includes('--write')) {
  const why = new Map(expected.map(d => [key(d), d.why]));
  fs.writeFileSync(EXPECT_FILE, JSON.stringify({ note: 'Differences between the request the definitions build and the one the existing UI sends, at defaults. Each says why it is accepted. tools/compare-request.ts checks this list.', differences: all.map(d => ({ ...d, why: why.get(key(d)) ?? 'TODO' })) }, null, 1).replace(/\n/g, '\r\n') + '\r\n');
}
console.log(`request at defaults, ${ENGINES.length} engines: ${all.length} differences from the existing UI, ${all.length - fresh.length} accepted in tools/fixtures/request-differences.json.`);
if (fresh.length || gone.length || expected.some(d => d.why === 'TODO')) {
  fresh.forEach(d => console.log(`  new: ${d.engine} ${d.field}: existing ${J(d.existing)}, definitions ${J(d.definitions)} (${d.kind})`));
  gone.forEach(d => console.log(`  gone: ${d.engine} ${d.field}: the list accepts a difference that is no longer there`));
  expected.filter(d => d.why === 'TODO').forEach(d => console.log(`  unexplained: ${d.engine} ${d.field}`));
  process.exit(1);
}
