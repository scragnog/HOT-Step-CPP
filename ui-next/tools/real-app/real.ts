// What the definitions' side of the request comparison is finished with: the real application's own request code,
// loaded in Node through hooks.mjs (registered by load.mjs), and the facts it reads. Both sides end in the server's
// resolveCreateIntent(), which is what /api/resolve/preview runs on what Create submits.
//
// Two of these live in the existing frontend: createContentParams() (ui/src/components/create/createContent.ts) and
// the app settings every request carries (DEFAULT_SETTINGS in ui/src/components/settings/SettingsPanel.tsx). The page
// will import them as the existing page does; before ui/ is removed they have to move with the page
// (docs/08-open-questions.md, question 4). The rest is the server's.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { RealApp } from '../../src/derive.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
/* the repository ui-next sits in; HOTSTEP_APP names it when ui-next is a copy somewhere else */
export const APP = process.env.HOTSTEP_APP ?? path.resolve(ROOT, '..');
export const url = (p: string) => 'file:///' + path.join(APP, p).split(path.sep).join('/');
export const FIX = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools', 'fixtures', 'real-app.json'), 'utf8'));

const { createContentParams, createContentFromDraft } = await import(url('ui/src/components/create/createContent.ts'));
const { resolveGenerationParams } = await import(url('server/src/services/generation/defaults.ts'));
const { resolveCreateIntent: resolveIntent } = await import(url('server/src/services/generation/resolve/resolveIntent.ts'));
/* DEFAULT_SETTINGS lives in a React component; the stand-in reads it from that file's text */
const { DEFAULT_SETTINGS } = await import(new URL('./stubs.mjs?m=SettingsPanel', import.meta.url).href);

/** The models the stand-in server reports as selected: real names, so a frozen model choice means something. */
export const SELECTION: Record<string, string> = {
  'heartmula.variant': 'heartmula-3b', 'heartmula.global': 'HeartMuLa-oss-3B-Q8_0.gguf', 'heartmula.local': 'HeartMuLa-local-Q8_0.gguf',
  'heartmula.flow': 'HeartCodec-flow-F16.gguf', 'heartmula.scalar_decoder': 'HeartCodec-scalar-F16.gguf', 'heartmula.lmAdapter': '',
};

export const SETTINGS = {
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
