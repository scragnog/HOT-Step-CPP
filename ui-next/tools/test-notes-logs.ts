// Tests of the core's addition for the engine log (package 11, src/derive/logs.ts engineLogModel), each against what the
// concept's log says (34-engine-panel.js paintLog) with the simulated server's lists, model files and figures (the
// mocks the page uses: services/mockEngine.ts, services/models.ts, services/logs.ts), and the page's words for the
// Concept notes (every section's heading and points, in the concept's order).
//
//   node tools/test-notes-logs.ts        exit 1 on a failure

import { DEFINITIONS } from '../src/definitions/index.ts';
import { ENGINE_IDS } from '../src/definitions/engines.ts';
import type { EngineId, Tier } from '../src/definitions/types.ts';
import { engineLogModel, indexOf, listKey, selectionKey, type Desk } from '../src/derive.ts';
import { say, type Slot } from '../src/derive/messages.ts';
import { mockLists } from '../src/app/services/mock/lists.ts';
import { MODEL_FILES, mockModels } from '../src/app/services/mock/models.ts';
import { WORDS } from '../src/app/i18n/words/notes.ts';

const J = (x: unknown) => JSON.stringify(x);
const fails: string[] = [];
let n = 0;
const ok = (what: string, cond: boolean, detail: unknown = '') => { n++; if (!cond) fails.push(`${what}${detail === '' ? '' : ': ' + (typeof detail === 'string' ? detail : J(detail))}`); };
const s = (m: Slot) => say(m);
const ix = indexOf(DEFINITIONS);

/* what the page is told: the server's lists, every engine's model files and the file it selected, the figures */
const told: Pick<Desk, 'lists' | 'selection' | 'vram'> = { lists: { ...await mockLists().lists() }, selection: {}, vram: await mockModels().vram() };
for (const e of ENGINE_IDS) for (const [b, files] of Object.entries(MODEL_FILES[e])) {
  told.lists![listKey({ from: 'models', bucket: b }, e)] = files; told.selection![selectionKey(e, b)] = files[0]!.value;
}
const desk = (engine: EngineId, tier: Tier = 'studio', vals: Desk['vals'] = {}): Desk =>
  ({ engine, tier, vals, auto: {}, state: {}, on: new Set(DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id)), ...told });
const UP = { up: true, engines: [...ENGINE_IDS] };
const log = (d: Desk, status: Parameters<typeof engineLogModel>[2] = UP, queue: Parameters<typeof engineLogModel>[3] = null) => {
  const m = engineLogModel(ix, d, status, queue); return { head: s(m.head), lines: m.lines.map(s) };
};

/* ---- the concept's log on ACE-Step's defaults (34-engine-panel.js paintLog) */
{
  const m = log(desk('ace'));
  ok('the head names the engine and its variant, sound and planner models, whole (F-34)', m.head === 'ACE-Step 1.5 · XL Turbo Q8_0, 4B Q8_0', m.head);
  ok('the five lines, as the concept writes them', J(m.lines) === J(['[engine]  ace-server up, 5 backends registered (HeartMuLa in integration testing)', '[ace]     sound XL Turbo Q8_0, planner 4B Q8_0, decoder ScragVAE',
    '[plugins] 46 solvers · 17 schedules · 18 guidance modes', '[vram]    13.3 / 31.8 GB loaded · the next Generate needs about 15.6 GB', '[queue]   idle']), m.lines);
}
/* ---- each engine's models by its roles, the tag padded to one column */
{
  const want: { [e in EngineId]: [string, string] } = {
    ace: ['ACE-Step 1.5 · XL Turbo Q8_0, 4B Q8_0', '[ace]     sound XL Turbo Q8_0, planner 4B Q8_0, decoder ScragVAE'],
    'minimax-m3': ['MiniMax-Music3 · Flow DiT Q8_0, 8B planner Q8_0', '[mm3]     sound Flow DiT Q8_0, planner 8B planner Q8_0, detail Depth decoder Q8_0'],
    yue2: ['YuE2 · YuE2 Q8_0', '[yue2]    sound YuE2 Q8_0, decoder Standard VAE'],
    sa3: ['Stable Audio 3 · SA3 DiT BF16', '[sa3]     sound SA3 DiT BF16'],
    heartmula: ['HeartMuLa · HeartMuLa 3B, Global Q8_0, Local Q8_0', '[hm]      variant HeartMuLa 3B, language Global Q8_0, decoder Local Q8_0, codec Codec flow Q8_0'],
  };
  for (const e of ENGINE_IDS) { const m = log(desk(e)); ok(`${e}: the head and the models line`, J([m.head, m.lines[1]]) === J(want[e]), [m.head, m.lines[1]]); }
}
/* ---- a model chosen is named as chosen, the VRAM following it */
{
  const m = log(desk('ace', 'studio', { 'model.dit': 'Turbo · Q4_K_M' }));
  ok('a model chosen: the head and its role name it whole', m.head === 'ACE-Step 1.5 · Turbo Q4_K_M, 4B Q8_0' && /^\[ace\]     sound Turbo Q4_K_M,/.test(m.lines[1]!), [m.head, m.lines[1]]);
  ok('and the VRAM is what it loads (5.3 GB of XL Turbo for 1.4 of Turbo Q4_K_M)', m.lines[3] === '[vram]    9.4 / 31.8 GB loaded · the next Generate needs about 11.7 GB', m.lines[3]);
}
/* ---- what the server says of itself */
{
  ok('before the server answers, the log says it waits', log(desk('ace'), null).lines[0] === '[engine]  waiting for ace-server');
  ok('a server that does not answer is said', log(desk('ace'), { up: false, engines: [] }).lines[0] === '[engine]  ace-server is not answering');
  ok('the engines registered are counted, and none in testing says nothing of testing', log(desk('ace'), { up: true, engines: ['ace', 'sa3'] }).lines[0] === '[engine]  ace-server up, 2 backends registered');
}
/* ---- the queue, as the generation package tells it (S-33's words) */
{
  ok('songs rendering and waiting', log(desk('ace'), UP, { rendering: 2, waiting: 3 }).lines[4] === '[queue]   2 rendering, 3 waiting');
  ok('on another engine, with what it holds loaded (B-34)', J(log(desk('ace'), UP, { rendering: 1, waiting: 0, engine: 'yue2', loaded: 5.7 }).lines.slice(3)) === J(['[vram]    5.7 / 31.8 GB loaded · the next Generate needs about 15.6 GB', '[queue]   1 rendering on YuE2']));
  ok('an empty queue is idle', log(desk('ace'), UP, { rendering: 0, waiting: 0 }).lines[4] === '[queue]   idle');
}
/* ---- the notes' words: the concept's sections, in order, each with its points */
{
  const count = (sec: string) => { let i = 1; while (`ui.notes.${sec}.${i}` in WORDS) i++; return i - 1; };
  ok('the notes hold the concept\'s five sections with their points', J(['tiers', 'adding', 'workflows', 'kept', 'grounding'].map(x => [WORDS[`ui.notes.${x}.title`], count(x)])) === J([['Tiers set how much each module shows', 8], ['Adding modules', 5], ['Workflows and revisions', 6], ['Your work is kept', 8], ['Grounding', 1]]));
  ok('every bold mark is closed, and no point holds a slot\'s braces', Object.entries(WORDS).filter(([k]) => k.startsWith('ui.notes.')).every(([, v]) => (v.match(/<b>/g) ?? []).length === (v.match(/<\/b>/g) ?? []).length && !/[{}]/.test(v)));
}

if (fails.length) { console.log(`notes and logs: ${fails.length} of ${n} failed`); fails.forEach(f => console.log('  ' + f)); process.exit(1); }
console.log(`notes and logs: ${n} checks of the engine log's core and the notes' words: ok`);
