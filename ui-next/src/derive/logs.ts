// The engine log (docs/09, package 11; the concept's 34-engine-panel.js paintLog): what the log docked at the window's
// foot says about the engine. Its head names the engine and the models it renders with; its five lines say what the
// server reports of itself (up, how many engines it has registered, which are in integration testing), the engine's
// models by their roles, the plugins the server lists, what the engine holds in VRAM and what the next Generate needs
// (vram.ts vramNeed, as the meter), and the queue. Every model is named whole, with no separator inside its name, and
// one separator between models (F-34). None has a branch for any engine or module: what differs per engine is data.

import { ENGINES } from '../definitions/engines.ts';
import type { EngineId } from '../definitions/types.ts';
import type { Role } from '../definitions/roles.ts';
import { type Desk, type Index, applies, formatValue, resolve, roleKey, shownValue, valueOf } from './core.ts';
import { EN, list, msg, type Msg, type Slot } from './messages.ts';
import { paramWords } from './strings.ts';
import { modelKeys, vramNeed } from './vram.ts';

/** What the server says about itself (GET /api/backends and its health): whether it answers, and the engines it has
 *  registered. Null until it has answered. */
export interface EngineStatus { up: boolean; engines: EngineId[] }
/** What the queue holds: how many songs render now and how many wait, and the engine that renders them when it is not
 *  the desk's (B-34), with what that engine holds loaded. The generation package tells it; null is an empty queue. */
export interface QueueSummary { rendering: number; waiting: number; engine?: EngineId; loaded?: number }

/** The core's words for the log, added to its table (messages.ts EN) as this file loads. */
const WORDS: { [id: string]: string } = {
  'log.head': '{engine} · {models}',
  'log.engine': '[engine]  ace-server up, {n} backends registered{testing}',
  'log.engine.testing': ' ({names} in integration testing)',
  'log.engine.down': '[engine]  ace-server is not answering',
  'log.engine.wait': '[engine]  waiting for ace-server',
  'log.models': '[{engine}]{pad}{models}',
  'log.model': '{role|lower} {file}',
  'log.plugins': '[plugins] {solvers} solvers · {schedules} schedules · {guidance} guidance modes',
  'log.vram': '[vram]    {loaded} / {card} GB loaded · the next Generate needs about {need} GB',
  'log.queue.idle': '[queue]   idle',
  'log.queue': '[queue]   {now} rendering{waiting}{on}',
  'log.queue.waiting': ', {n} waiting',
  'log.queue.on': ' on {engine}',
};
for (const [id, text] of Object.entries(WORDS)) {
  if (id in EN) throw new Error(`the core's word ${id} is defined twice`);
  EN[id] = text;
}

/** The models the head names, in this order (the variant, the sound model, the planner), and every model the engine
 *  loads, in the definitions' order, for the models line. */
const HEAD: Role[] = ['variant', 'soundModel', 'plannerModel'];
/** The plugin lists the server gives (parts.ts OPTION_SOURCES), counted on the plugins line. */
const PLUGINS = { solvers: 'plugins.solver', schedules: 'plugins.scheduler', guidance: 'plugins.guidance' } as const;

/** A model file as the log names it: the file in force ("" is the one the server selected), whole (F-34). */
function fileName(ix: Index, key: string, desk: Desk): string {
  const f = resolve(ix.params.get(key)!.p, desk.engine);
  return formatValue(f, shownValue(ix, key, desk, valueOf(ix, key, desk)!), desk.lists, desk.engine).replace(/ · /g, ' ');
}
const engineOf = (id: EngineId) => ENGINES.find(e => e.id === id)!;
const gb1 = (x: number) => x.toFixed(1);

export interface EngineLogModel {
  /** The head's line after "Engine": the engine and the models it renders with. */
  head: Msg;
  /** The log's lines, in order. */
  lines: Msg[];
}

/** The engine log for the desk, given what the server says about itself and what the queue holds. */
export function engineLogModel(ix: Index, desk: Desk, status: EngineStatus | null, queue: QueueSummary | null = null): EngineLogModel {
  const e = desk.engine, me = engineOf(e), used = modelKeys(ix, e).filter(k => applies(ix, k, desk));
  const head = msg('log.head', { engine: me.name, models: list(HEAD.map(r => roleKey(ix, r, desk.engine)).filter((k): k is string => !!k && used.includes(k)).map(k => fileName(ix, k, desk)), 'comma') });
  const testing = status ? status.engines.filter(id => 'testing' in engineOf(id) && (engineOf(id) as { testing?: boolean }).testing).map(id => engineOf(id).name) : [];
  const server = !status ? msg('log.engine.wait') : !status.up ? msg('log.engine.down')
    : msg('log.engine', { n: status.engines.length, testing: testing.length ? msg('log.engine.testing', { names: list(testing) }) : '' });
  const roles: Slot[] = used.map(k => { const w = paramWords(ix, k, desk); return msg('log.model', { role: w.short ?? w.label, file: fileName(ix, k, desk) }); });
  const tag = me.short, models = msg('log.models', { engine: tag, pad: ' '.repeat(Math.max(1, 10 - tag.length - 2)), models: list(roles, 'comma') });
  const count = (src: string) => desk.lists?.[src]?.length ?? 0;
  const plugins = msg('log.plugins', { solvers: count(PLUGINS.solvers), schedules: count(PLUGINS.schedules), guidance: count(PLUGINS.guidance) });
  const v = vramNeed(ix, desk), loaded = queue?.engine && queue.engine !== e && queue.loaded !== undefined ? queue.loaded : v.models;
  const vram = msg('log.vram', { loaded: gb1(loaded), card: String(+v.card.toFixed(1)), need: gb1(v.need) });
  const busy = !!queue && queue.rendering + queue.waiting > 0;
  const q = !busy ? msg('log.queue.idle') : msg('log.queue', {
    now: queue!.rendering, waiting: queue!.waiting ? msg('log.queue.waiting', { n: queue!.waiting }) : '',
    on: queue!.engine && queue!.engine !== e ? msg('log.queue.on', { engine: engineOf(queue!.engine).name }) : '',
  });
  return { head, lines: [server, models, plugins, vram, q] };
}
