// The engine panel and the VRAM meter (docs/09, package 5): what each engine's card says, what the panel's models
// heading says at the tier, each model row's words and option faces, and what the next Generate needs on the card,
// worked out from the definitions, the model files the server lists with their sizes (desk.lists) and the figures it
// tells (desk.vram), as the concept works them out (34-engine-panel.js vramParts, paintEngine, stepMeter; 35-gobar.js
// vramNeed). None has a branch for any module or engine: what differs per engine is data (engines.ts, the figures).

import type { EngineId, Node, Tier } from '../definitions/types.ts';
import { ENGINES, engineName } from '../definitions/engines.ts';
import { type Desk, type Index, applies, countOf, has, listKey, modelDefault, resolve, valueOf } from './core.ts';
import { type Msg, type Slot, msg } from './messages.ts';
import { paramWords, textMsg } from './strings.ts';
import { CARD_GB, optionFaces, type OptionFace } from './controls.ts';

/** How many songs one Generate makes at most on an engine: the takes count's range there (the definitions'). */
const TAKES = 'takes.n';

/** A model the engine loads (OptionsFrom `loaded`): its options are a bucket of the server's model files. */
const bucketOf = (ix: Index, key: string, e: EngineId) => {
  const f = resolve(ix.params.get(key)!.p, e), o = f.options;
  return o && !Array.isArray(o) && o.from === 'models' && o.loaded && o.bucket ? o : null;
};
/** Every model the engine loads, in the definitions' order. */
export const modelKeys = (ix: Index, e: EngineId): string[] => [...ix.params.keys()].filter(k => has(ix, k, e) && !!bucketOf(ix, k, e));

/** The size in GB of the file a model choice holds on the desk (its "" is the file the server has selected); 0 when the
 *  server does not list it with a size. */
export function modelGb(ix: Index, key: string, desk: Desk): number {
  const b = bucketOf(ix, key, desk.engine); if (!b) return 0;
  const v = valueOf(ix, key, desk), file = v === '' ? modelDefault(ix, key, desk) ?? v : v;
  return desk.lists?.[listKey(b, desk.engine)]?.find(x => x.value === file)?.gb ?? 0;
}

export interface VramNeed {
  /** The card, in GB. */
  card: number;
  /** What the engine holds loaded: its model files, the parts it always loads and its adapters. */
  models: number;
  /** What the next Generate needs: that, and what its takes add. */
  need: number;
  /** What each take adds, and whether the first take adds it too (else only the takes after the first). */
  each: number; first: boolean;
  /** How many takes fit on the card at once (what "Use N takes" and "Try N takes" offer). */
  fits: number;
  /** In how many passes it runs to fit the card (1: at once). */
  passes: number;
  /** The engine refuses it before loading anything. */
  refuses: boolean;
  /** It does not fit at once, and is not refused. */
  tight: boolean;
  /** What it adds up from: each model by its role, each fixed part, the adapters. */
  parts: { name: Slot; gb: number }[];
}

/** What the desk's next Generate needs on the card (35-gobar.js vramNeed), for `takes` songs (default: the takes count
 *  in force). Without the figures (desk.vram) only the model files count. */
export function vramNeed(ix: Index, desk: Desk, takes?: number): VramNeed {
  const e = desk.engine, card = desk.vram?.card ?? CARD_GB, fig = desk.vram?.engines[e];
  const parts: VramNeed['parts'] = modelKeys(ix, e).filter(k => applies(ix, k, desk)).map(k => ({ name: paramWords(ix, k, desk).short ?? paramWords(ix, k, desk).label, gb: modelGb(ix, k, desk) })).filter(x => x.gb > 0);
  for (const f of fig?.fixed ?? []) parts.push({ name: f.name, gb: f.gb });
  const ads = fig?.adapters ? Math.min(countOf(ix, 'slots', desk), fig.adapters.most) : 0;
  if (ads) parts.push({ name: msg('vram.adapters'), gb: ads * fig!.adapters!.gb });
  const n = takes ?? (has(ix, TAKES, e) ? Number(valueOf(ix, TAKES, desk) ?? 1) : 1);
  const t = fig?.take, each = !t || (t.when && !valueOf(ix, t.when, desk)) ? 0 : t.gb;
  const models = parts.reduce((a, x) => a + x.gb, 0), extra = t?.first ? n * each : Math.max(0, n - 1) * each, need = models + extra;
  const passes = fig?.over === 'passes' && need > card && each ? Math.ceil(extra / Math.max(each, card - models)) : 1;
  const refuses = fig?.over === 'refuse' && need > card;
  const fits = Math.max(1, Math.floor((card - models) / (each || 1)) + 1);
  return { card, models, need, each, first: !!t?.first, fits, passes, refuses, tight: !refuses && need > card, parts };
}

/** A number of GB as the meter writes it ("13.3"); the card as the page names it ("31.8"). */
const gb1 = (x: number) => x.toFixed(1);
const cardText = (card: number) => String(+card.toFixed(1));

export interface MeterModel {
  /** What is loaded and what the next Generate needs, in GB: where the solid and the outlined segments end. */
  use: number; est: number; card: number;
  /** The read-out: what is loaded, or what the next run needs when it will not fit at once; and "/ 31.8 ". */
  read: string; of: Msg;
  /** The next run is refused (over), or runs in passes or tight (tight). */
  over: boolean; tight: boolean;
  /** Where its peak line stands, and the ticks at 8, 16 and 24 GB, as fractions of the bar. */
  peak: number; ticks: number[];
  /** Its accessible name and its tooltip. */
  aria: Msg; title: Msg;
}

/** The VRAM meter in the top bar (34-engine-panel.js paintEngine): when the next run will be refused, the read-out shows
 *  what it needs, not what is loaded (G-B-08). */
export function meterModel(ix: Index, desk: Desk, run?: Desk | null): MeterModel {
  const v = vramNeed(ix, desk), card = cardText(v.card);
  /* while another engine renders, "loaded" is what that engine holds; the next Generate is the selected engine's (B-34) */
  const other = run && run.engine !== desk.engine ? run : null, held = other ? vramNeed(ix, other).models : v.models;
  return {
    use: held, est: v.need, card: v.card,
    read: gb1(v.refuses || v.tight ? v.need : held), of: msg('vram.of', { card }),
    over: v.refuses, tight: v.tight,
    peak: Math.min(1, v.need / v.card), ticks: [8, 16, 24].map(g => g / v.card),
    aria: other ? msg('vram.aria.run', { engine: engineName(other.engine), held: gb1(held), next: engineName(desk.engine), need: gb1(v.need), card })
      : msg('vram.aria', { loaded: gb1(v.models), need: gb1(v.need), card }), title: msg('vram.title', { card }),
  };
}
/** Where the meter's needle heads: what is loaded, and while a run renders, most of the way to what it needs, a little
 *  differently each frame (34-engine-panel.js stepMeter). */
export const meterTarget = (use: number, est: number, busy: boolean, rnd: () => number): number => use + (busy ? Math.max(0, est - use) * (0.75 + rnd() * 0.25) : 0);

/** The meter's LED segments: lit up to what it shows now (`cur`), outlined up to what the next run needs, amber from
 *  68% of the bar and red from 86% (34-engine-panel.js stepMeter). */
export function meterLeds(cur: number, est: number, card: number, n = 32): string[] {
  const lit = Math.round(cur / card * n), ghost = Math.round(est / card * n);
  return Array.from({ length: n }, (_, i) => [i < lit ? 'on' : i < ghost ? 'nx' : '', i >= n * 0.86 ? 'r' : i >= n * 0.68 ? 'a' : ''].filter(Boolean).join(' '));
}
/** One frame of the meter's needle: a fast attack and a slow release toward what it shows; at once without motion. */
export const meterEase = (cur: number, target: number, reduced: boolean): number => cur + (target - cur) * (reduced ? 1 : target > cur ? 0.35 : 0.08);

export interface EngineCard {
  id: EngineId; name: string;
  /** Its tagline, in two lines, so the cards' first lines and spec rows line up (F-20). */
  tag: [Msg, Msg];
  /** Its output rate, short and as a tip; how many songs a Generate makes, and its tip; its badge while in testing. */
  rate: Msg; rateTip: Msg; takes: Msg; takesTip: Msg; testing: Msg | null;
}
export interface EnginePanelModel {
  cards: EngineCard[];
  /** The models' heading ("ACE-Step 1.5 models"), what the tier shows of them, and that the engine is in testing. */
  head: Msg; sub: Msg; testing: Msg | null; sep: Msg;
}

/** The engine panel: a card for each engine and the heading of the desk's engine's models (34-engine-panel.js
 *  fillEngine). */
export function enginePanelModel(ix: Index, tier: Tier, engine: EngineId): EnginePanelModel {
  const me = ENGINES.find(x => x.id === engine)!;
  return {
    cards: ENGINES.map(e => {
      const most = has(ix, TAKES, e.id) ? resolve(ix.params.get(TAKES)!.p, e.id).range?.[1] ?? 1 : 1;
      return {
        id: e.id, name: e.name,
        tag: [{ id: `engine.${e.id}.tag1`, en: e.tagline[0] }, { id: `engine.${e.id}.tag2`, en: e.tagline[1] }],
        rate: msg('engine.rate.short', { khz: e.khz }), rateTip: msg('engine.rate', { khz: e.khz }),
        takes: msg('engine.takes', { n: most }), takesTip: msg('engine.takes.tip', { n: most }),
        testing: 'testing' in e && e.testing ? msg('engine.testing') : null,
      };
    }),
    head: msg('models.head', { engine: me.name }), sub: msg(tier === 'quick' ? 'models.sub.quick' : 'models.sub'),
    testing: 'testing' in me && me.testing ? msg('models.testing') : null, sep: msg('models.sep'),
  };
}

/** One model row of the engine panel: the model's role (Sound, Planner), its list's heading and the card it is drawn
 *  against, its accessible name with the technical name after the role (C-28), and every option's face. */
export interface ModelRow { key: string; role: Slot; title: Msg; aria: Msg; note: Msg; faces: OptionFace[] }

/** The rows a modelPicker node draws on the desk: each model choice it binds that applies on the engine. */
export function modelRows(ix: Index, node: Node, desk: Desk): ModelRow[] {
  const card = cardText(desk.vram?.card ?? CARD_GB);
  return [node.param, ...(node.params ?? [])].filter((k): k is string => !!k && has(ix, k, desk.engine) && applies(ix, k, desk)).map(k => {
    const w = paramWords(ix, k, desk), role = w.short ?? w.label;
    return { key: k, role, title: msg('models.title', { role }), aria: msg('models.aria', { role, label: w.label, value: textMsg(ix, k, desk) }), note: msg('models.vram', { card }), faces: optionFaces(ix, k, desk, 'model') };
  });
}
