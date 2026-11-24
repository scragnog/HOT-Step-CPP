// Generation (docs/09, package 6): what a press of Generate makes of the desk, and everything the go bar, the queue,
// Latest, the song menu and the player say, as the concept works them out (35-gobar.js, 35-songs.js, 35-player.js).
// The page decides none of it: it asks these functions and draws what they return. What renders, and how far it has
// got, is the server's (the generation service); what it says about a run in progress is worded here.
//
// Pure functions of the definitions, the desk and what the service reports. `rnd` is the caller's source of chance (a
// take's length and colour where the engine ends the song itself, a wildcard's pick), so a test can fix it.

import type { Definition, EngineId, Json } from '../definitions/types.ts';
import { engineName } from '../definitions/engines.ts';
import { type Desk, type Index, applies, changed, countOf, has, isAuto, mmss, onDesk, stateOf, valueOf } from './core.ts';
import { list, msg, type Msg, type Slot } from './messages.ts';
import { realState } from './request.ts';
import type { CreateIntent } from './request.ts';
import { defName, stepName } from './steps.ts';
import { tileAtTier, usedHere, parkedOn } from './desk.ts';
import { vramNeed } from './vram.ts';
import { switchEngine, type EngineBook } from './engine.ts';

const K = {
  prompt: 'prompt.text', lyrics: 'lyrics.text', instr: 'lyrics.instr', title: 'song.title', artist: 'song.artist', wild: 'wild.auto',
  takes: 'takes.n', steps: 'samp.steps', chain: 'pp.on', stream: 'mm.stream', dur: 'tempo.dur', max: 'tempo.max',
  preview: 'lm.preview', abc: 'yue.abc', replan: 'yue.replan', tries: 'yue.tries',
} as const;
const S = { source: 'source', region: 'region', latent: 'latent', cont: 'contFile', contSel: 'contSel', build: 'build', slots: 'slots' } as const;

const str = (v: unknown): string => (typeof v === 'string' ? v : '');
const val = (ix: Index, key: string, desk: Desk) => (ix.params.has(key) && has(ix, key, desk.engine) ? valueOf(ix, key, desk) : undefined);
const on = (ix: Index, key: string, desk: Desk) => ix.params.has(key) && applies(ix, key, desk) && val(ix, key, desk) === true;
const modOn = (ix: Index, id: string, desk: Desk) => desk.on.has(id) && !!ix.defs.find(d => d.id === id && usedHere(d, desk.engine));

/* ---------------------------------------------------------------- the main box's word, per engine (S-05) */

/** What the main text box is called on an engine (the concept's BOX, 01-engines.js). Package 4's derive/fields.ts
 *  states the same rule (boxWord, boxA); once both are on one branch, this one goes and those are called. */
const BOX: { [e: string]: string } = { 'minimax-m3': 'caption', sa3: 'prompt', heartmula: 'tags' };
export const genBox = (e: EngineId): Msg => msg(`gen.box.${BOX[e] ?? 'description'}`);
export const genBoxA = (e: EngineId): Msg => msg(`gen.box.${BOX[e] ?? 'description'}.a`);
/** The example a refusal offers (the box's first placeholder line, 33-prompt.js PH, PHT). */
export const exampleText = (e: EngineId): string => (BOX[e] === 'tags' ? 'indie folk, acoustic guitar, warm, female vocals' : 'dreamy indie folk, warm acoustic guitar, soft female vocals');

/* ---------------------------------------------------------------- the song a repaint works on */

/** The song a repaint works on, as its record holds it (package 3a's Repaint reads the same record). */
export interface SourceSong { song: number | string; title: string; style: string; dur: number; hue: number }
export function repaintSource(ix: Index, desk: Desk): SourceSong | null {
  const v = stateOf(ix, S.source, desk) as Partial<SourceSong> | null | undefined;
  return v && typeof v === 'object' && typeof v.dur === 'number' && typeof v.title === 'string' ? { song: v.song ?? '', title: v.title, style: str(v.style), dur: v.dur, hue: Number(v.hue) || 0 } : null;
}

/* ---------------------------------------------------------------- what a press of Generate makes */

/** A press that does not start a run, and why: a box that must be filled first (with its example), a score to preview
 *  first, two modules that cannot run together, or a run certain not to fit (F-21: the line says why, with its way
 *  out). */
export type Refusal =
  | { kind: 'empty'; says: Msg; example: string }
  | { kind: 'score' }
  | { kind: 'clash'; says: Msg }
  | { kind: 'wontFit'; says: Msg; fits: number };

export function refusalOf(ix: Index, desk: Desk): Refusal | null {
  const e = desk.engine, src = repaintSource(ix, desk), typed = str(val(ix, K.prompt, desk)).trim();
  if (!src && !typed) {
    const lyr = has(ix, K.lyrics, e) && str(val(ix, K.lyrics, desk)).trim(), needStyle = e !== 'ace';
    if (needStyle || (!lyr && val(ix, K.instr, desk) !== true)) return { kind: 'empty', says: needStyle ? msg('gen.needs', { engine: engineName(e), box: genBoxA(e) }) : msg('gen.needs.ace'), example: exampleText(e) };
  }
  if (on(ix, K.preview, desk) && !str(val(ix, K.abc, desk)).trim() && !src) return { kind: 'score' };
  const build = stateOf(ix, S.build, desk) as { planned?: unknown } | undefined;
  if (stateOf(ix, S.cont, desk) && modOn(ix, 'build', desk) && build?.planned) return { kind: 'clash', says: msg('gen.clash') };
  const n = takesOf(ix, desk), v = vramNeed(ix, desk, n);
  if (v.refuses) return { kind: 'wontFit', says: msg('gen.blocked', { need: v.need.toFixed(1), card: cardText(v.card), fits: lineFits(ix, desk) }), fits: lineFits(ix, desk) };
  return null;
}

/** What the main box holds now. */
export const boxHolds = (ix: Index, desk: Desk): string => str(val(ix, K.prompt, desk));

/** How many songs the next run makes: the takes on the desk (1 without them); a repaint makes one. */
export function takesOf(ix: Index, desk: Desk): number {
  if (repaintSource(ix, desk) && modOn(ix, 'repaint', desk)) return 1;
  return modOn(ix, 'takes', desk) ? Math.max(1, Number(val(ix, K.takes, desk) ?? 1)) : 1;
}
const cardText = (card: number) => String(+card.toFixed(1));

/** One song a run will make, as its card shows it while it renders and as the server is asked for it. */
export interface SongSpec {
  title: string; take?: number; of?: number;
  /** Its card's line: the artist, then the description (a Structured Caption's Global Metadata line, E-35). */
  style: string;
  /** The whole text it was made from, for Reuse (E-20). */
  text: string;
  /** Seconds. */
  dur: number;
  hue: number;
  seed?: number;
}
/** A run as it is started: its engine and everything it renders with, kept with it, so it keeps its engine whatever is
 *  selected later and goes back in the queue exactly as it was (F-13). */
export interface RunSpec {
  engine: EngineId;
  /** The desk it was started from: what it holds on the card (B-34) and what Undo of a cancel queues again. */
  desk: Desk;
  intent: CreateIntent;
  n: number; steps: number;
  /** The pedals that run after decoding, by name, for the stage line. */
  lit: Slot[];
  /** It can be played while it renders (MiniMax-Music3, E-40). */
  stream: boolean;
  /** It re-plans runaway scores, up to this many tries (YuE2): the dice light up as it tries. */
  tries: number;
  songs: SongSpec[];
  /** What it needs on the card, what each take adds and how many fit, for a refusal the server makes. */
  need: number;
  /** What the toast says as it starts (a latent or a recording it continues from). */
  says: Msg[];
}

/** A song's card line from what it was made from (E-35). */
export const songStyle = (p: string): string => ((/^Global Metadata:[ \t]*(.+)$/m.exec(p) ?? [])[1] ?? p).replace(/\n/g, ' ').trim();
/** One option of each {a|b|c} wildcard, chosen by `rnd`. Package 4's derive/fields.ts has the same (expandWild). */
const expand = (t: string, rnd: () => number): string => t.replace(/\{([^{}]*\|[^{}]*)\}/g, (_, g: string) => { const o = g.split('|'); return o[Math.floor(rnd() * o.length)]!.trim(); });
/** A song's title from its words: the first three words of the description, the first one capitalised. */
const titleFrom = (p: string): string => { const w = p.split(/[,.\n]/)[0]!.replace(/Global Metadata:\s*/, '').trim().split(/\s+/).slice(0, 3).join(' '); return w ? w[0]!.toUpperCase() + w.slice(1) : 'Untitled'; };

/** The request the page sends for the desk: the desk in the real application's names (request.ts realState). The real
 *  adapter (package 10) finishes it with the application's own resolver (request.ts createIntent). */
export function pageIntent(ix: Index, desk: Desk): CreateIntent {
  const rs = realState(ix, desk);
  return { kind: 'create', params: { backend: desk.engine, ...rs.params, ...(Object.keys(rs.backendParams).length ? { backendParams: rs.backendParams } : {}), ...rs.body, ...rs.content }, compose: (rs.intent.compose ?? {}) as { [k: string]: Json } };
}

/** What a press of Generate starts (35-gobar.js generate): one song per take, each titled, described and timed as the
 *  concept does it; `seed` is for this run only (a score being rendered, E-08). Refusals are refusalOf's. */
export function runOf(ix: Index, desk: Desk, rnd: () => number, seed?: number): RunSpec {
  const e = desk.engine, src = modOn(ix, 'repaint', desk) ? repaintSource(ix, desk) : null, typed = str(val(ix, K.prompt, desk)).trim();
  const p0 = typed || (src ? src.style : ''), p = on(ix, K.wild, desk) ? expand(p0, rnd) : p0;
  const n = takesOf(ix, desk), steps = Number(valueOf(ix, K.steps, desk) ?? 12);
  const lit: Slot[] = val(ix, K.chain, desk) === false ? [] : ix.defs.filter(d => d.kind === 'pedal' && desk.on.has(d.id) && usedHere(d, e) && d.on && valueOf(ix, d.on, desk) === true).sort((a, b) => a.at - b.at).map(d => defName(d, desk));
  const tempo = modOn(ix, 'tempo', desk), title = str(val(ix, K.title, desk)), artist = str(val(ix, K.artist, desk));
  const exact = tempo && applies(ix, K.dur, desk) && !isAuto(ix, K.dur, desk);
  const durMax = ix.params.get(K.dur)?.p.range?.[1] ?? 240;
  /* an exact length where the engine takes one; where the model ends the song itself, never past the longest it may run (S-07) */
  const cap = exact && e === 'heartmula' ? Number(val(ix, K.dur, desk)) : tempo && applies(ix, K.max, desk) ? Number(val(ix, K.max, desk)) : e === 'heartmula' ? Number(resolveMax(ix, desk) ?? durMax) : Infinity;
  const songs: SongSpec[] = Array.from({ length: n }, (_, i) => {
    const dur = src ? src.dur : (e === 'ace' || e === 'sa3') && exact ? Number(val(ix, K.dur, desk)) : e === 'sa3' ? 30 : Math.min(cap, 150 + Math.round(rnd() * 90));
    return {
      title: src ? `${src.title} (repainted)` : title || titleFrom(p), ...(n > 1 ? { take: i + 1, of: n } : {}),
      style: (artist ? artist + ' · ' : '') + songStyle(p), text: String(p).slice(0, 4000), dur,
      hue: src ? (src.hue + 25) % 360 : Math.round(rnd() * 360), ...(seed !== undefined ? { seed } : {}),
    };
  });
  const says: Msg[] = [];
  const latent = stateOf(ix, S.latent, desk), cont = stateOf(ix, S.cont, desk);
  if (e === 'ace' && typeof latent === 'string' && latent) says.push(msg('gen.continuing', { file: latent }));
  if (e === 'yue2' && modOn(ix, 'cont', desk) && typeof cont === 'string' && cont) {
    const sel = stateOf(ix, S.contSel, desk) as { start?: number; end?: number } | number[] | undefined;
    const [a, b] = Array.isArray(sel) ? sel : [sel?.start ?? 0, sel?.end ?? 0];
    says.push(msg('gen.continuing.part', { a: mmss(Number(a)), b: mmss(Number(b)), file: cont }));
  }
  return {
    engine: e, desk, intent: pageIntent(ix, desk), n, steps, lit, songs, says,
    stream: e === 'minimax-m3' && on(ix, K.stream, desk),
    tries: e === 'yue2' && on(ix, K.replan, desk) ? Math.max(1, Number(val(ix, K.tries, desk) ?? 1)) : 0,
    need: vramNeed(ix, desk, n).need,
  };
}
const resolveMax = (ix: Index, desk: Desk): number | undefined => { const hit = ix.params.get(K.dur); if (!hit) return undefined; const line = (hit.p.per ?? []).find(l => 'engines' in l && l.engines.includes(desk.engine) && 'range' in l) as { range?: [number, number, number] } | undefined; return (line?.range ?? hit.p.range)?.[1]; };

/** What a run says it is doing at `f` (0 to 1) of its own song (35-gobar.js stage): planning for the first fifth, then
 *  the steps, then decoding, then each pedal that runs. HeartMuLa writes as it goes (S-11). */
export function stageOf(f: number, e: EngineId, steps: number, lit: Slot[]): Msg {
  if (f < 0.2) return msg(`gen.stage.plan.${e}`);
  if (f < 0.8) return msg('gen.stage.step', { kind: e, i: Math.max(1, Math.round((f - 0.2) / 0.6 * steps)), n: steps });
  if (f < 0.88 || !lit.length) return msg('gen.stage.decode');
  return msg('gen.stage.pedal', { name: lit[Math.min(lit.length - 1, Math.floor((f - 0.88) / 0.12 * lit.length))]! });
}
/** How far along each take is when its run is at `f`: the takes start one after another. */
export const takeProgress = (f: number, i: number): number => Math.min(1, Math.max(0, f * 1.2 - i * 0.15));
/** How much of a take can be played while it renders: a window behind the render (E-40). */
export const readyOf = (ff: number): number => Math.min(1, Math.max(0, (ff - 0.25) / 0.75));
/** How many re-planning tries a YuE2 run has used at `f`, of the `att` it will take (the dice). */
export const attemptsAt = (f: number, att: number): number => (f < 0.2 ? Math.min(att, 1 + Math.floor(f / 0.2 * att)) : att);

/** A refusal the server makes when a run reaches the card (MiniMax-Music3 checks before it loads anything): the
 *  notice's words, and how many takes would fit on that card. */
export function refusedWords(e: EngineId, n: number, need: number, card: number, each: number, models: number): { why: Msg; more: Msg; retry: number } {
  const fits = Math.max(1, Math.floor((card - models) / (each || 1)) + 1);
  return { why: msg('gen.refused.why', { need: need.toFixed(1), n, card: cardText(card) }), more: msg('gen.refused.more', { engine: engineName(e) }), retry: Math.min(fits, n - 1) };
}

/* ---------------------------------------------------------------- the go bar's status line */

/** How many takes the line offers when the next run will not fit (35-gobar.js paintMeter). */
function lineFits(ix: Index, desk: Desk): number {
  const tk = takesOf(ix, desk), v = vramNeed(ix, desk, tk);
  return Math.min(tk - 1, Math.max(1, Math.floor((v.card - v.models) / (v.each || 1)) + 1));
}

/** The settings the line's "N changes" counts (S-04): every changed setting of the modules and pedals on the desk
 *  that the engine uses, a pedal's own foot switch left out, and of the built-in boxes the switches and choices (not
 *  their text); and each adapter loaded. */
export function changedKeys(ix: Index, desk: Desk): string[] {
  return ix.defs.filter(d => (onDesk(d, desk)) && usedHere(d, desk.engine) && !(d.kind === 'module' && d.source))
    .flatMap(d => d.params.filter(p => !(d.kind !== 'module' && d.on === p.key) && !(d.kind === 'module' && d.builtIn && p.type === 'text') && changed(ix, p.key, desk)).map(p => p.key));
}

export interface GoLine {
  /** The next run will not fit (refused before it loads): what it needs, the card, and the way out. */
  wontFit: { needs: Msg; use: Msg; fits: number } | null;
  /** It runs in passes (ACE-Step): the takes and the passes. */
  passes: { takes: Msg; long: Msg } | null;
  tier: Msg;
  /** What is on the desk, in order: modules, pedals, side units, hidden, parked. */
  rest: { says: Msg; hidden?: boolean }[];
  /** How many changes (0: everything on defaults), and the link's words. */
  changes: number; changesSay: Msg; rested: Msg;
}

/** The status line under the desk (35-gobar.js paintMeter): the next run judged where Generate is (G-B-08), the tier,
 *  what is on the desk, what this tier hides, what is parked (in every tier alike, S-01), and the changes. */
export function goLine(ix: Index, desk: Desk): GoLine {
  const e = desk.engine, tk = takesOf(ix, desk), v = vramNeed(ix, desk, tk), fits = lineFits(ix, desk);
  const live = (d: Definition) => d.kind === 'module' && !d.builtIn && !d.source && !d.board && desk.on.has(d.id) && usedHere(d, e);
  const mods = ix.defs.filter(live), shownN = mods.filter(d => tileAtTier(d, desk)).length, hid = mods.length - shownN;
  const pedals = ix.defs.filter(d => d.kind === 'pedal' && desk.on.has(d.id) && usedHere(d, e)).length;
  const units = ix.defs.filter(d => d.kind === 'unit' && desk.on.has(d.id) && usedHere(d, e)).length;
  const parked = parkedOn(ix, desk), np = parked.modules.length + parked.board.length;
  const ads = ix.defs.some(d => d.id === 'adapters' && usedHere(d, e)) ? countOf(ix, S.slots, desk) : 0;
  const ch = changedKeys(ix, desk).length + ads, tier = msg(`tier.${desk.tier}`);
  const rest: GoLine['rest'] = [{ says: shownN ? msg('gen.line.modules', { n: shownN }) : msg('gen.line.noModules') }];
  if (pedals) rest.push({ says: msg('gen.line.pedals', { n: pedals }) });
  if (units) rest.push({ says: msg('gen.line.units', { n: units }) });
  if (hid) rest.push({ says: msg('gen.line.hidden', { n: hid, tier }), hidden: true });
  if (np) rest.push({ says: msg('gen.line.parked', { n: np }) });
  return {
    wontFit: v.refuses ? { needs: msg('gen.line.needs', { need: v.need.toFixed(1), card: cardText(v.card) }), use: msg('gen.line.use', { n: fits }), fits } : null,
    passes: !v.refuses && v.passes > 1 ? { takes: msg('gen.line.takes', { n: tk }), long: msg('gen.line.passes', { n: v.passes }) } : null,
    tier, rest, changes: ch, changesSay: msg('gen.line.changes', { n: ch }), rested: ch ? msg('gen.line.rest') : msg('gen.line.defaults'),
  };
}

/** What "N changes" does when no changed control is drawn on the desk (35-gobar.js showChanges): changes only in
 *  modules this tier hides say so and offer Lab (Q-15); otherwise it names what is changed, never claiming a hidden
 *  module (S-04). */
export function changesUnmarked(ix: Index, desk: Desk): { says: Msg; lab: boolean } {
  const keys = changedKeys(ix, desk);
  const hid = ix.defs.filter(d => d.kind === 'module' && !d.builtIn && desk.on.has(d.id) && usedHere(d, desk.engine) && !tileAtTier(d, desk));
  const ks = hid.flatMap(d => d.params.filter(p => keys.includes(p.key)).map(p => p.key)), ms = hid.filter(d => d.params.some(p => keys.includes(p.key)));
  if (ms.length) return { says: msg('gen.changes.hidden', { n: ks.length, names: list(ms.map(d => defName(d, desk))), tier: msg(`tier.${desk.tier}`) }), lab: true };
  const names = [...new Map(keys.map(k => { const m = stepName(ix, k, desk); return [JSON.stringify(m), m]; })).values()];
  return { says: names.length ? msg('gen.changes.unmarked', { names: list(names) }) : msg('gen.changes.none'), lab: false };
}

/* ---------------------------------------------------------------- Generate and the queue */

export interface GoModel {
  /** Its label: Generate, or Repaint region while a repaint is on the desk. */
  label: Msg; title: Msg;
  /** The queue: how many songs render or wait (the badge), and Cancel's words. */
  queued: number; cancel: Msg; cancelLabel: Msg; count: string;
  wontFit: boolean;
}
/** Generate and Cancel (35-gobar.js updateGo): the button keeps its width and label; the queue is a badge on it (B-11,
 *  M-12); its title says how many render now and how many wait (C-26). */
export function goModel(ix: Index, desk: Desk, running: number, waiting: number): GoModel {
  const label = repaintSource(ix, desk) && modOn(ix, 'repaint', desk) ? msg('gen.go.repaint') : msg('gen.go'), q = running + waiting;
  return {
    label, title: q ? msg('gen.go.busy', { label, run: running, wait: waiting, q }) : msg('gen.go.title', { label }),
    queued: q, cancel: msg('gen.cancel', { q }), cancelLabel: msg('gen.cancel.label', { q }), count: q > 1 ? String(q) : '',
    wontFit: vramNeed(ix, desk, takesOf(ix, desk)).refuses,
  };
}
/** The document's title while songs render: their count before the page's own title. */
export const docTitle = (base: string, q: number): Msg => (q ? msg('gen.doc', { q, base }) : msg('gen.doc.idle', { base }));

/** A song's full name, for labels: its title, and its take when it is one of several. */
export const songName = (s: { title: string; take?: number; of?: number }): Msg => ((s.of ?? 0) > 1 ? msg('gen.song.take', { title: s.title, t: s.take ?? 1, of: s.of! }) : msg('gen.song', { title: s.title }));

/** What a card in the queue says (35-gobar.js tick): queued, in line behind the runs ahead, or its stage. */
export const queueStage = (ahead: number, f: number, stage: Slot): Slot => (ahead > 0 ? msg('gen.inLine', { n: ahead }) : f === 0 ? msg('gen.queued') : stage);

/** Cancelling (F-13): what the toast says, the step's name, and what Undo says when it puts the runs back. */
export function cancelWords(n: number, one: { title: string; take?: number; of?: number } | null): { says: Msg; label: Msg; put: Msg } {
  const put = msg('gen.cancel.put', { n });
  return one ? { says: msg('gen.cancelled.one', { name: songName(one) }), label: msg('gen.cancel.one', { name: songName(one) }), put }
    : { says: msg('gen.cancelled', { n }), label: msg('gen.cancel.step', { n }), put };
}
/** What Undo of a cancel says once the runs are back in the queue: the one that was rendering starts again. */
export const cancelBack = (n: number, restarts: boolean): Msg => msg('gen.cancel.back', { n, restarts: String(restarts) });
/** A finished run, said when its card is out of sight (the card is the message). */
export const newSongs = (n: number, e: EngineId): Msg => msg('gen.new', { n, engine: engineName(e) });
/** A run the server refused, said when its notice is out of sight. */
export const refusedSays = (n: number, e: EngineId): Msg => msg('gen.refused', { n, engine: engineName(e) });
/** Songs that were rendering when the page closed and that the server no longer has (F-27). */
export const lostSays = (n: number): Msg => msg('gen.lost', { n });

/* ---------------------------------------------------------------- Latest */

/** The concept's seeded generator (00-util.js rng): a song's art and waveform are the same on every visit. */
export const rng = (seed: number) => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const idNum = (id: string | number): number => (typeof id === 'number' ? id : /^\d+$/.test(id) ? +id : [...id].reduce((a, c) => (a * 31 + c.charCodeAt(0)) | 0, 7));
/** A song's art (35-songs.js artBg): two soft lights over a conic ground, in its hue. */
export function songArt(id: string | number, hue: number): string {
  const r = rng(idNum(id) * 977 + hue), h2 = (hue + 40 + r() * 80) % 360;
  return `radial-gradient(circle at ${20 + r() * 60}% ${20 + r() * 60}%, hsl(${hue} 92% 64%) 0, transparent ${40 + r() * 20}%),radial-gradient(circle at ${r() * 100}% ${r() * 100}%, hsl(${h2} 85% 55%) 0, transparent ${45 + r() * 20}%),conic-gradient(from ${r() * 360}deg at ${30 + r() * 40}% ${30 + r() * 40}%, hsl(${hue} 50% 22%), hsl(${h2} 60% 14%), hsl(${hue} 50% 22%))`;
}
/** The player's seek bar: the song's waveform (35-player.js playerWave, B-13). */
export function songWave(id: string | number): string {
  const r = rng(idNum(id) * 7919 + 3), n = 96; let d = '';
  for (let i = 0; i < n; i++) { const a = 2 + (0.25 + 0.75 * r()) * (Math.sin(i / n * Math.PI) * 0.5 + 0.5) * 8; d += `M${i * 3 + 1} ${10 - a}v${a * 2}`; }
  return d;
}

/** Which cards Latest shows (35-songs.js trimLatest): every song rendering or refused, so each can be cancelled or
 *  retried; finished ones in whole rows, up to two; a long queue folded to one row behind a "+N waiting" card that
 *  opens (C-26, C-47, D-26, B-10, F-14). `kinds` are the cards in order; `order` ranks a pending card (the songs
 *  rendering now first). */
export interface LatestShown { hidden: boolean[]; folded: boolean[]; fold: { after: number; open: boolean; title: Msg; sub: Msg } | null; count: Msg }
export function latestShown(kinds: ('song' | 'pending' | 'error')[], order: number[], cols: number, open: boolean): LatestShown {
  const n = kinds.length, pend = kinds.map((k, i) => (k === 'pending' ? i : -1)).filter(i => i >= 0), folded = Array<boolean>(n).fill(false);
  let fold: LatestShown['fold'] = null;
  if (pend.length > cols) {
    const keep = open ? pend.length : cols - 1, shown = new Set([...pend].sort((a, b) => order[a]! - order[b]!).slice(0, keep));
    for (const i of pend) folded[i] = !shown.has(i);
    fold = { after: pend[pend.length - 1]!, open, title: open ? msg('gen.fold.close') : msg('gen.fold', { n: pend.length - keep }), sub: open ? msg('gen.fold.close.sub') : msg('gen.fold.sub') };
  }
  const foldN = fold ? 1 : 0, cards = kinds.filter((k, i) => k !== 'error' && !folded[i]), busyN = cards.filter(k => k === 'pending').length + foldN, cn = cards.length + foldN;
  const rows = cn >= cols * 2 ? 2 : cn >= cols ? 1 : 0;
  let room = Math.max(rows ? rows * cols : cn, busyN) - foldN;
  const hidden = kinds.map((k, i) => { if (folded[i]) return true; const busy = k !== 'song'; const h = !busy && room <= 0; if (!h && k !== 'error') room--; return h; });
  const done = kinds.filter(k => k === 'song').length, shownDone = kinds.filter((k, i) => k === 'song' && !hidden[i]).length;
  /* the line says what this page holds, not what the Library holds (F-27) */
  return { hidden, folded, fold, count: done <= shownDone ? msg('gen.count', { n: done }) : msg('gen.count.of', { n: shownDone, kept: done > 60 ? 'newest 60' : String(Math.min(done, 60)) }) };
}

/** A song card's words (35-songs.js songHTML). */
export function songCard(s: { title: string; take?: number; of?: number; style: string; dur: number; engine: EngineId }): { name: Msg; take: Msg | null; takeTip: Msg; dur: string; engine: string; styleTip: Msg } {
  return { name: songName(s), take: (s.of ?? 0) > 1 ? msg('gen.take', { t: s.take ?? 1, of: s.of! }) : null, takeTip: msg('gen.take.tip', { t: s.take ?? 1, of: s.of ?? 1 }), dur: mmss(s.dur), engine: engineName(s.engine), styleTip: msg('gen.style.tip', { style: s.style, engine: engineName(s.engine) }) };
}

/* ---------------------------------------------------------------- the song menu */

export type SongAction = 'repaint' | 'reuse' | 'Cover Studio' | 'Song Builder' | 'Compare' | 'Stem Separator' | 'MIDI Studio';
/** The song menu's items, in order, with the break before the last three (35-songs.js). Repaint and Song Builder are
 *  ACE-Step's: from another engine they say they switch there (E-14). */
export function songMenu(engine: EngineId, songEngine: EngineId): { id: SongAction; says: Msg; small?: Msg; rule?: boolean }[] {
  const ace = engine === 'ace';
  return [
    { id: 'repaint', says: msg('gen.menu.repaint'), small: msg(ace ? 'gen.menu.repaint.here' : 'gen.menu.repaint.switch') },
    { id: 'reuse', says: msg('gen.menu.reuse', { box: genBox(songEngine) }), small: msg('gen.menu.reuse.small') },
    { id: 'Cover Studio', says: msg('gen.menu.cover'), small: msg('gen.menu.cover.small') },
    { id: 'Song Builder', says: msg('gen.menu.builder'), small: msg(ace ? 'gen.menu.builder.here' : 'gen.menu.builder.switch') },
    { id: 'Compare', says: msg('gen.menu.compare'), rule: true },
    { id: 'Stem Separator', says: msg('gen.menu.stems') },
    { id: 'MIDI Studio', says: msg('gen.menu.midi') },
  ];
}
/** What an item that opens another page says. */
export const sentTo = (page: string, title: string): Msg => msg('gen.sent', { page, title });

/** "Repaint a region": the song as the repaint's record holds it, the region it starts with (the concept's
 *  0.35 to 0.55 of the song), and the step's and the toast's words. From another engine it switches to ACE-Step first,
 *  as one step (E-14). */
export function repaintFrom(s: { id: string | number; title: string; take?: number; of?: number; style: string; dur: number; hue: number }, desk: Desk): { source: SourceSong; region: { start: number; end: number }; label: Msg; says: Msg; engine: EngineId } {
  const sw = desk.engine !== 'ace', name = songName(s);
  return {
    source: { song: typeof s.id === 'string' && /^\d+$/.test(s.id) ? +s.id : s.id, title: s.title, style: s.style, dur: s.dur, hue: s.hue }, region: { start: +(s.dur * 0.35).toFixed(2), end: +(s.dur * 0.55).toFixed(2) }, engine: 'ace',
    label: msg(sw ? 'gen.repaint.switch' : 'gen.repaint', { name }), says: msg('gen.repaint.says', { name }),
  };
}
/** The desk after "Repaint a region": on ACE-Step (switched there when it was elsewhere, as one step with the switch's
 *  own words, E-14), the song as the repaint's record, its first region, and Repaint first on the desk in place of any
 *  other source module. */
export function repaintDesk(ix: Index, desk: Desk, book: EngineBook, mods: string[], s: Parameters<typeof repaintFrom>[0]): { desk: Desk; book: EngineBook; mods: string[]; label: Msg; says: Msg } {
  const r = repaintFrom(s, desk), sw = desk.engine !== r.engine ? switchEngine(ix, desk, book, r.engine) : null;
  const d1 = sw ? sw.desk : desk, kept = mods.filter(id => (!sw || sw.desk.on.has(id)) && !ix.defs.some(d => d.id === id && d.kind === 'module' && d.source));
  return {
    desk: { ...d1, state: { ...d1.state, [S.source]: r.source as unknown as Json, [S.region]: r.region } }, book: sw ? sw.book : book, mods: ['repaint', ...kept],
    label: r.label, says: sw ? msg('gen.repaint.after', { news: sw.toast, says: r.says }) : r.says,
  };
}
/** "Reuse its description": the step's words, and the toast when it replaced what was in the box. */
export const reuseWords = (e: EngineId): { label: Msg; replaced: Msg } => ({ label: msg('gen.reuse', { box: genBox(e) }), replaced: msg('gen.reuse.replaced', { box: genBox(e) }) });

/* ---------------------------------------------------------------- the player */

/** The player's read-outs (35-player.js paintPlayer): the time, the seek bar's value words, the transport's words. */
export function playerWords(pos: number, dur: number, paused: boolean): { time: string; valueText: Msg; act: Msg; tip: Msg } {
  const act = msg(paused ? 'gen.player.play' : 'gen.player.pause');
  return { time: `${mmss(pos)} / ${mmss(dur)}`, valueText: msg('gen.player.at', { pos: mmss(pos), dur: mmss(dur) }), act, tip: msg('gen.player.key', { act }) };
}
/** Where a seek key takes the player (arrows 5 s, Shift or Page 30 s, Home, End), or null for another key. */
export function playerSeek(key: string, shift: boolean, pos: number, dur: number): number | null {
  const big = shift || key.startsWith('Page') ? 30 : 5;
  const v = ({ ArrowRight: pos + big, ArrowUp: pos + big, PageUp: pos + big, ArrowLeft: pos - big, ArrowDown: pos - big, PageDown: pos - big, Home: 0, End: dur - 1 } as Record<string, number>)[key];
  return v === undefined ? null : Math.min(Math.max(0, dur - 1), Math.max(0, Math.round(v)));
}
