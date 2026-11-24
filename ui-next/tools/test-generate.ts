// Tests of the core's additions for generation (package 6: src/derive/generate.ts, meterModel and meterTarget in
// src/derive/vram.ts, Desk.songs in src/derive/readouts.ts), each against what the concept does (35-gobar.js generate,
// runBatch, paintMeter, updateGo, cancelUndoable; 35-songs.js songHTML, trimLatest and the song menu; 35-player.js;
// 34-engine-panel.js paintEngine).
//
//   node tools/test-generate.ts        exit 1 on a failure

import { DEFINITIONS } from '../src/definitions/index.ts';
import type { EngineId, Node, Tier } from '../src/definitions/types.ts';
import {
  attemptsAt, cancelBack, cancelWords, changedKeys, changesUnmarked, docTitle, emptyBook, goLine, goModel, indexOf, latestShown, meterModel, meterTarget, playerSeek, playerWords,
  readoutModel, readyOf, refusalOf, refusedWords, repaintDesk, runOf, songCard, songMenu, stageOf, takeProgress, type Desk, type Msg, type VramFigures,
} from '../src/derive.ts';
import { draftRun } from '../src/app/services/mock/songs.ts';
import { say } from '../src/derive/messages.ts';

const ix = indexOf(DEFINITIONS);
const J = (x: unknown) => JSON.stringify(x);
const fails: string[] = [];
let n = 0;
const ok = (what: string, cond: boolean, detail: unknown = '') => { n++; if (!cond) fails.push(`${what}${detail === '' ? '' : ': ' + (typeof detail === 'string' ? detail : J(detail))}`); };
/* the concept's figures (services/models.ts), and a model list with sizes, so the VRAM sums are the concept's */
const FIG: VramFigures = {
  card: 31.8,
  engines: {
    ace: { fixed: [{ name: 'Text encoder', gb: 1.2 }, { name: 'Working memory', gb: 2.0 }], adapters: { gb: 0.15, most: 3 }, take: { gb: 2.3, when: 'planner', first: true }, over: 'passes' },
    'minimax-m3': { fixed: [{ name: 'Encoders and vocoder', gb: 0.8 }, { name: 'Working memory', gb: 3.0 }], adapters: { gb: 0.15, most: 1 }, take: { gb: 3.4, first: false }, over: 'refuse' },
    sa3: { fixed: [{ name: 'T5Gemma and SAME', gb: 1.6 }, { name: 'Working memory', gb: 1.2 }], take: { gb: 0, first: false }, over: 'tight' },
  },
};
const LISTS: Desk['lists'] = {
  'models.minimax-m3.dit': [{ value: 'Flow DiT · F16', label: 'Flow DiT · F16', gb: 4.9 }], 'models.minimax-m3.lm': [{ value: '8B planner · F16', label: '8B planner · F16', gb: 17.2 }], 'models.minimax-m3.depth': [{ value: 'Depth decoder · F16', label: 'Depth decoder · F16', gb: 1.3 }],
  'models.ace.dit': [{ value: 'XL Turbo · Q8_0', label: 'XL Turbo · Q8_0', gb: 5.3 }], 'models.ace.lm': [{ value: '4B · Q8_0', label: '4B · Q8_0', gb: 4.5 }], 'models.ace.vae': [{ value: 'ScragVAE', label: 'ScragVAE', gb: 0.34 }],
};
const SEL = { 'minimax-m3.dit': 'Flow DiT · F16', 'minimax-m3.lm': '8B planner · F16', 'minimax-m3.depth': 'Depth decoder · F16', 'ace.dit': 'XL Turbo · Q8_0', 'ace.lm': '4B · Q8_0', 'ace.vae': 'ScragVAE' };
const desk = (engine: EngineId, tier: Tier, on: string[], vals: Desk['vals'] = {}, state: Desk['state'] = {}): Desk =>
  ({ engine, tier, vals, auto: {}, state, lists: LISTS, selection: SEL, vram: FIG, on: new Set([...DEFINITIONS.filter(d => d.kind === 'module' && d.builtIn).map(d => d.id), ...on]) });
const s = (m: unknown) => say(m as Msg);
const fixed = (v: number) => () => v;

/* ---- what a press makes */
{
  const r0 = refusalOf(ix, desk('ace', 'studio', []));
  ok('an empty box on ACE-Step asks for a description, some lyrics or Instrumental, with an example', r0?.kind === 'empty' && s(r0.says) === 'ACE-Step needs a description, some lyrics, or Instrumental' && !!r0.example, r0);
  ok('Instrumental is enough on ACE-Step', refusalOf(ix, desk('ace', 'studio', [], { 'lyrics.instr': true })) === null);
  const sa = refusalOf(ix, desk('sa3', 'studio', [])), hm = refusalOf(ix, desk('heartmula', 'studio', [], { 'lyrics.instr': true }));
  ok('each engine names its box (S-05)', sa?.kind === 'empty' && s(sa.says) === 'Stable Audio 3 needs a prompt first' && hm?.kind === 'empty' && s(hm.says) === 'HeartMuLa needs some tags first', [sa, hm]);
  const big = desk('minimax-m3', 'studio', ['takes'], { 'prompt.text': 'big', 'takes.n': 4 }), rb = refusalOf(ix, big);
  ok('a run certain not to fit is refused before it starts (F-21), with the takes that fit', rb?.kind === 'wontFit' && rb.fits === 2 && /^Won't fit: needs 37\.4 of 31\.8 GB\. Use 2 takes$/.test(s(rb.says)), rb && 'says' in rb ? s(rb.says) : rb);
  /* what the page asks for (runOf), and what the simulated server makes of it (its draft) */
  const ask = runOf(ix, desk('ace', 'studio', ['takes'], { 'prompt.text': 'warm folk song, acoustic', 'takes.n': 3 }), fixed(0.5)), run = draftRun(ask, fixed(0.5));
  ok('the page asks for no title, length or colour it did not set: the server says them (RA-12)', ask.songs.every(x => x.title === undefined && x.dur === undefined && x.hue === undefined), ask.songs);
  ok('one song per take, titled from the first three words, each take numbered', run.songs.length === 3 && run.songs.every((x, i) => x.title === 'Warm folk song' && x.take === i + 1 && x.of === 3), run.songs);
  ok('a take\'s length where the model ends it: 150 s and up to 90 more', run.songs[0]!.dur === 195, run.songs[0]!.dur);
  ok('the run keeps the desk it started from', run.engine === 'ace' && run.desk.vals['takes.n'] === 3);
  const sc = runOf(ix, desk('minimax-m3', 'studio', [], { 'prompt.text': 'Global Metadata: city pop, bright\nVocal Details: female', 'song.artist': 'The Kettles' }), fixed(0));
  ok('a Structured Caption\'s card line is its Global Metadata, after the artist (E-35); Reuse keeps it whole (E-20)', sc.songs[0]!.style === 'The Kettles · city pop, bright' && sc.songs[0]!.text.startsWith('Global Metadata:'), sc.songs[0]);
  ok('Stable Audio 3 renders 30 s without a length', draftRun(runOf(ix, desk('sa3', 'studio', [], { 'prompt.text': 'pads' }), fixed(0)), fixed(0)).songs[0]!.dur === 30);
  const w = runOf(ix, desk('ace', 'studio', [], { 'prompt.text': 'a {b|c} d', 'wild.auto': true }), fixed(0.9));
  ok('wildcards are expanded at Generate', w.songs[0]!.style === 'a c d', w.songs[0]!.style);
  const lit = runOf(ix, desk('ace', 'lab', ['post', 'norm', 'gain'], { 'prompt.text': 'x', 'pp.norm.on': true, 'pp.gain.on': false }), fixed(0));
  ok('the pedals that run are the stage line\'s last steps, in chain order', lit.lit.length >= 1 && s(stageOf(0.95, 'ace', 12, lit.lit)) === s(lit.lit[lit.lit.length - 1]), lit.lit.map(s));
}

/* ---- what a run says */
{
  ok('planning, steps, decoding (35-gobar.js stage)', s(stageOf(0.1, 'yue2', 12, [])) === 'Composing the lead sheet' && s(stageOf(0.5, 'minimax-m3', 12, [])) === 'Flow step 6 of 12' && s(stageOf(0.5, 'ace', 30, [])) === 'Step 15 of 30' && s(stageOf(0.85, 'ace', 12, [])) === 'Decoding audio');
  ok('takes start one after another', Math.abs(takeProgress(0.5, 0) - 0.6) < 1e-9 && Math.abs(takeProgress(0.5, 2) - 0.3) < 1e-9 && takeProgress(1, 1) === 1);
  ok('what can be played is a window behind the render (E-40)', readyOf(0.25) === 0 && readyOf(1) === 1);
  ok('the dice light up through planning', attemptsAt(0, 3) === 1 && attemptsAt(0.15, 3) === 3 && attemptsAt(0.5, 3) === 3);
  const rf = refusedWords('minimax-m3', 2, { need: 30.6, card: 30, fits: 1 });
  ok('a refusal the server makes: its words and the takes that fit', s(rf.why) === 'Needs about 30.6 GB for 2 takes at once; the card has 30 GB.' && s(rf.more) === 'MiniMax-Music3 checks before it loads anything.' && rf.retry === 1, [s(rf.why), rf.retry]);
}

/* ---- the go bar */
{
  ok('the line at rest', J(['Studio', 'no modules', 'everything on defaults'].join(' · ')) === J([s(goLine(ix, desk('ace', 'studio', [])).tier), s(goLine(ix, desk('ace', 'studio', [])).rest[0]!.says), s(goLine(ix, desk('ace', 'studio', [])).rested)].join(' · ')));
  const hp = goLine(ix, desk('minimax-m3', 'quick', ['sampler', 'latent', 'perf', 'repetition', 'guidance', 'planner']));
  ok('hidden and parked counted once each (E-13)', hp.rest.some(r => r.hidden && s(r.says) === '2 hidden in Quick') && hp.rest.some(r => s(r.says) === '2 parked'), hp.rest.map(r => s(r.says)));
  const big = goLine(ix, desk('minimax-m3', 'studio', ['takes'], { 'takes.n': 4 }));
  ok('a run that will not fit is said with its way out (G-B-08)', !!big.wontFit && s(big.wontFit.needs) === ': needs 37.4 of 31.8 GB' && s(big.wontFit.use) === 'Use 2 takes', big.wontFit && [s(big.wontFit.needs), s(big.wontFit.use)]);
  const ch = desk('ace', 'lab', ['post', 'ss', 'norm'], { 'pp.ss.on': true });
  ok('a pedal\'s own switch is no change (S-04)', changedKeys(ix, ch).length === 0, changedKeys(ix, ch));
  ok('a box\'s text is no change; its switches are', changedKeys(ix, desk('ace', 'studio', [], { 'prompt.text': 'x', 'lyrics.instr': true })).join() === 'lyrics.instr', changedKeys(ix, desk('ace', 'studio', [], { 'prompt.text': 'x', 'lyrics.instr': true })));
  const hid = changesUnmarked(ix, desk('ace', 'studio', ['perf'], { 'perf.cache': 0.3 }));
  ok('changes only in a hidden module say so and offer Lab (Q-15)', hid.lab && / in .+, which Studio hides$/.test(s(hid.says)), s(hid.says));
  const g0 = goModel(ix, desk('ace', 'studio', []), 0, 0), g2 = goModel(ix, desk('ace', 'studio', []), 3, 3);
  ok('Generate at rest and with a queue (C-26)', s(g0.title) === 'Generate (Ctrl+Enter)' && s(g0.cancel) === 'Cancel' && s(g0.cancelLabel) === 'Cancel the song' && s(g2.title) === 'Generate (Ctrl+Enter). 3 running, 3 waiting; another one queues after them.' && s(g2.cancel) === 'Cancel all ' && g2.count === '6', [s(g0.title), s(g2.title)]);
  const rp = goModel(ix, desk('ace', 'studio', ['repaint'], {}, { source: { song: 1, title: 'A', style: 'b', dur: 100, hue: 4 } }), 0, 0);
  ok('with a repaint on the desk Generate repaints', s(rp.label) === 'Repaint region');
  ok('the title carries the count', s(docTitle('HOT-Step', 3)) === '(3) HOT-Step' && s(docTitle('HOT-Step', 0)) === 'HOT-Step');
}

/* ---- cancelling, Latest, the menu, the player */
{
  const c = cancelWords(2, null), one = cancelWords(1, { title: 'Song', take: 2, of: 3 });
  ok('cancelling says what it did, and its Undo what it put back (F-13)', s(c.says) === 'Cancelled 2 songs' && s(one.says) === 'Cancelled "Song, take 2 of 3"; the rest carry on' && s(cancelBack(2, true)) === 'They are back in the queue; the one that was rendering starts again' && s(cancelBack(1, false)) === 'It is back in the queue');
  const l1 = latestShown(Array(10).fill('song'), Array(10).fill(0), 4, false);
  ok('finished songs fill whole rows, two at most (D-26)', l1.hidden.filter(h => !h).length === 8 && s(l1.count) === '8 of your 10 songs');
  const kinds = [...Array(6).fill('pending'), ...Array(6).fill('song')] as ('song' | 'pending')[], l2 = latestShown(kinds, [0, 0, 0, 100, 100, 100, 0, 0, 0, 0, 0, 0], 4, false);
  ok('a queue longer than a row folds (C-47), the songs rendering now in sight', !!l2.fold && s(l2.fold.title) === '+3 waiting' && l2.folded.slice(0, 3).every(f => !f), l2);
  const l3 = latestShown(kinds, [0, 0, 0, 100, 100, 100, 0, 0, 0, 0, 0, 0], 4, true);
  ok('and opens', !!l3.fold && l3.folded.every(f => !f) && s(l3.fold.title) === 'Fold the queue');
  const card = songCard({ title: 'T', take: 1, of: 2, style: 'x', dur: 236, engine: 'sa3' });
  ok('a card\'s words', s(card.name) === 'T, take 1 of 2' && s(card.take!) === '1/2' && card.dur === '3:56' && card.engine === 'Stable Audio 3');
  ok('the menu says Repaint switches from another engine (E-14)', s(songMenu('minimax-m3', 'ace')[0]!.small!) === 'Switches to ACE-Step 1.5 and puts Repaint on the desk' && s(songMenu('ace', 'minimax-m3')[1]!.says) === 'Reuse its caption');
  const rd = repaintDesk(ix, desk('minimax-m3', 'studio', ['tempo', 'cont']), emptyBook(), ['tempo', 'cont'], { id: '1', title: 'Studio sketch 01', style: 'x', dur: 171, hue: 215 });
  ok('Repaint from another engine: on ACE-Step, Repaint first, the song as its source (E-14)', rd.desk.engine === 'ace' && rd.mods[0] === 'repaint' && (rd.desk.state.source as { song: number }).song === 1 && /^On ACE-Step 1\.5\..*Repaint is on the desk for Studio sketch 01/.test(s(rd.says)), [rd.mods, s(rd.says)]);
  const pw = playerWords(35, 236, true);
  ok('the player\'s words (C-27)', pw.time === '0:35 / 3:56' && s(pw.valueText) === '0:35 of 3:56' && s(pw.tip) === 'Play (Space)');
  ok('seek keys: 5 s, Shift or Page 30 s, Home, End', playerSeek('ArrowRight', false, 0, 236) === 5 && playerSeek('ArrowRight', true, 5, 236) === 35 && playerSeek('End', false, 0, 236) === 235 && playerSeek('Home', false, 9, 236) === 0 && playerSeek('a', false, 0, 236) === null);
}

/* ---- what package 5 and package 2 left here */
{
  const d = desk('sa3', 'studio', []), runDesk = desk('ace', 'studio', []), m = meterModel(ix, d, runDesk);
  ok('while another engine renders the meter shows what it holds (B-34)', /^VRAM: ACE-Step 1\.5 is rendering with \d+\.\d GB loaded\. Next: Stable Audio 3, about \d+\.\d GB, on a 31\.8 GB card$/.test(s(m.aria)) && m.use === meterModel(ix, runDesk).use, s(m.aria));
  ok('the needle swings toward what the run needs', meterTarget(10, 14, true, fixed(0)) === 13 && meterTarget(10, 14, false, fixed(0)) === 10);
  const replan = DEFINITIONS.find(x => x.id === 'replan')!, find = (ns: Node[]): Node | undefined => { for (const x of ns) { if (x.part === 'dice') return x; const y = find(x.children ?? []); if (y) return y; } };
  const dice = find((replan as { face: Node[] }).face)!, rd = { ...desk('yue2', 'lab', ['replan'], { 'yue.tries': 4 }), songs: { attempts: 2, hue: 40 } };
  ok('the dice light the tries the run reports', readoutModel(ix, dice, rd)?.dice?.used === 2, readoutModel(ix, dice, rd));
  ok('the cover swatch takes the newest song\'s colour', readoutModel(ix, { part: 'coverSwatch' } as Node, rd)?.cover?.hue === 40);
}

if (fails.length) { console.log(`generation: ${fails.length} of ${n} checks failed\n  ` + fails.join('\n  ')); process.exit(1); }
console.log(`generation: ${n} checks of the generation core: ok`);
