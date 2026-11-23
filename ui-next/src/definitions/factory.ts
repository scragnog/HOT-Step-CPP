// The factory workflows and their revisions: the desks the page ships with, and named settings saved under them.
// Concept: src/js/32-workflows.js (FACTORY, FACTORY_REVS). They are saved data in the page's own format (types.ts,
// "Saved data"), so the validator checks them as it checks a loaded file: every module and pedal exists, every
// value is one its parameter can hold on the revision's engine, and a revision holds only what a revision keeps.
//
// A factory revision lists only what it sets; loading it first resets every setting a revision keeps for its
// engine (docs/05, "Saved data").

import type { Revision, Workflow } from './types.ts';

/* The order of "Everything" composes at every desk width on every engine (the concept chose it with a search). */
const EVERYTHING = ['guidance', 'flow', 'perf', 'tempo', 'sampler', 'latent', 'repetition', 'planner', 'plans', 'seed', 'timbre', 'avoid', 'takes', 'score', 'replan', 'cont', 'adhere', 'adapters', 'yuebatch', 'build', 'post'];
const BOARD = ['tile', 'refiner', 'ss', 'ppvae', 'lift', 'nat', 'gain', 'vst', 'master', 'norm', 'lrc', 'fa', 'wh', 'art', 'q'];

export const FACTORY_WORKFLOWS: Workflow[] = [
  { v: 1, name: 'Start fresh', tier: 'quick', modules: [], board: [], note: 'Just the prompt and lyrics' },
  { v: 1, name: 'Songwriter', tier: 'quick', modules: ['tempo', 'avoid', 'takes'], board: ['lrc'] },
  { v: 1, name: 'Lo-fi tape', tier: 'studio', modules: ['adapters', 'tempo', 'sampler', 'post'], board: ['nat', 'norm'] },
  { v: 1, name: 'Radio master', tier: 'studio', modules: ['post'], board: ['ss', 'lift', 'norm', 'q'] },
  { v: 1, name: 'Seed hunter', tier: 'studio', modules: ['seed', 'sampler', 'takes'], board: [] },
  { v: 1, name: 'Deep sampler', tier: 'lab', modules: ['sampler', 'planner', 'guidance', 'perf', 'latent', 'repetition'], board: [] },
  { v: 1, name: 'Everything', tier: 'lab', modules: EVERYTHING, board: BOARD },
];

export const FACTORY_REVISIONS: Revision[] = [
  {
    v: 1, workflow: 'Songwriter', name: 'Pop 3 takes', engine: 'ace', withContent: true,
    vals: { 'takes.n': 3, 'prompt.text': 'bright pop, punchy drums, catchy female vocals' },
    note: 'Three takes of a bright pop idea',
  },
  {
    v: 1, workflow: 'Lo-fi tape', name: 'Tape at 82 BPM', engine: 'ace', withContent: true,
    vals: { 'samp.steps': 30, 'samp.sched': 'cosine', s0: 0.8, 'pp.nat.on': true, 'pp.norm.on': true, 'tempo.bpm': 82, 'prompt.text': 'lo-fi hip hop with tape hiss and a lazy Rhodes' },
    auto: { 'tempo.bpm': false },
    state: { slots: [{ file: 'tape-lofi-v2', type: 'LoRA' }] },
  },
  { v: 1, workflow: 'Radio master', name: 'Loud and clean', engine: 'ace', vals: { 'pp.ss.on': true, 'pp.ss.str': 25, 'pp.lift.on': true, 'pp.norm.on': true, 'pp.q.on': true } },
  { v: 1, workflow: 'Radio master', name: 'Loud and clean', engine: 'minimax-m3', vals: { 'pp.ss.on': true, 'pp.ss.str': 25, 'pp.norm.on': true, 'pp.q.on': true } },
  { v: 1, workflow: 'Seed hunter', name: 'Wide net', engine: 'ace', vals: { 'takes.n': 4, 'samp.retake': 0.3 }, note: 'Four takes with some retake variance' },
  { v: 1, workflow: 'Deep sampler', name: 'Quick ideas', engine: 'ace', vals: { 'samp.steps': 8, 'perf.cache': 0.3 }, note: 'Few steps and a step cache, for auditioning' },
  {
    v: 1, workflow: 'Deep sampler', name: 'Polishing', engine: 'ace',
    vals: { 'samp.steps': 40, 'samp.solver': 'dpm3m', 'samp.sched': 'beta57', 'samp.cs': 0.1, 'samp.ce': 0.85, 'apg.mom': 0.6 },
    note: 'More steps, a CFG interval, a gentler APG',
  },
  {
    v: 1, workflow: 'Deep sampler', name: 'Faithful render', engine: 'minimax-m3', vals: { 'samp.steps': 30, 'samp.cfg': 2.2, 'mm.temp': 0.9 },
    note: 'The checkpoint\'s own 30 flow steps, a little more guidance',
  },
  {
    v: 1, workflow: 'Deep sampler', name: 'Plan-Play loose', engine: 'yue2', vals: { 'samp.solver': 'ancestral', 'samp.sched': 'plan_loose', 'yp.temp': 0.8 },
    note: 'In the spirit of the importable Plan-Play presets',
  },
  { v: 1, workflow: 'Start fresh', name: 'Quick sketch', engine: 'sa3', vals: { 'samp.steps': 4 }, note: 'Four steps, for auditioning prompts' },
  { v: 1, workflow: 'Start fresh', name: 'Reference', engine: 'heartmula', vals: {}, note: 'HeartMuLa\'s reference sampling: temperature 1.0, top-K 50, CFG 1.5' },
];
