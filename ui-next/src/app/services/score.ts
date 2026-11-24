// The score the planner writes for the lead-sheet preview (YuE2's Chain of Thought, run alone), and the run that
// renders it. Server-owned: the planner's score is shown, never stored or undone.
//
// This is the simulated planner (the concept's 34-score-modal.js planScore): a score drawn from its seed, four bars a
// section, G major at 96 BPM; most plans are healthy, some run on past the lyrics ("long"), some loop their last
// section and never close ("runaway"). What is drawn is what plays (E-23). The real adapter is package 10's.
//
// Rendering a score is generation's (package 6): it registers how (`scoreRuns.use`); until it does, Continue only
// closes the preview.

import { defineService } from '../register/services.ts';
import { SCORE_Q, SCORE_SCALE, barChord, type Score, type ScoreAsk } from '../../derive.ts';

const rng = (seed: number) => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const DEFAULT_ORDER = ['Verse 1', 'Chorus', 'Verse 2', 'Chorus', 'Bridge', 'Chorus'];

export const scorePlanner = defineService('score', {
  /** A score for these sections (or the planner's own order without lyrics), from its seed. */
  plan(seed: number, ask: ScoreAsk): Promise<Score> {
    const r = rng(seed), order = ask.names.length ? ask.names : DEFAULT_ORDER;
    const roll = r(), verdict = roll < 0.72 ? 'healthy' : roll < 0.9 ? 'long' : 'runaway';
    const bar = (name: string) => Array.from({ length: 4 }, () => Math.floor(r() * 7) + (name.toLowerCase().startsWith('chorus') ? 2 : 0));
    const bars = order.map(name => Array.from({ length: 4 }, () => bar(name)));
    const meas: Score['meas'] = bars.flatMap((sec, si) => sec.map((b, bi) => ({ b, si, first: bi === 0 }))), lyricBars = meas.length, last = order.length - 1;
    if (verdict === 'long') { const extra = Math.round(lyricBars * 0.6); for (let i = 0; i < extra; i++) meas.push({ b: bar(order[last]!), si: last, past: true, first: i === 0 }); }
    if (verdict === 'runaway') bars[last]!.forEach((b, bi) => meas.push({ b, si: last, loop: true, first: bi === 0 }));
    const est = Math.round(meas.length * 4 * SCORE_Q), title = ask.title || 'Untitled', chords = ask.chords;
    const line = (ms: Score['meas']) => ms.map(m => (chords ? `"${barChord(m)}" ` : '') + m.b.map(x => SCORE_SCALE[Math.min(8, x)]).join(' ')).join(' | ');
    const abc = `X:1\nT:${title}\nM:4/4\nL:1/4\nQ:1/4=96\nK:G\n` + order.map((n, i) => `% ${n}\nP:${n}\nV:Vocal\n` + line(meas.filter(m => m.si === i && !m.past && !m.loop)) + ' |').join('\n')
      + (verdict === 'long' ? `\n% past the lyrics\n${line(meas.filter(m => m.past))} |]` : verdict === 'runaway' ? `\n% loops, never closes\n|: ${line(meas.filter(m => m.loop))} :|` : ']');
    return Promise.resolve({ seed, order, meas, lyricBars, verdict, est, abc, title, chords });
  },
});

/** How a planned score is rendered with its seed, for that run only (E-08): generation registers it. */
let runner: ((seed: number) => boolean) | null = null;
export const scoreRuns = {
  use(run: (seed: number) => boolean): void { runner = run; },
  /** Renders the score with its seed; false while nothing renders scores (or the run was refused). */
  run(seed: number): boolean { return runner ? runner(seed) : false; },
};
