// What the simulated server says the songs of a run will be (the generation service's draft), as the concept invents
// them (35-gobar.js generate): a title from the first words of the description, a repaint's after its song; a length of
// 150 s and up to 90 more where the model ends the song (never past the longest it may run), Stable Audio 3's 30 s
// without one; a colour at random, a repaint's beside its song's. What the page set is kept.

import type { RunAsk, RunSpec } from '../../../derive.ts';

/** A song's title from its words: the first three words of the description, the first one capitalised. */
const titleFrom = (p: string): string => { const w = p.split(/[,.\n]/)[0]!.replace(/Global Metadata:\s*/, '').trim().split(/\s+/).slice(0, 3).join(' '); return w ? w[0]!.toUpperCase() + w.slice(1) : 'Untitled'; };

/** The run with its songs said; `rnd` is the source of chance (a test fixes it). */
export function draftRun(ask: RunAsk, rnd: () => number): RunSpec {
  return { ...ask, songs: ask.songs.map(x => ({
    ...x,
    title: x.title ?? (x.repaints ? `${x.repaints.title} (repainted)` : titleFrom(x.text)),
    dur: x.dur ?? (ask.engine === 'sa3' ? 30 : Math.min(x.most ?? Infinity, 150 + Math.round(rnd() * 90))),
    hue: x.hue ?? (x.repaints ? (x.repaints.hue + 25) % 360 : Math.round(rnd() * 360)),
  })) };
}
