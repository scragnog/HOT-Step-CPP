// The score the planner writes for the lead-sheet preview (YuE2's Chain of Thought, run alone). Server-owned: the
// planner's score is shown, never stored or undone. The simulated planner's: services/mock/score.ts.

import type { Score, ScoreAsk } from '../../derive.ts';

export interface ScoreService {
  /** A score for these sections (or the planner's own order without lyrics), from its seed. */
  plan(seed: number, ask: ScoreAsk): Promise<Score>;
}
