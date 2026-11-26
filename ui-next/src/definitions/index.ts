// Every definition, in one list. Adding a definition file means adding its line here; the validator fails until you do.
// Each file checks itself with `satisfies`, so this list adds no second report of the same mistake.

import type { Definition } from './types.ts';
import { prompt } from './modules/prompt.ts';
import { lyrics } from './modules/lyrics.ts';
import { models } from './modules/models.ts';
import { repaint } from './modules/repaint.ts';
import { tempo } from './modules/tempo.ts';
import { avoid } from './modules/avoid.ts';
import { timbre } from './modules/timbre.ts';
import { adapters } from './modules/adapters.ts';
import { planner } from './modules/planner.ts';
import { repetition } from './modules/repetition.ts';
import { sampler } from './modules/sampler.ts';
import { guidance } from './modules/guidance.ts';
import { perf } from './modules/perf.ts';
import { latent } from './modules/latent.ts';
import { score } from './modules/score.ts';
import { build } from './modules/build.ts';
import { cont } from './modules/cont.ts';
import { flow } from './modules/flow.ts';
import { plans } from './modules/plans.ts';
import { replan } from './modules/replan.ts';
import { adhere } from './modules/adhere.ts';
import { yuebatch } from './modules/yuebatch.ts';
import { seed } from './modules/seed.ts';
import { takes } from './modules/takes.ts';
import { post } from './modules/post.ts';
import { tile } from './pedals/tile.ts';
import { refiner } from './pedals/refiner.ts';
import { ss } from './pedals/ss.ts';
import { ppvae } from './pedals/ppvae.ts';
import { lift } from './pedals/lift.ts';
import { nat } from './pedals/nat.ts';
import { gain } from './pedals/gain.ts';
import { vst } from './pedals/vst.ts';
import { master } from './pedals/master.ts';
import { norm } from './pedals/norm.ts';
import { lrc } from './units/lrc.ts';
import { fa } from './units/fa.ts';
import { wh } from './units/wh.ts';
import { art } from './units/art.ts';
import { q } from './units/q.ts';

export const DEFINITIONS = [
  prompt,
  lyrics,
  models,
  repaint,
  tempo,
  avoid,
  timbre,
  adapters,
  planner,
  repetition,
  sampler,
  guidance,
  perf,
  latent,
  score,
  build,
  cont,
  flow,
  plans,
  replan,
  adhere,
  yuebatch,
  seed,
  takes,
  post,
  tile,
  refiner,
  ss,
  ppvae,
  lift,
  nat,
  gain,
  vst,
  master,
  norm,
  lrc,
  fa,
  wh,
  art,
  q,
] as readonly Definition[];
