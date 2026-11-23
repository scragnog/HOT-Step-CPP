// Module "Seed". Concept: src/js/25-module-seed.js (and its css).

import type { ModuleDef } from '../types.ts';

export const seed = {
  kind: 'module',
  id: 'seed',
  name: 'Seed',
  desc: 'Reproduce a take, and keep the seeds you like.',
  at: 200,
  category: 'Engine',
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
  owner: 'song',
  stored: 'settings',
  size: { w: 1, grow: true, stretch: 144 },
  tierSummary: ['Roll or random', '+ saved seeds, keep this seed', '+ LM seed follows'],
  params: [
    {
      key: 'seed.lock',
      type: 'boolean',
      label: 'Keep this seed',
      default: false,
      wire: { to: 'param:randomSeed', invert: true, note: 'randomSeed is the opposite of Keep this seed.' },
    },
  ],
  state: [{ key: 'seed', kind: 'seed', noun: 'the seed', owner: 'song', stored: 'settings', wire: 'param:seed' }],
  face: [
    {
      part: 'splitFlap',
      opts: { state: 'seed' },
      children: [{ part: 'lockButton', param: 'seed.lock', tiers: ['studio', 'lab'] }],
    },
    { part: 'pill', param: 'lm.seedlock', label: 'LM seed follows', engines: ['ace'], tiers: ['lab'] },
    { part: 'savedSeeds', tiers: ['studio', 'lab'] },
  ],
  thumb: { file: 'seed' },
} satisfies ModuleDef;
