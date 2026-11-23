// Module "Seed". Concept: src/js/25-module-seed.js (and its css).

import type { ModuleDef } from '../types.ts';

export const seed = {
  kind: 'module',
  id: 'seed',
  starter: { at: 3 },
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
  state: [
    {
      key: 'seed',
      kind: 'seed',
      noun: 'Seed',
      empty: 'random',
      steps: { set: '{noun} {from} → {to}' },
      owner: 'song',
      stored: 'none',
      /* the seed is kept to 2^64-1, as the application keeps it; ACE-Step, MiniMax-Music3 and Stable Audio 3 are sent a
         number (the existing page's getGlobalParams sends Number(seed) there), YuE2 and HeartMuLa the decimal text */
      wire: [
        { engines: ['ace', 'minimax-m3', 'sa3'], wire: { to: 'param:seed', via: 'int' } },
        { engines: ['yue2', 'heartmula'], wire: 'param:seed' },
      ],
    },
    {
      key: 'stars',
      kind: 'list',
      noun: 'the saved seeds',
      steps: { save: 'Save seed {seed}', update: 'Update seed {seed}', delete: 'Delete seed {seed}', star: 'Star {seed}', unstar: 'Unstar {seed}' },
      owner: 'song',
      stored: 'none',
      wire: { none: 'The page\'s own list of saved seeds.' },
    },
  ],
  face: [
    /* the flaps with the lock beside them, then the LM seed's link under them (it reads the LM's switch), then Roll once
       and Random each time */
    {
      part: 'splitFlap',
      param: 'seed.lock',
      state: ['seed'],
      reads: ['lm.seedlock'],
      children: [
        { part: 'lockButton', param: 'seed.lock', tiers: ['studio', 'lab'] },
        { part: 'pill', param: 'lm.seedlock', label: 'LM seed follows', engines: ['ace'], tiers: ['lab'] },
      ],
    },
    { part: 'savedSeeds', param: 'seed.lock', state: ['stars', 'seed'], tiers: ['studio', 'lab'] },
  ],
  thumb: { file: 'seed' },
} satisfies ModuleDef;
