// Module "Guidance plugin". Concept: src/js/25-module-guidance.js (and its css).

import type { ModuleDef } from '../types.ts';

export const guidance = {
  kind: 'module',
  id: 'guidance',
  name: 'Guidance plugin',
  desc: 'The chosen guidance plugin\'s own settings, read from its Lua file.',
  at: 90,
  category: 'Engine',
  labUnder: 'sampler',
  engines: ['ace', 'minimax-m3', 'sa3', 'heartmula'],
  owner: 'engine',
  stored: 'settings',
  size: { w: 1, stretch: 144 },
  params: [
    {
      key: 'apg.mom',
      type: 'number',
      label: 'APG momentum',
      range: [0, 1, 0.01],
      default: 0.75,
      when: { param: 'samp.guide', is: 'apg' },
      engines: ['ace', 'sa3', 'heartmula'],
      wire: 'param:apgMomentum',
    },
    {
      key: 'apg.norm',
      type: 'number',
      label: 'APG norm threshold',
      range: [0, 10, 0.1],
      default: 2.5,
      when: { param: 'samp.guide', is: 'apg' },
      wire: 'param:apgNormThreshold',
    },
  ],
  strip: [
    {
      when: { not: { any: [{ engine: ['ace', 'yue2'] }, { param: 'samp.plugins', is: true }] } },
      says: 'This engine uses its own guidance',
      action: {
        do: 'set',
        param: 'samp.plugins',
        value: true,
        label: 'Turn on Lua plugins',
        tip: 'Lua plugins are experimental here: they were tuned for ACE-Step',
      },
    },
    { when: { param: 'samp.guide', is: '' }, says: 'Native guidance has no settings of its own' },
    { when: { param: 'samp.guide', not: 'apg' }, says: 'This guidance mode has no settings of its own' },
  ],
  face: [
    {
      part: 'xyPad',
      params: ['apg.mom', 'apg.norm'],
      opts: { look: 'apg', xl: 'momentum', yl: 'norm threshold' },
      engines: ['ace', 'sa3', 'heartmula'],
    },
    {
      part: 'note',
      text: 'Little effect on Stable Audio 3',
      engines: ['sa3'],
      tip: 'Stable Audio 3 has no unconditional prediction to steer against',
    },
    { part: 'row', engines: ['minimax-m3'], children: [{ part: 'knob', param: 'apg.norm' }] },
    { part: 'note', text: 'Only the norm threshold applies here', engines: ['minimax-m3'] },
  ],
  thumb: { file: 'guidance' },
} satisfies ModuleDef;
