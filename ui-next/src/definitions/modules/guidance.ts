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
      per: [{ engines: ['sa3', 'heartmula'], wire: { to: 'param:apgMomentum', differs: ['unread'], note: 'The concept has it here; nothing this engine runs reads it today, so it changes nothing until the server does (docs/08, question 3).' } }],
    },
    {
      key: 'apg.norm',
      type: 'number',
      label: 'APG norm threshold',
      range: [0, 10, 0.1],
      default: 2.5,
      when: { param: 'samp.guide', is: 'apg' },
      engines: ['ace', 'minimax-m3', 'sa3', 'heartmula'],
      wire: 'param:apgNormThreshold',
      per: [{ engines: ['sa3'], wire: { to: 'param:apgNormThreshold', differs: ['unread'], note: 'The concept has it here; nothing this engine runs reads it today, so it changes nothing until the server does (docs/08, question 3).' } }],
    },
  ],
  strip: [
    {
      when: { not: { any: [{ engine: ['ace', 'yue2'] }, { param: 'samp.plugins', is: true }] } },
      says: '{engine} uses its own guidance',
      actions: [
        {
          do: 'set',
          param: 'samp.plugins',
          value: true,
          label: 'Turn on Lua plugins',
          tip: 'Lua plugins are experimental on {engine}: they were tuned for ACE-Step',
        },
      ],
    },
    {
      when: { all: [{ engine: ['minimax-m3', 'sa3', 'heartmula'] }, { param: 'samp.guide', is: '' }] },
      says: 'Native guidance has no settings of its own',
    },
    { when: { param: 'samp.guide', not: 'apg' }, says: '{value:samp.guide} has no settings of its own' },
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
      tip: 'Stable Audio 3 has no unconditional prediction to steer against',
      text: 'Little effect on Stable Audio 3',
      engines: ['sa3'],
    },
    { part: 'row', engines: ['minimax-m3'], children: [{ part: 'knob', param: 'apg.norm' }] },
    { part: 'note', text: 'Only the norm threshold applies here', engines: ['minimax-m3'] },
  ],
  thumb: { file: 'guidance' },
} satisfies ModuleDef;
