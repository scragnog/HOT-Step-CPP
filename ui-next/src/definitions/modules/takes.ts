// Module "Takes". Concept: src/js/25-module-takes.js (and its css).

import type { ModuleDef } from '../types.ts';

export const takes = {
  kind: 'module',
  id: 'takes',
  name: 'Takes',
  desc: 'How many songs each Generate makes.',
  at: 210,
  category: 'Engine',
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
  owner: 'song',
  stored: 'settings',
  size: { w: 1, grow: true, stretch: 144 },
  per: { sa3: { size: { w: 4 } }, heartmula: { size: { w: 4 } } },
  tierSummary: ['Songs per run', 'The same', 'The same'],
  params: [
    {
      key: 'takes.n',
      type: 'number',
      label: 'Takes per run',
      tip: 'Each take is a separate song from the same settings, with its own seed',
      range: [1, 9, 1],
      default: 1,
      held: true,
      wire: 'param:batchSize',
      per: {
        'minimax-m3': {
          wire: 'ext:mm3Takes',
          range: [1, 4, 1],
          tip: 'MiniMax-Music3 calls these variations: songs rendered together from one plan',
        },
        yue2: {
          wire: 'ext:yue2BatchSize',
          range: [1, 4, 1],
          tip: 'YuE2 calls this the batch size: songs composed side by side',
        },
        sa3: { wire: { none: 'One song per run.' }, range: [1, 1, 1], tip: 'One song per run' },
        heartmula: { wire: { none: 'One song per run.' }, range: [1, 1, 1], tip: 'One song per run' },
        ace: { range: [1, 9, 1] },
      },
    },
  ],
  strip: [{ when: { engine: ['sa3', 'heartmula'] }, says: 'One song per run · Generate again to queue another' }],
  face: [{ part: 'takesReels', param: 'takes.n' }],
  thumb: { file: 'takes' },
} satisfies ModuleDef;
