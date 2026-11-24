// Module "Takes". Concept: src/js/25-module-takes.js (and its css).

import type { ModuleDef } from '../types.ts';

export const takes = {
  kind: 'module',
  id: 'takes',
  starter: { at: 4 },
  name: 'Takes',
  desc: 'How many songs each Generate makes.',
  at: 210,
  category: 'Engine',
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
  owner: 'song',
  stored: 'settings',
  size: { w: 1, grow: true, stretch: 144 },
  per: [{ engines: ['sa3', 'heartmula'], size: { w: 4, stripKeepsWidth: true } }],
  tierSummary: ['Songs per run', 'The same', 'The same'],
  params: [
    {
      key: 'takes.n',
      type: 'number',
      label: 'Takes per run',
      tip: 'Each take is a separate song from the same settings, with its own seed',
      range: [1, 8, 1],
      default: 1,
      engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
      held: { note: 'Your {value} takes return on {engine}', toast: 'Takes capped at {shown}', value: '{value} takes' },
      wire: { to: 'param:batchSize', differs: ['range'], note: 'The server takes 8 at most (batch.max); the existing slider goes to 9 and sends 8.' },
      per: [
        {
          engines: ['minimax-m3'],
          wire: 'ext:mm3Takes',
          range: [1, 4, 1],
          tip: 'MiniMax-Music3 calls these variations: songs rendered together from one plan',
        },
        {
          engines: ['yue2'],
          wire: { to: 'ext:yue2BatchSize', differs: ['range'], note: 'Declared with a maximum of 1 in the static manifest; the engine raises it to its batch limit.' },
          range: [1, 4, 1],
          tip: 'YuE2 calls this the batch size: songs composed side by side',
        },
        {
          engines: ['sa3', 'heartmula'],
          wire: { none: 'One song per run.' },
          range: [1, 1, 1],
          tip: 'One song per run',
        },
      ],
    },
  ],
  strip: [{ when: { engine: ['sa3', 'heartmula'] }, says: 'One song per run · Generate again to queue another' }],
  face: [
    { part: 'takesReels', param: 'takes.n', label: 'Takes per run', per: [{ when: { param: 'takes.n', is: 1 }, label: 'Take per run' }] },
  ],
  thumb: { file: 'takes' },
} satisfies ModuleDef;
