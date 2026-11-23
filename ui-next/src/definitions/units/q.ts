// Side unit "Quality evaluator". Concept: src/js/28-side-units.js.

import type { UnitDef } from '../types.ts';

export const q = {
  kind: 'unit',
  id: 'q',
  name: 'Quality evaluator',
  desc: 'A spectral score, shown as a badge in the Library.',
  at: 50,
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3'],
  on: 'pp.q.on',
  gated: true,
  owner: 'song',
  stored: 'settings',
  params: [
    { key: 'pp.q.on', type: 'boolean', label: 'Quality evaluator', default: false, wire: 'param:qualityEvalEnabled' },
    {
      key: 'pp.q.eval',
      type: 'choice',
      label: 'Evaluate',
      default: 'unmastered',
      options: [
        { value: 'unmastered', label: 'Unmastered', short: 'Raw' },
        { value: 'mastered', label: 'Mastered', short: 'Master' },
        { value: 'both', label: 'Both' },
      ],
      wire: 'param:qualityEvalTarget',
    },
  ],
  face: [
    { part: 'unitBadge', tip: 'The score each song gets, as its badge in the Library', text: '8.4' },
  ],
  thumb: { file: 'ex-q' },
} satisfies UnitDef;
