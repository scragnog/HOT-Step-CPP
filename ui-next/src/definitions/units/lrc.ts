// Side unit "Lyric timestamps". Concept: src/js/28-side-units.js.

import type { UnitDef } from '../types.ts';

export const lrc = {
  kind: 'unit',
  id: 'lrc',
  name: 'Lyric timestamps',
  desc: 'Synced lyric timing: from the DiT on ACE-Step, from the planner on MiniMax-Music3.',
  at: 10,
  engines: ['ace', 'minimax-m3'],
  on: 'pp.lrc.on',
  owner: 'song',
  stored: 'settings',
  params: [
    {
      key: 'pp.lrc.on',
      type: 'boolean',
      label: 'Lyric timestamps',
      default: true,
      wire: { to: 'param:skipLrc', invert: true, note: 'skipLrc is the opposite.' },
    },
  ],
  face: [{ part: 'lrcPreview' }],
  thumb: { file: 'ex-lrc' },
} satisfies UnitDef;
