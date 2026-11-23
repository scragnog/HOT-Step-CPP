// Side unit "Forced alignment". Concept: src/js/28-side-units.js.

import type { UnitDef } from '../types.ts';

export const fa = {
  kind: 'unit',
  id: 'fa',
  name: 'Forced alignment',
  desc: 'Lyric timestamps for YuE2, aligned after the render. Needs the MMS aligner model.',
  at: 20,
  engines: ['yue2'],
  on: 'pp.fa.on',
  owner: 'song',
  stored: 'settings',
  params: [
    {
      key: 'pp.fa.on',
      type: 'boolean',
      label: 'Lyric timestamps (forced alignment)',
      default: false,
      wire: 'param:yue2AlignLyrics',
    },
  ],
  face: [
    {
      part: 'unitStatus',
      tip: 'Aligns lyric timestamps after the render. Needs the MMS aligner model.',
      text: 'After the render · needs MMS',
    },
  ],
  thumb: { file: 'ex-fa' },
} satisfies UnitDef;
