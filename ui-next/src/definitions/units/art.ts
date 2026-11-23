// Side unit "Cover art". Concept: src/js/28-side-units.js.

import type { UnitDef } from '../types.ts';

export const art = {
  kind: 'unit',
  id: 'art',
  name: 'Cover art',
  desc: '1024×1024 cover from FLUX.2-klein-4B.',
  at: 40,
  engines: ['ace', 'minimax-m3', 'yue2'],
  on: 'pp.art.on',
  gated: true,
  owner: 'song',
  stored: 'settings',
  params: [
    { key: 'pp.art.on', type: 'boolean', label: 'Cover art', default: false, wire: 'param:coverArtEnabled' },
    {
      key: 'pp.art.subject',
      type: 'text',
      label: 'Cover art subject',
      tip: 'What the cover shows. Leave it empty to draw from the song\'s caption.',
      default: '',
      stored: 'content',
      wire: 'param:coverArtSubject',
    },
  ],
  face: [
    { part: 'coverSwatch' },
  ],
  thumb: { file: 'ex-art' },
} satisfies UnitDef;
