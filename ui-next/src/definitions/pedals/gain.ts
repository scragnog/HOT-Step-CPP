// Pedal "Pre-VST gain". Concept: src/js/27-pedals.js.

import type { PedalDef } from '../types.ts';

export const gain = {
  kind: 'pedal',
  id: 'gain',
  name: 'Pre-VST gain',
  desc: 'Headroom before your VST chain.',
  at: 70,
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
  hue: 210,
  on: null,
  main: 'pp.gain',
  thumbWidth: 1,
  owner: 'song',
  stored: 'settings',
  params: [
    {
      key: 'pp.gain',
      type: 'number',
      label: 'Gain',
      range: [-10, 10, 0.5],
      unit: 'dB',
      default: 0,
      wire: 'param:gainOffsetDb',
    },
  ],
  thumb: { file: 'pd-gain' },
} satisfies PedalDef;
