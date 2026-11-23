// Pedal "Spectral lifter". Concept: src/js/27-pedals.js.

import type { PedalDef } from '../types.ts';

export const lift = {
  kind: 'pedal',
  id: 'lift',
  name: 'Spectral lifter',
  desc: 'Removes AI shimmer and noise, and can extend the high end.',
  at: 50,
  engines: ['ace'],
  hue: 140,
  on: 'pp.lift.on',
  main: 'pp.lift.den',
  thumbWidth: 2,
  owner: 'song',
  stored: 'settings',
  params: [
    { key: 'pp.lift.on', type: 'boolean', label: 'On', default: false, wire: 'param:spectralLifterEnabled' },
    {
      key: 'pp.lift.den',
      type: 'number',
      label: 'Denoise',
      range: [0, 1, 0.05],
      default: 0.3,
      wire: 'param:slDenoiseStrength',
    },
    {
      key: 'pp.lift.floor',
      type: 'number',
      label: 'Floor',
      range: [0.01, 0.5, 0.01],
      default: 0.1,
      wire: 'param:slNoiseFloor',
    },
    {
      key: 'pp.lift.hf',
      type: 'number',
      label: 'HF ext',
      range: [0, 0.5, 0.01],
      default: 0,
      wire: 'param:slHfMix',
    },
    {
      key: 'pp.lift.tr',
      type: 'number',
      label: 'Transient',
      short: 'Attack',
      range: [0, 1, 0.05],
      default: 0,
      wire: 'param:slTransientBoost',
    },
    {
      key: 'pp.lift.shim',
      type: 'number',
      label: 'Shimmer',
      range: [0, 12, 0.5],
      unit: 'dB',
      default: 6,
      wire: 'param:slShimmerReduction',
    },
  ],
  thumb: { file: 'pd-lift' },
} satisfies PedalDef;
