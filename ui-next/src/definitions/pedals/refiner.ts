// Pedal "MM3 latent refiner". Concept: src/js/27-pedals.js.

import type { PedalDef } from '../types.ts';

export const refiner = {
  kind: 'pedal',
  id: 'refiner',
  name: 'MM3 latent refiner',
  short: 'MM3 refiner',
  desc: 'Whole-mix refinement. Tuned on MiniMax-Music3; it runs on any engine, untested outside it. Its pack is separate.',
  at: 20,
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
  hue: 200,
  on: 'pp.refiner.on',
  main: 'pp.refiner.mix',
  thumbWidth: 2,
  owner: 'song',
  stored: 'settings',
  params: [
    { key: 'pp.refiner.on', type: 'boolean', label: 'On', default: false, wire: 'param:mm3RefinerEnabled' },
    {
      key: 'pp.refiner.steps',
      type: 'number',
      label: 'Steps',
      range: [1, 64, 1],
      default: 32,
      wire: 'param:mm3RefinerSteps',
    },
    {
      key: 'pp.refiner.win',
      type: 'number',
      label: 'Window',
      range: [0, 60, 1],
      unit: 's',
      default: 30,
      wire: 'param:mm3RefinerWindowSeconds',
    },
    {
      key: 'pp.refiner.ov',
      type: 'number',
      label: 'Overlap',
      range: [0, 10, 0.5],
      unit: 's',
      default: 2,
      wire: 'param:mm3RefinerOverlapSeconds',
    },
    {
      key: 'pp.refiner.mix',
      type: 'number',
      label: 'Mix',
      range: [0, 100, 1],
      unit: '%',
      default: 100,
      wire: { to: 'param:mm3RefinerMix', scale: 0.01, note: 'Real is 0..1.' },
    },
  ],
  thumb: { file: 'pd-refiner' },
} satisfies PedalDef;
