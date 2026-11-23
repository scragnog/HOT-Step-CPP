// Pedal "PP-VAE". Concept: src/js/27-pedals.js.

import type { PedalDef } from '../types.ts';

export const ppvae = {
  kind: 'pedal',
  id: 'ppvae',
  name: 'PP-VAE',
  desc: 'A second autoencoder pass that cleans up fizz.',
  at: 40,
  engines: ['ace'],
  hue: 175,
  on: 'pp.ppvae.on',
  main: 'pp.ppvae.blend',
  thumbWidth: 1,
  owner: 'song',
  stored: 'settings',
  params: [
    { key: 'pp.ppvae.on', type: 'boolean', label: 'On', default: false, wire: 'param:ppVaeReencode' },
    {
      key: 'pp.ppvae.blend',
      type: 'number',
      label: 'Blend',
      range: [0, 1, 0.05],
      default: 0,
      wire: 'param:ppVaeBlend',
    },
  ],
  thumb: { file: 'pd-ppvae' },
} satisfies PedalDef;
