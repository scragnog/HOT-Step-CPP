// Pedal "Tiled decoder". Concept: src/js/27-pedals.js.

import type { PedalDef } from '../types.ts';

export const tile = {
  kind: 'pedal',
  id: 'tile',
  name: 'Tiled decoder',
  desc: 'Replaces the VAE decode with a tiled decode plugin: overlap crossfades, an optional dual pass and a small DSP chain. Stable Audio 3 needs Lua plugins on.',
  at: 10,
  engines: ['ace', 'sa3'],
  hue: 160,
  on: 'pp.tile.on',
  main: null,
  thumbWidth: 1,
  owner: 'song',
  stored: 'settings',
  params: [
    { key: 'pp.tile.on', type: 'boolean', label: 'On', default: false, wire: 'param:postprocessEnabled' },
    {
      key: 'pp.tile.plugin',
      type: 'choice',
      label: 'Plugin',
      default: 'md_audio_tiled V2.1',
      options: { from: 'plugins.postprocess' },
      wire: 'param:postprocessPlugin',
    },
  ],
  idle: [
    {
      when: { all: [{ engine: ['sa3'] }, { param: 'samp.plugins', is: false }] },
      says: 'Needs Lua plugins',
      action: {
        do: 'set',
        param: 'samp.plugins',
        value: true,
        label: 'Turn on',
        tip: 'Turns on the Sampler\'s Lua plugins (experimental on Stable Audio 3)',
      },
    },
  ],
  face: [{ part: 'lcdValue', param: 'pp.tile.plugin', label: 'decoder', tiers: ['studio'] }],
  thumb: { file: 'pd-tile' },
} satisfies PedalDef;
