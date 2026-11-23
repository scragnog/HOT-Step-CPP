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
  owner: 'song',
  stored: 'settings',
  params: [
    { key: 'pp.tile.on', type: 'boolean', label: 'On', default: false, wire: 'param:postprocessEnabled' },
    {
      key: 'pp.tile.plugin',
      type: 'choice',
      label: 'Plugin',
      default: 'md_audio_tiled V2.1',
      options: {
        from: 'plugins.postprocess',
        include: [{ value: 'md_audio_tiled V2.1', label: 'md_audio_tiled V2.1', lcd: 'V2.1' }],
      },
      wire: { to: 'param:postprocessPlugin', note: 'The concept starts on V2.1, so turning the decoder on uses it; the real app starts with none chosen.' },
    },
  ],
  idle: [
    {
      when: { all: [{ engine: ['sa3'] }, { param: 'samp.plugins', is: false }] },
      says: 'Needs Lua plugins',
      actions: [
        {
          do: 'set',
          param: 'samp.plugins',
          value: true,
          label: 'Turn on',
          tip: 'Turns on the Sampler\'s Lua plugins (experimental on Stable Audio 3)',
        },
      ],
    },
  ],
  face: [
    { part: 'lcdValue', reads: ['pp.tile.plugin'], label: 'decoder', tiers: ['studio'] },
  ],
  thumb: { file: 'pd-tile' },
} satisfies PedalDef;
