// Pedal "Final normalizer". Concept: src/js/27-pedals.js.

import type { PedalDef } from '../types.ts';

export const norm = {
  kind: 'pedal',
  id: 'norm',
  name: 'Final normalizer',
  desc: 'Hits a LUFS target with a look-ahead limiter. Always last.',
  at: 100,
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
  hue: 352,
  on: 'pp.norm.on',
  main: 'pp.norm.ceil',
  owner: 'song',
  stored: 'settings',
  params: [
    { key: 'pp.norm.on', type: 'boolean', label: 'On', default: false, wire: 'param:lufsEnabled' },
    {
      key: 'pp.norm.target',
      type: 'choice',
      label: 'Target',
      default: 'spotify',
      options: [
        { value: 'spotify', label: 'Spotify / YouTube · -14', lcd: '-14 LUFS', detail: 'Spotify / YouTube' },
        { value: 'apple', label: 'Apple Music / Tidal · -16', lcd: '-16 LUFS', detail: 'Apple Music / Tidal' },
        { value: 'ebu', label: 'EBU R128 · -23', lcd: '-23 LUFS', detail: 'EBU R128' },
        { value: 'club', label: 'Club / DJ · -8', lcd: '-8 LUFS', detail: 'Club / DJ' },
        { value: 'custom', label: 'Custom', lcd: 'CUSTOM', detail: 'set your own' },
      ],
      wire: [
        { to: 'ui:hs-lufsPreset', note: 'Form state: the real app keeps the preset to show it and sends only the target.' },
        {
          to: 'param:lufsTarget',
          map: { spotify: -14, apple: -16, ebu: -23, club: -8 },
          note: 'Custom sends the Custom knob\'s value instead.',
        },
      ],
    },
    {
      key: 'pp.norm.custom',
      type: 'number',
      label: 'Custom',
      range: [-30, -5, 0.5],
      unit: 'LUFS',
      default: -14,
      when: { param: 'pp.norm.target', is: 'custom' },
      wire: 'param:lufsTarget',
    },
    {
      key: 'pp.norm.ceil',
      type: 'number',
      label: 'Ceiling',
      range: [-6, 0, 0.1],
      unit: 'dBFS',
      default: -1,
      wire: 'param:lufsCeilingDb',
    },
  ],
  thumb: { file: 'pd-norm' },
} satisfies PedalDef;
