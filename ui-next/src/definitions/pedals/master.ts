// Pedal "Mastering". Concept: src/js/27-pedals.js.

import type { PedalDef } from '../types.ts';

export const master = {
  kind: 'pedal',
  id: 'master',
  name: 'Mastering',
  desc: 'Matches level, EQ and dynamics to a reference track.',
  at: 90,
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
  hue: 320,
  on: 'pp.master.on',
  main: null,
  owner: 'song',
  stored: 'settings',
  params: [
    { key: 'pp.master.on', type: 'boolean', label: 'On', default: false, wire: 'param:masteringEnabled' },
    {
      key: 'pp.master.timbre',
      type: 'boolean',
      label: 'Use as timbre ref',
      default: false,
      engines: ['ace'],
      wire: 'param:timbreReference',
    },
  ],
  state: [
    {
      key: 'masterRef',
      role: 'reference',
      kind: 'file',
      noun: 'the mastering reference',
      title: 'Mastering reference',
      tells: ['clear'],
      steps: { load: 'Load {file} as the mastering reference' },
      owner: 'song',
      stored: 'none',
      wire: 'param:masteringReference',
    },
  ],
  idle: [
    { when: { state: 'masterRef', set: false }, tiers: ['lab'], says: 'Needs a reference track' },
    {
      when: { state: 'masterRef', set: false },
      tiers: ['quick', 'studio'],
      says: 'Needs a reference track. Set it in Lab',
      actions: [{ do: 'tier', tier: 'lab', ring: ['masterRef'], label: 'Set it in Lab' }],
    },
  ],
  face: [
    {
      part: 'lcdValue',
      label: 'reference',
      text: 'REF',
      per: [
        {
          when: { state: 'masterRef', set: false },
          label: 'no reference',
          text: '—',
        },
        {
          when: { all: [{ state: 'masterRef', set: false }, { param: 'pp.master.on', is: true }] },
          actions: [{ do: 'tier', tier: 'lab', ring: ['masterRef'], label: 'set in Lab' }],
        },
      ],
      tiers: ['studio'],
    },
    { part: 'meter', opts: { min: -20, max: 3, red: 0 }, tiers: ['lab'] },
    {
      part: 'dropZone',
      state: ['masterRef'],
      label: 'Reference',
      opts: { look: 'slot', empty: 'Insert a track' },
      tiers: ['lab'],
    },
    {
      part: 'note',
      text: 'Lends its reference to Timbre once one is in',
      when: { all: [{ param: 'pp.master.timbre', is: true }, { state: 'masterRef', set: false }] },
      tiers: ['lab'],
    },
    {
      part: 'note',
      text: 'Timbre reference uses this track',
      when: { all: [{ param: 'pp.master.timbre', is: true }, { state: 'masterRef', set: true }, { onDesk: 'timbre' }] },
      tiers: ['lab'],
    },
    {
      part: 'note',
      text: 'Timbre reference is not on the desk',
      when: {
        all: [
          { param: 'pp.master.timbre', is: true },
          { state: 'masterRef', set: true },
          { not: { onDesk: 'timbre' } },
        ],
      },
      actions: [{ do: 'add', item: 'timbre', label: 'Add it' }],
      tiers: ['lab'],
    },
  ],
  thumb: { file: 'pd-master' },
} satisfies PedalDef;
