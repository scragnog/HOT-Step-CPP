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
  thumbWidth: 1,
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
      kind: 'file',
      noun: 'the mastering reference',
      load: 'Load {file} as the mastering reference',
      owner: 'song',
      stored: 'content',
      wire: 'param:masteringReference',
    },
  ],
  idle: [
    {
      when: { state: 'masterRef', set: false },
      says: 'Needs a reference track',
      action: { do: 'tier', tier: 'lab', label: 'Set it in Lab' },
    },
  ],
  face: [
    { part: 'lcdValue', label: 'reference', text: 'REF', tiers: ['studio'] },
    { part: 'vuMeter', tiers: ['lab'] },
    {
      part: 'dropZone',
      opts: { look: 'slot', state: 'masterRef', empty: 'Insert a track' },
      label: 'Reference',
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
        all: [{ param: 'pp.master.timbre', is: true }, { state: 'masterRef', set: true }, { not: { onDesk: 'timbre' } }],
      },
      action: { do: 'add', item: 'timbre', label: 'Add it' },
      tiers: ['lab'],
    },
  ],
  thumb: { file: 'pd-master' },
} satisfies PedalDef;
