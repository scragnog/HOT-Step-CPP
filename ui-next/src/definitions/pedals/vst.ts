// Pedal "VST3 chain". Concept: src/js/27-pedals.js.

import type { PedalDef } from '../types.ts';

export const vst = {
  kind: 'pedal',
  id: 'vst',
  name: 'VST3 chain',
  desc: 'Your own VST3 plugins, in order, with their own editors.',
  at: 80,
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
  hue: 225,
  on: 'pp.vst.on',
  main: null,
  owner: 'song',
  stored: 'settings',
  params: [
    {
      key: 'pp.vst.on',
      type: 'boolean',
      label: 'On',
      default: false,
      wire: {
        none: 'No chain switch in the real app: the VST3 chain runs while any plugin in it is enabled (vstChainStore chainEnabled).',
      },
    },
  ],
  idle: [
    { when: { not: { state: 'vstPlugins', atLeast: 1 } }, tiers: ['lab'], says: 'No plugins to run' },
    {
      when: { not: { state: 'vstPlugins', atLeast: 1 } },
      tiers: ['quick', 'studio'],
      says: 'No plugins to run. Add them in Lab',
      actions: [{ do: 'tier', tier: 'lab', label: 'Add them in Lab' }],
    },
  ],
  face: [
    {
      part: 'lcdValue',
      label: 'plugins',
      text: '{count:vstPlugins}',
      per: [{ when: { param: 'pp.vst.on', is: true }, actions: [{ do: 'tier', tier: 'lab', label: 'add in Lab' }] }],
      tiers: ['studio'],
    },
    { part: 'labelled', label: 'Chain', children: [{ part: 'unitLcd', text: 'NO PLUGINS' }], tiers: ['lab'] },
    { part: 'button', actions: [{ do: 'bespoke', name: 'vstHost', label: '+ Add' }], tiers: ['lab'] },
  ],
  thumb: { file: 'pd-vst' },
} satisfies PedalDef;
