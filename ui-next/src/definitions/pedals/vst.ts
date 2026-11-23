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
  thumbWidth: 1,
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
  idle: [{ when: { not: { state: 'vstPlugins', atLeast: 1 } }, says: 'No plugins to run' }],
  face: [
    { part: 'lcdValue', label: 'plugins', text: '0', tiers: ['studio'] },
    {
      part: 'labelled',
      label: 'Chain',
      children: [{ part: 'unitLcd', opts: { lines: ['"NO PLUGINS"'] } }],
      tiers: ['lab'],
    },
    { part: 'button', action: { do: 'bespoke', name: 'vstHost', label: '+ Add' }, tiers: ['lab'] },
  ],
  thumb: { file: 'pd-vst' },
} satisfies PedalDef;
