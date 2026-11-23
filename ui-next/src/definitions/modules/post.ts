// Module "Pedalboard". Concept: src/js/27-pedalboard.js and src/css/27-pedalboard.css.

import type { ModuleDef } from '../types.ts';

export const post = {
  kind: 'module',
  id: 'post',
  name: 'Pedalboard',
  desc: 'The post-processing chain: pedals in their chain order, and side units beside it.',
  at: 220,
  category: 'Chain',
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
  owner: 'song',
  stored: 'settings',
  size: { w: 4, quickW: 4 },
  params: [
    {
      key: 'pp.on',
      type: 'boolean',
      label: 'Chain on',
      undoName: 'Chain',
      format: { words: [{ value: false, text: 'bypassed' }, { value: true, text: 'on' }] },
      default: true,
      wire: 'param:postProcessingEnabled',
    },
  ],
  face: [
    { part: 'powerButton', param: 'pp.on', label: 'Chain' },
  ],
  thumb: { file: 'post' },
} satisfies ModuleDef;
