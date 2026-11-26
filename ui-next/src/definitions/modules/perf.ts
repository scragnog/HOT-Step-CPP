// Module "Performance trades". Concept: src/js/25-module-perf.js (and its css).

import type { ModuleDef } from '../types.ts';

export const perf = {
  kind: 'module',
  id: 'perf',
  name: 'Performance trades',
  desc: 'Trade quality for speed: CFG cutoff, LM CFG cutoff, step cache.',
  at: 100,
  category: 'Engine',
  labUnder: 'sampler',
  engines: ['ace'],
  owner: 'engine',
  stored: 'settings',
  size: { w: 1, stretch: 144 },
  params: [
    {
      key: 'perf.cfgcut',
      type: 'number',
      label: 'CFG cutoff',
      range: [0, 1, 0.05],
      default: 1,
      wire: 'param:cfgCutoffRatio',
    },
    {
      key: 'perf.lmcut',
      type: 'number',
      label: 'LM CFG cutoff',
      range: [0.3, 1, 0.05],
      default: 1,
      wire: 'param:lmCfgCutoffRatio',
    },
    {
      key: 'perf.cache',
      type: 'number',
      label: 'Step cache',
      range: [0, 0.7, 0.05],
      default: 0,
      wire: 'param:cacheRatio',
    },
  ],
  face: [
    /* the pad, and beside it the gauge over the LM cut knob (25-module-perf.js .padrow, .padside) */
    {
      part: 'row',
      children: [
        {
          part: 'xyPad',
          params: ['perf.cfgcut', 'perf.cache'],
          opts: { look: 'heat', xl: 'CFG cutoff', yl: 'step cache', corners: ['fastest', '', '', 'most faithful'] },
        },
        {
          part: 'row',
          children: [
            { part: 'speedGauge', reads: ['perf.cfgcut', 'perf.cache', 'perf.lmcut'] },
            { part: 'knob', param: 'perf.lmcut', label: 'LM cut' },
          ],
        },
      ],
    },
  ],
  thumb: { file: 'perf' },
} satisfies ModuleDef;
