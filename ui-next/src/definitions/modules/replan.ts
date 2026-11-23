// Module "Re-planning". Concept: src/js/25-module-replan.js (and its css).

import type { ModuleDef } from '../types.ts';

export const replan = {
  kind: 'module',
  id: 'replan',
  name: 'Re-planning',
  desc: 'How often YuE2 redraws a bad score, or recomposes a song that runs away.',
  at: 170,
  category: 'Engine',
  labUnder: 'planner',
  engines: ['yue2'],
  owner: 'engine',
  stored: 'settings',
  size: { w: 1 },
  params: [
    { key: 'yue.replan', type: 'boolean', label: 'Re-plan runaway scores', default: true, wire: 'ext:yue2AutoReplan' },
    {
      key: 'yue.replanflag',
      type: 'boolean',
      label: 'Re-plan flagged scores',
      default: true,
      wire: 'ext:yue2ReplanFlags',
    },
    {
      key: 'yue.tries',
      type: 'number',
      label: 'Re-plan tries',
      range: [1, 10, 1],
      default: 3,
      wire: 'ext:yue2ReplanAttempts',
    },
    {
      key: 'yue.recompose',
      type: 'number',
      label: 'Recompose runaway songs',
      tip: 'Fresh compositions if a good plan still runs away',
      range: [0, 10, 1],
      default: 2,
      wire: 'ext:yue2ComposeRetries',
    },
  ],
  face: [
    {
      part: 'labelled',
      label: 'Plan attempts',
      children: [{ part: 'dice', reads: ['yue.tries'], opts: { offWhen: 'yue.replan' } }],
    },
    {
      part: 'rack',
      children: [{ part: 'ledButton', param: 'yue.replan' }, { part: 'ledButton', param: 'yue.replanflag' }],
    },
    {
      part: 'row',
      children: [
        { part: 'knob', param: 'yue.tries', label: 'Tries' },
        { part: 'knob', param: 'yue.recompose', label: 'Recompose' },
      ],
    },
  ],
  thumb: { file: 'replan' },
} satisfies ModuleDef;
