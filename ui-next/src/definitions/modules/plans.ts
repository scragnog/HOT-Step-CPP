// Module "Plans & endings". Concept: src/js/25-module-plans.js (and its css).

import type { ModuleDef } from '../types.ts';

export const plans = {
  kind: 'module',
  id: 'plans',
  name: 'Plans & endings',
  desc: 'Reuse, save and replay plans, and how a song is allowed to end.',
  at: 160,
  category: 'Engine',
  labUnder: 'planner',
  engines: ['minimax-m3'],
  owner: 'song',
  stored: 'settings',
  size: { w: 1 },
  params: [
    { key: 'mm.reuse', type: 'boolean', label: 'Reuse planner output', default: true, wire: 'ext:mm3ReuseAr' },
    { key: 'mm.savear', type: 'boolean', label: 'Save the next Plank', default: false, wire: 'ext:mm3SaveArCodes' },
    {
      key: 'mm.plank',
      type: 'choice',
      label: 'Replay a Plank',
      default: '',
      options: { from: 'mm3.planks', include: [{ value: '', label: 'None' }] },
      wire: 'ext:mm3PlankPath',
    },
    {
      key: 'mm.savehid',
      type: 'boolean',
      label: 'Save the next full plan',
      default: false,
      wire: 'ext:mm3SaveHiddens',
    },
    {
      key: 'mm.hidname',
      type: 'text',
      label: 'Full plan name',
      default: '',
      when: { param: 'mm.savehid', is: true },
      wire: 'ext:mm3HiddensName',
    },
    {
      key: 'mm.hidpath',
      type: 'choice',
      label: 'Replay a full plan',
      default: '',
      options: { from: 'mm3.hiddens', include: [{ value: '', label: 'None' }] },
      wire: 'ext:mm3HiddensPath',
    },
    {
      key: 'mm.reqend',
      type: 'boolean',
      label: 'Require natural ending',
      tip: 'Render only a planned candidate that reaches its own ending; if none does before the ceiling, plan again with fresh seeds',
      default: true,
      wire: 'ext:mm3RequireEnding',
    },
    {
      key: 'mm.minlen',
      type: 'boolean',
      label: 'Reject short renders',
      tip: 'Throw away a render that ends too early and plan again',
      default: true,
      wire: 'ext:mm3MinLength',
    },
    {
      key: 'mm.basehand',
      type: 'boolean',
      label: 'Base model ending',
      tip: 'Let the base planner write the ending while the adapter plans the rest',
      default: true,
      when: { state: 'adapters', atLeast: 1 },
      wire: 'ext:mm3BaseHandoff',
    },
    {
      key: 'mm.kvq8',
      type: 'boolean',
      label: 'Compact planner cache',
      tip: 'Keep the planner cache at 8-bit to save VRAM while Plan is below 1',
      default: true,
      when: { all: [{ state: 'adapters', atLeast: 1 }, { param: 'ad.plan', below: 1 }] },
      wire: 'ext:mm3KvQ8',
    },
  ],
  face: [
    {
      part: 'deck',
      children: [
        {
          part: 'labelled',
          label: 'Last plan',
          children: [
            {
              part: 'ledButton',
              param: 'mm.reuse',
              label: 'Reuse',
              tip: 'Reuse the planner output: with a fixed seed, changing only flow settings reuses the last plan',
            },
            { part: 'ledButton', param: 'mm.kvq8' },
          ],
        },
        {
          part: 'labelled',
          label: 'Plank',
          tip: 'The planner\'s token plan for a song (its AR plan)',
          children: [
            {
              part: 'recButton',
              param: 'mm.savear',
              label: '● SAVE NEXT',
              tip: 'Save the next plan as a Plank: the planner token plan',
              dim: {
                when: { param: 'mm.plank', not: '' },
                says: 'Replaying a saved plan, so there is nothing new to save',
              },
            },
          ],
        },
        {
          part: 'labelled',
          label: 'Replay',
          children: [{ part: 'pick', param: 'mm.plank', opts: { variant: 'lcd' } }],
        },
        {
          part: 'labelled',
          label: 'Full plan',
          children: [
            {
              part: 'recButton',
              param: 'mm.savehid',
              label: '● SAVE NEXT',
              tip: 'Save the next plan to disk, with its hidden states',
              dim: {
                when: { all: [{ param: 'mm.hidpath', not: '' }, { param: 'mm.plank', is: '' }] },
                says: 'Replaying a saved plan, so there is nothing new to save',
              },
            },
          ],
        },
        {
          part: 'labelled',
          label: 'Name',
          children: [{ part: 'textField', param: 'mm.hidname', opts: { look: 'lcd' } }],
        },
        {
          part: 'labelled',
          label: 'Replay',
          children: [
            {
              part: 'pick',
              param: 'mm.hidpath',
              opts: { variant: 'lcd' },
              dim: {
                when: { all: [{ param: 'mm.plank', not: '' }, { param: 'mm.hidpath', not: '' }] },
                says: 'Not used: the Plank replay above wins',
              },
            },
          ],
        },
      ],
    },
    {
      part: 'labelled',
      label: 'Endings',
      children: [
        {
          part: 'rack',
          children: [
            { part: 'ledButton', param: 'mm.reqend' },
            { part: 'ledButton', param: 'mm.minlen' },
            { part: 'ledButton', param: 'mm.basehand' },
          ],
        },
      ],
    },
  ],
  thumb: { file: 'plans' },
} satisfies ModuleDef;
