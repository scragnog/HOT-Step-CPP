// Module "Latent & correction". Concept: src/js/25-module-latent.js (and its css).

import type { ModuleDef } from '../types.ts';

export const latent = {
  kind: 'module',
  id: 'latent',
  name: 'Latent & correction',
  desc: 'Custom timesteps, latent shift and rescale, DCW, denoiser and LSS.',
  at: 110,
  category: 'Engine',
  labUnder: 'sampler',
  engines: ['ace'],
  owner: 'engine',
  stored: 'settings',
  size: { w: 2, stretch: 144 },
  params: [
    {
      key: 'lat.ts',
      type: 'text',
      label: 'Custom timesteps',
      default: '',
      placeholder: '0.97,0.76,0.5,0.28,0.085,0',
      example: '0.97, 0.76, 0.5, 0.28, 0.085, 0',
      check: {
        rule: 'timesteps',
        message: 'Timesteps are numbers from 0 to 1, separated by commas, highest first.',
        keepInvalid: 'the last valid list',
      },
      wire: 'param:customTimesteps',
    },
    {
      key: 'lat.shift',
      type: 'number',
      label: 'Latent shift',
      range: [-2, 2, 0.05],
      default: 0,
      wire: 'param:latentShift',
    },
    {
      key: 'lat.rescale',
      type: 'number',
      label: 'Latent rescale',
      range: [0.1, 3, 0.05],
      default: 1,
      wire: 'param:latentRescale',
    },
    { key: 'dcw.on', type: 'boolean', label: 'DCW correction', default: false, wire: 'param:dcwEnabled' },
    {
      key: 'dcw.mode',
      type: 'choice',
      label: 'DCW mode',
      default: 'double',
      options: [
        { value: 'double', label: 'Both (Low + High)' },
        { value: 'low', label: 'Low-Frequency' },
        { value: 'high', label: 'High-Frequency' },
        { value: 'pix', label: 'Pixel-Space (No Wavelets)' },
      ],
      wire: 'param:dcwMode',
    },
    {
      key: 'den.on',
      type: 'boolean',
      label: 'Denoiser (spectral gate)',
      default: false,
      wire: {
        none: 'No switch in the real app: the denoiser runs while denoiseStrength is above 0. Off has to send denoiseStrength 0.',
      },
    },
    {
      key: 'lss.on',
      type: 'boolean',
      label: 'LSS (latent spectral suppressor)',
      default: false,
      wire: { none: 'No switch in the real app: LSS runs while lssStrength is above 0. Off has to send lssStrength 0.' },
    },
    {
      key: 'dcw.low',
      type: 'number',
      label: 'Low',
      range: [0, 1, 0.01],
      default: 0.2,
      labelWhen: [{ when: { param: 'dcw.mode', not: 'double' }, label: 'Scaler' }],
      wire: 'param:dcwLowScaler',
    },
    {
      key: 'dcw.high',
      type: 'number',
      label: 'High',
      range: [0, 1, 0.01],
      default: 0.2,
      when: { param: 'dcw.mode', is: 'double' },
      wire: 'param:dcwHighScaler',
    },
    {
      key: 'den.str',
      type: 'number',
      label: 'Strength',
      range: [0.01, 1, 0.01],
      default: 0.5,
      wire: 'param:denoiseStrength',
    },
    {
      key: 'den.smooth',
      type: 'number',
      label: 'Smoothing',
      range: [0, 1, 0.01],
      default: 0.7,
      wire: 'param:denoiseSmoothing',
    },
    { key: 'den.mix', type: 'number', label: 'Mix', range: [0, 1, 0.01], default: 0.25, wire: 'param:denoiseMix' },
    {
      key: 'lss.str',
      type: 'number',
      label: 'Strength',
      range: [0.01, 1, 0.01],
      default: 0.65,
      wire: 'param:lssStrength',
    },
    {
      key: 'lss.var',
      type: 'number',
      label: 'Var threshold',
      range: [0.01, 0.5, 0.01],
      default: 0.15,
      wire: 'param:lssVarThresh',
    },
    { key: 'lss.dc', type: 'boolean', label: 'DC remove', default: true, wire: 'param:lssDcRemove' },
  ],
  face: [
    {
      part: 'latentCurve',
      params: ['lat.shift', 'lat.rescale'],
      children: [
        {
          part: 'xyPad',
          params: ['lat.shift', 'lat.rescale'],
          opts: { look: 'overlay', xl: 'shift', yl: 'rescale' },
        },
      ],
    },
    {
      part: 'labelled',
      label: 'Custom timesteps · highest first',
      tip: 'Highest first. They replace the schedule, and the Sampler scope draws them.',
      children: [
        { part: 'textField', param: 'lat.ts', opts: { look: 'lcd' } },
        { part: 'link', action: { do: 'example', param: 'lat.ts', label: 'Insert example' } },
      ],
    },
    {
      part: 'row',
      children: [
        {
          part: 'group',
          dim: { when: { param: 'dcw.on', is: false }, says: 'DCW is off' },
          children: [
            { part: 'pill', param: 'dcw.on', label: 'DCW' },
            { part: 'segmented', param: 'dcw.mode', opts: { size: 'sm' } },
            { part: 'row', children: [{ part: 'knob', param: 'dcw.low' }, { part: 'knob', param: 'dcw.high' }] },
          ],
        },
        {
          part: 'group',
          dim: { when: { param: 'den.on', is: false }, says: 'Denoiser is off' },
          children: [
            { part: 'pill', param: 'den.on', label: 'Denoiser' },
            {
              part: 'row',
              children: [
                { part: 'knob', param: 'den.str' },
                { part: 'knob', param: 'den.smooth', label: 'Smooth' },
                { part: 'knob', param: 'den.mix' },
              ],
            },
          ],
        },
        {
          part: 'group',
          dim: { when: { param: 'lss.on', is: false }, says: 'LSS is off' },
          children: [
            { part: 'pill', param: 'lss.on', label: 'LSS' },
            { part: 'pill', param: 'lss.dc', label: 'DC remove' },
            {
              part: 'row',
              children: [{ part: 'knob', param: 'lss.str' }, { part: 'knob', param: 'lss.var', label: 'Variance' }],
            },
          ],
        },
      ],
    },
  ],
  thumb: { file: 'latent' },
} satisfies ModuleDef;
