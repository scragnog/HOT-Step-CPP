// Module "Flow renderer". Concept: src/js/25-module-flow.js (and its css).

import type { ModuleDef } from '../types.ts';

export const flow = {
  kind: 'module',
  id: 'flow',
  name: 'Flow renderer',
  desc: 'Window hop, CFG delta cache, the renderer and its precision, and play while rendering.',
  at: 150,
  category: 'Engine',
  labUnder: 'sampler',
  engines: ['minimax-m3'],
  owner: 'engine',
  stored: 'settings',
  size: { w: 2 },
  params: [
    {
      key: 'mm.hop',
      type: 'number',
      label: 'Window hop',
      range: [100, 160, 10],
      format: { template: '{v} fr ({s})', perSecond: 25 },
      default: 150,
      wire: { to: 'ext:mm3FlowHop', note: 'The concept stepped by 5; the server declares 10, which this follows.' },
    },
    {
      key: 'mm.dcache',
      type: 'number',
      label: 'CFG delta cache',
      range: [1, 4, 1],
      format: { template: 'every {v}', words: [{ value: 1, text: 'off' }] },
      default: 1,
      wire: 'ext:mm3FlowUncondInterval',
    },
    { key: 'mm.lowstep', type: 'boolean', label: 'Low-step compensation', default: true, wire: 'ext:mm3AutoLowStep' },
    { key: 'mm.fused', type: 'boolean', label: 'Fused depth decoder', default: true, wire: 'ext:mm3DepthFused' },
    {
      key: 'mm.rend',
      type: 'choice',
      label: 'Renderer',
      default: 'auto',
      options: [
        { value: 'auto', label: 'Auto', tip: 'TensorRT when this machine has the MM3 TensorRT pack, otherwise GGML' },
        { value: 'ggml', label: 'GGML' },
        { value: 'tensorrt', label: 'TensorRT' },
      ],
      wire: {
        to: 'ext:mm3DitBackend',
        note: 'Read by backends/minimax/generate.ts but no longer declared as an extension (its control was removed).',
      },
    },
    {
      key: 'mm.prec',
      type: 'choice',
      label: 'TensorRT precision',
      default: 'fp8',
      options: [{ value: 'fp8', label: 'FP8' }, { value: 'bf16', label: 'BF16' }],
      when: { param: 'mm.rend', not: 'ggml' },
      wire: { to: 'ext:mm3DitPrecision', note: 'Read by backends/minimax/generate.ts; not declared as an extension.' },
    },
    { key: 'mm.stream', type: 'boolean', label: 'Play while rendering', default: false, wire: 'ext:mm3Stream' },
  ],
  face: [
    { part: 'flowWindows', reads: ['mm.hop'] },
    {
      /* the knobs and the switches share one row (D-11) */
      part: 'row',
      children: [
        {
          part: 'row',
          children: [
            { part: 'knob', param: 'mm.hop', label: 'Hop' },
            { part: 'knob', param: 'mm.dcache', label: 'Δ cache' },
          ],
        },
        {
          part: 'rack',
          children: [
            { part: 'ledButton', param: 'mm.lowstep' },
            { part: 'ledButton', param: 'mm.fused' },
            { part: 'ledButton', param: 'mm.stream' },
          ],
        },
      ],
    },
    {
      part: 'foot',
      children: [
        { part: 'labelled', label: 'Renderer', children: [{ part: 'segmented', param: 'mm.rend' }] },
        { part: 'labelled', label: 'Precision', children: [{ part: 'segmented', param: 'mm.prec' }] },
        {
          part: 'note',
          text: 'Needs the MM3 TensorRT pack. The first render at each precision builds an engine, which takes a few minutes. FP8 is about 27% faster.',
          when: { param: 'mm.rend', is: 'tensorrt' },
        },
      ],
    },
  ],
  thumb: { file: 'flow' },
} satisfies ModuleDef;
