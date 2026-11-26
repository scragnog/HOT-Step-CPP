// Module "Throughput". Concept: src/js/25-module-yuebatch.js (and its css).

import type { ModuleDef } from '../types.ts';

export const yuebatch = {
  kind: 'module',
  id: 'yuebatch',
  name: 'Throughput',
  desc: 'Batch queued songs, noise variations, NAR cache and chunk length.',
  at: 190,
  category: 'Engine',
  labUnder: 'takes',
  engines: ['yue2'],
  owner: 'song',
  stored: 'settings',
  size: { w: 1 },
  params: [
    { key: 'yue.coalesce', type: 'boolean', label: 'Batch queued songs', default: true, wire: 'ext:yue2Coalesce' },
    {
      key: 'yue.noisevar',
      type: 'number',
      label: 'Noise variations',
      tip: 'Noise variations: the same composed song rendered from different noise',
      range: [1, 9, 1],
      default: 1,
      wire: { to: 'ext:yue2Variations', differs: ['range'], note: 'Declared with max 1 in the static manifest; the engine raises it.' },
    },
    {
      key: 'yue.narcache',
      type: 'number',
      label: 'NAR cache ratio',
      tip: 'Render (NAR) cache ratio',
      range: [0, 0.9, 0.05],
      default: 0,
      wire: 'ext:yue2NarCacheRatio',
    },
    {
      key: 'yue.chunk',
      type: 'number',
      label: 'NAR chunk length',
      tip: 'Splits a long render; 0 renders it whole',
      range: [0, 300, 10],
      format: { template: '{v} s', words: [{ value: 0, text: 'whole song' }] },
      default: 0,
      wire: 'ext:yue2NarChunkSeconds',
    },
  ],
  face: [
    { part: 'rack', children: [{ part: 'ledButton', param: 'yue.coalesce' }] },
    {
      part: 'row',
      children: [
        { part: 'knob', param: 'yue.noisevar', label: 'Variations' },
        { part: 'knob', param: 'yue.narcache', label: 'Cache' },
        { part: 'knob', param: 'yue.chunk', label: 'Chunk' },
      ],
    },
  ],
  thumb: { file: 'yuebatch' },
} satisfies ModuleDef;
