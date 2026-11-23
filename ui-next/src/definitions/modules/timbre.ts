// Module "Timbre reference". Concept: src/js/25-module-timbre.js (and its css).

import type { ModuleDef } from '../types.ts';

export const timbre = {
  kind: 'module',
  id: 'timbre',
  name: 'Timbre reference',
  desc: 'A track the DiT borrows tone and texture from.',
  at: 40,
  category: 'Compose',
  engines: ['ace'],
  owner: 'song',
  stored: 'content',
  size: { w: 1, grow: true, stretch: 144 },
  tierSummary: ['Drop a track', '+ strength knob', 'Same as Studio'],
  params: [
    {
      key: 'timbre.str',
      type: 'number',
      label: 'Reference blend',
      range: [0, 1, 0.05],
      format: { percent: true },
      default: 1,
      wire: 'param:timbreReferenceStrength',
    },
  ],
  state: [
    {
      key: 'timbreFile',
      kind: 'file',
      noun: 'the timbre reference',
      load: 'Load {file} as the timbre reference',
      owner: 'song',
      stored: 'content',
      wire: 'param:timbreAudioPath',
    },
  ],
  face: [
    { part: 'recentRefs', tiers: ['studio', 'lab'] },
    {
      part: 'dropZone',
      opts: { look: 'sleeve', state: 'timbreFile', empty: 'Drop a track', hint: 'or choose a file' },
      tiers: ['quick'],
    },
    {
      part: 'dropZone',
      opts: { look: 'sleeve', state: 'timbreFile', empty: 'Drop a reference track', hint: 'or click to choose audio' },
      tiers: ['studio', 'lab'],
    },
    {
      part: 'timbreBlend',
      param: 'timbre.str',
      dim: { when: { state: 'timbreFile', set: false }, says: 'Load a track to blend it in' },
      tiers: ['studio', 'lab'],
    },
  ],
  thumb: { file: 'timbre' },
} satisfies ModuleDef;
