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
  stored: 'settings',
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
      role: 'reference',
      kind: 'file',
      noun: 'the timbre reference',
      title: 'Timbre reference',
      tells: ['clear'],
      steps: { load: 'Load {file} as the timbre reference' },
      owner: 'song',
      stored: 'none',
      wire: 'param:timbreAudioPath',
    },
    {
      key: 'recentRefs',
      kind: 'list',
      noun: 'the recent references',
      owner: 'song',
      stored: 'none',
      wire: { none: 'The page\'s own list of recent reference files.' },
    },
  ],
  face: [
    /* "N more…" opens the rest of them (the part draws its words: how many more) */
    { part: 'recentRefs', state: ['recentRefs', 'timbreFile'], tiers: ['studio', 'lab'], actions: [{ do: 'bespoke', name: 'moreRefs', label: 'More recent references' }] },
    {
      part: 'dropZone',
      state: ['timbreFile'],
      opts: { look: 'sleeve', empty: 'Drop a reference track', hint: 'or click to choose audio', wave: true },
      per: [
        { tiers: ['quick'], opts: { look: 'sleeve', empty: 'Drop a track', hint: 'or choose a file', wave: true } },
        {
          when: {
            all: [{ param: 'pp.master.timbre', is: true }, { state: 'masterRef', set: true }, { onDesk: 'master' }],
          },
          state: ['masterRef'],
          opts: { look: 'sleeve', from: 'from Mastering', wave: true },
        },
      ],
    },
    {
      part: 'timbreBlend',
      param: 'timbre.str',
      dim: {
        when: {
          all: [
            { state: 'timbreFile', set: false },
            {
              not: {
                all: [{ param: 'pp.master.timbre', is: true }, { state: 'masterRef', set: true }, { onDesk: 'master' }],
              },
            },
          ],
        },
        says: 'Load a track to blend it in',
      },
      tiers: ['studio', 'lab'],
    },
  ],
  thumb: { file: 'timbre' },
} satisfies ModuleDef;
