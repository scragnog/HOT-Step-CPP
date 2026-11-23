// Module "Repaint". Concept: src/js/25-module-repaint.js (and its css).

import type { ModuleDef } from '../types.ts';

export const repaint = {
  kind: 'module',
  id: 'repaint',
  name: 'Repaint',
  desc: 'Regenerate a region of a song, chosen on its waveform.',
  at: 10,
  category: 'Compose',
  source: true,
  engines: ['ace'],
  owner: 'song',
  stored: 'none',
  size: { w: 4, quickW: 4 },
  params: [
    {
      key: 'rp.mode',
      type: 'choice',
      label: 'Repaint mode',
      default: 'balanced',
      options: [
        { value: 'conservative', label: 'Conservative' },
        { value: 'balanced', label: 'Balanced' },
        { value: 'aggressive', label: 'Aggressive' },
      ],
      wire: 'content:repaintMode',
    },
    {
      key: 'rp.xfade',
      type: 'number',
      label: 'Boundary crossfade',
      range: [0, 30, 1],
      format: { template: '{v} fr ({s})', perSecond: 25 },
      default: 10,
      wire: 'content:crossfadeFrames',
    },
    {
      key: 'rp.lyrics',
      type: 'text',
      label: 'Lyrics',
      undoName: 'the repaint lyrics',
      default: '',
      wire: { to: 'content:lyrics', note: 'The Repaint studio\'s own lyrics field (studioWorkflows repaint fields).' },
    },
  ],
  state: [
    {
      key: 'source',
      kind: 'record',
      noun: 'the song to repaint',
      owner: 'song',
      stored: 'none',
      wire: { to: 'content:sourceSong', note: 'The Repaint studio\'s source song fields (sourceSong, sourceAudioUrl).' },
    },
    {
      key: 'region',
      kind: 'span',
      noun: 'Repaint region',
      owner: 'song',
      stored: 'none',
      wire: {
        to: 'content:regionStart',
        note: 'regionStart and regionEnd, in seconds in the real app; a fraction of the song in the concept.',
      },
    },
  ],
  face: [
    {
      part: 'repaintRegion',
      children: [
        { part: 'segmented', param: 'rp.mode', tiers: ['lab'] },
        { part: 'knob', param: 'rp.xfade', label: 'Crossfade', opts: { small: true }, tiers: ['lab'] },
      ],
    },
    {
      part: 'row',
      children: [{ part: 'segmented', param: 'rp.mode' }, { part: 'knob', param: 'rp.xfade', label: 'Crossfade' }],
      tiers: ['studio'],
    },
    {
      part: 'textArea',
      param: 'rp.lyrics',
      opts: { look: 'paper' },
      tip: 'Lyrics for the repaint. They replace the whole track\'s lyrics.',
      tiers: ['lab'],
    },
    { part: 'note', text: 'An empty prompt reuses the song\'s style' },
  ],
} satisfies ModuleDef;
