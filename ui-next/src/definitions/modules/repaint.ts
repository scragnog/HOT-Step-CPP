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
      wire: { to: 'ui:repaint.mode', note: 'A field of the Repaint studio (studioWorkflows), not of the Create form.' },
    },
    {
      key: 'rp.xfade',
      type: 'number',
      label: 'Boundary crossfade',
      range: [0, 30, 1],
      format: { template: '{v} fr ({s})', perSecond: 25 },
      default: 10,
      wire: {
        to: 'ui:repaint.crossfadeFrames',
        note: 'A field of the Repaint studio (studioWorkflows), not of the Create form.',
      },
    },
    {
      key: 'rp.lyrics',
      type: 'text',
      label: 'Lyrics',
      undoName: 'the repaint lyrics',
      default: '',
      wire: { to: 'ui:repaint.lyrics', note: 'The Repaint studio\'s own lyrics field (studioWorkflows).' },
    },
  ],
  state: [
    {
      key: 'source',
      kind: 'record',
      noun: 'the song to repaint',
      steps: { pick: 'Repaint {title}' },
      owner: 'song',
      stored: 'none',
      wire: { to: 'ui:repaint.source', note: 'The Repaint studio\'s source song fields (sourceSong, sourceAudioUrl).' },
    },
    {
      key: 'region',
      kind: 'span',
      noun: 'Repaint region',
      steps: { move: '{noun} {from} → {to}' },
      owner: 'song',
      stored: 'none',
      wire: {
        to: 'ui:repaint.region',
        note: 'regionStart and regionEnd, in seconds in the real app; a fraction of the song in the concept.',
      },
    },
  ],
  face: [
    {
      part: 'repaintRegion',
      state: ['source', 'region'],
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
      tip: 'Lyrics for the repaint. They replace the whole track\'s lyrics.',
      opts: { look: 'paper' },
      tiers: ['lab'],
    },
    { part: 'note', text: 'An empty prompt reuses the song\'s style' },
  ],
} satisfies ModuleDef;
