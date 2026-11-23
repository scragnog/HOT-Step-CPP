// Module "Lyrics". Concept: src/js/25-module-lyrics.js (and its css).

import type { ModuleDef } from '../types.ts';

export const lyrics = {
  kind: 'module',
  id: 'lyrics',
  name: 'Lyrics',
  desc: 'The lyric sheet: section tags, the structure strip, the language and Instrumental.',
  at: 1,
  category: 'Words',
  builtIn: true,
  engines: ['ace', 'minimax-m3', 'yue2', 'heartmula'],
  owner: 'song',
  stored: 'content',
  size: { w: 2, quickW: 4, stretch: 0 },
  params: [
    { key: 'lyrics.text', type: 'text', label: 'Lyrics', default: '', wire: 'content:lyrics' },
    { key: 'lyrics.instr', type: 'boolean', label: 'Instrumental', default: false, wire: 'content:instrumental' },
    {
      key: 'lyrics.lang',
      type: 'choice',
      label: 'Vocal language',
      tip: 'The language the vocals are sung in.',
      default: 'en',
      options: { from: 'languages' },
      engines: ['ace', 'minimax-m3'],
      wire: 'content:vocalLanguage',
      per: {
        'minimax-m3': {
          label: 'Lyrics language',
          tip: 'MiniMax-Music3 has no language input; it follows the characters of the lyrics. Compose caption writes the language into Vocal Details.',
          wire: {
            to: 'ui:hs-vocalLanguage',
            note: 'Sent on every engine and read only by ACE-Step; Compose caption writes it into Vocal Details on MiniMax-Music3.',
          },
        },
      },
    },
    {
      key: 'lyrics.gender',
      type: 'choice',
      label: 'Vocal gender',
      tip: 'Compose caption writes this into Vocal Details',
      default: '',
      options: [
        { value: '', label: 'Any' },
        { value: 'female', label: 'Female' },
        { value: 'male', label: 'Male' },
        { value: 'duet', label: 'Duet' },
      ],
      engines: ['minimax-m3'],
      wire: {
        to: 'ui:hs-vocalGender',
        note: 'Deliberately not sent: it reaches the model only through Compose caption writing it into Vocal Details (createContent.ts). Real values are "", male, female, duet.',
      },
    },
  ],
  face: [
    { part: 'lyricSheet', param: 'lyrics.text', params: ['lyrics.lang', 'lyrics.instr', 'wild.auto'] },
    {
      part: 'foot',
      children: [
        { part: 'pill', param: 'lyrics.instr', label: 'Instrumental' },
        {
          part: 'labelled',
          label: 'Vocal',
          engines: ['minimax-m3'],
          children: [{ part: 'segmented', param: 'lyrics.gender' }],
          tiers: ['studio', 'lab'],
        },
      ],
    },
  ],
} satisfies ModuleDef;
