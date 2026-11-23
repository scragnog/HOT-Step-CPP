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
    {
      key: 'lyrics.text',
      type: 'text',
      label: 'Lyrics',
      undoName: 'the lyrics',
      placeholder: 'Or leave this empty and the planner writes them.',
      default: '',
      engines: ['ace', 'minimax-m3', 'yue2', 'heartmula'],
      wire: 'content:lyrics',
      per: [{ engines: ['heartmula'], placeholder: 'Leave this empty for an instrumental.' }],
    },
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
      per: [
        {
          engines: ['minimax-m3'],
          label: 'Lyrics language',
          tip: 'MiniMax-Music3 has no language input; it follows the characters of the lyrics. Compose caption writes the language into Vocal Details.',
          wire: {
            to: 'content:vocalLanguage',
            note: 'createContentParams sends it on every engine; only ACE-Step reads it. Compose caption writes it into Vocal Details on MiniMax-Music3.',
          },
        },
      ],
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
      stored: 'settings',
      wire: {
        to: 'ui:hs-vocalGender',
        note: 'Not sent: it reaches the model only through Compose caption writing it into Vocal Details (createContent.ts).',
      },
    },
  ],
  face: [
    {
      /* the sheet holds its foot (Instrumental, the vocal gender), where it puts its line count below Lab
         (25-module-lyrics.js lyricsInner); it reads Instrumental, the BPM field and the box its estimate falls back on */
      part: 'lyricSheet',
      param: 'lyrics.text',
      params: ['lyrics.lang'],
      reads: ['lyrics.instr', 'tempo.bpm', 'prompt.text'],
      per: [{ tiers: ['lab'], params: ['lyrics.lang', 'wild.auto'] }],
      actions: [
        { do: 'bespoke', name: 'insertSectionTag', label: 'Insert a section tag' },
        { do: 'bespoke', name: 'expandWildcards', label: 'Expand {a|b} now', tip: 'Replace each {a|b|c} with one of its choices' },
      ],
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
