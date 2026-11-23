// Module "Lead sheet". Concept: src/js/25-module-score.js (and its css).

import type { ModuleDef } from '../types.ts';

export const score = {
  kind: 'module',
  id: 'score',
  name: 'Lead sheet',
  desc: 'Paste an ABC lead sheet or preview the planned one, and set lyric timing.',
  at: 120,
  category: 'Compose',
  engines: ['yue2'],
  owner: 'song',
  stored: 'content',
  size: { w: 2, quickW: 2 },
  params: [
    {
      key: 'yue.abc',
      type: 'text',
      label: 'Lead sheet (ABC)',
      default: '',
      example: 'X:1\nT:Stay a little longer\nM:4/4\nL:1/4\nK:G\n"G" G A B d | "C" c B A G | "D" A B c d | "G" B2 G2 |]',
      when: { param: 'lm.cotmode', not: 'off' },
      wire: {
        to: 'ext:yue2Abc',
        note: 'Read by backends/yue2/generate.ts as a lead sheet marker; not declared as an extension.',
      },
    },
    {
      key: 'yue.timing',
      type: 'choice',
      label: 'Lyric timing',
      default: 'off',
      options: [{ value: 'off', label: 'Off' }, { value: 'bias', label: 'Soft' }, { value: 'mask', label: 'Hard' }],
      stored: 'settings',
      wire: 'ext:yue2LyricSchedule',
    },
  ],
  face: [
    { part: 'textArea', param: 'yue.abc', opts: { look: 'staff' } },
    {
      part: 'abcFoot',
      reads: ['yue.abc', 'lm.cotmode'],
      actions: [
        { do: 'example', param: 'yue.abc', label: 'Insert example' },
        { do: 'bespoke', name: 'scorePreview', label: 'Preview this score' },
        { do: 'set', param: 'lm.cotmode', value: 'full', label: 'Write a full score' },
      ],
    },
    {
      part: 'labelled',
      label: 'Lyric timing',
      children: [
        { part: 'segmented', param: 'yue.timing' },
        {
          part: 'note',
          text: 'Nudges each lyric block toward its section\'s first vocal note.',
          when: { param: 'yue.timing', is: 'bias' },
        },
        {
          part: 'note',
          text: 'Hides a section\'s lyrics until its first vocal note.',
          when: { param: 'yue.timing', is: 'mask' },
        },
        {
          part: 'note',
          text: 'It needs composer CFG 1, so your CFG skips it.',
          opts: { look: 'warn' },
          when: {
            all: [{ param: 'yue.timing', not: 'off' }, { param: 'yue.cfg', auto: false }, { param: 'yue.cfg', not: 1 }],
          },
          actions: [{ do: 'auto', param: 'yue.cfg', label: 'Set CFG to auto' }],
        },
      ],
      tiers: ['studio', 'lab'],
    },
  ],
  thumb: { file: 'score' },
} satisfies ModuleDef;
