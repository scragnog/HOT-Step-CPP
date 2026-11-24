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
      placeholder: '% empty: the planner writes one\n% e.g.  K:G   G A B d | c B A G |',
      example: 'X:1\nT:Stay a little longer\nM:4/4\nL:1/4\nK:G\n"G" G A B d | "C" c B A G | "D" A B c d | "G" B2 G2 |]',
      when: { param: 'lm.cotmode', not: 'off' },
      per: [{ when: { param: 'lm.cotmode', is: 'off' }, placeholder: '% empty: the Planner is set to write nothing down\n% e.g.  K:G   G A B d | c B A G |' }],
      wire: {
        to: 'ext:yue2Abc',
        note: 'Declared by the YuE2 backend and read by backends/yue2/generate.ts as a lead sheet marker.',
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
    {
      /* with Chain of Thought off the planner writes no score and the sheet is not sent: it stands back and says so,
         and can still be written on */
      part: 'textArea',
      param: 'yue.abc',
      opts: { look: 'staff' },
      dim: { when: { param: 'lm.cotmode', is: 'off' }, says: 'Not used while the Planner is set to write nothing down' },
    },
    {
      part: 'abcFoot',
      reads: ['yue.abc', 'lm.cotmode', 'lyrics.text', 'prompt.text', 'song.title'],
      opts: { chords: 'full' },
      actions: [
        { do: 'example', param: 'yue.abc', label: 'Insert example' },
        { do: 'bespoke', name: 'scorePreview', label: 'Preview this score' },
        { do: 'set', param: 'lm.cotmode', value: 'full', label: 'Write a full score' },
      ],
    },
    {
      /* the concept's .ctl-row: the choice, then one line saying what it does and what it needs */
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
          text: 'It needs composer CFG 1, so your {value:yue.cfg} skips it.',
          opts: { look: 'warn' },
          when: {
            all: [{ param: 'yue.timing', not: 'off' }, { param: 'yue.cfg', auto: false }, { param: 'yue.cfg', not: 1 }],
          },
          actions: [{ do: 'auto', param: 'yue.cfg', label: 'Set CFG to auto' }],
        },
        {
          part: 'note',
          text: 'Needs composer CFG 1 (auto).',
          when: {
            all: [{ param: 'yue.timing', not: 'off' }, { any: [{ param: 'yue.cfg', auto: true }, { param: 'yue.cfg', is: 1 }] }],
          },
        },
      ],
      tiers: ['studio', 'lab'],
    },
  ],
  thumb: { file: 'score' },
} satisfies ModuleDef;
