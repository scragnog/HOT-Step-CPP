// Module "Build by section". Concept: src/js/25-module-build.js (and its css).

import type { ModuleDef } from '../types.ts';

export const build = {
  kind: 'module',
  id: 'build',
  name: 'Build by section',
  desc: 'Build the song one section at a time: make a few takes, hear them, keep one, move on.',
  at: 130,
  category: 'Compose',
  engines: ['yue2'],
  owner: 'song',
  stored: 'settings',
  size: { w: 4, quickW: 4, stripKeepsWidth: true },
  tierSummary: [
    'Sections, takes, Keep and Render',
    '+ section mode and takes per section',
    '+ Follow score, Plan strength and the plan\'s seed',
  ],
  params: [
    {
      key: 'yue.btakes',
      type: 'number',
      label: 'Takes per section',
      range: [1, 4, 1],
      default: 2,
      wire: {
        none: 'The real Build by section (Yue2Build) has no takes-per-section setting; takes come one at a time from the build session.',
      },
    },
    {
      key: 'yue.bmode',
      type: 'choice',
      label: 'Section mode',
      default: 'new',
      options: [{ value: 'new', label: 'New chords per take' }, { value: 'follow', label: 'Follow the score' }],
      wire: { none: 'No section mode in the real Build by section.' },
    },
    {
      key: 'yue.bcond',
      type: 'number',
      label: 'Follow score',
      tip: 'How closely each take follows the score',
      range: [0, 2, 0.05],
      default: 1,
      wire: { to: 'ui:Yue2Build.cfgScale', note: 'BuildSettings.cfgScale, "Condition strength", 0.1 to 2.' },
    },
    {
      key: 'yue.bar',
      type: 'number',
      label: 'Plan strength',
      tip: 'How strongly the plan steers each take',
      range: [0, 2, 0.05],
      default: 1,
      wire: {
        to: 'ui:Yue2Build.arStrength',
        note: 'BuildSettings.arStrength; the real default is 0.6 (COVER_AR_DEFAULT), the concept\'s 1.',
      },
    },
  ],
  state: [
    {
      key: 'build',
      kind: 'record',
      noun: 'the build',
      owner: 'song',
      stored: 'none',
      wire: {
        to: 'ui:hs-create-buildSessionId',
        note: 'The real build lives in a server session (yue2CoverApi build*); the browser keeps its id and plan.',
      },
      steps: {
        plan: 'Plan the score',
        replan: 'Re-plan the score',
        relyric: 'Keep building on the new lyrics',
        keep: 'Keep take {n} for {section}',
        unkeep: 'Stop keeping take {n} for {section}',
        discard: 'Discard take {n}',
        render: 'Render "{title}"',
        reset: 'Start a new build',
      },
    },
  ],
  strip: [
    {
      when: { param: 'lm.cotmode', is: 'off' },
      tiers: ['quick'],
      says: 'The Planner is set to write nothing down, and this needs a score',
      actions: [{ do: 'set', param: 'lm.cotmode', value: 'full', label: 'Write a full score' }],
    },
    {
      when: { param: 'lm.cotmode', is: 'off' },
      tiers: ['studio', 'lab'],
      says: 'Needs a planned score, and Chain of Thought is off',
      actions: [{ do: 'set', param: 'lm.cotmode', value: 'full', label: 'Turn it on' }],
    },
    {
      /* untagged lyrics are one implicit Verse, so the board shows; only an empty sheet has nothing to build */
      when: { param: 'lyrics.text', is: '' },
      says: 'Tag the lyrics by section to build one at a time',
      actions: [
        { do: 'tag', tag: '[Verse]', label: '+ [Verse]', tip: 'Add a [Verse] tag to the lyrics' },
        { do: 'tag', tag: '[Chorus]', label: '+ [Chorus]', tip: 'Add a [Chorus] tag to the lyrics' },
      ],
    },
  ],
  face: [
    {
      /* the board holds its settings and draws them under the section being built (the concept's .b-ctl); while a keep
         locks Follow score and Plan strength it shows them as read-outs in their place */
      part: 'buildBoard',
      state: ['build'],
      reads: ['lyrics.text', 'lm.cotmode', 'yue.btakes', 'yue.bmode', 'yue.bcond', 'yue.bar', 'prompt.text', 'song.title'],
      opts: { ownChords: 'new' },
      children: [
        { part: 'segmented', param: 'yue.bmode', tiers: ['studio', 'lab'] },
        { part: 'labelled', label: 'Takes', children: [{ part: 'token', param: 'yue.btakes' }], tiers: ['studio', 'lab'] },
        { part: 'labelled', label: 'Follow score', tip: 'How closely each take follows the score', children: [{ part: 'token', param: 'yue.bcond' }], tiers: ['lab'] },
        { part: 'labelled', label: 'Plan strength', tip: 'How strongly the plan steers each take', children: [{ part: 'token', param: 'yue.bar' }], tiers: ['lab'] },
      ],
    },
  ],
  thumb: { file: 'build' },
} satisfies ModuleDef;
