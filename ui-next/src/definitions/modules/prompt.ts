// Module "Prompt". Concept: src/js/33-prompt.js and src/css/22-prompt.css.

import type { ModuleDef } from '../types.ts';

export const prompt = {
  kind: 'module',
  id: 'prompt',
  name: 'Prompt',
  desc: 'The main text box and the song info: what the engine reads first.',
  at: 0,
  category: 'Words',
  builtIn: true,
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
  owner: 'song',
  stored: 'content',
  size: { w: 4, quickW: 4 },
  per: { 'minimax-m3': { name: 'Caption' }, sa3: { name: 'Prompt' }, heartmula: { name: 'Tags' } },
  params: [
    {
      key: 'prompt.text',
      type: 'text',
      label: 'Describe the song',
      default: '',
      wire: 'content:caption',
      per: { 'minimax-m3': { label: 'Caption' }, sa3: { label: 'Prompt' }, heartmula: { label: 'Tags' } },
    },
    {
      key: 'wild.auto',
      type: 'boolean',
      label: 'Expand {a|b|c} wildcards at Generate',
      undoName: 'Expand on Generate',
      default: false,
      engines: ['ace', 'minimax-m3', 'yue2', 'sa3'],
      stored: 'settings',
      wire: 'ui:hs-main-auto-expand',
    },
    { key: 'song.artist', type: 'text', label: 'Artist', default: '', wire: 'content:artist' },
    { key: 'song.title', type: 'text', label: 'Title', default: '', wire: 'content:title' },
    { key: 'song.subject', type: 'text', label: 'Subject', default: '', wire: 'content:subject' },
    {
      key: 'cap.src',
      type: 'choice',
      label: 'Caption source',
      default: 'custom',
      options: {
        from: 'caption.sources',
        include: [{ value: 'custom', label: 'Custom' }, { value: 'auto', label: 'Automatic' }],
      },
      engines: ['minimax-m3', 'yue2'],
      wire: {
        to: 'ui:mm3CaptionSources.mode',
        note: 'The Send-to-Create handoff in localStorage; values custom, auto, track:<title>.',
      },
      per: {
        yue2: {
          label: 'Description source',
          wire: { to: 'ui:yue2CaptionSelection.mode', note: 'CreatePanel state; values custom, auto, track.' },
        },
      },
    },
    {
      key: 'cap.ds',
      type: 'choice',
      label: 'Dataset',
      default: '',
      options: { from: 'caption.datasets', include: [{ value: '', label: 'The adapter\'s dataset' }] },
      engines: ['minimax-m3', 'yue2'],
      wire: { none: 'MiniMax-Music3 takes its caption tracks from a Lyric Studio handoff, not from a dataset dropdown.' },
      per: { yue2: { wire: 'ui:hs-yue2CaptionDataset' } },
    },
  ],
  face: [
    {
      part: 'promptBox',
      param: 'prompt.text',
      params: ['song.artist', 'song.title', 'song.subject', 'cap.src', 'cap.ds', 'wild.auto'],
    },
  ],
} satisfies ModuleDef;
