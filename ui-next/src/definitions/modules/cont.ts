// Module "Continue from recording". Concept: src/js/25-module-cont.js (and its css).

import type { ModuleDef } from '../types.ts';

export const cont = {
  kind: 'module',
  id: 'cont',
  name: 'Continue from recording',
  desc: 'Start a new song from up to 30 seconds of a recording. Experimental.',
  at: 140,
  category: 'Compose',
  labUnder: 'score',
  engines: ['yue2'],
  owner: 'song',
  stored: 'content',
  size: { w: 2 },
  params: [],
  state: [
    {
      key: 'contFile',
      kind: 'file',
      noun: 'the recording',
      load: 'Load {file} to continue from',
      owner: 'song',
      stored: 'content',
      wire: {
        to: 'content:yue2Continuation',
        note: 'yue2Continuation (server contracts/yue2Continuation.ts) carries the source and the part.',
      },
    },
    {
      key: 'contSel',
      kind: 'span',
      noun: 'Part to continue from',
      owner: 'song',
      stored: 'content',
      wire: { to: 'content:yue2Continuation', note: 'Inside yue2Continuation.' },
    },
  ],
  face: [
    {
      part: 'dropZone',
      opts: {
        look: 'sleeve',
        state: 'contFile',
        empty: 'Drop a recording',
        hint: 'YuE2 continues a new song from up to 30 seconds of it',
      },
    },
    { part: 'recordingSpan', when: { state: 'contFile', set: true } },
    {
      part: 'note',
      text: 'Experimental',
      tip: 'The selected part is composed again, so it will differ from the original. It works on simple material and is unreliable on real vocal recordings.',
    },
    { part: 'note', text: 'Can\'t be combined with Build by section.', when: { onDesk: 'build' } },
  ],
  thumb: { file: 'cont' },
} satisfies ModuleDef;
