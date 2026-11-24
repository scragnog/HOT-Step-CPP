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
  params: [

  ],
  state: [
    {
      key: 'contFile',
      role: 'recording',
      kind: 'file',
      noun: 'the recording',
      steps: { load: 'Load {file} to continue from' },
      tells: ['load'],
      owner: 'song',
      stored: 'none',
      wire: { to: 'content:yue2Continuation', via: 'continuation' },
    },
    {
      key: 'contSel',
      role: 'recordingPart',
      kind: 'span',
      noun: 'Part to continue from',
      /* a recording's part is its first 20 seconds until it is moved */
      empty: '0:00–0:20',
      owner: 'song',
      stored: 'none',
      wire: { to: 'content:yue2Continuation', via: 'continuation' },
      steps: { move: '{noun} {from} → {to}' },
    },
  ],
  face: [
    {
      part: 'dropZone',
      state: ['contFile'],
      opts: {
        look: 'sleeve',
        empty: 'Drop a recording',
        hint: 'YuE2 continues a new song from up to 30 seconds of it',
      },
    },
    /* it stands while there is no recording too, drawing nothing, so that one chosen brings focus to it (E-23) */
    { part: 'recordingSpan', state: ['contSel'], reads: ['contFile'] },
    {
      part: 'note',
      tip: 'The selected part is composed again, so it will differ from the original. It works on simple material and is unreliable on real vocal recordings.',
      text: 'Experimental',
      opts: { look: 'badge' },
      per: [{ when: { onDesk: 'build' }, opts: { look: 'badge', after: 'Can’t be combined with Build by section.' } }],
    },
  ],
  thumb: { file: 'cont' },
} satisfies ModuleDef;
