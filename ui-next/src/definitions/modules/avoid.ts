// Module "Avoid". Concept: src/js/25-module-avoid.js (and its css).

import type { ModuleDef } from '../types.ts';

export const avoid = {
  kind: 'module',
  id: 'avoid',
  name: 'Avoid',
  desc: 'Negative prompt: sounds to steer away from.',
  at: 30,
  category: 'Compose',
  engines: ['ace'],
  owner: 'song',
  stored: 'content',
  size: { w: 1, stretch: 144 },
  tierSummary: ['A sticky note', 'The same note', 'The same note'],
  params: [
    {
      key: 'avoid.text',
      type: 'text',
      label: 'Avoid',
      default: '',
      placeholder: 'jazz, piano, slow...',
      wire: { to: 'content:negativePrompt', note: 'Sent as negative_prompt, only when not empty.' },
    },
  ],
  face: [{ part: 'textArea', param: 'avoid.text', opts: { look: 'note' } }],
  thumb: { file: 'avoid' },
} satisfies ModuleDef;
