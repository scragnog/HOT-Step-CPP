// Side unit "Whisper lyrics". Concept: src/js/28-side-units.js.

import type { UnitDef } from '../types.ts';

export const wh = {
  kind: 'unit',
  id: 'wh',
  name: 'Whisper lyrics',
  desc: 'Word-level lyric transcription, guided by your lyrics.',
  at: 30,
  engines: ['ace', 'minimax-m3', 'yue2'],
  on: 'pp.wh.on',
  wide: true,
  owner: 'song',
  stored: 'settings',
  params: [
    {
      key: 'pp.wh.on',
      type: 'boolean',
      label: 'Whisper lyrics',
      default: false,
      wire: 'param:whisperLyricsEnabled',
    },
    {
      key: 'pp.wh.model',
      type: 'choice',
      label: 'Model',
      default: '',
      options: [
        { value: '', label: 'Auto-detect', lcd: 'AUTO', detail: 'picks the installed one' },
        { value: 'large-v3-turbo', label: 'Large v3 Turbo', detail: '▮▮▮▯' },
        { value: 'large-v3', label: 'Large v3', detail: '▮▮▮▮' },
        { value: 'medium', label: 'Medium', detail: '▮▮▯▯' },
        { value: 'base', label: 'Base', detail: '▮▯▯▯' },
      ],
      wire: 'param:whisperModel',
    },
    {
      key: 'pp.wh.beam',
      type: 'number',
      label: 'Beam size',
      short: 'Beam',
      range: [1, 10, 1],
      default: 5,
      wire: 'param:whisperBeamSize',
    },
    {
      key: 'pp.wh.iso',
      type: 'boolean',
      label: 'Isolate vocals first',
      default: false,
      wire: 'param:whisperIsolateVocals',
    },
    {
      key: 'pp.wh.lang',
      type: 'choice',
      label: 'Language',
      default: 'auto',
      options: { from: 'languages', include: [{ value: 'auto', label: 'Auto-detect', lcd: 'any language' }] },
      wire: 'param:whisperLanguage',
    },
  ],
  face: [{ part: 'unitLcd', opts: { lines: ['pp.wh.model', 'pp.wh.lang'], prefix: 'WHISPER ', notAtLab: true } }],
  thumb: { file: 'ex-wh' },
} satisfies UnitDef;
