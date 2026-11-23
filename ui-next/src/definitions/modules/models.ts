// Module "Models". Concept: src/js/34-engine-panel.js and src/css/30-engine-panel.css.

import type { ModuleDef } from '../types.ts';

export const models = {
  kind: 'module',
  id: 'models',
  name: 'Models',
  desc: 'The model files the engine loads, in the engine panel.',
  at: 2,
  category: 'Engine',
  builtIn: true,
  engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
  owner: 'engine',
  stored: 'settings',
  size: { w: 2 },
  params: [
    {
      key: 'model.var',
      type: 'choice',
      label: 'Variant',
      undoName: 'Model variant',
      default: '',
      options: { from: 'models', bucket: 'variant', include: [{ value: '', label: 'Engine default' }] },
      engines: ['heartmula'],
      wire: [
        'model:heartmula.variant',
        {
          to: 'body:heartmulaVariant',
          note: 'Sent with the request so a queued job keeps the models it was queued with.',
        },
      ],
    },
    {
      key: 'model.dit',
      type: 'choice',
      label: 'DiT model',
      default: '',
      options: { from: 'models', bucket: 'dit', include: [{ value: '', label: 'Engine default' }] },
      engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],
      wire: 'param:ditModel',
      per: [
        { engines: ['ace'], undoName: 'Sound model' },
        { engines: ['minimax-m3'], label: 'Flow DiT', wire: 'model:minimax-m3.dit', undoName: 'Sound model' },
        {
          engines: ['yue2'],
          label: 'Language model (AR + NAR)',
          options: { from: 'models', bucket: 'lm', include: [{ value: '', label: 'Engine default' }] },
          wire: 'model:yue2.lm',
          undoName: 'Sound model',
        },
        { engines: ['sa3'], label: 'Flow DiT', wire: 'model:sa3.dit', undoName: 'Sound model' },
        {
          engines: ['heartmula'],
          label: 'Global LM',
          options: { from: 'models', bucket: 'global', include: [{ value: '', label: 'Engine default' }] },
          wire: [
            'model:heartmula.global',
            {
              to: 'body:heartmulaGlobalModel',
              note: 'Sent with the request so a queued job keeps the models it was queued with.',
            },
          ],
          undoName: 'Language model',
        },
      ],
    },
    {
      key: 'model.lm',
      type: 'choice',
      label: 'LM model',
      default: '',
      options: { from: 'models', bucket: 'lm', include: [{ value: '', label: 'Engine default' }] },
      engines: ['ace', 'minimax-m3', 'heartmula'],
      wire: 'param:lmModel',
      per: [
        { engines: ['ace'], undoName: 'Planner model' },
        { engines: ['minimax-m3'], label: 'Planner LM', wire: 'model:minimax-m3.lm', undoName: 'Planner model' },
        {
          engines: ['heartmula'],
          label: 'Local decoder',
          options: { from: 'models', bucket: 'local', include: [{ value: '', label: 'Engine default' }] },
          wire: [
            'model:heartmula.local',
            {
              to: 'body:heartmulaLocalModel',
              note: 'Sent with the request so a queued job keeps the models it was queued with.',
            },
          ],
          undoName: 'Decoder model',
        },
      ],
    },
    {
      key: 'model.x',
      type: 'choice',
      label: 'Depth decoder',
      default: '',
      options: { from: 'models', bucket: 'depth', include: [{ value: '', label: 'Engine default' }] },
      engines: ['minimax-m3', 'heartmula'],
      wire: 'model:minimax-m3.depth',
      per: [
        { engines: ['minimax-m3'], undoName: 'Detail model' },
        {
          engines: ['heartmula'],
          label: 'Codec flow',
          options: { from: 'models', bucket: 'flow', include: [{ value: '', label: 'Engine default' }] },
          wire: [
            'model:heartmula.flow',
            {
              to: 'body:heartmulaFlowModel',
              note: 'Sent with the request so a queued job keeps the models it was queued with.',
            },
          ],
          undoName: 'Codec model',
        },
      ],
    },
    {
      key: 'model.vae',
      type: 'choice',
      label: 'VAE decoder',
      undoName: 'Decoder model',
      default: '',
      options: { from: 'models', bucket: 'vae', include: [{ value: '', label: 'Engine default' }] },
      engines: ['ace', 'yue2'],
      wire: 'param:vaeModel',
      per: [{ engines: ['yue2'], label: 'VAE', wire: 'model:yue2.vae' }],
    },
  ],
  face: [
    { part: 'modelPicker', params: ['model.var', 'model.dit', 'model.lm', 'model.x', 'model.vae'] },
  ],
} satisfies ModuleDef;
