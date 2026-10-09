import fs from 'node:fs';
import path from 'node:path';
import type Database from 'better-sqlite3';
import { z } from 'zod/v4';
import { getDb } from '../../db/database.js';
import { config } from '../../config.js';
import { resolveAudioAsset } from '../assets/audioAssets.js';
import { getActiveBackendId, getBackend } from '../backends/registry.js';
import { readDuration } from '../training/audioMeta.js';
import { registerWorkflowKind } from '../../routes/workflows.js';
import { WorkflowError, type WorkflowContext, type WorkflowKind } from './workflowJobs.js';
import {
  repaintLayerSourceSchema as source, repaintRenderSchema as repaint, layerRenderSchema as layer,
  type RepaintInput, type LayerInput,
} from '../../contracts/studioWorkflows.js';

export type { RepaintInput, LayerInput };

export interface ResolvedSource { url: string; latentUrl?: string; path: string }

export function resolveRepaintLayerSource(db: Database.Database, ref: z.infer<typeof source>, userId: string,
  dirs: { dataDir: string; audioDir: string }): ResolvedSource {
  if (ref.kind === 'asset') {
    const asset = resolveAudioAsset(db, ref.id, userId, dirs.dataDir);
    if (asset.url !== ref.expectedUrl) throw new WorkflowError(409, 'Source changed; select it again');
    return { url: asset.url, path: asset.path };
  }
  const row = db.prepare('SELECT audio_url, latent_url FROM songs WHERE id = ? AND user_id = ?')
    .get(ref.id, userId) as { audio_url: string; latent_url: string | null } | undefined;
  if (!row?.audio_url) throw new WorkflowError(404, 'Source song was removed');
  if (row.audio_url !== ref.expectedUrl) throw new WorkflowError(409, 'Source song changed; select it again');
  if (!/^\/audio\/[^/]+$/.test(row.audio_url)) throw new WorkflowError(400, 'Unsupported source song URL');
  const file = path.join(dirs.audioDir, path.basename(row.audio_url));
  if (!fs.existsSync(file)) throw new WorkflowError(404, 'Source audio file was removed');
  return { url: row.audio_url, latentUrl: row.latent_url || undefined, path: file };
}

async function assertCapability(expectedBackend: string, feature: 'repaint' | 'lego'): Promise<void> {
  if (getActiveBackendId() !== expectedBackend) throw new WorkflowError(409, 'Active backend changed; start again');
  const backend = getBackend(expectedBackend);
  if (!backend) throw new WorkflowError(400, 'Unknown backend');
  const capabilities = await backend.capabilities();
  if (getActiveBackendId() !== expectedBackend) throw new WorkflowError(409, 'Active backend changed; start again');
  if (!capabilities.features[feature]) throw new WorkflowError(400, `${feature} is unavailable on this backend`);
}

export function effectiveRepaintRequest(input: RepaintInput, src: ResolvedSource): Record<string, unknown> {
  const ratios = { conservative: 0.7, balanced: 0.5, aggressive: 0.3 };
  return {
    ...input.engineParams, customMode: true, taskType: 'repaint', sourceAudioUrl: src.url,
    repaintingStart: input.regionStart, repaintingEnd: input.regionEnd,
    lyrics: input.lyrics || '[Instrumental]', style: input.styleCaption || input.engineParams.style || '',
    title: input.sourceName ? `${input.sourceName} (Repaint)` : 'Repaint',
    duration: 0, source: 'repaint', repaintCrossfadeFrames: input.crossfadeFrames,
    repaintInjectionRatio: ratios[input.repaintMode],
    ...(src.latentUrl ? { sourceLatentUrl: src.latentUrl } : {}),
    expectedBackend: input.expectedBackend,
  };
}

export function effectiveLayerRequest(input: LayerInput, src: ResolvedSource): Record<string, unknown> {
  return {
    ...input.engineParams, customMode: true, taskType: 'lego', trackName: input.trackName,
    sourceAudioUrl: src.url, caption: input.caption || '',
    lyrics: input.trackName === 'vocals' ? '' : '[Instrumental]', duration: 0,
    instrumental: input.trackName !== 'vocals', source: 'stem-builder',
    title: `${input.trackName.replace('_', ' ')} layer`, ditModel: input.buildModel,
    loraPath: '', loraScale: 0, adapterGroupScales: undefined, adapterMode: undefined,
    masteringEnabled: false, masteringReference: undefined, timbreReference: undefined,
    guidanceScale: 1.0, guidanceMode: 'apg', shift: 1.0, useCotCaption: false,
    inferMethod: 'euler', scheduler: 'linear', postProcessingEnabled: false,
    vocalNaturalizerEnabled: false, spectralLifterEnabled: false, ppVaeReencode: false,
    stableStepOn: false, denoiseStrength: 0, dcwEnabled: false, dcwMode: undefined,
    dcwLowScaler: undefined, dcwHighScaler: undefined, latentShift: 0.0,
    latentRescale: 1.0, customTimesteps: '', cfgCutoffRatio: 1.0,
    lmCfgCutoffRatio: 1.0, cacheRatio: 0, audioCoverStrength: 1.0,
    bpm: 0, keyScale: '', timeSignature: '', expectedBackend: input.expectedBackend,
  };
}

export interface RepaintLayerDeps {
  resolveSource?: (ref: z.infer<typeof source>, userId: string) => ResolvedSource;
  checkCapability?: (backend: string, feature: 'repaint' | 'lego') => Promise<void>;
  duration?: (path: string) => Promise<number>;
}

async function run(ctx: WorkflowContext<RepaintInput | LayerInput>, kind: 'repaint' | 'lego', deps: RepaintLayerDeps): Promise<unknown> {
  const input = ctx.input;
  await (deps.checkCapability || assertCapability)(input.expectedBackend, kind);
  const src = (deps.resolveSource || ((ref, userId) => resolveRepaintLayerSource(getDb(), ref, userId,
    { dataDir: config.data.dir, audioDir: config.data.audioDir })))(input.source, ctx.userId);
  if (kind === 'repaint') {
    const region = input as RepaintInput;
    if (region.regionEnd <= region.regionStart) throw new WorkflowError(400, 'Region end must follow start');
    const seconds = await (deps.duration || readDuration)(src.path);
    if (!(seconds > 0)) throw new WorkflowError(400, 'Source duration could not be read');
    if (region.regionStart >= seconds || region.regionEnd > seconds + 0.05)
      throw new WorkflowError(400, 'Region exceeds source duration');
  }
  ctx.throwIfCancelled();
  const request = kind === 'repaint'
    ? effectiveRepaintRequest(input as RepaintInput, src)
    : effectiveLayerRequest(input as LayerInput, src);
  ctx.emit('request', { source: src.url, kind });
  const item = ctx.audio.enqueue('render', request, { source: kind === 'repaint' ? 'repaint' : 'stem-builder' });
  const finished = await ctx.audio.wait(item.id);
  ctx.throwIfCancelled();
  if (finished.status !== 'succeeded') throw new Error(finished.error || `Audio ${finished.status}`);
  return { request, audioIntentId: item.id, audio: finished.result };
}

export function createRepaintLayerKinds(deps: RepaintLayerDeps = {}): WorkflowKind<any>[] {
  return [
    { kind: 'repaint-render', input: repaint, run: ctx => run(ctx, 'repaint', deps) },
    { kind: 'layer-render', input: layer, run: ctx => run(ctx, 'lego', deps) },
  ];
}

export function registerRepaintLayerWorkflows(): void {
  for (const kind of createRepaintLayerKinds()) registerWorkflowKind(kind);
}
