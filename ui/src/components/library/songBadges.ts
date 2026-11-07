import type { Song } from '../../types';
import { songBackend } from '../details/songFacts';

export interface SongBadge {
  label: 'ar' | 'vae' | 'cfg' | 'solver' | 'quant' | 'scale' | 'bias' | 'sound';
  value: string;
}

function params(song: Song): Record<string, any> {
  return song.generationParams || song.generation_params || {};
}

export function coverTag(song: Song): string {
  const gp = params(song);
  return (gp.source === 'cover-studio' || (songBackend(song) === 'yue2' && gp.yue2Cover)) && typeof gp.artistName === 'string'
    ? gp.artistName.trim() : '';
}

/** Read only the selected backend's parameters; global panels retain other backends' values. */
export function buildSongBadges(song: Song): SongBadge[] {
  const gp = params(song);
  const badges: SongBadge[] = [];
  const add = (label: SongBadge['label'], value: unknown) => {
    if (typeof value === 'number' && Number.isFinite(value)) badges.push({ label, value: String(value) });
    else if (typeof value === 'string' && value.trim()) badges.push({ label, value: value.trim() });
  };
  const pretty = (value: unknown) => typeof value === 'string'
    ? value.replace(/[_-]+/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : value;
  const quant = (model: unknown) => {
    if (typeof model !== 'string') return;
    const filename = model.split(/[\\/]/).pop() || '';
    add('quant', filename.match(/(?:^|[-_.])(BF16|F16|F32|Q\d(?:_[A-Z0-9]+)*)(?=[-_.]|$)/i)?.[1]?.toUpperCase());
  };
  switch (songBackend(song)) {
    case 'yue2':
      if (gp.yue2Adapters?.ar?.path) add('ar', gp.yue2Adapters.ar.scales?.global);
      add('vae', pretty(gp.yue2?.vae_variant ?? gp.yue2Request?.vae_variant ?? gp.yue2VaeVariant));
      break;
    case 'minimax-m3':
      add('cfg', gp.mm3?.cfg_flow ?? gp.mm3Request?.cfg_flow);
      if (gp.mm3LmAdapter) add('scale', gp.mm3LmAdapterScale);
      if (gp.mm3PlankPath && gp.mm3PlankSemanticOnly) add('bias', gp.mm3ForceBias === '' ? 'pin' : gp.mm3ForceBias);
      if (Number(gp.mm3A2aStrength) > 0) add('sound', gp.mm3A2aStrength);
      quant(gp.mm3Request?.model);
      break;
    case 'sa3':
      add('cfg', gp.sa3Request?.guidanceScale);
      add('solver', pretty(gp.sa3Request?.solver ?? gp.sa3Request?.sampler));
      break;
    default:
      add('cfg', gp.guidanceScale);
      add('solver', pretty(gp.inferMethod));
      quant(gp.ditModel || song.dit_model);
  }
  return badges;
}
