export type Yue2RenderMode = 'base' | 'artist' | 'custom';

type Pair = { ar?: string; nar?: string; artist?: string };

export function resolveYue2Render(mode: Yue2RenderMode, custom: Pair, existing: Pair) {
  const chosen = mode === 'artist' ? existing : custom;
  return {
    lmAdapterAr: mode === 'base' ? '' : chosen.ar || '',
    lmAdapterNar: mode === 'base' ? '' : chosen.nar || '',
    artist: chosen.artist?.trim() || '',
  };
}
