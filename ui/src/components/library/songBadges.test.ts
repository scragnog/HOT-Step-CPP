import assert from 'node:assert/strict';
import test from 'node:test';
import type { Song } from '@/types';
import { buildSongBadges, coverTag } from './songBadges';

const song = (backend: string, generationParams: Record<string, unknown>) => ({ backend, generationParams }) as Song;

test('YuE2 badges use rendered values and include zero AR strength', () => {
  const s = song('yue2', {
    guidanceScale: 99, inferMethod: 'stale', vaeModel: 'stale',
    yue2VaeVariant: 'legacy', yue2: { vae_variant: 'standard' },
    yue2Adapters: { ar: { path: 'adapter', scales: { global: 0 } } },
  });
  assert.deepEqual(buildSongBadges(s), [{ label: 'ar', value: '0' }, { label: 'vae', value: 'Standard' }]);
});

test('backend badges never read another backend\'s leftover settings', () => {
  const stale = { guidanceScale: 99, inferMethod: 'euler', ditModel: 'ace-BF16.gguf' };
  assert.deepEqual(buildSongBadges(song('minimax-m3', { ...stale, mm3: { cfg_flow: 0 }, mm3LmAdapterScale: 0.5 })),
    [{ label: 'cfg', value: '0' }]);
  assert.deepEqual(buildSongBadges(song('sa3', stale)), []);
  assert.deepEqual(buildSongBadges(song('yue2', stale)), []);
});

test('ACE badges show CFG, solver and only a recorded model quantization', () => {
  assert.deepEqual(buildSongBadges(song('ace', { guidanceScale: 4.5, inferMethod: 'wasserstein', ditModel: 'acestep-v15-Q8_0.gguf' })),
    [{ label: 'cfg', value: '4.5' }, { label: 'solver', value: 'Wasserstein' }, { label: 'quant', value: 'Q8_0' }]);
  assert.deepEqual(buildSongBadges({} as Song), []);
});

test('cover tag requires cover provenance and supports database field spelling', () => {
  assert.equal(coverTag(song('yue2', { artistName: 'iteration 0.9', yue2Cover: { sourceId: 'source' } })), 'iteration 0.9');
  assert.equal(coverTag({ generation_params: { source: 'cover-studio', artistName: 'take 2' } } as Song), 'take 2');
  assert.equal(coverTag(song('ace', { artistName: 'ordinary artist' })), '');
  assert.equal(coverTag(song('ace', { artistName: 'stale tag', yue2Cover: { sourceId: 'stale' } })), '');
});

test('MM3 and SA3 use their own request values, including zero and unity', () => {
  assert.deepEqual(buildSongBadges(song('minimax-m3', { mm3Request: { cfg_flow: 1.7 },
    mm3LmAdapter: 'adapter', mm3LmAdapterScale: 1 })), [{ label: 'cfg', value: '1.7' }, { label: 'scale', value: '1' }]);
  assert.deepEqual(buildSongBadges(song('sa3', { guidanceScale: 99,
    sa3Request: { guidanceScale: 0, solver: 'euler' } })), [{ label: 'cfg', value: '0' }, { label: 'solver', value: 'Euler' }]);
});
