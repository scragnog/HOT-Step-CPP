import assert from 'node:assert/strict';
import test from 'node:test';
import { resolveYue2Render } from './yue2RenderMode';

const custom = { ar: 'custom-ar', nar: 'custom-nar', artist: 'Custom band' };
const existing = { ar: 'album-ar', nar: 'album-nar', artist: 'Album artist' };

test('render mode sends only its visible adapter source', () => {
  assert.deepEqual(resolveYue2Render('base', custom, existing), { lmAdapterAr: '', lmAdapterNar: '', artist: 'Custom band' });
  assert.deepEqual(resolveYue2Render('artist', custom, existing), { lmAdapterAr: 'album-ar', lmAdapterNar: 'album-nar', artist: 'Album artist' });
  assert.deepEqual(resolveYue2Render('custom', custom, existing), { lmAdapterAr: 'custom-ar', lmAdapterNar: 'custom-nar', artist: 'Custom band' });
});
