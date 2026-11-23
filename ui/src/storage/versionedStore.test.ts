import assert from 'node:assert/strict';
import { beforeEach, describe, it } from 'node:test';
import { createVersionedStore, storageKey } from './versionedStore';
import { scopeFromAudioUrl } from './trackScope';
import {
  DEFAULT_REALIGN_SETTINGS, editorPrefs, loadRealignSettings, saveRealignSettings,
} from './lyricEditorSettings';

class MemoryStorage {
  private values = new Map<string, string>();
  getItem(key: string): string | null { return this.values.get(key) ?? null; }
  setItem(key: string, value: string): void { this.values.set(key, String(value)); }
  removeItem(key: string): void { this.values.delete(key); }
  clear(): void { this.values.clear(); }
  key(index: number): string | null { return [...this.values.keys()][index] ?? null; }
  get length(): number { return this.values.size; }
}

beforeEach(() => {
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: new MemoryStorage(),
  });
});

describe('versioned storage', () => {
  it('puts the version in the key, after the scope', () => {
    assert.equal(storageKey('realign', 1, 'song-1'), 'realign-song-1:1');
    assert.equal(storageKey('lyric-editor-zoom', 2), 'lyric-editor-zoom:2');
  });

  it('migrates an older version and writes it to the current key', () => {
    const store = createVersionedStore<{ n: number }>({
      namespace: 'probe',
      version: 2,
      scope: 's',
      migrate: value => {
        if (!value || typeof value !== 'object') return null;
        const n = Number((value as { n?: unknown }).n);
        return Number.isFinite(n) ? { n } : null;
      },
    });
    localStorage.setItem('probe-s:1', JSON.stringify({ n: 7 }));
    assert.deepEqual(store.read(), { n: 7 });
    assert.equal(localStorage.getItem('probe-s:2'), '{"n":7}');
  });

  it('reads an unversioned legacy key once', () => {
    localStorage.setItem('hs-lyricEditorFollow', '0');
    assert.equal(editorPrefs.follow.read(true), false);
    assert.equal(localStorage.getItem('lyric-editor-follow:1'), 'false');
  });

  it('falls back to defaults when a track has nothing stored', () => {
    assert.deepEqual(loadRealignSettings('song-1'), DEFAULT_REALIGN_SETTINGS);
  });

  it('keeps repair settings per track and round-trips them', () => {
    saveRealignSettings('song-1', { ...DEFAULT_REALIGN_SETTINGS, snap: false });
    saveRealignSettings('song-2', { ...DEFAULT_REALIGN_SETTINGS, repeats: false });
    assert.ok((localStorage.getItem('realign-song-1:1')).includes('"snap":false'));
    assert.equal(loadRealignSettings('song-1').snap, false);
    assert.equal(loadRealignSettings('song-1').repeats, true);
    assert.equal(loadRealignSettings('song-2').repeats, false);
  });

  it('fills a partly-written value from the defaults', () => {
    localStorage.setItem('realign-song-1:1', JSON.stringify({ cutBack: false }));
    assert.deepEqual(loadRealignSettings('song-1'), {
      ...DEFAULT_REALIGN_SETTINGS, cutBack: false,
    });
  });
});

describe('track scope', () => {
  it('ignores the query and is stable across variants of the same URL', () => {
    const base = scopeFromAudioUrl('/audio/take%201.wav');
    assert.equal(scopeFromAudioUrl('/audio/take%201.wav?_t=123'), base);
    assert.notEqual(scopeFromAudioUrl('/audio/take%201_mastered.wav'), base);
    assert.doesNotMatch(base, /[/?#\s]/);
  });

  it('does not share settings between two folders with the same basename', () => {
    assert.notEqual(scopeFromAudioUrl('/audio/a/song.wav'), scopeFromAudioUrl('/audio/b/song.wav'));
  });
});
