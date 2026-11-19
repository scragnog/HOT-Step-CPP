import { beforeEach, describe, expect, it } from 'vitest';
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
    expect(storageKey('realign', 1, 'song-1')).toBe('realign-song-1:1');
    expect(storageKey('lyric-editor-zoom', 2)).toBe('lyric-editor-zoom:2');
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
    expect(store.read()).toEqual({ n: 7 });
    expect(localStorage.getItem('probe-s:2')).toBe('{"n":7}');
  });

  it('reads an unversioned legacy key once', () => {
    localStorage.setItem('hs-lyricEditorFollow', '0');
    expect(editorPrefs.follow.read(true)).toBe(false);
    expect(localStorage.getItem('lyric-editor-follow:1')).toBe('false');
  });

  it('falls back to defaults when a track has nothing stored', () => {
    expect(loadRealignSettings('song-1')).toEqual(DEFAULT_REALIGN_SETTINGS);
  });

  it('keeps repair settings per track and round-trips them', () => {
    saveRealignSettings('song-1', { ...DEFAULT_REALIGN_SETTINGS, snap: false });
    saveRealignSettings('song-2', { ...DEFAULT_REALIGN_SETTINGS, repeats: false });
    expect(localStorage.getItem('realign-song-1:1')).toContain('"snap":false');
    expect(loadRealignSettings('song-1').snap).toBe(false);
    expect(loadRealignSettings('song-1').repeats).toBe(true);
    expect(loadRealignSettings('song-2').repeats).toBe(false);
  });

  it('fills a partly-written value from the defaults', () => {
    localStorage.setItem('realign-song-1:1', JSON.stringify({ cutBack: false }));
    expect(loadRealignSettings('song-1')).toEqual({
      ...DEFAULT_REALIGN_SETTINGS, cutBack: false,
    });
  });
});

describe('track scope', () => {
  it('ignores the query and is stable across variants of the same URL', () => {
    const base = scopeFromAudioUrl('/audio/take%201.wav');
    expect(scopeFromAudioUrl('/audio/take%201.wav?_t=123')).toBe(base);
    expect(scopeFromAudioUrl('/audio/take%201_mastered.wav')).not.toBe(base);
    expect(base).not.toMatch(/[/?#\s]/);
  });

  it('does not share settings between two folders with the same basename', () => {
    expect(scopeFromAudioUrl('/audio/a/song.wav')).not.toBe(scopeFromAudioUrl('/audio/b/song.wav'));
  });
});
