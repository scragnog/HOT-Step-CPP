/**
 * useAudioGeneration.ts — Send-to-Create flow for Lyric Studio V2.
 *
 * Handles: preset loading → localStorage writes → page navigation.
 *
 * NOTE: The generateAudio function and mergeCreatePanelSettings helper
 * were removed — all audio generation now flows through
 * audioGenQueueStore.enqueueAudioGen() which takes a getGlobalParams()
 * snapshot, ensuring 100% parity with the Create page path.
 */

import { useCallback } from 'react';
import { lireekApi } from '../../services/lireekApi';
import { writePersistedState } from '../../hooks/usePersistedState';
import type { Generation, Profile, AlbumPreset } from '../../services/lireekApi';

// ── Hook ─────────────────────────────────────────────────────────────────────

interface UseAudioGenerationOptions {
  profiles: Profile[];
  showToast: (msg: string) => void;
}

export function useAudioGeneration({ profiles, showToast }: UseAudioGenerationOptions) {

  const sendToCreate = useCallback(async (gen: Generation): Promise<void> => {
    const profile = profiles.find(p => p.id === gen.profile_id);
    let preset: AlbumPreset | null = null;
    if (profile) {
      try {
        const res = await lireekApi.getPreset(profile.lyrics_set_id);
        preset = res.preset;
      } catch { /* ignore */ }
    }

    // Write to hs-* localStorage keys AND fire same-tab StorageEvent so
    // usePersistedState hooks in the top bar update immediately.
    const write = (key: string, value: any) => writePersistedState(key, value);

    // Content
    write('hs-caption', gen.caption || '');
    write('hs-lyrics', gen.lyrics || '');
    write('hs-instrumental', false);

    // Song info (Title / Artist / Subject)
    write('hs-title', gen.title || '');
    write('hs-artist', gen.artist_name || '');
    write('hs-subject', gen.subject || '');

    // Metadata
    if (gen.bpm) write('hs-bpm', gen.bpm);
    if (gen.key) {
      // Normalize casing: LLM may produce "B Major" but dropdown expects "B major"
      const parts = gen.key.trim().split(/\s+/);
      const normalized = parts.length === 2
        ? `${parts[0]} ${parts[1].toLowerCase()}`
        : gen.key;
      write('hs-keyScale', normalized);
    }
    if (gen.duration) write('hs-duration', gen.duration);

    // Adapter from album preset — only set the path, scales are global
    if (preset?.adapter_path) {
      write('hs-adapter', preset.adapter_path);
      write('hs-adaptersOpen', true);
    }

    // Mastering reference from album preset (does NOT force-enable — respects global toggle)
    if (preset?.reference_track_path) {
      write('hs-masteringReference', preset.reference_track_path);
      write('hs-timbreReference', true);
    }

    console.log(`[LyricStudioV2] Send to Create: "${gen.title}" (adapter: ${preset?.adapter_path || 'none'}, mastering: ${preset?.reference_track_path || 'none'})`);

    // Navigate to Create page — save current LS URL first so sidebar can restore it
    try { localStorage.setItem('hs-lastLyricStudioUrl', window.location.pathname); } catch { /* ignore */ }
    window.history.pushState({}, '', '/');
    window.dispatchEvent(new PopStateEvent('popstate'));
  }, [profiles]);

  return { sendToCreate };
}
