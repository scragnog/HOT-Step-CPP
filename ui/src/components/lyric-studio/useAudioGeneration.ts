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
import { resolveDuration } from '../../utils/estimateDuration';
import { useGlobalParamsStore } from '../../stores/globalParamsStore';
import { captionForBackend } from '../../utils/captionForBackend';
import { normalizeKeyScale } from '../../utils/keyScale';
import { useLmAdapterEnabled } from '../../utils/lmAdapterPref';
import { useBackendStore } from '../../stores/backendStore';

// ── Hook ─────────────────────────────────────────────────────────────────────

interface UseAudioGenerationOptions {
  profiles: Profile[];
  showToast: (msg: string) => void;
}

export function useAudioGeneration({ profiles, showToast: _showToast }: UseAudioGenerationOptions) {

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

    // Content. The caption box holds ONE caption, so which of the generation's
    // two goes in it depends on the backend that is about to render it.
    write('hs-caption', captionForBackend(gen, useBackendStore.getState().activeBackendId));
    write('hs-lyrics', gen.lyrics || '');
    write('hs-instrumental', false);

    // Song info (Title / Artist / Subject)
    write('hs-title', gen.title || '');
    write('hs-artist', gen.artist_name || '');
    write('hs-subject', gen.subject || '');

    // Metadata
    if (gen.bpm) write('hs-bpm', gen.bpm);
    if (gen.key) write('hs-keyScale', normalizeKeyScale(gen.key));
    // Duration. resolveDuration estimates a length from the lyrics and tempo,
    // which is what ACE needs: its LM is TOLD a length and aims for it.
    // MM3 works the other way round — the length never reaches the planner LM
    // at all (it is not in the assembled prompt), so the request's duration is
    // only a frame ceiling and the song ends on the LM's own stop token. An
    // estimate there does nothing but cut the song off early, so send Auto.
    if (useBackendStore.getState().activeBackendId === 'minimax-m3') {
      write('hs-duration', -1);
    } else if (gen.duration || gen.bpm) {
      write('hs-duration', resolveDuration(gen.duration, gen.lyrics || '', gen.bpm || 120));
    }

    // Adapter from album preset — update Zustand store directly (writePersistedState
    // only touches localStorage, which the Zustand store doesn't listen to after init)
    const gps = useGlobalParamsStore.getState();
    if (preset?.adapter_path) {
      gps.setAdapter(preset.adapter_path);
      // An Advanced-mode adapter stack supersedes the single adapter in
      // getGlobalParams() — replace it too, otherwise the preset swap is
      // silently ignored and the previously stacked adapters keep playing.
      if (gps.advancedAdapters && gps.adapterStack && gps.adapterStack.length > 0) {
        gps.setAdapterStack([{ path: preset.adapter_path, scale: gps.adapterScale ?? 1.0 }]);
      }
      gps.setAdaptersOpen(true);
    }

    // Planner-LM adapter from album preset (path only — strength stays with
    // the global Adapters-menu slider, mirroring the DiT adapter semantics).
    // Gated on the sidebar's "Use LM Adapter" toggle, OFF by default; when it
    // is off the selection is cleared rather than merely skipped, since this
    // path writes the global and a stale preset adapter would otherwise ride
    // along into Create. See utils/lmAdapterPref.ts.
    if (useLmAdapterEnabled()) {
      if (preset?.lm_adapter_path) {
        gps.setLmAdapter(preset.lm_adapter_path);
      }
    } else {
      gps.setLmAdapter('');
    }

    // Mastering reference from album preset (does NOT force-enable — respects global toggle)
    if (preset?.reference_track_path) {
      gps.setMasteringReference(preset.reference_track_path);
      gps.setTimbreReference(true);
    }

    console.log(`[LyricStudioV2] Send to Create: "${gen.title}" (adapter: ${preset?.adapter_path || 'none'}, mastering: ${preset?.reference_track_path || 'none'})`);

    // Navigate to Create page — save current LS URL first so sidebar can restore it
    try { localStorage.setItem('hs-lastLyricStudioUrl', window.location.pathname); } catch { /* ignore */ }
    window.history.pushState({}, '', '/');
    window.dispatchEvent(new PopStateEvent('popstate'));
  }, [profiles]);

  return { sendToCreate };
}
