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
import { captionForBackend, MM3_BACKEND_ID } from '../../utils/captionForBackend';
import { ensureMm3SourceTracks } from '../../utils/mm3CaptionSource';
import {
  activeYue2AdapterPath, applyYue2PresetAdapters, ensureYue2SourceTracks, YUE2_BACKEND_ID,
} from '../../utils/yue2CaptionSource';
import {
  MM3_CAPTION_SOURCES_KEY, clearMm3CaptionSources,
  readMm3CaptionSelection, readMm3SourceTracks,
  type Mm3CaptionSourcesHandoff,
} from '../../utils/mm3CaptionSource';
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
    const backendId = useBackendStore.getState().activeBackendId;
    const lyricsSetId = profile?.lyrics_set_id;
    if (backendId === MM3_BACKEND_ID) await ensureMm3SourceTracks(lyricsSetId);
    // YuE2's equivalent, and it has to happen HERE rather than in the picker:
    // resolveYue2CaptionForGeneration reads the track list out of the cache the
    // Create picker fills, so a song generated from Lyric Studio without that
    // panel ever being opened resolved to the written caption and the album's own
    // captions never reached the model.
    if (backendId === YUE2_BACKEND_ID) {
      const yue2Adapter = activeYue2AdapterPath();
      if (yue2Adapter) await ensureYue2SourceTracks(yue2Adapter);
    }
    write('hs-caption', captionForBackend(gen, backendId, lyricsSetId));
    write('hs-lyrics', gen.lyrics || '');

    // MM3 caption SOURCE — hand the Create panel everything it needs to offer
    // the same three-way control (automatic by tempo / a named source track /
    // this song's own caption) without a server call of its own. The tracks
    // were cached when the album loaded in Lyric Studio; an album that has no
    // captioned tracks hands over an empty list, which the panel reads as
    // "custom only". Cleared outright on ACE so a stale MM3 handoff cannot
    // resurface the control after a backend switch.
    if (backendId === MM3_BACKEND_ID) {
      const sel = readMm3CaptionSelection(gen.id);
      write(MM3_CAPTION_SOURCES_KEY, {
        mode: sel.mode,
        selectedTitle: sel.selectedTitle,
        customCaption: gen.caption_mm3 || '',
        tracks: readMm3SourceTracks(lyricsSetId),
      } satisfies Mm3CaptionSourcesHandoff);
    } else {
      clearMm3CaptionSources();
    }
    write('hs-instrumental', false);

    // Song info (Title / Artist / Subject)
    write('hs-title', gen.title || '');
    write('hs-artist', gen.artist_name || '');
    write('hs-subject', gen.subject || '');

    // Metadata
    if (gen.bpm) write('hs-bpm', gen.bpm);
    if (gen.key) write('hs-keyScale', normalizeKeyScale(gen.key));
    // Duration — ACE only. resolveDuration estimates a length from the lyrics
    // and tempo, and ACE's LM is told that length and aims for it. MM3 has no
    // length input at all: the number becomes a frame cap, so it can only ever
    // truncate the song. (7d574365 sent Auto for MM3 and was reverted because
    // renders then ran to the 300s ceiling — the ending arbitration is what
    // actually fixes that, and it makes the cap pure downside.) The Create
    // panel hides the control in MM3 mode and the backend ignores the field, so
    // this only avoids leaving a stale number in a box the user cannot see.
    if (backendId !== MM3_BACKEND_ID && (gen.duration || gen.bpm)) {
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

    // MM3 adapter from album preset (2026-09-11): the backend param the global
    // LM Adapter dropdown edits. Set when the preset names one, CLEARED when it
    // does not — the param persists across sessions, so a previous album's
    // adapter would otherwise keep planning this one. Scales stay where the
    // dropdown left them, mirroring the DiT adapter semantics.
    if (backendId === MM3_BACKEND_ID) {
      gps.setBackendParam('mm3LmAdapter', preset?.mm3_adapter_path || '');
    }

    // YuE2's two halves, same intent as the MM3 line above and a different
    // mechanism: the adapter is merged into the resident LM, so it is engine
    // state rather than a request param and has to be POSTed like the picker
    // does. Fire-and-forget — a preset that cannot be applied must not block a
    // hand-off, and the picker still shows what is actually in force.
    if (backendId === YUE2_BACKEND_ID) {
      void applyYue2PresetAdapters(preset as Parameters<typeof applyYue2PresetAdapters>[0]);
    }

    // Mastering reference from album preset (does NOT force-enable — respects global toggle).
    // ACE only: the timbre reference is an ACE-Step conditioning input and the
    // MM3 runner never reads it, so in MM3 mode it is left alone.
    if (backendId !== MM3_BACKEND_ID && preset?.reference_track_path) {
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
