/**
 * useAudioGeneration.ts — Encapsulates the full audio generation flow
 * for Lyric Studio V2.
 *
 * Handles: preset loading → param merging → adapter params →
 * trigger word → mastering → generation → audio linking.
 *
 * Adapted for the C++ engine which accepts adapter params directly in generate call.
 */

import { useCallback } from 'react';
import { lireekApi } from '../../services/lireekApi';
import { generateApi } from '../../services/api';
import { useAuth } from '../../context/AuthContext';
import type { Generation, Profile, AlbumPreset } from '../../services/lireekApi';

// ── Helpers ──────────────────────────────────────────────────────────────────

function readPersisted(key: string): any {
  try {
    const raw = localStorage.getItem(key);
    return raw !== null ? JSON.parse(raw) : undefined;
  } catch { return undefined; }
}

/**
 * Read ALL generation params from the Create panel's localStorage.
 * These use the 'hs-' prefix and match the keys in CreatePanel.tsx.
 * We skip content fields (lyrics, caption, title, bpm, key, duration, timeSignature)
 * and adapter/mastering settings — those come from the album preset.
 */
function mergeCreatePanelSettings(params: Record<string, any>): void {
  // Every generation-affecting key from CreatePanel, mapped to the param name
  // that generateApi / translateParams expects.
  const map: [string, string][] = [
    // DiT settings
    ['hs-inferenceSteps', 'inferenceSteps'],
    ['hs-guidanceScale', 'guidanceScale'],
    ['hs-shift', 'shift'],
    ['hs-inferMethod', 'inferMethod'],
    ['hs-scheduler', 'scheduler'],
    ['hs-guidanceMode', 'guidanceMode'],
    // Seed
    ['hs-seed', 'seed'],
    ['hs-randomSeed', 'randomSeed'],
    ['hs-batchSize', 'batchSize'],
    // LM
    ['hs-skipLm', 'skipLm'],
    ['hs-useCotCaption', 'useCotCaption'],
    ['hs-lmTemperature', 'lmTemperature'],
    ['hs-lmCfgScale', 'lmCfgScale'],
    ['hs-lmTopK', 'lmTopK'],
    ['hs-lmTopP', 'lmTopP'],
    ['hs-lmNegativePrompt', 'lmNegativePrompt'],
    // Models
    ['hs-ditModel', 'ditModel'],
    ['hs-lmModel', 'lmModel'],
    ['hs-vaeModel', 'vaeModel'],
    // Solver sub-params
    ['hs-storkSubsteps', 'storkSubsteps'],
    ['hs-beatStability', 'beatStability'],
    ['hs-frequencyDamping', 'frequencyDamping'],
    ['hs-temporalSmoothing', 'temporalSmoothing'],
    // Guidance sub-params
    ['hs-apgMomentum', 'apgMomentum'],
    ['hs-apgNormThreshold', 'apgNormThreshold'],
    // Language
    ['hs-vocalLanguage', 'vocalLanguage'],
  ];
  for (const [storageKey, paramKey] of map) {
    const val = readPersisted(storageKey);
    if (val !== undefined && val !== null) {
      params[paramKey] = val;
    }
  }
}

function applyTriggerWord(params: Record<string, any>, adapterPath: string): void {
  const useFilename = localStorage.getItem('ace-globalTriggerUseFilename') === 'true';
  const placement = (localStorage.getItem('ace-globalTriggerPlacement') as 'prepend' | 'append' | 'replace') || 'prepend';
  if (!useFilename) return;
  const fileName = adapterPath.replace(/\\/g, '/').split('/').pop() || '';
  const triggerWord = fileName.replace(/\.safetensors$/i, '');
  if (!triggerWord) return;
  const current = ((params.caption as string) || '').trim();
  if (current.toLowerCase().includes(triggerWord.toLowerCase())) return;
  if (placement === 'replace') { params.caption = triggerWord; }
  else if (placement === 'append') { params.caption = current ? `${current}, ${triggerWord}` : triggerWord; }
  else { params.caption = current ? `${triggerWord}, ${current}` : triggerWord; }
  console.log(`[LyricStudioV2] Trigger word '${triggerWord}' ${placement}ed → '${params.caption}'`);
}

// ── Hook ─────────────────────────────────────────────────────────────────────

interface UseAudioGenerationOptions {
  profiles: Profile[];
  showToast: (msg: string) => void;
  onJobLinked?: (generationId: number, jobId: string) => void;
}

export function useAudioGeneration({ profiles, showToast, onJobLinked }: UseAudioGenerationOptions) {
  const { token } = useAuth();

  const generateAudio = useCallback(async (gen: Generation): Promise<string | null> => {
    if (!token) { showToast('Not authenticated'); return null; }

    try {
      // 1) Find album preset for this generation
      const profile = profiles.find(p => p.id === gen.profile_id);
      let preset: AlbumPreset | null = null;
      if (profile) {
        const res = await lireekApi.getPreset(profile.lyrics_set_id);
        preset = res.preset;
      }

      // 2) Build base params
      const params: Record<string, any> = {
        lyrics: gen.lyrics || '',
        caption: gen.caption || '',
        title: gen.title || '',
        instrumental: false,
        duration: gen.duration || 180,
      };
      if (gen.bpm) params.bpm = gen.bpm;
      if (gen.key) params.keyScale = gen.key;

      // 3) Merge persisted CreatePanel settings
      mergeCreatePanelSettings(params);

      // 4) Adapter — pass directly to generate in cpp engine
      if (preset?.adapter_path) {
        params.loraPath = preset.adapter_path;
        params.loraScale = preset.adapter_scale ?? 1.0;
        if (preset.adapter_group_scales) {
          params.adapterGroupScales = preset.adapter_group_scales;
        }
        // 5) Trigger word
        applyTriggerWord(params, preset.adapter_path);
      }

      // 6) Reference Track — timbre conditioning + mastering
      if (preset?.reference_track_path) {
        params.sourceAudioUrl = preset.reference_track_path;
        params.audioCoverStrength = preset.audio_cover_strength ?? 0.5;
        params.masteringEnabled = true;
        params.masteringReference = preset.reference_track_path;
      }

      // 7) Mark as Lyric Studio generation
      params.source = 'lyric-studio';

      // 8) Start generation
      console.log(`[LyricStudioV2] Starting generation — title: "${params.title}"`);
      const res = await generateApi.submit(params as any, token);
      const jobId = res.jobId;
      showToast(`Audio job queued: ${jobId}`);

      // 9) Link audio to lyric generation
      if (jobId) {
        await lireekApi.linkAudio(gen.id, jobId);
        onJobLinked?.(gen.id, jobId);
      }

      return jobId;
    } catch (err) {
      showToast(`Audio generation failed: ${(err as Error).message}`);
      return null;
    }
  }, [token, profiles, showToast, onJobLinked]);

  const sendToCreate = useCallback(async (gen: Generation): Promise<void> => {
    const profile = profiles.find(p => p.id === gen.profile_id);
    let preset: AlbumPreset | null = null;
    if (profile) {
      try {
        const res = await lireekApi.getPreset(profile.lyrics_set_id);
        preset = res.preset;
      } catch { /* ignore */ }
    }

    const importData: Record<string, any> = {
      title: gen.title || '',
      prompt: gen.lyrics || '',
      style: gen.caption || '',
      instrumental: false,
    };
    if (gen.bpm) importData.bpm = gen.bpm;
    if (gen.key) importData.keyScale = gen.key;
    if (gen.duration) importData.duration = gen.duration;
    if (preset?.adapter_path) {
      importData.loraPath = preset.adapter_path;
      importData.loraScale = preset.adapter_scale ?? 1.0;
    }
    if (preset?.reference_track_path) {
      importData.masteringEnabled = true;
      importData.masteringReference = preset.reference_track_path;
    }

    localStorage.setItem('hotstep_lireek_import', JSON.stringify(importData));
    window.history.pushState({}, '', '/create');
    window.dispatchEvent(new PopStateEvent('popstate'));
  }, [profiles]);

  return { generateAudio, sendToCreate };
}
