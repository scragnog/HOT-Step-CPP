// coverStemsStore.ts — Cover Studio's SuperSep split-stems result.
//
// Was component-local state in CoverStudio.tsx, so leaving the tab (which
// unmounts the component) destroyed the split and killed the in-flight
// progress poll (#135). Zustand so the result and the poll both survive
// navigation — same convention as globalParamsStore/trainingStore.

import { create } from 'zustand';
import {
  startSeparation as apiStartSeparation, waitForCompletion, getStemAudioUrl,
  releaseSeparation as apiReleaseSeparation,
  type SeparationLevel,
} from '../services/supersepApi';
import type { StemControl, MixerStemInfo } from '../components/shared/StemMixer';

interface CoverStemsState {
  sepJobId: string | null;
  sepStems: MixerStemInfo[] | null;
  stemControls: StemControl[];
  showMixer: boolean;
  isSeparating: boolean;
  sepProgress: number;
  sepMessage: string;

  setShowMixer(v: boolean): void;
  setStemControls(controls: StemControl[]): void;
  /** Starts a split and polls it to completion in the store, not the
   *  component, so a split survives the user navigating away mid-poll.
   *  Releases whatever job it replaces. Throws on failure — caller toasts. */
  startSeparation(sourceAudioUrl: string, level: SeparationLevel): Promise<void>;
  /** Discards the current split result and releases its server-side job. */
  clearStems(): void;
}

export const useCoverStemsStore = create<CoverStemsState>((set, get) => ({
  sepJobId: null,
  sepStems: null,
  stemControls: [],
  showMixer: false,
  isSeparating: false,
  sepProgress: 0,
  sepMessage: '',

  setShowMixer: (v) => set({ showMixer: v }),
  setStemControls: (controls) => set({ stemControls: controls }),

  startSeparation: async (sourceAudioUrl, level) => {
    const prevJobId = get().sepJobId;
    if (prevJobId) void apiReleaseSeparation(prevJobId);

    set({
      isSeparating: true, sepProgress: 0, sepMessage: 'Starting separation…',
      sepStems: null, sepJobId: null,
    });
    try {
      const jobId = await apiStartSeparation(sourceAudioUrl, level);
      set({ sepJobId: jobId });
      const result = await waitForCompletion(jobId, (progress, message) => {
        // A later clearStems()/startSeparation() call superseded this job —
        // don't let a straggling progress tick resurrect its state.
        if (get().sepJobId !== jobId) return;
        set({ sepProgress: progress, sepMessage: message });
      });
      if (get().sepJobId !== jobId) return;

      const mixerStems: MixerStemInfo[] = result.stems.map(s => ({
        name: s.name, category: s.category,
        audioUrl: getStemAudioUrl(jobId, s.index),
        index: s.index, stage: s.stage,
      }));
      set({
        sepStems: mixerStems,
        stemControls: mixerStems.map(s => ({ index: s.index, volume: 1.0, muted: false })),
        showMixer: true,
        isSeparating: false,
      });
    } catch (err) {
      set({ isSeparating: false });
      throw err;
    }
  },

  clearStems: () => {
    const jobId = get().sepJobId;
    if (jobId) void apiReleaseSeparation(jobId);
    set({
      sepJobId: null, sepStems: null, stemControls: [], showMixer: false,
      isSeparating: false, sepProgress: 0, sepMessage: '',
    });
  },
}));
