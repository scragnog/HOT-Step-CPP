// sheetTranscriber.ts — which lead-sheet transcriber YuE2 covers use.
//
// SheetSage2 or SheetSage2-Refined (with optional modal keys), chosen in the
// YuE2 Models dropdown and read when Cover Studio starts a transcription. Not a
// backendParams knob: those are sent with every generation and the server
// refuses undeclared ones, and this choice never reaches a render.

import { useSyncExternalStore } from 'react';

export type SheetTranscriber = 'sheetsage2' | 'sheetsage2-refined';
export interface SheetTranscriberChoice { transcriber: SheetTranscriber; keyMode: boolean }

const KEY = 'hs-yue2SheetTranscriber';
const DEFAULT: SheetTranscriberChoice = { transcriber: 'sheetsage2', keyMode: false };
const listeners = new Set<() => void>();
let cached: SheetTranscriberChoice | null = null;

export function sheetTranscriberChoice(): SheetTranscriberChoice {
  if (cached) return cached;
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? 'null') as Partial<SheetTranscriberChoice> | null;
    cached = {
      transcriber: v?.transcriber === 'sheetsage2-refined' ? 'sheetsage2-refined' : 'sheetsage2',
      keyMode: v?.keyMode === true,
    };
  } catch {
    cached = DEFAULT;
  }
  return cached;
}

export function setSheetTranscriberChoice(next: Partial<SheetTranscriberChoice>): void {
  cached = { ...sheetTranscriberChoice(), ...next };
  try { localStorage.setItem(KEY, JSON.stringify(cached)); } catch { /* storage full or blocked */ }
  listeners.forEach(l => l());
}

export function useSheetTranscriberChoice(): SheetTranscriberChoice {
  return useSyncExternalStore(
    (l) => { listeners.add(l); return () => listeners.delete(l); },
    sheetTranscriberChoice,
  );
}
