// sheetTranscriber.ts — which lead-sheet transcriber every SheetSage pass uses.
//
// SheetSage2 or SheetSage2-Refined (with optional modal keys), chosen in the
// YuE2 Models dropdown. Stored on the server (preferences family
// `sheet-transcriber`), which applies it to YuE2 cover transcription, the AR
// sheet stage and labelling alike. Not a backendParams knob: those go out with
// every generation and the server refuses undeclared ones.

import { useQuery } from '@tanstack/react-query';
import { queriedSettingsApi, settingOptions } from '../../queries/shell-queue/preferences';

export type SheetTranscriber = 'sheetsage2' | 'sheetsage2-refined';
export interface SheetTranscriberChoice { transcriber: SheetTranscriber; keyMode: boolean }

const DEFAULT: SheetTranscriberChoice = { transcriber: 'sheetsage2', keyMode: false };

export function useSheetTranscriber(): { choice: SheetTranscriberChoice; set: (next: Partial<SheetTranscriberChoice>) => void } {
  const { data } = useQuery(settingOptions<SheetTranscriberChoice>('sheet-transcriber'));
  const doc = data?.document;
  const choice = doc?.body ?? DEFAULT;
  const set = (next: Partial<SheetTranscriberChoice>) =>
    void queriedSettingsApi.upsert('sheet-transcriber', doc?.revision, { ...choice, ...next });
  return { choice, set };
}
