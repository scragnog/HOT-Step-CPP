import type { CoverDraft, CoverLeadSheetResult, CoverLeadSheetSource, HeartmulaCoverRolls } from '../contracts/studioWorkflows.js';
import { abcLeadSheetEvents, abcLeadSheetRoll, leadSheetRoll } from './leadSheetRoll.js';

/** Prefer a sounding Vocal; an explicit empty Vocal falls back to Ins. */
function coverRollAbc(abc: string): string {
  const vocal = abcLeadSheetEvents(abc);
  const hasVocal = /^V:\s*(Vocal|Vocals)\b/im.test(abc);
  if ((hasVocal && vocal.melody.length) || !/^V:\s*Ins\b/im.test(abc)) return abc;
  return abc.replace(/^(V:\s*)(Vocal|Vocals|Ins)\b/gim,
    (_all, prefix: string, id: string) => prefix + (/^Ins$/i.test(id) ? 'Vocal' : 'EmptyVocal'));
}
export const coverScoreEvents = (abc: string) => abcLeadSheetEvents(coverRollAbc(abc));

export function abcCoverLeadSheet(abc: string, source: CoverLeadSheetSource,
  scoreSource: CoverLeadSheetResult['scoreSource'] = source === 'sheetsage2-refined' ? 'sheetsage2' : source): CoverLeadSheetResult {
  const events = coverScoreEvents(abc);
  return { abc, scoreSource, leadSheetSource: source,
    chordQuality: scoreSource === 'muscriptor' ? 'estimated' : scoreSource === 'dataset' || scoreSource === 'musicxml' ? 'provided' : 'transcribed',
    timing: { conversionVersion: 1, path: 'abc', bpm: events.bpm,
      rawEvents: { melody: events.melody, chords: events.chords, drums: events.drums } } };
}

/** Prepare preserves incomplete ABC for editing. Only rendering requires a
 * readable tempo and supported notation; a failed preview must not lose work. */
export function tryAbcCoverLeadSheet(...args: Parameters<typeof abcCoverLeadSheet>): CoverLeadSheetResult | undefined {
  try { return abcCoverLeadSheet(...args); } catch { return undefined; }
}

/** The native wrapper in 6f supplies raw seconds separately from its editable ABC. */
export function nativeCoverLeadSheet(abc: string, timing: CoverLeadSheetResult['timing']): CoverLeadSheetResult {
  if (timing.path !== 'native' || timing.conversionVersion !== 1 || !timing.grid?.identity) {
    throw new Error('Native lead sheet needs raw timing and a grid identity');
  }
  // Validate and quantise from raw events only. Never recover native rolls from ABC.
  leadSheetRoll({ ...timing.rawEvents, bpm: timing.bpm, chordTiming: 'seconds' });
  return { abc, scoreSource: 'yourmt3-chordnet', leadSheetSource: 'yourmt3-chordnet', chordQuality: 'transcribed', timing };
}

export function coverRenderRolls(draft: CoverDraft): { rolls: HeartmulaCoverRolls; path: 'abc' | 'abc-edited' | 'native' } {
  if (!draft.approvedAbc) throw new Error('Review and approve the score first');
  const sheet = draft.leadSheet;
  const unchanged = sheet?.abc === draft.approvedAbc && sheet.leadSheetSource === draft.leadSheetSource;
  const native = unchanged && sheet.timing.path === 'native';
  const r = native ? leadSheetRoll({ ...sheet.timing.rawEvents, bpm: sheet.timing.bpm, chordTiming: 'seconds' })
    : abcLeadSheetRoll(coverRollAbc(draft.approvedAbc));
  return { path: native ? 'native' : sheet && !unchanged ? 'abc-edited' : 'abc',
    rolls: { version: 1, dtype: 'float32', bpm: r.bpm, length: r.length,
      pianoroll: Array.from(r.pianoroll), chord: Array.from(r.chord),
      drum_pianoroll: Array.from(r.drum_pianoroll), context_mask: Array.from(r.context_mask) } };
}

export function clearCoverScore(draft: CoverDraft, source: CoverLeadSheetSource): CoverDraft {
  const { leadSheet: _sheet, scoreSource: _score, ...rest } = draft;
  return { ...rest, leadSheetSource: source, abc: '', approvedAbc: '' };
}
