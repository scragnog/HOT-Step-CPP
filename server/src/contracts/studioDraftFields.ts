// contracts/studioDraftFields.ts — which persisted fields each studio draft
// holds, and the value each accepts (docs/dev/frontend-library.md).
//
// Draft fields keep the browser storage names they were saved under, so a
// draft written by the bundled UI and one written by another client agree.
// The server checks only the key's studio and its value type; it never
// interprets a value. Selection pointers, playback state and device handles
// are never draft fields.

import type { StudioKind } from './studioDrafts.js';
import { prepareV2Problem } from './yue2Prepare.js';

const keys = (list: string, prefix = '') => list.split(' ').map(x => `${prefix}${x}`);

const CREATE = new Set(keys('caption lyrics negative-prompt instrumental lora-trigger beat-intro intro-bars title artist subject bpm keyScale timeSignature duration vocalLanguage vocalGender sourceLatentUrl draftName', 'hs-'));
const COVER = new Set(keys('sourceFileName sourceAudioUrl sourceAssetId sourceSongId metadata analysis songArtist songTitle lyrics lyricsSource datasetAnalysis selectedArtistId selectedPreset artistCaption audioCoverStrength coverNoiseStrength coverNoiseMethod tempoScale pitchShift bpmCorrection bpmOverride keyOverride noFsq coverInstrumental sourceLatentUrl coverVocalLanguage coverTimbreOverride sepLevel sourceMode advancedMode pairMode yue2Ar yue2Nar coverVoices keepChords instrumentalMode coverTempoMode coverBpm coverKeyShift coverCfgScale coverGuide coverLyricSchedule captionMode sheetAbc prepareV2 scoreSections scoreCuts scoreTempoScale sheetAudioUrl approvedSheet selectedYue2ArtistId selectedYue2LyricsSetId coveringBand buildMode buildSessionId buildLyricsDraft draftName leadSheetSource sheetScoreSource chordQuality heartmulaTags heartmulaSeed sa3Transform sa3Seed sa3Adapter mm3Cover', 'cover-studio-'));
const REPAINT = new Set(keys('sourceSong sourceAssetId sourceAudioUrl sourceName regionStart regionEnd lyrics repaintMode crossfadeFrames styleCaption', 'hs-repaint-'));
const STORM = new Set(keys('caption lyrics neg lora instrumental beat-intro intro-bars bpm duration-v2 deck-a-caption deck-a-lyrics deck-a-bpm deck-b-caption deck-b-lyrics deck-b-bpm', 'hs-storm-'));
const STEM = new Set(keys('hs-stem-sourceUrl hs-stem-sourceFile hs-stem-sepLevel hs-stem-extractModel'));
const BUILDER = new Set(keys('hs-sb-sourceUrl hs-sb-sourceRef hs-sb-sourceFile hs-sb-model'));
/** Create's per-dataset caption-source keys, matched by pattern. */
export const CREATE_CAPTION_SOURCE_KEY = /^(hs-mm3CaptionSource:[^:]+|hs-mm3CaptionSources|hs-yue2CaptionSource:ds:[^:]+|hs-yue2CaptionSource:ds:song:[^:]+:[^:]+|hs-yue2CaptionDataset|hs-yue2CaptionSources)$/;

const BOOLEAN = new Set(keys('hs-instrumental hs-beat-intro cover-studio-datasetAnalysis cover-studio-noFsq cover-studio-coverInstrumental cover-studio-advancedMode cover-studio-keepChords cover-studio-buildMode hs-storm-instrumental hs-storm-beat-intro'));
const NUMBER = new Set(keys('hs-bpm hs-duration hs-intro-bars cover-studio-audioCoverStrength cover-studio-coverNoiseStrength cover-studio-tempoScale cover-studio-pitchShift cover-studio-bpmCorrection cover-studio-sepLevel cover-studio-coverBpm cover-studio-coverKeyShift cover-studio-coverCfgScale cover-studio-heartmulaSeed cover-studio-sa3Transform cover-studio-sa3Seed hs-repaint-regionStart hs-repaint-regionEnd hs-repaint-crossfadeFrames hs-storm-intro-bars hs-storm-bpm hs-storm-duration-v2 hs-storm-deck-a-bpm hs-storm-deck-b-bpm'));
const NULLABLE_NUMBER = new Set(keys('cover-studio-selectedArtistId cover-studio-bpmOverride cover-studio-selectedYue2ArtistId cover-studio-selectedYue2LyricsSetId'));
const NULLABLE_STRING = new Set(keys('cover-studio-lyricsSource cover-studio-keyOverride cover-studio-sheetScoreSource cover-studio-chordQuality'));
const OBJECT = new Set(keys('cover-studio-metadata cover-studio-analysis cover-studio-selectedPreset cover-studio-scoreTempoScale cover-studio-approvedSheet cover-studio-prepareV2 cover-studio-buildLyricsDraft cover-studio-coverGuide cover-studio-leadSheetSource cover-studio-heartmulaTags cover-studio-mm3Cover hs-repaint-sourceSong hs-sb-sourceRef'));
const ARRAY = new Set(keys('cover-studio-scoreSections cover-studio-scoreCuts'));

/** Keys whose browser value is a raw string, not JSON (imports keep it as is). */
export const RAW_STRING_DRAFT_KEYS: ReadonlySet<string> = new Set([...STEM, ...BUILDER].filter(x => x !== 'hs-sb-sourceRef'));

export type DraftFieldType = 'string' | 'boolean' | 'number' | 'number|null' | 'string|null' | 'object|null' | 'array';

/** Every fixed draft key, by studio, with the value type the server accepts.
 *  Create also accepts keys matching CREATE_CAPTION_SOURCE_KEY. */
export const STUDIO_DRAFT_FIELDS: Readonly<Record<StudioKind, Readonly<Record<string, DraftFieldType>>>> = (() => {
  const byStudio: Array<[StudioKind, Set<string>]> = [['create', CREATE], ['cover', COVER], ['repaint', REPAINT],
    ['storm', STORM], ['stem-studio', STEM], ['stem-builder', BUILDER]];
  return Object.fromEntries(byStudio.map(([studio, set]) =>
    [studio, Object.fromEntries([...set].map(key => [key, draftFieldType(key)]))])) as Record<StudioKind, Record<string, DraftFieldType>>;
})();

export function draftFieldType(key: string): DraftFieldType {
  if (BOOLEAN.has(key)) return 'boolean';
  if (NUMBER.has(key)) return 'number';
  if (NULLABLE_NUMBER.has(key)) return 'number|null';
  if (NULLABLE_STRING.has(key)) return 'string|null';
  if (ARRAY.has(key)) return 'array';
  if (OBJECT.has(key) || key.startsWith('hs-mm3CaptionSource:') || key === 'hs-mm3CaptionSources' ||
    key.startsWith('hs-yue2CaptionSource:') || key === 'hs-yue2CaptionSources') return 'object|null';
  return 'string';
}

/** Whether `value` is acceptable for `key`. */
export function validDraftField(key: string, value: unknown): boolean {
  switch (draftFieldType(key)) {
    case 'boolean': return typeof value === 'boolean';
    case 'number': return typeof value === 'number' && Number.isFinite(value);
    case 'number|null': return value === null || (typeof value === 'number' && Number.isFinite(value));
    case 'string|null': return value === null || typeof value === 'string';
    case 'object|null':
      // The Prepare document is checked whole (contracts/yue2Prepare.ts).
      if (key === 'cover-studio-prepareV2' && value !== null) return prepareV2Problem(value) === null;
      return value === null || (typeof value === 'object' && !Array.isArray(value));
    case 'array': return Array.isArray(value);
    default: return typeof value === 'string';
  }
}

/** The studio a draft key belongs to, or null for a key no draft holds. */
export function studioForDraftKey(key: string): StudioKind | null {
  if (CREATE.has(key) || CREATE_CAPTION_SOURCE_KEY.test(key)) return 'create';
  if (COVER.has(key)) return 'cover';
  if (REPAINT.has(key)) return 'repaint';
  if (STORM.has(key)) return 'storm';
  if (STEM.has(key)) return 'stem-studio';
  if (BUILDER.has(key)) return 'stem-builder';
  return null;
}

/** Fields that describe a draft's source. Changing any of them (or the
 *  draft's sourceAssetId, sourceSongId or sourceRevision) drops the
 *  DRAFT_SOURCE_RESULT_FIELDS on save: results of the old source. */
export const DRAFT_SOURCE_FIELDS: Readonly<Partial<Record<StudioKind, readonly string[]>>> = {
  cover: ['cover-studio-sourceAssetId', 'cover-studio-sourceSongId', 'cover-studio-sourceAudioUrl'],
  repaint: ['hs-repaint-sourceAssetId', 'hs-repaint-sourceSong', 'hs-repaint-sourceAudioUrl'],
  create: ['hs-sourceLatentUrl'],
};
export const DRAFT_SOURCE_RESULT_FIELDS: Readonly<Partial<Record<StudioKind, readonly string[]>>> = {
  cover: ['cover-studio-analysis', 'cover-studio-metadata', 'cover-studio-artistCaption', 'cover-studio-lyricsSource'],
  create: ['hs-mm3CaptionSources', 'hs-yue2CaptionSources'],
};
