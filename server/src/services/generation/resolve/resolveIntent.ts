// resolve/resolveIntent.ts — build the /api/generate body from an intent.
//
// Pure: every input it needs (the song row, album preset, source tracks)
// arrives as data, so the same function serves the preview route and the
// tests. Two callers are ported, step for step:
//
//   Create        ui/src/components/create/CreatePanel.tsx handleGenerate,
//                 plus its MM3 and YuE2 caption-source lock
//   Written song  ui/src/stores/audioGenQueueStore.ts _executeItem, steps 1-5
//
// What the browser also does there and Node deliberately does not: write the
// top bar's persisted adapter, LM adapter and mastering reference, and the MM3
// global adapter param. Those are client-visible selection changes; resolving
// a request must not make them. They are reported in `uiEffects` so the client
// can keep doing them for now.

import { createHash } from 'node:crypto';
import type {
  CreateIntent, Mm3SourceTrack, ResolveProvenance, WrittenSongIntent, Yue2SourceTrack,
} from '../../../contracts/resolution.js';
import { normalizeKeyScale } from '../../lireek/prompts.js';
import {
  captionForEngine, effectiveYue2Selection, normalizeMm3Selection, resolveMm3Caption, resolveYue2Caption,
  yue2CaptionAdapterPath, yue2PickAtEnqueue, MM3_ENGINE_ID, YUE2_ENGINE_ID,
} from './captionSource.js';
import { composeCreateCaption, createWildcardSeed, expandWildcards, hasWildcards, resolveDuration } from './content.js';

export interface ResolvedIntent {
  request: Record<string, unknown>;
  provenance: ResolveProvenance;
  warnings: string[];
  /** Client-side selection writes the browser made on this path. */
  uiEffects: Array<{ key: string; value: unknown }>;
}

/** The body asserts which engine it was resolved for. /api/generate refuses
 *  it with a 409 when the active engine differs (expectedBackendMismatch), so
 *  a preview made for one engine can never run on another after a switch. */
export class ResolveConflictError extends Error {}

function pinEngine(params: Record<string, unknown>, engine: string): void {
  const asserted = params.expectedBackend;
  if (asserted !== undefined && asserted !== engine) {
    throw new ResolveConflictError(`The request asserts expectedBackend '${String(asserted)}' but is being resolved for '${engine}'`);
  }
  params.expectedBackend = engine;
}

/** sha256 of the request with keys sorted at every level. */
export function requestVersion(request: Record<string, unknown>): string {
  const canon = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(canon);
    if (v && typeof v === 'object') {
      return Object.fromEntries(Object.keys(v as object).sort()
        .filter(k => (v as Record<string, unknown>)[k] !== undefined)
        .map(k => [k, canon((v as Record<string, unknown>)[k])]));
    }
    return v;
  };
  return createHash('sha256').update(JSON.stringify(canon(request))).digest('hex');
}

/** True when `body` is exactly the version a preview returned. */
export function verifyResolvedRequest(body: Record<string, unknown>, version: string): boolean {
  return typeof version === 'string' && requestVersion(body) === version;
}

// ── Create ───────────────────────────────────────────────────────────────────

export interface CreateData {
  /** Source tracks for the caption source, when one applies. */
  mm3Tracks?: Mm3SourceTrack[];
  yue2Tracks?: Yue2SourceTrack[];
}

export function resolveCreateIntent(intent: CreateIntent, engine: string, data: CreateData = {}): ResolvedIntent {
  const params = structuredClone(intent.params ?? {});
  const warnings: string[] = [];
  const typed = typeof params.caption === 'string' ? params.caption : '';
  const lyricsIn = typeof params.lyrics === 'string' ? params.lyrics : '';
  const bpm = typeof params.bpm === 'number' ? params.bpm : Number(params.bpm) || 0;

  // Caption source lock (CreatePanel mm3Resolved / yue2Resolved): only on the
  // matching engine and only with tracks to pick from.
  let base = typed;
  const captionProv: ResolveProvenance['caption'] = { source: 'box' };
  const src = intent.captionSource;
  if (src?.engine === YUE2_ENGINE_ID && engine === YUE2_ENGINE_ID && (data.yue2Tracks?.length ?? 0) > 0) {
    const sel = effectiveYue2Selection(src.datasetId, src.selection, src.adapterInForce);
    const r = resolveYue2Caption(typed, bpm, data.yue2Tracks!, sel);
    if (r.mode !== 'custom') {
      base = r.caption;
      Object.assign(captionProv, { source: 'dataset-track', mode: r.mode, fromTrack: r.fromTrack, datasetId: src.datasetId });
    } else Object.assign(captionProv, { mode: 'custom', datasetId: src.datasetId });
  } else if (src?.engine === MM3_ENGINE_ID && engine === MM3_ENGINE_ID && (data.mm3Tracks?.length ?? 0) > 0) {
    const r = resolveMm3Caption({ bpm, caption_mm3: src.customCaption }, data.mm3Tracks!, normalizeMm3Selection(src.selection));
    if (r.mode !== 'custom') {
      base = r.caption;
      Object.assign(captionProv, { source: 'album-track', mode: r.mode, fromTrack: r.fromTrack });
    } else captionProv.mode = 'custom';
  } else if (src && src.engine !== engine) {
    warnings.push(`Caption source is for '${src.engine}' but the request is for '${engine}'; the caption box is used`);
  }

  const compose = intent.compose ?? {};
  let caption = base;
  let lyrics = lyricsIn;
  let wildcards: ResolveProvenance['wildcards'];
  if (compose.autoExpand && (hasWildcards(caption) || hasWildcards(lyrics))) {
    const wc = createWildcardSeed(params.randomSeed, params.seed);
    const expandCaption = hasWildcards(caption);
    const expandLyrics = hasWildcards(lyrics);
    if (expandCaption) caption = expandWildcards(caption, wc.seed, 0);
    if (expandLyrics) lyrics = expandWildcards(lyrics, wc.seed, 0);
    wildcards = { seed: wc.seed, seedFrom: wc.seedFrom, caption: expandCaption, lyrics: expandLyrics };
  }
  params.caption = composeCreateCaption(caption, compose);
  params.lyrics = params.instrumental ? '[Instrumental]' : lyrics;
  const triggerWord = (compose.loraTrigger ?? '').trim();

  let duration: ResolveProvenance['duration'];
  if (engine === MM3_ENGINE_ID) {
    params.duration = -1;
    duration = { source: 'auto', value: -1 };
  } else if (typeof params.duration === 'number') {
    duration = { source: 'request', value: params.duration };
  }

  pinEngine(params, engine);
  return {
    request: params,
    provenance: {
      engine,
      caption: captionProv,
      ...(wildcards ? { wildcards } : {}),
      ...(duration ? { duration } : {}),
      trigger: triggerWord ? { source: 'compose', words: [triggerWord] } : { source: 'none' },
    },
    warnings,
    uiEffects: [],
  };
}

// ── Written song ─────────────────────────────────────────────────────────────

export interface WrittenSongData {
  gen: {
    id?: number; title?: string | null; lyrics?: string | null; caption?: string | null;
    caption_mm3?: string | null; caption_yue2?: string | null; bpm?: number | null; key?: string | null;
    duration?: number | null; subject?: string | null;
  };
  preset: {
    adapter_path?: string | null; lm_adapter_path?: string | null; mm3_adapter_path?: string | null;
    reference_track_path?: string | null; yue2_ar_adapter_path?: string | null; yue2_nar_adapter_path?: string | null;
  } | null;
  /** The target album's MM3-captioned source tracks. */
  mm3Tracks?: Mm3SourceTrack[];
  /** The dataset linked to the target album, and its tracks. */
  yue2Dataset?: { datasetId: string; tracks: Yue2SourceTrack[] };
}

const APP_FLAGS = ['coResident', 'cacheLmCodes', 'parallelWhisper', 'parallelQualityEval', 'parallelCoverArt'] as const;

export function resolveWrittenSongIntent(intent: WrittenSongIntent, engine: string, data: WrittenSongData): ResolvedIntent {
  const { gen, preset } = data;
  const settings = intent.settings ?? {};
  const snapshot = intent.params ?? {};
  const params: Record<string, unknown> = structuredClone(snapshot);
  const warnings: string[] = [];
  const uiEffects: ResolvedIntent['uiEffects'] = [];

  params.lyrics = gen.lyrics || '';

  let adapterInForce = false;
  if (engine === YUE2_ENGINE_ID) {
    const pick = intent.yue2Pick ?? yue2PickAtEnqueue(intent.yue2Defaults, preset);
    params.yue2Pick = pick;
    adapterInForce = !!yue2CaptionAdapterPath(pick);
  }
  const renderingAs = !!intent.sourceLyricsSetId && intent.sourceLyricsSetId !== intent.lyricsSetId;
  const datasetId = data.yue2Dataset?.datasetId ?? '';
  const song = { ...gen, bpm: gen.bpm ?? undefined };
  const cap = captionForEngine(song, engine, {
    mm3: { tracks: data.mm3Tracks ?? [], selection: normalizeMm3Selection(intent.mm3Selection) },
    yue2: { datasetId, tracks: data.yue2Dataset?.tracks ?? [], selection: effectiveYue2Selection(datasetId, intent.yue2Selection, adapterInForce) },
    renderingAs,
  });
  params.caption = cap.caption;
  params.title = gen.title || '';
  params.instrumental = false;

  let duration: ResolveProvenance['duration'];
  if (engine !== MM3_ENGINE_ID) {
    const d = resolveDuration(gen.duration ?? undefined, gen.lyrics || '', gen.bpm || 120, settings.useLlmDuration ?? true);
    params.duration = d.value;
    duration = d;
  } else {
    params.duration = -1;
    duration = { source: 'auto', value: -1 };
  }
  if (gen.bpm) params.bpm = gen.bpm;
  if (gen.key) params.keyScale = normalizeKeyScale(gen.key);
  if (intent.artistName) params.artist = intent.artistName;
  if (gen.subject) params.subject = gen.subject;

  if (!params.timeSignature) params.timeSignature = settings.timeSignature ?? '';
  if (!params.vocalLanguage) params.vocalLanguage = settings.vocalLanguage ?? 'en';

  if (settings.app) {
    for (const k of APP_FLAGS) params[k] = settings.app[k];
    if (typeof settings.app.generationTimeoutMinutes === 'number') params.generationTimeoutMinutes = settings.app.generationTimeoutMinutes;
  }

  let trigger: ResolveProvenance['trigger'] = { source: 'none' };
  if (preset?.adapter_path) {
    uiEffects.push({ key: 'hs-adapter', value: preset.adapter_path });
    params.loraPath = preset.adapter_path;
    params.loraStack = [{ path: preset.adapter_path, scale: typeof params.loraScale === 'number' ? params.loraScale : 1.0 }];
    delete params.triggerWord;
    delete params.triggerWords;
    delete params.triggerPlacement;
    trigger = { source: 'cleared' };
    if (settings.triggerUseFilename === true) {
      const fileName = preset.adapter_path.replace(/\\/g, '/').split('/').pop() || '';
      const word = fileName.replace(/\.safetensors$/i, '');
      if (word) {
        params.triggerWord = word;
        params.triggerWords = [word];
        params.triggerPlacement = settings.triggerPlacement || 'prepend';
        trigger = { source: 'preset-filename', words: [word] };
      }
    }
  }

  if (settings.useLmAdapter === true) {
    if (preset?.lm_adapter_path) {
      uiEffects.push({ key: 'hs-lmAdapter', value: preset.lm_adapter_path });
      params.lmAdapter = preset.lm_adapter_path;
    }
  } else {
    delete params.lmAdapter;
  }

  if (engine === MM3_ENGINE_ID) {
    const ref = preset?.mm3_adapter_path || '';
    params.mm3LmAdapter = ref;
    uiEffects.push({ key: 'backendParams.mm3LmAdapter', value: ref });
  }

  if (preset?.reference_track_path) {
    uiEffects.push({ key: 'hs-masteringReference', value: preset.reference_track_path }, { key: 'hs-timbreReference', value: true });
    params.masteringReference = preset.reference_track_path;
    params.timbreReference = typeof snapshot.timbreReference === 'string' && snapshot.timbreReference
      ? snapshot.timbreReference : true;
    if (settings.randomizeTimbreRef === true) params.randomizeTimbreRef = true;
  }

  params.taskType = 'text2music';
  params.source = 'lyric-studio';
  pinEngine(params, engine);

  return {
    request: params,
    provenance: {
      engine,
      caption: { source: cap.source, ...(cap.mode ? { mode: cap.mode } : {}), ...(cap.fromTrack ? { fromTrack: cap.fromTrack } : {}),
        ...(datasetId && engine === YUE2_ENGINE_ID ? { datasetId } : {}), renderingAs },
      duration,
      trigger,
      preset: {
        lyricsSetId: intent.lyricsSetId,
        ...(preset?.adapter_path ? { adapterPath: preset.adapter_path } : {}),
        ...(preset?.lm_adapter_path ? { lmAdapterPath: preset.lm_adapter_path } : {}),
        ...(preset?.mm3_adapter_path ? { mm3AdapterPath: preset.mm3_adapter_path } : {}),
        ...(preset?.reference_track_path ? { referenceTrack: preset.reference_track_path } : {}),
      },
    },
    warnings,
    uiEffects,
  };
}
