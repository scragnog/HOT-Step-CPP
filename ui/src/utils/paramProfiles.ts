// paramProfiles.ts — collect/apply the full generation-parameter state
//
// One shared implementation behind three features:
//   • Preset export/import (JSON file, GlobalParamBar)
//   • Parameter profiles (server-stored, ProfilesModal)
//   • Any future "restore these exact settings" flow
//
// Format v2 is a RAW snapshot of the globalParamsStore fields (exactly as
// set in the UI) plus the CreatePanel content fields. This differs from
// getGlobalParams(), which is a *request assembler*: it renames fields
// (adapter → loraPath), rescales values (dcwScaler) and omits knobs whose
// feature is currently disabled — fine for a request, lossy for a snapshot.
// v1 files (the old export, shaped like getGlobalParams() output) are still
// accepted via a small reverse-mapping in applyProfileData().

import { useGlobalParamsStore } from '../stores/globalParamsStore';
import { writePersistedState } from '../hooks/usePersistedState';

// ── Content fields (CreatePanel state, persisted via usePersistedState) ──
// profile key → localStorage key
const CONTENT_KEYS: Record<string, string> = {
  caption: 'hs-caption',
  lyrics: 'hs-lyrics',
  negativePrompt: 'hs-negative-prompt',
  instrumental: 'hs-instrumental',
  bpm: 'hs-bpm',
  duration: 'hs-duration',
  keyScale: 'hs-keyScale',
  timeSignature: 'hs-timeSignature',
  vocalLanguage: 'hs-vocalLanguage',
  loraTrigger: 'hs-lora-trigger',
  beatIntro: 'hs-beat-intro',
  introBars: 'hs-intro-bars',
};

// ── Store fields captured in a profile, grouped by domain ──
// Everything that affects generation. Deliberately excluded: adaptersOpen
// (accordion UI state) and adapterFolder (browse location preference).
// localStorage key is `hs-<field>` unless overridden in FIELD_KEY_OVERRIDES.
// This grouping is the single source of truth for collect/apply AND the
// profile inspector, so a field added here shows up in every path at once.
export const PARAM_GROUPS: { title: string; fields: string[] }[] = [
  {
    title: 'Models',
    fields: ['ditModel', 'lmModel', 'vaeModel', 'embeddingModel', 'useOrtVae'],
  },
  {
    title: 'Adapters',
    fields: [
      'adapter', 'adapterScale', 'adapterStack', 'adapterStackMode', 'adapterStackBudget',
      'adapterSectionAlignAt', 'adapterSectionIsolation', 'adapterMode', 'adapterRuntimeQuant',
      'adapterMergeLowVram', 'adapterGroupScales', 'rebaseSource', 'rebaseBeta', 'advancedAdapters',
    ],
  },
  {
    title: 'DiT Sampling',
    fields: [
      'inferenceSteps', 'guidanceScale', 'cfgCutoffRatio', 'lmCfgCutoffRatio', 'cacheRatio',
      'shift', 'inferMethod', 'scheduler', 'guidanceMode', 'seed', 'randomSeed',
      'batchSize', 'storkSubsteps', 'beatStability', 'frequencyDamping', 'temporalSmoothing',
      'apgMomentum', 'apgNormThreshold', 'dcwEnabled', 'dcwMode', 'dcwLowScaler', 'dcwHighScaler',
      'latentShift', 'latentRescale', 'customTimesteps',
      'denoiseStrength', 'denoiseSmoothing', 'denoiseMix',
      'lssStrength', 'lssVarThresh', 'lssDcRemove',
      'pluginParams',
    ],
  },
  {
    title: 'LM / Thinking',
    fields: [
      'lmSeed', 'lmSeedFollowsDit', 'skipLm', 'skipLrc', 'useCotCaption',
      'lmTemperature', 'lmCfgScale', 'lmTopK', 'lmTopP', 'lmNegativePrompt', 'lmCodesStrength',
    ],
  },
  {
    title: 'Post-Processing',
    fields: [
      'postProcessingEnabled', 'spectralLifterEnabled', 'slDenoiseStrength', 'slNoiseFloor',
      'slHfMix', 'slTransientBoost', 'slShimmerReduction',
      'masteringEnabled', 'masteringReference', 'timbreReference', 'timbreAudioPath',
      'vocalNaturalizerEnabled', 'gainOffsetDb', 'naturalizeAmount', 'natVibratoRate',
      'natVibratoDepth', 'natFormantStrength', 'natMetallicReduction', 'natQuantizationMask',
      'natTransitionSmooth', 'ppVaeReencode', 'ppVaeBlend', 'ppVaeUseOnnx',
      'postprocessEnabled', 'postprocessPlugin',
      'lufsEnabled', 'lufsPreset', 'lufsTarget',
      'autoTrimEnabled', 'durationBuffer', 'autoTrimFadeMs',
    ],
  },
  {
    title: 'Extras',
    fields: [
      'coverArtEnabled', 'coverArtSubject', 'qualityEvalEnabled', 'qualityEvalTarget',
      'whisperLyricsEnabled', 'whisperModel', 'whisperLanguage', 'whisperBeamSize', 'whisperIsolateVocals',
    ],
  },
];

const PROFILE_FIELDS: string[] = PARAM_GROUPS.flatMap(g => g.fields);

// Store fields whose localStorage key is not simply `hs-<field>`
const FIELD_KEY_OVERRIDES: Record<string, string> = {
  whisperLanguage: 'hs-whisperLang',
  whisperBeamSize: 'hs-whisperBeam',
  whisperIsolateVocals: 'hs-whisperIsolate',
};

function fieldStorageKey(field: string): string {
  return FIELD_KEY_OVERRIDES[field] ?? `hs-${field}`;
}

export type ProfileData = Record<string, unknown>;

function readLS<T>(key: string, fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return v !== null ? JSON.parse(v) : fallback;
  } catch { return fallback; }
}

// ── Collect ──────────────────────────────────────────────────────────────

/** Snapshot every generation parameter + content field, exactly as set. */
export function collectProfileData(): ProfileData {
  const s = useGlobalParamsStore.getState();
  const out: ProfileData = { _format: 'hot-step-preset', _version: 2 };
  for (const [field, lsKey] of Object.entries(CONTENT_KEYS)) {
    out[field] = readLS(lsKey, undefined);
  }
  for (const field of PROFILE_FIELDS) {
    out[field] = s[field];
  }
  return out;
}

// ── Apply ────────────────────────────────────────────────────────────────

// Reverse-map v1 preset files (shaped like getGlobalParams() output) onto
// store field names. Derived-only fields (triggerWord, loraStack) are ignored.
function mapV1(p: ProfileData): ProfileData {
  const out = { ...p };
  if (out.adapter === undefined && typeof out.loraPath === 'string') out.adapter = out.loraPath;
  if (out.adapterScale === undefined && typeof out.loraScale === 'number') out.adapterScale = out.loraScale;
  if (out.lmCodesStrength === undefined && typeof out.audioCoverStrength === 'number') out.lmCodesStrength = out.audioCoverStrength;
  // v1 stored timbreReference as either boolean or the audio path string
  if (typeof out.timbreReference === 'string') {
    out.timbreAudioPath = out.timbreReference;
    out.timbreReference = true;
  }
  // v1 pre-scaled the DCW scalers (dcwScaler = low ×0.05, or high ×0.02 in
  // high mode; double mode additionally exported dcwHighScaler ×0.02) — invert
  if (out.dcwLowScaler === undefined && typeof out.dcwScaler === 'number') {
    if (out.dcwMode === 'high') out.dcwHighScaler = out.dcwScaler / 0.02;
    else {
      out.dcwLowScaler = out.dcwScaler / 0.05;
      if (out.dcwMode === 'double' && typeof out.dcwHighScaler === 'number') {
        out.dcwHighScaler = out.dcwHighScaler / 0.02;
      }
    }
  }
  return out;
}

/**
 * Apply a preset/profile to the live UI — no page reload needed.
 * Content fields go through writePersistedState() so CreatePanel's
 * usePersistedState hooks pick them up in place; store fields are written
 * in one setState plus their hs-* localStorage keys.
 * Unknown keys are ignored, so both v1 and v2 files are safe.
 */
export function applyProfileData(raw: ProfileData): void {
  const p = raw._version === 2 ? raw : mapV1(raw);

  for (const [field, lsKey] of Object.entries(CONTENT_KEYS)) {
    if (p[field] !== undefined) writePersistedState(lsKey, p[field]);
  }

  const partial: Record<string, unknown> = {};
  for (const field of PROFILE_FIELDS) {
    if (p[field] !== undefined) {
      partial[field] = p[field];
      try { localStorage.setItem(fieldStorageKey(field), JSON.stringify(p[field])); } catch { /* full */ }
    }
  }
  useGlobalParamsStore.setState(partial);
}

/** Short human summary of a profile for list rows, e.g. "32s · euler · linear · apg 9". */
export function summarizeProfile(p: ProfileData): string {
  const parts: string[] = [];
  if (typeof p.inferenceSteps === 'number') parts.push(`${p.inferenceSteps}s`);
  if (typeof p.inferMethod === 'string' && p.inferMethod) parts.push(p.inferMethod);
  if (typeof p.scheduler === 'string' && p.scheduler) parts.push(p.scheduler);
  if (typeof p.guidanceMode === 'string' && p.guidanceMode) {
    const g = typeof p.guidanceScale === 'number' ? ` ${p.guidanceScale}` : '';
    parts.push(`${p.guidanceMode}${g}`);
  }
  const dit = typeof p.ditModel === 'string' ? p.ditModel.split(/[\\/]/).pop() : '';
  if (dit) parts.push(dit.replace(/\.(gguf|safetensors)$/i, ''));
  return parts.join(' · ');
}

// ── Inspector ──────────────────────────────────────────────────────────────

/** camelCase field name → "Camel Case" label for display. */
function prettifyKey(key: string): string {
  return key
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/^./, c => c.toUpperCase())
    .replace(/\bLm\b/g, 'LM')
    .replace(/\bDit\b/g, 'DiT')
    .replace(/\bDcw\b/g, 'DCW')
    .replace(/\bApg\b/g, 'APG')
    .replace(/\bLss\b/g, 'LSS')
    .replace(/\bVae\b/g, 'VAE')
    .replace(/\bLufs\b/g, 'LUFS')
    .replace(/\bCfg\b/g, 'CFG')
    .replace(/\bBpm\b/g, 'BPM')
    .replace(/\bOrt\b/g, 'ORT')
    .replace(/\bCot\b/g, 'CoT');
}

/** Human-readable value for the inspector. */
function formatValue(v: unknown): string {
  if (v === undefined || v === null || v === '') return '—';
  if (typeof v === 'boolean') return v ? 'On' : 'Off';
  if (typeof v === 'number') return String(v);
  if (typeof v === 'string') {
    const base = v.split(/[\\/]/).pop() || v;   // shorten model/adapter paths
    return base.length > 60 ? base.slice(0, 57) + '…' : base;
  }
  try {
    const json = JSON.stringify(v);
    return json.length > 80 ? json.slice(0, 77) + '…' : json;
  } catch { return String(v); }
}

export interface InspectRow { key: string; label: string; value: string }
export interface InspectGroup { title: string; rows: InspectRow[] }

/**
 * Break a profile's data into display groups for the inspector. Only fields
 * actually present in the profile are shown; a leading "Content" group carries
 * the CreatePanel fields (caption/lyrics/etc.).
 */
export function describeProfileGroups(data: ProfileData): InspectGroup[] {
  const groups: InspectGroup[] = [];

  const contentRows: InspectRow[] = [];
  for (const field of Object.keys(CONTENT_KEYS)) {
    if (data[field] !== undefined) {
      contentRows.push({ key: field, label: prettifyKey(field), value: formatValue(data[field]) });
    }
  }
  if (contentRows.length) groups.push({ title: 'Content', rows: contentRows });

  for (const g of PARAM_GROUPS) {
    const rows: InspectRow[] = [];
    for (const field of g.fields) {
      if (data[field] !== undefined) {
        rows.push({ key: field, label: prettifyKey(field), value: formatValue(data[field]) });
      }
    }
    if (rows.length) groups.push({ title: g.title, rows });
  }
  return groups;
}
