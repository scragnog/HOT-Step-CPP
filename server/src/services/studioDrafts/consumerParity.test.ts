// Slice I consumer parity: what Create, Cover and Repaint mirror into their
// drafts, and the training-to-Create handoff read through the drafts adapter.
// Request parity for resumed and applied drafts: ui/src/components/create/createContent.test.ts.
import test from 'node:test';
import assert from 'node:assert/strict';
import Database from 'better-sqlite3';
import { StudioDrafts, validateDraft, withoutStaleSourceResults } from './index.js';
import { WorkflowDocuments } from '../workflows/revisions.js';
import { createTrainingCreateDraft, mirroredGenerationDraft } from '../training/review/auditionDraft.js';
import type { AuditionPreview, AuditionSideResult } from '../training/types.js';
import type { StudioDraftBody } from '../../contracts/studioDrafts.js';

// Each studio's mirrored keys at their UI defaults (CreatePanel, CoverStudio,
// RepaintStudio), so false, zero, null and '' all round-trip.
const DEFAULTS: StudioDraftBody[] = [
  { studio: 'create', backendId: 'ace-step', fields: {
    'hs-caption': '', 'hs-lyrics': '', 'hs-negative-prompt': '', 'hs-instrumental': false, 'hs-lora-trigger': '',
    'hs-beat-intro': false, 'hs-intro-bars': 2, 'hs-title': '', 'hs-artist': '', 'hs-subject': '', 'hs-bpm': 0,
    'hs-keyScale': '', 'hs-timeSignature': '', 'hs-duration': -1, 'hs-vocalLanguage': 'en', 'hs-vocalGender': '',
    'hs-sourceLatentUrl': '', 'hs-mm3CaptionSources': null, 'hs-yue2CaptionDataset': '',
    'hs-yue2CaptionSource:ds:dataset-1': { mode: 'track', selectedName: 'one' } } },
  { studio: 'cover', fields: Object.fromEntries(Object.entries({
    sourceFileName: '', sourceAudioUrl: '', sourceAssetId: '', sourceSongId: '', metadata: null, analysis: null,
    songArtist: '', songTitle: '', lyrics: '', lyricsSource: null, datasetAnalysis: false, selectedArtistId: null,
    selectedPreset: null, artistCaption: '', audioCoverStrength: 0.5, coverNoiseStrength: 0, coverNoiseMethod: '',
    tempoScale: 1, pitchShift: 0, bpmCorrection: 1, bpmOverride: null, keyOverride: null, noFsq: false,
    coverInstrumental: false, sourceLatentUrl: '', coverVocalLanguage: 'en', coverTimbreOverride: '', sepLevel: 1,
  }).map(([k, v]) => [`cover-studio-${k}`, v])) },
  { studio: 'repaint', fields: Object.fromEntries(Object.entries({
    sourceSong: null, sourceAssetId: '', sourceAudioUrl: '', sourceName: '', regionStart: 0, regionEnd: 0,
    lyrics: '', repaintMode: 'balanced', crossfadeFrames: 10, styleCaption: '',
  }).map(([k, v]) => [`hs-repaint-${k}`, v])) },
];

test('each studio mirror passes draft validation and resumes byte-identical', () => {
  const store = new StudioDrafts(new Database(':memory:'));
  for (const body of DEFAULTS) {
    validateDraft(body);
    const saved = store.drafts.create('u', body, { origin: 'client' });
    assert.deepEqual(store.drafts.get(saved.id, 'u').body, body, body.studio);
  }
});

test('a resumed Create draft keeps its caption choices until its source changes', () => {
  const store = new StudioDrafts(new Database(':memory:'));
  const body = structuredClone(DEFAULTS[0]);
  body.fields['hs-mm3CaptionSources'] = { mode: 'track', selectedTitle: 'a', customCaption: 'mine', tracks: [{ title: 'a', caption: 'x' }] };
  const saved = store.drafts.create('u', body, { origin: 'client' });
  const edited = { ...body, fields: { ...body.fields, 'hs-caption': 'edited' } };
  // The PUT route's merge (routes/studioDrafts.ts).
  const kept = store.drafts.update(saved.id, 'u', saved.revision,
    (current: StudioDraftBody) => withoutStaleSourceResults(current, edited), { origin: 'client' });
  assert.deepEqual(kept.body.fields['hs-mm3CaptionSources'], body.fields['hs-mm3CaptionSources']);
  assert.equal(kept.body.fields['hs-caption'], 'edited');
  const moved = { ...edited, fields: { ...edited.fields, 'hs-sourceLatentUrl': '/latents/other.npy' } };
  const reset = store.drafts.update(saved.id, 'u', kept.revision,
    (current: StudioDraftBody) => withoutStaleSourceResults(current, moved), { origin: 'client' });
  assert.equal(Object.hasOwn(reset.body.fields, 'hs-mm3CaptionSources'), false);
  assert.equal(reset.body.fields['hs-caption'], 'edited');
});

const preview = {
  previewId: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', datasetId: 'dataset', kind: 'ab',
  createdAt: '2026-01-01T00:00:00.000Z', seed: 73, caption: 'tagged caption',
  captionInput: 'plain caption', lyrics: 'lines', durationSec: 12, lmModel: 'lm',
  ditModel: 'dit', renderDitModel: 'render dit', vaeModel: 'vae', renderDitAdapter: 'dit adapter',
  renderSteps: 8, bpm: 120, keyscale: 'C', timesignature: '4',
  lmTemperature: 0.7, lmTopP: 0.8, lmCfgScale: 3, lmRepPenalty: 1.2,
} as AuditionPreview;
const base = { slot: 'base', lmAdapter: '', lmAdapterScale: 1, ok: true } as AuditionSideResult;
const adapter = { slot: 'adapter', lmAdapter: 'lm adapter', lmAdapterScale: 0.8, ok: true } as AuditionSideResult;

test('a training handoff reads back unchanged through the drafts adapter, resets and caption asymmetry kept', () => {
  const db = new Database(':memory:');
  const docs = new WorkflowDocuments(db);
  const drafts = new StudioDrafts(db, docs);
  for (const [side, cell, caption] of [[base, 'bare', 'tagged caption'], [adapter, 'adapter', 'plain caption']] as const) {
    const draft = mirroredGenerationDraft(preview, side, cell);
    const created = createTrainingCreateDraft('u', `key-${side.slot}`, draft, docs);
    const read = drafts.handoff('u', created.id).data as unknown as typeof draft;
    assert.deepEqual(read.content, draft.content);
    assert.deepEqual(read.params, draft.params);
    // The LM adapter side gets the untagged input caption; the bare side the tagged one.
    assert.equal(read.content['hs-caption'], caption);
    // The trigger is never carried over; the handoff resets stacking and caching.
    assert.equal(read.content['hs-lora-trigger'], '');
    assert.deepEqual(read.params.adapterStack, []);
    assert.deepEqual(read.params.pluginParams, {});
    assert.equal(read.params.randomSeed, false);
    assert.deepEqual(read.settings, { cacheLmCodes: false });
    assert.equal(read.params.adapter, cell === 'adapter' ? 'dit adapter' : '');
  }
  assert.throws(() => drafts.handoff('other-user', docs.list('u', 'training-audition-create')[0].id));
});

