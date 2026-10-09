// Fixture coverage for contracts/studioWorkflows.ts: each schema here is the
// one import { ... } from this module hands to its workflow kind (not a
// parallel copy), so these fixtures are what that kind actually accepts or
// rejects on submit.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  instaInputSchema, instaApproveSchema, coverOpenSchema, coverCaptionSchema, coverTranscribeSchema,
  coverRenderSchema, repaintLayerSourceSchema, repaintRenderSchema, layerRenderSchema, lyricBatchInputSchema,
} from './studioWorkflows.js';

const ok = (schema: { safeParse: (v: unknown) => { success: boolean } }, value: unknown) =>
  assert.equal(schema.safeParse(value).success, true, JSON.stringify(value));
const bad = (schema: { safeParse: (v: unknown) => { success: boolean } }, value: unknown) =>
  assert.equal(schema.safeParse(value).success, false, JSON.stringify(value));

const instaBase = {
  caption: 'synthwave', genres: ['synthwave'], lyricMode: 'lyrics' as const,
  subject: '', randomSubject: false, provider: '', model: '',
  vocalLanguage: 'en', thinking: false, engineParams: {}, expectedBackend: 'ace',
  coResident: false, cacheLmCodes: false,
};

test('insta-preview/insta-direct: lyrics-ai requires a provider, and a subject unless random', () => {
  ok(instaInputSchema, instaBase);
  ok(instaInputSchema, { ...instaBase, lyricMode: 'lyrics-ai', provider: 'openai', subject: 'a lighthouse' });
  ok(instaInputSchema, { ...instaBase, lyricMode: 'lyrics-ai', provider: 'openai', randomSubject: true });
  bad(instaInputSchema, { ...instaBase, lyricMode: 'lyrics-ai', provider: '' });
  bad(instaInputSchema, { ...instaBase, lyricMode: 'lyrics-ai', provider: 'openai', subject: '', randomSubject: false });
  bad(instaInputSchema, { ...instaBase, caption: '' });
});

test('insta-preview/insta-direct: model must be present (empty string is fine), systemPrompt may be omitted', () => {
  ok(instaInputSchema, { ...instaBase, model: '' });
  const { model, ...withoutModel } = instaBase;
  bad(instaInputSchema, withoutModel);
});

test('insta-approve: a document reference, revision at least 1', () => {
  ok(instaApproveSchema, { documentId: '123e4567-e89b-12d3-a456-426614174000', revision: 1 });
  bad(instaApproveSchema, { documentId: 'not-a-uuid', revision: 1 });
  bad(instaApproveSchema, { documentId: '123e4567-e89b-12d3-a456-426614174000', revision: 0 });
});

test('cover-open: asset id, cached metadata/analysis optional but shaped', () => {
  ok(coverOpenSchema, { assetId: '123e4567-e89b-12d3-a456-426614174000' });
  ok(coverOpenSchema, { assetId: '123e4567-e89b-12d3-a456-426614174000', cached: {
    metadata: { artist: 'A', title: 'T', album: 'Al', duration: 180 },
    analysis: { bpm: 120, key: 'C major' },
  } });
  bad(coverOpenSchema, { assetId: '123e4567-e89b-12d3-a456-426614174000', cached: { metadata: { artist: 'A' } } });
});

const draftRef = { documentId: '123e4567-e89b-12d3-a456-426614174000', revision: 1 };

test('cover-caption: draft ref plus artist/provider/model/force', () => {
  ok(coverCaptionSchema, { ...draftRef, artistId: 1, provider: 'openai', model: 'gpt', force: false });
  bad(coverCaptionSchema, { ...draftRef, artistId: 0, provider: 'openai', model: 'gpt', force: false });
});

test('cover-transcribe: force defaults false', () => {
  assert.equal(coverTranscribeSchema.parse(draftRef).force, false);
  ok(coverTranscribeSchema, { ...draftRef, force: true });
});

const coverControls = {
  bpmOverride: null, bpmCorrection: 1, keyOverride: null, tempoScale: 1,
  pitchShift: 0, noFsq: false, audioCoverStrength: 1, coverNoiseStrength: 0,
  coverNoiseMethod: '', vocalLanguage: 'en', sourceLatentUrl: '', timbreOverridePath: '',
  presetAdapterPath: '', presetReferencePath: '', triggerUseFilename: false, triggerPlacement: '',
  voices: 'vocal' as const, keepChords: false, tempoMode: 'free' as const, coverBpm: 120,
  keyShift: 0, cfgScale: 1, lmAdapterAr: '', lmAdapterNar: '', pairMode: 'base' as const,
  yue2Pick: {}, captionMode: 'auto', captionTracks: [],
};
const coverRenderBase = {
  ...draftRef, expectedBackend: 'ace' as const, engineParams: {}, title: 'Cover', artistName: 'Src',
  targetArtistName: 'Dst', lyrics: 'la la', caption: 'synthwave', instrumental: false,
  lyricsSource: null, scoreSource: null, settings: {}, controls: coverControls,
};

test('cover-render: ace and yue2 both validate; pitchShift/keyShift stay in range', () => {
  ok(coverRenderSchema, coverRenderBase);
  ok(coverRenderSchema, { ...coverRenderBase, expectedBackend: 'yue2' });
  bad(coverRenderSchema, { ...coverRenderBase, controls: { ...coverControls, pitchShift: 13 } });
  bad(coverRenderSchema, { ...coverRenderBase, expectedBackend: 'suno' });
});

test('repaint/layer source: asset or song, both pinned by expectedUrl', () => {
  ok(repaintLayerSourceSchema, { kind: 'asset', id: '123e4567-e89b-12d3-a456-426614174000', expectedUrl: '/audio/a.wav' });
  ok(repaintLayerSourceSchema, { kind: 'song', id: 'song-1', expectedUrl: '/audio/a.wav' });
  bad(repaintLayerSourceSchema, { kind: 'asset', id: 'not-a-uuid', expectedUrl: '/audio/a.wav' });
  bad(repaintLayerSourceSchema, { kind: 'other', id: 'x', expectedUrl: '/audio/a.wav' });
});

const source = { kind: 'asset' as const, id: '123e4567-e89b-12d3-a456-426614174000', expectedUrl: '/audio/a.wav' };

test('repaint-render: region bounds and crossfade must be non-negative', () => {
  ok(repaintRenderSchema, { source, expectedBackend: 'ace', engineParams: {},
    regionStart: 0, regionEnd: 10, lyrics: '', styleCaption: '', sourceName: 'Take 1',
    repaintMode: 'balanced', crossfadeFrames: 0 });
  bad(repaintRenderSchema, { source, expectedBackend: 'ace', engineParams: {},
    regionStart: -1, regionEnd: 10, lyrics: '', styleCaption: '', sourceName: 'Take 1',
    repaintMode: 'balanced', crossfadeFrames: 0 });
});

test('layer-render: buildModel must be a plain Base DiT checkpoint', () => {
  ok(layerRenderSchema, { source, expectedBackend: 'ace', engineParams: {},
    trackName: 'vocals', buildModel: 'acestep-v15-base-foo', caption: '' });
  ok(layerRenderSchema, { source, expectedBackend: 'ace', engineParams: {},
    trackName: 'drums', buildModel: 'acestep-v15-xl-base-foo', caption: '' });
  bad(layerRenderSchema, { source, expectedBackend: 'ace', engineParams: {},
    trackName: 'drums', buildModel: 'acestep-v15-lora-foo', caption: '' });
  bad(layerRenderSchema, { source, expectedBackend: 'ace', engineParams: {},
    trackName: 'organ', buildModel: 'acestep-v15-base-foo', caption: '' });
});

test('lyric-batch: 1-200 items, mixed discriminated-union types', () => {
  ok(lyricBatchInputSchema, { items: [{ type: 'fetch', artist: 'Queen', maxSongs: 10 }] });
  ok(lyricBatchInputSchema, { items: [
    { type: 'profile', provider: 'openai', sourceId: 1, sourceRevision: 'r1', artist: 'Queen', songs: [] },
    { type: 'preflight-error', error: 'boom' },
  ] });
  bad(lyricBatchInputSchema, { items: [] });
  bad(lyricBatchInputSchema, { items: [{ type: 'fetch', artist: '', maxSongs: 10 }] });
});
