import test from 'node:test';
import assert from 'node:assert/strict';
import express from 'express';
import type { Server } from 'node:http';
import {
  captionForEngine, effectiveYue2Selection, normalizeMm3Selection, pickNearestMm3Track, pickNearestYue2Track,
  resolveMm3Caption, resolveYue2Caption, yue2CaptionAdapterPath, yue2PickAtEnqueue,
} from './captionSource.js';
import {
  composeCreateCaption, createWildcardSeed, estimateDuration, expandWildcards, hasWildcards, resolveDuration,
} from './content.js';
import {
  requestVersion, resolveCreateIntent, resolveWrittenSongIntent, verifyResolvedRequest, ResolveConflictError, type WrittenSongData,
} from './resolveIntent.js';
import { IntentDataError, loadCreateData, loadWrittenSongData, type IntentDataSources } from './loadIntentData.js';
import { createResolveRouter } from '../../../routes/resolve.js';
import { expectedBackendMismatch } from '../../../routes/generate.js';
import type { Mm3SourceTrack, Yue2SourceTrack, WrittenSongIntent, CreateIntent } from '../../../contracts/resolution.js';
import { resolveIntentSchema } from '../../../contracts/resolution.js';

const mm3Tracks: Mm3SourceTrack[] = [
  { title: 'Slow', bpm: 90, caption: 'slow caption' },
  { title: 'Mid A', bpm: 120, caption: 'mid a caption' },
  { title: 'Mid B', bpm: 120, caption: 'mid b caption' },
  { title: 'NoTempo', caption: 'no tempo caption' },
];
const yue2Tracks: Yue2SourceTrack[] = [
  { name: 'a.flac', caption: 'raw a', bpm: '~100', styled: 'styled a' },
  { name: 'b.flac', caption: 'raw b', bpm: '128.5' },
  { name: 'c.flac', caption: 'raw c', bpm: 128.5, styled: 'raw b' },
  { name: 'empty.flac', caption: '', styled: '' },
];

// ── MM3 caption source ───────────────────────────────────────────────────────

test('mm3: custom uses the song caption; an empty one falls through to the album', () => {
  assert.deepEqual(resolveMm3Caption({ bpm: 120, caption_mm3: ' mine ' }, mm3Tracks, { mode: 'custom' }), { caption: 'mine', mode: 'custom' });
  assert.deepEqual(resolveMm3Caption({ bpm: 120, caption_mm3: '  ' }, mm3Tracks, { mode: 'custom' }),
    { caption: 'mid a caption', mode: 'auto', fromTrack: 'Mid A' });
});

test('mm3: a picked track wins; a missing one falls back to auto', () => {
  assert.deepEqual(resolveMm3Caption({ bpm: 60 }, mm3Tracks, { mode: 'track', selectedTitle: 'Mid B' }),
    { caption: 'mid b caption', mode: 'track', fromTrack: 'Mid B' });
  assert.equal(resolveMm3Caption({ bpm: 60 }, mm3Tracks, { mode: 'track', selectedTitle: 'Gone' }).fromTrack, 'Slow');
  assert.equal(resolveMm3Caption({ bpm: 60 }, mm3Tracks, { mode: 'track' }).mode, 'auto');
});

test('mm3: nearest tempo, ties to the earlier track, no tempo means the first track', () => {
  assert.equal(pickNearestMm3Track(mm3Tracks, 119)?.title, 'Mid A');
  assert.equal(pickNearestMm3Track(mm3Tracks, 0)?.title, 'Slow');
  assert.equal(pickNearestMm3Track(mm3Tracks, undefined)?.title, 'Slow');
  assert.equal(pickNearestMm3Track([{ title: 'X', caption: 'x' }, { title: 'Y', caption: 'y' }], 120)?.title, 'X');
  assert.equal(pickNearestMm3Track([], 120), null);
});

test('mm3: no tracks at all lands on the song caption, even an empty one', () => {
  assert.deepEqual(resolveMm3Caption({ caption_mm3: 'own' }, [], { mode: 'auto' }), { caption: 'own', mode: 'custom' });
  assert.deepEqual(resolveMm3Caption({}, [], { mode: 'auto' }), { caption: '', mode: 'custom' });
});

test('mm3: an absent or unrecognised selection reads as auto', () => {
  assert.deepEqual(normalizeMm3Selection(undefined), { mode: 'auto' });
  assert.deepEqual(normalizeMm3Selection({ mode: 'bogus' } as any), { mode: 'auto' });
  assert.deepEqual(normalizeMm3Selection({ mode: 'custom' }), { mode: 'custom' });
});

// ── YuE2 caption source ──────────────────────────────────────────────────────

test('yue2: custom returns the own caption trimmed', () => {
  assert.deepEqual(resolveYue2Caption('  mine ', 128, yue2Tracks, { mode: 'custom' }), { caption: 'mine', mode: 'custom' });
});

test('yue2: a picked track uses styled, then caption; an empty or missing pick falls to auto', () => {
  assert.deepEqual(resolveYue2Caption('m', 0, yue2Tracks, { mode: 'track', selectedName: 'b.flac' }),
    { caption: 'raw b', mode: 'track', fromTrack: 'b.flac' });
  assert.equal(resolveYue2Caption('m', 0, yue2Tracks, { mode: 'track', selectedName: 'a.flac' }).caption, 'styled a');
  assert.equal(resolveYue2Caption('m', 128, yue2Tracks, { mode: 'track', selectedName: 'empty.flac' }).fromTrack, 'b.flac');
  assert.equal(resolveYue2Caption('m', 128, yue2Tracks, { mode: 'track', selectedName: 'gone.flac' }).mode, 'auto');
});

test('yue2: tempo strings parse like the browser; unparsable tempos are skipped', () => {
  assert.equal(pickNearestYue2Track(yue2Tracks, 100)?.name, 'b.flac');   // "~100" is not a number
  assert.equal(pickNearestYue2Track(yue2Tracks, 0)?.name, 'a.flac');
  assert.equal(pickNearestYue2Track([{ name: 'n', caption: 'c', bpm: 'fast' }], 120)?.name, 'n');
});

test('yue2: no usable track lands on the own caption', () => {
  assert.deepEqual(resolveYue2Caption(' mine ', 120, [], { mode: 'auto' }), { caption: 'mine', mode: 'custom' });
  assert.deepEqual(resolveYue2Caption('m', 120, [{ name: 'e', caption: '' }], { mode: 'auto' }), { caption: 'm', mode: 'custom' });
});

test('yue2: default selection is auto only with a dataset and an adapter', () => {
  assert.deepEqual(effectiveYue2Selection('', { mode: 'track', selectedName: 'x' }, true), { mode: 'custom' });
  assert.deepEqual(effectiveYue2Selection('ds', undefined, true), { mode: 'auto' });
  assert.deepEqual(effectiveYue2Selection('ds', undefined, false), { mode: 'custom' });
  assert.deepEqual(effectiveYue2Selection('ds', { mode: 'nope' } as any, true), { mode: 'auto' });
  assert.deepEqual(effectiveYue2Selection('ds', { mode: 'custom' }, true), { mode: 'custom' });
});

test('yue2: adapter path prefers AR, then NAR, then the legacy key', () => {
  assert.equal(yue2CaptionAdapterPath({ lmAdapterAr: ' ar ', lmAdapterNar: 'nar' }), 'ar');
  assert.equal(yue2CaptionAdapterPath({ lmAdapterAr: '', lmAdapterNar: 'nar' }), 'nar');
  assert.equal(yue2CaptionAdapterPath({ lmAdapter: 'old' }), 'old');
  assert.equal(yue2CaptionAdapterPath(undefined), '');
});

test('yue2: the enqueue pick keeps picker dials and takes the preset halves, absent half empty', () => {
  const defaults = { lmAdapterAr: 'pickAr', lmAdapterArScale: 0.8, lmAdapterNarScaleMid: 0.5, lmAdapterFolder: 'f', other: 1 };
  assert.deepEqual(yue2PickAtEnqueue(defaults, null), { lmAdapterAr: 'pickAr', lmAdapterArScale: 0.8, lmAdapterNarScaleMid: 0.5 });
  assert.deepEqual(yue2PickAtEnqueue(defaults, { yue2_ar_adapter_path: ' presetAr ' }),
    { lmAdapterAr: 'presetAr', lmAdapterArScale: 0.8, lmAdapterNarScaleMid: 0.5, lmAdapterNar: '' });
});

// ── Precedence (captionForBackend) ───────────────────────────────────────────

const song = { id: 7, bpm: 128, caption: ' ace caption ', caption_mm3: 'mm3 own', caption_yue2: 'yue2 own' };
const ds = { datasetId: 'ds1', tracks: yue2Tracks, selection: { mode: 'auto' as const } };

test('precedence: ACE always gets the ACE caption, untrimmed', () => {
  assert.deepEqual(captionForEngine(song, 'ace', { yue2: ds }), { caption: ' ace caption ', source: 'song' });
});

test('precedence: MM3 album track, own MM3 caption, then the ACE caption', () => {
  assert.equal(captionForEngine(song, 'minimax-m3', { mm3: { tracks: mm3Tracks, selection: { mode: 'auto' } } }).source, 'album-track');
  assert.deepEqual(captionForEngine(song, 'minimax-m3', { mm3: { tracks: mm3Tracks, selection: { mode: 'custom' } } }),
    { caption: 'mm3 own', mode: 'custom', source: 'song-mm3' });
  assert.deepEqual(captionForEngine({ ...song, caption_mm3: '' }, 'minimax-m3', { mm3: { tracks: [], selection: { mode: 'auto' } } }),
    { caption: ' ace caption ', source: 'song' });
});

test('precedence: YuE2 own caption beats the dataset unless rendering as another album', () => {
  assert.deepEqual(captionForEngine(song, 'yue2', { yue2: ds }), { caption: 'yue2 own', source: 'song-yue2' });
  const asOther = captionForEngine(song, 'yue2', { yue2: ds, renderingAs: true });
  assert.equal(asOther.source, 'dataset-track');
  assert.equal(asOther.caption, 'raw b');
});

test('precedence: Render-as with a custom pick falls back to the own YuE2 caption', () => {
  const r = captionForEngine(song, 'yue2', { yue2: { ...ds, selection: { mode: 'custom' } }, renderingAs: true });
  assert.deepEqual(r, { caption: 'yue2 own', source: 'song-yue2' });
});

test('precedence: no own YuE2 caption takes the dataset; no dataset takes the ACE caption trimmed', () => {
  assert.equal(captionForEngine({ ...song, caption_yue2: null }, 'yue2', { yue2: ds }).caption, 'raw b');
  assert.deepEqual(captionForEngine({ ...song, caption_yue2: '' }, 'yue2', { yue2: { datasetId: '', tracks: [], selection: { mode: 'custom' } } }),
    { caption: 'ace caption', mode: 'custom', source: 'song' });
  assert.deepEqual(captionForEngine({ caption: '', caption_yue2: '' }, 'yue2', {}), { caption: '', source: 'song' });
});

// ── Wildcards, compose helpers, duration ─────────────────────────────────────

test('wildcards: deterministic per seed, innermost first, empty and single options', () => {
  const t = 'a {x|y|z} b {p|{q|r}} c';
  assert.equal(expandWildcards(t, 42), expandWildcards(t, 42));
  assert.ok(!hasWildcards(expandWildcards(t, 42)));
  assert.equal(expandWildcards('{only}', 1), 'only');
  assert.equal(expandWildcards('a{|}b', 1), 'ab');
  assert.equal(expandWildcards('plain', 1), 'plain');
  const seen = new Set(Array.from({ length: 40 }, (_, s) => expandWildcards('{x|y|z}', s)));
  assert.deepEqual([...seen].sort(), ['x', 'y', 'z']);
});

test('wildcard seed: the DiT seed when fixed (zero included), a fresh draw when random', () => {
  assert.deepEqual(createWildcardSeed(false, 0), { seed: 0, seedFrom: 'dit-seed' });
  assert.deepEqual(createWildcardSeed(false, 1234), { seed: 1234, seedFrom: 'dit-seed' });
  const r = createWildcardSeed(true, 1234);
  assert.equal(r.seedFrom, 'random');
  assert.ok(Number.isSafeInteger(r.seed) && r.seed >= 0);
});

test('compose: trigger prepended once, as a whole word only; beat tail appended', () => {
  assert.equal(composeCreateCaption('rock', { loraTrigger: ' trig ' }), 'trig, rock');
  assert.equal(composeCreateCaption('  Trig, rock', { loraTrigger: 'trig' }), '  Trig, rock');
  assert.equal(composeCreateCaption('trig rock', { loraTrigger: 'trig' }), 'trig rock');
  assert.equal(composeCreateCaption('trigger rock', { loraTrigger: 'trig' }), 'trig, trigger rock');
  assert.equal(composeCreateCaption('rock', { beatIntro: true, introBars: 4 }), 'rock, with a clean 4-bar percussive intro and outro for DJ mixing');
  assert.equal(composeCreateCaption('rock', {}), 'rock');
});

test('duration: estimate clamps to 90..360, LLM value wins only when allowed and positive', () => {
  const lyrics = ['[Verse]', ...Array(10).fill('line'), '[Chorus]', ...Array(10).fill('line')].join('\n');
  assert.equal(estimateDuration(lyrics, 120), 90);       // 70 s + 8 s, clamped up
  assert.equal(estimateDuration(Array(200).fill('l').join('\n'), 120), 360);
  assert.equal(estimateDuration('', 120), 0);
  assert.equal(estimateDuration('x', 0), 0);
  assert.deepEqual(resolveDuration(200, lyrics, 120, true), { value: 200, source: 'llm' });
  assert.deepEqual(resolveDuration(200, lyrics, 120, false), { value: 90, source: 'estimate' });
  assert.deepEqual(resolveDuration(0, lyrics, 120, true), { value: 90, source: 'estimate' });
  assert.deepEqual(resolveDuration(null, '', 120, true), { value: 180, source: 'fallback' });
});

// ── Create intent ────────────────────────────────────────────────────────────

const createParams = { caption: 'my words', lyrics: '[Verse]\nla', bpm: 128, duration: 30, seed: 5, randomSeed: false, instrumental: false, source: 'create', taskType: 'text2music' };

test('create: the caption box, compose helpers and the request duration on ACE', () => {
  const intent: CreateIntent = { kind: 'create', params: { ...createParams }, compose: { loraTrigger: 'trig', beatIntro: true, introBars: 2 } };
  const r = resolveCreateIntent(intent, 'ace');
  assert.equal(r.request.caption, 'trig, my words, with a clean 2-bar percussive intro and outro for DJ mixing');
  assert.equal(r.request.duration, 30);
  assert.deepEqual(r.provenance.caption, { source: 'box' });
  assert.deepEqual(r.provenance.trigger, { source: 'compose', words: ['trig'] });
  assert.deepEqual(intent.params, createParams);   // input untouched
});

test('create: MM3 forces auto duration and locks to an album track', () => {
  const r = resolveCreateIntent({ kind: 'create', params: { ...createParams },
    captionSource: { engine: 'minimax-m3', customCaption: 'own mm3', selection: { mode: 'track', selectedTitle: 'Slow' } } },
  'minimax-m3', { mm3Tracks });
  assert.equal(r.request.caption, 'slow caption');
  assert.equal(r.request.duration, -1);
  assert.equal(r.provenance.caption.source, 'album-track');
});

test('create: an MM3 custom choice keeps the box, not the handed-over caption', () => {
  const r = resolveCreateIntent({ kind: 'create', params: { ...createParams },
    captionSource: { engine: 'minimax-m3', customCaption: 'own mm3', selection: { mode: 'custom' } } }, 'minimax-m3', { mm3Tracks });
  assert.equal(r.request.caption, 'my words');
});

test('create: YuE2 dataset lock, default auto only with an adapter', () => {
  const src = { engine: 'yue2' as const, datasetId: 'ds1' };
  assert.equal(resolveCreateIntent({ kind: 'create', params: { ...createParams }, captionSource: { ...src, adapterInForce: true } },
    'yue2', { yue2Tracks }).request.caption, 'raw b');
  assert.equal(resolveCreateIntent({ kind: 'create', params: { ...createParams }, captionSource: { ...src, adapterInForce: false } },
    'yue2', { yue2Tracks }).request.caption, 'my words');
  assert.equal(resolveCreateIntent({ kind: 'create', params: { ...createParams }, captionSource: { ...src, adapterInForce: true } },
    'yue2', { yue2Tracks: [] }).request.caption, 'my words');
});

test('create: a caption source for another engine is ignored with a warning', () => {
  const r = resolveCreateIntent({ kind: 'create', params: { ...createParams },
    captionSource: { engine: 'yue2', datasetId: 'ds1', adapterInForce: true } }, 'ace', { yue2Tracks });
  assert.equal(r.request.caption, 'my words');
  assert.equal(r.warnings.length, 1);
});

test('create: wildcards expand from the DiT seed before the trigger is added; instrumental lyrics', () => {
  const params = { ...createParams, caption: 'a {b|c}', lyrics: '{x|y}', seed: 0 };
  const r = resolveCreateIntent({ kind: 'create', params, compose: { autoExpand: true, loraTrigger: '{t|u}' } }, 'ace');
  assert.equal(r.request.caption, `{t|u}, a ${expandWildcards('{b|c}', 0)}`);
  assert.equal(r.request.lyrics, expandWildcards('{x|y}', 0));
  assert.deepEqual(r.provenance.wildcards, { seed: 0, seedFrom: 'dit-seed', caption: true, lyrics: true });
  const inst = resolveCreateIntent({ kind: 'create', params: { ...createParams, instrumental: true } }, 'ace');
  assert.equal(inst.request.lyrics, '[Instrumental]');
  const off = resolveCreateIntent({ kind: 'create', params: { ...createParams, caption: 'a {b|c}' } }, 'ace');
  assert.equal(off.request.caption, 'a {b|c}');
});

// ── Written song ─────────────────────────────────────────────────────────────

const gen = { id: 7, title: 'T', lyrics: '[Verse]\nla', caption: 'ace cap', caption_mm3: '', caption_yue2: '', bpm: 128, key: 'E Major', duration: 175, subject: 'S' };
const stack3 = [{ path: 'a', scale: 0.25 }, { path: 'b', scale: 0.25 }, { path: 'c', scale: 0.25 }];
const snapshot = { loraStack: stack3, loraPath: 'a', loraScale: 0.5, triggerWord: 'g', triggerWords: ['g1', 'g2'], triggerPlacement: 'append', lmAdapter: 'globalLm', timbreReference: 'dedicated.wav', seed: 1, randomSeed: true };
const base: WrittenSongIntent = { kind: 'written-song', generationId: 7, lyricsSetId: 3, params: snapshot };

test('written: a preset-less album keeps the global stack and triggers, strips the LM adapter', () => {
  const r = resolveWrittenSongIntent(base, 'ace', { gen, preset: null });
  assert.deepEqual(r.request.loraStack, stack3);
  assert.deepEqual(r.request.triggerWords, ['g1', 'g2']);
  assert.equal(r.request.lmAdapter, undefined);
  assert.equal(r.request.caption, 'ace cap');
  assert.equal(r.request.duration, 175);
  assert.equal(r.request.keyScale, 'E major');
  assert.equal(r.request.title, 'T');
  assert.equal(r.request.subject, 'S');
  assert.equal(r.request.instrumental, false);
  assert.equal(r.request.source, 'lyric-studio');
  assert.equal(r.request.timeSignature, '');
  assert.equal(r.request.vocalLanguage, 'en');
  assert.deepEqual(r.uiEffects, []);
  assert.deepEqual(base.params, snapshot);   // input untouched
});

test('written: a preset adapter replaces the stack with one entry and re-derives the filename trigger', () => {
  const preset = { adapter_path: 'C:\\ad\\album_v2.safetensors', lm_adapter_path: 'presetLm', reference_track_path: 'ref.wav' };
  const r = resolveWrittenSongIntent({ ...base, settings: { triggerUseFilename: true, triggerPlacement: 'append', useLmAdapter: true, randomizeTimbreRef: true } },
    'ace', { gen, preset });
  assert.deepEqual(r.request.loraStack, [{ path: preset.adapter_path, scale: 0.5 }]);
  assert.equal(r.request.loraPath, preset.adapter_path);
  assert.equal(r.request.triggerWord, 'album_v2');
  assert.deepEqual(r.request.triggerWords, ['album_v2']);
  assert.equal(r.request.triggerPlacement, 'append');
  assert.equal(r.request.lmAdapter, 'presetLm');
  assert.equal(r.request.masteringReference, 'ref.wav');
  assert.equal(r.request.timbreReference, 'dedicated.wav');
  assert.equal(r.request.randomizeTimbreRef, true);
  assert.deepEqual(r.uiEffects.map(e => e.key), ['hs-adapter', 'hs-lmAdapter', 'hs-masteringReference', 'hs-timbreReference']);
});

test('written: filename trigger off clears the global triggers; default placement is prepend', () => {
  const preset = { adapter_path: '/x/y.safetensors' };
  const off = resolveWrittenSongIntent(base, 'ace', { gen, preset });
  assert.equal(off.request.triggerWord, undefined);
  assert.equal(off.request.triggerWords, undefined);
  assert.equal(off.request.triggerPlacement, undefined);
  assert.equal(off.request.loraStack && (off.request.loraStack as any)[0].scale, 0.5);
  const on = resolveWrittenSongIntent({ ...base, settings: { triggerUseFilename: true } }, 'ace', { gen, preset });
  assert.equal(on.request.triggerPlacement, 'prepend');
  const noScale = resolveWrittenSongIntent({ ...base, params: { ...snapshot, loraScale: undefined } }, 'ace', { gen, preset });
  assert.equal((noScale.request.loraStack as any)[0].scale, 1.0);
});

test('written: preset reference without a dedicated timbre uses the reference itself', () => {
  const r = resolveWrittenSongIntent({ ...base, params: { ...snapshot, timbreReference: '' } }, 'ace', { gen, preset: { reference_track_path: 'r.wav' } });
  assert.equal(r.request.timbreReference, true);
  assert.equal(r.request.randomizeTimbreRef, undefined);
});

test('written: MM3 takes the preset adapter or none, auto duration, album-track caption', () => {
  const r = resolveWrittenSongIntent({ ...base, params: { ...snapshot, mm3LmAdapter: 'global' } }, 'minimax-m3',
    { gen, preset: null, mm3Tracks });
  assert.equal(r.request.mm3LmAdapter, '');
  assert.equal(r.request.duration, -1);
  assert.equal(r.request.caption, 'mid a caption');
  const p = resolveWrittenSongIntent(base, 'minimax-m3', { gen, preset: { mm3_adapter_path: 'mm3.pt' }, mm3Tracks, });
  assert.equal(p.request.mm3LmAdapter, 'mm3.pt');
});

test('written: YuE2 pick from the preset, dataset caption, Render-as', () => {
  const data: WrittenSongData = { gen, preset: { yue2_ar_adapter_path: 'ar', yue2_nar_adapter_path: '' }, yue2Dataset: { datasetId: 'ds1', tracks: yue2Tracks } };
  const r = resolveWrittenSongIntent({ ...base, yue2Defaults: { lmAdapterAr: 'old', lmAdapterArScale: 0.9 } }, 'yue2', data);
  assert.deepEqual(r.request.yue2Pick, { lmAdapterAr: 'ar', lmAdapterArScale: 0.9, lmAdapterNar: '' });
  assert.equal(r.request.caption, 'raw b');
  assert.equal(r.provenance.caption.renderingAs, false);
  // A queued pick wins over the derivation; no adapter in it keeps the own caption.
  const own = resolveWrittenSongIntent({ ...base, yue2Pick: { lmAdapterAr: '' } }, 'yue2', data);
  assert.equal(own.request.caption, 'ace cap');
  // Render-as: the target album's dataset pick leads even over a song YuE2 caption.
  const asOther = resolveWrittenSongIntent({ ...base, sourceLyricsSetId: 9 }, 'yue2', { ...data, gen: { ...gen, caption_yue2: 'own yue2' } });
  assert.equal(asOther.request.caption, 'raw b');
  assert.equal(asOther.provenance.caption.renderingAs, true);
  const sameSet = resolveWrittenSongIntent({ ...base, sourceLyricsSetId: 3 }, 'yue2', { ...data, gen: { ...gen, caption_yue2: 'own yue2' } });
  assert.equal(sameSet.request.caption, 'own yue2');
});

test('written: duration estimate when LLM duration is off; BPM default 120; app flags copied', () => {
  const r = resolveWrittenSongIntent({ ...base, settings: { useLlmDuration: false, app: { coResident: true, generationTimeoutMinutes: 30 } } },
    'ace', { gen: { ...gen, bpm: 0, key: '' }, preset: null });
  assert.equal(r.request.duration, 90);
  assert.equal(r.request.bpm, undefined);
  assert.equal(r.request.keyScale, undefined);
  assert.equal(r.request.coResident, true);
  assert.ok('cacheLmCodes' in r.request && r.request.cacheLmCodes === undefined);
  assert.equal(r.request.generationTimeoutMinutes, 30);
});

test('written: stored time signature and language fill only missing values', () => {
  const r = resolveWrittenSongIntent({ ...base, params: { ...snapshot, timeSignature: '3/4' }, settings: { timeSignature: '4/4', vocalLanguage: 'de' } },
    'ace', { gen, preset: null });
  assert.equal(r.request.timeSignature, '3/4');
  assert.equal(r.request.vocalLanguage, 'de');
});

// ── Version and schema ───────────────────────────────────────────────────────

test('version ignores key order and undefined values, and changes with content', () => {
  const v = requestVersion({ b: 1, a: { y: 2, x: [1, { q: 1, p: 2 }] }, z: undefined });
  assert.equal(v, requestVersion({ a: { x: [1, { p: 2, q: 1 }], y: 2 }, b: 1 }));
  assert.notEqual(v, requestVersion({ a: { x: [1, { p: 2, q: 1 }], y: 2 }, b: 2 }));
  assert.ok(verifyResolvedRequest({ b: 1, a: { y: 2, x: [1, { q: 1, p: 2 }] } }, v));
  assert.ok(!verifyResolvedRequest({ b: 1 }, v));
});

test('schema accepts both intents and rejects malformed ones', () => {
  assert.ok(resolveIntentSchema.safeParse({ kind: 'create', params: {} }).success);
  assert.ok(resolveIntentSchema.safeParse(base).success);
  assert.ok(!resolveIntentSchema.safeParse({ kind: 'written-song', params: {} }).success);
  assert.ok(!resolveIntentSchema.safeParse({ kind: 'create', params: {}, captionSource: { engine: 'yue2' } }).success);
  assert.ok(!resolveIntentSchema.safeParse({ kind: 'other', params: {} }).success);
});

// ── Engine pin ───────────────────────────────────────────────────────────────

test('every resolved body pins expectedBackend to the engine it was resolved for', () => {
  assert.equal(resolveCreateIntent({ kind: 'create', params: { ...createParams } }, 'yue2').request.expectedBackend, 'yue2');
  assert.equal(resolveWrittenSongIntent(base, 'minimax-m3', { gen, preset: null }).request.expectedBackend, 'minimax-m3');
  assert.equal(resolveCreateIntent({ kind: 'create', params: { ...createParams, expectedBackend: 'ace' } }, 'ace').request.expectedBackend, 'ace');
});

test('a conflicting supplied expectedBackend is refused', () => {
  assert.throws(() => resolveCreateIntent({ kind: 'create', params: { ...createParams, expectedBackend: 'ace' } }, 'yue2'), ResolveConflictError);
  assert.throws(() => resolveWrittenSongIntent({ ...base, params: { ...snapshot, expectedBackend: 'yue2' } }, 'ace', { gen, preset: null }), ResolveConflictError);
});

test('preview for an inactive engine, then submit: the generate guard refuses it', () => {
  // Resolved for YuE2 while ACE is active, or after a switch away from YuE2.
  const body = resolveWrittenSongIntent(base, 'yue2', { gen, preset: null, yue2Dataset: { datasetId: 'ds1', tracks: yue2Tracks } }).request;
  assert.deepEqual(expectedBackendMismatch(body, 'ace'), { expectedBackend: 'yue2', activeBackend: 'ace' });
  assert.equal(expectedBackendMismatch(body, 'yue2'), null);
});

// ── Loader source selection ──────────────────────────────────────────────────

function fakeSources() {
  const calls: string[] = [];
  const sources: IntentDataSources = {
    getGeneration: id => { calls.push(`gen:${id}`); return id === 7 ? { ...gen } : null; },
    getPreset: id => { calls.push(`preset:${id}`); return id === 3 ? { adapter_path: 'own.safetensors' } : id === 9 ? { adapter_path: 'target.safetensors' } : null; },
    getLyricsSet: id => { calls.push(`set:${id}`); return { songs: JSON.stringify([{ title: `T${id}`, bpm: 120, mm3Caption: `mm3 of ${id}` }]) }; },
    yue2DatasetCaptions: async by => {
      calls.push(`ds:${JSON.stringify(by)}`);
      return { datasetId: 'ds-x', datasetSlug: '', datasetName: '', tracks: [{ name: 'n', caption: `yue2 via ${JSON.stringify(by)}`, genre: '', bpm: '', key: '', styled: '' }] };
    },
  };
  return { sources, calls };
}

test('written-song loader reads the Render-as target album, never the own one', async () => {
  const { sources, calls } = fakeSources();
  const intent: WrittenSongIntent = { ...base, lyricsSetId: 9, sourceLyricsSetId: 3 };
  const mm3 = await loadWrittenSongData(intent, 'minimax-m3', sources);
  assert.equal(mm3.preset?.adapter_path, 'target.safetensors');
  assert.deepEqual(mm3.mm3Tracks?.map(t => t.caption), ['mm3 of 9']);
  const y = await loadWrittenSongData(intent, 'yue2', sources);
  assert.equal(y.yue2Dataset?.datasetId, 'ds-x');
  assert.deepEqual(calls, ['gen:7', 'preset:9', 'set:9', 'gen:7', 'preset:9', 'ds:{"lyricsSet":9}']);
});

test('written-song loader: ACE reads no caption sources; an unknown song is a 404', async () => {
  const { sources, calls } = fakeSources();
  const d = await loadWrittenSongData(base, 'ace', sources);
  assert.equal(d.mm3Tracks, undefined);
  assert.equal(d.yue2Dataset, undefined);
  assert.deepEqual(calls, ['gen:7', 'preset:3']);
  await assert.rejects(loadWrittenSongData({ ...base, generationId: 99 }, 'ace', sources),
    (e: unknown) => e instanceof IntentDataError && e.status === 404);
});

test('create loader: YuE2 by dataset id, MM3 by lyrics set or handed tracks, nothing for another engine', async () => {
  const { sources, calls } = fakeSources();
  const y = await loadCreateData({ kind: 'create', params: {}, captionSource: { engine: 'yue2', datasetId: 'd1', adapterInForce: true } }, 'yue2', sources);
  assert.equal(y.yue2Tracks?.[0].caption, 'yue2 via {"dataset":"d1"}');
  const m = await loadCreateData({ kind: 'create', params: {}, captionSource: { engine: 'minimax-m3', customCaption: '', lyricsSetId: 4 } }, 'minimax-m3', sources);
  assert.deepEqual(m.mm3Tracks?.map(t => t.caption), ['mm3 of 4']);
  const handed = await loadCreateData({ kind: 'create', params: {},
    captionSource: { engine: 'minimax-m3', customCaption: '', tracks: [{ title: 'h', caption: 'handed' }] } }, 'minimax-m3', sources);
  assert.deepEqual(handed.mm3Tracks?.map(t => t.caption), ['handed']);
  assert.deepEqual(await loadCreateData({ kind: 'create', params: {},
    captionSource: { engine: 'yue2', datasetId: 'd1', adapterInForce: true } }, 'ace', sources), {});
  assert.deepEqual(calls, ['ds:{"dataset":"d1"}', 'set:4']);
});

// ── Route ────────────────────────────────────────────────────────────────────

async function withRouter(run: (post: (body: unknown, token?: string) => Promise<Response>, pathGet: (token?: string) => Promise<Response>) => Promise<void>) {
  const { sources } = fakeSources();
  const app = express();
  app.use(express.json());
  app.use('/api/resolve', createResolveRouter({
    userId: req => (req.headers.authorization === 'Bearer good' ? 'u' : null),
    activeEngine: () => 'ace',
    isEngine: id => ['ace', 'minimax-m3', 'yue2'].includes(id),
    sources,
  }));
  const server: Server = await new Promise(resolve => { const s = app.listen(0, '127.0.0.1', () => resolve(s)); });
  const url = `http://127.0.0.1:${(server.address() as { port: number }).port}/api/resolve/preview`;
  const post = (body: unknown, token?: string) => fetch(url, { method: 'POST',
    headers: { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) }, body: JSON.stringify(body) });
  const pathGet = (token?: string) => fetch(url.replace('/preview', '/path'), {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });
  try { await run(post, pathGet); } finally { await new Promise(resolve => server.close(resolve)); }
}

test('path switch is authenticated and defaults to old unless explicitly resolved', async () => {
  const previous = process.env.GENERATION_INTENT_PATH;
  try {
    await withRouter(async (_post, pathGet) => {
      delete process.env.GENERATION_INTENT_PATH;
      assert.equal((await pathGet()).status, 401);
      assert.deepEqual(await (await pathGet('good')).json(), { path: 'old' });
      process.env.GENERATION_INTENT_PATH = 'resolved';
      assert.deepEqual(await (await pathGet('good')).json(), { path: 'resolved' });
      process.env.GENERATION_INTENT_PATH = 'other';
      assert.deepEqual(await (await pathGet('good')).json(), { path: 'old' });
    });
  } finally {
    if (previous === undefined) delete process.env.GENERATION_INTENT_PATH;
    else process.env.GENERATION_INTENT_PATH = previous;
  }
});

test('preview route: auth, validation, unknown engine, unknown song, conflicting assertion', async () => {
  await withRouter(async post => {
    assert.equal((await post({ kind: 'create', params: {} })).status, 401);
    assert.equal((await post({ kind: 'create', params: {} }, 'stale')).status, 401);
    assert.equal((await post({ kind: 'nope' }, 'good')).status, 400);
    assert.equal((await post({ kind: 'create', engine: 'other', params: {} }, 'good')).status, 400);
    assert.equal((await post({ ...base, generationId: 99 }, 'good')).status, 404);
    assert.equal((await post({ kind: 'create', engine: 'yue2', params: { expectedBackend: 'ace' } }, 'good')).status, 400);
  });
});

test('preview route: an explicit inactive engine resolves from its own sources and pins it', async () => {
  await withRouter(async post => {
    const res = await post({ ...base, engine: 'minimax-m3', lyricsSetId: 9, sourceLyricsSetId: 3 }, 'good');
    assert.equal(res.status, 200);
    const out = await res.json() as { request: Record<string, unknown>; version: string };
    assert.equal(out.request.expectedBackend, 'minimax-m3');
    assert.equal(out.request.caption, 'mm3 of 9');
    assert.equal(out.request.loraPath, 'target.safetensors');
    assert.equal(out.version, requestVersion(out.request));
    // No engine named: the active one, pinned.
    const def = await (await post({ ...base }, 'good')).json() as { request: Record<string, unknown> };
    assert.equal(def.request.expectedBackend, 'ace');
  });
});
