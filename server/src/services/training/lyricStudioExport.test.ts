import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import type { TrainingDatasetRow, TrainingSample } from './types.js';

test('instrumental and mixed exports preserve captions and merge linked adapters', async () => {
  const root = fs.mkdtempSync(path.resolve('.test-instrumental-export-'));
  process.env.DATA_DIR = root;
  process.env.TRAINING_DIR = path.join(root, 'training');
  const { config } = await import('../../config.js');
  config.lireek.geniusAccessToken = '';
  const database = await import('../../db/database.js');
  const rows = await import('../../db/lireekDb.js');
  const exp = await import('./lyricStudioExport.js');
  database.initDb();
  try {
    const ds = { id: 'fixture-export', slug: 'fixture-export', name: 'Fixture', sourceDir: root,
      defaultArtist: 'Fixture Artist', defaultAlbum: 'Fixture Album' } as TrainingDatasetRow;
    const sample = (id: string, instrumental: boolean, lyrics = '') => ({
      sampleId: id, filename: `${id}.wav`, relPath: `${id}.wav`, audioPath: path.join(root, `${id}.wav`),
      tagArtist: 'Fixture Artist', tagAlbum: 'Fixture Album', tagTitle: id,
      isInstrumental: instrumental, lyrics, caption: 'Quiet piano', genre: 'ambient', bpm: 80,
      key: 'C minor', signature: '4/4', language: '', duration: 180, excluded: false, fileMissing: false,
    }) as TrainingSample;
    const inst = sample('instrumental', true, 'stale text must not become lyrics');
    fs.writeFileSync(path.join(root, 'instrumental.mm3.txt'), 'MM3 piano caption');
    fs.writeFileSync(path.join(root, 'instrumental.yue2.txt'), 'YuE2 piano caption');
    exp.pointMm3PresetsAt(ds, 'fixture-export-run/ckpt-100/adapter_model.safetensors');
    const preview = await exp.previewLyricStudioExport(ds, [inst]);
    assert.equal(preview.songs.length, 1);
    assert.equal(preview.skippedInstrumental, 0);
    assert.equal(preview.songs[0].instrumental, true);
    const input = { artist: 'Fixture Artist', album: 'Fixture Album' };
    const result = await exp.commitLyricStudioExport(ds, [inst], input);
    const stored = rows.getLyricsSet(result.lyricsSetId)!.songs[0];
    assert.deepEqual([stored.title, stored.album], ['instrumental', 'Fixture Album']);
    assert.deepEqual([stored.lyrics, stored.instrumental, stored.caption, stored.mm3Caption, stored.yue2Caption,
      stored.bpm, stored.key, stored.duration], ['', true, 'Quiet piano', 'MM3 piano caption', 'YuE2 piano caption', 80, 'C minor', 180]);
    assert.equal(rows.getPreset(result.lyricsSetId)!.mm3_adapter_path,
      'fixture-export-run/ckpt-100/adapter_model.safetensors');
    const express = (await import('express')).default;
    const { registerLlmRoutes } = await import('../../routes/lireek/llmRoutes.js');
    const app = express();
    app.use(express.json());
    const router = express.Router();
    registerLlmRoutes(router);
    app.use(router);
    const http = app.listen(0, '127.0.0.1');
    await new Promise<void>(resolve => http.once('listening', resolve));
    try {
      const address = http.address() as { port: number };
      for (const endpoint of ['build-profile', 'build-profile-stream']) {
        const response = await fetch(`http://127.0.0.1:${address.port}/lyrics-sets/${result.lyricsSetId}/${endpoint}`,
          { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ provider_name: 'must-not-be-called' }) });
        assert.equal(response.status, 200);
        const profile = endpoint === 'build-profile'
          ? await response.json()
          : JSON.parse((await response.text()).match(/event: complete\ndata: ([^\n]+)/)![1]);
        assert.equal(profile.profile_data.instrumental, true);
        assert.deepEqual(profile.profile_data.audio_enrichment.captionExamples, ['Quiet piano']);
        assert.deepEqual(profile.profile_data.audio_enrichment.mm3CaptionExamples, ['MM3 piano caption']);
        assert.deepEqual(profile.profile_data.audio_enrichment.yue2CaptionExamples, ['YuE2 piano caption']);
        assert.equal(profile.profile_data.representative_excerpts, undefined);
      }
    } finally { await new Promise<void>((resolve, reject) => http.close(err => err ? reject(err) : resolve())); }
    const { captureLyricItems } = await import('../lireek/lyricWorkflow.js');
    const batch = captureLyricItems([{ type: 'profile', targetId: result.lyricsSetId, provider: 'must-not-be-called' }]);
    assert.equal(batch.items[0].type, 'profile');
    const reference = path.join(root, 'reference.wav');
    fs.writeFileSync(reference, 'fixture');
    rows.upsertPreset(result.lyricsSetId, { mm3AdapterPath: 'other/adapter.safetensors', audioCoverStrength: 0.42,
      referenceTrackPath: reference,
      adapterScale: 0.75, adapterGroupScales: { mlp: 0.5 } });
    const noLink = await exp.commitLyricStudioExport(ds, [inst], { ...input, linkAdapters: false });
    assert.equal(noLink.presetUpdated, false);
    assert.equal(rows.getPreset(result.lyricsSetId)!.mm3_adapter_path, 'other/adapter.safetensors');
    fs.writeFileSync(path.join(config.training.dir, 'yue2-linked.json'), JSON.stringify({
      [ds.slug]: { arPath: 'fixture/ar', narPath: 'fixture/nar', at: new Date().toISOString() },
    }));
    const mixed = [inst, sample('vocal', false, 'Real lyrics'), sample('unfinished', false),
      { ...inst, sampleId: 'excluded', excluded: true }];
    const p = await exp.previewLyricStudioExport(ds, mixed);
    assert.deepEqual([p.songs.length, p.skippedNoLyrics, p.skippedExcluded], [2, 1, 1]);
    const again = await exp.commitLyricStudioExport(ds, mixed, input);
    assert.equal(again.lyricsSetId, result.lyricsSetId);
    assert.equal(rows.getLyricsSet(result.lyricsSetId)!.songs.length, 2);
    const preset = rows.getPreset(result.lyricsSetId)!;
    assert.equal(preset.reference_track_path, reference);
    assert.deepEqual([preset.audio_cover_strength, preset.adapter_scale, preset.mm3_adapter_path,
      preset.yue2_ar_adapter_path, preset.yue2_nar_adapter_path],
    [0.42, 0.75, 'fixture-export-run/ckpt-100/adapter_model.safetensors', 'fixture/ar', 'fixture/nar']);
  } finally {
    database.closeDb();
    fs.rmSync(root, { recursive: true, force: true });
  }
});
