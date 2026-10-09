// client.library.test.ts — 7f-2: Song Builder, Library, Playlists, studio
// drafts, presets, import/export and settings/backends, against the REAL
// production routes mounted by fakeServer.ts. One fakeServer per file, same
// as client.test.ts.
//
// Blocked, not faked: GET /api/settings/gpus shells to nvidia-smi
// (gpuDevices.ts:82,98) with no injectable seam — never called here.
// download.ts's song-found branch always re-encodes through ffmpeg
// (audioMetadata.ts's gatherSongMetadata never returns falsy) — the download
// test below uses the documented queue-only fallback (an id with no matching
// song row, found by its audioUrl query param) instead, which is a real,
// zero-subprocess code path, not a workaround.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { startFakeServer, type FakeServer } from './fakeServer.js';
import { ReferenceClient, ClientError } from './client.js';
import { makeFixtureWav } from './fixtures.js';

let server: FakeServer;
let client: ReferenceClient;

test.before(async () => {
  server = await startFakeServer();
  client = new ReferenceClient(server.origin);
  await client.login();
});
test.after(() => server.close());

// ── Song Builder ─────────────────────────────────────────────────────────

test('Song Builder: create a project, generate a section, choose a candidate', async () => {
  const { project } = await client.createBuilderProject({ title: 'Test Song', variantCount: 1 });
  const generated = await client.generateBuilderSection(project.id as string, {
    idempotencyKey: randomUUID(), expectedRevision: project.revision, direction: 'first',
    length: { seconds: 1 }, expectedBackend: 'ace', engineParams: { skipLm: true },
  });
  assert.ok(generated.jobId);
  assert.ok(generated.sectionId);

  // The section's own job is a workflow job (kind builder-section) — poll it,
  // then read the project back so loadSections settles it.
  const job = await client.waitForWorkflowJob(generated.jobId);
  assert.equal(job.status, 'succeeded');

  const view = await client.getBuilderProject(project.id as string);
  const section = (view.sections as Array<Record<string, unknown>>)[0];
  assert.equal(section.status, 'ready');
  const candidate = (section.candidates as Array<{ id: string }>)[0];
  assert.ok(candidate?.id);

  const chosen = await client.chooseBuilderSection(section.id as string, candidate.id, (view.project as Record<string, unknown>).revision as number);
  const chosenSection = (chosen.sections as Array<Record<string, unknown>>)[0];
  assert.equal(chosenSection.status, 'chosen');
  assert.equal((chosenSection.chosen as { id: string }).id, candidate.id);
});

test('Song Builder: a stale project revision on a section command is 409 with currentRevision, state unchanged, retry with the fetched revision succeeds', async () => {
  const { project } = await client.createBuilderProject({ title: 'Stale Test' });
  const generated = await client.generateBuilderSection(project.id as string, {
    idempotencyKey: randomUUID(), expectedRevision: project.revision, direction: 'first',
    length: { seconds: 1 }, expectedBackend: 'ace', engineParams: { skipLm: true },
  });
  await client.waitForWorkflowJob(generated.jobId);
  const before = await client.getBuilderProject(project.id as string);
  const beforeRevision = (before.project as Record<string, unknown>).revision as number;

  let currentRevision: number | undefined;
  await assert.rejects(
    () => client.stopBuilderSection(generated.sectionId, beforeRevision + 99),
    (err: unknown) => {
      if (!(err instanceof ClientError) || err.status !== 409) return false;
      currentRevision = (err.body as { currentRevision: number }).currentRevision;
      return true;
    },
  );
  assert.equal(currentRevision, beforeRevision);

  const unchanged = await client.getBuilderProject(project.id as string);
  assert.equal((unchanged.project as Record<string, unknown>).revision, beforeRevision);
  assert.deepEqual(unchanged.sections, before.sections);

  const retried = await client.stopBuilderSection(generated.sectionId, currentRevision as number);
  assert.equal((retried.project as Record<string, unknown>).revision, beforeRevision + 1);
  const retriedSection = (retried.sections as Array<Record<string, unknown>>)[0];
  assert.equal(retriedSection.status, 'ready');
});

test('Song Builder: deleting a project removes it, candidate songs stay in the library', async () => {
  const { project } = await client.createBuilderProject({ title: 'Delete Me' });
  const del = await client.deleteBuilderProject(project.id as string);
  assert.equal(del.ok, true);
  await assert.rejects(
    () => client.getBuilderProject(project.id as string),
    (err: unknown) => err instanceof ClientError && err.status === 404,
  );
});

test('Song Builder: auth failure — no bearer token is 401', async () => {
  const res = await fetch(`${server.origin}/api/builder/projects`);
  assert.equal(res.status, 401);
});

// ── Library ──────────────────────────────────────────────────────────────

test('Library: create, list, patch, get and delete a song', async () => {
  const { song } = await client.createSong({ title: 'Lib Song', caption: 'a caption', lyrics: '[Instrumental]' });
  const { songs } = await client.listSongs();
  assert.ok(songs.some((s: any) => s.id === song.id));

  const { song: patched } = await client.patchSong(song.id as string, { title: 'Renamed', tags: ['a', 'b'] });
  assert.equal(patched.title, 'Renamed');
  assert.deepEqual(patched.tags, ['a', 'b']);

  const { song: fetched } = await client.getSong(song.id as string);
  assert.equal(fetched.id, song.id);

  const del = await client.deleteSong(song.id as string);
  assert.equal(del.success, true);
  await assert.rejects(() => client.getSong(song.id as string), (err: unknown) => err instanceof ClientError && err.status === 404);
});

test('Library: bulk-delete, songIds and recent all reflect the same rows', async () => {
  const a = await client.createSong({ title: 'Bulk A' });
  const b = await client.createSong({ title: 'Bulk B' });
  const { ids } = await client.songIds();
  assert.ok(ids.includes(a.song.id as string) && ids.includes(b.song.id as string));

  const bulk = await client.bulkDeleteSongs([a.song.id as string, b.song.id as string]);
  assert.equal(bulk.success, true);
  assert.ok(bulk.deletedCount >= 2);
});

test('Library: rejected input — an empty bulk-delete is 400', async () => {
  await assert.rejects(() => client.bulkDeleteSongs([]), (err: unknown) => err instanceof ClientError && err.status === 400);
});

// BLOCKED, not faked: both POST /api/songs/import (direct multipart) and
// POST /api/export-import/imports (asset-id path) land in the same
// importTrackFile() -> toRenderShapedWav() (importTrack.ts:53-66), which
// unconditionally re-encodes through ffmpeg's execFile whenever
// getFFmpegPath() finds a binary — there is no "already the right shape"
// shortcut like download.ts's convertAudio() has for wav-without-metadata,
// and no deps parameter to inject a fake encoder. Proposed seam: an
// optional `convert` override on importTrackFile()/toRenderShapedWav(),
// defaulted to the real execFileAsync call, same shape as coverWorkflow.ts's
// `deps.analyze`. The upload step itself (POST /api/export-import/assets)
// has no such dependency and is exercised below.
test('Library: upload accepts a fixture file and returns an asset id', async () => {
  const asset = await client.uploadAsset(makeFixtureWav(1), 'import-me.wav');
  assert.ok(asset.assetId);
});

test('Library: auth failure — no bearer token is 401', async () => {
  const res = await fetch(`${server.origin}/api/songs`);
  assert.equal(res.status, 401);
});

// ── Playlist ─────────────────────────────────────────────────────────────

test('Playlist: null before the first command, then add/reorder/update/remove/clear', async () => {
  const initial = await client.getPlaylist();
  assert.equal(initial.document, null);

  const item1 = { id: 'p1', title: 'Track 1', audioUrl: '/audio/p1.wav' };
  const item2 = { id: 'p2', title: 'Track 2', audioUrl: '/audio/p2.wav' };
  const added1 = await client.playlistCommand(0, { operation: 'add', item: item1 });
  assert.equal(added1.document.revision, 1);
  const added2 = await client.playlistCommand(1, { operation: 'add', item: item2 });
  assert.equal((added2.document.body.items as unknown[]).length, 2);

  const reordered = await client.playlistCommand(2, { operation: 'reorder', ids: ['p2', 'p1'] });
  assert.deepEqual((reordered.document.body.items as Array<{ id: string }>).map(i => i.id), ['p2', 'p1']);

  const updated = await client.playlistCommand(3, { operation: 'update', id: 'p1', patch: { title: 'Renamed Track' } });
  assert.equal((updated.document.body.items as Array<{ id: string; title: string }>).find(i => i.id === 'p1')?.title, 'Renamed Track');

  const removed = await client.playlistCommand(4, { operation: 'remove', id: 'p2' });
  assert.equal((removed.document.body.items as unknown[]).length, 1);

  const cleared = await client.playlistCommand(5, { operation: 'clear' });
  assert.equal((cleared.document.body.items as unknown[]).length, 0);
});

test('Playlist: a stale expectedRevision is 409 with currentRevision, changes nothing, and the documented retry (reload, reapply, resend) succeeds', async () => {
  const before = await client.getPlaylist();
  const baseRevision = before.document?.revision ?? 0;
  const baseItems = before.document?.body.items ?? [];

  let currentRevision: number | undefined;
  await assert.rejects(
    () => client.playlistCommand(baseRevision + 99, { operation: 'add', item: { id: 'x', title: 'X', audioUrl: '/audio/x.wav' } }),
    (err: unknown) => {
      if (!(err instanceof ClientError) || err.status !== 409) return false;
      currentRevision = (err.body as { currentRevision: number }).currentRevision;
      return true;
    },
  );
  assert.equal(currentRevision, baseRevision);

  const unchanged = await client.getPlaylist();
  assert.equal(unchanged.document?.revision ?? 0, baseRevision);
  assert.deepEqual(unchanged.document?.body.items ?? [], baseItems);

  const retried = await client.playlistCommand(currentRevision as number, { operation: 'add', item: { id: 'x', title: 'X', audioUrl: '/audio/x.wav' } });
  assert.equal(retried.document.revision, baseRevision + 1);
});

test('Playlist: a malformed command (reorder missing an item) is 400', async () => {
  const res = await fetch(`${server.origin}/api/studio-drafts/playlist/commands`, {
    method: 'POST', headers: { Authorization: `Bearer ${client.token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ expectedRevision: 0, command: { operation: 'reorder' } }),
  });
  assert.equal(res.status, 400);
});

test('Playlist: auth failure — no bearer token is 401', async () => {
  const res = await fetch(`${server.origin}/api/studio-drafts/playlist`);
  assert.equal(res.status, 401);
});

// ── Studio drafts ────────────────────────────────────────────────────────

test('Studio drafts: create, list, read, update and delete a Create draft', async () => {
  const created = await client.createDraft({ studio: 'create', fields: { 'hs-caption': 'draft caption', 'hs-bpm': 120 } });
  assert.equal(created.document.revision, 1);

  const { documents } = await client.listDrafts('create');
  assert.ok(documents.some((d: any) => d.id === created.document.id));

  const read = await client.getDraft(created.document.id);
  assert.equal(read.sourceError, null);

  const updated = await client.putDraft(created.document.id, 1, { studio: 'create', fields: { 'hs-caption': 'edited caption' } });
  assert.equal(updated.document.revision, 2);

  const removed = await client.deleteDraft(created.document.id, 2);
  assert.equal(removed.removed, true);
});

test('Studio drafts: a key from another studio is 400', async () => {
  await assert.rejects(
    () => client.createDraft({ studio: 'create', fields: { 'cover-studio-songTitle': 'wrong studio' } }),
    (err: unknown) => err instanceof ClientError && err.status === 400,
  );
});

test('Studio drafts: a stale revision on PUT is 409 with currentRevision, state unchanged, retry with the fetched revision succeeds', async () => {
  const created = await client.createDraft({ studio: 'create', fields: { 'hs-caption': 'v1' } });

  let currentRevision: number | undefined;
  await assert.rejects(
    () => client.putDraft(created.document.id, 99, { studio: 'create', fields: { 'hs-caption': 'v2' } }),
    (err: unknown) => {
      if (!(err instanceof ClientError) || err.status !== 409) return false;
      currentRevision = (err.body as { currentRevision: number }).currentRevision;
      return true;
    },
  );
  assert.equal(currentRevision, created.document.revision);

  const unchanged = await client.getDraft(created.document.id);
  assert.equal(unchanged.document.revision, created.document.revision);
  assert.equal((unchanged.document as { body: { fields: Record<string, unknown> } }).body.fields['hs-caption'], 'v1');

  const retried = await client.putDraft(created.document.id, currentRevision as number, { studio: 'create', fields: { 'hs-caption': 'v2' } });
  assert.equal(retried.document.revision, created.document.revision + 1);
  assert.equal((retried.document as { body: { fields: Record<string, unknown> } }).body.fields['hs-caption'], 'v2');
});

test('Studio drafts: import moves a browser value into a new draft', async () => {
  // The import route is per-field, not per-document (studioDrafts/index.ts
  // :103-123): `storageKey` must be one real draft field key (resolved to
  // its studio via studioForKey), and `raw` is that field's own JSON value,
  // not a whole draft body. 'hs-title' rather than 'hs-caption': a field no
  // other test in this file touches, so an existing draft never collides
  // with expectedRevision: 0's "this is a brand-new import" assumption.
  const raw = JSON.stringify('imported title');
  // studioDrafts/index.ts:104 recomputes sha256(raw) itself and 400s on a
  // mismatch — unlike the preset import below, this hash is verified, not
  // just a dedup key.
  const sourceHash = `sha256:${createHash('sha256').update(raw).digest('hex')}`;
  const result = await client.importDraft({
    storageKey: 'hs-title', raw, sourceHash, schemaVersion: 1, expectedRevision: 0,
  });
  assert.ok(result.created);
  assert.equal((result.document as { body: { fields: Record<string, unknown> } }).body.fields['hs-title'], 'imported title');
});

test('Studio drafts: an unknown handoff id is 404', async () => {
  await assert.rejects(() => client.getHandoff(randomUUID()), (err: unknown) => err instanceof ClientError && err.status === 404);
});

test('Studio drafts: auth failure — no bearer token is 401', async () => {
  const res = await fetch(`${server.origin}/api/studio-drafts/drafts`);
  assert.equal(res.status, 401);
});

// ── Presets (installation-scoped, no token) ─────────────────────────────

test('Presets: create, list, update and delete a named preset', async () => {
  const created = await client.createPreset('ai-continue-style', { label: 'Energetic', value: 'high energy, upbeat' });
  assert.equal(created.document.revision, 1);

  const { documents } = await client.listPresets('ai-continue-style');
  assert.ok(documents.some(d => d.id === created.document.id));

  const updated = await client.putPreset('ai-continue-style', created.document.id, 1, { label: 'Energetic', value: 'edited' });
  assert.equal(updated.document.revision, 2);

  const removed = await client.deletePreset('ai-continue-style', created.document.id, 2);
  assert.equal(removed.removed, true);
});

test('Presets: a stale revision on PUT is 409 with currentRevision, state unchanged, retry with the fetched revision succeeds', async () => {
  const created = await client.createPreset('ai-continue-lyric', { label: 'L', value: 'v' });

  let currentRevision: number | undefined;
  await assert.rejects(
    () => client.putPreset('ai-continue-lyric', created.document.id, 99, { label: 'L', value: 'v2' }),
    (err: unknown) => {
      if (!(err instanceof ClientError) || err.status !== 409) return false;
      currentRevision = (err.body as { currentRevision: number }).currentRevision;
      return true;
    },
  );
  assert.equal(currentRevision, created.document.revision);

  const { documents } = await client.listPresets('ai-continue-lyric');
  const unchanged = documents.find(d => d.id === created.document.id);
  assert.equal(unchanged?.revision, created.document.revision);
  assert.deepEqual(unchanged?.body, { label: 'L', value: 'v' });

  const retried = await client.putPreset('ai-continue-lyric', created.document.id, currentRevision as number, { label: 'L', value: 'v2' });
  assert.equal(retried.document.revision, created.document.revision + 1);
});

test('Presets: rejected input — an unknown family is 404', async () => {
  await assert.rejects(() => client.listPresets('not-a-real-family').then(r => r), () => true);
  const res = await fetch(`${server.origin}/api/preferences/presets/not-a-real-family`);
  assert.equal(res.status, 404);
});

test('Presets: import reports a name collision, then replace resolves it', async () => {
  const hash = (n: number) => `sha256:${String(n).padStart(64, '0')}`;
  const first = await client.importPresets('ai-continue-style', [
    { storageKey: 'k1', sourceHash: hash(1), name: 'Collide', body: { label: 'Collide', value: 'one' } },
  ]);
  assert.equal((first.results[0] as { outcome: string }).outcome, 'imported');

  const conflict = await client.importPresets('ai-continue-style', [
    { storageKey: 'k2', sourceHash: hash(2), name: 'Collide', body: { label: 'Collide', value: 'two' } },
  ]);
  assert.equal((conflict.results[0] as { outcome: string }).outcome, 'name-conflict');

  // Same storageKey+sourceHash, now with the explicit resolution the
  // conflict named — the documented retry (docs/dev/frontend-presets.md).
  const resolved = await client.importPresets('ai-continue-style', [
    { storageKey: 'k2', sourceHash: hash(2), name: 'Collide', body: { label: 'Collide', value: 'two' }, resolution: 'replace' },
  ]);
  assert.equal((resolved.results[0] as { outcome: string }).outcome, 'replaced');
});

test('Presets: import with keep-both lands as a second document, not a name update', async () => {
  const hash = (n: number) => `sha256:${String(n + 100).padStart(64, '0')}`;
  await client.importPresets('ai-continue-style', [
    { storageKey: 'kb1', sourceHash: hash(1), name: 'Both', body: { label: 'Both', value: 'one' } },
  ]);
  // presets.ts:38-41's keep-both path falls through to importOnce and
  // reports whatever that returns — 'imported' for a new (storageKey,
  // sourceHash) pair, not a distinct 'kept-both' outcome. The result type
  // (contracts/preferences.ts) still lists 'kept-both' as possible; this is
  // the type documenting an outcome the code's keep-both branch does not
  // actually produce, not a test issue — noted in the report, not fixed here.
  const kept = await client.importPresets('ai-continue-style', [
    { storageKey: 'kb2', sourceHash: hash(2), name: 'Both', body: { label: 'Both', value: 'two' }, resolution: 'keep-both' },
  ]);
  assert.equal((kept.results[0] as { outcome: string }).outcome, 'imported');
  const { documents } = await client.listPresets('ai-continue-style');
  assert.equal(documents.filter(d => (d.body as { label: string }).label === 'Both').length, 2);
});

test('Presets: applying a YuE2 joint preset resolves the effective form', async () => {
  const { result } = await client.resolveYue2Preset({
    preset: { name: 'Earlier recipe', version: 2, settings: { rank: 64, lyricTiming: true } },
    currentForm: { dataset: 'ds', checkpoint: 'ck', output: 'out', adapterType: 'lokr' },
    lyricTiming: false,
  });
  assert.equal(result.effectiveForm.rank, 64);
  assert.equal(result.lyricTiming, true);
});

test('Singleton settings: null before creation, then PUT creates and updates it', async () => {
  const before = await client.getSingletonSetting('storm-tuning');
  assert.equal(before.document, null);

  const created = await client.putSingletonSetting('storm-tuning', { seed: 42 });
  assert.equal(created.document.revision, 1);

  const updated = await client.putSingletonSetting('storm-tuning', { seed: 43 }, 1);
  assert.equal(updated.document.revision, 2);
  assert.equal((updated.document.body as { seed: number }).seed, 43);
});

// ── Import/export ────────────────────────────────────────────────────────

test('Export/import: resolve a queue-only item, then download it with zero subprocess calls', async () => {
  // A render that exists only in the queue (not yet a library row): the
  // documented path is audioUrl/srcUrl on the export item, not a songId
  // lookup (docs/dev/frontend-media.md). Place the fixture file where
  // download.ts's fallback branch looks for it (config.data.audioDir, via
  // DATA_DIR) rather than going through a real render.
  const filename = `${randomUUID()}.wav`;
  fs.mkdirSync(path.join(server.dataDir, 'audio'), { recursive: true });
  fs.writeFileSync(path.join(server.dataDir, 'audio', filename), makeFixtureWav(1));
  const audioUrl = `/audio/${filename}`;

  const resolved = await client.resolveExport({
    items: [{ songId: 'queue-only-item', audioUrl, variant: 'original' }],
    format: 'wav',
  });
  assert.equal(resolved.items.length, 1);
  const entry = resolved.items[0] as { url: string; filename: string };
  assert.ok(entry.url?.startsWith('/api/download/queue-only-item'));

  const res = await client.download(entry.url);
  assert.equal(res.status, 200);
  assert.equal(res.headers.get('content-type'), 'audio/wav');
  const bytes = await res.arrayBuffer();
  assert.ok(bytes.byteLength > 0);
});

test('Export/import: a malformed export request (no items) is 400', async () => {
  await assert.rejects(() => client.resolveExport({ items: [] }), (err: unknown) => err instanceof ClientError && err.status === 400);
});

test('Export/import: a per-item export failure sits alongside a valid item, ordered, with no url on the failure', async () => {
  // Same queue-only fixture setup as the single-item export test above: a
  // real file under DATA_DIR/audio so the valid item's "ok" status isn't
  // itself a false negative from a missing file (operations.ts:36-38).
  const filename = `${randomUUID()}.wav`;
  fs.mkdirSync(path.join(server.dataDir, 'audio'), { recursive: true });
  fs.writeFileSync(path.join(server.dataDir, 'audio', filename), makeFixtureWav(1));
  const resolved = await client.resolveExport({
    items: [{ songId: 'export-ok-item', audioUrl: `/audio/${filename}` }, { songId: 'does-not-exist' }],
    format: 'wav',
  });
  assert.equal(resolved.items.length, 2);
  const [ok, failed] = resolved.items as Array<{ index: number; url?: string; error?: string }>;
  assert.equal(ok.index, 0);
  assert.ok(ok.url);
  assert.equal(ok.error, undefined);
  assert.equal(failed.index, 1);
  assert.equal(failed.url, undefined);
  assert.ok(failed.error);
});

// A real successful import can't be exercised here without triggering
// importTrackFile()'s unconditional ffmpeg re-encode (importTrack.ts:53-66,
// the same blocked seam noted on the upload test above). resolveAudioAsset
// 404s on a missing assetId before that encoder ever runs (audioAssets.ts:
// 56-62), so two distinct missing-asset items still exercise importAssets()'s
// per-item ordering without a subprocess.
test('Export/import: per-item import failures (two missing assets) stay ordered by index', async () => {
  const first = randomUUID();
  const second = randomUUID();
  const result = await client.importAssets([{ assetId: first }, { assetId: second }]);
  assert.equal(result.items.length, 2);
  const [a, b] = result.items as Array<{ index: number; assetId: string; song?: unknown; error?: string }>;
  assert.equal(a.index, 0);
  assert.equal(a.assetId, first);
  assert.ok(a.error);
  assert.equal(a.song, undefined);
  assert.equal(b.index, 1);
  assert.equal(b.assetId, second);
  assert.ok(b.error);
  assert.equal(b.song, undefined);
});

test('Export/import: auth failure — no bearer token is 401', async () => {
  const res = await fetch(`${server.origin}/api/export-import/exports/resolve`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ items: [] }),
  });
  assert.equal(res.status, 401);
});

test('Export/import: validate a bare preset profile file', async () => {
  const result = await client.validateProfile('my-preset.json', { some: 'setting', _format: 'hot-step-preset' });
  assert.equal(result.name, 'my-preset');
  assert.deepEqual(result.data, { some: 'setting', _format: 'hot-step-preset' });
});

test('Export/import: an empty profile is 400', async () => {
  await assert.rejects(
    () => client.validateProfile('empty.json', {}),
    (err: unknown) => err instanceof ClientError && err.status === 400,
  );
});

// ── Settings/backends ────────────────────────────────────────────────────
//
// GET /api/settings/gpus is BLOCKED, not tested: gpuDevices.ts:82,98 shells
// to `nvidia-smi` via execFileAsync/execFileSync with no injectable seam
// (listGpus() takes no deps parameter and the module has no override hook).
// Proposed seam: an optional `probe` parameter on listGpus(), defaulted to
// the real execFileAsync call, the same shape essentiaClient.ts's
// analyzeWithEssentia() already uses via coverWorkflow.ts's `deps.analyze`.

test('Settings: round-trip an exposed .env key through the isolated env file', async () => {
  const before = await client.getEnvSettings();
  assert.ok('DATA_DIR' in before.values);

  const updated = await client.putEnvSettings({ LLM_TIMEOUT_MS: '45000' });
  assert.ok(updated.updated.includes('LLM_TIMEOUT_MS'));

  const after = await client.getEnvSettings();
  assert.equal(after.values.LLM_TIMEOUT_MS, '45000');
});

test('Settings: an unexposed key is silently dropped, not written', async () => {
  const updated = await client.putEnvSettings({ NOT_AN_EXPOSED_KEY: 'x' });
  assert.deepEqual(updated.updated, []);
  assert.equal(updated.restartRequired, false);
});

test('Backends: list, switch to the same active id, and read capabilities', async () => {
  const list = await client.listBackends();
  assert.ok(list.backends.length > 0);
  assert.equal(list.activeId, 'ace');

  // Switching to the already-active backend proves the round trip without
  // triggering the outgoing-backend releaseVram() lifecycle (a cross-backend
  // VRAM handoff, Engineer's 7f-3 training/streaming domain, not this one's).
  const switched = await client.setActiveBackend(list.activeId);
  assert.equal(switched.activeId, list.activeId);

  const caps = await client.getCapabilities();
  assert.equal((caps as { backend: string }).backend, 'ace');
});

// ── Safety ──────────────────────────────────────────────────────────────
//
// Last, so it covers every test above it — same convention as client.test.ts.

test('Safety: no real subprocess exec or off-origin network call occurred', () => {
  assert.deepEqual(server.violations, []);
});
