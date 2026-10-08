import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

test('capture pins provider, overrides and source revisions; invalid batch items remain visible', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'lyric-workflow-'));
  process.env.DATA_DIR = directory;
  const database = await import('../../db/database.js');
  const rows = await import('../../db/lireekDb.js');
  const { captureLyricItems, lyricBatchInput, runLyricBatch } = await import('./lyricWorkflow.js');
  const { WorkflowJobs } = await import('../workflows/workflowJobs.js');
  try {
    database.initDb();
    const artist = rows.getOrCreateArtist('Fixture Artist');
    const set = rows.saveLyricsSet(artist.id as number, 'Fixture Album', 1,
      [{ title: 'Fixture Song', lyrics: 'A verse and a chorus' }]);
    const p = rows.saveProfile(set.id as number, 'fixture-provider', 'fixture-model',
      { artist: 'Fixture Artist', themes: ['night'], common_subjects: [], rhyme_schemes: [], avg_verse_lines: 4, avg_chorus_lines: 4 });
    const g = rows.saveGeneration({ profileId: p.id as number, provider: 'fixture-provider', model: 'fixture-model', lyrics: 'Original', title: 'Fixture Song', subject: 'stars', key: 'C minor' });
    const batch = captureLyricItems([
      { type: 'profile', targetId: set.id as number, provider: 'fixture-provider', model: 'fixture-model' },
      { type: 'generate', targetId: p.id as number, provider: 'fixture-provider', model: 'fixture-model', count: 2, extraInstructions: 'Short verses', userSubject: 'clouds', noThink: true },
      { type: 'refine', targetId: g.id as number, provider: 'fixture-provider' },
      { type: 'profile', targetId: 999999, provider: 'fixture-provider' },
    ]);
    assert.equal(batch.items.length, 5);
    const generated = batch.items[1];
    assert.equal(generated.type, 'generate');
    if (generated.type !== 'generate') return;
    assert.deepEqual([generated.model, generated.extraInstructions, generated.userSubject, generated.noThink],
      ['fixture-model', 'Short verses', 'clouds', true]);
    assert.equal(generated.history.usedSubjects.includes('stars'), true);
    assert.equal(batch.items[4].type, 'preflight-error');
    let release!: () => void;
    const hold = new Promise<void>(resolve => { release = resolve; });
    const jobs = new WorkflowJobs({ db: database.getDb(), audio: {
      enqueue: () => { throw new Error('unexpected audio'); },
      get: () => { throw new Error('unexpected audio'); },
      cancel: () => { throw new Error('unexpected audio'); },
    } });
    jobs.register({ kind: 'lyric-capture-test', input: lyricBatchInput, run: async ctx => { await hold; return runLyricBatch(ctx); } });
    const submitted = jobs.submit({ kind: 'lyric-capture-test', idempotencyKey: 'revision', input: { items: [generated] } }, 'user').job;
    rows.updateProfileData(p.id as number, { ...p.profile_data, themes: ['changed'] });
    const changed = captureLyricItems([{ type: 'generate', targetId: p.id as number, provider: 'fixture-provider' }]).items[0];
    assert.equal(changed.type, 'generate');
    if (changed.type === 'generate') assert.notEqual(changed.sourceRevision, generated.sourceRevision);
    release();
    await jobs.settled();
    const result = jobs.get(submitted.id).result as { results: Array<{ error?: string }> };
    assert.match(result.results[0].error || '', /Source changed/);
  } finally {
    database.closeDb();
    fs.rmSync(directory, { recursive: true, force: true });
  }
});
