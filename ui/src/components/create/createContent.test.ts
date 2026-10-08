// createContent.test.ts — request parity for resumed and applied Create drafts.
// Runs server code (studio drafts, intent normalization), so use the server's tsx:
//   (cd server && node --import tsx --test ../ui/src/components/create/createContent.test.ts)
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { StudioDrafts } from '../../../../server/src/services/studioDrafts/index.js';
import { WorkflowDocuments } from '../../../../server/src/services/workflows/revisions.js';
import { createTrainingCreateDraft, mirroredGenerationDraft } from '../../../../server/src/services/training/review/auditionDraft.js';
import { resolveGenerationIntent } from '../../../../server/src/services/generation/intent.js';
import type { BackendExtensionParam } from '../../../../server/src/services/backends/types.js';
import type { AuditionPreview, AuditionSideResult } from '../../../../server/src/services/training/types.js';
import { applyCreateDraft, createContentFromDraft, createContentParams, createDraftBackendRefusal } from './createContent';
import { applyTrainingCreateDraft, type TrainingCreateDraftData } from './trainingCreateDraft';
import { createDraftMirror } from '../../services/studioDraftMirror';

// better-sqlite3 is a server dependency; resolve it from there.
const Database = createRequire(new URL('../../../../server/package.json', import.meta.url))('better-sqlite3') as typeof import('better-sqlite3');

// Create's mirrored keys at the form's defaults (CreatePanel).
const DEFAULTS = [
  { studio: 'create', backendId: 'ace-step', fields: {
    'hs-caption': '', 'hs-lyrics': '', 'hs-negative-prompt': '', 'hs-instrumental': false, 'hs-lora-trigger': '',
    'hs-beat-intro': false, 'hs-intro-bars': 2, 'hs-title': '', 'hs-artist': '', 'hs-subject': '', 'hs-bpm': 0,
    'hs-keyScale': '', 'hs-timeSignature': '', 'hs-duration': -1, 'hs-vocalLanguage': 'en', 'hs-vocalGender': '',
    'hs-sourceLatentUrl': '', 'hs-mm3CaptionSources': null, 'hs-yue2CaptionDataset': '',
    'hs-yue2CaptionSource:ds:dataset-1': { mode: 'track', selectedName: 'one' } } },
];

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


// Resumed-draft request parity: every captured Create request in the batch 1
// golden set, saved as a Create draft, resumed, and turned back into request
// content by the form's own mapping, still normalizes to the captured body.
interface GoldenCase { id: string; engine: string; branch: string;
  intent: { contract: 'generation-intent/1'; params: Record<string, unknown>; input: Record<string, unknown> };
  expected: Record<string, unknown>; extensions: BackendExtensionParam[] }
const golden = JSON.parse(readFileSync(new URL('../../../../server/src/services/generation/fixtures/batch1-intent-golden.json', import.meta.url), 'utf8')) as GoldenCase[];
const CONTENT_KEYS = ['caption', 'lyrics', 'instrumental', 'bpm', 'keyScale', 'timeSignature', 'vocalLanguage', 'duration', 'taskType', 'title', 'artist', 'subject'];

test('resumed Create drafts reproduce every captured Create request', async () => {
  const store = new StudioDrafts(new Database(':memory:'));
  const cases = golden.filter(c => c.intent.input.source === 'create');
  assert.ok(cases.length >= 15, `only ${cases.length} Create cases`);
  for (const c of cases) {
    const input = c.intent.input;
    const text = (key: string) => typeof input[key] === 'string' ? input[key] : '';
    const fields = { ...DEFAULTS[0].fields, 'hs-caption': input.caption, 'hs-lyrics': input.lyrics,
      'hs-instrumental': input.instrumental, 'hs-bpm': input.bpm, 'hs-keyScale': input.keyScale,
      'hs-timeSignature': input.timeSignature, // MM3 drafts keep a stale ACE length, as the hidden control does.
      'hs-duration': c.engine === 'minimax-m3' ? 200 : input.duration, 'hs-vocalLanguage': input.vocalLanguage,
      'hs-title': text('title'), 'hs-artist': text('artist'), 'hs-subject': text('subject') };
    const saved = store.drafts.create('u', { studio: 'create', backendId: c.engine, fields }, { origin: 'client' });
    const resumed = store.drafts.get(saved.id, 'u').body;
    // CreatePanel's load path: the mirror's load with CreatePanel's refusal and
    // applier. Another active backend refuses it and writes nothing; its own
    // backend applies it through the persisted keys, read back by the form's
    // mapping with the active backend's rules (not the fixture's engine).
    const loadAs = async (active: string) => {
      const storage = new Map<string, unknown>();
      let error = '';
      const mirror = createDraftMirror({ api: {} as never, pointer: { read: () => null, write: () => {} }, onError: m => { error = m; } });
      const loaded = await mirror.load('t', saved.id, {
        get: async () => ({ document: { ...saved, body: resumed }, sourceError: null }),
        current: () => ({ studio: 'create', fields: {} }), confirm: () => true,
        refuse: draft => createDraftBackendRefusal(draft, active, id => `name:${id}`),
        apply: draft => applyCreateDraft(draft, (key, value) => storage.set(key, value)),
      });
      return { loaded, storage, error };
    };
    for (const other of ['ace', 'yue2', 'minimax-m3'].filter(id => id !== c.engine)) {
      const refused = await loadAs(other);
      assert.equal(refused.loaded, false);
      assert.equal(refused.storage.size, 0);
      assert.equal(refused.error, `This draft was saved for name:${c.engine}. Switch the backend to name:${c.engine} to load it.`);
    }
    const active = c.engine;
    const { loaded, storage } = await loadAs(active);
    assert.equal(loaded, true);
    const content = createContentParams(createContentFromDraft(Object.fromEntries(storage)),
      { mm3Mode: active === 'minimax-m3' }) as Record<string, unknown>;
    for (const key of Object.keys(content)) assert.ok(CONTENT_KEYS.includes(key) && Object.hasOwn(input, key), `${c.id}: unexpected ${key}`);
    const resumedInput = { ...input, ...content };
    assert.deepEqual(resumedInput, input, `${c.id} (${c.branch})`);
    const actual = JSON.parse(JSON.stringify(resolveGenerationIntent({ ...c.intent, input: resumedInput }, c.engine, c.extensions)));
    assert.deepEqual(actual, c.expected, `${c.id} (${c.branch})`);
  }
});

test('a resumed draft composes the trigger and beat intro exactly as the form does', () => {
  const content = createContentParams(createContentFromDraft({ 'hs-caption': 'warm synths', 'hs-lora-trigger': 'trig',
    'hs-beat-intro': true, 'hs-intro-bars': 4, 'hs-duration': 90 }), { mm3Mode: true });
  assert.equal(content.caption, 'trig, warm synths, with a clean 4-bar percussive intro and outro for DJ mixing');
  assert.equal(content.duration, -1);
  assert.equal(createContentParams(createContentFromDraft({ 'hs-caption': 'Trig, typed', 'hs-lora-trigger': 'trig' }),
    { mm3Mode: false }).caption, 'Trig, typed');
});

// Saved-preset application parity: a Training Studio audition saved for
// Create, read back through the drafts adapter and applied by Create's own
// applier, writes exactly the former complete handoff (fixture), and the
// resulting request caption carries no trigger.
const former = JSON.parse(readFileSync(new URL('../../../../server/src/services/training/review/former-audition-drafts.json', import.meta.url), 'utf8')) as Array<{
  side: 'base' | 'adapter'; cell: 'bare' | 'adapter'; content: Record<string, unknown>; params: Record<string, unknown>; settings: Record<string, unknown> }>;

test('applying a saved training handoff writes the former handoff exactly', () => {
  const db = new Database(':memory:');
  const docs = new WorkflowDocuments(db);
  const drafts = new StudioDrafts(db, docs);
  assert.equal(former.length, 4);
  for (const fixture of former) {
    const side = fixture.side === 'base' ? base : adapter;
    const created = createTrainingCreateDraft('u', `apply-${fixture.side}-${fixture.cell}`, mirroredGenerationDraft(preview, side, fixture.cell), docs);
    const read = drafts.handoff('u', created.id).data as unknown as TrainingCreateDraftData;
    const writes: Record<string, unknown> = {};
    const set: Record<string, unknown> = {};
    const state: Record<string, unknown> = {};
    for (const key of Object.keys(fixture.params)) state[`set${key[0].toUpperCase()}${key.slice(1)}`] = (v: unknown) => { set[key] = v; };
    const storage = new Map<string, string>([['ace-settings', JSON.stringify({ cacheLmCodes: true, downloadFormat: 'flac' })]]);
    applyTrainingCreateDraft(read, {
      store: { getState: () => state, setState: partial => Object.assign(set, partial) },
      write: (key, value) => { writes[key] = value; },
      storage: { getItem: k => storage.get(k) ?? null, setItem: (k, v) => { storage.set(k, v); } },
      scopedKey: k => `${k}:ace`,
    });
    const { ['ace-settings']: settings, ...content } = writes;
    assert.deepEqual(content, fixture.content, `${fixture.side}/${fixture.cell}`);
    assert.deepEqual(set, fixture.params, `${fixture.side}/${fixture.cell}`);
    assert.deepEqual(settings, { cacheLmCodes: fixture.settings.cacheLmCodes, downloadFormat: 'flac' });
    const request = createContentParams(createContentFromDraft(content), { mm3Mode: false });
    assert.equal(request.caption, fixture.content['hs-caption']);
  }
});
