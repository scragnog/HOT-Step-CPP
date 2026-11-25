import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { muscriptorRoots, resolveModelDir, targetRoot, migrateLegacyModels, isCompleteModelDir } from './muscriptorPaths.js';
import { locateData } from '../dataroot/locate.js';
import { dataLayout, layoutDefaults, migrateEnvText } from '../dataroot/paths.js';

function fixture() {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'hs-musc-'));
  const models = path.join(base, 'models');
  const library = path.join(base, 'library');
  return { base, models, library, roots: muscriptorRoots(undefined, models, library) };
}

function put(dir: string, files: Record<string, string>) {
  fs.mkdirSync(dir, { recursive: true });
  for (const [n, c] of Object.entries(files)) fs.writeFileSync(path.join(dir, n), c);
}
const FULL = { 'model.safetensors': 'weights', 'config.json': '{}' };

test('dettato folders are read from one root, never from the pre-#228 legacy one', () => {
  const f = fixture();
  put(path.join(f.library, 'models', 'muscriptor', 'dettato'), FULL);
  put(path.join(f.models, 'muscriptor', 'dettato-grouper'), FULL);
  assert.equal(resolveModelDir(f.roots, 'dettato').complete, false);
  assert.equal(resolveModelDir(f.roots, 'dettato').dir, path.join(f.models, 'muscriptor', 'dettato'));
  assert.equal(resolveModelDir(f.roots, 'dettato-grouper').complete, true);
});

test('shared-only: read from <models>/muscriptor', () => {
  const f = fixture();
  put(path.join(f.models, 'muscriptor', 'small'), FULL);
  assert.deepEqual(resolveModelDir(f.roots, 'small'), { dir: path.join(f.models, 'muscriptor', 'small'), source: 'shared', complete: true });
  assert.equal(targetRoot(f.roots), path.join(f.models, 'muscriptor'));
});

test('legacy-only: a complete old download is reused in place', () => {
  const f = fixture();
  put(path.join(f.library, 'models', 'muscriptor', 'beat-this'), FULL);
  const r = resolveModelDir(f.roots, 'beat-this');
  assert.equal(r.source, 'legacy');
  assert.equal(r.dir, path.join(f.library, 'models', 'muscriptor', 'beat-this'));
  assert.equal(r.complete, true);
});

test('nothing anywhere: shared is the destination, not complete', () => {
  const f = fixture();
  assert.deepEqual(resolveModelDir(f.roots, 'large'), { dir: path.join(f.models, 'muscriptor', 'large'), source: 'shared', complete: false });
});

test('partial downloads never count as complete', () => {
  const f = fixture();
  put(path.join(f.models, 'muscriptor', 'medium'), { 'config.json': '{}', 'model.safetensors.part': 'half' });
  assert.equal(isCompleteModelDir(path.join(f.models, 'muscriptor', 'medium')), false);
  // a partial shared copy falls through to a complete legacy one
  put(path.join(f.library, 'models', 'muscriptor', 'medium'), FULL);
  assert.equal(resolveModelDir(f.roots, 'medium').source, 'legacy');
  // a partial legacy copy is not reused
  put(path.join(f.library, 'models', 'muscriptor', 'large'), { 'model.safetensors': 'w' });
  assert.equal(resolveModelDir(f.roots, 'large').complete, false);
  assert.equal(resolveModelDir(f.roots, 'large').source, 'shared');
});

test('both complete: shared wins', () => {
  const f = fixture();
  put(path.join(f.models, 'muscriptor', 'small'), FULL);
  put(path.join(f.library, 'models', 'muscriptor', 'small'), { ...FULL, 'model.safetensors': 'other' });
  assert.equal(resolveModelDir(f.roots, 'small').source, 'shared');
});

test('override is authoritative, raw value kept, even when empty', () => {
  const f = fixture();
  put(path.join(f.models, 'muscriptor', 'small'), FULL);
  put(path.join(f.library, 'models', 'muscriptor', 'small'), FULL);
  const roots = muscriptorRoots('rel/musc', f.models, f.library);
  assert.equal(targetRoot(roots), 'rel/musc');
  assert.deepEqual(resolveModelDir(roots, 'small'), { dir: path.join('rel/musc', 'small'), source: 'override', complete: false });
});

test('moved roots: resolution follows the roots passed in', () => {
  const f = fixture();
  put(path.join(f.models, 'muscriptor', 'small'), FULL);
  const moved = path.join(f.base, 'moved');
  fs.mkdirSync(moved);
  fs.renameSync(f.models, path.join(moved, 'models'));
  const roots = muscriptorRoots(undefined, path.join(moved, 'models'), path.join(moved, 'library'));
  assert.deepEqual(resolveModelDir(roots, 'small'), { dir: path.join(moved, 'models', 'muscriptor', 'small'), source: 'shared', complete: true });
  assert.equal(resolveModelDir(f.roots, 'small').complete, false);
});

test('migration copies, verifies and keeps the original by default', async () => {
  const f = fixture();
  const legacy = path.join(f.library, 'models', 'muscriptor', 'small');
  put(legacy, FULL);
  const [r] = await migrateLegacyModels(f.roots, ['small']);
  assert.equal(r.outcome, 'migrated');
  assert.equal(fs.readFileSync(path.join(f.models, 'muscriptor', 'small', 'model.safetensors'), 'utf8'), 'weights');
  assert.ok(isCompleteModelDir(legacy));
  assert.equal(resolveModelDir(f.roots, 'small').source, 'shared');
  assert.deepEqual(fs.readdirSync(path.join(f.models, 'muscriptor', 'small')).sort(), ['config.json', 'model.safetensors']);
  // again: identical, and removeLegacy now deletes the verified original
  const [again] = await migrateLegacyModels(f.roots, ['small'], { removeLegacy: true });
  assert.equal(again.outcome, 'identical');
  assert.equal(isCompleteModelDir(legacy), false);
});

test('migration never overwrites a conflicting file', async () => {
  const f = fixture();
  put(path.join(f.library, 'models', 'muscriptor', 'large'), FULL);
  put(path.join(f.models, 'muscriptor', 'large'), { 'config.json': '{"different":1}' });
  const [r] = await migrateLegacyModels(f.roots, ['large'], { removeLegacy: true });
  assert.equal(r.outcome, 'conflict');
  assert.equal(fs.readFileSync(path.join(f.models, 'muscriptor', 'large', 'config.json'), 'utf8'), '{"different":1}');
  assert.ok(isCompleteModelDir(path.join(f.library, 'models', 'muscriptor', 'large')), 'legacy kept on conflict');
  assert.equal(fs.existsSync(path.join(f.models, 'muscriptor', 'large', 'model.safetensors')), false, 'no weights published beside the conflict');
  assert.equal(resolveModelDir(f.roots, 'large').source, 'legacy');
});

test('migration with shared and legacy as one folder deletes nothing', async () => {
  const f = fixture();
  // ACESTEPCPP_MODELS=<library>/models: both roots are <library>/models/muscriptor
  const roots = muscriptorRoots(undefined, path.join(f.library, 'models'), f.library);
  put(path.join(roots.legacy, 'small'), FULL);
  const [r] = await migrateLegacyModels(roots, ['small'], { removeLegacy: true });
  assert.equal(r.outcome, 'identical');
  assert.ok(isCompleteModelDir(path.join(roots.shared, 'small')));
});

test('migration: partial legacy, no legacy and override are left alone', async () => {
  const f = fixture();
  put(path.join(f.library, 'models', 'muscriptor', 'medium'), { 'model.safetensors': 'w' });
  put(path.join(f.models, 'muscriptor', 'small'), { 'model.safetensors.part': 'x' });
  put(path.join(f.library, 'models', 'muscriptor', 'small'), FULL);
  const res = await migrateLegacyModels(f.roots, ['medium', 'large', 'small']);
  assert.deepEqual(res.map(r => r.outcome), ['none', 'none', 'migrated']);
  assert.equal(fs.existsSync(path.join(f.models, 'muscriptor', 'medium')), false);
  const over = await migrateLegacyModels(muscriptorRoots(path.join(f.base, 'o'), f.models, f.library), ['small']);
  assert.equal(over[0].outcome, 'override');
});

test('inspector and transfer defaults match the runtime roots', () => {
  const f = fixture();
  const root = path.join(f.base, 'data');
  const l = dataLayout(root);
  const runtime = muscriptorRoots(undefined, l.models, l.library);
  assert.equal(layoutDefaults(l).MUSCRIPTOR_MODELS_DIR, targetRoot(runtime));

  const env = { HOT_STEP_ROOT: path.join(f.base, 'app'), HOT_STEP_PROFILE_DIR: path.join(f.base, 'prof'), HOT_STEP_DATA_ROOT: root } as NodeJS.ProcessEnv;
  const c = locateData({ env }).children!;
  assert.equal(c.muscriptorModels, targetRoot(runtime));
  assert.equal(c.muscriptorLegacyModels, runtime.legacy);

  const over = locateData({ env: { ...env, MUSCRIPTOR_MODELS_DIR: path.join(f.base, 'ext') } }).children!;
  assert.equal(over.muscriptorModels, path.join(f.base, 'ext'));
  assert.equal(over.muscriptorLegacyModels, null);

  // A migrated .env drops the new default and keeps an explicit legacy path.
  const text = `MUSCRIPTOR_MODELS_DIR=${path.join(l.models, 'muscriptor')}\n`;
  assert.equal(migrateEnvText(text, root, [], l, null).trim(), '');
  const legacy = `MUSCRIPTOR_MODELS_DIR=${runtime.legacy}`;
  assert.equal(migrateEnvText(legacy, root, [], l, null), legacy);
});
