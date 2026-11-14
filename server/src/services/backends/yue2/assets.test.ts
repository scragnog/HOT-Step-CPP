import assert from 'node:assert/strict';
import test from 'node:test';
import { yue2Backend } from './index.js';
import type { GenerationJob } from '../../generation/jobTypes.js';
import type { GenerationContext } from '../types.js';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { yue2AssetReason, yue2ValidateAssets, yue2Preflight, yue2Props, type Yue2Props } from './client.js';

test('a missing selected VAE stops a generation before planning or synth submission', async (t) => {
  const calls: string[] = [];
  t.mock.method(globalThis, 'fetch', async (input: string | URL | Request) => {
    const url = String(input);
    calls.push(url);
    if (url.endsWith('/yue2/props')) return Response.json({
      available: true, files: { lm: { found: true, probe_ok: true }, vae_legacy: { found: false } },
    });
    throw new Error('No engine work should be submitted');
  });
  const job = { params: {}, envelope: { models: { lm: '', vae_variant: 'legacy' } } } as unknown as GenerationJob;
  await assert.rejects(yue2Backend.generate(job, {} as GenerationContext), /Cannot start generation:.*VAE.*legacy.*not installed/);
  assert.equal(calls.length, 1);
  assert.ok(calls[0].endsWith('/yue2/props'));
});

const manifest = (): Yue2Props => ({
  backend: 'yue2', model: 'YuE2', available: true, loaded: false,
  files: { lm: { found: true, probe_ok: true }, vae_standard: { found: true, probe_ok: true },
    vae_legacy: { found: false }, vae_dec2x: { found: true, probe_ok: true } },
});

test('validates the selected VAE, including failed metadata probes and decoder variants', () => {
  const props = manifest();
  assert.doesNotThrow(() => yue2ValidateAssets(props, {}));
  assert.doesNotThrow(() => yue2ValidateAssets(props, { vae_variant: 'dec2x' }));
  assert.throws(() => yue2ValidateAssets(props, { vae_variant: 'legacy' }), /VAE \(legacy\) not installed/);
  // rs-x3's props key is vae_rs_x3 (the engine's JSON key has no hyphen).
  props.files!.vae_rs_x3 = { found: false };
  assert.throws(() => yue2ValidateAssets(props, { vae_variant: 'rs-x3' }), /VAE \(rs-x3\) not installed/);
  props.files!.vae_rs_x3 = { found: true, probe_ok: true };
  assert.doesNotThrow(() => yue2ValidateAssets(props, { vae_variant: 'rs-x3' }));
  props.files!.vae_standard = { found: true, probe_ok: false, error: 'wrong architecture' };
  assert.throws(() => yue2ValidateAssets(props, {}), /metadata probe failed: wrong architecture/);
  assert.throws(() => yue2ValidateAssets(props, { vae_variant: 'unknown' }), /unknown YuE2 VAE variant/);
});

test('plan and semantic probes need their LM but no VAE', () => {
  const props = manifest();
  for (const stage of [{ plan_only: true }, { semantic_only: true }]) {
    assert.doesNotThrow(() => yue2ValidateAssets(props, { ...stage, vae_variant: 'legacy' }));
    assert.throws(() => yue2ValidateAssets({ ...props, files: { lm: { found: false } } }, stage), /base LM not installed/);
  }
  assert.throws(() => yue2ValidateAssets(props, { lm_type: 'convrot' }), /ConvRot LM asset availability/);
  assert.throws(() => yue2ValidateAssets({ ...props, variants: { lm: { available: [], selected: '', requested: '' } } },
    { lm_type: 'q8_0' }), /LM \(q8_0\) not installed/);
  assert.throws(() => yue2ValidateAssets(null, {}), /availability could not be verified/);
});

test('an explicit LM pick checks its own asset rather than a different resident selection', () => {
  const props = manifest();
  props.files!.lm = { found: false };
  props.variants = { lm: { selected: 'bf16', requested: '', available: [{ type: 'q8_0', filename: 'yue2-lm-q8_0.gguf', bytes: 1 }] } };
  assert.doesNotThrow(() => yue2ValidateAssets(props, { lm_type: 'q8_0' }));
  props.variants.lm!.selected = 'q8_0';
  assert.throws(() => yue2ValidateAssets(props, { lm_type: 'q8_0' }), /LM \(q8_0\) not installed/);
});

test('disk checks catch deletion after discovery, empty files and directories', (t) => {
  const root = fs.mkdtempSync(path.resolve('.tmp-yue2-assets-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  fs.mkdirSync(path.join(root, 'yue2'));
  const file = path.join(root, 'yue2', 'decoder.gguf');
  fs.writeFileSync(file, 'fixture');
  const props = { ...manifest(), models_dir: root };
  const found = { found: true, probe_ok: true, name: 'decoder.gguf' };
  assert.equal(yue2AssetReason(props, found), undefined);
  fs.unlinkSync(file);
  assert.match(yue2AssetReason(props, found)!, /Not installed/);
  fs.writeFileSync(file, '');
  assert.match(yue2AssetReason(props, found)!, /empty on disk/);
  fs.unlinkSync(file);
  fs.mkdirSync(file);
  assert.match(yue2AssetReason(props, found)!, /Not installed/);
  fs.rmdirSync(file);
  fs.writeFileSync(path.join(root, 'decoder.gguf'), 'root fixture');
  assert.equal(yue2AssetReason(props, found), undefined);
});

test('an unavailable engine falls back to its manifest but still rejects missing assets', async (t) => {
  let busy = false;
  t.mock.method(globalThis, 'fetch', async () => {
    if (busy) throw new Error('engine busy');
    return Response.json(manifest());
  });
  await yue2Props();
  busy = true;
  await assert.rejects(yue2Preflight({ vae_variant: 'legacy' }), /VAE \(legacy\) not installed/);
});

test('catalogue explains unavailable VAEs and submit rejects an unavailable captured pick', (t) => {
  const root = fs.mkdtempSync(path.resolve('.tmp-yue2-catalogue-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const script = `
    import assert from 'node:assert/strict';
    import { initDb } from './src/db/database.js';
    import { yue2Backend } from './src/services/backends/yue2/index.js';
    globalThis.fetch = async () => Response.json(${JSON.stringify(manifest())});
    initDb();
    const catalogue = await yue2Backend.models();
    assert.ok(catalogue.buckets.vae.includes('legacy'));
    assert.match(catalogue.meta.vae.legacy.disabledReason, /Not installed/);
    assert.equal(catalogue.meta.vae.standard.disabledReason, undefined);
    assert.throws(() => yue2Backend.resolveRequest({caption: 'fixture', yue2Pick: {vae: 'legacy'}}), /VAE \\(legacy\\) not installed/);
  `;
  execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
    timeout: 30_000, stdio: 'pipe',
    env: { ...process.env, DATA_DIR: path.join(root, 'library'), TRAINING_DIR: path.join(root, 'training'), ACESTEPCPP_ADAPTERS: path.join(root, 'adapters') },
  });
});
