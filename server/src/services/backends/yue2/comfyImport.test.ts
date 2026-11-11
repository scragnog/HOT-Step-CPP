import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test, { after } from 'node:test';

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-comfy-import-'));
process.env.TRAINING_DIR = path.join(tmp, 'training'); // keep the run index out of the real one
const { importYue2ComfyAdapter } = await import('./comfyImport.js');
after(() => fs.rmSync(tmp, { recursive: true, force: true }));

/** One-layer, rank-1 file; every F32 element is its own index,
 *  so a wrong slice shows up as the wrong numbers. */
function writeFixture(file: string, options: {
  experts?: Array<'diffusion_model' | 'text_encoders' | 'bare'>;
  sites?: Record<string, [number, number]>;
  suffixes?: readonly [string, string];
  key?: (name: string) => string;
  alphas?: Record<string, number>;
  flow?: 'diff' | 'replacement';
} = {}): void {
  const sites = options.sites ?? {
    'self_attn.qkv_proj': [2048, 4096], 'self_attn.o_proj': [2048, 2048],
    'mlp.gate_up_proj': [2048, 12288], 'mlp.down_proj': [6144, 2048],
  };
  const header: Record<string, unknown> = {
    __metadata__: { format: 'pt', ss_tag_frequency: '{"1_mytrig": {"mytrig": 1}}', training_info: '{"step": 7}' },
  };
  const parts: Buffer[] = [];
  let off = 0;
  for (const expert of options.experts ?? ['diffusion_model', 'text_encoders']) {
    const prefix = expert === 'bare' ? '' : `${expert}.model.`;
    for (const [site, [inp, outp]] of Object.entries(sites)) {
      const [down, up] = options.suffixes ?? ['lora_A.weight', 'lora_B.weight'];
      for (const [suffix, shape] of [[down, [1, inp]], [up, [outp, 1]]] as const) {
        const n = shape[0] * shape[1];
        const data = Buffer.from(Float32Array.from({ length: n }, (_, i) => i).buffer);
        const key = `${prefix}layers.0.${site}.${suffix}`;
        header[options.key?.(key) ?? key] = { dtype: 'F32', shape, data_offsets: [off, off + data.length] };
        parts.push(data);
        off += data.length;
      }
      if (options.alphas?.[site] !== undefined) {
        const data = Buffer.alloc(4); data.writeFloatLE(options.alphas[site]);
        const key = `${prefix}layers.0.${site}.alpha`;
        header[options.key?.(key) ?? key] = { dtype: 'F32', shape: [], data_offsets: [off, off + 4] };
        parts.push(data); off += 4;
      }
    }
  }
  if (options.flow) {
    for (const [site, shape, bias] of [['llm2vae', [64, 2048], false], ['llm2vae', [64], true],
      ['vae2llm', [2048, 64], false], ['vae2llm', [2048], true]] as const) {
      const data = Buffer.alloc(shape.reduce((a, b) => a * b, 1) * 4); data.writeFloatLE(0.25);
      const suffix = options.flow === 'diff' ? bias ? 'diff_b' : 'diff' : bias ? 'bias' : 'weight';
      header[`diffusion_model.${site}.${suffix}`] = { dtype: 'F32', shape, data_offsets: [off, off + data.length] };
      parts.push(data); off += data.length;
    }
  }
  const json = Buffer.from(JSON.stringify(header));
  const len = Buffer.alloc(8);
  len.writeBigUInt64LE(BigInt(json.length));
  fs.writeFileSync(file, Buffer.concat([len, json, ...parts]));
}

function read(file: string) {
  const buf = fs.readFileSync(file);
  const n = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + n).toString());
  const first = (name: string) => buf.readFloatLE(8 + n + header[name].data_offsets[0]);
  return { header, first };
}

test('a ComfyUI joint adapter splits into native AR and NAR files', () => {
  const src = path.join(tmp, 'joint.safetensors');
  writeFixture(src);
  const r = importYue2ComfyAdapter(src, path.join(tmp, 'adapters'));
  assert.equal(r.trigger, 'mytrig');
  assert.equal(r.steps, 7);
  assert.equal(r.alpha, 1); // no alpha in the file = scale 1.0, like ComfyUI

  const ar = read(r.arPath!), nar = read(r.narPath!);
  assert.equal(ar.header.__metadata__.format, 'yue2-ar-lora-v1');
  assert.equal(nar.header.__metadata__.format, 'yue2-nar-lora-v1');
  assert.equal(nar.header.__metadata__.style_template, 'upstream');
  assert.ok(ar.header['yue2.blk.0.attn_q.lora_A.weight']);
  assert.ok(nar.header['yue2.blk.0.nar_attn_q.lora_A.weight']);
  assert.ok(!Object.keys(ar.header).some(k => k.includes('nar_')));

  // fused rows: q 0..2047, k 2048..3071, v 3072..4095; gate 0..6143, up 6144..
  assert.deepEqual(nar.header['yue2.blk.0.nar_attn_k.lora_B.weight'].shape, [1024, 1]);
  assert.equal(nar.first('yue2.blk.0.nar_attn_k.lora_B.weight'), 2048);
  assert.equal(nar.first('yue2.blk.0.nar_attn_v.lora_B.weight'), 3072);
  assert.equal(ar.first('yue2.blk.0.ffn_up.lora_B.weight'), 6144);
  assert.equal(ar.first('yue2.blk.0.ffn_down.lora_B.weight'), 0);

  const index = JSON.parse(fs.readFileSync(path.join(process.env.TRAINING_DIR!, 'yue2-aitk-runs.json'), 'utf8'));
  assert.equal(index[0].options.source, 'comfyui-import');
  assert.equal(index[0].checkpoints[0].arPath, r.arPath);
});

test('lora_down and lora_up weights become native LoRA pairs', () => {
  const src = path.join(tmp, 'down-up.safetensors');
  writeFixture(src, { sites: { 'self_attn.qkv_proj': [2048, 4096] },
    suffixes: ['lora_down.weight', 'lora_up.weight'] });
  const r = importYue2ComfyAdapter(src, path.join(tmp, 'adapters'));
  const ar = read(r.arPath!), nar = read(r.narPath!);
  assert.equal(ar.first('yue2.blk.0.attn_k.lora_B.weight'), 2048);
  assert.equal(nar.first('yue2.blk.0.nar_attn_v.lora_B.weight'), 3072);
  assert.ok(ar.header['yue2.blk.0.attn_q.lora_A.weight']);
});

test('unfused projections map directly to all seven native sites', () => {
  const src = path.join(tmp, 'unfused.safetensors');
  writeFixture(src, {
    experts: ['bare', 'diffusion_model'],
    sites: {
      q_proj: [2048, 2048], k_proj: [2048, 1024], v_proj: [2048, 1024],
      o_proj: [2048, 2048], gate_proj: [2048, 6144],
      up_proj: [2048, 6144], down_proj: [6144, 2048],
    },
    suffixes: ['lora_A', 'lora_B'],
  });
  const r = importYue2ComfyAdapter(src, path.join(tmp, 'adapters'));
  const names = ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down'];
  for (const [file, prefix] of [[r.arPath!, ''], [r.narPath!, 'nar_']] as const) {
    const { header } = read(file);
    for (const name of names) {
      assert.ok(header[`yue2.blk.0.${prefix}${name}.lora_A.weight`]);
      assert.ok(header[`yue2.blk.0.${prefix}${name}.lora_B.weight`]);
    }
    assert.equal(Object.keys(header).length, 22); // seven pairs, seven module alphas, metadata
  }
});

test('an AR-only file writes and registers only its AR half', () => {
  const src = path.join(tmp, 'ar-only.safetensors');
  writeFixture(src, { experts: ['text_encoders'] });
  const r = importYue2ComfyAdapter(src, path.join(tmp, 'adapters'));
  assert.ok(r.arPath && fs.existsSync(r.arPath));
  assert.equal(r.narPath, undefined);
  assert.equal(fs.existsSync(path.join(path.dirname(r.arPath!), 'native-nar.safetensors')), false);
  const index = JSON.parse(fs.readFileSync(path.join(process.env.TRAINING_DIR!, 'yue2-aitk-runs.json'), 'utf8'));
  const ckpt = index.find((run: { output: string }) => run.output === r.output).checkpoints[0];
  assert.equal(ckpt.arPath, r.arPath);
  assert.equal(ckpt.narPath, undefined);
});

test('a NAR-only file writes and registers only its NAR half', () => {
  const src = path.join(tmp, 'nar-only.safetensors');
  writeFixture(src, { experts: ['diffusion_model'] });
  const r = importYue2ComfyAdapter(src, path.join(tmp, 'adapters'));
  assert.equal(r.arPath, undefined);
  assert.ok(r.narPath && fs.existsSync(r.narPath));
});

test('a file without YuE2 modules is refused', () => {
  const src = path.join(tmp, 'other.safetensors');
  writeFixture(src, { sites: { other_proj: [2048, 2048] } });
  assert.throws(() => importYue2ComfyAdapter(src, path.join(tmp, 'adapters')), /No YuE2 adapter tensors/);
});

test('fused slider modules retain their different alpha values after splitting', () => {
  const src = path.join(tmp, 'slider.safetensors');
  writeFixture(src, { experts: ['text_encoders'], alphas: { 'self_attn.qkv_proj': 24, 'self_attn.o_proj': 8 } });
  const result = importYue2ComfyAdapter(src, path.join(tmp, 'adapters'));
  const { first } = read(result.arPath!);
  assert.equal(first('yue2.blk.0.attn_q.alpha'), 24);
  assert.equal(first('yue2.blk.0.attn_k.alpha'), 24);
  assert.equal(first('yue2.blk.0.attn_output.alpha'), 8);
  assert.equal(result.alpha, undefined);
});

test('ordinary native slider names map to the AR family', () => {
  const src = path.join(tmp, 'native-slider.safetensors');
  writeFixture(src, { experts: ['bare'], sites: { 'self_attn.k_proj': [2048, 1024] },
    key: key => key.replace('layers.0.self_attn.k_proj', 'adapters.model-layers-0-self_attn-k_proj') });
  const result = importYue2ComfyAdapter(src, path.join(tmp, 'adapters'));
  assert.ok(read(result.arPath!).header['yue2.blk.0.attn_k.lora_A.weight']);
  assert.equal(result.narPath, undefined);
});

test('sidecar alpha is captured in each module and baked module alpha takes precedence', () => {
  const dir = path.join(tmp, 'sidecar'); fs.mkdirSync(dir);
  const src = path.join(dir, 'adapter.safetensors');
  writeFixture(src, { experts: ['bare'], alphas: { 'self_attn.o_proj': 8 } });
  fs.writeFileSync(path.join(dir, 'adapter_config.json'), JSON.stringify({ lora_alpha: 0.5 }));
  const result = importYue2ComfyAdapter(src, path.join(tmp, 'adapters'));
  const { first } = read(result.arPath!);
  assert.equal(first('yue2.blk.0.attn_q.alpha'), 0.5);
  assert.equal(first('yue2.blk.0.attn_output.alpha'), 8);
});

for (const flow of ['diff', 'replacement'] as const) test(`NAR ${flow} flow heads survive conversion and disable the companion`, () => {
  const src = path.join(tmp, `flow-${flow}.safetensors`);
  writeFixture(src, { experts: ['diffusion_model'], flow });
  const { header, first } = read(importYue2ComfyAdapter(src, path.join(tmp, 'adapters')).narPath!);
  assert.equal(header.__metadata__.decoder_companion, 'off');
  assert.equal(first(`yue2.vae2llm.${flow === 'diff' ? 'diff_b' : 'bias'}`), 0.25);
  assert.deepEqual(header[`yue2.llm2vae.${flow === 'diff' ? 'diff' : 'weight'}`].shape, [64, 2048]);
});
