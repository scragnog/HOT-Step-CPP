import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-comfy-import-'));
process.env.TRAINING_DIR = path.join(tmp, 'training'); // keep the run index out of the real one
const { importYue2ComfyAdapter } = await import('./comfyImport.js');

/** A one-layer, rank-1 ComfyUI joint file; every F32 element is its own index,
 *  so a wrong slice shows up as the wrong numbers. */
function writeFixture(file: string): void {
  const sites: Record<string, [number, number]> = {
    'self_attn.qkv_proj': [2048, 4096], 'self_attn.o_proj': [2048, 2048],
    'mlp.gate_up_proj': [2048, 12288], 'mlp.down_proj': [6144, 2048],
  };
  const header: Record<string, unknown> = {
    __metadata__: { format: 'pt', ss_tag_frequency: '{"1_mytrig": {"mytrig": 1}}', training_info: '{"step": 7}' },
  };
  const parts: Buffer[] = [];
  let off = 0;
  for (const expert of ['diffusion_model', 'text_encoders']) {
    for (const [site, [inp, outp]] of Object.entries(sites)) {
      for (const [suffix, shape] of [['lora_A.weight', [1, inp]], ['lora_B.weight', [outp, 1]]] as const) {
        const n = shape[0] * shape[1];
        const data = Buffer.from(Float32Array.from({ length: n }, (_, i) => i).buffer);
        header[`${expert}.model.layers.0.${site}.${suffix}`] = { dtype: 'F32', shape, data_offsets: [off, off + data.length] };
        parts.push(data);
        off += data.length;
      }
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
  try {
    const src = path.join(tmp, 'joint.safetensors');
    writeFixture(src);
    const r = importYue2ComfyAdapter(src, path.join(tmp, 'adapters'));
    assert.equal(r.trigger, 'mytrig');
    assert.equal(r.steps, 7);
    assert.equal(r.alpha, 1); // no alpha in the file = scale 1.0, like ComfyUI

    const ar = read(r.arPath), nar = read(r.narPath);
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
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});
