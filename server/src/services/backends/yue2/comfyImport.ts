// comfyImport.ts — bring a ComfyUI / ai-toolkit YuE2 adapter into the app.
//
// Those tools save ONE file holding both halves in the upstream module tree:
// `diffusion_model.model.layers.N.*` is the NAR half, `text_encoders.model.
// layers.N.*` the AR half, with fused `self_attn.qkv_proj` and
// `mlp.gate_up_proj`. The engine loads neither that naming nor a file that
// spans both halves (yue2-adapter.h refuses it on purpose), so this splits it
// into the two native files our own trainer exports, byte for byte the layout
// of engine/src/train/yue2-aitk-native-adapter-io.h: LoRA splits the fused
// sites by B rows with A shared, LoKr by w1 rows with w2 whole. Row slicing of
// a row-major tensor is a contiguous byte range, so no dtype is decoded except
// a per-module alpha scalar.
//
// The pair lands in the joint-adapter tree and the joint-run index, so the
// picker lists it exactly like a checkpoint trained here.

import fs from 'fs';
import path from 'path';
import { config } from '../../../config.js';
import { runStamp } from '../../training/adapterLayout.js';
import { recordYue2AitkRun } from '../../training/yue2AitkRuns.js';

const KEY = /^(?:model\.)?(diffusion_model|text_encoders)\.model\.layers\.(\d+)\.(self_attn\.qkv_proj|self_attn\.o_proj|mlp\.gate_up_proj|mlp\.down_proj)\.(.+)$/;
const SITES: Record<string, [number, number]> = {
  'self_attn.qkv_proj': [2048, 4096], 'self_attn.o_proj': [2048, 2048],
  'mlp.gate_up_proj': [2048, 12288], 'mlp.down_proj': [6144, 2048],
};
const ELEM: Record<string, number> = { F32: 4, F16: 2, BF16: 2 };
const MAX_BYTES = 2 * 1024 ** 3;

interface Tensor { dtype: string; shape: number[]; data: Buffer }

function splits(site: string, layer: number, ar: boolean): Array<[string, number, number]> {
  const b = `blk.${layer}.`;
  const n = ar ? '' : 'nar_';
  if (site === 'self_attn.qkv_proj') return [[`${b}${n}attn_q`, 0, 2048], [`${b}${n}attn_k`, 2048, 1024], [`${b}${n}attn_v`, 3072, 1024]];
  if (site === 'self_attn.o_proj') return [[`${b}${n}attn_output`, 0, 2048]];
  if (site === 'mlp.gate_up_proj') return [[`${b}${n}ffn_gate`, 0, 6144], [`${b}${n}ffn_up`, 6144, 6144]];
  return [[`${b}${n}ffn_down`, 0, 2048]];
}

function readSafetensors(file: string): { tensors: Map<string, Tensor>; meta: Record<string, string> } {
  const buf = fs.readFileSync(file);
  const n = Number(buf.readBigUInt64LE(0));
  if (!Number.isFinite(n) || n <= 0 || 8 + n > buf.length) throw new Error('not a safetensors file (bad header length)');
  const header = JSON.parse(buf.subarray(8, 8 + n).toString('utf8')) as Record<string, any>;
  const meta = (header.__metadata__ ?? {}) as Record<string, string>;
  delete header.__metadata__;
  const base = 8 + n;
  const tensors = new Map<string, Tensor>();
  for (const [name, e] of Object.entries(header)) {
    const [s, t] = e.data_offsets as [number, number];
    if (!ELEM[e.dtype]) throw new Error(`tensor ${name} has dtype ${e.dtype}; only F32, F16 and BF16 adapters are supported`);
    const count = (e.shape as number[]).reduce((a, d) => a * d, 1);
    if (t - s !== count * ELEM[e.dtype] || base + t > buf.length) throw new Error(`tensor ${name} is truncated or malformed`);
    tensors.set(name, { dtype: e.dtype, shape: e.shape, data: buf.subarray(base + s, base + t) });
  }
  return { tensors, meta };
}

function writeSafetensors(file: string, tensors: Array<[string, Tensor]>, meta: Record<string, string>): void {
  const header: Record<string, unknown> = { __metadata__: meta };
  let off = 0;
  for (const [name, t] of tensors) {
    header[name] = { dtype: t.dtype, shape: t.shape, data_offsets: [off, off + t.data.length] };
    off += t.data.length;
  }
  let json = JSON.stringify(header);
  json += ' '.repeat((8 - (json.length % 8)) % 8);
  const len = Buffer.alloc(8);
  len.writeBigUInt64LE(BigInt(Buffer.byteLength(json)));
  fs.writeFileSync(file, Buffer.concat([len, Buffer.from(json, 'utf8'), ...tensors.map(([, t]) => t.data)]));
}

/** Rows [first, first+count) of a row-major 2-D tensor. */
function rows(t: Tensor, first: number, count: number): Tensor {
  const row = t.data.length / t.shape[0];
  return { dtype: t.dtype, shape: [count, t.shape[1]], data: t.data.subarray(first * row, (first + count) * row) };
}

function scalar(t: Tensor): number {
  if (t.dtype === 'F32') return t.data.readFloatLE(0);
  if (t.dtype === 'BF16') { const b = Buffer.alloc(4); b.writeUInt16LE(t.data.readUInt16LE(0), 2); return b.readFloatLE(0); }
  const h = t.data.readUInt16LE(0);
  const e = (h >> 10) & 31, m = h & 1023, s = h >> 15 ? -1 : 1;
  return s * (e === 0 ? m * 2 ** -24 : 2 ** (e - 15) * (1 + m / 1024));
}

/** The trigger a kohya-style trainer recorded: the most frequent tag in
 *  ss_tag_frequency, else the `<repeats>_<name>` folder name it came from. */
export function comfyTrigger(meta: Record<string, string>): string {
  if (meta.trigger) return meta.trigger.trim();
  try {
    const freq = JSON.parse(meta.ss_tag_frequency ?? '{}') as Record<string, Record<string, number>>;
    const tags = Object.values(freq).flatMap(f => Object.entries(f)).sort((a, b) => b[1] - a[1]);
    if (tags.length) return tags[0][0].trim();
    const folder = Object.keys(freq)[0];
    if (folder) return folder.replace(/^\d+_/, '').trim();
  } catch { /* no usable tag record */ }
  return '';
}

export interface Yue2ComfyImport {
  output: string; arPath: string; narPath: string;
  name: string; kind: 'lora' | 'lokr'; rank: number; alpha?: number; steps: number; trigger: string;
}

/** Split `src` into native AR + NAR files under the joint-adapter tree. */
export function importYue2ComfyAdapter(src: string, adaptersRoot = config.aceServer.adapters): Yue2ComfyImport {
  if (!path.isAbsolute(src) || !/\.safetensors$/i.test(src) || !fs.existsSync(src) || !fs.statSync(src).isFile()) {
    throw new Error(`Needs an absolute path to an existing .safetensors file: ${src}`);
  }
  if (fs.statSync(src).size > MAX_BYTES) throw new Error('File is larger than 2 GB, which no YuE2 adapter is');
  const { tensors, meta } = readSafetensors(src);
  if (/^yue2-(ar|nar)-/.test(meta.format ?? '')) {
    throw new Error('This is already a native YuE2 adapter. Point the adapter folder at it instead of importing it.');
  }

  const groups = new Map<string, { ar: boolean; layer: number; site: string; parts: Record<string, Tensor> }>();
  const unknown: string[] = [];
  for (const [name, t] of tensors) {
    const m = KEY.exec(name);
    if (!m) { unknown.push(name); continue; }
    const id = `${m[1]}|${m[2]}|${m[3]}`;
    if (!groups.has(id)) groups.set(id, { ar: m[1] === 'text_encoders', layer: Number(m[2]), site: m[3], parts: {} });
    groups.get(id)!.parts[m[4]] = t;
  }
  if (!groups.size) throw new Error('No YuE2 adapter tensors found. Expected ComfyUI naming (diffusion_model.model.layers.* / text_encoders.model.layers.*).');
  if (unknown.length) throw new Error(`${unknown.length} tensors this importer does not recognise, e.g. ${unknown.slice(0, 3).join(', ')}`);
  const kinds = new Set([...groups.values()].map(g => Object.keys(g.parts).some(k => k.startsWith('lokr')) ? 'lokr' : 'lora'));
  if (kinds.size !== 1) throw new Error('File mixes LoRA and LoKr modules');
  const kind = [...kinds][0] as 'lora' | 'lokr';
  const alphas = new Set([...groups.values()].filter(g => g.parts.alpha).map(g => scalar(g.parts.alpha)));
  if (alphas.size > 1) throw new Error(`Modules carry different alphas (${[...alphas].slice(0, 4).join(', ')}); not supported`);

  const out: Record<'ar' | 'nar', Array<[string, Tensor]>> = { ar: [], nar: [] };
  let rank = 0;
  const sorted = [...groups.values()].sort((a, b) => Number(a.ar) - Number(b.ar) || a.layer - b.layer || a.site.localeCompare(b.site));
  for (const g of sorted) {
    const [inp, outp] = SITES[g.site];
    const dst = out[g.ar ? 'ar' : 'nar'];
    const where = `${g.ar ? 'AR' : 'NAR'} layer ${g.layer} ${g.site}`;
    if (kind === 'lora') {
      const A = g.parts['lora_A.weight'], B = g.parts['lora_B.weight'];
      if (!A || !B) throw new Error(`${where}: missing lora_A or lora_B`);
      const r = A.shape[0];
      if (A.shape[1] !== inp || B.shape[0] !== outp || B.shape[1] !== r) throw new Error(`${where}: shapes ${A.shape} / ${B.shape} do not fit YuE2`);
      rank ||= r;
      for (const [name, first, count] of splits(g.site, g.layer, g.ar)) {
        dst.push([`yue2.${name}.lora_A.weight`, A], [`yue2.${name}.lora_B.weight`, rows(B, first, count)]);
      }
    } else {
      const w1 = g.parts.lokr_w1, w2 = g.parts.lokr_w2, w2a = g.parts.lokr_w2_a, w2b = g.parts.lokr_w2_b;
      if (!w1 || !(w2 || (w2a && w2b))) throw new Error(`${where}: incomplete LoKr module`);
      const outK = w2 ? w2.shape[0] : w2a.shape[0], inN = w2 ? w2.shape[1] : w2b.shape[1];
      if (w1.shape[0] * outK !== outp || w1.shape[1] * inN !== inp) throw new Error(`${where}: LoKr shapes do not fit YuE2`);
      if (w2a) rank ||= w2a.shape[1];
      for (const [name, first, count] of splits(g.site, g.layer, g.ar)) {
        if (first % outK || count % outK) throw new Error(`${where}: LoKr factor does not divide the fused split`);
        dst.push([`yue2.${name}.lokr_w1`, rows(w1, first / outK, count / outK)]);
        if (w2) dst.push([`yue2.${name}.lokr_w2`, w2]);
        else dst.push([`yue2.${name}.lokr_w2_a`, w2a], [`yue2.${name}.lokr_w2_b`, w2b]);
      }
    }
  }
  if (!out.ar.length || !out.nar.length) {
    throw new Error(`File only holds the ${out.ar.length ? 'AR' : 'NAR'} half; a joint ComfyUI adapter carries both`);
  }

  // No alpha in the file means ComfyUI merges at scale 1.0, i.e. alpha = rank.
  // Recording that keeps the app playing it the way ComfyUI does.
  const alpha = alphas.size ? [...alphas][0] : (rank || undefined);
  let steps = 0;
  try { steps = Number(JSON.parse(meta.training_info ?? '{}').step) || 0; } catch { /* unknown */ }
  const trigger = comfyTrigger(meta);
  const name = (meta.ss_output_name || path.basename(src, path.extname(src)))
    .replace(/[^a-zA-Z0-9._-]+/g, '_').replace(/^[._]+|[._]+$/g, '').slice(0, 100) || 'imported';

  const stamp = runStamp();
  const output = path.join(adaptersRoot, 'yue2-joint-adapters', `${name}_${stamp}`);
  const dir = path.join(output, `checkpoint-step${steps}`);
  fs.mkdirSync(dir, { recursive: true });
  const paths = { ar: path.join(dir, 'native-ar.safetensors'), nar: path.join(dir, 'native-nar.safetensors') };
  for (const half of ['ar', 'nar'] as const) {
    const md: Record<string, string> = {
      format: `yue2-${half}-${kind}-v1`, rank: String(rank), steps: String(steps),
      yue2_adapter_layout: 'native_split_v1', converted_from: 'comfyui', source_file: path.basename(src),
    };
    if (alpha !== undefined) md.alpha = String(alpha);
    if (meta.lokr_dim) md.lokr_dim = meta.lokr_dim;
    if (meta.lokr_factor) md.lokr_factor = meta.lokr_factor;
    if (trigger) Object.assign(md, { trigger, style_template: 'upstream' });
    writeSafetensors(paths[half], out[half], md);
  }

  const now = Date.now();
  recordYue2AitkRun({
    version: 1, jobId: `import-${name}-${stamp}`, datasetId: '', datasetSlug: '', method: 'aitk', output,
    options: { source: 'comfyui-import', name, sourceFile: src, trigger, steps },
    status: 'done', createdAt: now, updatedAt: now, checkpoints: [],
  });
  return { output, arPath: paths.ar, narPath: paths.nar, name, kind, rank, alpha, steps, trigger };
}
