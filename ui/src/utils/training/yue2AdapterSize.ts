// Expected on-disk size of a YuE2 joint adapter, per half (AR planner and NAR
// decoder share the same shapes). Mirrors the engine: 28 layers, four fused
// sites per layer (engine/src/train/yue2-aitk-lora.h), LyCORIS factorization
// (engine/src/lokr-common.h), BF16 files. The exported split files repeat the
// shared factor in every q/k/v and gate/up slice (LoRA's A, LoKr's w2), which
// is what reproduces the measured 279 MB (LoRA r64) and 106 MB (LoKr 64/4).

const H = 2048, Q = 2048, KV = 1024, FF = 6144, LAYERS = 28, BYTES = 2;
// [input, output, slices in the split export]
const SITES: [number, number, number][] = [[H, Q + 2 * KV, 3], [Q, H, 1], [H, 2 * FF, 2], [FF, H, 1]];

function factorization(dimension: number, factor: number): [number, number] {
  if (factor > 0 && dimension % factor === 0) {
    const n = dimension / factor;
    return factor > n ? [n, factor] : [factor, n];
  }
  const cap = factor < 0 ? dimension : factor;
  let m = 1, n = dimension;
  const length = m + n;
  while (m < n) {
    let nm = m + 1;
    while (dimension % nm !== 0) nm++;
    const nn = dimension / nm;
    if (nm + nn > length || nm > cap) break;
    m = nm; n = nn;
  }
  return m > n ? [n, m] : [m, n];
}

/** Bytes for ONE half, or null when the LoKr factor cannot split the fused
 *  q/k/v or gate/up outputs on their boundaries (the trainer refuses it). */
export function yue2AdapterHalfBytes(a: { type: 'lora'; rank: number } | { type: 'lokr'; dim: number; factor: number }): number | null {
  let params = 0;
  for (const [input, output, slices] of SITES) {
    if (a.type === 'lora') {
      params += a.rank * input * slices + a.rank * output;
      continue;
    }
    const [outL, outK] = factorization(output, a.factor);
    const [inM, inN] = factorization(input, a.factor);
    if (slices === 3 && (Q % outK !== 0 || KV % outK !== 0)) return null;
    if (slices === 2 && FF % outK !== 0) return null;
    const mono = !(a.dim < Math.max(outK, inN) / 2);
    const w2 = mono ? inN * outK : a.dim * (outK + inN);
    params += inM * outL + w2 * slices;
  }
  return params * LAYERS * BYTES;
}

export const formatMB = (bytes: number) => `${Math.round(bytes / 1e6)} MB`;
