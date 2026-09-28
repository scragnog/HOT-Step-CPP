// Run: npx tsx scripts/check-yue2-adapter-size.ts
// The size estimate must reproduce the measured on-disk sizes of the two
// shipped defaults; if the engine's shapes or export layout change, this fails.
import { yue2AdapterHalfBytes as half } from '../src/utils/yue2AdapterSize';

const mb = (b: number | null) => Math.round((2 * (b ?? NaN)) / 1e6);
const lora = mb(half({ type: 'lora', rank: 64 }));
const lokr = mb(half({ type: 'lokr', dim: 64, factor: 4 }));
if (lora !== 279 || lokr !== 106) {
  console.error(`FAIL: LoRA r64 ${lora} MB (want 279), LoKr 64/4 ${lokr} MB (want 106)`);
  process.exit(1);
}
console.log(`ok: LoRA r64 ${lora} MB, LoKr 64/4 ${lokr} MB`);
