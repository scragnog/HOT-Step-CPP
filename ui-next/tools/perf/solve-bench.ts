// How long the pure layout takes on the recorded desks (tools/fixtures/pack.json): node tools/perf/solve-bench.ts
import fs from 'node:fs';
import { deskSolve } from '../../src/app/desk/pack.ts';
const { cases } = JSON.parse(fs.readFileSync(new URL('../fixtures/pack.json', import.meta.url), 'utf8'));
const big = [...cases].sort((a, b) => b.items.length - a.items.length).slice(0, 5);
for (const c of big) {
  const t = performance.now(); let n = 0; while (performance.now() - t < 300) { deskSolve(c.cols, c.items); n++; }
  console.log(c.id, c.items.length, 'items', c.cols, 'cols:', ((performance.now() - t) / n).toFixed(2), 'ms a solve');
}
