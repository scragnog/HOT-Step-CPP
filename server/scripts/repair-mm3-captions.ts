// repair-mm3-captions.ts — re-run the MM3 normaliser and fact substitution over
// a dataset's existing .mm3.txt files, without re-captioning.
//
//   cd server && npx tsx scripts/repair-mm3-captions.ts <dataset-dir> [--out <dir>]
//
// Facts (bpm, key, genre, caption) come from each track's .txt sidecar, the
// same fields the labeling pass feeds applyFactSubstitution. Without --out the
// files are rewritten in place; with it, repaired copies go to <dir> so
// check-mm3-format.ts can vet them first. .mm3.txt.prev backups are untouched.
// Use after a normaliser fix, then run check-mm3-format.ts on the result.

import fs from 'fs';
import path from 'path';
import { applyFactSubstitution } from '../src/services/training/mossCaption.js';
import { readSidecar } from '../src/services/training/sidecarIO.js';

const args = process.argv.slice(2);
const outAt = args.indexOf('--out');
const outDir = outAt >= 0 ? args.splice(outAt, 2)[1] : null;
const dir = args[0];
if (!dir) {
  console.error('usage: npx tsx scripts/repair-mm3-captions.ts <dataset-dir> [--out <dir>]');
  process.exit(2);
}
if (outDir) fs.mkdirSync(outDir, { recursive: true });

let changed = 0, total = 0;
for (const name of fs.readdirSync(dir).filter(n => n.endsWith('.mm3.txt'))) {
  total++;
  const file = path.join(dir, name);
  const before = fs.readFileSync(file, 'utf8').replace(/\r\n/g, '\n');
  const meta = readSidecar(path.join(dir, name.replace(/\.mm3\.txt$/, '.txt')));
  const after = applyFactSubstitution(before, {
    bpm: Number(meta.bpm) || null,
    keyscale: meta.key || null,
    genre: meta.genre || null,
    observedCaption: meta.caption || null,
  });
  if (after !== before.replace(/\n+$/, '')) changed++;
  fs.writeFileSync(outDir ? path.join(outDir, name) : file, after, 'utf8');
}
console.log(`${total} files, ${changed} changed${outDir ? ` (written to ${outDir})` : ' (in place)'}`);
