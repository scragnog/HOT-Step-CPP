// The test build (dist-test/index.html, `npm run build:test`), built again when it is missing or older than anything it
// is built from: src/, index.html, the Vite and TypeScript configs, package.json. Used by the runners that test it
// (tools/visual-baseline/visual-all.mjs, tools/behaviour/switches.mjs).
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

export const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const testPage = path.join(repo, 'dist-test', 'index.html');

const newest = p => { const st = fs.statSync(p); if (!st.isDirectory()) return st.mtimeMs; let m = 0; for (const f of fs.readdirSync(p)) m = Math.max(m, newest(path.join(p, f))); return m; };
const SOURCES = ['src', 'index.html', 'vite.config.ts', 'package.json', 'tsconfig.json', 'tsconfig.app.json'];

/** Builds the test page if it is missing or stale; returns why it was built, or null. Throws if the build fails. */
export function ensureTestBuild() {
  const src = Math.max(...SOURCES.map(s => path.join(repo, s)).filter(fs.existsSync).map(newest));
  const why = !fs.existsSync(testPage) ? 'missing' : fs.statSync(testPage).mtimeMs < src ? 'older than its sources' : null;
  if (!why) return null;
  console.log(`dist-test/index.html is ${why}: npm run build:test`);
  const b = spawnSync('npm run build:test', { cwd: repo, shell: true, encoding: 'utf8' });
  if (b.status !== 0) throw new Error('npm run build:test failed:\n' + (b.stdout + b.stderr).split('\n').filter(l => /error/i.test(l)).slice(0, 10).join('\n'));
  return why;
}
