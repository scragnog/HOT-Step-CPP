import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const out = path.join(path.dirname(fileURLToPath(import.meta.url)), 'evidence');
const root = path.resolve(process.env.UPDATE_GATE_ROOT || 'D:/Ace-Step-Latest/hotstep-1148-fixture', 'source');
const files = dir => fs.readdirSync(dir, { withFileTypes: true }).flatMap(e => e.isDirectory() ? files(path.join(dir, e.name)) : [path.join(dir, e.name)]);
const results = [];
for (const [name, cwd, args] of [
  ['server-tsc', 'server', ['node_modules/typescript/bin/tsc', '--noEmit']],
  ['ui-tsc', 'ui', ['node_modules/typescript/bin/tsc', '-b']],
  ['server-tests', 'server', ['--import', 'tsx', '--test', '--test-timeout=120000', ...files(path.join(root, 'server/src')).filter(f => f.endsWith('.test.ts')).map(f => path.relative(path.join(root, 'server'), f))]],
  ['ui-tests', 'ui', ['--import', '../server/node_modules/tsx/dist/loader.mjs', '--test', '--test-force-exit', '--test-timeout=120000', ...files(path.join(root, 'ui/src')).filter(f => /\.test\.(ts|tsx)$/.test(f)).map(f => path.relative(path.join(root, 'ui'), f))]],
  ['release-tests', '', ['--test', '--test-timeout=120000', ...files(path.join(root, 'release')).filter(f => f.endsWith('.test.mjs') && !f.includes('node_modules')).map(f => path.relative(root, f))]],
]) {
  console.log(`START ${name}`);
  const fd = fs.openSync(path.join(out, `${name}.log`), 'w');
  const started = Date.now();
  const r = spawnSync(process.execPath, args, { cwd: path.join(root, cwd), timeout: 900_000, windowsHide: true, stdio: ['ignore', fd, fd] });
  fs.closeSync(fd);
  const log = fs.readFileSync(path.join(out, `${name}.log`), 'utf8');
  const result = { name, command: ['node', ...args].join(' '), cwd: path.join(root, cwd), exit: r.status, error: r.error?.message, seconds: (Date.now() - started) / 1000, summary: log.split('\n').filter(l => /^(?:#|ℹ) (tests|pass|fail|cancelled|skipped|duration_ms)/.test(l)) };
  results.push(result);
  console.log(JSON.stringify(result));
  fs.writeFileSync(path.join(out, 'suites.json'), JSON.stringify(results, null, 2));
}
process.exitCode = results.some(r => r.exit !== 0) ? 1 : 0;
