import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const batch = process.env.UPDATE_GATE_BATCH || 'round2';
const out = path.join(path.dirname(fileURLToPath(import.meta.url)), 'evidence', batch);
const root = path.resolve(process.env.UPDATE_GATE_ROOT || 'D:/Ace-Step-Latest/hotstep-1148-fixture');
const read = f => JSON.parse(fs.readFileSync(f, 'utf8'));
const suites = read(path.join(out, 'suites.json'));
const cases = ['clean', 'failed-health', 'no-models', 'both-no-models', 'before-pointer', 'after-pointer', 'mid-restore', 'after-commit', 'committed-missing', 'busy-lock', 'protocol-zero', 'recovery-no-server'].map(name => read(path.join(out, `${name}-result.json`)));
const passed = cases.every(c => c.verdict === 'PASS') && suites.every(s => s.exit === 0);
const source = cases.find(c => c.name === 'clean')?.health?.build?.sha;
const major = process.env.UPDATE_GATE_MAJOR || '92';
const summary = {
  source,
  verdict: passed ? 'PASS: CPU Windows portable acceptance' : 'FAIL: see case and suite receipts',
  packages: [0, 1, 2].map(patch => ({ version: `${major}.0.${patch}`, protocol: patch === 2 ? 0 : 1, buildLog: `build-${major}.0.${patch}.stdout.log` })),
  commands: ['pwsh -NoProfile -File tools/update-gate/build-fixtures.ps1 -Reuse', 'node tools/update-gate/suites.mjs', 'node tools/update-gate/run.mjs all', "$env:UPDATE_GATE_CASE_SUFFIX='retry-'; node tools/update-gate/run.mjs clean,failed-health,no-models,both-no-models,after-pointer,mid-restore,committed-missing", "$env:UPDATE_GATE_CASE_SUFFIX='retry2-'; node tools/update-gate/run.mjs failed-health,committed-missing", "$env:UPDATE_GATE_CASE_SUFFIX='retry3-'; node tools/update-gate/run.mjs committed-missing"],
  method: {
    feed: 'Process-local interception of exact release/feed URLs. Production launcher/server/UI sources are unchanged.',
    native: 'CPU package with reused binaries (-SkipEngine), metadata-only GGUF stubs; no weights loaded or generation. Native version appears in clean health receipt.',
    data: 'Isolated profile/data folders; ports 13948/18948. Exact owned process trees checked against daemon PID before termination.',
    comparisons: 'All database rows retained. Equality excludes only server_instances process heartbeat rows. Settings .env compared byte-for-byte. Failed-health compares every persistent row to the actual pre-update SQLite snapshot, because refreshing Settings before Apply can save a model selection. Post-commit sentinel proves recovery does not restore old data.',
    faults: 'Missing engine path and reversible candidate data changes force rollback. Filesystem hooks freeze at four actual durable transaction boundaries. Empty models uses an empty directory, with no data mutation.',
    ui: 'Real Chromium time and real failed requests. Matching stale receipts cannot finish the attempt. Clean/rollback dialogs retain the old document until manual Reload. Damaged snapshot waits five minutes once.',
    busy: 'A real model download reads controlled slow HTTP bytes; it continues after apply refusal, then the harness cancels it.',
    protocolZero: 'Flat package layout checked; packaged server started directly on isolated ports to avoid the flat launcher browser URL hardcoded to 3001.',
  },
  limits: ['CUDA package not run: local GPU prohibited for this batch.', 'No visual or listening approval claimed.', 'Packager npm install changes fixture-only ui/package-lock.json metadata; application source remains the requested commit.'],
  harnessRetries: 'Initial attempts are retained under initial-harness, first-retry and second-retry. Harness corrections: nullable success message, stale query after API preparation, model-selection baseline timing, actual snapshot comparison, health field names, and precise visible engine-status matching. No application code or package archive changed between attempts.',
  cases, suites, evidenceDirectory: out, fixtureDirectory: root,
};
fs.writeFileSync(path.join(out, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify({ verdict: summary.verdict, cases: cases.map(({ name, verdict }) => ({ name, verdict })), suites: suites.map(({ name, exit, summary }) => ({ name, exit, summary })) }, null, 2));
