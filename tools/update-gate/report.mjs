import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
const out = path.join(path.dirname(fileURLToPath(import.meta.url)), 'evidence');
const root = path.resolve(process.env.UPDATE_GATE_ROOT || 'D:/Ace-Step-Latest/hotstep-1148-fixture');
const read = f => JSON.parse(fs.readFileSync(f, 'utf8'));
const write = (f, value) => fs.writeFileSync(f, JSON.stringify(value, null, 2) + '\n');
const suites = read(path.join(out, 'suites.json'));
for (const suite of suites) suite.summary = fs.readFileSync(path.join(out, `${suite.name}.log`), 'utf8').split('\n').filter(l => /^(?:#|ℹ) (tests|pass|fail|cancelled|skipped|duration_ms)/.test(l));
write(path.join(out, 'suites.json'), suites);
const cases = ['clean', 'failed-health', 'no-models', 'before-pointer', 'after-pointer', 'mid-restore', 'after-commit', 'busy-lock', 'protocol-zero', 'recovery-no-server'].map(name => read(path.join(out, `${name}-result.json`)));
for (const c of cases) {
  if (c.details?.images) c.details.images = c.details.images.map(({ alt }) => ({ alt }));
  write(path.join(out, `${c.name}-result.json`), c);
}
const require = createRequire(import.meta.url);
const Database = require(path.join(root, 'source/server/node_modules/better-sqlite3'));
const db = new Database(path.join(root, 'failed-health-run/data/library/hotstep.db'), { readonly: true });
const failedRows = {
  song: db.prepare("SELECT id,title FROM songs WHERE id='update-gate-song'").get(),
  setting: db.prepare("SELECT key,value FROM settings WHERE key='update-gate'").get(),
};
db.close();
write(path.join(out, 'failed-health-retained-candidate-rows.json'), failedRows);
const summary = {
  source: read(path.join(root, 'clean-confirm/install/app', read(path.join(root, 'clean-confirm/install/current.json')).payload, 'build-info.json')).sha,
  recommendation: 'Keep UPDATER_PROTOCOL=0. Fix failed-engine attestation and preserve updater reconnect UI through global offline handling; then rerun the failed cases and obtain the owner portable check.',
  packages: ['91.0.0', '91.0.1', '91.0.2'].map(version => ({ version, protocol: version === '91.0.2' ? 0 : 1, buildLog: `build-${version}.stdout.log` })),
  method: {
    commands: ['node tools/update-gate/run.mjs failed-health', 'node tools/update-gate/run.mjs clean,before-pointer,after-pointer,mid-restore,after-commit', 'node tools/update-gate/run.mjs busy-lock,protocol-zero,recovery-no-server', 'node tools/update-gate/run.mjs recovery-no-server', 'node tools/update-gate/run.mjs no-models'],
    feed: 'NODE_OPTIONS --import process-local fetch replacement for exact GitHub latest-release, commit, archive and checksum URLs; all package/launcher/server/UI code unmodified',
    native: 'Copied engine binaries via -SkipEngine; CPU backend; metadata-only GGUF fixtures; no weight loading or generation',
    data: 'Separate profile/data folders for every case; server 13948, engine 18948; process trees recorded with Get-CimInstance and killed by exact owned PID after daemon PID check',
    comparisons: 'All 31 table row sets retained as evidence. Equality excludes only server_instances, whose UUID/PID/heartbeat change on restart. Settings .env compared byte-for-byte. After-commit case deliberately changes a setting after commit and proves it survives recovery.',
    failedHealth: 'Candidate preload makes its engine path absent and changes a song, setting and .env. Real attestation incorrectly accepts it. Mid-restore uses an injected HTTP 503 to enter rollback independently of this defect.',
    busy: 'Real model-download job fed a slow local HTTP response through a process-local HTTPS redirect. No model loaded. Job continues after refusal, then the test cancels it.',
    protocolZero: 'Default flat archive verified. Its packaged server command is run directly with isolated ports because the old flat bat hardcodes its browser helper to port 3001.',
    recoveryUi: 'Final reproduction uses real browser time. Offline transition unmounts the updater overlay before its timeout; screenshot and launcher recovery-required receipt retained. No five-minute timeout is claimed to have elapsed.',
  },
  failedRows,
  cases,
  suites,
  evidenceDirectory: out,
  cleanup: { processes: 'All fixture process trees stopped; final port/process inspection saved separately', folders: root, reason: 'Automatic approval review rejected recursive cleanup with blocked by policy. Fixtures and disposable clone retained.' },
};
write(path.join(out, 'summary.json'), summary);
console.log(JSON.stringify({ cases: cases.map(({ name, verdict }) => ({ name, verdict })), suites: suites.map(({ name, exit, summary }) => ({ name, exit, summary })), failedRows }, null, 2));
