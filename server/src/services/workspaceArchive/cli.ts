// workspaceArchive/cli.ts — Export or import the workspace without the app (GH #228).
//
//   cd server
//   npx tsx src/services/workspaceArchive/cli.ts export --out <folder or file.zip>
//       [--no-training] [--no-adapters] [--models] [--secrets]
//   npx tsx src/services/workspaceArchive/cli.ts import --zip <file.zip> --into <new folder>
//
// Export refuses while a data-folder change is pending or a HOT-Step server is
// running on this data folder (it cannot hold that server's writes; export
// from Settings instead). Import only builds the new data folder and prints
// its report: it never changes which folder the app uses (switch to it with
// Settings > Storage & Data, "use an existing data folder"). It may run while
// the app runs; it only reads this machine's DB and settings.
// Exit 0 on success, 1 on failure.

import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { locateData } from '../../dataroot/locate.js';
import { DEFAULT_CHOICE, type CategoryChoice } from './manifest.js';

const USAGE = 'usage: cli.ts export --out <folder or file.zip> [--no-training] [--no-adapters] [--models] [--secrets]\n'
  +'       cli.ts import --zip <file.zip> --into <new folder>';

export function parseImportArgs(argv: string[]): { zip: string; into: string } {
  let zip = '', into = '';
  for (let i = 1; i < argv.length; i++) {
    if (argv[i] === '--zip') zip = argv[++i] ?? '';
    else if (argv[i] === '--into') into = argv[++i] ?? '';
    else throw new Error(`unknown argument ${argv[i]}\n${USAGE}`);
  }
  if (!zip || !into) throw new Error(USAGE);
  return { zip: path.resolve(zip), into: path.resolve(into) };
}

async function importMain(argv: string[]): Promise<number> {
  const { zip, into } = parseImportArgs(argv);
  const d = locateData();
  if (d.pendingMove) throw new Error(`A data-folder ${d.pendingMove.kind} is pending; start HOT-Step once to finish it, then import.`);
  if (!d.children) throw new Error('This install has no data folder yet.');
  if (d.root) process.env.HOT_STEP_DATA_ROOT = d.root;
  const { importWorkspace } = await import('./import.js');
  const { config } = await import('../../config.js');
  let last = 0;
  const r = await importWorkspace({
    zipFile: zip, target: into,
    here: {
      appRoot: d.installRoot, dataRoot: d.root, children: d.children,
      paths: { WHISPER_EXE: config.whisper.exe, WHISPER_MODELS_DIR: config.whisper.modelsDir, TENSORRT_LIBS: config.aceServer.trtLibs },
    },
    onProgress: p => {
      if (Date.now() - last < 2000 && p.filesDone !== p.filesTotal) return;
      last = Date.now();
      console.error(`[Archive] ${p.phase}: ${p.filesDone}/${p.filesTotal} files, ${(p.bytesDone / 1e9).toFixed(2)}/${(p.bytesTotal / 1e9).toFixed(2)} GB`);
    },
  });
  console.log(JSON.stringify(r, null, 2));
  return 0;
}

export function parseArgs(argv: string[]): { out: string; choice: CategoryChoice } {
  if (argv[0] !== 'export') throw new Error(USAGE);
  const choice = { ...DEFAULT_CHOICE };
  let out = '';
  const flags: Record<string, () => void> = {
    '--no-training': () => { choice.training = false; }, '--training': () => { choice.training = true; },
    '--no-adapters': () => { choice.adapters = false; }, '--adapters': () => { choice.adapters = true; },
    '--models': () => { choice.models = true; }, '--no-models': () => { choice.models = false; },
    '--secrets': () => { choice.secrets = true; }, '--no-secrets': () => { choice.secrets = false; },
  };
  for (let i = 1; i < argv.length; i++) {
    if (argv[i] === '--out') out = argv[++i] ?? '';
    else if (flags[argv[i]]) flags[argv[i]]();
    else throw new Error(`unknown argument ${argv[i]}\n${USAGE}`);
  }
  if (!out) throw new Error(USAGE);
  out = path.resolve(out);
  if (!/\.zip$/i.test(out)) out = path.join(out, `HOT-Step-workspace-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.zip`);
  return { out, choice };
}

async function main(): Promise<number> {
  if (process.argv[2] === 'import') return importMain(process.argv.slice(2));
  const { out, choice } = parseArgs(process.argv.slice(2));
  const d = locateData();
  if (d.pendingMove) throw new Error(`A data-folder ${d.pendingMove.kind} is pending; start HOT-Step once to finish it, then export.`);
  if (!d.children) throw new Error('This install has no data folder yet.');
  // config.ts (imported below through db/backup.ts) bootstraps the data root;
  // pinning it here means it can only open the folder we just located.
  if (d.root) process.env.HOT_STEP_DATA_ROOT = d.root;
  const { exportWorkspace, otherLiveServers } = await import('./export.js');
  const { APP_VERSION } = await import('../../config.js');
  const live = otherLiveServers(d.children.db, null);
  if (live.length) throw new Error(`${live.join('; ')}. Stop it, or export from Settings > Storage & Data.`);

  let last = 0;
  const r = await exportWorkspace({
    children: d.children, dataRoot: d.root, outFile: out, choice, appVersion: APP_VERSION,
    onProgress: p => {
      if (Date.now() - last < 2000 && p.filesDone !== p.filesTotal) return;
      last = Date.now();
      console.error(`[Archive] ${p.phase}: ${p.filesDone}/${p.filesTotal} files, ${(p.bytesDone / 1e9).toFixed(2)}/${(p.bytesTotal / 1e9).toFixed(2)} GB`);
    },
  });
  const m = r.manifest;
  console.log(JSON.stringify({
    outFile: r.outFile, bytes: r.bytes, files: m.files.length, sources: m.sources,
    skipped: m.skipped.length + m.skippedNotListed, externals: m.externals.length,
    credentialRowsRemoved: m.db?.credentialRowsRemoved ?? 0, envKeysDropped: m.env?.droppedKeys ?? [],
  }, null, 2));
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().then(code => { process.exitCode = code; }, (e: Error) => { console.error(`[Archive] ${e.message}`); process.exitCode = 1; });
}
