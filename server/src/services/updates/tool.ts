// tool.ts — bundled as server/updater-tool.mjs in every payload. The stable
// launcher runs a payload's own copy with that payload's Node, so its DB is
// snapshotted and restored by the SQLite build that wrote it. One JSON line on
// stdout on success; a message on stderr and exit 1 on failure.
//   updater-tool.mjs snapshot <dir> '{"dbs":[...],"files":[...]}'  -> {"manifestSha256":"..."}
//   updater-tool.mjs restore <dir> <manifestSha256>                 -> {"ok":true}
import { createSnapshot, restoreSnapshot, verifyRestored } from './maintenance.js';

export async function runTool(argv: string[]): Promise<object> {
  const [cmd, dir, arg] = argv;
  if (!dir || !arg) throw new Error('usage: updater-tool snapshot <dir> <set-json> | restore <dir> <manifest-sha256>');
  if (cmd === 'snapshot') {
    const set = JSON.parse(arg) as { dbs: string[]; files: string[] };
    if (!Array.isArray(set?.dbs) || !Array.isArray(set?.files)) throw new Error('snapshot set needs dbs and files arrays');
    return { manifestSha256: (await createSnapshot(dir, set)).manifestSha256 };
  }
  if (cmd === 'restore') {
    await restoreSnapshot(dir, arg);
    verifyRestored(dir, arg);
    return { ok: true };
  }
  throw new Error(`unknown command ${cmd}`);
}

const entry = process.argv[1]?.replace(/\\/g, '/') ?? '';
if (/(?:updater-tool\.mjs|updates\/tool\.ts)$/.test(entry)) {
  runTool(process.argv.slice(2)).then(
    r => { process.stdout.write(JSON.stringify(r) + '\n'); process.exit(0); },
    e => { process.stderr.write(`${(e as Error).message}\n`); process.exit(1); },
  );
}
