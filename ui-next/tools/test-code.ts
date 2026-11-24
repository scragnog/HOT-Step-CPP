// The code's contract with the definitions, read from the source (no build, no browser): what makes a rename, a removal
// or an addition the maintainer makes in the definitions either work or fail here with what to do.
//
//   1. The core and the view name no parameter key, state key or definition id (RA-02): what they need in particular
//      they find by its role (definitions/roles.ts, `roleKey`), a module by what it holds (model/index.ts LEAD, BOARD,
//      MODELS) or by the definition a role is played in (`roleOn`). A literal key is refused with its file and line.
//   2. Every part a definition names has a component registered (`definePart` in src/app/parts), and every named action
//      (parts.ts ACTIONS) a definition runs has a runner (`defineAction` in src/app): a part without one would draw as a
//      placeholder, an action without one would only say so (RA-08).
//
//   node tools/test-code.ts        exit 1 on a failure

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFINITIONS } from '../src/definitions/index.ts';
import { ACTIONS } from '../src/definitions/parts.ts';
import type { Node } from '../src/definitions/types.ts';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const fails: string[] = [];
const walk = (dir: string): string[] => fs.readdirSync(dir).flatMap(f => { const p = path.join(dir, f); return fs.statSync(p).isDirectory() ? walk(p) : /\.(ts|tsx)$/.test(f) ? [p] : []; });
const rel = (p: string) => path.relative(ROOT, p).replace(/\\/g, '/');

/* ---------------------------------------------------------------- 1. no key or id in the code */

const keys = new Set<string>(), states = new Set<string>(), ids = new Set<string>();
for (const d of DEFINITIONS) { ids.add(d.id); for (const p of d.params) keys.add(p.key); for (const s of d.state ?? []) states.add(s.key); }
/* the test layer speaks the concept's names on purpose, the page's words are words, and the simulated server is data */
const CODE = [...walk(path.join(ROOT, 'src', 'derive')), ...walk(path.join(ROOT, 'src', 'app'))].map(rel)
  .filter(f => !/^src\/app\/(testing|i18n\/words|services\/mock)\//.test(f));
/* a state key or an id is ambiguous as a word (a kind, a class, a role's name), so only where it is used as one:
   reading a state, finding a definition, testing the desk */
const ID_USE = [/\bstates\.get\(\s*['"`]([\w.]+)['"`]/g, /\bstateOf\(\s*\w+,\s*['"`]([\w.]+)['"`]/g, /\.state\[\s*['"`]([\w.]+)['"`]\s*\]/g,
  /\bdefOf\(\s*['"`]([\w.]+)['"`]/g, /\.id\s*===\s*['"`]([\w.]+)['"`]/g, /\bon\.has\(\s*['"`]([\w.]+)['"`]/g, /\bdataset\.m\s*[!=]==\s*['"`]([\w.]+)['"`]/g, /data-m="([\w.]+)"/g];
let scanned = 0;
for (const f of CODE) {
  const text = fs.readFileSync(path.join(ROOT, f), 'utf8'), lines = text.split(/\r?\n/);
  scanned++;
  lines.forEach((line, i) => {
    if (/^\s*(\/\/|\*|\/\*)/.test(line)) return;
    /* the concept's saved shape (version 0) keeps its own keys, frozen (read through `was` after a rename) */
    if (/frozen/.test(lines[i - 1] ?? '') || /frozen/.test(lines[i - 2] ?? '')) return;
    for (const m of line.matchAll(/['"`]([a-zA-Z][\w]*\.[\w.]+)['"`]/g)) if (keys.has(m[1]!)) fails.push(`${f}:${i + 1}: names the parameter "${m[1]}": find it by its role (definitions/roles.ts, roleKey), so a rename keeps it working`);
    for (const re of ID_USE) for (const m of line.matchAll(re)) {
      const s = m[1]!;
      if (states.has(s)) fails.push(`${f}:${i + 1}: names the state "${s}": find it by its role (definitions/roles.ts, roleKey)`);
      else if (ids.has(s)) fails.push(`${f}:${i + 1}: names the definition "${s}": find it by a role its parameters or state play (roleOn, roleOwner) or by what it holds (model/index.ts)`);
    }
  });
}

/* ---------------------------------------------------------------- 2. every part and action named is built */

const partsUsed = new Map<string, string>(), actionsUsed = new Map<string, string>();
const visit = (n: Node, d: string) => {
  if (!partsUsed.has(n.part)) partsUsed.set(n.part, d);
  for (const a of [...(n.actions ?? []), ...((n.per ?? []) as { actions?: Node['actions'] }[]).flatMap(l => l.actions ?? [])]) if (a.do === 'bespoke') actionsUsed.set(a.name, d);
  (n.children ?? []).forEach(c => visit(c, d));
};
for (const d of DEFINITIONS) for (const n of 'face' in d ? d.face ?? [] : []) visit(n, d.id);
/* what the definitions' JSON names as an action (a strip's, a pedal's), whatever field carries it */
for (const d of DEFINITIONS) for (const name of Object.keys(ACTIONS)) if (JSON.stringify(d).includes(`"name":"${name}"`) && !actionsUsed.has(name)) actionsUsed.set(name, d.id);
const app = walk(path.join(ROOT, 'src', 'app')).filter(f => !/[\\/]testing[\\/]/.test(f)).map(f => fs.readFileSync(f, 'utf8')).join('\n');
const defined = new Set([...app.matchAll(/definePart\(\s*['"`](\w+)['"`]/g)].map(m => m[1]!));
const runners = new Set([...app.matchAll(/defineAction\(\s*['"`](\w+)['"`]/g)].map(m => m[1]!));
/* what the shell draws itself: the generic renderer's containers and the tile's own pieces */
const SHELL = new Set(fs.readFileSync(path.join(ROOT, 'src', 'app', 'parts', 'containers.tsx'), 'utf8').match(/definePart\(\s*['"`](\w+)['"`]/g)?.map(x => x.replace(/.*['"`](\w+)['"`]/, '$1')) ?? []);
for (const [p, d] of partsUsed) if (!defined.has(p) && !SHELL.has(p)) fails.push(`${d} names the part "${p}", which has no component: add src/app/parts/<Part>.tsx with definePart('${p}', ...) (docs/06, section 4)`);
for (const [a, d] of actionsUsed) if (!runners.has(a)) fails.push(`${d} runs the action "${a}", which nothing runs: register it with defineAction('${a}', ...) in a file of src/app/features (docs/06, "Smaller jobs")`);

console.log(`the code and the definitions: ${scanned} files read, no key, state or id named in them${fails.some(f => / names the (parameter|state|definition)/.test(f)) ? ' (NOT)' : ''}; ${partsUsed.size} parts named, each with a component; ${actionsUsed.size} actions run, each with a runner`);
if (fails.length) { console.log(`\n${fails.length} failed:`); fails.slice(0, 60).forEach(f => console.log('  ' + f)); process.exit(1); }
