// Writes tools/fixtures/real-app.json: what the real application declares, read from its source, so the validator
// can check every real name, range, default and option a definition states without a running server.
//
//   node --import ./tools/real-app/load.mjs tools/make-fixtures.ts
//
// (npm run fixtures). Run it after the server, the engine's registries or the Lua plugins change, and commit the
// result. What it reads:
//   generationDefaults   GENERATION_DEFAULTS (server/src/contracts/generationControls.ts), imported
//   createContent        the Create form's content at its defaults (createContentFromDraft({}), ui/src), and the
//                        fields its interface declares
//   extensions           each backend's capabilities().extensions, read from its index.ts
//   reads                the backend-prefixed fields each backend's code reads (params.mm3Xxx and the like), so a
//                        field that is read but not declared can be told from a misspelt one
//   fields               the GENERATION_DEFAULTS fields each engine's code reads, directly or through a field
//                        resolveGenerationParams() makes from them
//   plugins              the solver, scheduler, guidance and post-process ids: the engine's registries and the Lua
//                        plugins' `name`
//   buckets              each backend's model buckets (POST /api/backends/models)
//   sliders              the existing UI's slider for each GENERATION_DEFAULTS field: min, max, step
//   values               the values a request field takes, where the source spells them ('merge' | 'runtime'): the
//                        comments and types of generationControls.ts and ui/src/types.ts, and the guards a backend
//                        throws from (`if (x !== 'ggml' && x !== 'tensorrt') throw`)
//   core                 each backend's batch and duration limits (capabilities().core), null where the server
//                        works them out at run time
//
// It reads the repository ui-next sits in, or the one HOTSTEP_APP names, and writes beside this file.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { ENGINES, FIELD_PREFIXES } from '../src/definitions/engines.ts';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = process.env.HOTSTEP_APP ?? path.resolve(HERE, '..', '..');
const read = (p: string) => fs.readFileSync(path.join(APP, p), 'utf8');
const BACKENDS: Record<string, string> = Object.fromEntries(ENGINES.map(e => [e.id, e.backend]));

const { GENERATION_DEFAULTS } = await import(pathOf('server/src/contracts/generationControls.ts'));
const { createContentFromDraft } = await import(pathOf('ui/src/components/create/createContent.ts'));
function pathOf(p: string) { return 'file:///' + path.join(APP, p).replace(/\\/g, '/'); }

/* ---- extensions: the array literal after `extensions:` (or the named constant it points at), evaluated with every
   free name standing for a constant of the backend's own files, `sd(stage, field, fallback)` for its fallback, and
   any other call for a list the server fills at run time ---- */
function arrayAfter(src: string, at: number): string {
  let i = src.indexOf('[', at), depth = 0, quote = '';
  const start = i;
  for (; i < src.length; i++) {
    const c = src[i]!;
    if (quote) { if (c === '\\') i++; else if (c === quote) quote = ''; continue; }
    if (c === "'" || c === '"' || c === '`') { quote = c; continue; }
    if (c === '/' && src[i + 1] === '/') { i = src.indexOf('\n', i); continue; }
    if (c === '/' && src[i + 1] === '*') { i = src.indexOf('*/', i) + 1; continue; }
    if (c === '[') depth++;
    else if (c === ']' && --depth === 0) return src.slice(start, i + 1);
  }
  throw new Error('unterminated array');
}
function constantsOf(dir: string): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const f of fs.readdirSync(path.join(APP, dir)).filter(f => f.endsWith('.ts') && !f.includes('.test.'))) {
    for (const m of read(path.join(dir, f)).matchAll(/(?:export )?const ([A-Z][A-Z0-9_]+)\s*(?::\s*[\w.]+\s*)?=\s*(-?[\d.]+|'[^']*')\s*;/g)) {
      out[m[1]!] = Function(`return ${m[2]}`)();
    }
  }
  return out;
}
const RUNTIME = '<server list>';
function extensionsOf(id: string, dir: string): unknown[] {
  if (id === 'ace') return [];
  const src = read(`server/src/services/backends/${dir}/index.ts`);
  let at = src.search(/const \w+: BackendExtensionParam\[\]\s*=/);
  at = at >= 0 ? src.indexOf('=', at) : src.search(/extensions:\s*\[/);
  const text = arrayAfter(src, at).replace(/\bas const\b/g, '').replace(/\bsatisfies \w+/g, '');
  const consts = constantsOf(`server/src/services/backends/${dir}`);
  const scope = new Proxy({}, {
    has: (_, k) => typeof k === 'string' && !(k in globalThis),
    get: (_, k) => k === Symbol.unscopables ? undefined
      : typeof k === 'string' && k in consts ? consts[k]
      : k === 'sd' ? (_s: string, _f: string, fallback: unknown) => fallback
      : () => RUNTIME,
  });
  const list = Function('scope', `with (scope) { return ${text}; }`)(scope) as unknown[];
  return list.flat().map(x => { const { hint: _h, section_hint: _s, ...rest } = x as Record<string, unknown>; return rest; });
}

/* ---- the backend-prefixed fields each backend reads ---- */
function readsOf(dir: string): string[] {
  const found = new Set<string>();
  const root = path.join(APP, 'server/src/services/backends', dir);
  for (const f of fs.readdirSync(root).filter(f => f.endsWith('.ts') && !f.includes('.test.'))) {
    const src = fs.readFileSync(path.join(root, f), 'utf8');
    for (const m of src.matchAll(new RegExp(`(?:\\.|['"\`])((?:${FIELD_PREFIXES.join('|')})[A-Z][A-Za-z0-9]*|samplerPluginsEnabled)\\b`, 'g'))) found.add(m[1]!);
  }
  return [...found].sort();
}

/* ---- the GENERATION_DEFAULTS fields each engine's code reads. A field counts as read when the engine's own code
   (its backend folder; on ACE-Step also translateParams.ts) or the code every engine runs (services/generation
   other than the resolver) names it, or names a request field resolveGenerationParams() makes from it: the
   resolver's return object is read with the compiler, each property with the input fields its expression uses,
   through the local constants it uses ---- */
function resolverInputs(): Map<string, Set<string>> {
  const file = 'server/src/services/generation/defaults.ts';
  const src = ts.createSourceFile(file, read(file), ts.ScriptTarget.ES2022, true);
  const locals = new Map<string, ts.Node>(), out = new Map<string, Set<string>>();
  const fn = src.statements.find((x): x is ts.FunctionDeclaration => ts.isFunctionDeclaration(x) && x.name?.text === 'resolveGenerationParams')!;
  let ret: ts.ObjectLiteralExpression | undefined;
  const visit = (n: ts.Node): void => {
    if (ts.isVariableDeclaration(n) && ts.isIdentifier(n.name) && n.initializer) locals.set(n.name.text, n.initializer);
    if (ts.isReturnStatement(n) && n.expression && ts.isObjectLiteralExpression(n.expression)) ret = n.expression;
    ts.forEachChild(n, visit);
  };
  visit(fn.body!);
  const depsOf = (n: ts.Node, seen = new Set<string>()): Set<string> => {
    const acc = new Set<string>();
    const walk = (x: ts.Node): void => {
      if (ts.isPropertyAccessExpression(x) && ts.isIdentifier(x.expression) && x.expression.text === 's') acc.add(x.name.text);
      else if (ts.isIdentifier(x) && locals.has(x.text) && !seen.has(x.text)) { seen.add(x.text); depsOf(locals.get(x.text)!, seen).forEach(d => acc.add(d)); }
      ts.forEachChild(x, walk);
    };
    walk(n); return acc;
  };
  for (const p of ret!.properties) if (ts.isPropertyAssignment(p) && ts.isIdentifier(p.name)) out.set(p.name.text, depsOf(p.initializer));
  return out;
}
const RESOLVER = resolverInputs();
const SHARED = fs.readdirSync(path.join(APP, 'server/src/services/generation')).filter(f => f.endsWith('.ts') && !f.includes('.test.') && !['defaults.ts', 'translateParams.ts'].includes(f)).map(f => `server/src/services/generation/${f}`);
function fieldsOf(id: string, dir: string): string[] {
  const own = fs.readdirSync(path.join(APP, 'server/src/services/backends', dir)).filter(f => f.endsWith('.ts') && !f.includes('.test.')).map(f => `server/src/services/backends/${dir}/${f}`);
  const text = [...own, ...(id === 'ace' ? ['server/src/services/generation/translateParams.ts'] : []), ...SHARED].map(read).join('\n');
  const named = (k: string) => new RegExp(`\\b${k}\\b`).test(text);
  const read2 = new Set<string>();
  for (const [outKey, inputs] of RESOLVER) if (named(outKey)) inputs.forEach(k => read2.add(k));
  return Object.keys(GENERATION_DEFAULTS).filter(k => read2.has(k) || (!RESOLVER.has(k) ? false : named(k))).sort();
}

/* ---- plugin ids ---- */
function registry(file: string): string[] {
  return [...read(file).matchAll(/\{\s*"([a-z0-9_]+)"\s*,\s*"/g)].map(m => m[1]!);
}
function lua(kind: string): string[] {
  const dir = path.join(APP, 'plugins', kind);
  return fs.readdirSync(dir).filter(f => f.endsWith('.lua')).flatMap(f => {
    const m = /\bname\s*=\s*"([^"]+)"/.exec(fs.readFileSync(path.join(dir, f), 'utf8'));
    return m ? [m[1]!] : [];
  });
}
const plugins = {
  solver: [...registry('engine/src/solvers/solver-registry.h'), ...lua('solvers')],
  /* the three parameterised schedules are written "beta:<a>:<b>", "power:<p>", "composite:<A>+<B>:<x>:<split>";
     their bare names are what the picker lists */
  scheduler: [...registry('engine/src/schedulers/scheduler-registry.h'), 'beta', 'composite', ...lua('schedulers')],
  guidance: [...registry('engine/src/guidance/guidance-registry.h'), ...lua('guidance')],
  postprocess: lua('postprocess'),
};

/* ---- model buckets, as each backend's models() returns them (read by hand; the names are stable) ---- */
const buckets = {
  ace: ['dit', 'lm', 'vae', 'embedding'],
  'minimax-m3': ['lm', 'dit', 'depth', 'cond', 'voc'],
  yue2: ['lm', 'vae', 'ar', 'nar'],
  sa3: ['dit'],
  heartmula: ['variant', 'global', 'local', 'flow', 'scalar_decoder', 'lmAdapter'],
};

/* ---- the existing UI's sliders: ACE-Step has no manifest, so its ranges are stated only by these. For every
   GENERATION_DEFAULTS field, the min, max and step of each JSX element whose value is that field. ---- */
function sliders(): Record<string, [number | null, number | null, number | null][]> {
  const files: string[] = [];
  const walkDir = (d: string) => { for (const f of fs.readdirSync(d, { withFileTypes: true })) { const p = path.join(d, f.name); if (f.isDirectory()) walkDir(p); else if (f.name.endsWith('.tsx') && !f.name.includes('.test.')) files.push(p); } };
  walkDir(path.join(APP, 'ui/src/components'));
  const texts = files.map(f => fs.readFileSync(f, 'utf8'));
  const out: Record<string, [number | null, number | null, number | null][]> = {};
  for (const field of Object.keys(GENERATION_DEFAULTS)) {
    const re = new RegExp(`<[A-Z][A-Za-z]*\\b[^<>]*?value=\\{(?:gp\\.|g\\.|params\\.|p\\.|s\\.)?${field}\\}[^<>]*?>`, 'gs');
    const pick = (h: string, a: string) => { const m = new RegExp(`\\b${a}=\\{(-?[0-9.]+)\\}`).exec(h); return m ? +m[1]! : null; };
    for (const t of texts) for (const m of t.matchAll(re)) {
      const r: [number | null, number | null, number | null] = [pick(m[0], 'min'), pick(m[0], 'max'), pick(m[0], 'step')];
      if (r[0] === null && r[1] === null) continue;
      (out[field] ||= []).push(r);
    }
  }
  return out;
}

/* ---- the values a field takes, where the source spells them ---- */
function values(): Record<string, string[]> {
  const out: Record<string, Set<string>> = {};
  const add = (field: string, list: string) => { for (const m of list.matchAll(/'([A-Za-z0-9_.-]+)'/g)) (out[field] ||= new Set()).add(m[1]!); };
  const LINE = /^\s+(\w+)\??:\s*(?:'[^']*' as |z\.unknown\(\)\.optional\(\),\s*\/\/\s*)?('[A-Za-z0-9_.-]+'(?:\s*\|\s*'[A-Za-z0-9_.-]+')+)/;
  for (const file of ['server/src/contracts/generationControls.ts', 'ui/src/types.ts']) {
    for (const line of read(file).split('\n')) { const m = LINE.exec(line); if (m && m[1]! in GENERATION_DEFAULTS) add(m[1]!, m[2]!); }
  }
  const GUARD = /const (\w+) = params\.(\w+) \?\? ('[^']+');\s*\n\s*if \(([^)]*)\)/g;
  for (const { backend } of ENGINES) {
    const dir = path.join(APP, 'server/src/services/backends', backend);
    for (const f of fs.readdirSync(dir).filter(f => f.endsWith('.ts') && !f.includes('.test.'))) {
      for (const m of fs.readFileSync(path.join(dir, f), 'utf8').matchAll(GUARD)) {
        const allowed = [...m[4]!.matchAll(new RegExp(`${m[1]} !== ('[^']+')`, 'g'))].map(x => x[1]);
        if (allowed.length) add(m[2]!, [m[3], ...allowed].join(' | '));
      }
    }
  }
  return Object.fromEntries(Object.entries(out).sort().map(([k, v]) => [k, [...v].sort()]));
}

/* ---- each backend's batch and duration limits (capabilities().core) ---- */
function core(): Record<string, { batchMax: number | null; durationMax: number | null }> {
  return Object.fromEntries(ENGINES.map(e => {
    const src = read(`server/src/services/backends/${e.backend}/index.ts`), consts = constantsOf(`server/src/services/backends/${e.backend}`);
    const num = (re: RegExp) => { const m = re.exec(src); if (!m) return null; const v = /^\d+$/.test(m[1]!) ? +m[1]! : consts[m[1]!]; return typeof v === 'number' ? v : null; };
    return [e.id, { batchMax: num(/batch:\s*\{\s*max:\s*(\w+)\s*\}/), durationMax: num(/duration:\s*\{\s*max:\s*(\w+)\b/) }];
  }));
}

const out = {
  note: 'Generated by tools/make-fixtures.ts from the real application\'s source. Do not edit by hand.',
  sliders: sliders(),
  generationDefaults: GENERATION_DEFAULTS,
  createContent: createContentFromDraft({}),
  /* the fields CreateContent declares, the optional ones included */
  createContentFields: [...read('ui/src/components/create/createContent.ts').match(/export interface CreateContent \{([\s\S]*?)\n\}/)![1]!.matchAll(/(\w+)\??:/g)].map(m => m[1]),
  extensions: Object.fromEntries(Object.entries(BACKENDS).map(([id, dir]) => [id, extensionsOf(id, dir)])),
  reads: Object.fromEntries(Object.entries(BACKENDS).map(([id, dir]) => [id, readsOf(dir)])),
  fields: Object.fromEntries(Object.entries(BACKENDS).map(([id, dir]) => [id, fieldsOf(id, dir)])),
  plugins,
  buckets,
  values: values(),
  core: core(),
  /* the Create intent's parts beside its params (server/src/contracts/resolution.ts createIntentSchema) */
  intent: {
    compose: [...read('server/src/contracts/resolution.ts').match(/compose: z\.object\(\{([\s\S]*?)\}\)/)![1]!.matchAll(/(\w+): z\./g)].map(m => m[1]),
  },
};
const file = path.join(HERE, 'fixtures', 'real-app.json');
fs.mkdirSync(path.dirname(file), { recursive: true });
fs.writeFileSync(file, JSON.stringify(out, null, 1).replace(/\n/g, '\r\n') + '\r\n');
console.log(`wrote ${path.relative(process.cwd(), file)}: ${Object.entries(out.extensions).map(([k, v]) => `${k} ${(v as unknown[]).length}`).join(', ')} extensions; `
  + `${Object.entries(plugins).map(([k, v]) => `${k} ${v.length}`).join(', ')} plugins`);
