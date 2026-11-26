// The part contract, enforced (docs/09, "How a part is written"): the lint refuses a part that reaches past what it is
// given. Three versions of the review's planted part (tools/fixtures/planted/StepSlider.tsx.txt) are linted as if they
// were src/app/parts/StepSlider.tsx:
//   1. as the review wrote it: it imports the store and the core and commits two other modules' values under a made-up
//      undo name: the lint must refuse it, on its imports;
//   2. the same part reaching the store through the page's globals instead (window): refused;
//   3. the same part written with the kit, using only what it is given: accepted.
// And the view's rule against deciding about values (eslint.config.js rule 2): a component that names an engine,
// formats a number or writes English must be refused. The run-time half of the contract (a setter writes only what its
// node binds, against the desk in force) is tested in tools/test-view.ts.
//
//   node tools/test-parts-contract.mjs        exit 1 on a failure

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ESLint } from 'eslint';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const eslint = new ESLint({ cwd: ROOT });
const lint = async (text, file) => (await eslint.lintText(text, { filePath: path.join(ROOT, file) }))[0].messages.filter(m => m.severity === 2);
const fails = [];
const expect = (what, msgs, want) => {
  const got = msgs.map(m => m.ruleId);
  if (want === 'clean' ? msgs.length : !want.every(r => got.includes(r))) fails.push(`${what}: ${want === 'clean' ? 'should pass, got' : `should be refused by ${want.join(', ')}, got`} ${msgs.map(m => `${m.ruleId} (${m.line}: ${m.message.slice(0, 70)})`).join('; ') || 'nothing'}`);
};

const planted = fs.readFileSync(path.join(ROOT, 'tools', 'fixtures', 'planted', 'StepSlider.tsx.txt'), 'utf8');
const PART = 'src/app/parts/StepSlider.tsx';

/* 1. the review's part */
const m1 = await lint(planted, PART);
expect('the planted part (store and core imported, two other modules written)', m1, ['no-restricted-imports']);
const refusedImports = m1.filter(m => m.ruleId === 'no-restricted-imports').map(m => m.line);

/* 2. through the window */
const viaWindow = `import { definePart, type PartView } from '../kit/index.ts';
export function StepSlider(v: PartView) {
  const go = () => { (window as unknown as { useDesk: { getState(): { commit(c: unknown): void } } }).useDesk.getState().commit({ vals: { 'pp.ss.str': 59 }, words: { label: 'Nothing to see' } }); };
  return <div onClick={go} />;
}
definePart('stepSlider' as never, StepSlider);
`;
expect('the planted part reaching the store through window', await lint(viaWindow, PART), ['no-restricted-syntax']);

/* 3. the same part, written with the kit */
const withKit = `// stepSlider: a short slider that snaps to labelled stops.
import type { KeyboardEvent } from 'react';
import { definePart, useSay, type PartView } from '../kit/index.ts';
export function StepSlider(v: PartView) {
  const say = useSay(), p = v.params[0];
  if (!p?.options) return null;
  const at = Math.max(0, p.options.findIndex(o => o.on));
  const key = (e: KeyboardEvent<HTMLDivElement>) => { const d = ({ ArrowRight: 1, ArrowLeft: -1 } as Record<string, number>)[e.key]; if (!d) return; e.preventDefault(); const o = p.options![Math.min(p.options!.length - 1, Math.max(0, at + d))]; if (o) p.set(o.value); };
  return <div className={\`stepsl\${p.changed ? ' changed' : ''}\`} data-ctl={p.key} role="slider" tabIndex={0} aria-label={say(v.label)} onKeyDown={key}>{p.options.map(o => <span key={String(o.value)}>{say(o.label)}</span>)}</div>;
}
definePart('stepSlider' as never, StepSlider);
`;
expect('the same part written with the kit', await lint(withKit, PART), 'clean');

/* the view's rule: a shell component that decides about a value */
const deciding = `import { useSay } from '../i18n/index.ts';
export function Bad({ engine, v }: { engine: string; v: number }) {
  const say = useSay(); void say;
  return <p title="Loudness">{engine === 'sa3' ? 'Length' : v.toLocaleString()} and {parseFloat('1.5')}</p>;
}
`;
const md = await lint(deciding, 'src/app/shell/Bad.tsx');
const why = new Set(md.map(m => m.message.slice(0, 30)));
if (md.length < 4) fails.push(`a component naming an engine, formatting a number, reading one and writing English: ${md.length} refusals (want 4 or more): ${[...why].join(' | ')}`);

console.log(`parts' contract (lint): the planted part refused on its imports (lines ${refusedImports.join(', ')}), through window refused, written with the kit accepted; a component deciding about values refused ${md.length} times`);
if (fails.length) { console.log(`\n${fails.length} failed:`); fails.forEach(f => console.log('  ' + f)); process.exit(1); }
