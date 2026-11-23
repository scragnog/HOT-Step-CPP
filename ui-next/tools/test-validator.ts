// The validator's own test: plausible mistakes a maintainer could make, one at a time, each on a fresh copy of
// ui-next, with what the type check and the validator say. Every mistake must be caught (a type error or a
// validator error), except the few this format has made impossible, which must now pass.
//
//   node tools/test-validator.ts          all of them; exit 1 if one is not caught
//   node tools/test-validator.ts M07 M12  only these
//
// The list started as the 40 of the format's review (definitions-review, work/mutate.py), rewritten for this
// format; a mistake found later belongs here too.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TMP = path.join(os.tmpdir(), 'ui-next-mutations');
const D = 'src/definitions/';
const PARAMS = '  params: [\n';
type Edit = [file: string, old: string, now: string];
interface Mutation { id: string; says: string; edits: Edit[]; gone?: string }

const M: Mutation[] = [
  { id: 'M01', says: 'a parameter added without `engines` beside its neighbours, drawn on YuE2 only', edits: [
    ['modules/planner.ts', PARAMS, PARAMS + "    { key: 'yp.minp', type: 'number', label: 'Plan min-P', range: [0, 1, 0.01], default: 0.05, wire: 'ext:yue2PlanTopP' },\n"],
    ['modules/planner.ts', "{ part: 'token', param: 'yp.topp', label: '--top-p' },", "{ part: 'token', param: 'yp.topp', label: '--top-p' }, { part: 'token', param: 'yp.minp', label: '--min-p' },"]] },
  { id: 'M02', says: 'a default off the step grid', edits: [
    ['modules/replan.ts', PARAMS, PARAMS + "    { key: 'yue.x', type: 'number', label: 'X', range: [0, 1, 0.1], default: 0.05, wire: 'ext:yue2ComposeRetries' },\n"],
    ['modules/replan.ts', "{ part: 'knob', param: 'yue.tries', label: 'Tries' },", "{ part: 'knob', param: 'yue.tries', label: 'Tries' }, { part: 'knob', param: 'yue.x' },"]] },
  { id: 'M03', says: 'a parameter nobody can change: listed only by a read-out', edits: [
    ['modules/replan.ts', PARAMS, PARAMS + "    { key: 'yue.x', type: 'number', label: 'X', range: [0, 1, 0.1], default: 0, wire: 'ext:yue2ComposeRetries' },\n"],
    ['modules/replan.ts', "reads: ['yue.tries'],", "reads: ['yue.tries', 'yue.x'],"]] },
  { id: 'M04', says: 'a parameter shown only by a node with tiers: []', edits: [
    ['modules/avoid.ts', PARAMS, PARAMS + "    { key: 'avoid.x', type: 'number', label: 'X', range: [0, 1, 0.1], default: 0, wire: 'param:cacheRatio' },\n"],
    ['modules/avoid.ts', '  face: [', "  face: [{ part: 'knob', param: 'avoid.x', tiers: [] }, "]] },
  { id: 'M05', says: 'a child tiered for a tier its parent never shows', edits: [
    ['modules/tempo.ts', "{ part: 'lcdButton', param: 'tempo.trim', label: 'AUTO-TRIM', tiers: ['lab'] },", "{ part: 'lcdButton', param: 'tempo.trim', label: 'AUTO-TRIM', tiers: ['quick'] },"]] },
  { id: 'M06', says: 'a misspelt real name', edits: [['modules/sampler.ts', "wire: 'param:inferenceSteps',", "wire: 'param:inferenceStep',"]] },
  { id: 'M07', says: 'two parameters on one engine sent to one real field', edits: [['pedals/lift.ts', "wire: 'param:slNoiseFloor',", "wire: 'param:slHfMix',"]] },
  { id: 'M08', says: 'invert on a number, scale on a switch', edits: [
    ['pedals/gain.ts', "wire: 'param:gainOffsetDb',", "wire: { to: 'param:gainOffsetDb', invert: true },"],
    ['pedals/ppvae.ts', "wire: 'param:ppVaeReencode'", "wire: { to: 'param:ppVaeReencode', scale: 1000 }"]] },
  { id: 'M09', says: 'an automatic parameter whose wire does not say what auto sends', edits: [['modules/tempo.ts', "        auto: 0,\n        note: 'Auto is 0. createContentParams", "        note: 'Auto is 0. createContentParams"]] },
  { id: 'M10', says: 'wire.auto on a parameter that has no auto', edits: [['pedals/gain.ts', "wire: 'param:gainOffsetDb',", "wire: { to: 'param:gainOffsetDb', auto: -1 },"]] },
  { id: 'M11', says: 'Quick choices off the step grid, and none equal to the default', edits: [['modules/sampler.ts', "choices: [['Loose', 5], ['Balanced', 9], ['Strict', 13]]", "choices: [['Loose', 5.55], ['Balanced', 8.123], ['Strict', 13]]"]] },
  { id: 'M12', says: 'Quick choices missing for one engine the row is on', edits: [['modules/sampler.ts', "        { engines: ['heartmula'], choices: [['Draft', 6], ['Standard', 10], ['High', 20]] },\n", '']] },
  { id: 'M13', says: 'two Quick choices with one word, two with one value', edits: [['modules/sampler.ts', "choices: [['Draft', 8], ['Standard', 12], ['High', 30]]", "choices: [['Draft', 8], ['Draft', 12], ['High', 12]]"]] },
  { id: 'M14', says: 'part options of the wrong type, an unknown look, a power chip\'s on outside the range', edits: [
    ['modules/guidance.ts', "opts: { look: 'apg', xl: 'momentum', yl: 'norm threshold' },", "opts: { look: 'purple', xl: 7, yl: ['a'] },"],
    ['modules/adhere.ts', 'opts: { off: 1, on: 1.5 }', 'opts: { off: 1, on: 99 }']] },
  { id: 'M15', says: 'an option and a read naming parameters that do not exist', edits: [
    ['modules/replan.ts', "opts: { offWhen: 'yue.replan' }", "opts: { offWhen: 'yue.nope' }"],
    ['units/wh.ts', "reads: ['pp.wh.model', 'pp.wh.lang']", "reads: ['pp.wh.modle', 'pp.wh.lang']"]] },
  { id: 'M16', says: 'a named action nothing declares', edits: [['pedals/vst.ts', "name: 'vstHost'", "name: 'vstHots'"]] },
  { id: 'M17', says: '"Insert example" on a parameter with no example; a set action far outside the range', edits: [
    ['modules/latent.ts', "do: 'example', param: 'lat.ts'", "do: 'example', param: 'lat.shift'"],
    ['pedals/tile.ts', "          param: 'samp.plugins',\n          value: true,", "          param: 'samp.steps',\n          value: 100000,"]] },
  { id: 'M18', says: 'conditions outside the range, contradictory, and circular', edits: [
    ['pedals/lift.ts', '      default: 6,', "      default: 6,\n      when: { param: 'pp.lift.den', above: 7, is: 5, not: 5 },"],
    ['pedals/lift.ts', '      default: 0.3,', "      default: 0.3,\n      when: { param: 'pp.lift.shim', above: 0 },"]] },
  { id: 'M19', says: 'a pedal\'s foot switch hidden by `when`', edits: [['pedals/ppvae.ts', "label: 'On', default: false,", "label: 'On', default: false, when: { param: 'pp.ppvae.blend', above: 0.5 },"]] },
  { id: 'M20', says: 'a range end whose partner has another range', edits: [['modules/adhere.ts', "      staysBelow: 'yue.pcfg2',", "      staysBelow: 'yue.pcfg',"]] },
  { id: 'M21', says: 'a range end that stays below itself', edits: [['modules/adhere.ts', "      staysBelow: 'yue.pcfg2',", "      staysBelow: 'yue.pcfg1',"]] },
  { id: 'M22', says: 'a key that collides with another parameter\'s auto switch (tempo.bpm.auto)', gone: 'auto switches are a map of their own, so tempo.bpm.auto is an ordinary key', edits: [
    ['modules/tempo.ts', PARAMS, PARAMS + "    { key: 'tempo.bpm.auto', type: 'boolean', label: 'Follow', default: false, engines: ['ace', 'minimax-m3', 'yue2'], wire: 'ui:x' },\n"],
    ['modules/tempo.ts', "            { part: 'tapTempo', param: 'tempo.bpm' },", "            { part: 'tapTempo', param: 'tempo.bpm' }, { part: 'lcdButton', param: 'tempo.bpm.auto' },"]] },
  { id: 'M23', says: 'two keys that give one translation key (perf.a-b and perf-a.b)', edits: [
    ['modules/perf.ts', PARAMS, PARAMS + "    { key: 'perf.a-b', type: 'boolean', label: 'One', default: false, wire: 'ui:x' },\n    { key: 'perf-a.b', type: 'boolean', label: 'Two', default: false, wire: 'ui:y' },\n"],
    ['modules/perf.ts', '  face: [\n', "  face: [\n    { part: 'pill', param: 'perf.a-b' }, { part: 'pill', param: 'perf-a.b' },\n"]] },
  { id: 'M24', says: 'two modules at one place', edits: [['modules/planner.ts', '  at: 60,', '  at: 80,']] },
  { id: 'M25', says: 'a Lab module under a module that is itself Lab-only', edits: [['modules/replan.ts', "labUnder: 'planner'", "labUnder: 'perf'"]] },
  { id: 'M26', says: 'a server-listed choice with a default the list cannot have', edits: [['modules/sampler.ts', "      default: 'euler',", "      default: 'eulr',"]] },
  { id: 'M27', says: 'a condition on a server-listed choice with a value it cannot have', edits: [['modules/sampler.ts', "when: { param: 'samp.sched', is: 'beta' },", "when: { param: 'samp.sched', is: 'Beta (Custom)' },"]] },
  { id: 'M28', says: 'a control part with nothing bound', edits: [['modules/replan.ts', "{ part: 'knob', param: 'yue.tries', label: 'Tries' },", "{ part: 'knob', param: 'yue.tries', label: 'Tries' }, { part: 'knob' }, { part: 'segmented', label: 'Mode' },"]] },
  { id: 'M29', says: 'terminal flags in another order than their controls', gone: 'a flag is its control\'s label, so it cannot be out of step with it', edits: [
    ['modules/planner.ts', "{ part: 'token', param: 'yp.temp', label: '--temp' },", "{ part: 'token', param: 'yp.temp', label: '--top-k' },"]] },
  { id: 'M30', says: 'kept per engine and saved as song content', edits: [['modules/planner.ts', "  owner: 'song',\n  stored: 'settings',", "  owner: 'engine',\n  stored: 'content',"]] },
  { id: 'M31', says: 'a definition on an engine where none of its parameters or nodes are', edits: [['modules/perf.ts', "engines: ['ace'],", "engines: ['ace', 'sa3'],"]] },
  { id: 'M32', says: 'a pedal thumbnail one unit wide with five knobs', gone: 'the width is worked out from the number of knobs; there is no field to get wrong', edits: [] },
  { id: 'M33', says: 'format words for a value it cannot hold, a template without {v}, negative decimals', edits: [['modules/timbre.ts', '      range: [0, 1, 0.05],\n      default: 1,', "      range: [0, 1, 0.05],\n      format: { words: [{ value: 55, text: 'never' }], template: 'lots', decimals: -3, perSecond: 0 },\n      default: 1,"]] },
  { id: 'M34', says: 'held on a pedal number with no words for the held note', edits: [['pedals/ss.ts', "      default: 8,\n      wire: 'param:stableStepSteps',", "      default: 8,\n      held: true,\n      engines: ['ace', 'minimax-m3', 'yue2', 'heartmula'],\n      per: [{ engines: ['heartmula'], range: [1, 32, 1] }],\n      wire: 'param:stableStepSteps',"]] },
  { id: 'M35', says: 'a strip that can never hold, with empty words', edits: [['modules/takes.ts', "when: { engine: ['sa3', 'heartmula'] }, says: 'One song per run · Generate again to queue another'", "when: { all: [{ engine: ['ace'] }, { engine: ['sa3'] }] }, says: ''"]] },
  { id: 'M36', says: 'an empty label; two options with one label', edits: [
    ['units/q.ts', "{ value: 'mastered', label: 'Mastered' },", "{ value: 'mastered', label: 'Unmastered' },"],
    ['units/q.ts', "label: 'Evaluate',", "label: '',"]] },
  { id: 'M37', says: 'a pedal\'s switch naming another pedal\'s parameter', edits: [['pedals/ppvae.ts', "on: 'pp.ppvae.on',", "on: 'pp.lift.on',"]] },
  { id: 'M38', says: 'a drop zone holding another module\'s state', edits: [['modules/timbre.ts', "      state: ['timbreFile'],\n      opts: { look: 'sleeve', empty:", "      state: ['masterRef'],\n      opts: { look: 'sleeve', empty:"]] },
  { id: 'M39', says: 'state declared and named by no node', edits: [['modules/perf.ts', PARAMS, "  state: [{ key: 'perfThing', kind: 'file', noun: 'the thing', owner: 'song', stored: 'settings', wire: 'ui:x' }],\n" + PARAMS]] },
  { id: 'M40', says: 'a range on one engine that leaves out the base default', edits: [['modules/sampler.ts', "{ engines: ['sa3'], range: [1, 64, 1], default: 8, wire: 'ext:sa3Steps' },", "{ engines: ['sa3'], range: [1, 6, 1], wire: 'ext:sa3Steps' },"]] },
  /* later ones */
  { id: 'M41', says: 'an engine added to a module, with a parameter that differs per engine left to inherit another engine\'s', edits: [['modules/takes.ts', "      engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],\n      held:", "      held:"]] },
  { id: 'M42', says: 'a read-out given a parameter to edit', edits: [['pedals/tile.ts', "{ part: 'lcdValue', reads: ['pp.tile.plugin'],", "{ part: 'lcdValue', param: 'pp.tile.plugin',"]] },
  { id: 'M43', says: 'a word slot that names nothing', edits: [['modules/guidance.ts', "says: '{engine} uses its own guidance'", "says: '{engin} uses its own guidance {value:samp.gide}'"]] },
  { id: 'M44', says: 'a transform name that does not exist', edits: [['modules/sampler.ts', "to: 'param:retakeSeed', via: 'int'", "to: 'param:retakeSeed', via: 'integer'"]] },
  { id: 'M45', says: 'a default that differs from the real app\'s, with no word why', edits: [['pedals/gain.ts', '      default: 0,', '      default: 3,']] },
  { id: 'M46', says: 'a factory revision setting a value its engine cannot take', edits: [['factory.ts', "vals: { 'takes.n': 4, 'samp.retake': 0.3 }", "vals: { 'takes.n': 12, 'samp.retake': 0.3 }"]] },
];

/* ---------------------------------------------------------------- the runner */
function fresh(): void {
  fs.rmSync(TMP, { recursive: true, force: true });
  fs.mkdirSync(TMP, { recursive: true });
  for (const x of ['src', 'tools', 'package.json', 'tsconfig.json']) fs.cpSync(path.join(ROOT, x), path.join(TMP, x), { recursive: true });
  fs.symlinkSync(path.join(ROOT, 'node_modules'), path.join(TMP, 'node_modules'), 'junction');
}
function run(cmd: string, args: string[]): { ok: boolean; out: string } {
  try { return { ok: true, out: execFileSync(cmd, args, { cwd: TMP, encoding: 'utf8', stdio: 'pipe', shell: process.platform === 'win32' }) }; }
  catch (e) { const x = e as { stdout?: string; stderr?: string }; return { ok: false, out: `${x.stdout ?? ''}${x.stderr ?? ''}` }; }
}
const only = process.argv.slice(2);
let bad = 0;
fresh();
const tsc = path.join(ROOT, 'node_modules', 'typescript', 'bin', 'tsc');
const base = [run('node', [tsc, '-p', 'tsconfig.json']), run('node', ['tools/validate-definitions.ts'])];
if (!base.every(r => r.ok)) { console.log('the unchanged copy does not pass:\n' + base.map(r => r.out).join('\n')); process.exit(1); }
for (const m of M.filter(x => !only.length || only.includes(x.id))) {
  fresh();
  for (const [file, old, now] of m.edits) {
    const p = path.join(TMP, D, file), text = fs.readFileSync(p, 'utf8').replace(/\r\n/g, '\n');
    if (!text.includes(old)) { console.log(`${m.id}: the text to change is not in ${file}; update the mutation`); bad++; continue; }
    fs.writeFileSync(p, text.replace(old, now));
  }
  const t = run('node', [tsc, '-p', 'tsconfig.json']), v = run('node', ['tools/validate-definitions.ts']);
  const tErr = t.out.split('\n').filter(l => l.includes('error TS')).map(l => l.replace(/^.*?error /, 'type: '));
  const vErr = v.out.includes('errors (') ? v.out.slice(v.out.indexOf('errors (')).split('\n').slice(1).map(l => l.trim()).filter(Boolean) : [];
  const caught = !t.ok || !v.ok;
  const verdict = m.gone ? (caught ? 'CAUGHT (expected to pass now)' : 'passes, as it should: ' + m.gone) : caught ? 'CAUGHT' : 'NOT CAUGHT';
  if (m.gone ? false : !caught) bad++;
  console.log(`${m.id} ${m.says}\n   -> ${verdict}`);
  [...tErr.slice(0, 2), ...vErr.slice(0, 3)].forEach(l => console.log('      ' + l.slice(0, 220)));
}
fs.rmSync(TMP, { recursive: true, force: true });
console.log(bad ? `\n${bad} not caught.` : '\nEvery mistake caught.');
process.exit(bad ? 1 : 0);
