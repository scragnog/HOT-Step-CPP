// CPU-only checks of the request -> CLI optimizer contract and saved-run defaults.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

const base = new URL('../src/services/training/', import.meta.url);
const source = name => fs.readFileSync(new URL(name, base), 'utf8');
function evaluate(text, globals = {}) {
  const exports = {};
  const js = ts.transpileModule(text, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText;
  vm.runInNewContext(js, { exports, ...globals });
  return exports;
}
const optim = evaluate(source('yue2Optim.ts'));
function declaration(file, name) {
  const text = source(file);
  const ast = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
  const node = ast.statements.find(s => s.name?.text === name ||
    (ts.isVariableStatement(s) && s.declarationList.declarations.some(d => d.name.text === name)));
  assert.ok(node, name);
  return text.slice(node.getStart(ast), node.end);
}
const globals = { ...optim, resolveYue2TrainModels: () => ({ lm: 'fixture.gguf' }),
  YUE2_ADAPTER_STEM: 'nar', YUE2_AR_ADAPTER_STEM: 'ar' };

test('new AR/NAR defaults select Prodigy, AR rank and alpha are 128', () => {
  const ar = evaluate(declaration('yue2ArTrain.ts', 'YUE2_AR_DEFAULTS'), globals).YUE2_AR_DEFAULTS;
  const nar = evaluate(declaration('yue2Train.ts', 'YUE2_NAR_DEFAULTS'), globals).YUE2_NAR_DEFAULTS;
  assert.equal(ar.optimizer, 'prodigy'); assert.equal(nar.optimizer, 'prodigy');
  assert.equal(ar.rank, 128); assert.equal(ar.alpha, 128);
});

test('both CLI builders pass optimizer settings; legacy jobs remain AdamW', () => {
  for (const suffix of ['', 'Ar']) {
    const name = `buildYue2${suffix}TrainArgs`;
    const build = evaluate(declaration(`yue2${suffix}Train.ts`, name), globals)[name];
    for (const optimizer of ['prodigy', 'adamw', 'muon']) {
      const args = build({ ...optim.yue2OptimRequest({ optimizer }), prodigyD0: 2e-6, muonLrScale: 7, muonNsSteps: 3 });
      assert.equal(args[args.indexOf('--optimizer') + 1], optimizer);
      assert.equal(args.includes('--prodigy-d0'), optimizer === 'prodigy');
      assert.equal(args.includes('--muon-lr-scale'), optimizer === 'muon');
      if (optimizer === 'prodigy') assert.equal(args[args.indexOf('--prodigy-d0') + 1], '0.000002');
      if (optimizer === 'muon') assert.equal(args[args.indexOf('--muon-lr-scale') + 1], '7');
    }
    const legacy = build({ resume: true });
    assert.equal(legacy[legacy.indexOf('--optimizer') + 1], 'adamw');
    assert.ok(legacy.includes('--resume'));
  }
});

test('invalid optimizer settings are rejected instead of silently changing optimizer', () => {
  for (const b of [{ optimizer: 'typo' }, { prodigyD0: 0 }, { prodigyD0: Infinity },
    { muonLrScale: -1 }, { muonNsSteps: 1.5 }, { muonNsSteps: 21 }]) {
    assert.throws(() => optim.yue2OptimRequest(b));
  }
  assert.equal(optim.yue2StoredOptimizer({}).optimizer, 'adamw');
  assert.equal(optim.yue2StoredOptimizer({ optimizer: 'muon' }).optimizer, 'muon');
});
