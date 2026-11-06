import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

// Runs in a child process with a throwaway data dir: the selection reads the
// settings table, and the real DB must never be touched by a test.
test('MM3 default quants: LM q8_0, depth NVFP4 on sm >= 120 else q8_0, persisted picks win', (t) => {
  const root = fs.mkdtempSync(path.resolve('.tmp-mm3-selection-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const script = `
    import assert from 'node:assert/strict';
    import { initDb, closeDb } from './src/db/database.js';
    import { setSetting } from './src/db/lireekDb.js';
    import { mm3EffectiveSelection } from './src/services/backends/minimax/index.js';
    const v = (...q) => ({ available: q.map(quant => ({ quant, filename: '', bytes: 1 })), selected: '', requested: '' });
    const props = (sm, depth) => ({
      dit_runtime: { supported: true, available: true, backend: 'tensorrt', reason: '', sm },
      variants: { lm: v('f16', 'q8_0', 'NVFP4'), depth, cond: v('f16'), dit: v('f16', 'q8_0'), voc: v('f16') },
    });
    initDb();
    try {
      const all = v('f16', 'q8_0', 'NVFP4');
      assert.deepEqual(mm3EffectiveSelection(props(120, all)), { lm: 'q8_0', depth: 'NVFP4', cond: '', dit: '', voc: '' });
      assert.equal(mm3EffectiveSelection(props(89, all)).depth, 'q8_0');
      assert.equal(mm3EffectiveSelection(props(120, v('f16', 'q8_0'))).depth, 'q8_0', 'NVFP4 not installed');
      assert.equal(mm3EffectiveSelection(props(120, v('f16'))).depth, '', 'nothing on the ladder = auto');
      assert.equal(mm3EffectiveSelection({ variants: { depth: all } }).depth, 'q8_0', 'no sm reported');
      setSetting('mm3_depth_quant', 'f16');
      setSetting('mm3_lm_quant', 'Q6_K-imat');
      const sel = mm3EffectiveSelection(props(120, all));
      assert.equal(sel.depth, 'f16', 'a persisted pick wins');
      assert.equal(sel.lm, 'q8_0', 'a persisted pick whose file is gone falls to the ladder');
      assert.equal(mm3EffectiveSelection(null).lm, 'Q6_K-imat', 'unknown variants keep the persisted pick');
    } finally { closeDb(); }
  `;
  execFileSync(process.execPath, ['--import', 'tsx/esm', '--eval', script], {
    timeout: 30_000,
    env: { ...process.env, DATA_DIR: path.join(root, 'library'),
      ACESTEPCPP_MODELS: path.join(root, 'models'), TRAINING_DIR: path.join(root, 'training') },
    stdio: 'pipe',
  });
});
