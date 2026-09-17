import assert from 'node:assert/strict';
import test from 'node:test';
import path from 'node:path';
import { yue2JointOutputDirectory } from './yue2AitkRuns.js';

test('joint output uses the configured adapter root, trigger and local timestamp', () => {
  const root = path.resolve('custom-adapters');
  const when = new Date(2026, 8, 17, 13, 14, 15);
  assert.equal(yue2JointOutputDirectory(root, 'alk3_infirmary', when),
    path.join(root, 'yue2-joint-adapters', 'alk3_infirmary_2026-09-17_13-14-15'));
  const escaped = yue2JointOutputDirectory(root, '../../bad / trigger', when);
  assert.equal(path.dirname(escaped), path.join(root, 'yue2-joint-adapters'));
});
