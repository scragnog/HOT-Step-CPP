import { test } from 'node:test';
import assert from 'node:assert/strict';

import { buildYue2TokenizeArgs, yue2CodesReady, yue2TokenizerOfFile } from './yue2Tokenize.js';

test('a cache is ready only for the tokenizer that made its codes', () => {
  const done = (tokenizer?: string) => ({ done: true, status: tokenizer === undefined ? null : { tokenizer } });
  assert.equal(yue2CodesReady(done('yue2-tok-f16.gguf'), 'v9'), true);
  assert.equal(yue2CodesReady(done('yue2-tok-f16.gguf'), 'wannabe-v2'), false);
  assert.equal(yue2CodesReady(done('yue2-wannabe-v2-f16.gguf'), 'wannabe-v2'), true);
  assert.equal(yue2CodesReady(done('yue2-wannabe-v2-f16.gguf'), 'wannabe-v1'), false);
  assert.equal(yue2CodesReady(done(), 'v9'), true);  // stamped before the choice existed
  assert.equal(yue2CodesReady({ done: false }, 'v9'), false);
  assert.equal(yue2TokenizerOfFile('C:/m/yue2/yue2-wannabe-v1-f16.gguf'), 'wannabe-v1');
  assert.equal(yue2TokenizerOfFile('something-else.gguf'), '');
});

test('v9 keeps engine discovery; a WannabeTokenizer is named with --tok', () => {
  const base = { manifest: 'm.json', decode: 'auto' as const, only: '', limit: 0, force: false, datasetSlug: 'd' };
  assert.ok(!buildYue2TokenizeArgs({ ...base, tokenizer: 'v9' }).includes('--tok'));
  assert.ok(buildYue2TokenizeArgs({ ...base, tokenizer: 'wannabe-v2' }).includes('--tok'));
});
