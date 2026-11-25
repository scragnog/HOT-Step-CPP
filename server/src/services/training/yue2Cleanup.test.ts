import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { dirBytes } from './yue2Cleanup.js';

test('dirBytes counts a hard-linked file once per shared set, and not when the kept rung holds it', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-cleanup-'));
  try {
    const [keep, a, b] = ['keep', 'a', 'b'].map(name => { const d = path.join(root, name); fs.mkdirSync(d); return d; });
    fs.writeFileSync(path.join(a, 'ar.safetensors'), Buffer.alloc(1000));
    fs.linkSync(path.join(a, 'ar.safetensors'), path.join(b, 'ar.safetensors'));
    fs.writeFileSync(path.join(a, 'nar.safetensors'), Buffer.alloc(10));
    fs.writeFileSync(path.join(b, 'nar.safetensors'), Buffer.alloc(10));
    assert.equal(dirBytes(a) + dirBytes(b), 2020);                 // no set: as before
    const seen = new Set<string>();
    assert.equal(dirBytes(a, seen) + dirBytes(b, seen), 1020);     // the link counts once
    fs.linkSync(path.join(a, 'ar.safetensors'), path.join(keep, 'ar.safetensors'));
    const linked = new Set<string>(); dirBytes(keep, linked);
    assert.equal(dirBytes(a, linked) + dirBytes(b, linked), 20);   // the kept rung's AR is not freed
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});
