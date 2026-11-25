import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { dirBytes, freedBytes } from './yue2Cleanup.js';

test('freedBytes credits a hard-linked file only when every link is deleted', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'yue2-cleanup-'));
  try {
    const [keep, a, b] = ['keep', 'a', 'b'].map(name => { const d = path.join(root, name); fs.mkdirSync(d); return d; });
    fs.writeFileSync(path.join(a, 'ar.safetensors'), Buffer.alloc(1000));
    fs.linkSync(path.join(a, 'ar.safetensors'), path.join(b, 'ar.safetensors'));
    fs.writeFileSync(path.join(a, 'nar.safetensors'), Buffer.alloc(10));
    fs.writeFileSync(path.join(b, 'nar.safetensors'), Buffer.alloc(10));
    assert.equal(dirBytes(a) + dirBytes(b), 2020);              // dirBytes: every name, as before
    assert.equal(freedBytes([a, b]), 1020);                     // another run: both links go, one inode freed
    assert.equal(freedBytes([a]), 10);                          // b still holds the AR
    fs.linkSync(path.join(a, 'ar.safetensors'), path.join(keep, 'ar.safetensors'));
    assert.equal(freedBytes([a, b]), 20);                       // the kept rung's link survives
    assert.equal(freedBytes([a, a, b]), 20);                    // a folder named twice is walked once
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});
