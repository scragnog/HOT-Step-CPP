// Concatenated-WAV stream framing (docs/dev/frontend-media.md): the reference
// reader in contracts/streaming.ts and Node's own copy in streamSessions, run
// against the same byte streams.
import test from 'node:test';
import assert from 'node:assert/strict';
import { WavFrameReader } from './streaming.js';
import { WavSplitter, wavHeader } from '../services/streamSessions/wav.js';

function wav(frames: number, fill: number): Uint8Array {
  const data = new Uint8Array(frames * 4).fill(fill);
  const out = new Uint8Array(44 + data.length);
  out.set(wavHeader({ audioFormat: 1, channels: 2, sampleRate: 48000, bitsPerSample: 16 }, data.length));
  out.set(data, 44);
  return out;
}
const concat = (...parts: Uint8Array[]) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
};
const same = (a: Uint8Array[], b: Uint8Array[]) =>
  assert.deepEqual(a.map(x => Buffer.from(x).toString('hex')), b.map(x => Buffer.from(x).toString('hex')));

const A = wav(3, 0x11), B = wav(7, 0x22), C = wav(1, 0x33);
const STREAM = concat(A, B, C);
const readers = {
  reference: () => { const r = new WavFrameReader(); return (c: Uint8Array) => r.push(c); },
  node: () => { const r = new WavSplitter(); return (c: Uint8Array) => r.push(c); },
};

for (const [name, make] of Object.entries(readers)) {
  test(`${name}: every single split point yields the same three WAVs`, () => {
    for (let p = 0; p <= STREAM.length; p++) {
      const push = make();
      same([...push(STREAM.subarray(0, p)), ...push(STREAM.subarray(p))], [A, B, C]);
    }
  });

  test(`${name}: splits at every pair of header byte boundaries, and byte-by-byte`, () => {
    const headerBytes = [A.length, A.length + B.length].flatMap(start => Array.from({ length: 45 }, (_, i) => start + i)).concat(Array.from({ length: 45 }, (_, i) => i));
    for (const p of headerBytes) for (const q of headerBytes) {
      if (q <= p) continue;
      const push = make();
      same([...push(STREAM.subarray(0, p)), ...push(STREAM.subarray(p, q)), ...push(STREAM.subarray(q))], [A, B, C]);
    }
    const push = make();
    const out: Uint8Array[] = [];
    for (const byte of STREAM) out.push(...push(Uint8Array.of(byte)));
    same(out, [A, B, C]);
  });

  test(`${name}: several WAVs in one read come out together`, () => {
    same(make()(STREAM), [A, B, C]);
  });

  test(`${name}: a truncated last WAV is never emitted`, () => {
    for (const cut of [1, 4, 30, C.length - 1]) same(make()(STREAM.subarray(0, STREAM.length - cut)), [A, B]);
  });

  test(`${name}: leading garbage and an impossible size field resynchronise on the next RIFF`, () => {
    same(make()(concat(Uint8Array.of(0, 1, 2, 0x52, 0x49), A, B)), [A, B]);
    const broken = A.slice();
    new DataView(broken.buffer).setUint32(4, 2, true);           // size 2: impossible
    same(make()(concat(broken, B, C)), [B, C]);
  });
}

test('the reference reader reports truncated bytes at end of stream and resets', () => {
  const r = new WavFrameReader();
  same(r.push(STREAM.subarray(0, STREAM.length - 5)), [A, B]);
  assert.deepEqual(r.end(), { truncatedBytes: C.length - 5 });
  same(r.push(C), [C]);
  assert.deepEqual(r.end(), { truncatedBytes: 0 });
  const g = new WavFrameReader();
  g.push(Uint8Array.of(9, 9, 9, 9, 9));
  assert.equal(g.skipped, 2, 'keeps the last three bytes in case they start a RIFF');
});
