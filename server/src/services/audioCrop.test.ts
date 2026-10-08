import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { wavDurationSec } from './audioCrop.js';

/** A minimal RIFF/WAVE file: `format` 1 = PCM, 3 = IEEE float. */
function wav(file: string, opts: { format: number; bits: number; rate: number; channels: number; frames: number }): void {
  const blockAlign = opts.channels * opts.bits / 8;
  const dataSize = opts.frames * blockAlign;
  const b = Buffer.alloc(44 + dataSize);
  b.write('RIFF', 0); b.writeUInt32LE(36 + dataSize, 4); b.write('WAVE', 8);
  b.write('fmt ', 12); b.writeUInt32LE(16, 16); b.writeUInt16LE(opts.format, 20); b.writeUInt16LE(opts.channels, 22);
  b.writeUInt32LE(opts.rate, 24); b.writeUInt32LE(opts.rate * blockAlign, 28); b.writeUInt16LE(blockAlign, 32); b.writeUInt16LE(opts.bits, 34);
  b.write('data', 36); b.writeUInt32LE(dataSize, 40);
  fs.writeFileSync(file, b);
}

test('wavDurationSec measures the engine\'s float32 output and PCM16 alike', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wavdur-'));
  try {
    // The ACE engine writes a 10 s request with a 47-code plan as 453120
    // stereo float32 frames at 48 kHz: 9.44 s, which is what the song row
    // must say (not the 10 s that was asked for).
    const f32 = path.join(dir, 'f32.wav');
    wav(f32, { format: 3, bits: 32, rate: 48000, channels: 2, frames: 453120 });
    assert.equal(wavDurationSec(f32), 9.44);
    const pcm = path.join(dir, 'pcm.wav');
    wav(pcm, { format: 1, bits: 16, rate: 44100, channels: 2, frames: 44100 * 8 });
    assert.equal(wavDurationSec(pcm), 8);
    assert.equal(wavDurationSec(path.join(dir, 'missing.wav')), 0);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
