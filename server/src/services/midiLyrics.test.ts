import assert from 'node:assert/strict';
import test from 'node:test';
import { addMidiLyrics } from './midiLyrics.js';

function track(body: number[]): Buffer {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(body.length);
  return Buffer.concat([Buffer.from('MTrk'), length, Buffer.from(body)]);
}

function fixture(): Buffer {
  return Buffer.concat([
    Buffer.from([0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 1, 0, 2, 1, 0xe0]),
    track([0, 0xff, 0x51, 3, 0x07, 0xa1, 0x20, 0, 0xff, 0x2f, 0]),
    track([0, 0x90, 60, 100, 0x60, 0x80, 60, 0, 0, 0xff, 0x2f, 0]),
  ]);
}

function readVlq(bytes: Buffer, start: number): [number, number] {
  let value = 0;
  let pos = start;
  while (true) {
    const byte = bytes[pos++];
    value = (value << 7) | (byte & 0x7f);
    if (!(byte & 0x80)) return [value, pos];
  }
}

function decodeMeta(bytes: Buffer): { tick: number; type: number; text: string }[] {
  const end = 22 + bytes.readUInt32BE(18);
  const events: { tick: number; type: number; text: string }[] = [];
  let pos = 22;
  let tick = 0;
  while (pos < end) {
    const [delta, afterDelta] = readVlq(bytes, pos);
    tick += delta;
    assert.equal(bytes[afterDelta], 0xff);
    const type = bytes[afterDelta + 1];
    const [length, afterLength] = readVlq(bytes, afterDelta + 2);
    events.push({ tick, type, text: bytes.toString('utf8', afterLength, afterLength + length) });
    pos = afterLength + length;
  }
  assert.equal(pos, end);
  return events;
}

test('adds lyric events at zero and fractional ticks, in timestamp order', () => {
  const result = addMidiLyrics(fixture(), '[00:01.25] Second\n[00:00.00] First');
  const events = decodeMeta(result);
  assert.deepEqual(events.map(({ tick, type }) => [tick, type]), [
    [0, 0x51], [0, 0x05], [1200, 0x05], [1200, 0x2f],
  ]);
  assert.deepEqual(events.filter(e => e.type === 0x05).map(e => e.text), ['First', 'Second']);
});

test('keeps equal-time lyric order and UTF-8 text with variable-length payloads', () => {
  const long = 'é'.repeat(70);
  const result = addMidiLyrics(fixture(), `[00:02.00] ${long}\n[00:02.00] 星の声`);
  const lyrics = decodeMeta(result).filter(e => e.type === 0x05);
  assert.deepEqual(lyrics.map(e => e.tick), [1920, 1920]);
  assert.deepEqual(lyrics.map(e => e.text), [long, '星の声']);
});

test('preserves existing meta events while inserting lyrics between their ticks', () => {
  const input = Buffer.concat([
    fixture().subarray(0, 14),
    track([0, 0xff, 0x51, 3, 0x07, 0xa1, 0x20,
      0x60, 0xff, 0x01, 3, 0x4f, 0x6c, 0x64,
      0x60, 0xff, 0x2f, 0]),
    fixture().subarray(22 + fixture().readUInt32BE(18)),
  ]);
  const events = decodeMeta(addMidiLyrics(input, '[00:00.15] New'));
  assert.deepEqual(events.map(e => [e.tick, e.type]), [
    [0, 0x51], [96, 0x01], [144, 0x05], [192, 0x2f],
  ]);
  assert.equal(events[1].text, 'Old');
});

test('skips malformed lines, empty text and section labels', () => {
  const result = addMidiLyrics(fixture(), 'bad\n[00:01.00] [Chorus]\n[00:02.00]\n[00:03.00] Keep');
  assert.deepEqual(decodeMeta(result).filter(e => e.type === 0x05).map(e => e.text), ['Keep']);
});

test('missing, empty and wholly malformed sidecars preserve the input bytes', () => {
  const input = fixture();
  for (const text of ['', '\n', 'bad\n[Chorus]\n[00:03.00] [Verse]']) {
    assert.strictEqual(addMidiLyrics(input, text), input);
  }
});

test('note tracks remain byte-identical and end-of-track stays last', () => {
  const input = fixture();
  const result = addMidiLyrics(input, '[00:04.00] Later');
  const originalNotes = input.subarray(22 + input.readUInt32BE(18));
  const resultNotes = result.subarray(22 + result.readUInt32BE(18));
  assert.deepEqual(resultNotes, originalNotes);
  assert.deepEqual(decodeMeta(result).at(-1), { tick: 3840, type: 0x2f, text: '' });
  assert.equal(result.readUInt32BE(18), resultNotes.byteOffset - result.byteOffset - 22);
});

test('rejects timing headers or tempo that differ from ace-midi', () => {
  for (const offset of [9, 13, 27]) {
    const input = fixture();
    input[offset] ^= 1;
    assert.throws(() => addMidiLyrics(input, '[00:01.00] Line'));
  }
});
