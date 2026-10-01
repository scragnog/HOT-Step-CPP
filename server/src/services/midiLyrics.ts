// Add timestamped LRC lines to the tempo track written by ace-midi.

interface MetaEvent {
  tick: number;
  bytes: Buffer;
  type: number;
}

const MAX_VLQ = 0x0fffffff;

function readVlq(bytes: Buffer, end: number, start: number): { value: number; next: number } {
  let value = 0;
  let pos = start;
  for (let i = 0; i < 4; i++) {
    if (pos >= end) throw new Error('MIDI meta track has a truncated variable-length value');
    const byte = bytes[pos++];
    value = (value << 7) | (byte & 0x7f);
    if (!(byte & 0x80)) return { value, next: pos };
  }
  throw new Error('MIDI meta track has an oversized variable-length value');
}

function writeVlq(value: number): Buffer {
  if (!Number.isSafeInteger(value) || value < 0 || value > MAX_VLQ) {
    throw new Error('Lyric timestamp exceeds the MIDI tick range');
  }
  const result = [value & 0x7f];
  while (value >>= 7) result.unshift((value & 0x7f) | 0x80);
  return Buffer.from(result);
}

function parseLrc(text: string): MetaEvent[] {
  const lines: MetaEvent[] = [];
  for (const line of text.replace(/\r/g, '').split('\n')) {
    const match = line.match(/^\[(\d+):(\d+)(?:\.(\d+))?\]\s*(.*)$/);
    if (!match) continue;
    const lyric = match[4].trim();
    if (!lyric || /^\[.*\]$/.test(lyric)) continue;
    const minutes = parseInt(match[1], 10);
    const seconds = parseInt(match[2], 10);
    const centiseconds = match[3] ? parseInt(match[3].padEnd(2, '0').slice(0, 2), 10) : 0;
    const tick = Math.round((minutes * 60 + seconds + centiseconds / 100) * 960);
    const utf8 = Buffer.from(lyric, 'utf8');
    lines.push({ tick, bytes: Buffer.concat([Buffer.from([0xff, 0x05]), writeVlq(utf8.length), utf8]), type: 0x05 });
  }
  return lines.sort((a, b) => a.tick - b.tick);
}

/** Return the original bytes for absent lyrics; reject a MIDI layout whose timing is unknown. */
export function addMidiLyrics(midiBytes: Buffer, lrcText: string): Buffer {
  const lyrics = parseLrc(lrcText);
  if (!lyrics.length) return midiBytes;
  if (midiBytes.length < 22 || midiBytes.toString('ascii', 0, 4) !== 'MThd'
    || midiBytes.readUInt32BE(4) !== 6 || midiBytes.readUInt16BE(8) !== 1
    || midiBytes.readUInt16BE(10) < 1 || midiBytes.readUInt16BE(12) !== 480) {
    throw new Error('Expected ace-midi format 1, 480 PPQ header');
  }
  const trackStart = 14;
  if (midiBytes.toString('ascii', trackStart, trackStart + 4) !== 'MTrk') {
    throw new Error('MIDI track 0 is missing');
  }
  const trackLength = midiBytes.readUInt32BE(trackStart + 4);
  const end = trackStart + 8 + trackLength;
  if (end > midiBytes.length) throw new Error('MIDI meta track is truncated');

  const events: MetaEvent[] = [];
  let pos = trackStart + 8;
  let tick = 0;
  let tempoFound = false;
  while (pos < end) {
    const delta = readVlq(midiBytes, end, pos);
    tick += delta.value;
    pos = delta.next;
    if (pos + 2 > end || midiBytes[pos++] !== 0xff) {
      throw new Error('MIDI track 0 contains an unsupported event');
    }
    const type = midiBytes[pos++];
    const length = readVlq(midiBytes, end, pos);
    pos = length.next;
    if (length.value > end - pos) throw new Error('MIDI meta event is truncated');
    if (type === 0x51) {
      if (tick !== 0 || length.value !== 3 || midiBytes.readUIntBE(pos, 3) !== 500000) {
        throw new Error('Expected ace-midi tempo 500000 at tick 0');
      }
      tempoFound = true;
    }
    if (type === 0x2f && (length.value !== 0 || pos !== end)) {
      throw new Error('MIDI end-of-track must be last');
    }
    events.push({ tick, type, bytes: midiBytes.subarray(delta.next, pos + length.value) });
    pos += length.value;
  }
  if (!tempoFound || events.at(-1)?.type !== 0x2f) {
    throw new Error('MIDI track 0 needs ace-midi tempo and end-of-track events');
  }

  const endOfTrack = events.pop()!;
  const ordered = [...events, ...lyrics].sort((a, b) => a.tick - b.tick);
  ordered.push({ ...endOfTrack, tick: Math.max(endOfTrack.tick, ordered.at(-1)?.tick ?? 0) });
  const parts: Buffer[] = [];
  let previousTick = 0;
  for (const event of ordered) {
    parts.push(writeVlq(event.tick - previousTick), event.bytes);
    previousTick = event.tick;
  }
  const body = Buffer.concat(parts);
  const length = Buffer.alloc(4);
  length.writeUInt32BE(body.length);
  return Buffer.concat([midiBytes.subarray(0, trackStart + 4), length, body, midiBytes.subarray(end)]);
}
