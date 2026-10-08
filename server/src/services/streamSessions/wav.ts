// streamSessions/wav.ts — the engine's stream chunks are self-contained RIFF
// WAVs (STORM: one float32 WAV per slot; MM3: one WAV per window). This reads
// their format and samples, splits a byte stream into whole WAVs and writes
// the canonical concatenation back out.

export interface WavFormat {
  /** 1 = integer PCM, 3 = IEEE float (WAVE_FORMAT_EXTENSIBLE is resolved to its subformat). */
  audioFormat: 1 | 3;
  channels: number;
  sampleRate: number;
  bitsPerSample: number;
}

export interface ParsedWav extends WavFormat {
  /** The data chunk's bytes: interleaved samples exactly as the engine wrote them. */
  data: Uint8Array;
  frames: number;
}

const tag = (b: Uint8Array, at: number) => String.fromCharCode(b[at], b[at + 1], b[at + 2], b[at + 3]);

/** Parse one complete WAV. Walks the chunk list, so fact/LIST chunks and
 *  WAVE_FORMAT_EXTENSIBLE headers (common for float output) are handled. */
export function parseWav(bytes: Uint8Array): ParsedWav {
  if (bytes.length < 12 || tag(bytes, 0) !== 'RIFF' || tag(bytes, 8) !== 'WAVE') throw new Error('Not a RIFF/WAVE chunk');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let format: WavFormat | null = null;
  let at = 12;
  while (at + 8 <= bytes.length) {
    const id = tag(bytes, at);
    const size = view.getUint32(at + 4, true);
    const body = at + 8;
    if (id === 'fmt ') {
      let audioFormat = view.getUint16(body, true);
      // WAVE_FORMAT_EXTENSIBLE: the real format is the first two bytes of the subformat GUID.
      if (audioFormat === 0xfffe && size >= 26) audioFormat = view.getUint16(body + 24, true);
      if (audioFormat !== 1 && audioFormat !== 3) throw new Error(`Unsupported WAV format ${audioFormat}`);
      format = { audioFormat, channels: view.getUint16(body + 2, true), sampleRate: view.getUint32(body + 4, true),
        bitsPerSample: view.getUint16(body + 14, true) };
    } else if (id === 'data') {
      if (!format) throw new Error('WAV data before fmt');
      const bytesPerFrame = format.channels * format.bitsPerSample / 8;
      if (!bytesPerFrame || ![16, 24, 32].includes(format.bitsPerSample) || (format.audioFormat === 3 && format.bitsPerSample !== 32))
        throw new Error(`Unsupported WAV sample layout ${format.audioFormat}/${format.bitsPerSample}`);
      const end = Math.min(bytes.length, body + size);
      const frames = Math.floor((end - body) / bytesPerFrame);
      return { ...format, data: bytes.subarray(body, body + frames * bytesPerFrame), frames };
    }
    at = body + size + (size & 1);
  }
  throw new Error('WAV has no data chunk');
}

/** Channel 0 as floats in [-1, 1), the same values Web Audio's decodeAudioData
 *  gives a browser for these layouts, so analysis matches the client's. */
export function channel0(wav: ParsedWav): Float32Array {
  const out = new Float32Array(wav.frames);
  const view = new DataView(wav.data.buffer, wav.data.byteOffset, wav.data.byteLength);
  const stride = wav.channels * wav.bitsPerSample / 8;
  for (let i = 0, at = 0; i < wav.frames; i++, at += stride) {
    if (wav.audioFormat === 3) out[i] = view.getFloat32(at, true);
    else if (wav.bitsPerSample === 16) out[i] = view.getInt16(at, true) / 32768;
    else if (wav.bitsPerSample === 24) out[i] = ((view.getUint8(at) | (view.getUint8(at + 1) << 8) | (view.getInt8(at + 2) << 16))) / 8388608;
    else out[i] = view.getInt32(at, true) / 2147483648;
  }
  return out;
}

/** Split whole WAVs off the front of a growing byte stream. Same framing rule
 *  as the client's extractWav (ui/src/utils/wavStream.ts): RIFF size + 8, and
 *  resynchronise on the next "RIFF" after garbage. */
export class WavSplitter {
  private buf = new Uint8Array(0);
  push(chunk: Uint8Array): Uint8Array[] {
    const merged = new Uint8Array(this.buf.length + chunk.length);
    merged.set(this.buf); merged.set(chunk, this.buf.length);
    this.buf = merged;
    const out: Uint8Array[] = [];
    for (;;) {
      if (this.buf.length < 44) break;
      if (tag(this.buf, 0) !== 'RIFF') {
        let next = -1;
        for (let i = 1; i < this.buf.length - 4; i++) if (tag(this.buf, i) === 'RIFF') { next = i; break; }
        this.buf = next < 0 ? this.buf.slice(Math.max(0, this.buf.length - 3)) : this.buf.slice(next);
        if (next < 0) break;
        continue;
      }
      const total = new DataView(this.buf.buffer, this.buf.byteOffset).getUint32(4, true) + 8;
      if (total < 44 || total > 500_000_000) { this.buf = this.buf.slice(4); continue; }
      if (this.buf.length < total) break;
      out.push(this.buf.slice(0, total));
      this.buf = this.buf.slice(total);
    }
    return out;
  }
}

/** A canonical WAV header for `dataBytes` bytes of samples in `format`. */
export function wavHeader(format: WavFormat, dataBytes: number): Uint8Array {
  const header = new Uint8Array(44);
  const view = new DataView(header.buffer);
  const write = (at: number, s: string) => { for (let i = 0; i < 4; i++) header[at + i] = s.charCodeAt(i); };
  const blockAlign = format.channels * format.bitsPerSample / 8;
  write(0, 'RIFF'); view.setUint32(4, 36 + dataBytes, true); write(8, 'WAVE');
  write(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, format.audioFormat, true);
  view.setUint16(22, format.channels, true); view.setUint32(24, format.sampleRate, true);
  view.setUint32(28, format.sampleRate * blockAlign, true); view.setUint16(32, blockAlign, true);
  view.setUint16(34, format.bitsPerSample, true);
  write(36, 'data'); view.setUint32(40, dataBytes, true);
  return header;
}

export const sameFormat = (a: WavFormat, b: WavFormat) =>
  a.audioFormat === b.audioFormat && a.channels === b.channels && a.sampleRate === b.sampleRate && a.bitsPerSample === b.bitsPerSample;
