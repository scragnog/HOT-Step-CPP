// wavStream.ts — pulling self-contained WAVs out of a chunked byte stream.
//
// Both of the app's audio streams put concatenated, complete RIFF files on the
// wire rather than a header plus raw frames:
//
//   POST /api/generate/storm/stream       one WAV per STORM slot
//   GET  /api/generate/mm3/stream/:jobId  one WAV per MiniMax-Music3 window
//
// so a reader only has to find the boundaries. This lived inside
// useStreamAudio.ts until the MM3 player needed the same three functions; it
// moved here rather than being copied, because a boundary parser that exists
// twice is a parser that will disagree with itself eventually.

/** Split the leading complete WAV off `buf`, or null if one is not there yet.
 *
 *  Tolerates leading garbage by resynchronising on the next "RIFF" — a stream
 *  that has lost framing is better resynced than silently mis-decoded. */
export function extractWav(buf: Uint8Array): { data: Uint8Array; remaining: Uint8Array } | null {
  if (buf.length < 44) return null;
  if (buf[0] !== 0x52 || buf[1] !== 0x49 || buf[2] !== 0x46 || buf[3] !== 0x46) {
    for (let i = 1; i < buf.length - 4; i++)
      if (buf[i] === 0x52 && buf[i + 1] === 0x49 && buf[i + 2] === 0x46 && buf[i + 3] === 0x46)
        return extractWav(buf.slice(i));
    return null;
  }
  const sz = buf[4] | (buf[5] << 8) | (buf[6] << 16) | (buf[7] << 24);
  const total = sz + 8;
  if (total < 44 || total > 500000000 || buf.length < total) return null;
  return { data: buf.slice(0, total), remaining: buf.slice(total) };
}

/** Sample rate from a classic 44-byte RIFF header (offset 24, LE u32).
 *
 *  Read BEFORE decoding, not after: `decodeAudioData` resamples into the
 *  AudioContext's rate, so an AudioBuffer can never tell you what the source
 *  actually was. MiniMax-Music3 is 44.1 kHz where the rest of the app is 48 —
 *  a context built at the wrong rate resamples every chunk and the exact
 *  sample offsets a gapless splice depends on stop being exact. */
export function wavSampleRate(wav: Uint8Array): number {
  if (wav.length < 28) return 0;
  return wav[24] | (wav[25] << 8) | (wav[26] << 16) | (wav[27] << 24);
}

/** Frame count (samples per channel) from a classic 44-byte RIFF header.
 *
 *  Derived from the data size, bit depth and channel count rather than from
 *  the decoded buffer, so it is available before decode and is unaffected by
 *  any resampling the browser might do. */
export function wavFrameCount(wav: Uint8Array): number {
  if (wav.length < 44) return 0;
  const channels = wav[22] | (wav[23] << 8);
  const bits = wav[34] | (wav[35] << 8);
  const dataSize = wav[40] | (wav[41] << 8) | (wav[42] << 16) | (wav[43] << 24);
  const bytesPerFrame = channels * (bits / 8);
  return bytesPerFrame > 0 ? Math.floor(dataSize / bytesPerFrame) : 0;
}
