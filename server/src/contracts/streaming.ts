// contracts/streaming.ts — the two live audio streams and their session
// links (docs/dev/frontend-media.md).
//
//   STORM: POST /api/generate/storm/stream     one complete WAV per slot
//   MM3:   GET  /api/generate/mm3/stream/:id   one complete WAV per window
//
// Both bodies are complete RIFF/WAV files concatenated end to end, with no
// other framing. Node forwards them unchanged. Crossfades, scheduling, the
// mix and device capture are the player's business and happen only in the
// client. Node's copy of a stream (analysis, canonical recording) is the
// stream session named in the X-Stream-Session response header
// (contracts/streamSessions.ts).

/** Response header naming the stream's Node session. Absent when Node could
 *  not open one; the audio is unaffected. */
export const STREAM_SESSION_HEADER = 'X-Stream-Session';

// ── STORM ──

/** POST /api/generate/storm/stream. Any Create generation parameter is also
 *  accepted and becomes the base of every slot; these are the stream's own. */
export interface StormStreamStart extends Record<string, unknown> {
  /** Keys the stream for control and stop. Default 'default'; DJ mode runs one per deck. */
  streamId?: string;
  /** Base seed; slot n uses seed + n unless seed_lock is on. Random when absent. */
  seed?: number;
  /** Keep the DiT and VAE loaded between slots. */
  coResident?: boolean;
  pluginParams?: Record<string, number | string>;
}

/** POST /api/generate/storm/control: changes for the next slot onwards.
 *  Fields of the wrong type are ignored. `next_*`, `seed`, `prompt` and
 *  `lyrics` apply to one slot; `stick_*` persist until set to null. */
export interface StormControl {
  streamId?: string;
  guidance_scale?: number;
  lss_strength?: number;
  inference_steps?: number;
  duration?: number;
  bpm?: number;
  next_bpm?: number;
  next_duration?: number;
  seed_lock?: boolean;
  seed?: number;
  prompt?: string;
  lyrics?: string;
  stick_prompt?: string | null;
  stick_lyrics?: string | null;
  /** True holds slot generation (the player's buffer is full); false resumes. */
  stream_pause?: boolean;
  infer_method?: string;
  scheduler?: string;
  guidance_mode?: string;
  plugin_params?: Record<string, number | string>;
}
export interface StormControlResponse { ok: true; streamId: string }
/** POST /api/generate/storm/stop: cancels the slot in flight and ends the response. */
export interface StormStop { streamId?: string }
export interface StormStopResponse { ok: true }

/** STORM start refusals, as JSON before any audio: 503 engine not ready, 409
 *  generation jobs active. After audio starts, the stream ends by closing the
 *  response (stop, a failed slot, a slot timeout); there is no error frame. */
export interface StreamError { error: string }

// ── MM3 ──

/** The /api/generate/status/:id fields that say when and how to open
 *  GET /api/generate/mm3/stream/:id?take=N. */
export interface Mm3StreamStatus {
  /** Open the stream only once this is true. */
  mm3_streaming: boolean;
  mm3_interleaved: boolean | null;
  /** The render's full length in seconds, known before audio arrives. */
  mm3_duration: number | null;
  /** Takes rendered together; open one stream per take (`take` 0..n-1). */
  mm3_takes: number;
  mm3_take_seeds: number[] | null;
}

/** MM3 refusals, as JSON: 404 unknown job; 409 not handed to the engine yet,
 *  not a streaming job, or the engine refused (finished, or the take already
 *  has a reader: the engine allows one); 502 the engine stream failed. */
export type Mm3StreamError = StreamError;

// ── Framing ──

/** Largest WAV a reader accepts before treating the size field as corrupt. */
export const MAX_STREAM_WAV_BYTES = 500_000_000;
const RIFF = [0x52, 0x49, 0x46, 0x46];

/** Reference reader for a concatenated-WAV stream: push bytes as they
 *  arrive, in any chunking, and take back each complete WAV once all of it
 *  is in. Several WAVs in one read come out together; a WAV split across
 *  reads, at any byte including inside its header, comes out when the last
 *  byte lands. Bytes before a "RIFF" are skipped, and so is a "RIFF" whose
 *  size field is impossible, so a damaged stream resynchronises rather than
 *  stalling. It holds at most one incomplete WAV (bounded by
 *  MAX_STREAM_WAV_BYTES). Node's own copy uses the same rule
 *  (services/streamSessions/wav.ts). */
export class WavFrameReader {
  private buf = new Uint8Array(0);
  /** Bytes dropped while resynchronising. */
  skipped = 0;

  push(chunk: Uint8Array): Uint8Array[] {
    const merged = new Uint8Array(this.buf.length + chunk.length);
    merged.set(this.buf);
    merged.set(chunk, this.buf.length);
    this.buf = merged;
    const out: Uint8Array[] = [];
    for (;;) {
      const start = this.findRiff();
      if (start < 0) {
        // Keep the last 3 bytes: they may be the start of a "RIFF".
        const keep = Math.min(3, this.buf.length);
        this.skipped += this.buf.length - keep;
        this.buf = this.buf.slice(this.buf.length - keep);
        return out;
      }
      if (start > 0) { this.skipped += start; this.buf = this.buf.slice(start); }
      if (this.buf.length < 8) return out;
      const total = new DataView(this.buf.buffer, this.buf.byteOffset, this.buf.byteLength).getUint32(4, true) + 8;
      if (total < 44 || total > MAX_STREAM_WAV_BYTES) { this.skipped += 4; this.buf = this.buf.slice(4); continue; }
      if (this.buf.length < total) return out;
      out.push(this.buf.slice(0, total));
      this.buf = this.buf.slice(total);
    }
  }

  /** At end of stream (EOF, stop, disconnect): the bytes of a WAV that never
   *  completed. A reader discards them; they are not playable audio. */
  end(): { truncatedBytes: number } {
    const truncatedBytes = this.buf.length;
    this.buf = new Uint8Array(0);
    return { truncatedBytes };
  }

  private findRiff(): number {
    for (let i = 0; i + 4 <= this.buf.length; i++) {
      if (this.buf[i] === RIFF[0] && this.buf[i + 1] === RIFF[1] && this.buf[i + 2] === RIFF[2] && this.buf[i + 3] === RIFF[3]) return i;
    }
    return -1;
  }
}
