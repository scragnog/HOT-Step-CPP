// latentFormat.ts — HSLAT (HOT-Step LATent) file format reader/writer
//
// HSLAT wraps raw float32 post-DiT latent data with a JSON metadata header:
//
//   ┌────────────────────────────────────────────┐
//   │ Magic: "HSLAT\x01"           (6 bytes)     │
//   │ JSON length: uint32_le       (4 bytes)     │
//   │ JSON metadata: UTF-8 string  (variable)    │
//   │ Padding: zeros to 256-byte boundary        │
//   │ Latent data: float32[T × 64] (T×256 bytes) │
//   └────────────────────────────────────────────┘
//
// The C++ engine only sees raw float32 — all header work happens here.

const HSLAT_MAGIC = Buffer.from('HSLAT\x01', 'ascii'); // 6 bytes
const ALIGNMENT = 256; // latent data starts at a 256-byte boundary

/** Metadata embedded in an HSLAT file. All fields optional. */
export interface HslatMetadata {
  // Musical properties
  duration?: number;      // seconds
  bpm?: number;
  key_scale?: string;     // e.g. "C major"
  time_signature?: string; // e.g. "4/4"

  // Content
  caption?: string;
  lyrics?: string;

  // Generation params
  seed?: number;
  inference_steps?: number;
  guidance_scale?: number;
  shift?: number;
  task_type?: string;

  // Model info
  adapter?: string;
  adapter_scale?: number;
  dit_model?: string;
  vae_model?: string;
  emb_model?: string;

  // Meta
  created_at?: string;    // ISO 8601 timestamp
  [key: string]: unknown; // extensible
}

/** Result of reading an HSLAT file. */
export interface HslatFile {
  metadata: HslatMetadata;
  rawLatent: Buffer;
}

/** Check if a buffer starts with the HSLAT magic bytes. */
export function isHslat(buf: Buffer): boolean {
  if (buf.length < HSLAT_MAGIC.length) return false;
  return buf.subarray(0, HSLAT_MAGIC.length).equals(HSLAT_MAGIC);
}

/**
 * Write an HSLAT file from raw latent bytes and metadata.
 * Returns a Buffer containing the complete HSLAT file.
 */
export function writeHslat(rawLatent: Buffer, metadata: HslatMetadata): Buffer {
  const jsonStr = JSON.stringify(metadata);
  const jsonBuf = Buffer.from(jsonStr, 'utf-8');

  // Header: magic (6) + json_len (4) + json (variable)
  const headerLen = HSLAT_MAGIC.length + 4 + jsonBuf.length;
  // Pad to alignment boundary
  const paddedHeaderLen = Math.ceil(headerLen / ALIGNMENT) * ALIGNMENT;
  const paddingLen = paddedHeaderLen - headerLen;

  const totalLen = paddedHeaderLen + rawLatent.length;
  const out = Buffer.alloc(totalLen, 0); // zeros for padding

  // Write magic
  HSLAT_MAGIC.copy(out, 0);
  // Write JSON length (uint32_le)
  out.writeUInt32LE(jsonBuf.length, HSLAT_MAGIC.length);
  // Write JSON
  jsonBuf.copy(out, HSLAT_MAGIC.length + 4);
  // Padding is already zeros from Buffer.alloc
  // Write raw latent data
  rawLatent.copy(out, paddedHeaderLen);

  return out;
}

/**
 * Read an HSLAT file. Handles both HSLAT-wrapped and raw float32 files.
 *
 * If the file doesn't start with HSLAT magic, it's treated as raw float32
 * (upstream-compatible) with empty metadata.
 */
export function readHslat(buf: Buffer): HslatFile {
  // Backward compat: if no magic, treat entire buffer as raw latent
  if (!isHslat(buf)) {
    return { metadata: {}, rawLatent: buf };
  }

  if (buf.length < HSLAT_MAGIC.length + 4) {
    throw new Error('HSLAT file too short: missing header');
  }

  // Read JSON length
  const jsonLen = buf.readUInt32LE(HSLAT_MAGIC.length);
  const headerLen = HSLAT_MAGIC.length + 4 + jsonLen;

  if (buf.length < headerLen) {
    throw new Error(`HSLAT file truncated: expected ${headerLen} header bytes, got ${buf.length}`);
  }

  // Parse JSON metadata
  const jsonBuf = buf.subarray(HSLAT_MAGIC.length + 4, HSLAT_MAGIC.length + 4 + jsonLen);
  let metadata: HslatMetadata;
  try {
    metadata = JSON.parse(jsonBuf.toString('utf-8'));
  } catch {
    throw new Error('HSLAT file: invalid JSON metadata');
  }

  // Compute padded header length
  const paddedHeaderLen = Math.ceil(headerLen / ALIGNMENT) * ALIGNMENT;

  // Extract raw latent data
  const rawLatent = buf.subarray(paddedHeaderLen);

  // Validate: latent data must be a multiple of 256 bytes (64 floats × 4 bytes)
  if (rawLatent.length > 0 && rawLatent.length % 256 !== 0) {
    throw new Error(
      `HSLAT file: latent data size ${rawLatent.length} is not a multiple of 256 bytes (64 × float32)`
    );
  }

  return { metadata, rawLatent };
}

/**
 * Get the number of latent frames from a raw latent buffer.
 * Each frame is 64 × float32 = 256 bytes.
 */
export function latentFrameCount(rawLatent: Buffer): number {
  return Math.floor(rawLatent.length / 256);
}

/**
 * Get the duration in seconds from a raw latent buffer (25 Hz latent rate).
 */
export function latentDuration(rawLatent: Buffer): number {
  return latentFrameCount(rawLatent) / 25;
}
