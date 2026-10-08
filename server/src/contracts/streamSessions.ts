// Stream sessions (/api/stream-sessions): Node's view of an engine audio
// stream it is proxying (STORM slots, MM3 windows), with per-chunk analysis
// and an explicit canonical recording. Playback and device capture stay in
// the client.
import { z } from 'zod/v4';

export const STREAM_EXPORT_FORMATS = ['wav', 'flac', 'mp3', 'opus'] as const;
export type StreamExportFormat = typeof STREAM_EXPORT_FORMATS[number];

export interface StreamChunkAnalysis {
  /** 0-based chunk index within the session (STORM slot, MM3 window). */
  index: number;
  sampleRate: number;
  channels: number;
  frames: number;
  /** 0 = unknown. */
  bpm: number;
  /** '' = unknown; 'A', 'F#m'. */
  key: string;
  /** First onset, seconds into the chunk. */
  firstBeat: number;
}

export type StreamRecordingStatus = 'idle' | 'recording' | 'stopped' | 'capped' | 'failed';

export interface StreamSessionSummary {
  id: string;
  kind: 'storm' | 'mm3';
  /** STORM: the client's streamId. MM3: Node job id and take. */
  source: { streamId: string } | { jobId: string; take: number };
  status: 'live' | 'ended';
  createdAt: number;
  endedAt: number | null;
  /** When an ended session and its recording are deleted. */
  expiresAt: number | null;
  chunkCount: number;
  /** The most recent chunks' analysis (at most the last 64). */
  chunks: StreamChunkAnalysis[];
  recording: {
    status: StreamRecordingStatus;
    /** First and last chunk index recorded; null before any chunk. */
    fromChunk: number | null;
    toChunk: number | null;
    frames: number;
    bytes: number;
    sampleRate: number | null;
    channels: number | null;
    error: string | null;
  };
  limits: { maxRecordingBytes: number; endedTtlMs: number; maxSessions: number };
}

export const recordingCommandSchema = z.object({
  action: z.enum(['start', 'stop', 'discard']),
});

export const exportQuerySchema = z.object({
  format: z.enum(STREAM_EXPORT_FORMATS).default('wav'),
  bitrate: z.coerce.number().int().min(32).max(512).optional(),
});
