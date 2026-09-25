import type { AceRequest } from '../aceClient.js';
import type { GenerationAttempt, GenerationEnvelope } from '../backends/types.js';

/** Internal job state */
/** Timing data for a single pipeline stage. */
export interface StageTiming {
  name: string;
  ms: number;
}

export interface GenerationJob {
  id: string;
  userId: string;
  envelope: Readonly<GenerationEnvelope>;
  attempts: GenerationAttempt[];
  status: 'pending' | 'running' | 'lm_running' | 'synth_running' | 'saving' | 'succeeded' | 'failed' | 'cancelled';
  stage?: string;
  progress?: number;
  aceJobId?: string;  // Current ace-server job ID (LM or synth)
  /** Fine-grained engine phase pulled from GET /job (e.g. "adapter_precompute").
   *  Surfaces "stuck in the ~17 s adapter precompute" vs. "actually failed". */
  acePhase?: string;
  /** Sub-phase progress formatted by pollUntilDone, e.g. "step 12/50", or
   *  empty when phase_total is 0. Surfaced to /status as ace_phase_progress. */
  acePhaseProgress?: string;
  lmResults?: AceRequest[];
  result?: {
    audioUrls: string[];
    songIds: string[];
    bpm?: number;
    duration?: number;
    keyScale?: string;
    timeSignature?: string;
    masteredAudioUrl?: string;
    /** No-adapter reference render (bare-DiT low-step output), when enabled */
    noAdapterAudioUrl?: string;
    /** Per-track mastered renders, index-aligned with audioUrls. The scalar
     *  above is only ever the FIRST non-empty one, so a multi-take render used
     *  to hand every take take 0's master — or, on MM3, no master at all.
     *  '' means that track produced none. */
    masteredAudioUrls?: string[];
    /** Per-track no-adapter reference renders, index-aligned with audioUrls. */
    noAdapterAudioUrls?: string[];
    /** Per-track durations in seconds, index-aligned with audioUrls. MM3 takes
     *  each stop at their own EOS, so the scalar `duration` (the longest) is
     *  wrong for every take but one. */
    durations?: number[];
    timing?: StageTiming[];
    totalMs?: number;
  };
  error?: string;
  params: any;
  createdAt: number;
  /** Set while another job renders this one inside its engine batch (YuE2
   *  queue coalescing). The backend returns it untouched on its own lane
   *  turn; cleared if the batch fails so it renders alone. */
  coalescedInto?: string;
  /** MiniMax-Music3 "play while rendering": true once the engine has confirmed
   *  it will serve this job's audio on GET /mm3/stream. The ENGINE's answer,
   *  not the request's — it may decline, and the UI must then behave exactly
   *  as it does with streaming off. */
  mm3Streaming?: boolean;
  /** True when the engine is dispatching windows DURING planning (both model
   *  stacks co-resident), false when it fell back to dispatching them after.
   *  Undefined until the engine has decided. Purely informational — a serial
   *  stream is still a stream. */
  mm3Interleaved?: boolean;
  /** Resolved render length in seconds, echoed by the engine at submit. Sent to
   *  the browser so a streaming track knows its full duration before its first
   *  window exists. */
  mm3Duration?: number;
  /** Ensemble takes this render is producing — the CLAMPED count the engine
   *  actually accepted, so it is how many streams exist to open and how many
   *  cards belong on screen. 1 (or absent) is an ordinary render. Known as
   *  soon as the engine has the job, i.e. long before any audio. */
  mm3Takes?: number;
  /** Each take's seed, as a DECIMAL STRING. Strings because these are uint64 —
   *  18226392072674864222 and its two successors all collapse to the same
   *  float64, which is exactly what made three distinct takes report one seed
   *  and become individually unreproducible. */
  mm3TakeSeeds?: string[];
  /** MM3 "require natural ending" outcome, set once the engine has finished.
   *  Present only on a render that ran the arbitration, and only then — so
   *  `dropped > 0` is the honest answer to "why did I ask for three and get
   *  two?" rather than something the UI has to infer from a count. */
  mm3Ending?: {
    /** Candidate plans drawn across every round. */
    planned: number;
    /** Candidates that reached EOS and became songs. */
    rendered: number;
    /** Candidates that hit the frame cap and were thrown away unrendered. */
    dropped: number;
    /** Planning rounds it took (1 on the common path). */
    rounds: number;
  };
  /** Stream preview WAV files emitted by the DEMON-style ring buffer */
  streamPreviews?: Array<{
    path: string;
    step: number;
    totalSteps: number;
    slot: number;
    timestamp: number;
  }>;
}

export const ACTIVE_STATUSES = ['pending', 'running', 'lm_running', 'synth_running', 'saving'] as const;

export function isActiveJob(job: Pick<GenerationJob, 'status'>): boolean {
  return (ACTIVE_STATUSES as readonly string[]).includes(job.status);
}
