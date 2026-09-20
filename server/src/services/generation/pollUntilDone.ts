import { aceClient } from '../aceClient.js';
import type { GenerationJob } from './jobTypes.js';

/** Poll ace-server job until completion, with stall detection watchdog.
 *  If job.stage/progress don't change for STALE_TIMEOUT_MS, the job is
 *  considered stalled and is cancelled + failed. This prevents a single
 *  wedged generation from blocking the entire queue for 45 minutes. */
export async function pollUntilDone(aceJobId: string, job: GenerationJob, signal: AbortSignal, timeoutMinutes?: number): Promise<void> {
  const POLL_INTERVAL = 500;          // ms — 500ms is tight enough for UI, avoids hammering
  // User-configurable wall-clock timeout, clamped to [5, 120] min. Default 45 min.
  const clampedTimeout = Math.max(5, Math.min(360, timeoutMinutes || 45));
  const MAX_WALL_MS = clampedTimeout * 60 * 1000;
  const STALE_TIMEOUT_MS = 120_000;   // 2 min with no progress = stalled
  // VAE decode reports once per tile and nothing in between, so its quiet
  // period is a whole tile. At chunk=1024 that is minutes on slower hardware,
  // and the flat 2 min window cancelled live decodes as wedged (#96). The
  // engine then finished the track anyway and the user was left with a job
  // marked red and no way to reach the audio.
  const STALE_TIMEOUT_VAE_MS = 900_000;  // 15 min — one tile, generously
  // The same shape applies to every stage that is one fixed string for the
  // whole of a long operation: model loads, the adapter merge (a LoKR with F32
  // promotion can take minutes and prints nothing until it is done, #106), FSQ
  // decode, DiT setup, the VAE encodes, and "Synthesizing..." before the first
  // step line (#96). Only a stage that ticks ("... Step n/total") can be
  // judged on a 2 min silence; everything else gets the quiet window, and the
  // wall-clock timeout below remains the backstop for a genuine hang.
  const STALE_TIMEOUT_QUIET_MS = STALE_TIMEOUT_VAE_MS;
  const isTickingStage = (s: unknown): boolean => typeof s === 'string' && /: Step \d+/.test(s);
  // A failing poll means "I can't see the job", not "the job stopped". The
  // engine stops answering HTTP while a single long op holds its request
  // thread, and counting that blind time as no-progress is what cancelled a
  // YuE2 run that had already finished (#158). While blind, the wall-clock
  // timeout below is the only backstop — which is what it is for.
  const BLIND_GRACE_MS = 10_000;
  // A stage that WAS ticking and has gone silent is the ambiguous case, and it
  // is the one that keeps costing users renders (#96, #158). The tight window
  // is only honest while the ticks are arriving: the moment they stop, the
  // stage may have died, or it may simply have ENDED and handed over to a long
  // quiet operation whose own first report has not landed yet. #158 was the
  // second kind — the semantic stage's last step stayed on screen while an
  // 11-minute NAR chunk solved underneath it, on an Apple Silicon box where
  // that chunk is slow enough to matter.
  //
  // So the 2 min mark is where we ASK the engine, not where we kill the job.
  // If it answers "still running", the stage falls back to the quiet window
  // and the wall-clock timeout remains the backstop. A genuinely wedged engine
  // is still caught quickly, because it answers "failed" or stops answering at
  // all, and neither of those paths waits.
  const RUNNING_PROBE_MS = 30_000;
  let lastProbeAt = 0;
  let stillRunningSince = 0;
  const startedAt = Date.now();
  let lastProgressAt = Date.now();
  let lastPollOkAt = Date.now();
  let lastStage = job.stage;
  let lastProgress = job.progress;
  // Heartbeat derived from the ace-server poll body (phase + phase_step).
  // The engine bumps phase_step every ~5 s during a long synchronous load
  // via LoadProgressTicker, so this changes even when no stdout log line
  // moves the parsed stage/progress. Without it a load that starts while
  // lastStage is still a ticking stage is judged on the tight 2 min window
  // and killed mid-load.
  let lastAceHeartbeat = '';

  while (true) {
    if (signal.aborted || job.status === 'cancelled') {
      await aceClient.cancelJob(aceJobId).catch(() => {});
      throw new Error('Cancelled');
    }

    // Detect progress changes (set by subscribeLines callbacks in runGeneration)
    if (job.stage !== lastStage || job.progress !== lastProgress) {
      lastProgressAt = Date.now();
      lastStage = job.stage;
      lastProgress = job.progress;
      stillRunningSince = 0;
    }

    // Stall detection: no progress update for the window this stage allows
    const stalledFor = Date.now() - lastProgressAt;
    const inVaeDecode = typeof lastStage === 'string' && lastStage.startsWith('Decoding audio (VAE)');
    const staleLimit = inVaeDecode ? STALE_TIMEOUT_VAE_MS
                     : isTickingStage(lastStage) ? STALE_TIMEOUT_MS : STALE_TIMEOUT_QUIET_MS;
    if (stalledFor > staleLimit && Date.now() - lastPollOkAt < BLIND_GRACE_MS
        && Date.now() - lastProbeAt > RUNNING_PROBE_MS) {
      lastProbeAt = Date.now();
      // Never let the watchdog destroy work the engine has already finished.
      // #158 and #96 both ended with a completed track thrown away because
      // this branch fired on a job that was already done.
      const final = await aceClient.pollJob(aceJobId).catch(() => null);
      if (final?.status === 'done') return;
      if (final?.status !== 'failed' && final?.status !== 'cancelled') {
        // The engine says it is still working. On a ticking stage that only
        // means the ticks stopped, so give it the quiet window before doing
        // anything irreversible — and say so once, because a user watching a
        // frozen step counter deserves to know it is being waited on rather
        // than ignored.
        if (isTickingStage(lastStage) && stalledFor < STALE_TIMEOUT_QUIET_MS) {
          if (!stillRunningSince) {
            stillRunningSince = Date.now();
            console.warn(
              `[Generate] ${aceJobId}: no progress for ${Math.round(stalledFor / 1000)}s at ` +
              `"${lastStage}", but the engine reports it is still running — waiting up to ` +
              `${Math.round(STALE_TIMEOUT_QUIET_MS / 60_000)} min before treating it as stalled.`
            );
          }
        } else {
          await aceClient.cancelJob(aceJobId).catch(() => {});
          throw new Error(
            `Generation stalled — no progress for ${Math.round(stalledFor / 1000)}s ` +
            `(last stage: "${lastStage}")`
          );
        }
      }
    }

    // Absolute wall-clock timeout
    if (Date.now() - startedAt > MAX_WALL_MS) {
      await aceClient.cancelJob(aceJobId).catch(() => {});
      throw new Error(`Generation timed out (${clampedTimeout} min limit)`);
    }

    // Poll ace-server — wrap in try-catch so transient HTTP timeouts
    // (e.g. ace-server busy mid-DiT-step) don't kill the loop
    try {
      const status = await aceClient.pollJob(aceJobId);
      lastPollOkAt = Date.now();
      // Surface the engine's fine-grained phase + step counter so /status can
      // return ace_phase / ace_phase_progress. Optional on the wire (older
      // ace-server builds omit them), so guard.
      if (status.phase) {
        job.acePhase = status.phase;
        const step = status.phase_step ?? 0;
        const total = status.phase_total ?? 0;
        job.acePhaseProgress = total > 0 ? `step ${step}/${total}` : '';
        // Feed the engine phase + step into the watchdog heartbeat so a
        // long load keeps the stall timer from firing.
        const aceHeartbeat = `${status.phase}|${step}|${total}`;
        if (aceHeartbeat !== lastAceHeartbeat) {
          lastProgressAt = Date.now();
          lastAceHeartbeat = aceHeartbeat;
        }
      }
      if (status.status === 'done') return;
      if (status.status === 'failed') throw new Error('Generation failed on ace-server');
      if (status.status === 'cancelled') throw new Error('Cancelled by ace-server');
    } catch (pollErr: any) {
      // Re-throw non-transient errors (actual generation failures)
      if (pollErr.message?.includes('Generation failed') ||
          pollErr.message?.includes('Cancelled')) {
        throw pollErr;
      }
      // Transient poll error (timeout, connection refused) — log and retry
      console.warn(`[Generate] Poll error for job ${aceJobId}: ${pollErr.message} (will retry)`);
    }

    await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
  }
}
