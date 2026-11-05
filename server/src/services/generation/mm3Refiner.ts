// Optional MM3 latent refinement, before separation and the existing PP chain.
import { aceClient } from '../aceClient.js';
import { parseWav } from './audioLevel.js';

/** Saved settings and generation requests use these same names on every backend. */
export interface Mm3RefinerParams {
  mm3RefinerEnabled?: boolean;
  mm3RefinerSteps?: number;
  mm3RefinerWindowSeconds?: number;
  mm3RefinerOverlapSeconds?: number;
  mm3RefinerMix?: number;
}

export function normalizeMm3RefinerParams(raw: Mm3RefinerParams): Required<Mm3RefinerParams> {
  return {
    mm3RefinerEnabled: raw.mm3RefinerEnabled === true,
    mm3RefinerSteps: raw.mm3RefinerSteps ?? 32,
    mm3RefinerWindowSeconds: raw.mm3RefinerWindowSeconds ?? 30,
    mm3RefinerOverlapSeconds: raw.mm3RefinerOverlapSeconds ?? 2,
    mm3RefinerMix: raw.mm3RefinerMix ?? 1,
  };
}

type LogFn = (level: 'INFO' | 'DEBUG' | 'WARNING' | 'ERROR', message: string) => void;

/** Null means bypass or a declined stage. The caller keeps the original copy.
 *  In particular, mix=0 must preserve WAV headers and PCM bytes, whereas an
 *  engine round trip would re-encode them into float32 stereo. */
export async function runMm3Refiner(
  source: Buffer,
  raw: Mm3RefinerParams,
  log: LogFn,
  setStage: (stage: string) => void,
): Promise<Buffer | null> {
  const p = normalizeMm3RefinerParams(raw);
  if (!p.mm3RefinerEnabled || p.mm3RefinerMix === 0) return null;
  let id: string | undefined;
  let terminal = false;
  try {
    const { mm3RefinerSteps: steps, mm3RefinerWindowSeconds: window,
      mm3RefinerOverlapSeconds: overlap, mm3RefinerMix: mix } = p;
    if (!Number.isInteger(steps) || steps < 1 || steps > 64
      || !Number.isFinite(window) || window < 0
      || !Number.isFinite(overlap) || overlap < 0 || (window > 0 && overlap >= window)
      || !Number.isFinite(mix) || mix < 0 || mix > 1) {
      throw new Error('Invalid parameters: steps 1-64, window >= 0, overlap >= 0 and shorter than a nonzero window, mix 0-1');
    }
    setStage('MM3 refiner: queued');
    id = await aceClient.submitMm3Refine(source, { steps, window, overlap, mix });
    const deadline = Date.now() + 30 * 60_000;
    for (;;) {
      const progress = await aceClient.pollJob(id) as Awaited<ReturnType<typeof aceClient.pollJob>> & { error?: string };
      if (progress.status === 'done') { terminal = true; break; }
      if (progress.status === 'failed' || progress.status === 'cancelled') {
        terminal = true;
        throw new Error(progress.error || `Job ${progress.status}`);
      }
      // Conditioning counts windows twice, once for DAV and once for MERT.
      // Show the engine's current pass without assuming monotonically rising steps.
      const count = progress.phase_total ? ` (${progress.phase_step ?? 0}/${progress.phase_total})` : '';
      setStage(`MM3 refiner: ${progress.phase ?? 'queued'}${count}`);
      if (Date.now() >= deadline) throw new Error('Job timed out after 30 minutes');
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    const response = await aceClient.getJobResult(id);
    if (!response.ok) throw new Error(`Result (${response.status}): ${await response.text()}`);
    const result = Buffer.from(await response.arrayBuffer());
    const before = parseWav(source), after = parseWav(result);
    if (before.sampleRate !== after.sampleRate || before.channels[0].length !== after.channels[0].length
      || after.channels.some(channel => channel.some(value => !Number.isFinite(value)))) {
      throw new Error('Result has invalid samples, sample rate or length');
    }
    log('INFO', `[MM3 refiner] Complete: steps=${steps}, window=${window}, overlap=${overlap}, mix=${mix}`);
    return result;
  } catch (error) {
    if (id && !terminal) {
      await aceClient.cancelJob(id).catch(() => {});
      // Keep the GPU lane until the queued worker observes cancellation.
      const deadline = Date.now() + 30_000;
      while (Date.now() < deadline) {
        const progress = await aceClient.pollJob(id).catch(() => null);
        if (!progress || progress.status !== 'running') break;
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }
    log('WARNING', `[MM3 refiner] Skipped; raw audio kept: ${error instanceof Error ? error.message : String(error)}`);
    return null;
  }
}
