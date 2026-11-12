import type { HeartmulaJob } from './client.js';

/** The queue card's text and percent for one /heartmula/job reading. The LM's frame
 *  count is a ceiling (audio EOS can end the song early), so its share is an upper
 *  bound. Ticking stages end in ": Step n/total", the pattern pollUntilDone's stall
 *  watchdog judges on a short window. */
export function heartmulaStageText(d: Partial<HeartmulaJob>): { stage: string; progress: number } | null {
  const frac = (done = 0, total = 0) => total > 0 ? Math.min(1, Math.max(0, done / total)) : 0;
  switch (d.stage) {
    case 'queued': return { stage: 'HeartMuLa: waiting for the GPU...', progress: 2 };
    case 'style': return { stage: 'MuLaCover: encoding style tags...', progress: 3 };
    case 'admission': return { stage: 'HeartMuLa: checking VRAM...', progress: 3 };
    case 'loading_lm': return { stage: 'Loading HeartMuLa LM...', progress: 4 };
    case 'loading_adapter': return { stage: 'Loading HeartMuLa adapter...', progress: 6 };
    case 'prefill':
      return { stage: `HeartMuLa LM: reading prompt: Step ${d.prefill_done ?? 0}/${d.prompt_tokens ?? 0}`,
        progress: Math.round(7 + 3 * frac(d.prefill_done, d.prompt_tokens)) };
    case 'frames': {
      const max = d.max_frames ?? 0, done = d.frames_done ?? 0;
      return { stage: `HeartMuLa LM: ${(done / 12.5).toFixed(1)}s of up to ${Math.round(max / 12.5)}s: Step ${done}/${max}`,
        progress: Math.round(10 + 60 * frac(done, max)) };
    }
    case 'loading_codec': return { stage: 'Loading HeartCodec...', progress: 71 };
    case 'codec_decode': {
      const chunks = d.codec_chunks ?? 0, steps = d.codec_steps ?? 0, step = d.codec_step ?? 0;
      return { stage: `HeartCodec decode: chunk ${Math.min((d.codec_chunk ?? 0) + 1, Math.max(chunks, 1))}/${chunks}: Step ${step}/${chunks * steps}`,
        progress: Math.round(72 + 21 * frac(step, chunks * steps)) };
    }
    case 'encoding_output': return { stage: 'HeartMuLa: encoding audio...', progress: 94 };
    default: return null;
  }
}
