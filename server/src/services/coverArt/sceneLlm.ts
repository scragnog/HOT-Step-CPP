import { config } from '../../config.js';
import { getProvider } from '../lireek/llm/registry.js';
import type { LLMProvider } from '../lireek/llm/base.js';
import type { CoverArtPromptOpts } from './promptBuilder.js';

const SCENE_TIMEOUT_MS = 20_000;
const MAX_SCENE_LENGTH = 220;

type SceneProvider = Pick<LLMProvider, 'isAvailable' | 'call'>;

interface SceneDeps {
  provider?: SceneProvider | null;
  timeoutMs?: number;
}

/** Return a visual subject for automatic covers, or let the keyword path run. */
export async function resolveCoverScene(opts: CoverArtPromptOpts, deps: SceneDeps = {}): Promise<string | null> {
  if (opts.prompt?.trim() || opts.subject?.trim()) return null;

  try {
    const provider = deps.provider === undefined
      ? getProvider(config.lireek.defaultProvider)
      : deps.provider;
    if (!provider?.isAvailable()) {
      console.log('[CoverArt] Scene unavailable; using keyword fallback');
      return null;
    }

    const lyricLines = (opts.lyrics || '').split(/\r?\n/)
      .map(line => line.trim())
      .filter(line => line && !/^\[.*\]$/.test(line))
      .slice(0, 4).join(' ').slice(0, 400);
    const context = [
      opts.title?.trim() && `Song title: ${opts.title.trim().slice(0, 120)}`,
      opts.style?.trim() && `Style: ${opts.style.trim().slice(0, 180)}`,
      lyricLines && `Lyrics: ${lyricLines}`,
    ].filter(Boolean).join('\n') || 'No song details supplied.';
    const instruction = 'Write one short line describing a concrete visual scene inspired by the song. Describe only what can be seen. Do not mention text, lettering, words, titles, or album covers. Return the scene alone.';

    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<null>(resolve => {
      timer = setTimeout(() => resolve(null), deps.timeoutMs ?? SCENE_TIMEOUT_MS);
    });
    let reply: string | null;
    try {
      reply = await Promise.race([provider.call(instruction, context), timeout]);
    } finally {
      if (timer) clearTimeout(timer);
    }
    const firstLine = reply?.split(/\r?\n/).map(line => line.trim()).find(Boolean);
    const candidate = firstLine?.slice(0, MAX_SCENE_LENGTH).trim() || null;
    const scene = candidate && !/\b(text|lettering|words?|titles?|album|covers?|typography|writing|caption|logo|signage)\b/i.test(candidate)
      ? candidate : null;
    console.log(scene ? '[CoverArt] Using caption LLM scene' : '[CoverArt] Scene unavailable; using keyword fallback');
    return scene;
  } catch {
    console.log('[CoverArt] Scene unavailable; using keyword fallback');
    return null;
  }
}
