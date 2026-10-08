// llm/registry.ts — Provider registry and lookup

import { LLMProvider } from './base.js';
import type { ProviderInfo } from './types.js';
import { GeminiProvider } from './gemini.js';
import { OpenAIProvider } from './openai.js';
import { AnthropicProvider } from './anthropic.js';
import { OllamaProvider } from './ollama.js';
import { LMStudioProvider } from './lmstudio.js';
import { UnslothProvider } from './unsloth.js';
import { OpenAICompatProvider } from './openai-compat.js';
import { LlamaCppProvider } from './llamacpp.js';

const providers: Record<string, LLMProvider> = {
  gemini: new GeminiProvider(),
  openai: new OpenAIProvider(),
  anthropic: new AnthropicProvider(),
  ollama: new OllamaProvider(),
  lmstudio: new LMStudioProvider(),
  unsloth: new UnslothProvider(),
  llamacpp: new LlamaCppProvider(),
  'openai-compat': new OpenAICompatProvider(),
};

export function getProvider(name: string): LLMProvider {
  const provider = providers[name];
  if (!provider) throw new Error(`Unknown LLM provider: ${name}`);
  return provider;
}

export async function listProviders(): Promise<ProviderInfo[]> {
  const PROVIDER_TIMEOUT_MS = 5000;

  const promises = Object.values(providers).map(async (p): Promise<ProviderInfo> => {
    try {
      if (p instanceof GeminiProvider || p instanceof AnthropicProvider || p instanceof OllamaProvider || p instanceof LMStudioProvider || p instanceof UnslothProvider || p instanceof LlamaCppProvider || p instanceof OpenAICompatProvider) {
        // Race against a timeout so one dead provider can't block the rest
        const info = await Promise.race([
          p.toInfoAsync(),
          new Promise<ProviderInfo>((_, reject) =>
            setTimeout(() => reject(new Error(`${p.name} timed out`)), PROVIDER_TIMEOUT_MS)
          ),
        ]);
        return info;
      } else {
        return p.toInfo();
      }
    } catch (err: any) {
      console.warn(`[LLM Registry] Provider ${p.name} failed: ${err.message}`);
      // Return the provider as unavailable rather than dropping it
      return { ...p.toInfo(), available: false, models: p.defaultModel ? [p.defaultModel] : [] };
    }
  });

  const settled = await Promise.allSettled(promises);
  return settled
    .filter((r): r is PromiseFulfilledResult<ProviderInfo> => r.status === 'fulfilled')
    .map(r => r.value);
}
