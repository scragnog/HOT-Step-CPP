// llm/registry.ts — Provider registry and lookup

import { LLMProvider } from './base.js';
import type { ProviderInfo } from './types.js';
import { GeminiProvider } from './gemini.js';
import { OpenAIProvider } from './openai.js';
import { AnthropicProvider } from './anthropic.js';
import { OllamaProvider } from './ollama.js';
import { LMStudioProvider } from './lmstudio.js';
import { UnslothProvider } from './unsloth.js';

const providers: Record<string, LLMProvider> = {
  gemini: new GeminiProvider(),
  openai: new OpenAIProvider(),
  anthropic: new AnthropicProvider(),
  ollama: new OllamaProvider(),
  lmstudio: new LMStudioProvider(),
  unsloth: new UnslothProvider(),
};

export function getProvider(name: string): LLMProvider {
  const provider = providers[name];
  if (!provider) throw new Error(`Unknown LLM provider: ${name}`);
  return provider;
}

export async function listProviders(): Promise<ProviderInfo[]> {
  const results = [];
  for (const p of Object.values(providers)) {
    if (p instanceof GeminiProvider || p instanceof OllamaProvider || p instanceof LMStudioProvider || p instanceof UnslothProvider) {
      results.push(await p.toInfoAsync());
    } else {
      results.push(p.toInfo());
    }
  }
  return results;
}
