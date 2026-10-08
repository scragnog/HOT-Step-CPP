// llm/anthropic.ts — Anthropic / Claude provider

import { config } from '../../../config.js';
import { LLMProvider, readSSE } from './base.js';
import type { ChunkCallback, ProviderInfo } from './types.js';

const FALLBACK_MODELS = ['claude-fable-5-1', 'claude-opus-5-5', 'claude-sonnet-5', 'claude-haiku-4-5-20251001'];

export class AnthropicProvider extends LLMProvider {
  id = 'anthropic';
  name = 'Anthropic / Claude';
  get defaultModel() { return config.lireek.anthropicModel; }
  availableModels = [...FALLBACK_MODELS];

  private fallbackModels(): string[] {
    this.availableModels = [...FALLBACK_MODELS];
    return this.availableModels;
  }

  private async getRemoteModels(): Promise<string[]> {
    if (!this.isAvailable()) return this.fallbackModels();

    try {
      const models: string[] = [];
      const seen = new Set<string>();
      let afterId: string | undefined;

      do {
        const url = new URL('https://api.anthropic.com/v1/models');
        url.searchParams.set('limit', '1000');
        url.searchParams.set('lifecycle', 'active');
        if (afterId) url.searchParams.set('after_id', afterId);

        const resp = await fetch(url, {
          headers: {
            'x-api-key': config.lireek.anthropicApiKey,
            'anthropic-version': '2023-06-01',
          },
          signal: AbortSignal.timeout(4000),
        });
        if (!resp.ok) return this.fallbackModels();

        const page: { data?: Array<{ id?: string }>; has_more?: boolean; last_id?: string | null } = await resp.json();
        if (!Array.isArray(page.data)) return this.fallbackModels();
        for (const model of page.data) {
          if (typeof model.id === 'string' && !seen.has(model.id)) {
            seen.add(model.id);
            models.push(model.id);
          }
        }
        if (!page.has_more) break;
        if (!page.last_id || page.last_id === afterId) return this.fallbackModels();
        afterId = page.last_id;
      } while (true);

      this.availableModels = models.length ? models : [...FALLBACK_MODELS];
      return this.availableModels;
    } catch {
      return this.fallbackModels();
    }
  }

  async toInfoAsync(): Promise<ProviderInfo> {
    const models = await this.getRemoteModels();
    return {
      ...this.toInfo(),
      models,
      default_model: this.preferredModel(models),
    };
  }

  isAvailable() { return !!config.lireek.anthropicApiKey; }

  async call(systemPrompt: string, userPrompt: string, model?: string, onChunk?: ChunkCallback): Promise<string> {
    const url = 'https://api.anthropic.com/v1/messages';
    const payload = {
      model: model || this.defaultModel,
      max_tokens: 4096,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }],
      stream: !!onChunk,
    };

    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': config.lireek.anthropicApiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(config.lireek.llmTimeoutMs),
    });

    if (!resp.ok) throw new Error(`Anthropic error: ${resp.status} ${await resp.text()}`);

    if (onChunk) {
      let fullText = '';
      await readSSE(resp, (text) => { fullText += text; onChunk(text); }, (data) => {
        if (data.type === 'content_block_delta' && data.delta?.text) return data.delta.text;
        return null;
      });
      return fullText;
    } else {
      const data: { content?: Array<{ type: string; text?: string }> } = await resp.json();
      return data.content?.map(block => block.type === 'text' ? block.text ?? '' : '').join('') ?? '';
    }
  }
}
