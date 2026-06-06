// llm/anthropic.ts — Anthropic / Claude provider

import { config } from '../../../config.js';
import { LLMProvider, readSSE } from './base.js';
import type { ChunkCallback } from './types.js';

export class AnthropicProvider extends LLMProvider {
  id = 'anthropic';
  name = 'Anthropic / Claude';
  get defaultModel() { return config.lireek.anthropicModel; }
  availableModels = ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229'];

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
      signal: AbortSignal.timeout(300_000),
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
      const data = await resp.json();
      return data.content?.[0]?.text || '';
    }
  }
}
