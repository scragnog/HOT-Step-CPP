// llm/openai.ts — OpenAI / ChatGPT provider

import { config } from '../../../config.js';
import { LLMProvider, readSSE } from './base.js';
import type { ChunkCallback } from './types.js';

export class OpenAIProvider extends LLMProvider {
  id = 'openai';
  name = 'OpenAI / ChatGPT';
  get defaultModel() { return config.lireek.openaiModel; }
  availableModels = ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'];

  isAvailable() { return !!config.lireek.openaiApiKey; }

  async call(systemPrompt: string, userPrompt: string, model?: string, onChunk?: ChunkCallback): Promise<string> {
    const url = 'https://api.openai.com/v1/chat/completions';
    const payload = {
      model: model || this.defaultModel,
      messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: userPrompt }],
      stream: !!onChunk,
    };

    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${config.lireek.openaiApiKey}`,
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(300_000),
    });

    if (!resp.ok) throw new Error(`OpenAI error: ${resp.status} ${await resp.text()}`);

    if (onChunk) {
      return await readSSE(resp, onChunk, (data) => data.choices?.[0]?.delta?.content || null, (data) => data.choices?.[0]?.delta?.reasoning_content || null);
    } else {
      const data = await resp.json();
      return data.choices?.[0]?.message?.content || '';
    }
  }
}
