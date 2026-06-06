// llm/lmstudio.ts — LM Studio provider

import { config } from '../../../config.js';
import { LLMProvider, readSSE } from './base.js';
import type { ProviderInfo, ChunkCallback } from './types.js';

export class LMStudioProvider extends LLMProvider {
  id = 'lmstudio';
  name = 'LM Studio';
  get defaultModel() { return config.lireek.lmstudioModel; }

  isAvailable() { return true; }

  private async getLocalModels(): Promise<string[]> {
    try {
      const baseUrl = config.lireek.lmstudioBaseUrl.replace('/v1', '');
      const resp = await fetch(`${baseUrl}/v1/models`, { signal: AbortSignal.timeout(3000) });
      if (!resp.ok) return [];
      const data = await resp.json();
      this.availableModels = data.data?.map((m: any) => m.id).sort().reverse() || [];
      return this.availableModels;
    } catch { return []; }
  }

  async toInfoAsync(): Promise<ProviderInfo> {
    const models = await this.getLocalModels();
    return {
      ...this.toInfo(),
      models: models.length ? models : (this.defaultModel ? [this.defaultModel] : []),
      default_model: models.length ? models[0] : this.defaultModel,
    };
  }

  async call(systemPrompt: string, userPrompt: string, model?: string, onChunk?: ChunkCallback): Promise<string> {
    const baseUrl = config.lireek.lmstudioBaseUrl;
    const url = `${baseUrl}/chat/completions`;
    const modelName = model || (await this.getLocalModels())[0] || this.defaultModel;
    
    if (!modelName) throw new Error("No models loaded in LM Studio");

    const payload = {
      model: modelName,
      messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: userPrompt }],
      stream: !!onChunk,
    };

    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(300_000),
    });

    if (!resp.ok) throw new Error(`LM Studio error: ${resp.status} ${await resp.text()}`);

    if (onChunk) {
      return await readSSE(resp, onChunk, (data) => data.choices?.[0]?.delta?.content || null, (data) => data.choices?.[0]?.delta?.reasoning_content || null);
    } else {
      const data = await resp.json();
      return data.choices?.[0]?.message?.content || '';
    }
  }
}
