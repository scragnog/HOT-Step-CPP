// llm/openai-compat.ts — Generic OpenAI-compatible provider
//
// Connects to any server that implements the OpenAI API format
// (oMLX, vLLM, text-generation-webui, LocalAI, etc.)

import { config } from '../../../config.js';
import { LLMProvider, readSSE } from './base.js';
import type { ProviderInfo, ChunkCallback } from './types.js';

export class OpenAICompatProvider extends LLMProvider {
  id = 'openai-compat';
  get name() { return config.lireek.openaiCompatName || 'OpenAI Compatible'; }
  get defaultModel() { return config.lireek.openaiCompatModel; }

  isAvailable() { return !!config.lireek.openaiCompatBaseUrl; }

  private async getRemoteModels(): Promise<string[]> {
    try {
      const baseUrl = config.lireek.openaiCompatBaseUrl.replace(/\/+$/, '');
      const headers: Record<string, string> = {};
      if (config.lireek.openaiCompatApiKey) {
        headers['Authorization'] = `Bearer ${config.lireek.openaiCompatApiKey}`;
      }
      const resp = await fetch(`${baseUrl}/models`, { headers, signal: AbortSignal.timeout(3000) });
      if (!resp.ok) return [];
      const data = await resp.json();
      this.availableModels = data.data?.map((m: any) => m.id).sort().reverse() || [];
      return this.availableModels;
    } catch { return []; }
  }

  async toInfoAsync(): Promise<ProviderInfo> {
    const models = await this.getRemoteModels();
    return {
      ...this.toInfo(),
      models: models.length ? models : (this.defaultModel ? [this.defaultModel] : []),
      default_model: models.length ? models[0] : this.defaultModel,
    };
  }

  async call(systemPrompt: string, userPrompt: string, model?: string, onChunk?: ChunkCallback): Promise<string> {
    const baseUrl = config.lireek.openaiCompatBaseUrl.replace(/\/+$/, '');
    const url = `${baseUrl}/chat/completions`;
    const modelName = model || (await this.getRemoteModels())[0] || this.defaultModel;

    if (!modelName) throw new Error(`No models available on ${this.name}`);

    const payload: Record<string, any> = {
      model: modelName,
      messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: userPrompt }],
      stream: !!onChunk,
      // Force thinking/reasoning for Qwen3-style models on oMLX/vLLM.
      // Servers that don't support this parameter will safely ignore it.
      enable_thinking: true,
    };

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (config.lireek.openaiCompatApiKey) {
      headers['Authorization'] = `Bearer ${config.lireek.openaiCompatApiKey}`;
    }

    const resp = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(300_000),
    });

    if (!resp.ok) throw new Error(`${this.name} error: ${resp.status} ${await resp.text()}`);

    if (onChunk) {
      return await readSSE(resp, onChunk, (data) => data.choices?.[0]?.delta?.content || null, (data) => data.choices?.[0]?.delta?.reasoning_content || null);
    } else {
      const data = await resp.json();
      return data.choices?.[0]?.message?.content || '';
    }
  }
}
