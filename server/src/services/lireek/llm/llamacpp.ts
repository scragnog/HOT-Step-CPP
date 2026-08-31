// llm/llamacpp.ts — llama.cpp server provider
//
// Connects to a llama.cpp server (llama-server / llama-cli --server)
// which exposes an OpenAI-compatible API at /v1/chat/completions and /v1/models.
// Default endpoint: http://127.0.0.1:8080/v1

import { config } from '../../../config.js';
import { LLMProvider, readSSE, noThinkSystemPrompt } from './base.js';
import type { ProviderInfo, ChunkCallback, CallOptions } from './types.js';

export class LlamaCppProvider extends LLMProvider {
  id = 'llamacpp';
  get local() { return true; }
  name = 'llama.cpp';
  get defaultModel() { return config.lireek.llamacppModel; }

  isAvailable() { return true; }

  private async getLocalModels(): Promise<string[]> {
    try {
      const baseUrl = config.lireek.llamacppBaseUrl.replace(/\/+$/, '');
      const resp = await fetch(`${baseUrl}/models`, { signal: AbortSignal.timeout(3000) });
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
      // The model chosen on the Settings page wins whenever the server
      // is actually serving it. This used to be `models[0]`, so the
      // configured model was only ever used when the server was
      // unreachable — i.e. the setting was dead exactly when it could
      // have worked, and the picker silently defaulted to whatever the
      // server happened to list first.
      default_model: this.preferredModel(models),
    };
  }

  async call(systemPrompt: string, userPrompt: string, model?: string, onChunk?: ChunkCallback, options?: CallOptions): Promise<string> {
    const baseUrl = config.lireek.llamacppBaseUrl.replace(/\/+$/, '');
    const url = `${baseUrl}/chat/completions`;
    const modelName = model || (await this.getLocalModels())[0] || this.defaultModel;

    if (!modelName) throw new Error('No models available on llama.cpp server');

    const noThink = !!options?.noThink;
    const payload: Record<string, any> = {
      model: modelName,
      messages: [
        { role: 'system', content: noThink ? noThinkSystemPrompt(systemPrompt) : systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      stream: !!onChunk,
    };
    // Officially supported by llama-server: skips the thinking block for
    // templates that take an enable_thinking kwarg (Qwen3, GLM, ...).
    if (noThink) {
      payload.chat_template_kwargs = { enable_thinking: false };
      // Qwen's official non-thinking sampling profile — presence_penalty=1.5
      // is their documented guard against degenerate repetition loops in this
      // mode. Explicit CallOptions values win; max_tokens bounds runaways.
      payload.temperature = options?.temperature ?? 0.7;
      payload.top_p = options?.top_p ?? 0.8;
      payload.top_k = 20;
      payload.presence_penalty = 1.5;
      payload.max_tokens = 8192;
    }

    // 5-min timeout — prevents hung requests from blocking the generation queue
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(300_000),
    });

    if (!resp.ok) throw new Error(`llama.cpp error: ${resp.status} ${await resp.text()}`);

    if (onChunk) {
      return await readSSE(resp, onChunk, (data) => data.choices?.[0]?.delta?.content || null, (data) => data.choices?.[0]?.delta?.reasoning_content || null);
    } else {
      const data = await resp.json();
      return data.choices?.[0]?.message?.content || '';
    }
  }
}
