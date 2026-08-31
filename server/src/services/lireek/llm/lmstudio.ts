// llm/lmstudio.ts — LM Studio provider

import { config } from '../../../config.js';
import { LLMProvider, readSSE, noThinkSystemPrompt } from './base.js';
import type { ProviderInfo, ChunkCallback, CallOptions } from './types.js';

export class LMStudioProvider extends LLMProvider {
  id = 'lmstudio';
  get local() { return true; }
  name = 'LM Studio';
  get defaultModel() { return config.lireek.lmstudioModel; }

  isAvailable() { return true; }

  /** Bearer header for an LM Studio server with Authentication enabled.
   *  Empty object when no key is set, which is the default install (#121). */
  private authHeaders(): Record<string, string> {
    const key = config.lireek.lmstudioApiKey;
    return key ? { Authorization: `Bearer ${key}` } : {};
  }

  private async getLocalModels(): Promise<string[]> {
    try {
      const baseUrl = config.lireek.lmstudioBaseUrl.replace('/v1', '');
      const resp = await fetch(`${baseUrl}/v1/models`, {
        headers: this.authHeaders(),
        signal: AbortSignal.timeout(3000),
      });
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
    const baseUrl = config.lireek.lmstudioBaseUrl;
    const url = `${baseUrl}/chat/completions`;
    const modelName = model || (await this.getLocalModels())[0] || this.defaultModel;

    if (!modelName) throw new Error("No models loaded in LM Studio");

    const noThink = !!options?.noThink;
    const payload: Record<string, any> = {
      model: modelName,
      messages: [
        { role: 'system', content: noThink ? noThinkSystemPrompt(systemPrompt) : systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      stream: !!onChunk,
    };
    if (noThink) {
      // Empirically verified against LM Studio + Qwen3.6 (2026-07-09): this is
      // the field LM Studio honours — reasoning drops to zero and content is
      // answered directly. Non-thinking models (gemma) accept it harmlessly.
      payload.reasoning_effort = 'none';
      // llama.cpp-style template kwarg — ignored by LM Studio today (verified),
      // kept because it is harmless and honoured if support lands.
      payload.chat_template_kwargs = { enable_thinking: false };
      // Qwen's OFFICIAL non-thinking sampling profile. Without thinking, low-
      // entropy sampling degenerates into endless repetition loops ("the cord
      // is a square / the cord is a circle" ...); presence_penalty=1.5 is
      // Qwen's documented anti-loop knob for this mode. Explicit CallOptions
      // values win. max_tokens bounds any residual runaway.
      payload.temperature = options?.temperature ?? 0.7;
      payload.top_p = options?.top_p ?? 0.8;
      payload.top_k = 20;
      payload.presence_penalty = 1.5;
      payload.max_tokens = 8192;
    }

    const doFetch = () => fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...this.authHeaders() },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(300_000),
    });

    let resp = await doFetch();
    // If a server/model combination rejects the non-standard fields, retry
    // once without them rather than failing the generation.
    if (!resp.ok && noThink && resp.status === 400) {
      delete payload.chat_template_kwargs;
      delete payload.reasoning_effort;
      resp = await doFetch();
    }

    if (!resp.ok) throw new Error(`LM Studio error: ${resp.status} ${await resp.text()}`);

    if (onChunk) {
      return await readSSE(resp, onChunk, (data) => data.choices?.[0]?.delta?.content || null, (data) => data.choices?.[0]?.delta?.reasoning_content || null);
    } else {
      const data = await resp.json();
      return data.choices?.[0]?.message?.content || '';
    }
  }
}
