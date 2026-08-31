// llm/openai-compat.ts — Generic OpenAI-compatible provider
//
// Connects to any server that implements the OpenAI API format
// (oMLX, vLLM, NInfer, text-generation-webui, LocalAI, etc.)

import { config } from '../../../config.js';
import { LLMProvider, readSSE, noThinkSystemPrompt } from './base.js';
import type { ProviderInfo, ChunkCallback, CallOptions } from './types.js';

export class OpenAICompatProvider extends LLMProvider {
  id = 'openai-compat';
  get name() { return config.lireek.openaiCompatName || 'OpenAI Compatible'; }
  get defaultModel() { return config.lireek.openaiCompatModel; }

  isAvailable() { return !!config.lireek.openaiCompatBaseUrl; }

  /** Unlike the other providers this one's locality is not fixed — the user
   *  supplies the base URL, and it is just as likely to be vLLM on this machine
   *  as a hosted endpoint. Decided from the host rather than guessed, so the UI
   *  does not call someone's localhost server "cloud". Anything unparseable or
   *  routable counts as remote, which is the safer way to be wrong. */
  get local(): boolean {
    try {
      const host = new URL(config.lireek.openaiCompatBaseUrl).hostname.toLowerCase();
      return host === 'localhost' || host === '::1' || host.endsWith('.local')
        || /^127\./.test(host) || /^0\.0\.0\.0$/.test(host)
        || /^10\./.test(host) || /^192\.168\./.test(host)
        || /^172\.(1[6-9]|2\d|3[01])\./.test(host);
    } catch { return false; }
  }

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
      // The model chosen on the Settings page wins whenever the server
      // is actually serving it. This used to be `models[0]`, so the
      // configured model was only ever used when the server was
      // unreachable — i.e. the setting was dead exactly when it could
      // have worked, and the picker silently defaulted to whatever the
      // server happened to list first.
      default_model: this.preferredModel(models),
    };
  }

  async call(
    systemPrompt: string,
    userPrompt: string,
    model?: string,
    onChunk?: ChunkCallback,
    options?: CallOptions
  ): Promise<string> {
    const baseUrl = config.lireek.openaiCompatBaseUrl.replace(/\/+$/, '');
    const url = `${baseUrl}/chat/completions`;
    const modelName = model || (await this.getRemoteModels())[0] || this.defaultModel;

    if (!modelName) throw new Error(`No models available on ${this.name}`);

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
      // NInfer honours `reasoning_effort` natively (none|low|medium|xhigh) as a
      // real chat-template change, not a sampling alias. LM Studio honours the
      // same field. Layered with the llama.cpp template kwarg, which is
      // harmless where unsupported.
      payload.reasoning_effort = 'none';
      payload.chat_template_kwargs = { enable_thinking: false };
      // Qwen's official non-thinking sampling profile — presence_penalty=1.5 is
      // their documented guard against degenerate repetition loops in this mode.
      // Explicit CallOptions values win; max_tokens bounds runaways.
      payload.temperature = options?.temperature ?? 0.7;
      payload.top_p = options?.top_p ?? 0.8;
      payload.top_k = 20;
      payload.presence_penalty = 1.5;
      payload.max_tokens = options?.max_tokens ?? 8192;
    } else {
      // Force thinking/reasoning for Qwen3-style models on oMLX/vLLM.
      // Servers that don't support this parameter will safely ignore it.
      payload.enable_thinking = true;
      // Intermediate efforts ('low' | 'medium' | 'xhigh'). NInfer treats these
      // as distinct reasoning budgets; 'low' is the useful middle ground
      // between full deliberation and none at all. A per-request value wins;
      // otherwise fall back to the configured default, and send nothing at all
      // when that is empty so endpoints which reject the field still work.
      const effort = options?.reasoning_effort || config.lireek.openaiCompatReasoningEffort;
      if (effort) payload.reasoning_effort = effort;
      if (options?.max_tokens) payload.max_tokens = options.max_tokens;
      if (options?.temperature !== undefined) payload.temperature = options.temperature;
      if (options?.top_p !== undefined) payload.top_p = options.top_p;
    }

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (config.lireek.openaiCompatApiKey) {
      headers['Authorization'] = `Bearer ${config.lireek.openaiCompatApiKey}`;
    }

    const doFetch = () => fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(300_000),
    });

    // Staged fallback. NInfer hard-400s on chat_template_kwargs but DOES honour
    // reasoning_effort (verified 2026-08-18: 'none' drops reasoning to zero).
    // A catch-all retry that strips both would silently restore full thinking, so
    // drop the llama.cpp-only kwarg first and only surrender the effort field if
    // the server still refuses.
    let resp = await doFetch();
    if (!resp.ok && resp.status === 400 && payload.chat_template_kwargs) {
      delete payload.chat_template_kwargs;
      resp = await doFetch();
    }
    if (!resp.ok && resp.status === 400) {
      delete payload.reasoning_effort;
      delete payload.enable_thinking;
      resp = await doFetch();
    }

    if (!resp.ok) throw new Error(`${this.name} error: ${resp.status} ${await resp.text()}`);

    if (onChunk) {
      return await readSSE(resp, onChunk, (data) => data.choices?.[0]?.delta?.content || null, (data) => data.choices?.[0]?.delta?.reasoning_content || null);
    } else {
      const data = await resp.json();
      return data.choices?.[0]?.message?.content || '';
    }
  }
}
