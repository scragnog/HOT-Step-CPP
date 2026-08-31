// llm/ollama.ts — Ollama (Local) provider

import { config } from '../../../config.js';
import { LLMProvider, noThinkSystemPrompt } from './base.js';
import type { ProviderInfo, ChunkCallback, CallOptions } from './types.js';
import { skipThinkingSignal } from './types.js';

export class OllamaProvider extends LLMProvider {
  id = 'ollama';
  get local() { return true; }
  name = 'Ollama (Local)';
  get defaultModel() { return config.lireek.ollamaModel; }
  
  isAvailable() { return true; }
  
  private async getLocalModels(): Promise<string[]> {
    try {
      const resp = await fetch(`${config.lireek.ollamaBaseUrl}/api/tags`, { signal: AbortSignal.timeout(3000) });
      if (!resp.ok) return [];
      const data = await resp.json();
      this.availableModels = data.models?.map((m: any) => m.name) || [];
      return this.availableModels;
    } catch { return []; }
  }

  async toInfoAsync(): Promise<ProviderInfo> {
    const models = await this.getLocalModels();
    return {
      ...this.toInfo(),
      models: models.length ? models : [this.defaultModel],
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
    const url = `${config.lireek.ollamaBaseUrl}/api/chat`;
    const noThink = !!options?.noThink;
    const payload: Record<string, any> = {
      model: model || this.defaultModel,
      messages: [
        { role: 'system', content: noThink ? noThinkSystemPrompt(systemPrompt) : systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      stream: !!onChunk,
      options: { num_predict: 8196 }
    };
    // Native Ollama switch for thinking models (qwen3, deepseek-r1, ...).
    if (noThink) {
      payload.think = false;
      // Qwen's official non-thinking sampling profile — presence_penalty=1.5
      // is their documented guard against degenerate repetition loops.
      payload.options = {
        ...payload.options,
        temperature: options?.temperature ?? 0.7,
        top_p: options?.top_p ?? 0.8,
        top_k: 20,
        presence_penalty: 1.5,
      };
    }

    const doFetch = () => fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(300_000),
    });

    let resp = await doFetch();
    // Non-thinking models reject the `think` field — retry once without it.
    if (!resp.ok && noThink && resp.status === 400 && 'think' in payload) {
      delete payload.think;
      resp = await doFetch();
    }

    if (!resp.ok) throw new Error(`Ollama error: ${resp.status} ${await resp.text()}`);

    if (onChunk) {
      if (!resp.body) return '';
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let fullText = '';
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunkLines = decoder.decode(value, { stream: true }).split('\n');
          for (const line of chunkLines) {
            if (!line.trim()) continue;
            try {
              const parsed = JSON.parse(line);
              // Display-only: stream reasoning to UI but don't keep in result
              const reasoning = parsed.message?.reasoning_content;
              if (reasoning) {
                onChunk(reasoning);
                continue;
              }
              const content = parsed.message?.content;
              if (content) {
                fullText += content;
                onChunk(content);
                if (skipThinkingSignal && fullText.includes('<think>') && !fullText.includes('</think>')) {
                  reader.cancel();
                  return fullText;
                }
              }
            } catch (e) {}
          }
        }
      } finally { reader.releaseLock(); }
      return fullText;
    } else {
      const data = await resp.json();
      return data.message?.content || '';
    }
  }
}
