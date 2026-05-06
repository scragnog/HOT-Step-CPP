// llm/ollama.ts — Ollama (Local) provider

import { config } from '../../../config.js';
import { LLMProvider } from './base.js';
import type { ProviderInfo, ChunkCallback } from './types.js';
import { skipThinkingSignal } from './types.js';

export class OllamaProvider extends LLMProvider {
  id = 'ollama';
  name = 'Ollama (Local)';
  get defaultModel() { return config.lireek.ollamaModel; }
  
  isAvailable() { return true; }
  
  private async getLocalModels(): Promise<string[]> {
    try {
      const resp = await fetch(`${config.lireek.ollamaBaseUrl}/api/tags`);
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
      default_model: models.length ? models[0] : this.defaultModel,
    };
  }

  async call(systemPrompt: string, userPrompt: string, model?: string, onChunk?: ChunkCallback): Promise<string> {
    const url = `${config.lireek.ollamaBaseUrl}/api/chat`;
    const payload = {
      model: model || this.defaultModel,
      messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: userPrompt }],
      stream: !!onChunk,
      options: { num_predict: 8196 }
    };

    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

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
