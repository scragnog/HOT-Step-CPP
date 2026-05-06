// llm/base.ts — LLMProvider abstract base class + SSE streaming helper

import type { ProviderInfo, ChunkCallback, CallOptions } from './types.js';
import { skipThinkingSignal } from './types.js';

export abstract class LLMProvider {
  abstract id: string;
  abstract name: string;
  abstract defaultModel: string;
  availableModels: string[] = [];

  abstract isAvailable(): boolean;
  
  abstract call(
    systemPrompt: string, 
    userPrompt: string, 
    model?: string, 
    onChunk?: ChunkCallback, 
    options?: CallOptions
  ): Promise<string>;

  toInfo(): ProviderInfo {
    return {
      id: this.id,
      name: this.name,
      available: this.isAvailable(),
      models: this.availableModels.length ? this.availableModels : (this.defaultModel ? [this.defaultModel] : []),
      default_model: this.defaultModel,
    };
  }
}

export async function readSSE(
  response: Response,
  onChunk: ChunkCallback,
  extractText: (data: any) => string | null,
  extractDisplayOnly?: (data: any) => string | null
): Promise<string> {
  if (!response.body) return '';
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let fullText = '';
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep the last incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const dataStr = line.slice(6).trim();
          if (dataStr === '[DONE]') {
            reader.cancel();
            return fullText;
          }
          
          try {
            const parsed = JSON.parse(dataStr);

            // Break on finish_reason (OpenAI-compatible sentinel)
            if (parsed.choices?.[0]?.finish_reason) {
              const lastText = extractText(parsed);
              if (lastText) {
                fullText += lastText;
                onChunk(lastText);
              }
              continue;
            }

            // Display-only content (e.g. reasoning/thinking) — stream to UI but don't keep
            if (extractDisplayOnly) {
              const displayText = extractDisplayOnly(parsed);
              if (displayText) {
                onChunk(displayText);
                // Track for skip-thinking detection but don't add to returned text
                if (skipThinkingSignal) {
                  const thinkCheck = fullText + displayText;
                  if (thinkCheck.includes('<think>') && !thinkCheck.includes('</think>')) {
                    reader.cancel();
                    return fullText;
                  }
                }
                continue;
              }
            }

            const text = extractText(parsed);
            if (text) {
              fullText += text;
              onChunk(text);
              
              if (skipThinkingSignal && fullText.includes('<think>') && !fullText.includes('</think>')) {
                reader.cancel();
                return fullText;
              }
            }
          } catch (e) {}
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
  return fullText;
}
