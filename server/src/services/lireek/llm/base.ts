// llm/base.ts — LLMProvider abstract base class + SSE streaming helper

import { Agent } from 'undici';
import { config } from '../../../config.js';
import type { ProviderInfo, ChunkCallback, CallOptions } from './types.js';
import { skipThinkingSignal } from './types.js';

// The completion timer owns the budget, including waits for headers and body.
// A separate pool prevents undici's five-minute defaults from capping it.
const completionDispatcher = new Agent({ headersTimeout: 0, bodyTimeout: 0 });

export abstract class LLMProvider {
  abstract id: string;
  abstract name: string;
  abstract defaultModel: string;
  availableModels: string[] = [];
  /** Runs on the user's machine. A getter rather than a field so a provider
   *  whose locality depends on configuration (openai-compat, whose base URL
   *  the user sets) can decide at call time. See ProviderInfo.local. */
  get local(): boolean { return false; }

  abstract isAvailable(): boolean;
  
  abstract call(
    systemPrompt: string, 
    userPrompt: string, 
    model?: string, 
    onChunk?: ChunkCallback, 
    options?: CallOptions
  ): Promise<string>;

  /** Which model the picker should land on, given what the server reports.
   *
   *  The configured model (Settings) wins whenever the server is actually
   *  serving it. If it is configured but no longer loaded, the server's first
   *  entry decides — the setting cannot be honoured either way, and failing the
   *  run would punish anyone whose setting has gone stale. Keeping the answer
   *  inside `models` also stops the UI rendering a <select> whose value is not
   *  one of its options. */
  protected preferredModel(models: string[]): string {
    if (this.defaultModel && (models.length === 0 || models.includes(this.defaultModel))) {
      return this.defaultModel;
    }
    return models[0] || this.defaultModel || '';
  }

  protected async withCompletion<T>(
    onChunk: ChunkCallback | undefined,
    run: (request: typeof fetch, onChunk: ChunkCallback | undefined) => Promise<T>,
    streaming = !!onChunk,
  ): Promise<T> {
    const timeoutMs = config.lireek.llmTimeoutMs;
    const controller = new AbortController();
    const timeout = new Error(
      `${this.name} LLM timeout after ${timeoutMs} ms${streaming ? ' without response tokens' : ''}. ` +
      `Raise LLM_TIMEOUT_MS (currently ${timeoutMs} ms) in Settings > AI Services > "LLM call timeout (ms)".`,
    );
    timeout.name = 'TimeoutError';
    const timer = setTimeout(() => controller.abort(timeout), timeoutMs);
    const request: typeof fetch = (input, init) => fetch(input, {
      ...init,
      signal: controller.signal,
      dispatcher: completionDispatcher,
    } as RequestInit & { dispatcher: Agent });
    const chunk: ChunkCallback | undefined = onChunk ? text => {
      if (text && streaming && !controller.signal.aborted) timer.refresh();
      onChunk(text);
    } : undefined;
    try {
      const result = await run(request, chunk);
      if (controller.signal.aborted) throw timeout;
      return result;
    } catch (error) {
      if (controller.signal.aborted) throw timeout;
      throw error;
    } finally {
      clearTimeout(timer);
      // Also closes an unfinished response after an early parser return/error.
      controller.abort();
    }
  }

  toInfo(): ProviderInfo {
    return {
      id: this.id,
      name: this.name,
      available: this.isAvailable(),
      models: this.availableModels.length ? this.availableModels : (this.defaultModel ? [this.defaultModel] : []),
      default_model: this.defaultModel,
      local: this.local,
    };
  }
}

// Qwen3-family soft switch: a bare `/no_think` in the system prompt makes the
// chat template skip the thinking block. Plain text to every other model, so
// it is safe to send unconditionally when CallOptions.noThink is set.
export function noThinkSystemPrompt(systemPrompt: string): string {
  return `${systemPrompt}\n\n/no_think`;
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
