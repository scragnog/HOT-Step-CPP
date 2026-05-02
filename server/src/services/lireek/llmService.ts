import { config } from '../../config.js';
import * as slopDetector from './slopDetector.js';
import { 
  GENERATION_SYSTEM_PROMPT, 
  SONG_METADATA_SYSTEM_PROMPT, 
  REFINEMENT_SYSTEM_PROMPT,
  TITLE_DERIVATION_PROMPT 
} from './prompts.js';
import { LyricsProfile } from './profilerService.js';

export interface ProviderInfo {
  id: string;
  name: string;
  available: boolean;
  models: string[];
  default_model: string;
}

export interface GenerationResponse {
  lyrics: string;
  provider: string;
  model: string;
  title: string;
  subject: string;
  bpm: number;
  key: string;
  caption: string;
  duration: number;
  system_prompt: string;
  user_prompt: string;
}

export type ChunkCallback = (chunk: string) => void;

export interface CallOptions {
  temperature?: number;
  top_p?: number;
  [key: string]: any;
}

// Global skip thinking signal
export let skipThinkingSignal = false;
export function setSkipThinking() {
  skipThinkingSignal = true;
  console.log('[LLM] Skip-thinking signal received');
}
export function resetSkipThinking() {
  skipThinkingSignal = false;
}

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

async function readSSE(
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

// ── Gemini ──────────────────────────────────────────────────────────────────

export class GeminiProvider extends LLMProvider {
  id = 'gemini';
  name = 'Google Gemini';
  get defaultModel() { return config.lireek.geminiModel; }
  availableModels = ['gemini-2.5-flash'];  // initial fallback, replaced by API fetch

  /** Cache fetched models so we don't hit the API on every listProviders() call */
  private _cachedModels: string[] | null = null;
  private _cacheExpiry = 0;
  private static CACHE_TTL = 5 * 60 * 1000; // 5 minutes

  isAvailable() { return !!config.lireek.geminiApiKey; }

  /** Fetch available models from the Gemini API, filtered to those that support generateContent */
  private async getRemoteModels(): Promise<string[]> {
    const now = Date.now();
    if (this._cachedModels && now < this._cacheExpiry) return this._cachedModels;

    try {
      const allModels: Array<{ name: string; supportedGenerationMethods?: string[] }> = [];
      let pageToken: string | undefined;

      // Paginate through all models
      do {
        const url = new URL('https://generativelanguage.googleapis.com/v1beta/models');
        url.searchParams.set('key', config.lireek.geminiApiKey);
        url.searchParams.set('pageSize', '100');
        if (pageToken) url.searchParams.set('pageToken', pageToken);

        const resp = await fetch(url.toString(), { signal: AbortSignal.timeout(8000) });
        if (!resp.ok) {
          console.warn(`[Gemini] models.list failed: ${resp.status}`);
          break;
        }
        const data = await resp.json();
        if (Array.isArray(data.models)) allModels.push(...data.models);
        pageToken = data.nextPageToken;
      } while (pageToken);

      if (allModels.length === 0) return this.availableModels;

      // Filter to models that support generateContent (excludes embedding, AQA, etc.)
      const generative = allModels
        .filter(m => m.supportedGenerationMethods?.includes('generateContent'))
        .map(m => m.name.replace(/^models\//, ''))
        // Exclude tuning, embedding, and legacy models cluttering the list
        .filter(name => !name.includes('embedding') && !name.includes('aqa') && !name.includes('tunedModels'));

      if (generative.length === 0) return this.availableModels;

      // Sort: prefer 2.5 > 2.0 > 1.5, flash before pro, shorter names first (base > dated variants)
      generative.sort((a, b) => {
        // Extract version number for primary sort
        const verA = parseFloat(a.match(/(\d+\.\d+)/)?.[1] || '0');
        const verB = parseFloat(b.match(/(\d+\.\d+)/)?.[1] || '0');
        if (verB !== verA) return verB - verA;
        // Same version: flash before pro
        const isFlashA = a.includes('flash') ? 0 : 1;
        const isFlashB = b.includes('flash') ? 0 : 1;
        if (isFlashA !== isFlashB) return isFlashA - isFlashB;
        // Same tier: shorter names (base model) before dated variants
        return a.length - b.length;
      });

      this._cachedModels = generative;
      this._cacheExpiry = now + GeminiProvider.CACHE_TTL;
      this.availableModels = generative;
      return generative;
    } catch (err: any) {
      console.warn(`[Gemini] Failed to fetch models: ${err.message}`);
      return this.availableModels;
    }
  }

  async toInfoAsync(): Promise<ProviderInfo> {
    const models = await this.getRemoteModels();
    return {
      ...this.toInfo(),
      models: models.length ? models : [this.defaultModel],
      default_model: this.defaultModel,
    };
  }

  async call(systemPrompt: string, userPrompt: string, model?: string, onChunk?: ChunkCallback): Promise<string> {
    const modelName = model || this.defaultModel;
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:${onChunk ? 'streamGenerateContent?alt=sse&' : 'generateContent?'}key=${config.lireek.geminiApiKey}`;
    
    const payload = {
      systemInstruction: { parts: [{ text: systemPrompt }] },
      contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
    };

    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) throw new Error(`Gemini API error: ${resp.status} ${await resp.text()}`);

    if (onChunk) {
      return await readSSE(resp, onChunk, (data) => data.candidates?.[0]?.content?.parts?.[0]?.text || null);
    } else {
      const data = await resp.json();
      return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
    }
  }
}

// ── OpenAI ──────────────────────────────────────────────────────────────────

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

// ── Anthropic ───────────────────────────────────────────────────────────────

export class AnthropicProvider extends LLMProvider {
  id = 'anthropic';
  name = 'Anthropic / Claude';
  get defaultModel() { return config.lireek.anthropicModel; }
  availableModels = ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229'];

  isAvailable() { return !!config.lireek.anthropicApiKey; }

  async call(systemPrompt: string, userPrompt: string, model?: string, onChunk?: ChunkCallback): Promise<string> {
    const url = 'https://api.anthropic.com/v1/messages';
    const payload = {
      model: model || this.defaultModel,
      max_tokens: 4096,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }],
      stream: !!onChunk,
    };

    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': config.lireek.anthropicApiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) throw new Error(`Anthropic error: ${resp.status} ${await resp.text()}`);

    if (onChunk) {
      let fullText = '';
      await readSSE(resp, (text) => { fullText += text; onChunk(text); }, (data) => {
        if (data.type === 'content_block_delta' && data.delta?.text) return data.delta.text;
        return null;
      });
      return fullText;
    } else {
      const data = await resp.json();
      return data.content?.[0]?.text || '';
    }
  }
}

// ── Ollama ──────────────────────────────────────────────────────────────────

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

// ── LM Studio ───────────────────────────────────────────────────────────────

export class LMStudioProvider extends LLMProvider {
  id = 'lmstudio';
  name = 'LM Studio';
  get defaultModel() { return config.lireek.lmstudioModel; }

  isAvailable() { return true; }

  private async getLocalModels(): Promise<string[]> {
    try {
      const baseUrl = config.lireek.lmstudioBaseUrl.replace('/v1', '');
      const resp = await fetch(`${baseUrl}/v1/models`);
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

// ── Unsloth Studio ──────────────────────────────────────────────────────────

export class UnslothProvider extends LLMProvider {
  id = 'unsloth';
  name = 'Unsloth Studio';
  get defaultModel() { return config.lireek.unslothModel; }
  
  private cachedToken = '';
  private tokenExpiry = 0;

  isAvailable() { return !!config.lireek.unslothUsername && !!config.lireek.unslothPassword; }

  private async authenticate(): Promise<string> {
    const now = Date.now() / 1000;
    if (this.cachedToken && now < this.tokenExpiry - 60) return this.cachedToken;

    const payload = { email: config.lireek.unslothUsername, password: config.lireek.unslothPassword };
    const resp = await fetch(`${config.lireek.unslothBaseUrl}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    // Unsloth also sometimes takes username instead of email, trying generic handling if this fails
    if (!resp.ok) {
        const payload2 = { username: config.lireek.unslothUsername, password: config.lireek.unslothPassword };
        const resp2 = await fetch(`${config.lireek.unslothBaseUrl}/api/auth/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload2)
        });
        if (!resp2.ok) throw new Error("Failed to authenticate with Unsloth");
        const data = await resp2.json();
        this.cachedToken = data.access_token || data.token;
    } else {
      const data = await resp.json();
      this.cachedToken = data.access_token || data.token;
    }

    try {
      const payloadB64 = this.cachedToken.split('.')[1];
      const decoded = JSON.parse(Buffer.from(payloadB64, 'base64').toString());
      this.tokenExpiry = decoded.exp || (now + 3600);
    } catch {
      this.tokenExpiry = now + 3600;
    }
    return this.cachedToken;
  }

  private async getLocalModels(): Promise<string[]> {
    try {
      const token = await this.authenticate();
      const resp = await fetch(`${config.lireek.unslothBaseUrl}/v1/models`, {
        headers: { 'Authorization': `Bearer ${token}` }
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
      default_model: models.length ? models[0] : this.defaultModel,
    };
  }

  async call(systemPrompt: string, userPrompt: string, model?: string, onChunk?: ChunkCallback): Promise<string> {
    const token = await this.authenticate();
    const modelName = model || (await this.getLocalModels())[0] || this.defaultModel;
    if (!modelName) throw new Error("No models loaded in Unsloth Studio");

    const url = `${config.lireek.unslothBaseUrl}/v1/chat/completions`;
    const payload = {
      model: modelName,
      messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: userPrompt }],
      stream: true, // Unsloth often requires stream: true
    };

    const resp = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) throw new Error(`Unsloth error: ${resp.status} ${await resp.text()}`);

    return await readSSE(resp, onChunk || (() => {}), (data) => data.choices?.[0]?.delta?.content || null, (data) => data.choices?.[0]?.delta?.reasoning_content || null);
  }
}

// ── Registry ────────────────────────────────────────────────────────────────

const providers: Record<string, LLMProvider> = {
  gemini: new GeminiProvider(),
  openai: new OpenAIProvider(),
  anthropic: new AnthropicProvider(),
  ollama: new OllamaProvider(),
  lmstudio: new LMStudioProvider(),
  unsloth: new UnslothProvider(),
};

export function getProvider(name: string): LLMProvider {
  const provider = providers[name];
  if (!provider) throw new Error(`Unknown LLM provider: ${name}`);
  return provider;
}

export async function listProviders(): Promise<ProviderInfo[]> {
  const results = [];
  for (const p of Object.values(providers)) {
    if (p instanceof GeminiProvider || p instanceof OllamaProvider || p instanceof LMStudioProvider || p instanceof UnslothProvider) {
      results.push(await p.toInfoAsync());
    } else {
      results.push(p.toInfo());
    }
  }
  return results;
}

// ── Orchestration Functions ─────────────────────────────────────────────────

export function stripThinkingBlocks(text: string): string {
  // Remove standard <think>...</think> blocks
  let result = text.replace(/<think>[\s\S]*?<\/think>/g, '');
  // Remove other reasoning model tags
  result = result.replace(/<analysis>[\s\S]*?<\/analysis>/g, '');
  result = result.replace(/<reasoning>[\s\S]*?<\/reasoning>/g, '');
  result = result.replace(/<reflection>[\s\S]*?<\/reflection>/g, '');
  result = result.replace(/<thought>[\s\S]*?<\/thought>/g, '');
  // Remove LM Studio channel-based thinking: <|channel>thought...<channel|>
  result = result.replace(/<\|channel>thought[\s\S]*?<channel\|>/g, '');
  // Handle unclosed thinking blocks (model stopped mid-thought)
  result = result.replace(/<(?:think|analysis|reasoning|reflection|thought)>[\s\S]*/g, '');
  result = result.replace(/<\|channel>thought[\s\S]*/g, '');
  // Plain text CoT (LM Studio GGUF quirks)
  const cotMatch = result.match(/^(?:\s*\*+\s*)?(?:Thinking Process|Thought Process|Thinking|Reasoning):\s*[\s\S]*?(?:---|[*]{3,}|={3,})\s*/i);
  if (cotMatch) {
    result = result.slice(cotMatch[0].length);
  }
  return result.trim();
}

// ── Post-processing pipeline (ported from HOT-Step 9000) ────────────────────

const SECTION_KEYWORDS = [
  'Intro', 'Verse', 'Pre-Chorus', 'Chorus', 'Post-Chorus',
  'Bridge', 'Interlude', 'Outro', 'Hook', 'Refrain',
];

const SECTION_LINE_RE = new RegExp(
  '^\\[?(' + SECTION_KEYWORDS.map(k => k.replace(/[-/]/g, '\\$&')).join('|') + ')\\s*(\\d*)\\]?\\s*$',
  'i'
);

const PUNCTUATION_ENDINGS = new Set('.,!?;:-…)"\'');

/** Wrap section headers in brackets, add missing punctuation to lyric lines. */
function postprocessLyrics(text: string): string {
  const resultLines: string[] = [];
  for (const line of text.split('\n')) {
    const stripped = line.trim();
    if (!stripped) { resultLines.push(''); continue; }
    const m = SECTION_LINE_RE.exec(stripped);
    if (m) {
      const sectionName = m[1].charAt(0).toUpperCase() + m[1].slice(1).toLowerCase();
      const sectionNum = m[2];
      resultLines.push(sectionNum ? `[${sectionName} ${sectionNum}]` : `[${sectionName}]`);
      continue;
    }
    if (/^\[.+\]$/.test(stripped)) { resultLines.push(stripped); continue; }
    if (stripped && !PUNCTUATION_ENDINGS.has(stripped[stripped.length - 1])) {
      resultLines.push(stripped + ',');
    } else {
      resultLines.push(stripped);
    }
  }
  return resultLines.join('\n');
}

/** Rename invalid section labels to valid ACE-Step ones. */
function fixSectionLabels(text: string): string {
  const INVALID_TO_VALID: Record<string, string> = {
    'x': 'Interlude', 'breakdown': 'Bridge', 'drop': 'Chorus',
    'solo': 'Interlude', 'hook': 'Chorus', 'rap': 'Verse', 'spoken': 'Verse',
  };

  const lines = text.split('\n');
  const result: string[] = [];
  const sectionHeaders: { lineIdx: number; header: string }[] = [];

  for (const line of lines) {
    const stripped = line.trim();
    const m = stripped.match(/^\[(.+?)(?:\s+\d+)?\]$/);
    if (m) {
      let label = m[1].trim().toLowerCase();
      let newStripped = stripped;
      if (INVALID_TO_VALID[label]) {
        const newLabel = INVALID_TO_VALID[label];
        const numMatch = stripped.match(/\d+/);
        newStripped = numMatch ? `[${newLabel} ${numMatch[0]}]` : `[${newLabel}]`;
      }
      sectionHeaders.push({ lineIdx: result.length, header: newStripped });
      result.push(newStripped);
    } else {
      result.push(stripped.startsWith('[') && stripped.endsWith(']') ? stripped : line);
    }
  }

  // If no Chorus exists but Bridge appears multiple times, fix it
  const bridgeIndices = sectionHeaders
    .map((h, i) => ({ i, h }))
    .filter(x => x.h.header.toLowerCase().includes('bridge'));
  const chorusExists = sectionHeaders.some(h => h.header.toLowerCase().includes('chorus'));

  if (!chorusExists && bridgeIndices.length >= 2) {
    for (const bi of bridgeIndices.slice(0, -1)) {
      result[sectionHeaders[bi.i].lineIdx] = '[Chorus]';
    }
  }

  return result.join('\n');
}

/** Enforce valid line counts: verses=4|8, choruses=4|6|8. */
function enforceLineCounts(text: string): string {
  const VERSE_VALID = new Set([4, 8]);
  const CHORUS_VALID = new Set([4, 6, 8]);
  const sections: { header: string; lines: string[] }[] = [];
  let currentHeader = '';
  let currentLines: string[] = [];

  for (const line of text.split('\n')) {
    const stripped = line.trim();
    if (/^\[.+\]$/.test(stripped)) {
      if (currentHeader || currentLines.length) {
        sections.push({ header: currentHeader, lines: currentLines });
      }
      currentHeader = stripped;
      currentLines = [];
    } else {
      currentLines.push(line);
    }
  }
  if (currentHeader || currentLines.length) {
    sections.push({ header: currentHeader, lines: currentLines });
  }

  const resultParts: string[] = [];
  for (const { header, lines } of sections) {
    const lyricLines = lines.filter(l => l.trim());
    const count = lyricLines.length;
    const headerLower = header.toLowerCase();
    const isVerse = headerLower.includes('verse');
    const isChorus = headerLower.includes('chorus') || headerLower.includes('hook');
    let target: number | null = null;

    if (isVerse && !VERSE_VALID.has(count)) {
      target = count <= 6 ? 4 : 8;
    } else if (isChorus && !CHORUS_VALID.has(count)) {
      if (count <= 5) target = 4;
      else if (count <= 7) target = 6;
      else target = 8;
    }

    let finalLines = lines;
    if (target !== null && target < count) {
      let kept = 0;
      finalLines = [];
      for (const l of lines) {
        if (l.trim()) {
          if (kept < target) { finalLines.push(l); kept++; }
        } else {
          if (kept < target) finalLines.push(l);
        }
      }
    }

    if (header) resultParts.push(header);
    resultParts.push(...finalLines);
  }
  return resultParts.join('\n');
}

/** Remove 'a-' prefix from non-gerund words. */
const BAD_A_PREFIX_RE = /\ba-(?!\w+ing\b)(?!\w+in'\b)/gi;
function fixAPrefix(text: string): string {
  return text.replace(BAD_A_PREFIX_RE, '');
}

/** Remove quoted lyric fragments from profile text to prevent copying. */
function stripLyricQuotes(text: string): string {
  return text.replace(/'[^']{4,}'/g, '[quote removed]');
}

/** Estimate track duration in seconds from lyrics content and BPM. */
function estimateDuration(lyrics: string, bpm: number): number {
  if (!lyrics.trim() || bpm <= 0) return 0;
  const barDuration = 240.0 / Math.max(bpm, 40);
  const lines = lyrics.trim().split('\n');
  let sectionCount = 0;
  let lyricLineCount = 0;
  for (const line of lines) {
    const stripped = line.trim();
    if (!stripped) continue;
    if (SECTION_LINE_RE.test(stripped) || (stripped.startsWith('[') && stripped.endsWith(']'))) {
      sectionCount++;
    } else {
      lyricLineCount++;
    }
  }
  const vocalSeconds = lyricLineCount * 3.5;
  const breakSeconds = Math.max(sectionCount - 1, 0) * 4 * barDuration;
  const estimated = Math.floor(vocalSeconds + breakSeconds);
  return Math.max(90, Math.min(estimated, 360));
}

/** Pick the most interesting blueprint from a list. */
function selectBestBlueprint(blueprints: string[]): string {
  if (!blueprints.length) return 'V-C-V-C-B-C';
  return blueprints
    .map(bp => {
      // Safety: truncate anything after an Outro (medley artefacts)
      const parts = bp.split('-');
      const outroIdx = parts.indexOf('O');
      return outroIdx >= 0 ? parts.slice(0, outroIdx + 1).join('-') : bp;
    })
    .reduce((best, bp) => {
      const parts = bp.split('-');
      const unique = new Set(parts).size;
      const hasBridge = parts.includes('B') ? 1 : 0;
      const score = unique * 10 + hasBridge * 100 + parts.length;
      const bestParts = best.split('-');
      const bestUnique = new Set(bestParts).size;
      const bestBridge = bestParts.includes('B') ? 1 : 0;
      const bestScore = bestUnique * 10 + bestBridge * 100 + bestParts.length;
      return score > bestScore ? bp : best;
    });
}

/**
 * Plans the metadata before generating.
 */
async function planSongMetadata(
  profile: LyricsProfile,
  usedSubjects: string[],
  usedBpms: number[],
  usedKeys: string[],
  usedDurations: number[],
  providerName: string,
  modelName: string,
  onChunk?: ChunkCallback
): Promise<any> {
  const provider = getProvider(providerName);
  const lines: string[] = [`Artist: ${profile.artist}`];
  if (profile.album) lines.push(`Album style: ${profile.album}`);
  if (profile.themes?.length) lines.push(`Themes: ${profile.themes.join(', ')}`);
  if (profile.tone_and_mood) lines.push(`Tone & mood: ${profile.tone_and_mood}`);
  if (profile.additional_notes) lines.push(`Additional notes: ${profile.additional_notes}`);
  if (profile.perspective) lines.push(`Perspective / voice: ${profile.perspective}`);

  if (profile.song_subjects && typeof profile.song_subjects === 'object') {
    lines.push('\nOriginal song subjects (for reference):');
    for (const [songTitle, subject] of Object.entries(profile.song_subjects)) {
      lines.push(`  • ${songTitle}: ${subject}`);
    }
  }
  if (profile.subject_categories?.length) {
    lines.push(`\nThematic categories: ${profile.subject_categories.join(', ')}`);
  }
  if (usedSubjects?.length) {
    lines.push('\nSubjects ALREADY USED (do NOT repeat these):');
    for (const s of usedSubjects) lines.push(`  ✗ ${s}`);
  }
  if (usedBpms?.length) lines.push(`\nBPMs ALREADY USED (avoid ±5 of these): ${usedBpms.join(', ')}`);
  if (usedKeys?.length) lines.push(`\nKeys ALREADY USED (try different ones): ${usedKeys.join(', ')}`);
  if (usedDurations?.length) lines.push(`\nDurations ALREADY USED (avoid ±10 of these): ${usedDurations.join(', ')}`);
  lines.push('\nPlan the metadata for the next song:');

  const prompt = lines.join('\n');
  console.log('[LLM] Planning song metadata via', providerName, modelName);
  const responseJsonStr = await provider.call(SONG_METADATA_SYSTEM_PROMPT, prompt, modelName, onChunk);
  const cleaned = stripThinkingBlocks(responseJsonStr);
  const cleanJson = cleaned.replace(/^```(?:json)?\s*|\s*```$/gm, '').trim();
  try {
    return JSON.parse(cleanJson);
  } catch (err) {
    // Try extracting JSON object with depth tracking
    const start = cleanJson.indexOf('{');
    if (start !== -1) {
      let depth = 0;
      for (let i = start; i < cleanJson.length; i++) {
        if (cleanJson[i] === '{') depth++;
        else if (cleanJson[i] === '}') { depth--; if (depth === 0) { try { return JSON.parse(cleanJson.slice(start, i + 1)); } catch {} break; } }
      }
    }
    console.error("Failed to parse metadata JSON:", cleanJson.slice(0, 300));
    return { subject: '', bpm: 0, key: '', caption: '', duration: 0 };
  }
}

const BLUEPRINT_LABEL_NAMES: Record<string, string> = {
  V: 'Verse', C: 'Chorus', B: 'Bridge', PC: 'Pre-Chorus',
  POC: 'Post-Chorus', I: 'Intro', O: 'Outro', IL: 'Interlude',
};

function buildGenerationPrompt(profile: LyricsProfile, extraInstructions?: string): string {
  const lines: string[] = [`Artist: ${profile.artist}`];
  if (profile.album) lines.push(`Album style: ${profile.album}`);

  // === STYLISTIC PROFILE ===
  lines.push('', '=== STYLISTIC PROFILE ===', '');
  lines.push(`Themes: ${(profile.themes || []).join(', ')}`);
  lines.push(`Common subjects / motifs: ${(profile.common_subjects || []).join(', ')}`);
  lines.push(`Rhyme schemes: ${(profile.rhyme_schemes || []).join(', ')}`);
  lines.push(`Average verse length: ${profile.avg_verse_lines} lines`);
  lines.push(`Average chorus length: ${profile.avg_chorus_lines} lines`);
  if (profile.vocabulary_notes) lines.push(`Vocabulary: ${stripLyricQuotes(profile.vocabulary_notes)}`);
  if (profile.tone_and_mood) lines.push(`Tone & mood: ${stripLyricQuotes(profile.tone_and_mood)}`);
  if (profile.structural_patterns) lines.push(`Structural patterns: ${stripLyricQuotes(profile.structural_patterns)}`);

  // === SONG STRUCTURE (MANDATORY) ===
  if (profile.structure_blueprints?.length) {
    const bp = selectBestBlueprint(profile.structure_blueprints);
    lines.push('', '=== SONG STRUCTURE (MANDATORY) ===');
    lines.push(`Blueprint: ${bp}`);
    const parts = bp.split('-');
    let verseNum = 0;
    const sectionList: string[] = [];
    for (const part of parts) {
      let name = BLUEPRINT_LABEL_NAMES[part] || part;
      if (part === 'V') { verseNum++; name = `Verse ${verseNum}`; }
      sectionList.push(`[${name}]`);
    }
    lines.push(`You MUST write these sections in this exact order: ${sectionList.join(' → ')}`);
    if (parts.includes('B')) lines.push("This artist uses bridges — you MUST include a [Bridge] section.");
  }

  if (profile.perspective) lines.push(`Perspective / voice: ${profile.perspective}`);

  // === LINE LENGTH & METER ===
  const ms = profile.meter_stats;
  if (ms) {
    lines.push('', '=== LINE LENGTH & METER ===');
    lines.push(`Average: ~${ms.avg_syllables_per_line ?? '?'} syllables/line, ~${ms.avg_words_per_line ?? '?'} words/line`);
    lines.push(`Standard deviation: ±${ms.syllable_std_dev ?? '?'} syllables (VARY your line lengths!)`);
    const llv = ms.line_length_variation;
    if (llv?.histogram) {
      const histStr = Object.entries(llv.histogram).map(([k, v]) => `${k} syl: ${v}%`).join(', ');
      lines.push(`Syllable distribution: ${histStr}`);
      lines.push('Match this distribution — NOT all lines the same length!');
    }
  }

  // === REPETITION & HOOKS ===
  const rs = profile.repetition_stats;
  if (rs) {
    lines.push('', '=== REPETITION & HOOKS ===');
    lines.push(`Chorus repetition: ${rs.chorus_repetition_pct ?? 0}% of chorus lines are repeats`);
    lines.push(`Pattern: ${rs.pattern || 'unknown'}`);
    if ((rs.chorus_repetition_pct ?? 0) >= 20) {
      lines.push('You MUST use repeated lines in your chorus to create a hook effect.');
    }
    if (rs.hook_examples?.length) {
      lines.push(`Hook examples: ${rs.hook_examples.slice(0, 3).join('; ')}`);
    }
  }

  // === VOCABULARY ===
  const vs = profile.vocabulary_stats;
  if (vs) {
    lines.push('', '=== VOCABULARY ===');
    lines.push(`Level: ${vs.contraction_pct ?? 0}% contractions, ${vs.profanity_pct ?? 0}% profanity`);
    lines.push(`Type-token ratio: ${vs.type_token_ratio ?? '?'} (${vs.unique_words ?? '?'} unique / ${vs.total_words ?? '?'} total)`);
    if (vs.distinctive_words?.length) {
      lines.push(`Use words like: ${vs.distinctive_words.slice(0, 10).join(', ')}`);
    }
  }

  // === RHYME QUALITY ===
  if (profile.rhyme_quality) {
    const rq = profile.rhyme_quality;
    const total = Object.values(rq).reduce((a, b) => a + b, 0);
    if (total > 0) {
      lines.push(`Rhyme mix: ${Math.round(100 * (rq.perfect || 0) / total)}% perfect, ${Math.round(100 * (rq.slant || 0) / total)}% slant, ${Math.round(100 * (rq.assonance || 0) / total)}% assonance`);
    }
  }

  // Deep stylistic fields
  if (profile.narrative_techniques) lines.push(`Narrative techniques: ${stripLyricQuotes(profile.narrative_techniques)}`);
  if (profile.imagery_patterns) lines.push(`Imagery patterns: ${stripLyricQuotes(profile.imagery_patterns)}`);
  if (profile.signature_devices) lines.push(`Signature devices: ${stripLyricQuotes(profile.signature_devices)}`);
  if (profile.emotional_arc) lines.push(`Emotional arc: ${stripLyricQuotes(profile.emotional_arc)}`);

  // === PROSE SUMMARY ===
  if (profile.raw_summary) {
    lines.push('', '=== PROSE SUMMARY ===', '', stripLyricQuotes(profile.raw_summary));
  }

  // === EXTRA INSTRUCTIONS ===
  if (extraInstructions) {
    lines.push('', '=== EXTRA INSTRUCTIONS ===', '', extraInstructions);
  }

  // === LYRIC EXCERPTS ===
  if (profile.representative_excerpts?.length) {
    lines.push('', '=== REPRESENTATIVE EXCERPTS (STYLE REFERENCE ONLY — DO NOT COPY) ===');
    lines.push(...profile.representative_excerpts.slice(0, 10).flatMap(e => [e, '---']));
  }

  // === FINAL REMINDERS ===
  lines.push(
    '', '=== FINAL REMINDERS ===',
    '1. VERSE LINE COUNT: Exactly 4 or 8 lines per verse.',
    '2. CHORUS LINE COUNT: Exactly 4, 6, or 8 lines per chorus. Each chorus MUST have a hook line that repeats.',
    '3. *** ZERO TOLERANCE FOR COPYING ***',
    '4. NO SLOP: Do not use neon, fluorescent, embers, silhouette, static, void, ethereal, or any AI cliché.',
    '5. MINIMIZE OVERUSED WORDS: heavy, broken, cold, dust, ghost, machine, nothing, nowhere, searching — use at most ONCE if at all.',
    "6. VOCABULARY DIVERSITY: A Snoop Dogg song must NOT sound like a Joy Division song. Use THIS artist's actual vocabulary.",
    '',
    'Now write the song (lyrics only, starting with [Intro] or [Verse 1] — no title line):',
  );
  return lines.join('\n');
}

export async function generateLyricsStreaming(
  profile: LyricsProfile,
  providerName: string,
  model?: string,
  extraInstructions?: string,
  usedSubjects: string[] = [],
  usedBpms: number[] = [],
  usedKeys: string[] = [],
  usedTitles: string[] = [],
  usedDurations: number[] = [],
  onChunk?: ChunkCallback,
  onPhase?: (phase: string) => void
): Promise<GenerationResponse> {
  const provider = getProvider(providerName);
  const effectiveModel = model || provider.defaultModel;

  if (onPhase) onPhase("Planning song metadata…");
  let metadata = { subject: '', bpm: 0, key: '', caption: '', duration: 0 };
  
  if (profile.song_subjects || (profile.themes && profile.themes.length)) {
    try {
      metadata = await planSongMetadata(profile, usedSubjects, usedBpms, usedKeys, usedDurations, providerName, effectiveModel, onChunk);
      console.log("Planned metadata:", metadata);
    } catch(e) {
      console.warn("Failed to plan metadata", e);
    }
  }

  if (metadata.subject) {
    extraInstructions = `The song must be about: ${metadata.subject}\n\n${extraInstructions || ''}`;
  }

  if (onPhase) onPhase("Writing lyrics…");
  const userPrompt = buildGenerationPrompt(profile, extraInstructions);

  let raw = await provider.call(GENERATION_SYSTEM_PROMPT, userPrompt, effectiveModel, onChunk);
  
  // Post-process: strip reasoning, tokens, role markers
  raw = stripThinkingBlocks(raw);
  raw = raw.replace(/<\|[a-z_]+\|>/g, '');
  raw = raw.replace(/\[?(System|User|Assistant)\]?:.*/gi, '');
  raw = raw.replace(/\s*\((?:Hook|You|Repeat|x\d|Refrain|Spoken|Whispered|Ad[- ]?lib|Echo)\)\s*/gi, '');
  raw = raw.replace(/ +$/gm, '');

  // Strip any Title: line the LLM might still emit (old habit)
  const rawLines = raw.trim().split('\n');
  for (let i = 0; i < rawLines.length; i++) {
    const match = rawLines[i].match(/^(?:Title:\s*|#\s*)(.*)/i);
    if (match) {
      const rest = rawLines.slice(i + 1);
      while (rest.length && !rest[0].trim()) rest.shift();
      raw = rest.join('\n');
      break;
    }
    // Stop looking once we hit a section header or lyric content
    if (rawLines[i].trim().startsWith('[') || (rawLines[i].trim() && i > 2)) break;
  }

  // Full post-processing pipeline
  raw = postprocessLyrics(raw);
  raw = fixSectionLabels(raw);
  raw = fixAPrefix(raw);
  raw = enforceLineCounts(raw);

  // Slop check
  const slopResult = slopDetector.scanForSlop(raw);
  if (slopResult.ai_score > 0) {
    console.warn(`Generation slop scan: score=${slopResult.ai_score} severity=${slopResult.severity}`,
      'words:', slopResult.layers.blacklisted_words.found,
      'phrases:', slopResult.layers.blacklisted_phrases.found,
      'overuse:', slopResult.layers.overuse.found.map(o => `${o.word}(${o.count}x)`).join(', ') || 'none');
  }

  // === POST-HOC TITLE DERIVATION ===
  if (onPhase) onPhase("Choosing title…");
  let title = '';
  try {
    // Build a focused prompt: artist context + the lyrics + used titles to avoid
    const titleLines: string[] = [`Artist: ${profile.artist}`];
    if (profile.album) titleLines.push(`Album style: ${profile.album}`);
    if (usedTitles?.length) {
      titleLines.push('\nTitles already used (avoid these and their key words):');
      for (const t of usedTitles) titleLines.push(`  ✗ ${t}`);
    }
    titleLines.push('\n--- LYRICS ---', raw, '--- END LYRICS ---');
    titleLines.push('\nChoose the best title for this song:');

    let titleRaw = await provider.call(TITLE_DERIVATION_PROMPT, titleLines.join('\n'), effectiveModel, onChunk);
    titleRaw = stripThinkingBlocks(titleRaw).trim();
    // Clean up: remove quotes, "Title:" prefix, markdown
    titleRaw = titleRaw.replace(/^(?:Title:\s*|#\s*)/i, '').replace(/^["'`]|["'`]$/g, '').trim();
    // Take only the first line if multi-line
    title = titleRaw.split('\n')[0].trim();
    console.log('[LLM] Derived title:', title);
  } catch (err) {
    console.warn('[LLM] Title derivation failed, falling back to empty:', err);
  }

  // Duration estimation fallback
  let duration = metadata.duration || 0;
  if (metadata.bpm > 0 && !duration) {
    duration = estimateDuration(raw, metadata.bpm);
  }

  return {
    lyrics: raw,
    provider: providerName,
    model: effectiveModel,
    title,
    subject: metadata.subject,
    bpm: metadata.bpm,
    key: metadata.key,
    caption: metadata.caption,
    duration,
    system_prompt: GENERATION_SYSTEM_PROMPT,
    user_prompt: userPrompt
  };
}

function buildRefinementPrompt(originalLyrics: string, artistName: string, title: string, profile?: LyricsProfile, originalSlop?: string[]): string {
  const lines = [`Artist: ${artistName}`, `Original Title: ${title}`, ''];

  if (profile) {
    lines.push('=== INTENDED LANE PROFILE (match this style) ===');
    lines.push(`Themes: ${(profile.themes || []).slice(0, 8).join(', ')}`);
    if (profile.tone_and_mood) lines.push(`Tone & mood: ${profile.tone_and_mood}`);
    if (profile.vocabulary_notes) lines.push(`Vocabulary: ${profile.vocabulary_notes}`);
    if (profile.imagery_patterns) lines.push(`Imagery patterns: ${profile.imagery_patterns}`);
    if (profile.signature_devices) lines.push(`Signature devices: ${profile.signature_devices}`);
    if (profile.narrative_techniques) lines.push(`Narrative techniques: ${profile.narrative_techniques}`);
    if (profile.emotional_arc) lines.push(`Emotional arc: ${profile.emotional_arc}`);
    if (profile.structural_patterns) lines.push(`Structure: ${profile.structural_patterns}`);
    if (profile.perspective) lines.push(`Perspective / voice: ${profile.perspective}`);

    // Rhyme behavior
    if (profile.rhyme_schemes?.length) {
      lines.push(`Rhyme schemes: ${profile.rhyme_schemes.join(', ')}`);
    }
    if (profile.rhyme_quality) {
      const rq = profile.rhyme_quality;
      const total = Object.values(rq).reduce((a, b) => a + b, 0);
      if (total > 0) {
        lines.push(`Rhyme mix: ${Math.round(100 * (rq.perfect || 0) / total)}% perfect, ${Math.round(100 * (rq.slant || 0) / total)}% slant, ${Math.round(100 * (rq.assonance || 0) / total)}% assonance`);
      }
    }

    // Line density / meter
    const ms = profile.meter_stats;
    if (ms) {
      lines.push(`Line density: ~${ms.avg_syllables_per_line ?? '?'} syl/line (σ=${ms.syllable_std_dev ?? '?'}), ~${ms.avg_words_per_line ?? '?'} words/line`);
    }

    // Repetition / hook behavior
    const rs = profile.repetition_stats;
    if (rs) {
      lines.push(`Hook behavior: ${rs.pattern || 'unknown'} (${rs.chorus_repetition_pct ?? 0}% chorus repetition)`);
      if ((rs.chorus_repetition_pct ?? 0) >= 20) {
        lines.push('Calibration: This artist uses heavy chorus repetition — ensure hook lines repeat.');
      } else if ((rs.chorus_repetition_pct ?? 0) < 15) {
        lines.push('Calibration: This artist uses light repetition — be subtle with hooks.');
      }
    }

    // Verse/chorus contrast stats
    if (profile.avg_verse_lines || profile.avg_chorus_lines) {
      lines.push(`Verse/chorus: avg ${profile.avg_verse_lines} verse lines, avg ${profile.avg_chorus_lines} chorus lines`);
    }

    // Original song titles for plagiarism checking
    if (profile.song_subjects && typeof profile.song_subjects === 'object') {
      const titles = Object.keys(profile.song_subjects);
      if (titles.length) {
        lines.push('');
        lines.push('=== ORIGINAL SONG TITLES (check for plagiarism) ===');
        for (const t of titles) lines.push(`  • ${t}`);
      }
    }
    lines.push('');
  }

  if (originalSlop?.length) {
    lines.push('=== KNOWN ISSUES TO FIX ===');
    lines.push('The original lyrics contain the following AI-clichés or circular phrases that MUST be replaced:');
    lines.push(`Words/Phrases to Remove: ${originalSlop.join(', ')}`);
    lines.push('');
  }

  lines.push('=== ORIGINAL LYRICS ===', '', originalLyrics, '', '=== INSTRUCTIONS ===', '');
  lines.push('Refine the lyrics above according to the refinement rules.');
  lines.push('Keep as much of the original as possible — only change what genuinely needs fixing.');
  lines.push(`Maintain ${artistName}'s distinctive style throughout.`);
  lines.push('Now output the refined version (Title line first, then lyrics with [Section] headers):');

  return lines.join('\n');
}

export async function refineLyricsStreaming(
  originalLyrics: string,
  artistName: string,
  title: string,
  providerName: string,
  model?: string,
  profile?: LyricsProfile,
  onChunk?: ChunkCallback
): Promise<GenerationResponse> {
  const provider = getProvider(providerName);
  const effectiveModel = model || provider.defaultModel;

  const slopScan = slopDetector.scanForSlop(originalLyrics);
  const foundSlop = [...slopScan.layers.blacklisted_words.found, ...slopScan.layers.blacklisted_phrases.found];

  const userPrompt = buildRefinementPrompt(originalLyrics, artistName, title, profile, foundSlop);

  let raw = await provider.call(REFINEMENT_SYSTEM_PROMPT, userPrompt, effectiveModel, onChunk);

  // Post-process: strip reasoning, tokens, performance notes
  raw = stripThinkingBlocks(raw);
  raw = raw.replace(/<\|[a-z_]+\|>/g, '');
  raw = raw.replace(/\s*\((?:Hook|You|Repeat|x\d|Refrain|Spoken|Whispered|Ad[- ]?lib|Echo)\)\s*/gi, '');
  raw = raw.replace(/ +$/gm, '');
  
  // Extract title
  let refinedTitle = title;
  const lines = raw.trim().split('\n');
  for (let i = 0; i < lines.length; i++) {
    const match = lines[i].match(/^(?:Title:\s*|#\s*)(.*)/i);
    if (match) {
      refinedTitle = match[1].trim().replace(/^['"]|['"]$/g, '');
      const rest = lines.slice(i + 1);
      while (rest.length && !rest[0].trim()) rest.shift();
      raw = rest.join('\n');
      break;
    }
  }

  // Full post-processing pipeline
  raw = postprocessLyrics(raw);
  raw = fixSectionLabels(raw);
  raw = fixAPrefix(raw);
  raw = enforceLineCounts(raw);

  // Slop check on refined output
  const slopResult = slopDetector.scanForSlop(raw);
  if (slopResult.ai_score > 0) {
    console.warn(`Refinement slop scan: score=${slopResult.ai_score} severity=${slopResult.severity}`,
      'words:', slopResult.layers.blacklisted_words.found,
      'phrases:', slopResult.layers.blacklisted_phrases.found,
      'overuse:', slopResult.layers.overuse.found.map(o => `${o.word}(${o.count}x)`).join(', ') || 'none');
  }

  return {
    lyrics: raw,
    provider: providerName,
    model: effectiveModel,
    title: refinedTitle,
    subject: '',
    bpm: 0,
    key: '',
    caption: '',
    duration: 0,
    system_prompt: REFINEMENT_SYSTEM_PROMPT,
    user_prompt: userPrompt
  };
}
