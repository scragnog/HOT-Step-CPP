import { config } from '../../config.js';
import * as slopDetector from './slopDetector.js';
import { 
  GENERATION_SYSTEM_PROMPT, 
  SONG_METADATA_SYSTEM_PROMPT, 
  REFINEMENT_SYSTEM_PROMPT 
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

// ── Shared SSE Utility ──────────────────────────────────────────────────────

async function readSSE(response: Response, onChunk: ChunkCallback, extractText: (data: any) => string | null): Promise<string> {
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
          if (dataStr === '[DONE]') continue;
          
          try {
            const parsed = JSON.parse(dataStr);
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
  availableModels = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'];

  isAvailable() { return !!config.lireek.geminiApiKey; }

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
      return await readSSE(resp, onChunk, (data) => data.choices?.[0]?.delta?.content || null);
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
      return await readSSE(resp, onChunk, (data) => data.choices?.[0]?.delta?.content || null);
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

    return await readSSE(resp, onChunk || (() => {}), (data) => data.choices?.[0]?.delta?.content || null);
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
    if (p instanceof OllamaProvider || p instanceof LMStudioProvider || p instanceof UnslothProvider) {
      results.push(await p.toInfoAsync());
    } else {
      results.push(p.toInfo());
    }
  }
  return results;
}

// ── Orchestration Functions ─────────────────────────────────────────────────

export function stripThinkingBlocks(text: string): string {
  // Removes <think>...</think> blocks from text
  return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

/**
 * Plans the metadata before generating.
 */
async function planSongMetadata(
  profile: LyricsProfile,
  usedSubjects: string[],
  usedBpms: number[],
  usedKeys: string[],
  providerName: string,
  modelName: string,
  onChunk?: ChunkCallback
): Promise<any> {
  const provider = getProvider(providerName);
  const prompt = `
Themes to explore: ${(profile.themes || []).join(', ')}
Typical subjects: ${(profile.subject_categories || []).join(', ')}
  
Already used subjects (DO NOT REPEAT): ${(usedSubjects || []).join(' || ')}
Already used BPMs: ${(usedBpms || []).join(', ')}
Already used Keys: ${(usedKeys || []).join(', ')}
`;

  const responseJsonStr = await provider.call(SONG_METADATA_SYSTEM_PROMPT, prompt, modelName, onChunk);
  const cleanJson = stripThinkingBlocks(responseJsonStr).replace(/```json/g, '').replace(/```/g, '').trim();
  try {
    return JSON.parse(cleanJson);
  } catch (err) {
    console.error("Failed to parse metadata JSON:", cleanJson);
    return { subject: '', bpm: 0, key: '', caption: '', duration: 0 };
  }
}

function buildGenerationPrompt(profile: LyricsProfile, extraInstructions?: string, usedTitles: string[] = []): string {
  let prompt = '';
  if (profile.raw_summary) prompt += `Artist Summary:\n${profile.raw_summary}\n\n`;
  
  if (profile.rhyme_schemes?.length) prompt += `Preferred Rhyme Schemes: ${profile.rhyme_schemes.join(', ')}\n`;
  if (profile.repetition_stats?.hook_examples?.length) {
    prompt += `Hook Examples:\n${profile.repetition_stats.hook_examples.join('\n')}\n\n`;
  }
  if (profile.structural_patterns) prompt += `Structural Patterns:\n${profile.structural_patterns}\n\n`;
  
  if (usedTitles?.length) prompt += `DO NOT USE THESE TITLES:\n${usedTitles.join('\n')}\n\n`;
  
  if (extraInstructions) prompt += `=== EXTRA INSTRUCTIONS ===\n${extraInstructions}\n\n`;
  if (profile.examples?.length && profile.representative_excerpts?.length) {
    prompt += `=== LYRIC EXCERPTS ===\n${profile.representative_excerpts.slice(0, 10).join('\n---\n')}\n`;
  }
  return prompt;
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
  onChunk?: ChunkCallback,
  onPhase?: (phase: string) => void
): Promise<GenerationResponse> {
  const provider = getProvider(providerName);
  const effectiveModel = model || provider.defaultModel;

  if (onPhase) onPhase("Planning song metadata…");
  let metadata = { subject: '', bpm: 0, key: '', caption: '', duration: 0 };
  
  if (profile.song_subjects || (profile.themes && profile.themes.length)) {
    try {
      metadata = await planSongMetadata(profile, usedSubjects, usedBpms, usedKeys, providerName, effectiveModel, onChunk);
      console.log("Planned metadata:", metadata);
    } catch(e) {
      console.warn("Failed to plan metadata", e);
    }
  }

  if (metadata.subject) {
    extraInstructions = `The song must be about: ${metadata.subject}\n\n${extraInstructions || ''}`;
  }

  if (onPhase) onPhase("Writing lyrics…");
  const userPrompt = buildGenerationPrompt(profile, extraInstructions, usedTitles);

  let raw = await provider.call(GENERATION_SYSTEM_PROMPT, userPrompt, effectiveModel, onChunk);
  
  // Post process
  raw = stripThinkingBlocks(raw);
  raw = raw.replace(/<\\|im_end\\|>/g, '');
  raw = raw.replace(/\\[?(System|User|Assistant)\\]?:.*/gi, '');

  let title = '';
  const lines = raw.trim().split('\n');
  for (let i = 0; i < lines.length; i++) {
    const match = lines[i].match(/^Title:\s*(.+)/i);
    if (match) {
      title = match[1].trim().replace(/^['"]|['"]$/g, '');
      raw = lines.slice(i + 1).join('\n').trim();
      break;
    }
  }

  // Slop check
  const slopResult = slopDetector.scanForSlop(raw);
  if (slopResult.ai_score > 0) {
    console.warn(`Generation slop scan: score=${slopResult.ai_score}`);
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
    duration: metadata.duration,
    system_prompt: GENERATION_SYSTEM_PROMPT,
    user_prompt: userPrompt
  };
}

function buildRefinementPrompt(originalLyrics: string, artistName: string, title: string, profile?: LyricsProfile, originalSlop?: string[]): string {
  let lines = [`Artist: ${artistName}`, `Original Title: ${title}`, ''];

  if (profile) {
    lines.push('=== ARTIST STYLE CONTEXT (MATCH THIS) ===');
    lines.push(`Themes: ${profile.themes.slice(0, 8).join(', ')}`);
    if (profile.tone_and_mood) lines.push(`Tone/Mood: ${profile.tone_and_mood}`);
    if (profile.vocabulary_notes) lines.push(`Vocabulary Notes: ${profile.vocabulary_notes}`);
    lines.push('');
  }

  if (originalSlop?.length) {
    lines.push('=== KNOWN ISSUES TO FIX ===');
    lines.push('The original lyrics contain the following AI-cliches or circular phrases that MUST be replaced:');
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

  raw = stripThinkingBlocks(raw);
  raw = raw.replace(/<\\|im_end\\|>/g, '');
  raw = raw.replace(/\\s*\\((?:Hook|You|Repeat|x\\d|Refrain|Spoken|Whispered|Ad[- ]?lib|Echo)\\)\\s*/gi, '');
  
  let refinedTitle = title;
  const lines = raw.trim().split('\n');
  for (let i = 0; i < lines.length; i++) {
    const match = lines[i].match(/^Title:\s*(.+)/i);
    if (match) {
      refinedTitle = match[1].trim().replace(/^['"]|['"]$/g, '');
      raw = lines.slice(i + 1).join('\n').trim();
      break;
    }
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
