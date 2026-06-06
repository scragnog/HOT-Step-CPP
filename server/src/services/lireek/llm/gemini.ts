// llm/gemini.ts — Google Gemini provider

import { config } from '../../../config.js';
import { LLMProvider, readSSE } from './base.js';
import type { ProviderInfo, ChunkCallback } from './types.js';

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
      signal: AbortSignal.timeout(300_000),
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
