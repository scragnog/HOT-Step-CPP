// llm/unsloth.ts — Unsloth Studio provider

import { config } from '../../../config.js';
import { LLMProvider, readSSE } from './base.js';
import type { ProviderInfo, ChunkCallback } from './types.js';

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
        headers: { 'Authorization': `Bearer ${token}` },
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
      signal: AbortSignal.timeout(300_000),
    });

    if (!resp.ok) throw new Error(`Unsloth error: ${resp.status} ${await resp.text()}`);

    return await readSSE(resp, onChunk || (() => {}), (data) => data.choices?.[0]?.delta?.content || null, (data) => data.choices?.[0]?.delta?.reasoning_content || null);
  }
}
