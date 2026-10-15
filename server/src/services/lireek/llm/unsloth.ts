// llm/unsloth.ts — Unsloth Studio provider

import { config } from '../../../config.js';
import { LLMProvider, readSSE } from './base.js';
import type { ProviderInfo, ChunkCallback } from './types.js';

export class UnslothProvider extends LLMProvider {
  id = 'unsloth';
  get local() { return true; }
  name = 'Unsloth Studio';
  get defaultModel() { return config.lireek.unslothModel; }

  // True when Studio is running on this machine. A local instance serves its
  // /v1 endpoints without enforcing a real token — the login/JWT step only
  // exists for remote installs — so a local one needs no credentials at all
  // and we skip the /api/auth/login round-trip (see authenticate()).
  private get isLocal(): boolean {
    try {
      const host = new URL(config.lireek.unslothBaseUrl)
        .hostname.replace(/^\[|\]$/g, '').toLowerCase();
      return host === 'localhost' || host === '127.0.0.1' || host === '::1';
    } catch {
      return false;
    }
  }

  // The provider appends /v1/... (and /api/auth/login) to the base itself, so a
  // base pasted from an OpenAI-compat example — which already ends in /v1 — would
  // double up to /v1/v1/... and 404. Strip a trailing /v1 (and any trailing slash)
  // so both `http://host:8888` and `http://host:8888/v1` work.
  private get baseUrl(): string {
    return config.lireek.unslothBaseUrl.replace(/\/+$/, '').replace(/\/v1$/i, '');
  }
  
  private cachedToken = '';
  private tokenExpiry = 0;

  isAvailable() {
    // Local Studio needs no credentials (see isLocal), so it's always
    // available; a remote one still requires both username and password.
    if (this.isLocal) return true;
    return !!config.lireek.unslothUsername && !!config.lireek.unslothPassword;
  }

  private async authenticate(): Promise<string> {
    // Local Studio accepts any Bearer token on its /v1 endpoints (it only
    // enforces a real JWT on the login route), so skip authentication and use
    // a static placeholder — no username/password required.
    if (this.isLocal) return 'not-needed';

    const now = Date.now() / 1000;
    if (this.cachedToken && now < this.tokenExpiry - 60) return this.cachedToken;

    const payload = { email: config.lireek.unslothUsername, password: config.lireek.unslothPassword };
    const resp = await fetch(`${this.baseUrl}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    // Unsloth also sometimes takes username instead of email, trying generic handling if this fails
    if (!resp.ok) {
        const payload2 = { username: config.lireek.unslothUsername, password: config.lireek.unslothPassword };
        const resp2 = await fetch(`${this.baseUrl}/api/auth/login`, {
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
      const resp = await fetch(`${this.baseUrl}/v1/models`, {
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
      // The model chosen on the Settings page wins whenever the server
      // is actually serving it. This used to be `models[0]`, so the
      // configured model was only ever used when the server was
      // unreachable — i.e. the setting was dead exactly when it could
      // have worked, and the picker silently defaulted to whatever the
      // server happened to list first.
      default_model: this.preferredModel(models),
    };
  }

  async call(systemPrompt: string, userPrompt: string, model?: string, onChunk?: ChunkCallback): Promise<string> {
    const token = await this.authenticate();
    const modelName = model || (await this.getLocalModels())[0] || this.defaultModel;
    if (!modelName) throw new Error("No models loaded in Unsloth Studio");

    const url = `${this.baseUrl}/v1/chat/completions`;
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
      signal: AbortSignal.timeout(config.lireek.llmTimeoutMs),
    });

    if (!resp.ok) throw new Error(`Unsloth error: ${resp.status} ${await resp.text()}`);

    return await readSSE(resp, onChunk || (() => {}), (data) => data.choices?.[0]?.delta?.content || null, (data) => data.choices?.[0]?.delta?.reasoning_content || null);
  }
}
