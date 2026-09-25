// assistant.ts — AI Assistant chat route with SSE streaming
//
// Provides a stateless chat endpoint that:
// 1. Loads the static knowledge base (cached in memory)
// 2. Injects the user's current generation settings as context
// 3. Streams the LLM response back via SSE
//
// Uses the existing LLM provider registry — same API keys as Lyric Studio.

import { Router } from 'express';
import type { Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { config, PORTABLE_MODE, PROJECT_ROOT } from '../config.js';
import { getProvider, listProviders } from '../services/lireek/llm/registry.js';
import { gpuLaneBusy } from '../services/generation/gpuLane.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const router = Router();

// ── Knowledge base (loaded once, cached in memory) ────────────────────────────

let knowledgeBase: string | null = null;

function loadKnowledge(): string {
  if (knowledgeBase) return knowledgeBase;
  const filePath = PORTABLE_MODE
    ? path.join(PROJECT_ROOT, 'server', 'data', 'assistant-knowledge.md')
    : path.resolve(__dirname, '../data/assistant-knowledge.md');
  try {
    knowledgeBase = fs.readFileSync(filePath, 'utf-8');
    console.log(`[Assistant] Knowledge base loaded (${(knowledgeBase.length / 1024).toFixed(1)} KB)`);
  } catch (err: any) {
    console.error(`[Assistant] Failed to load knowledge base: ${err.message}`);
    knowledgeBase = 'You are the HOT-Step Assistant. Help users configure their music generation settings.';
  }
  return knowledgeBase;
}

// ── SSE helpers ───────────────────────────────────────────────────────────────

function initSse(res: Response): (type: string, data: any) => void {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();
  return (type: string, data: any) => {
    res.write(`event: ${type}\ndata: ${JSON.stringify(data)}\n\n`);
  };
}

// ── Build system prompt ───────────────────────────────────────────────────────

function buildSystemPrompt(currentSettings: Record<string, any>): string {
  const knowledge = loadKnowledge();

  // Extract and label the active view for the LLM
  const viewLabels: Record<string, string> = {
    create: 'Create (text-to-music)',
    lyric: 'Lyric Studio (AI songwriting)',
    cover: 'Cover Studio (reference-based covers)',
    stems: 'Stem Studio (audio separation)',
    stemBuilder: 'Stem Builder (stem composition)',
    settings: 'Settings',
  };
  const activeView = currentSettings._activeView || 'create';
  const modeLabel = viewLabels[activeView] || activeView;

  // Remove internal-only fields before serializing
  const { _activeView, ...settingsForLLM } = currentSettings;
  const settingsJson = JSON.stringify(settingsForLLM, null, 2);

  return `${knowledge}

---

## User's Current Mode

The user is currently in: **${modeLabel}**

Tailor your responses to this mode. For example, if they're in Cover Studio, focus on cover-related settings and workflow. If they're in Create mode, focus on generation parameters and lyrics.

## User's Current Configuration

The following JSON represents the user's current generation settings. Reference these when answering questions or suggesting changes.

\`\`\`json
${settingsJson}
\`\`\`
`;
}

// ── Build user prompt with multi-turn history ────────────────────────────────

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

function buildUserPrompt(history: ChatMessage[], currentMessage: string): string {
  if (!history.length) return currentMessage;

  const lines: string[] = [];
  for (const msg of history) {
    const prefix = msg.role === 'user' ? 'User' : 'Assistant';
    lines.push(`${prefix}: ${msg.content}`);
  }
  lines.push(`User: ${currentMessage}`);
  return lines.join('\n\n');
}

// ── Routes ────────────────────────────────────────────────────────────────────

/**
 * POST /api/assistant/chat
 * 
 * Streams an assistant response via SSE.
 * Body: {
 *   message: string,
 *   history: ChatMessage[],
 *   currentSettings: Record<string, any>,
 *   provider: string,
 *   model?: string
 * }
 * 
 * SSE events:
 *   event: chunk  — { text: "..." }          (streaming token)
 *   event: complete — { text: "..." }        (full response)
 *   event: error — { error: "..." }          (on failure)
 */
router.post('/chat', async (req: Request, res: Response) => {
  try {
    const {
      message,
      history = [],
      currentSettings = {},
      provider: providerName,
      model,
    } = req.body;

    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'message is required' });
    }
    if (!providerName || typeof providerName !== 'string') {
      return res.status(400).json({ error: 'provider is required' });
    }

    // Resolve provider
    let provider;
    try {
      provider = getProvider(providerName);
    } catch {
      return res.status(400).json({ error: `Unknown provider: ${providerName}` });
    }

    if (!provider.isAvailable()) {
      return res.status(503).json({ error: `Provider ${providerName} is not available. Check API keys in Settings → AI Services.` });
    }

    // Build prompts
    const systemPrompt = buildSystemPrompt(currentSettings);
    const userPrompt = buildUserPrompt(history, message);
    const resolvedModel = model || provider.defaultModel;

    console.log(`[Assistant] Chat via ${providerName}/${resolvedModel} (${(systemPrompt.length / 1024).toFixed(1)}K system, ${(userPrompt.length / 1024).toFixed(1)}K user)`);

    // Set up SSE
    const sendSse = initSse(res);
    let fullText = '';

    // Call the provider with streaming
    const result = await provider.call(
      systemPrompt,
      userPrompt,
      resolvedModel,
      (chunk: string) => {
        fullText += chunk;
        sendSse('chunk', { text: chunk });
      },
    );

    // If the provider returned without streaming (some don't support onChunk),
    // send the full result as a single chunk
    if (!fullText && result) {
      fullText = result;
      sendSse('chunk', { text: result });
    }

    sendSse('complete', { text: fullText || result });
    res.end();
  } catch (err: any) {
    // The provider's abort fires exactly when a render holds the GPU and a
    // local model is being starved (#136): say so, and name the setting.
    if (/abort|timeout/i.test(`${err?.name} ${err?.message}`) && gpuLaneBusy()) {
      err.message = `The assistant timed out while the engine was busy with a generation or stem job. Wait for it to finish, or raise LLM_TIMEOUT_MS (currently ${config.lireek.llmTimeoutMs} ms) in Settings > Environment.`;
    }
    console.error('[Assistant] Chat error:', err.message);
    // If headers already sent (SSE started), send error event
    if (res.headersSent) {
      res.write(`event: error\ndata: ${JSON.stringify({ error: err.message })}\n\n`);
      res.end();
    } else {
      res.status(500).json({ error: err.message });
    }
  }
});

/**
 * GET /api/assistant/providers
 * 
 * Returns available LLM providers (reuses the same registry as Lyric Studio).
 */
router.get('/providers', async (_req: Request, res: Response) => {
  try {
    const providers = await listProviders();
    res.json(providers);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

export default router;
