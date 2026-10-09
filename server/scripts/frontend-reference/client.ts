// client.ts — an independent reference client for HOT-Step CPP.
//
// Imports shared contracts only (types from server/src/contracts/), talks
// HTTP, and never imports ui/src or an execution service — not even
// transitively. This is the proof that the documented contracts are enough
// to build a replacement frontend without reading server/src's internals.

import type { WorkflowJob } from '../../src/contracts/workflow.js';
import { randomUUID } from 'node:crypto';

export class ClientError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

async function asJson<T>(res: Response): Promise<T> {
  const text = await res.text();
  let body: unknown;
  try { body = text ? JSON.parse(text) : undefined; } catch { body = text; }
  if (!res.ok) {
    const message = typeof body === 'object' && body && 'error' in (body as Record<string, unknown>)
      ? String((body as Record<string, unknown>).error)
      : text;
    throw new ClientError(res.status, message);
  }
  return body as T;
}

export class ReferenceClient {
  private _token = '';

  constructor(private readonly origin: string) {}

  /** The bearer token from login(), for a caller that needs to hit a route
   *  this client has no wrapper for yet (e.g. to assert on a raw error body). */
  get token(): string { return this._token; }

  private headers(extra: Record<string, string> = {}): Record<string, string> {
    return { Authorization: `Bearer ${this._token}`, ...extra };
  }

  // ── Auth ───────────────────────────────────────────────────────────────
  async login(): Promise<{ id: string; username: string }> {
    const body = await asJson<{ user: { id: string; username: string }; token: string }>(
      await fetch(`${this.origin}/api/auth/auto`),
    );
    this._token = body.token;
    return body.user;
  }

  // ── Create (text2music) ─────────────────────────────────────────────────
  async submitGeneration(body: Record<string, unknown>): Promise<{ jobId: string; status: string }> {
    return asJson(await fetch(`${this.origin}/api/generate`, {
      method: 'POST',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(body),
    }));
  }

  async generationStatus(jobId: string): Promise<Record<string, unknown>> {
    return asJson(await fetch(`${this.origin}/api/generate/status/${jobId}`, { headers: this.headers() }));
  }

  async cancelGeneration(jobId: string): Promise<Record<string, unknown>> {
    return asJson(await fetch(`${this.origin}/api/generate/cancel/${jobId}`, { method: 'POST', headers: this.headers() }));
  }

  /** Poll GET /api/generate/status/:id until a terminal status or timeoutMs. */
  async waitForGeneration(jobId: string, timeoutMs = 20_000): Promise<Record<string, unknown>> {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const status = await this.generationStatus(jobId);
      if (['succeeded', 'failed', 'cancelled'].includes(String(status.status))) return status;
      if (Date.now() > deadline) throw new Error(`generation ${jobId} did not finish within ${timeoutMs}ms (last status: ${status.status})`);
      await new Promise(r => setTimeout(r, 300));
    }
  }

  // ── Generic workflow envelope (Insta-Gen, Cover, Repaint, Lego) ─────────
  async submitWorkflowJob(kind: string, input: Record<string, unknown>, idempotencyKey = randomUUID()): Promise<WorkflowJob> {
    const body = await asJson<{ job: WorkflowJob }>(await fetch(`${this.origin}/api/workflows/jobs`, {
      method: 'POST',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ kind, idempotencyKey, input }),
    }));
    return body.job;
  }

  async getWorkflowJob(id: string): Promise<WorkflowJob> {
    const body = await asJson<{ job: WorkflowJob }>(await fetch(`${this.origin}/api/workflows/jobs/${id}`, { headers: this.headers() }));
    return body.job;
  }

  async cancelWorkflowJob(id: string): Promise<WorkflowJob> {
    const body = await asJson<{ job: WorkflowJob }>(await fetch(`${this.origin}/api/workflows/jobs/${id}/cancel`, { method: 'POST', headers: this.headers() }));
    return body.job;
  }

  async retryWorkflowJob(id: string): Promise<WorkflowJob> {
    const body = await asJson<{ job: WorkflowJob }>(await fetch(`${this.origin}/api/workflows/jobs/${id}/retry`, { method: 'POST', headers: this.headers() }));
    return body.job;
  }

  /** Poll GET /api/workflows/jobs/:id until a terminal status or timeoutMs. */
  async waitForWorkflowJob(id: string, timeoutMs = 20_000): Promise<WorkflowJob> {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const job = await this.getWorkflowJob(id);
      if (['succeeded', 'failed', 'cancelled', 'interrupted'].includes(job.status)) return job;
      if (Date.now() > deadline) throw new Error(`workflow job ${id} did not finish within ${timeoutMs}ms (last status: ${job.status})`);
      await new Promise(r => setTimeout(r, 300));
    }
  }

  async getDocument(id: string): Promise<{ document: { id: string; revision: number; data: unknown } }> {
    return asJson(await fetch(`${this.origin}/api/workflows/documents/${id}`, { headers: this.headers() }));
  }

  async putDocument(id: string, expectedRevision: number, data: unknown): Promise<{ document: { id: string; revision: number; data: unknown } }> {
    return asJson(await fetch(`${this.origin}/api/workflows/documents/${id}`, {
      method: 'PUT',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ expectedRevision, data }),
    }));
  }

  // ── Upload (source asset acquisition for Cover/stem workflows) ──────────
  async uploadAudio(wav: Buffer, filename = 'fixture.wav'): Promise<{ audio_url: string; filename: string; asset_id: string }> {
    const form = new FormData();
    form.append('audio', new Blob([wav], { type: 'audio/wav' }), filename);
    return asJson(await fetch(`${this.origin}/api/upload/audio`, {
      method: 'POST',
      headers: this.headers(),
      body: form,
    }));
  }

  // ── Stem Studio (extract) ───────────────────────────────────────────────
  async stemExtract(body: Record<string, unknown>): Promise<{ id: string }> {
    return asJson(await fetch(`${this.origin}/api/stem-studio/extract`, {
      method: 'POST',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(body),
    }));
  }

  async stemSupersep(body: Record<string, unknown>): Promise<{ id: string }> {
    return asJson(await fetch(`${this.origin}/api/stem-studio/supersep`, {
      method: 'POST',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(body),
    }));
  }

  async stemProgress(jobId: string): Promise<Record<string, unknown>> {
    return asJson(await fetch(`${this.origin}/api/stem-studio/${jobId}/progress`, { headers: this.headers() }));
  }

  async stemResult(jobId: string): Promise<Record<string, unknown>> {
    return asJson(await fetch(`${this.origin}/api/stem-studio/${jobId}/result`, { headers: this.headers() }));
  }

  /** Poll GET /api/stem-studio/:jobId/progress until done/failed or timeoutMs. */
  async waitForStemJob(jobId: string, timeoutMs = 20_000): Promise<Record<string, unknown>> {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const progress = await this.stemProgress(jobId);
      if (['done', 'failed'].includes(String(progress.status))) return progress;
      if (Date.now() > deadline) throw new Error(`stem job ${jobId} did not finish within ${timeoutMs}ms (last status: ${progress.status})`);
      await new Promise(r => setTimeout(r, 300));
    }
  }

  // ── SuperSep proxy (/api/supersep) ──────────────────────────────────────
  async supersepSeparate(audioUrl: string, level = 0): Promise<{ id: string }> {
    return asJson(await fetch(`${this.origin}/api/supersep/separate?level=${level}`, {
      method: 'POST',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ audioUrl }),
    }));
  }

  async supersepProgress(jobId: string): Promise<Record<string, unknown>> {
    return asJson(await fetch(`${this.origin}/api/supersep/${jobId}/progress`, { headers: this.headers() }));
  }

  async supersepResult(jobId: string): Promise<{ stems: Array<Record<string, unknown>> }> {
    return asJson(await fetch(`${this.origin}/api/supersep/${jobId}/result`, { headers: this.headers() }));
  }

  async supersepRelease(jobId: string): Promise<Record<string, unknown>> {
    return asJson(await fetch(`${this.origin}/api/supersep/${jobId}/release`, { method: 'POST', headers: this.headers() }));
  }
}
