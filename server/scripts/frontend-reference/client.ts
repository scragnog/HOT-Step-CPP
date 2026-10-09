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

  // ── Resolve (durable Create path: preview, then enqueue) ────────────────
  async resolvePreview(intent: Record<string, unknown>): Promise<{ request: Record<string, unknown>; version: string }> {
    return asJson(await fetch(`${this.origin}/api/resolve/preview`, {
      method: 'POST',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(intent),
    }));
  }

  async enqueueAudioIntent(request: Record<string, unknown>, idempotencyKey = randomUUID()): Promise<{ item: Record<string, unknown> }> {
    return asJson(await fetch(`${this.origin}/api/audio-queue/items`, {
      method: 'POST',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ idempotencyKey, request }),
    }));
  }

  async audioQueueItem(id: string): Promise<{ item: Record<string, unknown> }> {
    return asJson(await fetch(`${this.origin}/api/audio-queue/items/${id}`, { headers: this.headers() }));
  }

  /** Poll GET /api/audio-queue/items/:id until a terminal status or timeoutMs. */
  async waitForAudioIntent(id: string, timeoutMs = 20_000): Promise<Record<string, unknown>> {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const { item } = await this.audioQueueItem(id);
      if (['succeeded', 'failed', 'cancelled', 'interrupted'].includes(String(item.status))) return item;
      if (Date.now() > deadline) throw new Error(`audio intent ${id} did not finish within ${timeoutMs}ms (last status: ${item.status})`);
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

  // ── 7f-2: Song Builder, Library, Playlists, studio drafts, presets,
  // import/export and settings/backends. Added after the 7f-1 methods
  // above; those are unchanged. ──────────────────────────────────────────

  // ── Song Builder (/api/builder) ─────────────────────────────────────────
  async createBuilderProject(body: Record<string, unknown> = {}): Promise<{ project: Record<string, unknown>; sections: unknown[] }> {
    return asJson(await fetch(`${this.origin}/api/builder/projects`, {
      method: 'POST', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify(body),
    }));
  }

  async getBuilderProject(id: string): Promise<{ project: Record<string, unknown>; sections: unknown[] }> {
    return asJson(await fetch(`${this.origin}/api/builder/projects/${id}`, { headers: this.headers() }));
  }

  async listBuilderProjects(): Promise<{ projects: unknown[] }> {
    return asJson(await fetch(`${this.origin}/api/builder/projects`, { headers: this.headers() }));
  }

  async patchBuilderProject(id: string, body: Record<string, unknown>): Promise<{ project: Record<string, unknown>; sections: unknown[] }> {
    return asJson(await fetch(`${this.origin}/api/builder/projects/${id}`, {
      method: 'PATCH', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify(body),
    }));
  }

  async deleteBuilderProject(id: string): Promise<{ ok: true }> {
    return asJson(await fetch(`${this.origin}/api/builder/projects/${id}`, { method: 'DELETE', headers: this.headers() }));
  }

  async generateBuilderSection(projectId: string, body: Record<string, unknown>): Promise<{ project: Record<string, unknown>; sections: unknown[]; jobId: string; sectionId: string }> {
    return asJson(await fetch(`${this.origin}/api/builder/projects/${projectId}/sections/generate`, {
      method: 'POST', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify(body),
    }));
  }

  async chooseBuilderSection(sectionId: string, songId: string, expectedRevision: number): Promise<{ project: Record<string, unknown>; sections: unknown[] }> {
    return asJson(await fetch(`${this.origin}/api/builder/sections/${sectionId}/choose`, {
      method: 'POST', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify({ songId, expectedRevision }),
    }));
  }

  async stopBuilderSection(sectionId: string, expectedRevision: number): Promise<{ project: Record<string, unknown>; sections: unknown[] }> {
    return asJson(await fetch(`${this.origin}/api/builder/sections/${sectionId}/stop`, {
      method: 'POST', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify({ expectedRevision }),
    }));
  }

  async patchBuilderSection(sectionId: string, body: Record<string, unknown>): Promise<{ project: Record<string, unknown>; sections: unknown[] }> {
    return asJson(await fetch(`${this.origin}/api/builder/sections/${sectionId}`, {
      method: 'PATCH', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify(body),
    }));
  }

  async deleteBuilderSection(sectionId: string, expectedRevision: number): Promise<{ project: Record<string, unknown>; sections: unknown[] }> {
    return asJson(await fetch(`${this.origin}/api/builder/sections/${sectionId}?expectedRevision=${expectedRevision}`, {
      method: 'DELETE', headers: this.headers(),
    }));
  }

  // ── Library (/api/songs) ────────────────────────────────────────────────
  async listSongs(source?: string): Promise<{ songs: Array<Record<string, unknown>> }> {
    const qs = source ? `?source=${encodeURIComponent(source)}` : '';
    return asJson(await fetch(`${this.origin}/api/songs${qs}`, { headers: this.headers() }));
  }

  async songIds(): Promise<{ ids: string[] }> {
    return asJson(await fetch(`${this.origin}/api/songs/ids`, { headers: this.headers() }));
  }

  async recentSongs(source = 'all', limit = 50): Promise<{ songs: Array<Record<string, unknown>> }> {
    return asJson(await fetch(`${this.origin}/api/songs/recent?source=${source}&limit=${limit}`, { headers: this.headers() }));
  }

  async getSong(id: string): Promise<{ song: Record<string, unknown> }> {
    return asJson(await fetch(`${this.origin}/api/songs/${id}`, { headers: this.headers() }));
  }

  async createSong(body: Record<string, unknown>): Promise<{ song: Record<string, unknown> }> {
    return asJson(await fetch(`${this.origin}/api/songs`, {
      method: 'POST', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify(body),
    }));
  }

  async patchSong(id: string, body: Record<string, unknown>): Promise<{ song: Record<string, unknown> }> {
    return asJson(await fetch(`${this.origin}/api/songs/${id}`, {
      method: 'PATCH', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify(body),
    }));
  }

  async deleteSong(id: string): Promise<{ success: true }> {
    return asJson(await fetch(`${this.origin}/api/songs/${id}`, { method: 'DELETE', headers: this.headers() }));
  }

  async bulkDeleteSongs(ids: string[]): Promise<{ success: true; deletedCount: number }> {
    return asJson(await fetch(`${this.origin}/api/songs/bulk-delete`, {
      method: 'POST', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify({ ids }),
    }));
  }

  async importSongs(files: Array<{ buf: Buffer; filename: string }>, description?: string): Promise<{ songs: unknown[]; errors: unknown[]; accepted: string[] }> {
    const form = new FormData();
    for (const f of files) form.append('audio', new Blob([f.buf], { type: 'audio/wav' }), f.filename);
    if (description !== undefined) form.append('description', description);
    return asJson(await fetch(`${this.origin}/api/songs/import`, { method: 'POST', headers: this.headers(), body: form }));
  }

  // ── Playlist (/api/studio-drafts/playlist) ──────────────────────────────
  async getPlaylist(): Promise<{ document: { id: string; revision: number; body: { items: unknown[] } } | null }> {
    return asJson(await fetch(`${this.origin}/api/studio-drafts/playlist`, { headers: this.headers() }));
  }

  async playlistCommand(expectedRevision: number, command: Record<string, unknown>): Promise<{ document: { id: string; revision: number; body: { items: unknown[] } } }> {
    return asJson(await fetch(`${this.origin}/api/studio-drafts/playlist/commands`, {
      method: 'POST', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify({ expectedRevision, command }),
    }));
  }

  // ── Studio drafts (/api/studio-drafts/drafts) ───────────────────────────
  async listDrafts(studio?: string): Promise<{ documents: unknown[] }> {
    const qs = studio ? `?studio=${encodeURIComponent(studio)}` : '';
    return asJson(await fetch(`${this.origin}/api/studio-drafts/drafts${qs}`, { headers: this.headers() }));
  }

  async createDraft(body: Record<string, unknown>): Promise<{ document: { id: string; revision: number } }> {
    return asJson(await fetch(`${this.origin}/api/studio-drafts/drafts`, {
      method: 'POST', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify({ body }),
    }));
  }

  async getDraft(id: string): Promise<{ document: { id: string; revision: number }; sourceError: string | null }> {
    return asJson(await fetch(`${this.origin}/api/studio-drafts/drafts/${id}`, { headers: this.headers() }));
  }

  async putDraft(id: string, expectedRevision: number, body: Record<string, unknown>): Promise<{ document: { id: string; revision: number } }> {
    return asJson(await fetch(`${this.origin}/api/studio-drafts/drafts/${id}`, {
      method: 'PUT', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify({ body, expectedRevision }),
    }));
  }

  async deleteDraft(id: string, expectedRevision: number): Promise<{ removed: true }> {
    return asJson(await fetch(`${this.origin}/api/studio-drafts/drafts/${id}?expectedRevision=${expectedRevision}`, {
      method: 'DELETE', headers: this.headers(),
    }));
  }

  async importDraft(body: Record<string, unknown>): Promise<{ receipt: unknown; document: unknown; created: boolean }> {
    return asJson(await fetch(`${this.origin}/api/studio-drafts/import`, {
      method: 'POST', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify(body),
    }));
  }

  async getHandoff(id: string): Promise<{ document: unknown }> {
    return asJson(await fetch(`${this.origin}/api/studio-drafts/handoffs/${id}`, { headers: this.headers() }));
  }

  // ── Presets (/api/preferences) — installation-scoped, no token ─────────
  async listPresets(family: string): Promise<{ documents: Array<{ id: string; revision: number; body: unknown }> }> {
    return asJson(await fetch(`${this.origin}/api/preferences/presets/${family}`));
  }

  async createPreset(family: string, body: unknown): Promise<{ document: { id: string; revision: number; body: unknown } }> {
    return asJson(await fetch(`${this.origin}/api/preferences/presets/${family}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ body }),
    }));
  }

  async putPreset(family: string, id: string, expectedRevision: number, body: unknown): Promise<{ document: { id: string; revision: number; body: unknown } }> {
    return asJson(await fetch(`${this.origin}/api/preferences/presets/${family}/${id}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ expectedRevision, body }),
    }));
  }

  async deletePreset(family: string, id: string, expectedRevision: number): Promise<{ removed: true }> {
    return asJson(await fetch(`${this.origin}/api/preferences/presets/${family}/${id}?expectedRevision=${expectedRevision}`, { method: 'DELETE' }));
  }

  async importPresets(family: string, items: Array<Record<string, unknown>>): Promise<{ results: unknown[] }> {
    return asJson(await fetch(`${this.origin}/api/preferences/presets/${family}/import`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ items }),
    }));
  }

  async resolveYue2Preset(body: Record<string, unknown>): Promise<{ result: { effectiveForm: Record<string, unknown>; lyricTiming: boolean } }> {
    return asJson(await fetch(`${this.origin}/api/preferences/presets/yue2-joint/resolve`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }));
  }

  async getSingletonSetting(family: string): Promise<{ document: { id: string; revision: number; body: unknown } | null }> {
    return asJson(await fetch(`${this.origin}/api/preferences/settings/${family}`));
  }

  async putSingletonSetting(family: string, body: unknown, expectedRevision?: number): Promise<{ document: { id: string; revision: number; body: unknown } }> {
    return asJson(await fetch(`${this.origin}/api/preferences/settings/${family}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ expectedRevision, body }),
    }));
  }

  // ── Import/export (/api/export-import) ──────────────────────────────────
  async uploadAsset(wav: Buffer, filename = 'fixture.wav'): Promise<{ assetId: string }> {
    const form = new FormData();
    form.append('audio', new Blob([wav], { type: 'audio/wav' }), filename);
    return asJson(await fetch(`${this.origin}/api/export-import/assets`, { method: 'POST', headers: this.headers(), body: form }));
  }

  async importAssets(items: Array<{ assetId: string; description?: string }>): Promise<{ items: Array<Record<string, unknown>> }> {
    return asJson(await fetch(`${this.origin}/api/export-import/imports`, {
      method: 'POST', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify({ items }),
    }));
  }

  async resolveExport(body: Record<string, unknown>): Promise<{ items: Array<Record<string, unknown>> }> {
    return asJson(await fetch(`${this.origin}/api/export-import/exports/resolve`, {
      method: 'POST', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify(body),
    }));
  }

  /** GET the resolved download URL; returns the raw Response (binary body,
   *  no token needed) so the caller can check headers and bytes. */
  async download(url: string): Promise<Response> {
    return fetch(`${this.origin}${url}`);
  }

  async validateProfile(filename: string, profile: Record<string, unknown>): Promise<{ name: string; data: Record<string, unknown> }> {
    return asJson(await fetch(`${this.origin}/api/export-import/profiles/validate`, {
      method: 'POST', headers: this.headers({ 'Content-Type': 'application/json' }), body: JSON.stringify({ filename, profile }),
    }));
  }

  // ── Settings/backends (/api/settings, /api/backends, /api/capabilities) ─
  async getEnvSettings(): Promise<{ values: Record<string, string>; restartKeys: string[] }> {
    return asJson(await fetch(`${this.origin}/api/settings/env`));
  }

  async putEnvSettings(values: Record<string, string>): Promise<{ updated: string[]; restartRequired: boolean }> {
    return asJson(await fetch(`${this.origin}/api/settings/env`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values }),
    }));
  }

  async listBackends(): Promise<{ backends: Array<Record<string, unknown>>; activeId: string }> {
    return asJson(await fetch(`${this.origin}/api/backends`));
  }

  async setActiveBackend(id: string): Promise<{ activeId: string }> {
    return asJson(await fetch(`${this.origin}/api/backends/active`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id }),
    }));
  }

  async getCapabilities(backend?: string): Promise<Record<string, unknown>> {
    const qs = backend ? `?backend=${encodeURIComponent(backend)}` : '';
    return asJson(await fetch(`${this.origin}/api/capabilities${qs}`));
  }
}
