// api.ts — Frontend API client
//
// Thin wrapper around fetch() for all server endpoints.
// Each method is standalone — import only what you need.

import type { Song, GenerationParams, GenerationJob, AuthState, AceModels } from '../types';

const BASE = '/api';

async function get<T>(path: string, token?: string | null): Promise<T> {
  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error: ${res.status}`);
  }
  return res.json();
}

async function post<T>(path: string, body?: unknown, token?: string | null): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error: ${res.status}`);
  }
  return res.json();
}

async function patch<T>(path: string, body: unknown, token?: string | null): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, {
    method: 'PATCH',
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error: ${res.status}`);
  }
  return res.json();
}

async function del<T>(path: string, token?: string | null): Promise<T> {
  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE}${path}`, { method: 'DELETE', headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `API error: ${res.status}`);
  }
  return res.json();
}

// ── Auth ────────────────────────────────────────────────────
export const authApi = {
  autoLogin: () => get<AuthState>('/auth/auto'),
  getMe: (token: string) => get<{ user: AuthState['user'] }>('/auth/me', token),
  updateUsername: (username: string, token: string) =>
    patch<AuthState>('/auth/username', { username }, token),
};

// ── Songs ────────────────────────────────────────────────────
export const songApi = {
  list: (token: string) => get<{ songs: Song[] }>('/songs', token),
  get: (id: string) => get<{ song: Song }>(`/songs/${id}`),
  create: (song: Partial<Song>, token: string) => post<{ song: Song }>('/songs', song, token),
  update: (id: string, data: Partial<Song>, token: string) =>
    patch<{ song: Song }>(`/songs/${id}`, data, token),
  delete: (id: string, token: string) => del<{ success: boolean }>(`/songs/${id}`, token),
  deleteAll: (token: string) => del<{ success: boolean; deletedCount: number }>('/songs', token),
};

// ── Generation ──────────────────────────────────────────────
export const generateApi = {
  submit: (params: GenerationParams, token: string) =>
    post<{ jobId: string; status: string }>('/generate', params, token),
  status: (jobId: string) => get<GenerationJob>(`/generate/status/${jobId}`),
  cancel: (jobId: string) => post<{ success: boolean }>(`/generate/cancel/${jobId}`),
  cancelAll: () => post<{ success: boolean; cancelled: number }>('/generate/cancel-all'),
};

// ── Models ──────────────────────────────────────────────────
export const modelApi = {
  list: () => get<AceModels>('/models'),
  health: () => get<{ aceServer: string }>('/models/health'),
};

// ── Health ──────────────────────────────────────────────────
export const healthApi = {
  check: () => get<{
    status: string;
    aceServer: { status: string; url: string; version: string };
    server: { port: number; uptime: number };
  }>('/health'),
};

// ── Shutdown ────────────────────────────────────────────────
export const shutdownApi = {
  quit: () => post<{ success: boolean; message: string }>('/shutdown'),
};

