// preferencesApi.ts — client for /api/preferences (server/src/routes/preferences.ts).
//
// No auth token: every family here is installation-scoped (no user id in any
// source browser key), same as /api/vst/chain. Throws WorkflowRequestError
// on a non-2xx response, carrying currentRevision for a 409 and 'name-conflict'
// results come back inline (not an error) so a batch import can report
// per-item outcomes.
import type {
  AiContinuePresetBody, AiContinueTemplateBody, ImportPreferenceItem, ImportPreferenceResult,
  StormTuningBody, VstChainPresetBody, Yue2JointPresetBody,
} from '../../../server/src/contracts/preferences';
import type { TypedDocument } from '../../../server/src/contracts/workflow';
import { WorkflowRequestError } from './workflowApi';

export type { ImportPreferenceItem, ImportPreferenceResult };
export type PresetFamily = 'vst-chain' | 'ai-continue-style' | 'ai-continue-lyric' | 'yue2-joint';
export type SettingsFamily = 'ai-continue-template' | 'storm-tuning';

async function request<T>(url: string, method = 'GET', body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const value = await res.json().catch(() => ({}));
  if (!res.ok) throw new WorkflowRequestError(res.status, value.error || `Request failed (${res.status})`, value.currentRevision, value.reason);
  return value as T;
}

const id = (v: string) => encodeURIComponent(v);

/** Browser-safe sha256 of a serialized value, in the `sha256:<hex>` form the
 *  server expects (same hashing as ui/src/components/lyric-studio/playlistStore.ts). */
export async function hashImportValue(raw: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(raw));
  return `sha256:${Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join('')}`;
}

export const preferencesApi = {
  presets: {
    list: <T>(family: PresetFamily) => request<{ documents: TypedDocument<T>[] }>(`/api/preferences/presets/${family}`),
    create: <T>(family: PresetFamily, body: T) => request<{ document: TypedDocument<T> }>(`/api/preferences/presets/${family}`, 'POST', { body }),
    update: <T>(family: PresetFamily, docId: string, expectedRevision: number, body: T) =>
      request<{ document: TypedDocument<T> }>(`/api/preferences/presets/${family}/${id(docId)}`, 'PUT', { expectedRevision, body }),
    remove: (family: PresetFamily, docId: string, expectedRevision: number) =>
      request<{ removed: true }>(`/api/preferences/presets/${family}/${id(docId)}?expectedRevision=${expectedRevision}`, 'DELETE'),
    import: (family: PresetFamily, items: ImportPreferenceItem[]) =>
      request<{ results: ImportPreferenceResult[] }>(`/api/preferences/presets/${family}/import`, 'POST', { items }),
  },
  settings: {
    get: <T>(family: SettingsFamily) => request<{ document: TypedDocument<T> | null }>(`/api/preferences/settings/${family}`),
    upsert: <T>(family: SettingsFamily, expectedRevision: number | undefined, body: T) =>
      request<{ document: TypedDocument<T> }>(`/api/preferences/settings/${family}`, 'PUT', { expectedRevision, body }),
    import: (family: SettingsFamily, item: ImportPreferenceItem) =>
      request<{ results: ImportPreferenceResult[] }>(`/api/preferences/settings/${family}/import`, 'POST', { items: [item] })
        .then(({ results }) => results[0]!),
  },
};

export type { VstChainPresetBody, AiContinuePresetBody, AiContinueTemplateBody, Yue2JointPresetBody, StormTuningBody };
