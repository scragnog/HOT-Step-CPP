// httpClient.ts — one configurable transport for every /api/* call.
//
// Replaces the get/post/patch/del boilerplate that api.ts and several
// per-studio clients each hand-rolled. Covers JSON, multipart upload (with
// progress), bearer auth, AbortSignal cancellation and media/audio URLs. SSE
// is not reimplemented here — it already has one shared connection manager
// (sharedEventSource.ts); this module only builds the URL.
//
// Incremental, not a rewrite: existing per-studio clients (lireekApi,
// trainingApi, stemStudioApi, ...) keep their own fetch wrappers for now and
// move onto this one in later slices. Nothing here changes a request body,
// header or error shape that a server route already expects.
//
// Error shape: every non-OK response throws ApiError, a superset of
// workflowApi's WorkflowRequestError (status + optional currentRevision +
// reason), so a revisioned caller can switch to this client later without
// losing that detail — `instanceof ApiError` is enough, callers don't need a
// second error class.

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;
  /** Set on a 409 caused by a stale revision (revisioned document/section writes). */
  readonly currentRevision?: number;
  /** e.g. 'unsupported-version' on a 409 for a stored document this build can't read. */
  readonly reason?: string;
  constructor(status: number, message: string, body: unknown, currentRevision?: number, reason?: string) {
    super(message);
    this.status = status;
    this.body = body;
    this.currentRevision = currentRevision;
    this.reason = reason;
  }
}

export interface ClientOptions {
  /** Defaults to '/api'. A client for a different mount (e.g. a per-studio
   *  prefix) sets its own; it does not need a second copy of this file. */
  baseUrl?: string;
  token?: string | null;
  signal?: AbortSignal;
}

function authHeaders(token?: string | null): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function parseErrorBody(res: Response): Promise<any> {
  return res.json().catch(() => ({ error: res.statusText }));
}

async function throwIfNotOk(res: Response): Promise<void> {
  if (res.ok) return;
  const body = await parseErrorBody(res);
  throw new ApiError(res.status, body.error || `API error: ${res.status}`, body, body.currentRevision, body.reason);
}

export class ApiClient {
  private readonly base: string;
  constructor(opts: ClientOptions = {}) {
    this.base = opts.baseUrl ?? '/api';
  }

  private url(path: string): string {
    return path.startsWith('http') ? path : `${this.base}${path}`;
  }

  async get<T>(path: string, opts: ClientOptions = {}): Promise<T> {
    const res = await fetch(this.url(path), { headers: authHeaders(opts.token), signal: opts.signal });
    await throwIfNotOk(res);
    return res.json();
  }

  async post<T>(path: string, body?: unknown, opts: ClientOptions = {}): Promise<T> {
    return this.send<T>('POST', path, body, opts);
  }

  async patch<T>(path: string, body?: unknown, opts: ClientOptions = {}): Promise<T> {
    return this.send<T>('PATCH', path, body, opts);
  }

  async put<T>(path: string, body?: unknown, opts: ClientOptions = {}): Promise<T> {
    return this.send<T>('PUT', path, body, opts);
  }

  async delete<T>(path: string, opts: ClientOptions = {}): Promise<T> {
    const res = await fetch(this.url(path), { method: 'DELETE', headers: authHeaders(opts.token), signal: opts.signal });
    await throwIfNotOk(res);
    return res.json();
  }

  private async send<T>(method: string, path: string, body: unknown, opts: ClientOptions): Promise<T> {
    const res = await fetch(this.url(path), {
      method,
      headers: { ...authHeaders(opts.token), ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: opts.signal,
    });
    await throwIfNotOk(res);
    return res.json();
  }

  /** Multipart upload via XHR, so `onProgress` can report upload fraction —
   *  fetch has no upload-progress event. `fields` are appended before the
   *  files, so a server that reads the form in order sees them first. */
  upload<T>(
    path: string,
    files: { field: string; file: File }[],
    opts: ClientOptions & { fields?: Record<string, string>; onProgress?: (fraction: number) => void } = {},
  ): Promise<T> {
    const form = new FormData();
    for (const [key, value] of Object.entries(opts.fields ?? {})) form.append(key, value);
    for (const { field, file } of files) form.append(field, file, file.name);

    return new Promise<T>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', this.url(path));
      if (opts.token) xhr.setRequestHeader('Authorization', `Bearer ${opts.token}`);
      if (opts.onProgress) {
        xhr.upload.onprogress = (e) => { if (e.lengthComputable) opts.onProgress!(e.loaded / e.total); };
      }
      if (opts.signal) {
        if (opts.signal.aborted) { reject(new DOMException('Aborted', 'AbortError')); return; }
        opts.signal.addEventListener('abort', () => xhr.abort());
      }
      xhr.onload = () => {
        let parsed: any = {};
        try { parsed = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(parsed);
        else reject(new ApiError(xhr.status, parsed.error || `Upload failed (HTTP ${xhr.status})`, parsed));
      };
      xhr.onerror = () => reject(new ApiError(0, 'Upload failed — the server closed the connection', null));
      xhr.onabort = () => reject(new DOMException('Aborted', 'AbortError'));
      xhr.send(form);
    });
  }

  /** The URL to pass to EventSource/sharedEventSource for a streaming route
   *  under this client's base. Opening the connection stays the caller's
   *  job — SSE auth here is a query param, never a header, since EventSource
   *  cannot set one. */
  streamUrl(path: string, opts: { token?: string | null } = {}): string {
    const url = this.url(path);
    if (!opts.token) return url;
    const sep = url.includes('?') ? '&' : '?';
    return `${url}${sep}token=${encodeURIComponent(opts.token)}`;
  }

  /** A `/audio` or `/references` URL for `<audio>`/`<img>` src — these are
   *  served from the app root, not under this client's `/api` base. */
  mediaUrl(path: string): string {
    return path.startsWith('http') || path.startsWith('/') ? path : `/${path}`;
  }
}

/** The default client, base `/api` — what api.ts and most callers want. */
export const apiClient = new ApiClient();
