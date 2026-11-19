// axiosClient.ts — one shared axios instance for all JSON /api/* traffic.
//
// The transport layer between TanStack Query (caching, dedup, retry) and REST:
// - the request interceptor injects the bearer from a module-level session
//   accessor that AuthProvider updates at the same site that calls
//   startAuthSession — the token lives in auth state, not localStorage, and
//   fetchers no longer thread it per call;
// - the response interceptor is the global catch: every failure is normalized
//   to ApiError, logged once here, and toasted per the filters below. Errors
//   have exactly one notifier — this module. Query-layer callbacks (onError,
//   MutationCache) never toast; success toasts are contextual and live in
//   mutation onSuccess.
//
// The adapter runs on globalThis.fetch (not axios's built-in adapters) so the
// Node test stubs — which replace globalThis.fetch — keep intercepting every
// request unchanged, and abort/timeout semantics stay explicit. Multipart
// uploads stay on XHR in httpClient.ts (only XHR reports upload progress);
// their failures call handleFailure() below, so there is still exactly one
// place that decides what is logged and toasted.

import axios, { AxiosError, CanceledError, type AxiosResponse, type InternalAxiosRequestConfig } from 'axios';
import { toast } from 'react-toastify';
import i18n from '@/i18n';
import { ApiError, HttpStatus } from './apiError';
import { errorMessage } from '@/utils/format/errorMessage';

/** Default per-request timeout. Known-slow routes pass SLOW_ROUTE_TIMEOUT_MS
 *  instead — a blanket short timeout would break them. */
export const DEFAULT_TIMEOUT_MS = 60_000;

/** For routes that block server-side for minutes (adapter scans, folder
 *  browse, Whisper retranscribe, mastering runs, external LLM calls): without
 *  the override they die at the default and report an unreachable server. */
export const SLOW_ROUTE_TIMEOUT_MS = 300_000;

// ---------------------------------------------------------------------------
// Auth session accessor — the only place the transport reads the bearer from.
// ---------------------------------------------------------------------------

let authToken: string | null = null;
/** AuthProvider calls this at the same site that calls startAuthSession, so no
 *  request ever pairs a new token with the old session's keys. */
export function setAuthToken(token: string | null): void {
  authToken = token;
}
export function getAuthToken(): string | null {
  return authToken;
}

/** Registered by AuthProvider: re-run auto-login when the session expires. The
 *  interceptor fires it (and the matching toast) at most once per cooldown, so
 *  a burst of 401s while re-auth runs triggers one re-login and one warning. */
let onUnauthorized: (() => void) | null = null;
export function setOnUnauthorized(cb: (() => void) | null): void {
  onUnauthorized = cb;
}

// ---------------------------------------------------------------------------
// Global error filtering + notification (shared with the XHR upload path).
// ---------------------------------------------------------------------------

/** Endpoints that are never user-facing even as commands — silent always. */
export const IGNORED_URL_PATTERNS: RegExp[] = [
  /\/api\/health/,
];

const UNAUTHORIZED_COOLDOWN_MS = 30_000;
let lastUnauthorizedAt = 0;

/** One translation key per status the app branches on; any other status gets
 *  the generic line. The keys are flat `api.error.*` names in the locale
 *  files (i18n/locales/*.json) — non-English locales fall back to English
 *  until they are translated. Status codes stay out of the text: end users
 *  are not developers, a number tells them nothing they can act on. */
const STATUS_MESSAGE_KEYS: Partial<Record<number, string>> = {
  [HttpStatus.unauthorized]: 'api.error.sessionExpired',
  [HttpStatus.forbidden]: 'api.error.refused',
  [HttpStatus.notFound]: 'api.error.notFound',
};

function messageFor(status: number, body: any): string {
  if (!status) return i18n.t('api.error.unreachable');
  const server = typeof body?.error === 'string' ? body.error : undefined;
  if (server) return server;
  const key = STATUS_MESSAGE_KEYS[status];
  return key ? i18n.t(key) : i18n.t('api.error.serverError');
}

function toastIfWindow(fn: () => void): void {
  // Node test runs have no Toaster mounted; react-toastify would warn per call.
  if (typeof window === 'undefined') return;
  fn();
}

export interface FailureInfo {
  status: number;
  body: unknown;
  method?: string;
  url?: string;
  /** The request opted out via X-Skip-Global-Error (ClientOptions.silent). */
  skip?: boolean;
}

/** The one place a transport failure becomes an ApiError and gets logged or
 *  toasted. Filters, in order: aborts never surface; GET errors never toast
 *  (reads retry per the contract policy and surface inline isError state — this
 *  is the primary storm guard, a failing 1.5 s poll toasts zero times); then
 *  the per-request skip flag and IGNORED_URL_PATTERNS. 401s get one deduped
 *  warning plus the re-auth callback regardless of method. */
export function handleFailure(info: FailureInfo): never {
  const body = info.body as any;
  const apiErr = new ApiError(
    info.status,
    messageFor(info.status, body),
    info.body ?? null,
    body?.currentRevision,
    body?.reason,
  );

  if (info.status === HttpStatus.unauthorized) {
    const now = Date.now();
    if (now - lastUnauthorizedAt >= UNAUTHORIZED_COOLDOWN_MS) {
      lastUnauthorizedAt = now;
      console.error('[api] 401 — session expired', info.url ?? '');
      onUnauthorized?.();
      toastIfWindow(() => toast.warn(apiErr.message));
    }
    throw apiErr;
  }

  const method = (info.method ?? 'get').toLowerCase();
  const silent = method === 'get' || info.skip === true
    || IGNORED_URL_PATTERNS.some(re => re.test(info.url ?? ''));
  if (!silent) {
    console.error('[api]', method.toUpperCase(), info.url ?? '', info.status || 'network', apiErr.message);
    toastIfWindow(() => toast.error(apiErr.message));
  }
  throw apiErr;
}

function toAxiosError(err: unknown, config?: InternalAxiosRequestConfig): AxiosError {
  if (err instanceof AxiosError) return err;
  const name = (err as any)?.name;
  if (name === 'TimeoutError') return new AxiosError(`timeout of ${config?.timeout}ms exceeded`, 'ECONNABORTED', config);
  if (name === 'AbortError' || config?.signal?.aborted) return new CanceledError(undefined, config);
  return new AxiosError(errorMessage(err), 'ERR_NETWORK', config);
}

/** The axios response interceptor: normalize, log/toast per the filters, rethrow. */
export function reportApiError(err: unknown): never {
  if (axios.isCancel(err)) throw err; // aborts are not errors: no log, no toast
  const ax = toAxiosError(err, err instanceof AxiosError ? err.config : undefined);
  return handleFailure({
    status: ax.response?.status ?? 0,
    body: ax.response?.data ?? null,
    method: ax.config?.method,
    url: ax.config?.url,
    skip: Boolean(ax.config?.headers?.['X-Skip-Global-Error']),
  });
}

// ---------------------------------------------------------------------------
// Adapter on globalThis.fetch — see the module header for why.
// ---------------------------------------------------------------------------

function isAbsoluteUrl(url: string): boolean {
  return /^[a-z][a-z\d+\-.]*:/i.test(url);
}

function headersRecord(h: Headers): Record<string, string> {
  const out: Record<string, string> = {};
  h.forEach((v, k) => { out[k] = v; });
  return out;
}

async function fetchAdapter(config: InternalAxiosRequestConfig): Promise<AxiosResponse> {
  const requested = config.url ?? '';
  let url = isAbsoluteUrl(requested) ? requested : `${config.baseURL ?? ''}${requested}`;
  if (config.params) {
    const sp = new URLSearchParams();
    for (const [k, v] of Object.entries(config.params)) {
      if (v === undefined || v === null) continue;
      sp.set(k, typeof v === 'object' ? JSON.stringify(v) : String(v));
    }
    const qs = sp.toString();
    if (qs) url += (url.includes('?') ? '&' : '?') + qs;
  }

  // config.headers is an AxiosHeaders instance by adapter time; it has no
  // forEach in axios 1.x — toJSON() is the plain-record view.
  const headers: Record<string, string> = {};
  for (const [key, value] of Object.entries(config.headers?.toJSON() ?? {})) {
    if (value !== undefined && value !== null) headers[key] = String(value);
  }
  let body: BodyInit | undefined;
  if (config.data !== undefined) {
    if (typeof config.data === 'string' || config.data instanceof FormData || config.data instanceof Blob) {
      body = config.data as BodyInit;
    } else {
      body = JSON.stringify(config.data);
      headers['Content-Type'] = 'application/json';
    }
  }

  // axios types config.signal as its own GenericAbortSignal; every caller here
  // passes a native AbortSignal (TanStack Query, our own controllers).
  const signals: AbortSignal[] = [];
  if (config.signal) signals.push(config.signal as AbortSignal);
  if (config.timeout) signals.push(AbortSignal.timeout(config.timeout));
  const signal = signals.length === 1 ? signals[0] : signals.length > 1 ? AbortSignal.any(signals) : undefined;

  let res: Response;
  try {
    res = await globalThis.fetch(url, { method: (config.method ?? 'get').toUpperCase(), headers, body, signal });
  } catch (err) {
    throw toAxiosError(err, config);
  }

  const responseHeaders = headersRecord(res.headers);
  if (!res.ok) {
    let data: unknown;
    try { data = await res.json(); } catch { data = undefined; } // non-JSON error page
    const response: AxiosResponse = { data, status: res.status, statusText: res.statusText, headers: responseHeaders, config };
    throw new AxiosError(`Request failed with status code ${res.status}`, String(res.status), config, {}, response);
  }

  // Binary routes (e.g. supersep recombine returns a WAV) opt in with
  // responseType 'blob'; everything else is JSON or null on a non-JSON body.
  let data: unknown;
  if (config.responseType === 'blob') data = await res.blob();
  else { try { data = await res.json(); } catch { data = null; } }
  return { data, status: res.status, statusText: res.statusText, headers: responseHeaders, config, request: {} };
}

// ---------------------------------------------------------------------------
// The instance.
// ---------------------------------------------------------------------------

export const http = axios.create({
  baseURL: '/api',
  timeout: DEFAULT_TIMEOUT_MS,
  // Stringify in the adapter instead, so the wire bytes (and its exact
  // 'application/json' content type) match the old fetch transport.
  transformRequest: [(data: unknown) => data],
  adapter: fetchAdapter,
});

http.interceptors.request.use((config) => {
  const token = getAuthToken();
  if (token) config.headers.set('Authorization', `Bearer ${token}`);
  return config;
});

http.interceptors.response.use(
  (res) => res,
  reportApiError,
);
