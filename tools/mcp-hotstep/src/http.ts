// http.ts — the one HTTP client every tool goes through.
//
// Auto-login (GET /api/auth/auto) on first use, one retry on a 401 with a
// fresh login — except for a caller that opts out (gen_submit: resending a
// generation body after a silent relogin risks a double-submitted GPU job,
// so a submit that 401s returns the failure and clears the cached token for
// the NEXT call instead of retrying this one).

const BASE_URL = (process.env.HOTSTEP_URL || 'http://127.0.0.1:3001').replace(/\/+$/, '');
const TIMEOUT_MS = 10_000;

let token: string | null = null;

/** Test-only: force a relogin on the next request. */
export function resetToken(): void {
  token = null;
}

export function hotstepBaseUrl(): string {
  return BASE_URL;
}

/** True when HOTSTEP_URL points at this machine — the only case a filesystem
 *  path alongside a song's audio_url means anything. */
export function hotstepIsLoopback(): boolean {
  try {
    const host = new URL(BASE_URL).hostname;
    return host === '127.0.0.1' || host === 'localhost' || host === '::1';
  } catch {
    return false;
  }
}

/** One AbortSignal that aborts as soon as any of `signals` does — portable
 *  (AbortSignal.any needs Node 20.3+; this repo supports 18-22). */
function anySignal(signals: AbortSignal[]): AbortSignal {
  const controller = new AbortController();
  for (const s of signals) {
    if (s.aborted) { controller.abort(s.reason); break; }
    s.addEventListener('abort', () => controller.abort(s.reason), { once: true });
  }
  return controller.signal;
}

/** No timeout of its own — shares whatever signal the caller built. */
async function login(signal: AbortSignal): Promise<string> {
  const res = await fetch(`${BASE_URL}/api/auth/auto`, { signal });
  const text = await res.text();
  if (!res.ok) throw new Error(`auto-login failed: HTTP ${res.status} ${text}`);
  let data: { token?: string };
  try { data = JSON.parse(text); } catch { throw new Error(`auto-login returned non-JSON: ${text.slice(0, 200)}`); }
  if (!data.token) throw new Error('auto-login response had no token');
  token = data.token;
  return token;
}

export interface HttpResult<T = unknown> {
  ok: boolean;
  status: number;
  data?: T;
  /** The raw response body — tools return this verbatim on failure rather
   *  than paraphrasing whatever the server said. 0 status + this message is
   *  a local abort/timeout: the caller's signal fired, or nothing arrived in
   *  time — login, the fetch, the 401 retry, and reading the body are ALL
   *  covered by this, never left to throw past this function. */
  text: string;
}

export async function request<T = unknown>(
  method: string,
  path: string,
  body?: unknown,
  opts: { retryOn401?: boolean; signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<HttpResult<T>> {
  // One combined signal for the WHOLE call — login, both sends, and reading
  // the body all share it, so a caller's deadline/cancel bounds every step,
  // not just the final fetch.
  const timeout = AbortSignal.timeout(Math.max(0, opts.timeoutMs ?? TIMEOUT_MS));
  const signal = opts.signal ? anySignal([timeout, opts.signal]) : timeout;
  try {
    if (!token) await login(signal);
    const send = () => fetch(`${BASE_URL}${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${token}`,
        ...(body !== undefined ? { 'content-type': 'application/json' } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal,
    });

    let res = await send();
    if (res.status === 401) {
      token = null;
      if (opts.retryOn401 !== false) {
        await login(signal);
        res = await send();
      }
    }

    const text = await res.text();
    let data: T | undefined;
    try { data = text ? JSON.parse(text) : undefined; } catch { /* non-JSON error body — text carries it */ }
    return { ok: res.ok, status: res.status, data, text };
  } catch (err: any) {
    return { ok: false, status: 0, text: err?.message || String(err) };
  }
}
