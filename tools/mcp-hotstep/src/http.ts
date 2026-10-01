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

async function login(): Promise<string> {
  const res = await fetch(`${BASE_URL}/api/auth/auto`, { signal: AbortSignal.timeout(TIMEOUT_MS) });
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
   *  than paraphrasing whatever the server said. */
  text: string;
}

export async function request<T = unknown>(
  method: string,
  path: string,
  body?: unknown,
  opts: { retryOn401?: boolean } = {},
): Promise<HttpResult<T>> {
  if (!token) await login();
  const send = () => fetch(`${BASE_URL}${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(body !== undefined ? { 'content-type': 'application/json' } : {}),
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });

  let res = await send();
  if (res.status === 401) {
    token = null;
    if (opts.retryOn401 !== false) {
      await login();
      res = await send();
    }
  }

  const text = await res.text();
  let data: T | undefined;
  try { data = text ? JSON.parse(text) : undefined; } catch { /* non-JSON error body — text carries it */ }
  return { ok: res.ok, status: res.status, data, text };
}
