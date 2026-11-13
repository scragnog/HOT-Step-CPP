// Bind-host and Origin policy for the Node server. Pure functions so they can be unit tested.
//
// Default bind is loopback. LAN or Tailscale access is opt-in via SERVER_HOST.
// Browsers always send Origin on cross-origin requests, so a request whose Origin
// is present but not trusted is treated as cross-site and refused when it would
// change state. Requests with no Origin (curl, native DAW plugins, same-origin GETs)
// are not affected by this check.

const LOOPBACK_ORIGIN = /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d{1,5})?$/i;
const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS']);

export function defaultServerHost(env: string | undefined): string {
  return env && env.trim() ? env.trim() : '127.0.0.1';
}

export function isLoopbackHost(host: string): boolean {
  const h = host.trim().toLowerCase().replace(/^\[|\]$/g, '');
  return h === 'localhost' || h === '::1' || /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(h);
}

/** CORS_ORIGINS: comma separated exact origins, trailing slashes ignored. */
export function parseCorsOrigins(raw: string | undefined): string[] {
  return (raw ?? '').split(',').map(s => s.trim().replace(/\/+$/, '')).filter(Boolean);
}

const IP_HOST = /^(\d{1,3}(\.\d{1,3}){3}|\[[0-9a-f:.]+\])(:\d{1,5})?$/i;

/** `host` is the request Host header. A same-origin page reached by IP literal (LAN or
 *  Tailscale) is allowed; hostnames are not, so DNS rebinding cannot pass this way. */
export function isAllowedOrigin(origin: string | undefined, extra: readonly string[], host?: string): boolean {
  if (!origin) return true;
  if (LOOPBACK_ORIGIN.test(origin)) return true;
  const o = origin.replace(/\/+$/, '');
  if (host && IP_HOST.test(host) && (o === `http://${host}` || o === `https://${host}`)) return true;
  return extra.includes(o);
}

/** Returns 403 when the request must be refused for its Origin, else null. */
export function originRejection(method: string, path: string, origin: string | undefined, extra: readonly string[], host?: string): number | null {
  if (isAllowedOrigin(origin, extra, host)) return null;
  const isDaw = path === '/api/v1/daw' || path.startsWith('/api/v1/daw/');
  if (isDaw || !SAFE_METHODS.has(method.toUpperCase())) return 403;
  return null;
}
