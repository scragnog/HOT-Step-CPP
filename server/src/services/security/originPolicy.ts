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
  const o = origin.replace(/\/+$/, '').toLowerCase();
  const hl = host?.trim().toLowerCase();
  if (hl && IP_HOST.test(hl) && (o === `http://${hl}` || o === `https://${hl}`)) return true;
  if (extra.some(e => e.toLowerCase() === o)) return true;
  return false;
}

/** Returns 403 when the request must be refused for its Origin, else null. */
export function originRejection(method: string, path: string, origin: string | undefined, extra: readonly string[], host?: string): number | null {
  if (isAllowedOrigin(origin, extra, host)) return null;
  const isDaw = path === '/api/v1/daw' || path.startsWith('/api/v1/daw/');
  if (isDaw || !SAFE_METHODS.has(method.toUpperCase())) return 403;
  return null;
}

/** Splits a Host header into its lowercase hostname, without port or IPv6 brackets. */
export function hostnameOf(hostHeader: string): string {
  const h = hostHeader.trim().toLowerCase();
  const v6 = /^\[([^\]]+)\](:\d{1,5})?$/.exec(h);
  if (v6) return v6[1];
  return h.replace(/:\d{1,5}$/, '');
}

function isIpLiteral(h: string): boolean {
  return /^\d{1,3}(\.\d{1,3}){3}$/.test(h) || (h.includes(':') && /^[0-9a-f:.]+$/.test(h));
}

/** Hostnames allowed in the Host header besides loopback and IP literals: SERVER_HOST,
 *  the hosts of CORS_ORIGINS, and ALLOWED_HOSTS (comma list, for LAN or MagicDNS names). */
export function buildAllowedHosts(serverHost: string | undefined, corsOrigins: readonly string[], allowedHostsEnv: string | undefined): Set<string> {
  const set = new Set<string>();
  if (serverHost && serverHost.trim()) set.add(hostnameOf(serverHost));
  for (const o of corsOrigins) {
    try { set.add(hostnameOf(new URL(o).host)); } catch { /* ignore malformed */ }
  }
  for (const h of (allowedHostsEnv ?? '').split(',')) if (h.trim()) set.add(hostnameOf(h));
  return set;
}

/** DNS rebinding guard. A rebinding attack needs a hostname, so loopback names and IP
 *  literals pass; other hostnames must be listed. A missing Host header is allowed. */
export function isAllowedHost(hostHeader: string | undefined, allowed: ReadonlySet<string>): boolean {
  if (!hostHeader || !hostHeader.trim()) return true;
  const h = hostnameOf(hostHeader);
  if (isLoopbackHost(h) || isIpLiteral(h)) return true;
  return allowed.has(h);
}
