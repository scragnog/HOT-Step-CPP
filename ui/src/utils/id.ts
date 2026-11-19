/**
 * A fresh client-side id.
 *
 * `crypto.randomUUID` only exists in a secure context, so on the LAN origin the
 * app is commonly opened from (`http://192.168.x.x:port`) it is undefined and
 * every idempotency key or draft id threw. newId keeps the strong path and falls
 * back to a time+random id that is unique enough for a request key.
 */
export function newId(): string {
  const c = typeof crypto === 'undefined' ? undefined : crypto;
  if (c && typeof c.randomUUID === 'function') return c.randomUUID();
  return `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
