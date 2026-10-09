// safetyGuards.ts — fail-closed subprocess/network guards for fakeServer.ts.
//
// The fixture-only contract (plan root, "Safety limits"): nothing mounted in
// this harness may spawn a real process or reach a real network endpoint.
// Blocking the call isn't enough on its own — several production call sites
// (essentiaClient.ts's run(), training's workerUpdate.ts) deliberately
// swallow a subprocess failure and degrade to a null/skip result, so a
// thrown guard error can get caught and hidden exactly like a real
// missing-binary error would. `violations` is the record that survives
// that: it is pushed to before the guard throws, so a test can still assert
// it is empty at the end of a run even though production never saw anything
// but an ordinary error (safetyGuards.test.ts demonstrates this directly).
//
// Installed before any production module is imported (same ordering
// constraint fakeServer.ts documents for DATA_DIR/ACESTEPCPP_*), so every
// route module's own `import { execFile } from 'node:child_process'` and
// bare `fetch(...)` call resolve to the guarded version, not the real one —
// Node's CJS/ESM interop binds a named import as a live getter onto the
// SAME canonical module object `require()` returns, so mutating that object
// here is visible to a caller that imported `execFile` by name, even one
// that did so before this file ever ran. `import cp from 'node:child_process'`
// does NOT reach that object — a builtin's default export is a distinct
// value from the one backing its named-export getters, verified against
// Node 24.18 by patching through each separately and watching which one a
// child module's named import actually observed — so this must go through
// createRequire(), not a default ESM import.

import { createRequire } from 'node:module';
const cp = createRequire(import.meta.url)('node:child_process') as typeof import('node:child_process');

export const violations: string[] = [];

const SUBPROCESS_FNS = ['execFile', 'exec', 'spawn', 'execFileSync', 'spawnSync', 'execSync', 'fork'] as const;

// requestCapture.ts's checkoutState() and workerUpdate.ts's startupCommit
// both call `execFileSync('git', ['rev-parse'|'status', ...])` to stamp a
// commit/dirty flag on a captured fixture or a training status read. A real
// git process is still a real subprocess — round 2's "isolated but real" git
// call was wrong per the plan's explicit "no real subprocess" limit, not
// just risky — so this returns a fixed fixture string for exactly those two
// read-only subcommands instead of ever invoking git. Not a violation: it's
// a deliberate substitute answer, the same shape fakeEngine.ts is for the
// real ace-server.
const GIT_FIXTURE: Record<string, string> = {
  'rev-parse': '0000000000000000000000000000000000000000',
  status: '',
};
function gitFixture(file: unknown, args: unknown): string | undefined {
  if (file !== 'git' || !Array.isArray(args) || typeof args[0] !== 'string') return undefined;
  return GIT_FIXTURE[args[0]];
}

/** Replaces every process-spawning entry point in node:child_process with one
 *  that records the attempt and throws. Nothing else this batch's six groups
 *  cover needs a real subprocess — ffmpeg, essentia, whisper and the training
 *  runners are all either unreached or must be fixtured instead. */
export function installSubprocessGuard(): void {
  for (const name of SUBPROCESS_FNS) {
    (cp as unknown as Record<string, unknown>)[name] = (...args: unknown[]) => {
      if (name === 'execFileSync') {
        const fixture = gitFixture(args[0], args[1]);
        if (fixture !== undefined) return fixture;
      }
      violations.push(`child_process.${name}(${args.map(a => (typeof a === 'string' ? a : typeof a)).slice(0, 2).join(', ')})`);
      throw new Error(`[safety-guard] child_process.${name}() blocked — fixture it, never exec a real subprocess here`);
    };
  }
}

/** Exact origins (scheme+host+port) this harness's own fetch calls may reach
 *  — the fake server and the fake engine, added by fakeServer.ts once each
 *  is actually listening. Nothing else: not "127.0.0.1" generally, which
 *  would also let a real local service on another port (llama.cpp's default
 *  :8080, a worker, a dev server) answer as if it were the fixture. */
export const allowedOrigins = new Set<string>();

/** Replaces global fetch with one that only reaches `allowedOrigins`, exactly,
 *  and never follows a redirect — a redirect response is blocked even when
 *  its target would itself be on the allowlist, since nothing this harness
 *  fixtures ever legitimately redirects. */
export function installNetworkGuard(): void {
  const real = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    let origin: string | null = null;
    try {
      origin = new URL(raw).origin;
    } catch {
      // Not an absolute URL — could be a legitimate relative path (resolves
      // against some base, so same-origin and allowed) or genuinely
      // malformed (reaches native fetch, which rejects it anyway, but with
      // no violation recorded — Reviewer's 7f-1 nit). Tell those apart by
      // trying the relative parse explicitly rather than assuming "not
      // absolute" means "relative".
      try { new URL(raw, 'http://127.0.0.1'); } catch {
        violations.push(`fetch(malformed: ${raw})`);
        throw new Error(`[safety-guard] fetch to malformed URL '${raw}' blocked`);
      }
    }
    if (origin && !allowedOrigins.has(origin)) {
      violations.push(`fetch(${raw})`);
      throw new Error(`[safety-guard] fetch to '${raw}' blocked — only ${[...allowedOrigins].join(', ') || '(nothing yet)'} are allowed in this fixture`);
    }
    const res = await real(input, { ...init, redirect: 'manual' });
    // Node's fetch (undici) does not implement the browser opaqueredirect
    // response type for a manual redirect — it hands back the raw 3xx with
    // its Location header intact (verified against Node 24.18), so that is
    // what a redirect attempt looks like here, not `res.type`.
    if (res.status >= 300 && res.status < 400 && res.headers.has('location')) {
      violations.push(`fetch redirect from ${raw} to ${res.headers.get('location')}`);
      throw new Error(`[safety-guard] redirect from '${raw}' blocked — no fixture origin may redirect`);
    }
    return res;
  }) as typeof fetch;
}
