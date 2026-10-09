// safetyGuards.ts — fail-closed subprocess/network guards for fakeServer.ts.
//
// The fixture-only contract (plan root, "Safety limits"): nothing mounted in
// this harness may spawn a real process or reach a real network endpoint.
// Blocking the call isn't enough on its own — several production call sites
// (essentiaClient.ts's run(), training runners) deliberately swallow a
// subprocess failure and degrade to a null/skip result, so a thrown guard
// error can get caught and hidden exactly like a real missing-binary error
// would. `violations` is the record that survives that: it is pushed to
// before the guard throws, so a test can assert it is empty at the end of a
// run even though production never saw anything but an ordinary error.
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

const SUBPROCESS_FNS = ['execFile', 'exec', 'spawn', 'execFileSync', 'spawnSync', 'fork'] as const;

/** requestCapture.ts's checkoutState() and workerUpdate.ts's startupCommit
 *  both shell out to `git rev-parse HEAD` / `git status --porcelain` purely
 *  to stamp a commit/dirty flag on a captured fixture or a training status
 *  read — read-only repo introspection, not a "runner, model download, LLM or
 *  remote worker" in the plan's sense, and harmless here since it runs
 *  against the isolated HOT_STEP_ROOT temp dir, which has no .git and
 *  nothing to read. Anything other than these two read-only subcommands
 *  still blocks — this is not a general 'git' allowlist. */
const ALLOWED_GIT_SUBCOMMANDS = ['rev-parse', 'status'];
function isAllowedGitCall(file: unknown, args: unknown): boolean {
  return file === 'git' && Array.isArray(args) && ALLOWED_GIT_SUBCOMMANDS.includes(args[0]);
}

/** Replaces every process-spawning entry point in node:child_process with one
 *  that records the attempt and throws. Nothing else this batch's six groups
 *  cover needs a real subprocess — ffmpeg, essentia, whisper and the training
 *  runners are all either unreached or must be fixtured instead. */
export function installSubprocessGuard(): void {
  for (const name of SUBPROCESS_FNS) {
    const real = (cp as unknown as Record<string, (...a: unknown[]) => unknown>)[name];
    (cp as unknown as Record<string, unknown>)[name] = (...args: unknown[]) => {
      if (isAllowedGitCall(args[0], args[1])) return real(...args);
      violations.push(`child_process.${name}(${args.map(a => (typeof a === 'string' ? a : typeof a)).slice(0, 2).join(', ')})`);
      throw new Error(`[safety-guard] child_process.${name}() blocked — fixture it, never exec a real subprocess here`);
    };
  }
}

/** Replaces global fetch with one that only forwards requests to this
 *  harness's own loopback origins (the fake server and fake engine); anything
 *  else — a real model download, remote worker or LLM endpoint — is recorded
 *  and refused. */
export function installNetworkGuard(allowedHosts: readonly string[]): void {
  const real = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const raw = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    let host = '';
    try { host = new URL(raw).hostname; } catch { /* relative URL: same-origin, allowed */ }
    if (host && !allowedHosts.includes(host)) {
      violations.push(`fetch(${raw})`);
      throw new Error(`[safety-guard] fetch to '${raw}' blocked — only ${allowedHosts.join(', ')} are allowed in this fixture`);
    }
    return real(input, init);
  }) as typeof fetch;
}
