// requestCapture.ts — dev-only recording of POST /api/generate request bodies.
//
// Off unless HOTSTEP_GENERATE_CAPTURE is set in the environment of a server
// started by dev.bat (HOT_STEP_DEV); no launcher sets the capture variable.
// It exists to collect "before" fixtures for the frontend-decoupling work: the exact body each caller (Create, Lyric Studio,
// ...) sends, recorded at the handler entry before the envelope is built or
// anything is normalized, so a later server-side resolver can be compared
// against it field for field.
//
//   HOTSTEP_GENERATE_CAPTURE=record        record, then generate as normal
//   HOTSTEP_GENERATE_CAPTURE=capture-only  record and return; nothing is queued
//
// The fixture is built from a deep copy. The request object the route goes on
// to use is never written to. Request headers are not stored (so no auth
// token), and body keys that look like credentials are replaced by a marker.

import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { PROJECT_ROOT, config } from '../../config.js';

export type GenerateCaptureMode = 'off' | 'record' | 'capture-only';

export const GENERATE_CAPTURE_SCHEMA = 'hotstep.generate-capture/1';
export const REDACTED = '[redacted]';

/** Honoured only under dev.bat, which sets HOT_STEP_DEV for the server; the
 *  end-user launchers never do, so the variable alone cannot turn it on. */
export function generateCaptureMode(env: NodeJS.ProcessEnv = process.env): GenerateCaptureMode {
  if (!env.HOT_STEP_DEV) return 'off';
  const v = (env.HOTSTEP_GENERATE_CAPTURE ?? '').trim().toLowerCase();
  return v === 'record' || v === 'capture-only' ? v : 'off';
}

/** A key that names a credential. `token` alone or as a suffix (hfToken,
 *  authToken) counts; `tokens` (maxTokens, lmMaxNewTokens) does not. */
const CREDENTIAL_KEY = /(api[-_]?key|secret|passw(or)?d|authori[sz]ation|cookie|credential|token$)/i;

/** Deep copy of `value` with credential-like keys replaced, plus the dotted
 *  paths that were replaced. Never mutates `value`. */
export function redactCredentials(value: unknown): { value: unknown; paths: string[] } {
  const paths: string[] = [];
  const walk = (v: unknown, at: string): unknown => {
    if (Array.isArray(v)) return v.map((item, i) => walk(item, `${at}[${i}]`));
    if (v && typeof v === 'object') {
      const out: Record<string, unknown> = {};
      for (const [k, child] of Object.entries(v as Record<string, unknown>)) {
        const p = at ? `${at}.${k}` : k;
        if (CREDENTIAL_KEY.test(k) && child !== undefined && child !== null && child !== '') {
          out[k] = REDACTED;
          paths.push(p);
        } else {
          out[k] = walk(child, p);
        }
      }
      return out;
    }
    return v;
  };
  return { value: walk(structuredClone(value), ''), paths };
}

export interface GenerateCaptureContext {
  mode: Exclude<GenerateCaptureMode, 'off'>;
  commit: string;
  dirty: boolean | null;
  activeBackendId: string;
  /** Optional `X-HotStep-Capture-Caller` header, for headless harnesses. */
  callerLabel?: string;
  /** Path of the page that sent the request (Referer, path only). */
  refererPath?: string;
  now?: Date;
  id?: string;
}

export interface GenerateCaptureFixture {
  schema: typeof GENERATE_CAPTURE_SCHEMA;
  id: string;
  capturedAt: string;
  mode: GenerateCaptureContext['mode'];
  route: 'POST /api/generate';
  commit: string;
  dirty: boolean | null;
  caller: { source: string | null; label: string | null; refererPath: string | null };
  /** Server-side state that changes how the body is resolved. Client settings
   *  (coResident, cacheLmCodes, ...) travel in the body itself. */
  settings: { activeBackendId: string; submittedBackend: string | null };
  seed: { seed: unknown; randomSeed: unknown; lmSeed: unknown };
  redactedPaths: string[];
  body: unknown;
}

export function buildGenerateCapture(body: unknown, ctx: GenerateCaptureContext): GenerateCaptureFixture {
  const { value, paths } = redactCredentials(body ?? null);
  const b = (value && typeof value === 'object' && !Array.isArray(value) ? value : {}) as Record<string, unknown>;
  const str = (v: unknown) => (typeof v === 'string' && v ? v : null);
  return {
    schema: GENERATE_CAPTURE_SCHEMA,
    id: ctx.id ?? randomUUID(),
    capturedAt: (ctx.now ?? new Date()).toISOString(),
    mode: ctx.mode,
    route: 'POST /api/generate',
    commit: ctx.commit,
    dirty: ctx.dirty,
    caller: { source: str(b.source), label: str(ctx.callerLabel), refererPath: str(ctx.refererPath) },
    settings: { activeBackendId: ctx.activeBackendId, submittedBackend: str(b.backend) },
    seed: { seed: b.seed, randomSeed: b.randomSeed, lmSeed: b.lmSeed },
    redactedPaths: paths,
    body: value,
  };
}

function git(args: string[]): string {
  return execFileSync('git', args, { cwd: PROJECT_ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
}

let commitCache: string | undefined;
function currentCommit(): string {
  if (commitCache === undefined) { try { commitCache = git(['rev-parse', 'HEAD']); } catch { commitCache = ''; } }
  return commitCache;
}
function checkoutDirty(): boolean | null {
  try { return git(['status', '--porcelain', '--untracked-files=no']).length > 0; } catch { return null; }
}

export function generateCaptureDir(): string {
  return path.join(config.data.dir, 'dev-captures', 'generate');
}

/** Write one fixture and return it. Throws on a write failure; the route
 *  reports that rather than silently dropping a fixture. */
export function recordGenerateRequest(
  body: unknown,
  ctx: Omit<GenerateCaptureContext, 'commit' | 'dirty'>,
): { fixture: GenerateCaptureFixture; file: string } {
  const fixture = buildGenerateCapture(body, { ...ctx, commit: currentCommit(), dirty: checkoutDirty() });
  const dir = generateCaptureDir();
  fs.mkdirSync(dir, { recursive: true });
  const caller = (fixture.caller.label ?? fixture.caller.source ?? 'unknown').replace(/[^A-Za-z0-9_-]/g, '_');
  const file = path.join(dir, `${fixture.capturedAt.replace(/[:.]/g, '-')}_${caller}_${fixture.id}.json`);
  fs.writeFileSync(file, `${JSON.stringify(fixture, null, 1)}\n`, { flag: 'wx' });
  return { fixture, file };
}
