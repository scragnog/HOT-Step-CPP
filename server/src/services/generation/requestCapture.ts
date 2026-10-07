// requestCapture.ts — dev-only recording of POST /api/generate request bodies.
//
// Off unless HOTSTEP_GENERATE_CAPTURE is set in the environment of a server
// started by dev.bat (HOT_STEP_DEV); no launcher sets the capture variable.
// It exists to collect "before" fixtures for the frontend-decoupling work: the
// exact body each caller (Create, Lyric Studio, ...) sends, recorded before the
// envelope is built or anything is normalized, so a later server-side resolver
// can be compared against it field for field.
//
//   HOTSTEP_GENERATE_CAPTURE=record        record, then generate as normal
//   HOTSTEP_GENERATE_CAPTURE=capture-only  record and return; nothing is queued
//
// It runs as middleware ahead of the generate handler and needs the same
// bearer token the handler does, so an unauthenticated caller writes nothing.
// The fixture is built from a deep copy; the request is never written to.
// No request header is stored as text. Body keys that name a credential are
// replaced by a marker; free-text values (lyrics, captions) are stored as sent.

import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import type { Request, RequestHandler } from 'express';
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

/** Key names that hold a credential, compared after dropping `-` and `_` and
 *  lowercasing, so apiKey, api_key and API-KEY all match. `token` counts as a
 *  whole word or suffix (hfToken, sessionToken); `tokens` (maxTokens) does not.
 *  Matching is by key name only: a secret pasted into a free-text field such
 *  as the caption cannot be detected and is stored as sent. */
const CREDENTIAL_PARTS = [
  'apikey', 'secret', 'password', 'passwd', 'passphrase', 'authorization', 'cookie',
  'credential', 'privatekey', 'accesskey', 'signingkey', 'clientkey', 'bearer',
];
export function isCredentialKey(key: string): boolean {
  const k = key.toLowerCase().replace(/[-_]/g, '');
  return k.endsWith('token') || CREDENTIAL_PARTS.some(part => k.includes(part));
}

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
        if (isCredentialKey(k) && child !== undefined && child !== null && child !== '') {
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

/** A caller label is kept only when it is a short identifier ("create",
 *  "lyric-studio", "harness-2"); anything else is dropped, never stored. */
const LABEL = /^[A-Za-z0-9._-]{1,64}$/;
export function boundedLabel(v: unknown): string | null {
  return typeof v === 'string' && LABEL.test(v) ? v : null;
}

/** The Referer's path, kept only when it is a short plain path ("/create").
 *  Host, query and fragment are never stored. */
const PATH = /^\/[A-Za-z0-9._~/-]{0,127}$/;
export function boundedRefererPath(referer: string | undefined): string | null {
  if (!referer) return null;
  try {
    const p = new URL(referer).pathname;
    return PATH.test(p) ? p : null;
  } catch { return null; }
}

export interface GenerateCaptureContext {
  mode: Exclude<GenerateCaptureMode, 'off'>;
  commit: string;
  dirty: boolean | null;
  activeBackendId: string;
  /** `X-HotStep-Capture-Caller` header, for headless harnesses. */
  callerLabel?: string;
  /** Referer header; only a bounded path survives. */
  referer?: string;
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
  return {
    schema: GENERATE_CAPTURE_SCHEMA,
    id: ctx.id ?? randomUUID(),
    capturedAt: (ctx.now ?? new Date()).toISOString(),
    mode: ctx.mode,
    route: 'POST /api/generate',
    commit: ctx.commit,
    dirty: ctx.dirty,
    caller: { source: boundedLabel(b.source), label: boundedLabel(ctx.callerLabel), refererPath: boundedRefererPath(ctx.referer) },
    settings: { activeBackendId: ctx.activeBackendId, submittedBackend: boundedLabel(b.backend) },
    seed: { seed: b.seed, randomSeed: b.randomSeed, lmSeed: b.lmSeed },
    redactedPaths: paths,
    body: value,
  };
}

/** HEAD and dirty state, read fresh for every capture: a commit made while the
 *  server runs (no source save, so no restart) must not be misattributed. */
export function checkoutState(cwd: string = PROJECT_ROOT): { commit: string; dirty: boolean | null } {
  const git = (args: string[]) =>
    execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], windowsHide: true }).trim();
  let commit = '';
  let dirty: boolean | null = null;
  try { commit = git(['rev-parse', 'HEAD']); } catch { /* not a checkout */ }
  try { dirty = git(['status', '--porcelain', '--untracked-files=no']).length > 0; } catch { /* unknown */ }
  return { commit, dirty };
}

/** Write one fixture into `dir` and return it. Throws on a write failure. */
export function writeGenerateCapture(dir: string, fixture: GenerateCaptureFixture): string {
  fs.mkdirSync(dir, { recursive: true });
  const caller = fixture.caller.label ?? fixture.caller.source ?? 'unknown';
  const file = path.join(dir, `${fixture.capturedAt.replace(/[:.]/g, '-')}_${caller}_${fixture.id}.json`);
  fs.writeFileSync(file, `${JSON.stringify(fixture, null, 1)}\n`, { flag: 'wx' });
  return file;
}

export interface GenerateCaptureDeps {
  mode: () => GenerateCaptureMode;
  /** The route's own token check (routes/auth.ts getUserId). */
  userId: (req: Request) => string | null;
  activeBackendId: () => string;
  dir: () => string;
  checkout: () => { commit: string; dirty: boolean | null };
}

/** Middleware for POST /api/generate. Off: passes straight through. On: an
 *  unauthenticated request gets the same 401 the handler gives and nothing is
 *  written; an authenticated one is recorded, then either continues (record)
 *  or returns without creating a job (capture-only). Engine readiness is not
 *  checked, so capture-only works while the engine is down. */
export function createGenerateCapture(deps: GenerateCaptureDeps): RequestHandler {
  return (req, res, next) => {
    const mode = deps.mode();
    if (mode === 'off') { next(); return; }
    if (!deps.userId(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
    let fixture: GenerateCaptureFixture;
    try {
      fixture = buildGenerateCapture(req.body, {
        mode,
        ...deps.checkout(),
        activeBackendId: deps.activeBackendId(),
        callerLabel: req.get('x-hotstep-capture-caller'),
        referer: req.get('referer'),
      });
      writeGenerateCapture(deps.dir(), fixture);
    } catch (err) {
      res.status(500).json({ error: `Request capture failed: ${err instanceof Error ? err.message : String(err)}` });
      return;
    }
    if (mode === 'capture-only') {
      res.json({ jobId: null, status: 'captured', captureId: fixture.id });
      return;
    }
    next();
  };
}

export function generateCaptureDir(): string {
  return path.join(config.data.dir, 'dev-captures', 'generate');
}
