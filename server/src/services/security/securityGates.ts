// Applies the request security gates (Host allowlist, Origin policy, CORS) and the CSP report route to an express app.
// Kept out of index.ts so the wiring can be tested without starting the whole server.
import express, { type Express } from 'express';
import cors from 'cors';
import { CSP_REPORT_PATH } from '../../contracts/csp.js';
import { buildAllowedHosts, isAllowedHost, isAllowedOrigin, originRejection, parseCorsOrigins } from './originPolicy.js';

export interface SecurityGateOptions {
  serverHost?: string;
  corsOrigins?: string;
  allowedHosts?: string;
}

export function applySecurityGates(app: Express, opts: SecurityGateOptions = {}): void {
  const corsOrigins = parseCorsOrigins(opts.corsOrigins);
  const allowedHosts = buildAllowedHosts(opts.serverHost, corsOrigins, opts.allowedHosts);
  // Host allowlist first: refuses DNS rebinding before anything else runs.
  app.use((req, res, next) => {
    if (!isAllowedHost(req.get('host'), allowedHosts)) { res.status(403).json({ error: 'Host not allowed' }); return; }
    next();
  });
  app.use((req, res, next) => {
    const status = originRejection(req.method, req.path, req.get('origin'), corsOrigins, req.get('host'));
    if (status) { res.status(status).json({ error: 'Origin not allowed' }); return; }
    next();
  });
  app.use(cors((req, cb) => {
    cb(null, { origin: isAllowedOrigin(req.headers.origin, corsOrigins, req.headers.host) });
  }));
  // CSP violation reports (csp.ts): browsers POST them as application/csp-report.
  app.post(CSP_REPORT_PATH, express.json({ type: ['application/csp-report', 'application/json'], limit: '64kb' }), (req, res) => {
    logCspReport(req.body);
    res.status(204).end();
  });
}

/** One console line per distinct violation (directive, blocked URI, source); capped so a loop can't flood the log. */
const seenCsp = new Set<string>();
export function logCspReport(body: unknown): string | null {
  const r = (body as { 'csp-report'?: Record<string, unknown> } | null)?.['csp-report'];
  if (!r) return null;
  const where = r['source-file'] ? ` at ${r['source-file']}:${r['line-number'] ?? '?'}` : '';
  const line = `[CSP] ${r['disposition'] ?? 'report'} ${r['effective-directive'] ?? r['violated-directive']} ${r['blocked-uri'] || '(inline)'} on ${r['document-uri']}${where}`;
  if (seenCsp.has(line) || seenCsp.size >= 500) return null;
  seenCsp.add(line);
  console.warn(line);
  return line;
}
