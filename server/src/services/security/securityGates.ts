// Applies the request security gates (Host allowlist, Origin policy, CORS) to an express app.
// Kept out of index.ts so the wiring can be tested without starting the whole server.
import type { Express } from 'express';
import cors from 'cors';
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
}
