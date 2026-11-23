// Serves the built UI (ui/dist) with its CSP and cache headers, plus the SPA fallback.
// Kept out of index.ts so the production header path can be tested on a real server.
import path from 'path';
import express, { type Express } from 'express';
import { CSP_HEADER, contentSecurityPolicy } from '../contracts/csp.js';

const NO_CACHE = 'no-cache, no-store, must-revalidate';

export function serveUiDist(app: Express, uiDistPath: string): void {
  // The UI document and its service worker carry the CSP; API, audio and /listening do not.
  const csp = contentSecurityPolicy();
  app.use((_req, res, next) => { res.setHeader(CSP_HEADER, csp); next(); });
  // Assets with content hashes get long cache; index.html always revalidates
  app.use(express.static(uiDistPath, {
    setHeaders: (res, filePath) => {
      if (filePath.endsWith('.webmanifest')) res.setHeader('Content-Type', 'application/manifest+json');
      // index.html, the service worker and the manifest must revalidate so updates reach installed apps.
      if (filePath.endsWith('.html') || filePath.endsWith('sw.js') || filePath.endsWith('.webmanifest')) {
        res.setHeader('Cache-Control', NO_CACHE);
      } else {
        res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
      }
    },
  }));
  // SPA fallback: serve index.html for all unmatched routes
  app.get('/{*splat}', (_req, res) => {
    res.setHeader('Cache-Control', NO_CACHE);
    res.sendFile(path.join(uiDistPath, 'index.html'));
  });
}
