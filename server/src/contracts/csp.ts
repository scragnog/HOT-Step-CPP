// Content-Security-Policy for the UI document: the page (skins included) may only
// run scripts from, and connect to, this server. Images from any HTTPS host stay
// allowed (img-src below), a known leak channel the doc spells out. Node sends it with the production build (index.ts) and the
// Vite dev server sends the dev variant (ui/vite.config.ts), so both read this file,
// which is why it sits in contracts/. No imports. Policy and how to extend it:
// docs/dev/architecture.md, "Content-Security-Policy".

/** Browsers POST violation reports here; the server logs them. */
export const CSP_REPORT_PATH = '/api/csp-report';

/** false sends Content-Security-Policy-Report-Only: violations are reported, nothing is blocked. */
export const CSP_ENFORCE = true;

export const CSP_HEADER = CSP_ENFORCE ? 'Content-Security-Policy' : 'Content-Security-Policy-Report-Only';

// Inter (ui/src/index.css) and the /kit specimen fonts come from Google Fonts.
const GOOGLE_FONTS_CSS = 'https://fonts.googleapis.com';
const GOOGLE_FONTS_FILES = 'https://fonts.gstatic.com';

export function contentSecurityPolicy(dev = false): string {
  const directives: Record<string, string[]> = {
    'default-src': ["'self'"],
    // Dev: Vite injects the React Refresh preamble as an inline module script.
    'script-src': ["'self'", ...(dev ? ["'unsafe-inline'"] : [])],
    // Inline styles: React style props, Vite's dev CSS, library-injected <style> tags.
    'style-src': ["'self'", "'unsafe-inline'", GOOGLE_FONTS_CSS],
    'font-src': ["'self'", 'data:', GOOGLE_FONTS_FILES],
    // data: the favicon; blob: local previews. https: Lyric Studio artist and album art,
    // stored as outside URLs (Genius, or any URL the user pastes). Page code can leak data
    // in an image URL; outside shells wait until that is accepted or a server artwork
    // route lets this drop to 'self' (docs/dev/architecture.md).
    'img-src': ["'self'", 'data:', 'blob:', 'https:'],
    // blob: streamed previews (useStreamAudio) and abcjs MIDI playback.
    'media-src': ["'self'", 'data:', 'blob:'],
    // 'self' also matches same-origin ws:, which the Vite HMR socket uses in dev.
    'connect-src': ["'self'"],
    'worker-src': ["'self'", 'blob:'],
    'object-src': ["'none'"],
    'base-uri': ["'self'"],
    'form-action': ["'self'"],
    'report-uri': [CSP_REPORT_PATH],
  };
  return Object.entries(directives).map(([k, v]) => `${k} ${v.join(' ')}`).join('; ');
}
