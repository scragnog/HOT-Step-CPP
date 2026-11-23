// Remote artwork goes through the server cache (GET /api/artwork) so the page's CSP
// img-src can stay 'self'. Local paths, data: and blob: URLs pass through unchanged.
export function artworkSrc(url: string): string;
export function artworkSrc(url: string | undefined | null): string | undefined;
export function artworkSrc(url: string | undefined | null): string | undefined {
  if (!url) return undefined;
  return /^https?:\/\//i.test(url) ? `/api/artwork?url=${encodeURIComponent(url)}` : url;
}
