// Raw responses preserve the shell's offline, download and fallback handling.
export function probeServerHealth(): Promise<Response> {
  return fetch('/api/health', { signal: AbortSignal.timeout(2000) });
}

export function revealDownload(path?: string): Promise<Response> {
  return fetch('/api/download/reveal', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }),
  });
}

export function exportSongsZip(ids: string[], token?: string | null): Promise<Response> {
  return fetch('/api/songs/export-zip', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: JSON.stringify({ ids }),
  });
}

export function saveDownload(url: string): Promise<Response> {
  const [pathname, query] = url.split('?');
  return fetch(`${pathname}/save${query ? `?${query}` : ''}`, { method: 'POST' });
}
