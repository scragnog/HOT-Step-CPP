// viewRoutes.ts — the URL <-> view-name mapping App uses for routing.

/** Derive top-level view from the browser URL */
export function viewFromUrl(path = window.location.pathname): string {
  if (path.startsWith('/insta-gen')) return 'insta-gen';
  if (path.startsWith('/lyric-studio')) return 'lyric-studio';
  if (path.startsWith('/cover-studio')) return 'cover-studio';
  if (path.startsWith('/stem-studio')) return 'stem-studio';
  if (path.startsWith('/stem-builder')) return 'stem-builder';
  if (path.startsWith('/song-builder')) return 'song-builder';
  if (path.startsWith('/storm')) return 'storm';
  if (path.startsWith('/midi-studio')) return 'midi-studio';
  if (path.startsWith('/training-studio')) return 'training-studio';
  if (path.startsWith('/compare')) return 'compare-studio';
  if (path.startsWith('/repaint')) return 'repaint';
  if (path.startsWith('/library')) return 'library';
  if (path.startsWith('/settings')) return 'settings';
  return 'create';
}

/** Map view names to URL paths */
export function urlForView(view: string): string {
  if (view === 'insta-gen') return '/insta-gen';
  if (view === 'lyric-studio') {
    // Restore the last deep URL (artist/album/tab) if we have one
    try {
      const saved = localStorage.getItem('hs-lastLyricStudioUrl');
      if (saved) return saved;
    } catch { /* ignore */ }
    return '/lyric-studio';
  }
  if (view === 'cover-studio') return '/cover-studio';
  if (view === 'stem-studio') return '/stem-studio';
  if (view === 'stem-builder') return '/stem-builder';
  if (view === 'song-builder') return '/song-builder';
  if (view === 'storm') return '/storm';
  if (view === 'midi-studio') return '/midi-studio';
  if (view === 'training-studio') {
    // Restore the last deep URL (dataset/phase) if we have one
    try {
      const saved = localStorage.getItem('hs-lastTrainingStudioUrl');
      if (saved) return saved;
    } catch { /* ignore */ }
    return '/training-studio';
  }
  if (view === 'compare-studio') return '/compare';
  if (view === 'repaint') return '/repaint';
  if (view === 'library') return '/library';
  if (view === 'settings') return '/settings';
  return '/';
}

