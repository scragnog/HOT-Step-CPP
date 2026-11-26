// The page's web fonts (index.html's stylesheet) are all loaded as the page opens, not the first time a word needs
// one: a face that arrives later lays the whole document out again (a dropdown's first opening, the notes drawer's), and
// the desk measures its tiles again. Every face the stylesheet offers is loaded (each family and weight in each range of
// characters: a list's words reach beyond Latin); a face that cannot load (no network) is left to the fallback, as
// before. Drawing only.

function loadAll(): void {
  document.fonts?.forEach(f => { if (f.status === 'unloaded') f.load().catch(() => {}); });
}
/* the stylesheet blocks the first paint, so it has usually arrived by now; else once it has */
const link = document.querySelector<HTMLLinkElement>('link[rel="stylesheet"][href*="fonts"]');
if (!link || link.sheet) loadAll(); else link.addEventListener('load', loadAll, { once: true });
