// geniusService.ts — Lyrics acquisition via the Genius API
//
// Port of Python genius_service.py.
// Uses fetch() for the Genius REST API and cheerio for HTML parsing.

import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import * as cheerio from 'cheerio';
import { config } from '../../config.js';

const execFileP = promisify(execFile);

const API_ROOT = 'https://api.genius.com';

const BROWSER_HEADERS: Record<string, string> = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
  'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
  'Accept-Language': 'en-US,en;q=0.9',
};

export interface SongLyrics {
  title: string;
  album: string | null;
  lyrics: string;
}

export interface LyricsSearchResponse {
  artist: string;
  album: string | null;
  songs: SongLyrics[];
  total_songs: number;
}


// ── Helpers ─────────────────────────────────────────────────────────────────

function getAuthHeaders(): Record<string, string> {
  const token = config.lireek.geniusAccessToken;
  if (!token) {
    throw new Error('GENIUS_ACCESS_TOKEN is not set. Please add it to your .env file.');
  }
  return { Authorization: `Bearer ${token}` };
}

function cleanLyrics(raw: string): string {
  if (!raw) return '';
  // Remove contributor/title header (captures up to first section header)
  let text = raw.replace(/^\d+\s*Contributors?.*?Lyrics.*?(?=\[)/is, '');
  // Simpler header strip if no section header found
  text = text.replace(/^\d+\s*Contributors?.*?Lyrics\s*\n?/is, '');
  // Remove 'You might also like'
  text = text.replace(/You might also like\s*/g, '');
  // Remove trailing 'Embed'
  text = text.replace(/\d*Embed$/, '').trim();
  // Collapse 3+ consecutive blank lines into 2
  text = text.replace(/\n{3,}/g, '\n\n');
  return text.trim();
}

/** genius.com pages are behind a Cloudflare bot check that Node can't pass. */
export class GeniusBlockedError extends Error {
  constructor() {
    super('Genius is blocking lyric page downloads (Cloudflare bot check). The API key works; '
      + 'the lyrics pages are refused. Make sure curl is installed and on PATH.');
  }
}

type Page = { ok: boolean; status: number; text: string; blocked: boolean };

const isChallenge = (status: number, text: string) => status === 403 && text.includes('challenge-platform');

/** Set once Node's own fetch has been challenged, so later pages go straight to curl. */
let nodeChallenged = false;

/** GET a page with the system curl. Null when curl is missing or fails. */
async function curlPage(url: string): Promise<Page | null> {
  const args = ['-sSL', '--max-time', '30', '-w', '\n%{http_code}'];
  for (const [k, v] of Object.entries(BROWSER_HEADERS)) args.push('-H', `${k}: ${v}`);
  try {
    const { stdout } = await execFileP('curl', [...args, url], { maxBuffer: 32 << 20, windowsHide: true });
    const cut = stdout.lastIndexOf('\n');
    const status = Number(stdout.slice(cut + 1));
    const text = stdout.slice(0, cut);
    return { ok: status >= 200 && status < 300, status, text, blocked: isChallenge(status, text) };
  } catch (err) {
    console.warn(`[Genius] curl fallback failed for ${url}:`, (err as Error).message);
    return null;
  }
}

/**
 * GET a genius.com HTML page (lyrics, album, artist).
 *
 * Since 2026-11 Cloudflare answers Node's TLS client on genius.com with a 403
 * JS challenge whatever the headers; api.genius.com is unaffected. Windows
 * curl (Schannel) still gets the page with a browser User-Agent, so a
 * challenged page is retried through the system curl. An OpenSSL curl on Linux
 * was still challenged when this was written, so `blocked` can stay true.
 */
async function fetchPage(url: string): Promise<Page> {
  if (!nodeChallenged) {
    const resp = await fetch(url, { headers: BROWSER_HEADERS, redirect: 'follow' });
    const text = await resp.text();
    if (!isChallenge(resp.status, text)) return { ok: resp.ok, status: resp.status, text, blocked: false };
    nodeChallenged = true;
    console.warn('[Genius] Node fetch is being challenged by Cloudflare; using curl for genius.com pages');
  }
  return await curlPage(url) ?? { ok: false, status: 403, text: '', blocked: true };
}

async function scrapeLyrics(songUrl: string): Promise<string | null> {
  try {
    const resp = await fetchPage(songUrl);
    if (resp.blocked) throw new GeniusBlockedError();
    if (!resp.ok) {
      console.warn(`[Genius] Failed to scrape ${songUrl}: ${resp.status}`);
      return null;
    }

    const $ = cheerio.load(resp.text.replace(/<br\/?>/gi, '\n'));

    // Modern Genius layout: data-lyrics-container divs
    const containers = $('div[data-lyrics-container="true"]');
    if (containers.length) {
      const parts: string[] = [];
      containers.each((_, el) => { parts.push($(el).text()); });
      return parts.join('\n');
    }

    // Fallback: root lyrics div
    const root = $('div.lyrics, div[class*="Lyrics__Root"]');
    if (root.length) return root.text();

    console.warn(`[Genius] No lyrics div found on: ${songUrl}`);
    return null;
  } catch (err) {
    if (err instanceof GeniusBlockedError) throw err;
    console.warn(`[Genius] Scrape error for ${songUrl}:`, err);
    return null;
  }
}

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));


// ── Authenticated Genius API calls ──────────────────────────────────────────

/** Statuses worth trying again — rate limit, timeout, transient upstream. */
const TRANSIENT_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);

/**
 * fetch() with a short backoff on transient failures.
 *
 * There was no retry anywhere in this module, so a single 429 or dropped socket
 * dropped that track's lyrics permanently — the sidecar is simply written
 * without them and nothing ever revisits it. A bulk label run over ~180 albums
 * left ~22 single-track holes whose queries match perfectly on a later attempt,
 * which is exactly the signature of an unretried blip (2026-08-05).
 *
 * Honours Retry-After when the server sends one.
 */
async function apiFetch(url: string, headers: Record<string, string>, attempts = 3): Promise<Response> {
  let lastErr: unknown = null;
  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      const resp = await fetch(url, { headers });
      if (resp.ok || !TRANSIENT_STATUS.has(resp.status) || attempt === attempts - 1) return resp;
      const retryAfter = Number(resp.headers.get('retry-after'));
      await sleep(Number.isFinite(retryAfter) && retryAfter > 0
        ? Math.min(retryAfter * 1000, 15_000)
        : 800 * 2 ** attempt);
    } catch (err) {
      lastErr = err;
      if (attempt === attempts - 1) throw err;
      await sleep(800 * 2 ** attempt);
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error('Genius request failed');
}

async function apiSearch(query: string, perPage = 20): Promise<any[]> {
  const headers = getAuthHeaders();
  const url = `${API_ROOT}/search?q=${encodeURIComponent(query)}&per_page=${perPage}`;
  const resp = await apiFetch(url, headers);
  if (!resp.ok) throw new Error(`Genius search failed: ${resp.status}`);
  const data = await resp.json() as any;
  return data.response.hits;
}

async function apiGetArtistSongs(artistId: number, perPage = 20, sort = 'popularity'): Promise<any[]> {
  const headers = getAuthHeaders();
  const url = `${API_ROOT}/artists/${artistId}/songs?per_page=${perPage}&page=1&sort=${sort}`;
  const resp = await fetch(url, { headers });
  if (!resp.ok) throw new Error(`Genius artist songs failed: ${resp.status}`);
  const data = await resp.json() as any;
  return data.response.songs;
}

async function apiGetAlbumTracks(albumId: number): Promise<any[]> {
  const headers = getAuthHeaders();
  const tracks: any[] = [];
  let page: number | null = 1;

  while (page) {
    const url = `${API_ROOT}/albums/${albumId}/tracks?per_page=50&page=${page}`;
    const resp = await fetch(url, { headers });
    if (!resp.ok) throw new Error(`Genius album tracks failed: ${resp.status}`);
    const data = await resp.json() as any;
    tracks.push(...data.response.tracks);
    page = data.response.next_page ?? null;
  }
  return tracks;
}

async function apiGetArtistDetails(artistId: number): Promise<any> {
  const headers = getAuthHeaders();
  const resp = await fetch(`${API_ROOT}/artists/${artistId}`, { headers });
  if (!resp.ok) throw new Error(`Genius artist details failed: ${resp.status}`);
  const data = await resp.json() as any;
  return data.response.artist;
}

async function getArtistIdFromUrl(url: string): Promise<number | null> {
  try {
    const resp = await fetchPage(url);
    if (!resp.ok) return null;
    const html = resp.text;
    const match = html.match(/\\?"artist_id\\?":\s*(\d+)/) ?? html.match(/content="genius:\/\/artists\/(\d+)"/);
    return match ? parseInt(match[1], 10) : null;
  } catch { return null; }
}

async function findArtistId(artistName: string): Promise<number | null> {
  const hits = await apiSearch(artistName, 5);
  const nameLower = artistName.toLowerCase();
  for (const hit of hits) {
    const primary = hit.result?.primary_artist;
    if (primary?.name?.toLowerCase() === nameLower) return primary.id;
  }
  return hits[0]?.result?.primary_artist?.id ?? null;
}

// Patterns indicating bonus/non-original tracks
const BONUS_PATTERN = /\([^)]*(?:Demo|Live|Outtake|Cassette|Remix|Acoustic|Remaster|Session)[^)]*\)/i;
function isBonusTrack(title: string): boolean {
  return BONUS_PATTERN.test(title);
}

function slugifyForGenius(name: string): string {
  return name.replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '-');
}

async function findAlbumIdByPage(albumName: string, artistName: string): Promise<number | null> {
  const slug = `${slugifyForGenius(artistName)}/${slugifyForGenius(albumName)}`;
  const url = `https://genius.com/albums/${slug}`;
  const artistLower = artistName.toLowerCase();

  try {
    const resp = await fetchPage(url);
    if (!resp.ok) return null;
    const $ = cheerio.load(resp.text);
    const links = $('div.chart_row a.u-display_block').toArray();
    if (!links.length) return null;

    for (const link of links.slice(0, 3)) {
      const h3 = $(link).find('h3');
      let titleText = (h3.length ? h3.text() : $(link).text()).trim();
      titleText = titleText.replace(/\s*Lyrics$/, '').trim();
      if (!titleText) continue;

      const hits = await apiSearch(`${titleText} ${artistName}`, 5);
      for (const hit of hits) {
        const result = hit.result ?? {};
        const songId = result.id;
        if (!songId) continue;
        if (result.primary_artist?.name?.toLowerCase() !== artistLower) continue;

        const headers = getAuthHeaders();
        const sResp = await fetch(`${API_ROOT}/songs/${songId}`, { headers });
        if (!sResp.ok) continue;
        const sData = await sResp.json() as any;
        const album = sData.response.song.album;
        if (album && albumName.toLowerCase().includes(album.name?.toLowerCase())) {
          return album.id;
        }
      }
    }
  } catch {}
  return null;
}

async function findAlbumId(albumName: string, artistName: string): Promise<number | null> {
  const headers = getAuthHeaders();
  const query = `${albumName} ${artistName}`;
  const hits = await apiSearch(query, 10);
  const albumLower = albumName.toLowerCase();
  const artistLower = artistName.toLowerCase();

  // Sort: artist matches first
  const sorted = [...hits].sort((a, b) => {
    const aMatch = a.result?.primary_artist?.name?.toLowerCase() === artistLower ? 1 : 0;
    const bMatch = b.result?.primary_artist?.name?.toLowerCase() === artistLower ? 1 : 0;
    return bMatch - aMatch;
  });

  const candidates: { id: number; name: string }[] = [];
  const seenIds = new Set<number>();

  for (const hit of sorted) {
    const songId = hit.result?.id;
    if (!songId || candidates.length >= 3) break;

    try {
      const resp = await fetch(`${API_ROOT}/songs/${songId}`, { headers });
      if (!resp.ok) continue;
      const data = await resp.json() as any;
      const album = data.response.song.album;
      if (album && album.name?.toLowerCase().includes(albumLower)) {
        if (!seenIds.has(album.id)) {
          seenIds.add(album.id);
          candidates.push({ id: album.id, name: album.name });
        }
      }
    } catch {}
  }

  if (candidates.length) {
    // Prefer exact name match, then shortest name (less likely deluxe/expanded)
    const exact = candidates.find(c => c.name.toLowerCase() === albumLower);
    if (exact) return exact.id;
    candidates.sort((a, b) => a.name.length - b.name.length);
    return candidates[0].id;
  }

  // Fallback: direct page scrape
  return findAlbumIdByPage(albumName, artistName);
}

async function scrapeAlbumPageTracks(
  albumName = '', artistName = '', url?: string,
): Promise<{ title: string; url: string }[]> {
  if (!url) {
    const slug = `${slugifyForGenius(artistName)}/${slugifyForGenius(albumName)}`;
    url = `https://genius.com/albums/${slug}`;
  }
  try {
    const resp = await fetchPage(url);
    if (resp.blocked) throw new GeniusBlockedError();
    if (!resp.ok) return [];
    const $ = cheerio.load(resp.text);

    const tracks: { title: string; url: string }[] = [];
    $('div.chart_row a.u-display_block').each((_, el) => {
      let href = $(el).attr('href') ?? '';
      if (!href) return;
      if (!href.startsWith('http')) href = `https://genius.com${href}`;
      const h3 = $(el).find('h3');
      let title = (h3.length ? h3.text() : $(el).text()).trim();
      title = title.replace(/\s*Lyrics$/, '').trim();
      if (title) tracks.push({ title, url: href });
    });
    return tracks;
  } catch (err) {
    if (err instanceof GeniusBlockedError) throw err;
    return [];
  }
}


// ── Public API ──────────────────────────────────────────────────────────────

export async function fetchLyrics(
  artistName: string,
  albumName?: string | null,
  maxSongs = 10,
): Promise<LyricsSearchResponse> {
  const scrapeDelay = 300;
  const songs: SongLyrics[] = [];
  let artistId: number | null = null;

  // Handle Genius artist URL input
  if (artistName.includes('genius.com/artists/')) {
    const urlMatch = artistName.match(/(https?:\/\/(?:www\.)?genius\.com\/artists\/[^\s]+)/);
    const url = urlMatch ? urlMatch[1] : artistName.trim();
    artistId = await getArtistIdFromUrl(url);
    if (artistId) {
      try {
        const details = await apiGetArtistDetails(artistId);
        if (details.name) artistName = details.name;
      } catch {}
    }
  }

  // Handle album URL input
  let albumUrl: string | undefined;
  if (albumName && albumName.includes('genius.com/albums/')) {
    const urlMatch = albumName.match(/(https?:\/\/(?:www\.)?genius\.com\/albums\/[^\s]+)/);
    albumUrl = urlMatch ? urlMatch[1] : albumName.trim();
    const parts = albumUrl.replace(/\/$/, '').split('/');
    if (parts.length >= 2) {
      albumName = parts[parts.length - 1].replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      if (!artistId) {
        const artistSlug = parts[parts.length - 2];
        artistName = artistSlug.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      }
    }
  }

  if (albumName) {
    console.log(`[Genius] Fetching album '${albumName}' by '${artistName}'`);

    let albumIdResolved: number | null = null;
    if (!albumUrl) {
      albumIdResolved = await findAlbumId(albumName, artistName);
    }

    if (!albumUrl && albumIdResolved != null) {
      // API-based album track fetch
      const tracks = await apiGetAlbumTracks(albumIdResolved);
      const seenTitles = new Set<string>();

      for (const track of tracks) {
        const songInfo = track.song ?? {};
        const url = songInfo.url;
        const title = songInfo.title ?? 'Unknown';
        if (!url || songInfo.lyrics_state !== 'complete' || songInfo.instrumental) continue;
        if (isBonusTrack(title)) continue;
        const baseTitle = title.replace(/\s*\(.*\)/, '').trim().toLowerCase();
        if (seenTitles.has(baseTitle)) continue;
        seenTitles.add(baseTitle);

        try {
          await sleep(scrapeDelay);
          const raw = await scrapeLyrics(url);
          if (raw) songs.push({ title, album: albumName, lyrics: cleanLyrics(raw) });
        } catch (err) {
          if (err instanceof GeniusBlockedError) throw err;
          console.warn(`[Genius] Failed to scrape '${title}':`, err);
        }
      }
    } else {
      // Fallback: scrape tracks from album page
      const pageTracks = await scrapeAlbumPageTracks(albumName, artistName, albumUrl);
      for (const pt of pageTracks) {
        if (isBonusTrack(pt.title)) continue;
        try {
          await sleep(scrapeDelay);
          const raw = await scrapeLyrics(pt.url);
          if (raw) songs.push({ title: pt.title, album: albumName, lyrics: cleanLyrics(raw) });
        } catch (err) {
          if (err instanceof GeniusBlockedError) throw err;
          console.warn(`[Genius] Failed to scrape '${pt.title}':`, err);
        }
      }
    }
  } else {
    // General artist search (no specific album)
    console.log(`[Genius] Fetching up to ${maxSongs} songs by '${artistName}'`);
    if (!artistId) artistId = await findArtistId(artistName);
    if (!artistId) {
      throw new Error(`Could not find artist '${artistName}' on Genius.`);
    }

    const apiSongs = await apiGetArtistSongs(artistId, maxSongs);
    for (const songInfo of apiSongs.slice(0, maxSongs)) {
      const url = songInfo.url;
      const title = songInfo.title ?? 'Unknown';
      const songAlbum = songInfo.album?.name ?? null;
      if (!url) continue;

      try {
        await sleep(scrapeDelay);
        const raw = await scrapeLyrics(url);
        if (raw) songs.push({ title, album: songAlbum, lyrics: cleanLyrics(raw) });
      } catch (err) {
        if (err instanceof GeniusBlockedError) throw err;
        console.warn(`[Genius] Failed to scrape '${title}':`, err);
      }
    }
  }

  if (!songs.length) {
    throw new Error(
      `No lyrics found for '${artistName}'` +
      (albumName ? ` – album '${albumName}'` : '') +
      '. Please check the spelling and try again.',
    );
  }

  return { artist: artistName, album: albumName ?? null, songs, total_songs: songs.length };
}

/**
 * Search for a single song's lyrics on Genius.
 */
/** Lowercase, strip parentheticals/brackets and punctuation, collapse spaces. */
function normalizeTitle(s: string): string {
  return s.toLowerCase()
    .replace(/\([^)]*\)|\[[^\]]*\]/g, ' ')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Artist names compared the way titles already are.
 *
 * Titles went through normalizeTitle() while ARTISTS were compared as raw
 * lowercased strings, and that asymmetry silently threw away correct matches:
 *
 *   tag "Guns N' Roses"          vs Genius "Guns N’ Roses"          (U+2019)
 *   tag "KC & The Sunshine Band" vs Genius "KC and the Sunshine Band"
 *
 * In both cases the RIGHT song was the top hit with the title already agreeing,
 * and the artist gate — exact and relaxed, since `includes()` fails just as hard
 * when one character differs — rejected it. That cost gnr_appetite 12/12 and
 * kcsunshineband 9/9 in the bulk run (verified against the live API 2026-08-05).
 *
 * '&' becomes 'and' BEFORE punctuation is stripped, so both spellings converge.
 *
 * Diacritics are folded too (NFD + combining-mark strip), because rippers write
 * the ASCII spelling and Genius carries the real one. Same failure, same cost:
 *
 *   tag "Motorhead"    vs Genius "Motörhead"
 *   tag "Touche Amore" vs Genius "Touché Amoré"
 *
 * Both albums lost every track in the bulk run (motorhead_aceofspades 12/12,
 * toucheamore_stagefour 12/12) with the correct song sitting at hit #1 and its
 * title already agreeing — verified against the live API 2026-08-19.
 */
function normalizeArtist(s: string): string {
  return s.toLowerCase()
    .normalize('NFD')
    .replace(/\p{M}/gu, '')   // ö -> o, é -> e — rip tags are ASCII, Genius is not
    .replace(/[‘’ʼ′]/g, "'")   // curly/modifier apostrophes
    .replace(/[“”]/g, '"')
    .replace(/\s*&\s*/g, ' and ')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Relaxed artist agreement: every word of the shorter name appears in the longer.
 *
 * `includes()` is a SUBSTRING test, so it only forgives text bolted onto an end.
 * Genius routinely expands a credited duo into both full names, which interleaves
 * new words instead:
 *
 *   tag "Hall & Oates" -> "hall and oates"
 *   Genius            -> "daryl hall and john oates"
 *
 * "hall and oates" is not a substring of that (a "john" sits in the middle), so
 * the relaxed gate rejected it and hallandoates_privateeyes lost 11/11 with the
 * right song at hit #1. A token-subset test accepts it and still refuses the
 * covers that share a title — "royal blood" has no word in common with
 * "motorhead" — and the caller only reaches relaxed mode after three exact
 * attempts have failed, with title agreement still required alongside it.
 *
 * Both names must carry at least two tokens: a one-word name is a subset of far
 * too much ("Yes", "Bush", "Cream", "Live"), and the title gate alone is not
 * enough to hold that back.
 */
function artistTokensAgree(a: string, b: string): boolean {
  const at = a.split(' ').filter(Boolean);
  const bt = b.split(' ').filter(Boolean);
  if (at.length < 2 || bt.length < 2) return false;
  const [short, long] = at.length <= bt.length ? [at, bt] : [bt, at];
  const set = new Set(long);
  return short.every(t => set.has(t));
}

export async function searchSongLyrics(
  artist: string, title: string,
  opts?: { relaxed?: boolean },
): Promise<{ title: string; lyrics: string; url: string } | null> {
  const hits = await apiSearch(`${title} ${artist}`, 5);
  const artistNorm = normalizeArtist(artist);
  const titleNorm = normalizeTitle(title);

  for (const hit of hits) {
    const result = hit.result ?? {};
    const primaryNorm = normalizeArtist(result.primary_artist?.name ?? '');
    if (opts?.relaxed) {
      // Collabs and feats break exact matching ("Electric Callboy & BABYMETAL",
      // "Artist feat. X") — accept containment either way, but then require the
      // titles to agree so a popular unrelated song can't slip in.
      const artistOk = !!primaryNorm
        && (primaryNorm.includes(artistNorm) || artistNorm.includes(primaryNorm)
          || artistTokensAgree(primaryNorm, artistNorm));
      if (!artistOk) continue;
      const resultNorm = normalizeTitle(result.title ?? '');
      const titleOk = !!titleNorm && !!resultNorm
        && (resultNorm.includes(titleNorm) || titleNorm.includes(resultNorm));
      if (!titleOk) continue;
    } else if (primaryNorm !== artistNorm) {
      continue;
    }
    const songUrl = result.url;
    if (!songUrl) continue;

    const raw = await scrapeLyrics(songUrl);
    if (raw) {
      return { title: result.title ?? title, lyrics: cleanLyrics(raw), url: songUrl };
    }
  }
  return null;
}

/**
 * Refresh an artist's image URL from Genius.
 * Returns the image URL or null if not found.
 */
export async function getArtistImageUrl(artistName: string): Promise<string | null> {
  try {
    const id = await findArtistId(artistName);
    if (!id) return null;
    const details = await apiGetArtistDetails(id);
    return details.image_url ?? null;
  } catch { return null; }
}

/**
 * Get an album cover image URL from Genius.
 */
export async function getAlbumImageUrl(albumName: string, artistName: string): Promise<string | null> {
  try {
    const albumId = await findAlbumId(albumName, artistName);
    if (!albumId) return null;
    const headers = getAuthHeaders();
    const resp = await fetch(`${API_ROOT}/albums/${albumId}`, { headers });
    if (!resp.ok) return null;
    const data = await resp.json() as any;
    return data.response.album?.cover_art_url ?? null;
  } catch { return null; }
}
