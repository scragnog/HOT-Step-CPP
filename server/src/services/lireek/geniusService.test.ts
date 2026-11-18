import test from 'node:test';
import assert from 'node:assert/strict';
import { config } from '../../config.js';
import { fetchLyrics, searchSongLyrics } from './geniusService.js';

const song = { id: 1, title: 'Fixture Song', url: 'https://genius.com/fixture-lyrics',
  primary_artist: { id: 2, name: 'Fixture Artist' }, album: { id: 3, name: 'Fixture Album' }, lyrics_state: 'complete' };
const hf = { rows: [{ row_idx: 17, row: { artist: 'Fixture Artist', title: song.title, lyrics: '[Verse]\nFixture words' } }] };

test('single-track and album fetches survive missing Genius pages and preserve the source', async (t) => {
  const token = config.lireek.geniusAccessToken;
  config.lireek.geniusAccessToken = 'fixture-token';
  t.after(() => { config.lireek.geniusAccessToken = token; });
  // A recorded Genius API envelope and page failure; payload text is fixture data.
  const calls: string[] = [];
  t.mock.method(globalThis, 'fetch', async (input: string) => {
    const url = String(input);
    calls.push(url);
    let data: unknown;
    if (url.includes('api.genius.com/search?')) data = { response: { hits: [{ result: song }] } };
    else if (url.includes('/songs/1')) data = { response: { song } };
    else if (url.includes('/albums/3/tracks')) data = { response: { tracks: [{ song }, { song: { ...song, title: 'Second Song' } }], next_page: null } };
    else if (url.startsWith('https://genius.com/')) return new Response('Unavailable', { status: 503 });
    else if (url.includes('huggingface.co/search')) data = new URL(url).searchParams.get('query') === `Fixture Artist ${song.title}` ? hf : { rows: [] };
    else if (url.includes('lrclib.net/api/get?')) data = { id: 42, artistName: 'Fixture Artist', trackName: 'Second Song', plainLyrics: 'Second fixture lyric', instrumental: false };
    else throw new Error(`Unexpected network request ${url}`);
    return new Response(JSON.stringify(data));
  });
  const single = await searchSongLyrics('Fixture Artist', song.title);
  assert.equal(single?.source, 'hf-genius');
  const album = await fetchLyrics('Fixture Artist', 'Fixture Album');
  assert.deepEqual(album.songs.map(row => row.source), ['hf-genius', 'lrclib']);
  assert.equal(album.songs[1].title, 'Second Song');
  assert.match(album.songs[1].url, /lrclib/);
  assert.ok(calls.some(url => url.includes('/albums/3/tracks')));
});

test('working Genius lyrics win without a fallback request', async (t) => {
  const token = config.lireek.geniusAccessToken;
  config.lireek.geniusAccessToken = 'fixture-token';
  t.after(() => { config.lireek.geniusAccessToken = token; });
  t.mock.method(globalThis, 'fetch', async (input: string) => {
    if (String(input).includes('api.genius.com/search?')) return Response.json({ response: { hits: [{ result: song }] } });
    assert.equal(String(input), song.url);
    return new Response('<div data-lyrics-container="true">[Verse]<br>Fixture words</div>');
  });
  assert.equal((await searchSongLyrics('Fixture Artist', song.title))?.source, 'genius');
});
