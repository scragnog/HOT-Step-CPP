import test from 'node:test';
import assert from 'node:assert/strict';
import { createLyricFallbacks } from './lyricFallbacks.js';

// Recorded API envelopes, with artist, title and lyric text replaced by fixtures.
const hf = { rows: [{ row_idx: 17, row: { artist: 'Fixture Artist', title: 'Fixture Song', lyrics: '[Verse]\nFixture words' } }] };
const lrc = { id: 42, artistName: 'Fixture Artist', trackName: 'Fixture Song', instrumental: false, plainLyrics: 'Fixture words', syncedLyrics: null };
function client(responses: Array<[number, unknown] | Error>) {
  const calls: string[] = [];
  const request = (async (url: string | URL | Request, init?: RequestInit) => {
    calls.push(String(url));
    assert.ok(init?.signal, 'every fallback request has a deadline');
    const next = responses.shift();
    assert.ok(next, `Unexpected request: ${url}`);
    if (next instanceof Error) throw next;
    return new Response(JSON.stringify(next[1]), { status: next[0] });
  }) as typeof fetch;
  return { lookup: createLyricFallbacks(request), calls };
}

test('HF supplies tagged lyrics and a row URL before LRCLIB is called', async () => {
  const { lookup, calls } = client([[200, hf]]);
  const hit = await lookup('Fixture Artist', 'Fixture Song');
  assert.equal(hit?.source, 'hf-genius');
  assert.equal(hit?.lyrics, '[Verse]\nFixture words');
  assert.match(hit!.url, /row=17$/);
  assert.equal(calls.length, 1);
  assert.equal(new URL(calls[0]).searchParams.get('query'), 'Fixture Artist Fixture Song');
});

test('HF misses and wrong artists fall through to plain LRCLIB lyrics', async () => {
  for (const response of [{ rows: [] }, { rows: [{ ...hf.rows[0], row: { ...hf.rows[0].row, artist: 'Another Artist' } }] }]) {
    const { lookup, calls } = client([[200, response], [200, lrc]]);
    assert.deepEqual(await lookup('Fixture Artist', 'Fixture Song'), {
      title: 'Fixture Song', lyrics: 'Fixture words', source: 'lrclib', url: 'https://lrclib.net/api/get/42',
    });
    assert.equal(calls.length, 2);
  }
});

test('HF timeout, malformed JSON, 429 and recorded 500 permit LRCLIB and cool down HF', async () => {
  for (const response of [new DOMException('Timed out', 'TimeoutError'), [429, {}],
    [500, { error: 'Authentication check on the Hugging Face Hub failed or timed out.' }], [200, null]] as Array<[number, unknown] | Error>) {
    const { lookup, calls } = client([response, [200, lrc], [200, lrc]]);
    assert.equal((await lookup('Fixture Artist', 'Fixture Song'))?.source, 'lrclib');
    assert.equal((await lookup('Fixture Artist', 'Fixture Song'))?.source, 'lrclib');
    assert.equal(calls.filter(url => url.includes('huggingface')).length, 1);
  }
});

test('HF recognizes deleted accented letters but still requires the requested title', async () => {
  const { lookup } = client([[200, hf]]);
  assert.equal((await lookup('Fixture Artïist', 'Fixture Söong'))?.source, 'hf-genius');
  const wrong = client([[200, { rows: [{ ...hf.rows[0], row: { ...hf.rows[0].row, title: 'Different Song' } }] }], [404, {}]]);
  assert.equal(await wrong.lookup('Fixture Artist', 'Fixture Song'), null);
});

test('LRCLIB rejects covers, wrong titles, instrumentals and empty lyrics', async () => {
  for (const row of [{ ...lrc, artistName: 'Cover Artist' }, { ...lrc, trackName: 'Different Song' },
    { ...lrc, instrumental: true }, { ...lrc, plainLyrics: '  ' }]) {
    const { lookup } = client([[200, { rows: [] }], [200, row]]);
    assert.equal(await lookup('Fixture Artist', 'Fixture Song'), null);
  }
});

test('both sources unavailable and empty inputs return no lyric', async () => {
  const { lookup, calls } = client([[503, {}], new Error('offline')]);
  assert.equal(await lookup('', 'Fixture Song'), null);
  assert.equal(calls.length, 0);
  assert.equal(await lookup('Fixture Artist', 'Fixture Song'), null);
});
