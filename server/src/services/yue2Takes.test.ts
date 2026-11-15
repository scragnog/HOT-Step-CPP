import assert from 'node:assert/strict';
import test from 'node:test';
import { fillStarts, lyricsThrough, sectionStarts, YUE2_FPS } from './yue2Takes.js';
import type { Yue2AlignWord } from './backends/yue2/align.js';
import { composeArrangement } from './backends/yue2/coverArrangement.js';
import fs from 'node:fs';
import path from 'node:path';
import { config } from '../config.js';
import { createBuildSession, generateBuildTakes, getBuildSession, cancelBuildTakes, keepBuildTake, mapTakeTracks } from './yue2Takes.js';
import type { Yue2SynthRequest, Yue2TrackDetail } from './backends/yue2/client.js';
import { resetGpuLane, runOnGpuLane } from './generation/gpuLane.js';
import { listBackends } from './backends/registry.js';

// Take and preview lanes evict the other families first; with no engine in
// these tests, every other family's release is a confirmed no-op.
for (const b of listBackends()) if (b.id !== 'yue2' && b.releaseVram) b.releaseVram = async () => {};

const score = [
  'X:1', 'M:3/4', 'L:1/8', 'Q:1/8=120', 'K:C',
  '% intro', 'V: Vocal', 'z6|z6|',
  '% verse', 'V: Vocal', 'C6|D6|',
  '% chorus', 'V: Vocal', 'E6|F6|',
  '% outro', 'V: Vocal', 'z6|z6|', '',
].join('\n');
const lyrics = '[intro]\n\n[verse]\nfirst line here\n\n[chorus]\nhook line now\n\n[outro]';
/** Each section's lyric span, in source order. */
const spans = composeArrangement(score, lyrics, [0, 1, 2, 3]).occurrences.map(o => o.lyric);

function words(times: Array<[string, number]>): Yue2AlignWord[] {
  return times.map(([text, start]) => {
    const char0 = Array.from(lyrics.slice(0, lyrics.indexOf(text))).length;
    return { char0, char1: char0 + text.length, start, end: start + 0.4, score: 0.05 };
  });
}

test('a section starts halfway through the gap before its first sung word, by the take\'s own timing', () => {
  // Sung later than the score clock says (verse at 6 s, chorus at 12 s). With
  // two words the median start is the second.
  const starts = sectionStarts(spans, words([['first', 8], ['line here', 8.5], ['hook', 15], ['line now', 15.5]]), 30 * YUE2_FPS);
  assert.deepEqual(starts.map(f => (f === null ? null : f / YUE2_FPS)), [0, 8, 12.2, 16.4]);
});

test('squashed words, or a section past the take\'s end, give no start', () => {
  const squashed = words([['first', 8], ['line here', 8.5]]).concat(
    words([['hook', 9.9], ['line now', 9.92]]).map(w => ({ ...w, end: w.start + 0.01 })));
  const starts = sectionStarts(spans, squashed, 11 * YUE2_FPS);
  assert.equal(starts[2], null);
  assert.equal(starts[3], null);
});

test('a take that stops early is aligned only against lyrics up to the next section', () => {
  assert.equal(lyricsThrough(lyrics, spans, 1), '[intro]\n\n[verse]\nfirst line here\n\n');
  assert.equal(lyricsThrough(lyrics, spans, 3), lyrics);
});

test('a start that comes far sooner than the score allows falls back to the score clock', () => {
  // Intro 372 frames on the score; the aligner read the verse at 223.
  const misread: Array<number | null> = [0, 223];
  fillStarts(misread, [0, 372, 1366], 0, 1, 0, 584);
  assert.deepEqual(misread, [0, 372]);
  // A take a little faster than the score keeps its own reading.
  const fast: Array<number | null> = [0, 300];
  fillStarts(fast, [0, 372, 1366], 0, 1, 0, 584);
  assert.deepEqual(fast, [0, 300]);
  // Resumed at 223: a chorus read at 831 is too soon after the verse's 994 frames.
  const verse: Array<number | null> = [0, 223, 831];
  fillStarts(verse, [0, 372, 1366], 1, 2, 223, 1615);
  assert.deepEqual(verse, [0, 223, 1217]);
  // Past the take's end the estimate is not given.
  const short: Array<number | null> = [0, 100];
  fillStarts(short, [0, 372, 1366], 0, 1, 0, 300);
  assert.deepEqual(short, [0, null]);
});

test('a start at or before the previous section\'s is dropped', () => {
  const starts = sectionStarts(spans, words([['first', 8], ['line here', 8.5], ['hook', 4], ['line now', 4.5]]), 30 * YUE2_FPS);
  assert.equal(starts[2], null);
});

test('a waveform reads the peak level of each slice of a PCM WAV', async () => {
  const { wavPeaks } = await import('./yue2Takes.js');
  const rate = 48000, seconds = 2, frames = rate * seconds;
  const wav = Buffer.alloc(44 + frames * 4);
  wav.write('RIFF', 0); wav.writeUInt32LE(36 + frames * 4, 4); wav.write('WAVE', 8);
  wav.write('fmt ', 12); wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(2, 22);
  wav.writeUInt32LE(rate, 24); wav.writeUInt32LE(rate * 4, 28); wav.writeUInt16LE(4, 32); wav.writeUInt16LE(16, 34);
  wav.write('data', 36); wav.writeUInt32LE(frames * 4, 40);
  // Silent first second, half scale in the second.
  for (let f = rate; f < frames; f++) wav.writeInt16LE(16384, 44 + f * 4);
  const peaks = wavPeaks(wav, 0, 2);
  assert.equal(peaks.length, 120);
  assert.equal(peaks[10], 0);
  assert.equal(peaks[100], 0.5);
});

function pcm(marker: number): Buffer {
  const wav = Buffer.alloc(44 + 1920);
  wav.write('RIFF'); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(2, 22);
  wav.writeUInt32LE(48000, 24); wav.writeUInt16LE(16, 34); wav.write('data', 36);
  wav.writeUInt32LE(1920, 40);
  for (let i = 44; i < wav.length; i += 2) wav.writeInt16LE(marker * 1000, i);
  return wav;
}

test('section take chunks save every song with its own WAV, codes and seed', async t => {
  const previousDir = config.data.dir;
  const dir = fs.mkdtempSync(path.resolve('.test-yue2-takes-'));
  config.data.dir = dir;
  t.after(() => { config.data.dir = previousDir; fs.rmSync(dir, { recursive: true, force: true }); });
  const requests: Yue2SynthRequest[] = [];
  const results = new Map<string, { tracks: Yue2TrackDetail[]; parts: Buffer[] }>();
  let mode = 'done';
  let cancels = 0;
  let polls = 0;
  let aligns = 0;
  t.mock.method(globalThis, 'fetch', async (input: string | URL | Request, init?: RequestInit) => {
    const url = new URL(String(input));
    if (url.pathname === '/yue2/synth') {
      const req = JSON.parse(String(init?.body)) as Yue2SynthRequest;
      requests.push(req);
      const seeds = req.songs?.map(s => s.seed!) ?? [req.seed!];
      // A rebuilt engine echoes seeds as decimal strings.
      const tracks = seeds.map((seed, song) => ({ song, seed: String(seed), noise_seed: String(seed), variation: 0,
        semantic_ids: [...(req.codec_ids ?? []), ...Array<number>(400).fill(song + 1)], frames: (req.codec_ids?.length ?? 0) + 400 }));
      const parts = seeds.map((_, i) => pcm(i + 1));
      // Reversed response order must still map each WAV to its song and seed.
      tracks.reverse(); parts.reverse();
      if (mode === 'drop') { tracks.splice(1, 1); parts.splice(1, 1); }
      if (mode === 'mismatch') tracks[0].seed = String(Number(tracks[0].seed) + 1);
      const id = String(requests.length);
      results.set(id, { tracks, parts });
      return Response.json({ id });
    }
    if (url.pathname === '/yue2/align') { aligns++; return Response.json({ words: [] }); }
    assert.equal(url.pathname, '/job');
    const result = results.get(url.searchParams.get('id')!)!;
    if (url.searchParams.has('cancel')) { cancels++; return Response.json({}); }
    if (url.searchParams.has('result')) {
      if (result.parts.length === 1) return new Response(new Uint8Array(result.parts[0]), { headers: { 'content-type': 'audio/wav' } });
      const body = Buffer.concat(result.parts.flatMap(part => [Buffer.from('--takes\r\nContent-Type: audio/wav\r\n\r\n'), part, Buffer.from('\r\n')])
        .concat(Buffer.from('--takes--\r\n')));
      return new Response(new Uint8Array(body), { headers: { 'content-type': 'multipart/mixed; boundary=takes' } });
    }
    polls++;
    if (mode === 'running' || mode === 'timeout') return Response.json({ status: cancels ? 'cancelled' : 'running', phase: 'semantic', phase_step: 2, phase_total: 4 });
    if (mode === 'failed') return Response.json({ status: 'failed', error: 'row failed' });
    return Response.json({ status: 'done', tracks: result.tracks });
  });
  const session = () => createBuildSession({}, { style: 'ambient', cot: 'off', abc: score, lyrics: 'test lyrics', lm_batch_size: 4, synth_batch_size: 3, semantic_retries: 2 }, 'user');
  const until = async (condition: () => boolean) => {
    const deadline = Date.now() + 5000;
    while (!condition()) {
      assert.ok(Date.now() < deadline, 'take worker finished within deadline');
      await new Promise(resolve => setTimeout(resolve, 5));
    }
  };
  for (const [count, widths] of [[1, [1]], [4, [4]], [5, [4, 1]], [10, [4, 4, 2]]] as const) {
    await t.test(`${count} takes use ${widths.join('+')}`, async () => {
      const s = session();
      const first = requests.length;
      generateBuildTakes(s.id, 'user', 0, count, 0);
      await until(() => getBuildSession(s.id, 'user').pending.length === 0);
      const done = getBuildSession(s.id, 'user');
      assert.equal(done.error, null);
      assert.equal(done.takes.length, count);
      const sent = requests.slice(first);
      assert.deepEqual(sent.map(r => r.songs?.length ?? 1), widths);
      for (const req of sent) {
        assert.equal(req.synth_batch_size, 1);
        assert.equal(req.lm_batch_size, undefined);
        assert.equal(req.semantic_retries, undefined);
        const seeds = req.songs?.map(s => s.seed!) ?? [req.seed!];
        for (let song = 0; song < seeds.length; song++) {
          const take = done.takes.find(take => take.seed === seeds[song])!;
          const file = path.join(config.data.audioDir, 'takes', s.id, `${take.id}.wav`);
          assert.deepEqual(fs.readFileSync(file), pcm(song + 1));
          assert.deepEqual(JSON.parse(fs.readFileSync(file.replace('.wav', '.codes.json'), 'utf8')), Array(400).fill(song + 1));
          assert.equal(take.starts[0], 0);
          assert.equal(take.peaks?.length, 120);
        }
      }
      if (count === 4) {
        keepBuildTake(s.id, 'user', done.takes[0].id, 0);
        generateBuildTakes(s.id, 'user', 1, 4, 0);
        await until(() => getBuildSession(s.id, 'user').pending.length === 0);
        const req = requests.at(-1)!;
        assert.equal(req.continue_codec_ids, true);
        assert.deepEqual(req.codec_ids, Array(150).fill(1));
        assert.equal(req.songs?.length, 4);
      }
    });
  }
  assert.ok(aligns > 0);
  for (const failure of ['drop', 'mismatch', 'failed']) {
    await t.test(`${failure} rejects the chunk and clears all pending takes`, async () => {
      mode = failure;
      const first = requests.length;
      const s = session();
      generateBuildTakes(s.id, 'user', 0, 10, 0);
      await until(() => !getBuildSession(s.id, 'user').pending.length);
      const done = getBuildSession(s.id, 'user');
      assert.equal(done.takes.length, 0);
      assert.match(done.error!, /chunk|row failed/);
      assert.equal(requests.length - first, 1);
    });
  }
  await t.test('cancel shares chunk progress, stops its engine job and starts no later chunks', async () => {
    mode = 'running'; cancels = 0; polls = 0;
    const first = requests.length;
    const s = session();
    generateBuildTakes(s.id, 'user', 0, 10, 0);
    await until(() => getBuildSession(s.id, 'user').pending[0]?.progress === 50);
    assert.equal(getBuildSession(s.id, 'user').pending[0].activeCount, 4);
    assert.throws(() => generateBuildTakes(s.id, 'user', 0, 1, 0), /pending takes/);
    await assert.rejects(cancelBuildTakes(s.id, 'other-user'), /not found/);
    await cancelBuildTakes(s.id, 'user');
    await until(() => !getBuildSession(s.id, 'user').pending.length);
    assert.ok(cancels > 0);
    assert.equal(requests.length - first, 1);
    assert.equal(getBuildSession(s.id, 'user').takes.length, 0);
    assert.match(getBuildSession(s.id, 'user').error!, /cancelled/);
  });
  await t.test('timeout cancels the engine and clears pending counts', async st => {
    mode = 'timeout'; cancels = 0;
    let now = 0;
    st.mock.method(Date, 'now', () => now += 16 * 60_000);
    const s = session();
    generateBuildTakes(s.id, 'user', 0, 5, 0);
    // Use the event loop rather than a clock that this test has replaced.
    while (getBuildSession(s.id, 'user').pending.length) await new Promise(resolve => setImmediate(resolve));
    assert.match(getBuildSession(s.id, 'user').error!, /timed out/);
    assert.ok(cancels > 0);
  });
  await t.test('queue reset before submission starts no takes', async () => {
    mode = 'done';
    let release!: () => void;
    const occupied = runOnGpuLane(() => new Promise<void>(resolve => { release = resolve; }));
    await new Promise(resolve => setImmediate(resolve));
    const s = session();
    const first = requests.length;
    generateBuildTakes(s.id, 'user', 0, 5, 0);
    resetGpuLane(); release(); await occupied;
    await until(() => !getBuildSession(s.id, 'user').pending.length);
    assert.equal(requests.length, first);
  });
  await t.test('a take caption styles single and batched takes, is recorded, and empty inherits', async () => {
    mode = 'done';
    const composed: Array<Record<string, unknown>> = [];
    const toStyle = (params: Record<string, unknown>) => { composed.push(params); return `trig, in the style of trig. ${params.prompt}`; };
    for (const [count, caption, style] of [[1, '  brighter, more drums ', 'trig, in the style of trig. brighter, more drums'],
      [3, 'brighter', 'trig, in the style of trig. brighter'], [2, '   ', 'ambient'], [1, undefined, 'ambient']] as const) {
      const s = session();
      const first = requests.length;
      generateBuildTakes(s.id, 'user', 0, count, 0, caption, toStyle);
      await until(() => !getBuildSession(s.id, 'user').pending.length);
      const sent = requests.slice(first);
      assert.deepEqual(sent.flatMap(r => r.songs?.map(x => x.style) ?? [r.style]), Array(count).fill(style));
      const done = getBuildSession(s.id, 'user');
      assert.equal(done.revision, 0);
      for (const take of done.takes) {
        assert.equal(take.style, style);
        assert.equal(take.caption, style === 'ambient' ? undefined : caption!.trim());
      }
    }
    // Composed from the session's params with only the caption replaced.
    assert.deepEqual(composed.map(p => p.prompt), ['brighter, more drums', 'brighter']);
    const s = session();
    assert.throws(() => generateBuildTakes(s.id, 'user', 0, 1, 0, 7, toStyle), /caption must be text/);
    assert.throws(() => generateBuildTakes(s.id, 'user', 0, 1, 0, 'x'.repeat(4001), toStyle), /limited/);
  });
  await t.test('a queued take keeps the caption it was submitted with', async () => {
    mode = 'done';
    let release!: () => void;
    const occupied = runOnGpuLane(() => new Promise<void>(resolve => { release = resolve; }));
    await new Promise(resolve => setImmediate(resolve));
    const s = session();
    const first = requests.length;
    // The composer answers differently by the time the lane frees: the take must not ask again.
    let style = 'first';
    generateBuildTakes(s.id, 'user', 0, 1, 0, 'first', () => style);
    style = 'second';
    assert.equal(getBuildSession(s.id, 'user').pending[0].phase, 'queued');
    release(); await occupied;
    await until(() => !getBuildSession(s.id, 'user').pending.length);
    assert.equal(requests[first].style, 'first');
    assert.equal(getBuildSession(s.id, 'user').takes[0].caption, 'first');
  });
});

test('track mapping rejects duplicate indices, unknown indices, extra variations and missing audio', () => {
  const tracks = [0, 1].map(song => ({ song, seed: 10 + song, noise_seed: 10 + song, variation: 0, semantic_ids: [song] }));
  for (const bad of [{ song: 0 }, { song: 3 }, { variation: 1 }, { seed: 999 }, { semantic_ids: [] }]) {
    assert.throws(() => mapTakeTracks([10, 11], [tracks[0], { ...tracks[1], ...bad }], [pcm(1), pcm(2)]), /chunk/);
  }
  assert.throws(() => mapTakeTracks([10, 11], tracks, [pcm(1)]), /audio parts/);
  assert.throws(() => mapTakeTracks([10, 11], tracks, [pcm(1), Buffer.from('invalid audio')]), /valid WAV/);
});

test('cover-local lyrics: validated, revisioned, isolated, and the next take follows them', async t => {
  const { applyBuildLyrics, deleteBuildTake, undoBuildKeep, lyricsProblem } = await import('./yue2Takes.js');
  const previousDir = config.data.dir;
  const dir = fs.mkdtempSync(path.resolve('.test-yue2-lyrics-'));
  config.data.dir = dir;
  t.after(() => { config.data.dir = previousDir; fs.rmSync(dir, { recursive: true, force: true }); });
  const requests: Yue2SynthRequest[] = [];
  const results = new Map<string, Yue2TrackDetail[]>();
  let hold = false, cancelled = false;
  t.mock.method(globalThis, 'fetch', async (input: string | URL | Request, init?: RequestInit) => {
    const url = new URL(String(input));
    if (url.pathname === '/yue2/synth') {
      const req = JSON.parse(String(init?.body)) as Yue2SynthRequest;
      requests.push(req);
      const id = String(requests.length);
      results.set(id, [{ song: 0, seed: req.seed!, noise_seed: req.seed!, variation: 0,
        semantic_ids: [...(req.codec_ids ?? []), ...Array<number>(400).fill(1)], frames: 400 }]);
      return Response.json({ id });
    }
    if (url.pathname === '/yue2/align') return Response.json({ words: [] });
    if (url.searchParams.has('cancel')) { cancelled = true; return Response.json({}); }
    if (url.searchParams.has('result')) return new Response(new Uint8Array(pcm(1)), { headers: { 'content-type': 'audio/wav' } });
    if (hold) return Response.json({ status: cancelled ? 'cancelled' : 'running', phase: 'semantic', phase_step: 1, phase_total: 4 });
    return Response.json({ status: 'done', tracks: results.get(url.searchParams.get('id')!) });
  });
  const until = async (condition: () => boolean) => {
    const deadline = Date.now() + 5000;
    while (!condition()) { assert.ok(Date.now() < deadline, 'finished within deadline'); await new Promise(r => setTimeout(r, 5)); }
  };
  const toRequest = (p: Record<string, unknown>) => ({ style: 'ambient', cot: 'off', abc: score, lyrics: String(p.lyrics) }) as Yue2SynthRequest;
  const start = (user = 'user') => createBuildSession({ lyrics }, toRequest({ lyrics }), user);
  const settle = (id: string) => until(() => !getBuildSession(id, 'user').pending.length);
  const edited = lyrics.replace('hook line now', 'a brand new hook');

  await t.test('length and section tags are validated', () => {
    assert.equal(lyricsProblem(lyrics), null);
    assert.match(lyricsProblem('')!, /empty/);
    assert.match(lyricsProblem('x'.repeat(20_001))!, /limited/);
    assert.match(lyricsProblem('just words')!, /section tags/);
    assert.match(lyricsProblem('words first\n[verse]\nmore')!, /above the first line/);
    assert.match(lyricsProblem('[verse\nmore')!, /Line 1/);
    assert.match(lyricsProblem('[ ]\nmore')!, /Line 1/);
    const s = start();
    assert.throws(() => applyBuildLyrics(s.id, 'user', 0, 'no tags here', toRequest), /section tags/);
    assert.equal(getBuildSession(s.id, 'user').revision, 0);
  });

  await t.test('apply changes params and request together, survives reload, and bumps the revision once', async () => {
    const s = start();
    assert.throws(() => applyBuildLyrics(s.id, 'user', 3, edited, toRequest), /changed since/);
    const next = applyBuildLyrics(s.id, 'user', 0, edited, toRequest);
    assert.equal(next.revision, 1);
    assert.equal(next.params.lyrics, edited);
    assert.equal(getBuildSession(s.id, 'user').params.lyrics, edited);
    assert.equal(applyBuildLyrics(s.id, 'user', 1, edited, toRequest).revision, 1);
    generateBuildTakes(s.id, 'user', 0, 1, 1);
    await settle(s.id);
    assert.equal(requests.at(-1)!.lyrics, edited);
    assert.equal(getBuildSession(s.id, 'user').takes[0].revision, 1);
  });

  await t.test('two sessions and two users stay apart', () => {
    const a = start(), b = start(), other = start('other');
    applyBuildLyrics(a.id, 'user', 0, edited, toRequest);
    assert.equal(getBuildSession(b.id, 'user').params.lyrics, lyrics);
    assert.equal(getBuildSession(other.id, 'other').params.lyrics, lyrics);
    assert.throws(() => applyBuildLyrics(a.id, 'other', 1, lyrics, toRequest), /not found/);
    assert.equal(getBuildSession(a.id, 'user').params.lyrics, edited);
  });

  await t.test('updates are refused while takes are pending', async () => {
    hold = true;
    const s = start();
    generateBuildTakes(s.id, 'user', 0, 1, 0);
    await until(() => getBuildSession(s.id, 'user').pending[0]?.progress === 25);
    assert.throws(() => applyBuildLyrics(s.id, 'user', 0, edited, toRequest), /pending/);
    await cancelBuildTakes(s.id, 'user');
    await settle(s.id);
    hold = false;
  });

  await t.test('delete build: refused while rendering and for another user, then removes the folder', async () => {
    const { deleteBuild } = await import('./yue2Takes.js');
    hold = true; cancelled = false;
    const s = start();
    generateBuildTakes(s.id, 'user', 0, 1, 0);
    await until(() => getBuildSession(s.id, 'user').pending[0]?.progress === 25);
    assert.throws(() => deleteBuild(s.id, 'user'), /still rendering/);
    await cancelBuildTakes(s.id, 'user');
    await settle(s.id);
    hold = false;
    const dir = path.join(config.data.audioDir, 'takes', s.id);
    assert.throws(() => deleteBuild(s.id, 'other'), /not found/);
    assert.throws(() => deleteBuild(s.id, 'user', id => id === s.id), /Save song render is queued/);
    assert.ok(fs.existsSync(dir));
    deleteBuild(s.id, 'user');
    assert.ok(!fs.existsSync(dir));
    assert.throws(() => getBuildSession(s.id, 'user'), /not found/);
  });

  await t.test('candidates made before an edit cannot be kept; mutations on the old revision fail', async () => {
    const s = start();
    generateBuildTakes(s.id, 'user', 0, 1, 0);
    await settle(s.id);
    const old = getBuildSession(s.id, 'user').takes[0];
    applyBuildLyrics(s.id, 'user', 0, edited, toRequest);
    assert.throws(() => keepBuildTake(s.id, 'user', old.id, 0), /changed since/);
    assert.throws(() => keepBuildTake(s.id, 'user', old.id, 1), /before the last change/);
    assert.throws(() => generateBuildTakes(s.id, 'user', 0, 1, 0), /changed since/);
    assert.throws(() => deleteBuildTake(s.id, 'user', old.id, 0), /changed since/);
    assert.throws(() => undoBuildKeep(s.id, 'user', undefined, 0), /changed since/);
  });

  await t.test('lyrics of built sections are locked, later ones are free, kept audio stays', async () => {
    const s = start();
    generateBuildTakes(s.id, 'user', 1, 1, 0);
    await settle(s.id);
    const take = getBuildSession(s.id, 'user').takes[0];
    keepBuildTake(s.id, 'user', take.id, 0);
    assert.throws(() => applyBuildLyrics(s.id, 'user', 0, lyrics.replace('first line here', 'another line'), toRequest), /locked/);
    assert.throws(() => applyBuildLyrics(s.id, 'user', 0, lyrics.replace('[intro]\n', '[intro]\nla la\n'), toRequest), /locked/);
    assert.equal(getBuildSession(s.id, 'user').revision, 0);
    const next = applyBuildLyrics(s.id, 'user', 0, edited, toRequest);
    assert.deepEqual(next.kept, [take.id]);
    assert.ok(fs.existsSync(path.join(config.data.audioDir, 'takes', s.id, `${take.id}.wav`)));
    // The kept take stays the spine; the next section resumes it under the new text.
    generateBuildTakes(s.id, 'user', 2, 1, 1);
    await settle(s.id);
    assert.equal(requests.at(-1)!.lyrics, edited);
    assert.equal(requests.at(-1)!.continue_codec_ids, true);
  });

  await t.test('a session written before revisions opens as version 2', async () => {
    const s = start();
    generateBuildTakes(s.id, 'user', 0, 1, 0);
    await settle(s.id);
    const file = path.join(config.data.audioDir, 'takes', s.id, 'session.json');
    const stored = JSON.parse(fs.readFileSync(file, 'utf8'));
    keepBuildTake(s.id, 'user', stored.takes[0].id, 0);
    const legacy = JSON.parse(fs.readFileSync(file, 'utf8'));
    for (const key of ['version', 'revision', 'arrangement']) delete legacy[key];
    legacy.takes = legacy.takes.map(({ revision: _r, parent: _p, occurrences: _o, ...take }: Record<string, unknown>) => take);
    fs.writeFileSync(file, JSON.stringify(legacy));
    const opened = getBuildSession(s.id, 'user');
    assert.equal(opened.version, 2);
    assert.equal(opened.revision, 0);
    assert.equal(opened.takes[0].parent, null);
    assert.deepEqual(opened.kept, [stored.takes[0].id]);
    generateBuildTakes(s.id, 'user', 1, 1, 0);
    await settle(s.id);
    assert.equal(getBuildSession(s.id, 'user').takes.at(-1)!.parent, stored.takes[0].id);
  });
});

test('arrangements: takes and Save song follow the plan, by occurrence', async t => {
  const { editBuildPlan, buildSessionRender, undoBuildKeep } = await import('./yue2Takes.js');
  const { buildYue2LyricSchedule } = await import('./backends/yue2/lyricSchedule.js');
  const previousDir = config.data.dir;
  const dir = fs.mkdtempSync(path.resolve('.test-yue2-arrangement-'));
  config.data.dir = dir;
  t.after(() => { config.data.dir = previousDir; fs.rmSync(dir, { recursive: true, force: true }); });
  const requests: Yue2SynthRequest[] = [];
  const aligned: string[] = [];
  const results = new Map<string, Yue2TrackDetail[]>();
  let hold = false, cancelled = false;
  t.mock.method(globalThis, 'fetch', async (input: string | URL | Request, init?: RequestInit) => {
    const url = new URL(String(input));
    if (url.pathname === '/yue2/synth') {
      const req = JSON.parse(String(init?.body)) as Yue2SynthRequest;
      requests.push(req);
      const id = String(requests.length);
      // Each take's new codes carry its request number, so prefixes are traceable.
      const length = (req.codec_ids?.length ?? 0) + (req.continue_codec_ids || !req.codec_ids ? 400 : 0);
      results.set(id, (req.songs?.map(s => s.seed!) ?? [req.seed!]).map((seed, song) => ({ song, seed, noise_seed: seed, variation: 0,
        semantic_ids: [...(req.codec_ids ?? []), ...Array<number>(length - (req.codec_ids?.length ?? 0)).fill(requests.length)], frames: length,
        audio_from: Math.min(req.render_from_frame ?? 0, length - 1) })));
      return Response.json({ id });
    }
    if (url.pathname === '/yue2/align') { aligned.push(String((init?.body as FormData).get('lyrics'))); return Response.json({ words: [] }); }
    if (url.searchParams.has('cancel')) { cancelled = true; return Response.json({}); }
    const tracks = results.get(url.searchParams.get('id')!)!;
    if (url.searchParams.has('result')) {
      if (tracks.length === 1) return new Response(new Uint8Array(pcm(1)), { headers: { 'content-type': 'audio/wav' } });
      const body = Buffer.concat(tracks.flatMap(() => [Buffer.from('--t\r\nContent-Type: audio/wav\r\n\r\n'), pcm(1), Buffer.from('\r\n')])
        .concat(Buffer.from('--t--\r\n')));
      return new Response(new Uint8Array(body), { headers: { 'content-type': 'multipart/mixed; boundary=t' } });
    }
    if (hold) return Response.json({ status: cancelled ? 'cancelled' : 'running', phase: 'semantic', phase_step: 1, phase_total: 4 });
    return Response.json({ status: 'done', tracks });
  });
  const until = async (condition: () => boolean) => {
    const deadline = Date.now() + 5000;
    while (!condition()) { assert.ok(Date.now() < deadline, 'finished within deadline'); await new Promise(r => setTimeout(r, 5)); }
  };
  const options = { mode: 'bias', bias: -4, abc: false, leadSec: 0, behind: -1 } as const;
  const start = (text = lyrics, abc = score) => createBuildSession({ lyrics: text }, { style: 'ambient', cot: 'full', abc, lyrics: text,
    lyric_schedule: buildYue2LyricSchedule(abc, text, options).wire }, 'user');
  const settle = (id: string) => until(() => !getBuildSession(id, 'user').pending.length);
  const take = async (id: string, to: number, revision: number, count = 1) => {
    const before = getBuildSession(id, 'user').takes.length;
    generateBuildTakes(id, 'user', to, count, revision);
    await settle(id);
    const after = getBuildSession(id, 'user');
    assert.equal(after.error, null);
    return after.takes.slice(before);
  };
  // Skip the intro, repeat the chorus: verse, chorus, chorus, outro.
  const composed = composeArrangement(score, lyrics, [1, 2, 2, 3]);

  await t.test('the source order sends the stored score and lyrics unchanged', async () => {
    const s = start();
    await take(s.id, 0, 0);
    assert.equal(requests.at(-1)!.abc, score);
    assert.equal(requests.at(-1)!.lyrics, lyrics);
    assert.equal(requests.at(-1)!.lyric_schedule, undefined, 'Lyric Timing is off for builds');
  });

  await t.test('an Improv section is composed blind: score and lyrics hidden for its bars only', async () => {
    const improv = score.replace('% chorus', '% Improv\nV: Vocal\nz6|z6|\n% chorus');
    const s = start(lyrics, improv);
    const [opening] = await take(s.id, 1, 0);
    assert.equal(requests.at(-1)!.lyric_schedule, undefined);
    keepBuildTake(s.id, 'user', opening.id, 0);
    await take(s.id, 2, 0);
    const blind = requests.at(-1)!.lyric_schedule!;
    // Intro and verse 0-12 s, improv 12-18 s; sampling resumes after frame 300.
    assert.deepEqual([blind.mode, blind.sections, blind.blind!.map(w => [w.from_sec, w.to_sec])], ['mask', [], [[12.04, 18]]]);
    assert.deepEqual(blind.blind![0].lyric, [[0, Array.from(lyrics.trimEnd()).length]]);
    assert.ok(blind.blind![0].abc!.some(([a, b]) => Array.from(improv).slice(a, b).join('').includes('E6|F6|')), 'the chorus lines are hidden too');
    assert.ok(blind.blind![0].abc!.every(([a, b]) => !/^[MLQKV]:/m.test(Array.from(improv).slice(a, b).join(''))), 'field lines stay visible');
    keepBuildTake(s.id, 'user', getBuildSession(s.id, 'user').takes.at(-1)!.id, 0);
    await take(s.id, 3, 0);
    assert.equal(requests.at(-1)!.lyric_schedule, undefined, 'the take after it reads the score again');
  });

  await t.test('an improv added to the plan writes its rest bars and is composed blind', async () => {
    const s = start();
    editBuildPlan(s.id, 'user', 0, ['s0', 's1', 'improv:2', 's2'], false);
    const [opening] = await take(s.id, 1, 1);
    keepBuildTake(s.id, 'user', opening.id, 1);
    const [improv] = await take(s.id, 2, 1);
    const req = requests.at(-1)!;
    assert.match(req.abc!, /% Improv\nV: Vocal\nz6\|z6\|\n% chorus/);
    assert.deepEqual(req.lyric_schedule!.blind!.map(w => [w.from_sec, w.to_sec]), [[12.04, 18]]);
    assert.equal(improv.occurrences.length, 1);
  });

  await t.test('skip, repeat and continue: each take resumes the kept output under the composed prompt', async () => {
    const s = start();
    const planned = editBuildPlan(s.id, 'user', 0, ['s1', 's2', 's2', 's3'], true);
    assert.equal(planned.revision, 1);
    assert.deepEqual(planned.arrangement.plan.map(o => o.section), ['s1', 's2', 's2', 's3']);
    // Identical labels, distinct occurrences that never reuse an earlier id.
    assert.deepEqual(planned.arrangement.plan.map(o => o.id), ['o4', 'o5', 'o6', 'o7']);
    const [verse] = await take(s.id, 0, 1);
    const first = requests.at(-1)!;
    assert.equal(first.abc, composed.abc);
    assert.equal(first.lyrics, composed.lyrics);
    assert.equal(first.codec_ids, undefined);
    // Two 3 s bars of verse on the composed clock: 150 frames, then slack.
    assert.equal(first.preview_max_frames, Math.round(150 * 1.3) + 100);
    assert.equal(first.lyric_schedule, undefined);
    assert.deepEqual(verse.occurrences, ['o4']);
    keepBuildTake(s.id, 'user', verse.id, 1);
    const [chorus] = await take(s.id, 1, 1);
    assert.deepEqual(requests.at(-1)!.codec_ids, Array(150).fill(requests.length - 1));
    assert.equal(requests.at(-1)!.abc, composed.abc);
    assert.deepEqual([chorus.parent, chorus.from, chorus.occurrences], [verse.id, 1, ['o5']]);
    keepBuildTake(s.id, 'user', chorus.id, 1);
    const [again] = await take(s.id, 2, 1);
    // The repeat resumes the first chorus's own stream where the repeat starts.
    assert.equal(requests.at(-1)!.codec_ids!.length, 300);
    assert.deepEqual(requests.at(-1)!.codec_ids!.slice(150), Array(150).fill(requests.length - 1));
    assert.deepEqual(again.occurrences, ['o6']);
    keepBuildTake(s.id, 'user', again.id, 1);
    // Save song: the spine's whole stream under the same arranged prompt.
    const render = buildSessionRender(s.id, 'user');
    assert.equal(render.abc, composed.abc);
    assert.equal(render.lyrics, composed.lyrics);
    assert.equal(render.codec_ids!.length, 300 + 400);
    assert.equal(render.lyric_schedule, undefined);
    // The last occurrence ends the song: no cap.
    await take(s.id, 3, 1);
    assert.equal(requests.at(-1)!.preview_max_frames, undefined);
  });

  await t.test('kept occurrences stay put; an unfinished plan builds its last one into the next source section', async () => {
    const s = start();
    editBuildPlan(s.id, 'user', 0, ['s1', 's2'], false);
    const [open] = await take(s.id, 1, 1);
    // The chorus runs on into the outro that follows it in the score, and stops there.
    assert.equal(requests.at(-1)!.abc, composeArrangement(score, lyrics, [1, 2, 3]).abc);
    assert.notEqual(requests.at(-1)!.preview_max_frames, undefined);
    assert.equal(open.starts.length, 3);
    const [verse] = await take(s.id, 0, 1);
    keepBuildTake(s.id, 'user', verse.id, 1);
    // Upcoming chorus o5 keeps its identity; a new outro follows it.
    const next = editBuildPlan(s.id, 'user', 1, ['o5', 's3'], true);
    assert.deepEqual(next.arrangement.plan.map(o => o.id), ['o4', 'o5', 'o6']);
    assert.equal(next.revision, 2);
    assert.equal(editBuildPlan(s.id, 'user', 2, ['o5', 'o6'], true).revision, 2);
    assert.throws(() => editBuildPlan(s.id, 'user', 2, ['o4'], true), /No such section/);
    assert.throws(() => editBuildPlan(s.id, 'user', 2, 'o5', true), /must list/);
    assert.deepEqual(getBuildSession(s.id, 'user').kept, [verse.id]);
    assert.deepEqual(getBuildSession(s.id, 'user').arrangement.plan[0].id, 'o4');
  });

  await t.test('a crop that keeps the start of the run-on section carries on from the cut', async () => {
    const { cropBuildTake } = await import('./yue2Takes.js');
    const s = start();
    editBuildPlan(s.id, 'user', 0, ['s1', 's2'], false);
    const [open] = await take(s.id, 1, 1);
    assert.equal(open.after, 's3', 'the chorus ran on into the outro');
    keepBuildTake(s.id, 'user', open.id, 1);
    const cut = open.starts[2]! + 10;
    const kept = cropBuildTake(s.id, 'user', open.id, 1, cut, 'regenerate');
    const spine = kept.takes.find(x => x.id === kept.kept.at(-1))!;
    assert.deepEqual([spine.to, spine.resume], [1, cut]);
    // Another section next would sing over the kept start of the outro.
    let rev = editBuildPlan(s.id, 'user', 1, ['s0'], false).revision;
    assert.throws(() => generateBuildTakes(s.id, 'user', 2, 1, rev), /carries on into outro/i);
    rev = editBuildPlan(s.id, 'user', rev, ['s3'], true).revision;
    await take(s.id, 2, rev);
    assert.equal(requests.at(-1)!.codec_ids!.length, cut, 'the next take resumes at the cut');
  });

  await t.test('a plan edit after a take leaves it keepable while its own sections are untouched', async () => {
    const s = start();
    editBuildPlan(s.id, 'user', 0, ['s0', 's1'], false);
    const [both] = await take(s.id, 1, 1);
    // Appending the next section, or moving what follows the take, does not touch it.
    let now = editBuildPlan(s.id, 'user', 1, ['o4', 'o5', 's2'], false);
    assert.equal(now.takes.find(x => x.id === both.id)!.revision, now.revision);
    // Its own sections changed: not keepable.
    now = editBuildPlan(s.id, 'user', now.revision, ['o5', 'o4', 'o6'], false);
    assert.notEqual(now.takes.find(x => x.id === both.id)!.revision, now.revision);
    assert.throws(() => keepBuildTake(s.id, 'user', both.id, now.revision), /before the last change/);
  });

  await t.test('an unbuilt repeat sings its own lines; built ones and tags are refused', async () => {
    const { setOccurrenceLyrics } = await import('./yue2Takes.js');
    const s = start();
    editBuildPlan(s.id, 'user', 0, ['s1', 's2', 's2'], true);
    const [verse] = await take(s.id, 0, 1);
    keepBuildTake(s.id, 'user', verse.id, 1);
    const [, , second] = getBuildSession(s.id, 'user').arrangement.plan;
    const edited = setOccurrenceLyrics(s.id, 'user', 1, { [second.id]: 'a new chorus\r\nline two' });
    assert.equal(edited.revision, 2);
    assert.equal(edited.arrangement.plan[2].lyrics, 'a new chorus\nline two');
    await take(s.id, 2, 2);
    const sent = requests.at(-1)!.lyrics!;
    assert.ok(sent.includes('a new chorus\nline two'));
    // The first chorus still sings the build's words.
    assert.equal(sent.split('hook line now').length, 2);
    assert.throws(() => setOccurrenceLyrics(s.id, 'user', 2, { [verse.occurrences[0]]: 'x' }), /is built/);
    assert.throws(() => setOccurrenceLyrics(s.id, 'user', 2, { [second.id]: 'x\n[Bridge]\ny' }), /section tag/);
    assert.throws(() => setOccurrenceLyrics(s.id, 'user', 2, { [second.id]: '  ' }), /needs words/);
    assert.throws(() => setOccurrenceLyrics(s.id, 'user', 1, { [second.id]: null }), /changed/);
    assert.equal(setOccurrenceLyrics(s.id, 'user', 2, { [second.id]: null }).arrangement.plan[2].lyrics, undefined);
  });

  await t.test('an unbuilt repeat sings an octave up; built ones and improvs are refused', async () => {
    const { setOccurrenceOctave } = await import('./yue2Takes.js');
    const s = start();
    editBuildPlan(s.id, 'user', 0, ['s1', 's2', 's2', 'improv:2'], true);
    const [verse] = await take(s.id, 0, 1);
    keepBuildTake(s.id, 'user', verse.id, 1);
    const [, , second, improv] = getBuildSession(s.id, 'user').arrangement.plan;
    const edited = setOccurrenceOctave(s.id, 'user', 1, { [second.id]: 1 });
    assert.equal(edited.revision, 2);
    assert.equal(edited.arrangement.plan[2].octave, 1);
    await take(s.id, 2, 2);
    const plan = [1, 2, 2, { improv: 2 }];
    const plain = composeArrangement(score, lyrics, plan).abc;
    const sent = requests.at(-1)!.abc!;
    assert.notEqual(sent, plain);
    assert.equal(sent, composeArrangement(score, lyrics, plan, [], [undefined, undefined, 1]).abc);
    assert.throws(() => setOccurrenceOctave(s.id, 'user', 2, { [verse.occurrences[0]]: 1 }), /is built/);
    assert.throws(() => setOccurrenceOctave(s.id, 'user', 2, { [improv.id]: 1 }), /improv/);
    assert.throws(() => setOccurrenceOctave(s.id, 'user', 2, { [second.id]: 2 }), /-1, 0 or 1/);
    assert.equal(setOccurrenceOctave(s.id, 'user', 2, { [second.id]: 0 }).arrangement.plan[2].octave, undefined);
  });

  await t.test('stale revisions, stale candidates and stale parents are refused', async () => {
    const s = start();
    const [a, b] = await take(s.id, 0, 0, 2);
    keepBuildTake(s.id, 'user', a.id, 0);
    const [fromA] = await take(s.id, 1, 0);
    undoBuildKeep(s.id, 'user', undefined, 0);
    keepBuildTake(s.id, 'user', b.id, 0);
    // fromA resumed a, which is no longer the spine.
    assert.throws(() => keepBuildTake(s.id, 'user', fromA.id, 0), /does not follow/);
    const [fromB] = await take(s.id, 1, 0);
    editBuildPlan(s.id, 'user', 0, ['s2', 's2', 's3'], true);
    assert.throws(() => editBuildPlan(s.id, 'user', 0, ['s2'], true), /changed since/);
    assert.throws(() => keepBuildTake(s.id, 'user', fromB.id, 1), /before the last change/);
  });

  await t.test('only the kept occurrences lock their lyrics, whatever their source position', async () => {
    const { applyBuildLyrics } = await import('./yue2Takes.js');
    const toRequest = (p: Record<string, unknown>) => ({ style: 'ambient', cot: 'full', abc: score, lyrics: String(p.lyrics) }) as Yue2SynthRequest;
    const s = start();
    editBuildPlan(s.id, 'user', 0, ['s2', 's0', 's1'], true);
    const [chorus] = await take(s.id, 0, 1);
    keepBuildTake(s.id, 'user', chorus.id, 1);
    // The intro sits before the kept chorus in the source, but is not built yet.
    const sungIntro = lyrics.replace('[intro]\n', '[intro]\nla la la\n');
    assert.equal(applyBuildLyrics(s.id, 'user', 1, sungIntro, toRequest).revision, 2);
    assert.throws(() => applyBuildLyrics(s.id, 'user', 2, sungIntro.replace('hook line now', 'other words'), toRequest), /locked/);
  });

  await t.test('plan edits wait for pending takes', async () => {
    hold = true;
    const s = start();
    generateBuildTakes(s.id, 'user', 0, 1, 0);
    await until(() => getBuildSession(s.id, 'user').pending[0]?.progress === 25);
    assert.throws(() => editBuildPlan(s.id, 'user', 0, ['s2'], true), /pending/);
    await cancelBuildTakes(s.id, 'user');
    await settle(s.id);
    hold = false;
    assert.equal(getBuildSession(s.id, 'user').revision, 0);
  });

  await t.test('a plan whose text would pair a block with the wrong occurrence sends no schedule', async () => {
    // The unsung second chorus moved ahead of the sung one would take its
    // block if the composed text were paired again.
    const abc = ['X:1', 'M:4/4', 'L:1/16', 'Q:1/4=120', 'K:C', '% verse', 'V: Vocal', 'C16|',
      '% chorus', 'V: Vocal', 'c16|', '% chorus', 'V: Vocal', 'Z|', ''].join('\n');
    const text = '[Verse]\nla la\n\n[Chorus]\noh oh\n';
    const plan = [0, 2, 1];
    assert.equal(composeArrangement(abc, text, plan).textPairingMatches, false);
    const s = start(text, abc);
    editBuildPlan(s.id, 'user', 0, plan.map(i => `s${i}`), true);
    await take(s.id, 0, 1);
    assert.equal(requests.at(-1)!.lyric_schedule, undefined);
    assert.equal(requests.at(-1)!.abc, composeArrangement(abc, text, plan).abc);
  });

  await t.test('a later take renders only from its lead-in and aligns only its own lyrics; Play build renders the whole build once', async () => {
    const { renderBuildPreview } = await import('./yue2Takes.js');
    const s = start();
    const [a] = await take(s.id, 0, 0);
    assert.equal(requests.at(-1)!.render_from_frame, undefined, 'the first take renders from the top');
    assert.equal(a.audioStart, undefined);
    keepBuildTake(s.id, 'user', a.id, 0);
    const [b] = await take(s.id, 1, 0);
    const lead = b.prefix! - 4 * YUE2_FPS;
    assert.equal(requests.at(-1)!.render_from_frame, lead);
    assert.equal(b.audioStart, lead);
    assert.ok(aligned.at(-2)!.includes('[intro]') && !aligned.at(-1)!.includes('[intro]'), 'the kept section is not aligned again');
    // Play build: the spine's whole stream, prefilled, rendered from the top.
    keepBuildTake(s.id, 'user', b.id, 0);
    assert.equal(renderBuildPreview(s.id, 'user').preview?.status, 'rendering');
    await until(() => getBuildSession(s.id, 'user').preview?.status !== 'rendering');
    const preview = getBuildSession(s.id, 'user').preview!;
    assert.deepEqual([preview.status, preview.spine], ['done', b.id]);
    assert.ok(fs.existsSync(path.join(config.data.audioDir, 'takes', s.id, path.basename(preview.audioUrl!))));
    const req = requests.at(-1)!;
    assert.deepEqual([req.codec_ids!.length, req.continue_codec_ids, req.render_from_frame, req.lyric_schedule], [b.frames, undefined, undefined, undefined]);
  });

  await t.test('trim: only a first-section take starts later, and only while a version of it is the first keep', async () => {
    const { setBuildTrim, cropBuildTake } = await import('./yue2Takes.js');
    const s = start();
    const [a, other] = await take(s.id, 0, 0, 2);
    const room = a.frames / YUE2_FPS - 1;
    assert.throws(() => setBuildTrim(s.id, 'user', a.id, 0, room + 0.5), new RegExp(`Start between 0 and ${room.toFixed(2)} s`));
    assert.throws(() => setBuildTrim(s.id, 'user', a.id, 0, 0), /Start between/);
    assert.throws(() => setBuildTrim(s.id, 'user', a.id, 7, 1), /changed since/);
    assert.equal(setBuildTrim(s.id, 'user', a.id, 0, 1.25).trim?.seconds, 1.25);
    keepBuildTake(s.id, 'user', other.id, 0);
    assert.equal(buildSessionRender(s.id, 'user').trimStart, undefined, 'another take is the first keep');
    undoBuildKeep(s.id, 'user', undefined, 0);
    keepBuildTake(s.id, 'user', a.id, 0);
    assert.equal(buildSessionRender(s.id, 'user').trimStart, 1.25);
    // A crop of the trimmed take keeps the trim.
    cropBuildTake(s.id, 'user', a.id, 0, 120, 'advance');
    assert.equal(buildSessionRender(s.id, 'user').trimStart, 1.25);
    const [b] = await take(s.id, 1, 0);
    assert.throws(() => setBuildTrim(s.id, 'user', b.id, 0, 1), /first section/);
    assert.equal(setBuildTrim(s.id, 'user', '', 0, null).trim, null);
    assert.equal(buildSessionRender(s.id, 'user').trimStart, undefined);
  });

  await t.test('crop: the next take and Save song resume exactly the cut prefix, and the original comes back', async () => {
    const { cropBuildTake, deleteBuildTake } = await import('./yue2Takes.js');
    const codes = (sid: string, take: string) => JSON.parse(fs.readFileSync(path.join(config.data.audioDir, 'takes', sid, `${take}.codes.json`), 'utf8')) as number[];
    const s = start();
    const [a] = await take(s.id, 0, 0);
    keepBuildTake(s.id, 'user', a.id, 0);
    // b resumes a's 150 frames and composes the verse; the chorus would start at 300.
    const [b] = await take(s.id, 1, 0);
    assert.deepEqual([b.prefix, b.starts], [150, [0, 150, 300, null]]);
    keepBuildTake(s.id, 'user', b.id, 0);
    const [d] = await take(s.id, 2, 0);
    assert.throws(() => cropBuildTake(s.id, 'user', b.id, 0, 200, undefined), /inside verse: choose/);
    assert.throws(() => cropBuildTake(s.id, 'user', b.id, 1, 200, 'advance'), /changed since/);
    assert.throws(() => cropBuildTake(s.id, 'user', a.id, 0, 100, 'advance'), /Only the last kept/);
    assert.throws(() => cropBuildTake(s.id, 'user', b.id, 0, 150, 'advance'), /inside the part this take composed/);

    // Move on: the chorus starts at the cut.
    const cut = cropBuildTake(s.id, 'user', b.id, 0, 200, 'advance');
    const c = cut.takes.find(x => x.id === cut.kept.at(-1))!;
    assert.deepEqual([c.cropOf, c.frames, c.to, c.starts, c.occurrences, cut.revision], [b.id, 200, 1, [0, 150, 200, null], b.occurrences, 0]);
    assert.deepEqual(codes(s.id, c.id), codes(s.id, b.id).slice(0, 200));
    assert.deepEqual(codes(s.id, b.id).length, 550);
    assert.deepEqual(buildSessionRender(s.id, 'user').codec_ids, codes(s.id, b.id).slice(0, 200));
    // d continued the uncut take, so it waits for the original.
    assert.throws(() => keepBuildTake(s.id, 'user', d.id, 0), /does not follow/);
    assert.throws(() => deleteBuildTake(s.id, 'user', b.id, 0), /undo the crop/);
    const [e] = await take(s.id, 2, 0);
    assert.deepEqual(requests.at(-1)!.codec_ids, codes(s.id, b.id).slice(0, 200));
    assert.deepEqual([e.parent, e.prefix, e.from], [c.id, 200, 2]);

    // Restore: the original is the spine again, the crop stays for its take e.
    const restored = cropBuildTake(s.id, 'user', c.id, 0, null, undefined);
    assert.equal(restored.kept.at(-1), b.id);
    assert.ok(restored.takes.some(x => x.id === c.id));
    assert.throws(() => keepBuildTake(s.id, 'user', e.id, 0), /does not follow/);
    assert.throws(() => cropBuildTake(s.id, 'user', b.id, 0, null, undefined), /not cropped/);

    // Compose it again: the verse is open, and the next take resumes at the cut.
    const again = cropBuildTake(s.id, 'user', b.id, 0, 200, 'regenerate');
    const r = again.takes.find(x => x.id === again.kept.at(-1))!;
    assert.deepEqual([r.to, r.resume, r.starts, r.occurrences], [0, 200, [0, 150, null, null], []]);
    assert.equal(getBuildSession(s.id, 'user').arrangement.plan.length, 4);
    const [v] = await take(s.id, 1, 0);
    assert.deepEqual(requests.at(-1)!.codec_ids, codes(s.id, b.id).slice(0, 200));
    assert.deepEqual([v.from, v.parent, v.prefix, v.starts[1]], [1, r.id, 200, 150]);
    keepBuildTake(s.id, 'user', v.id, 0);
    assert.equal(codes(s.id, v.id).length, 600);
    // A crop with no takes made from it goes when it is replaced.
    undoBuildKeep(s.id, 'user', undefined, 0);
    undoBuildKeep(s.id, 'user', undefined, 0);
    keepBuildTake(s.id, 'user', b.id, 0);
    const exact = cropBuildTake(s.id, 'user', b.id, 0, 300, undefined);
    const x = exact.takes.find(t => t.id === exact.kept.at(-1))!;
    assert.deepEqual([x.to, x.resume, x.frames], [1, undefined, 300]);
    const back = cropBuildTake(s.id, 'user', x.id, 0, null, undefined);
    assert.ok(!back.takes.some(t => t.id === x.id));
    assert.ok(!fs.existsSync(path.join(config.data.audioDir, 'takes', s.id, `${x.id}.codes.json`)));
  });

  await t.test('a section reopened by a crop and then changed cannot come back through undo or a later crop', async () => {
    const { cropBuildTake, applyBuildLyrics } = await import('./yue2Takes.js');
    const s = start();
    // Intro and verse in one take; the chorus would start at 300.
    const [both] = await take(s.id, 1, 0);
    keepBuildTake(s.id, 'user', both.id, 0);
    const reopened = cropBuildTake(s.id, 'user', both.id, 0, 200, 'regenerate');
    const crop = reopened.kept.at(-1)!;
    // Swap the reopened verse and the chorus.
    const plan = editBuildPlan(s.id, 'user', 0, ['o2', 'o1', 'o3'], true);
    assert.deepEqual(plan.arrangement.plan.map(o => o.id), ['o0', 'o2', 'o1', 'o3']);
    assert.throws(() => cropBuildTake(s.id, 'user', crop, 1, null, undefined), /no longer matches/);
    assert.throws(() => cropBuildTake(s.id, 'user', crop, 1, 250, 'advance'), /no longer matches/);
    assert.throws(() => cropBuildTake(s.id, 'user', crop, 1, 300, undefined), /no longer matches/);
    const after = getBuildSession(s.id, 'user');
    assert.equal(after.kept.at(-1), crop);
    assert.equal(after.takes.length, 2);
    // Cutting within what is built still works.
    const shorter = cropBuildTake(s.id, 'user', crop, 1, 180, 'regenerate');
    assert.deepEqual(shorter.takes.find(x => x.id === shorter.kept.at(-1))!.occurrences, ['o0']);

    // A lyric edit to the reopened verse also blocks it.
    const t2 = start();
    const [pair] = await take(t2.id, 1, 0);
    keepBuildTake(t2.id, 'user', pair.id, 0);
    const c2 = cropBuildTake(t2.id, 'user', pair.id, 0, 200, 'regenerate').kept.at(-1)!;
    const toRequest = (p: Record<string, unknown>) => ({ style: 'ambient', cot: 'full', abc: score, lyrics: String(p.lyrics) }) as Yue2SynthRequest;
    applyBuildLyrics(t2.id, 'user', 0, lyrics.replace('first line here', 'other words now'), toRequest);
    assert.throws(() => cropBuildTake(t2.id, 'user', c2, 1, null, undefined), /no longer matches/);
  });

  await t.test('crop waits for pending takes', async () => {
    const { cropBuildTake } = await import('./yue2Takes.js');
    const s = start();
    const [a] = await take(s.id, 0, 0);
    keepBuildTake(s.id, 'user', a.id, 0);
    hold = true; cancelled = false;
    generateBuildTakes(s.id, 'user', 1, 1, 0);
    await until(() => getBuildSession(s.id, 'user').pending[0]?.progress === 25);
    assert.throws(() => cropBuildTake(s.id, 'user', a.id, 0, 100, 'advance'), /pending/);
    await cancelBuildTakes(s.id, 'user');
    await settle(s.id);
    hold = false;
  });
});

test('crop plans: a cut on a frame inside the composed part, by occurrence', async () => {
  const { cropPlan, cropWav } = await import('./yue2Takes.js');
  const take = { from: 1, to: 2, frames: 400, starts: [0, 150, 250, 330], prefix: 150 };
  for (const bad of [150, 400, 200.5, '200']) assert.throws(() => cropPlan(take, bad, 'advance'), /Crop between/);
  assert.deepEqual(cropPlan(take, 250, undefined), { to: 1, starts: [0, 150, 250, null], resume: undefined, prefix: 150 });
  assert.throws(() => cropPlan(take, 200, undefined), /inside section 2/);
  assert.deepEqual(cropPlan(take, 200, 'advance'), { to: 1, starts: [0, 150, 200, null], resume: undefined, prefix: 150 });
  assert.deepEqual(cropPlan(take, 200, 'regenerate'), { to: 0, starts: [0, 150, null, null], resume: 200, prefix: 150 });
  // Past the next section's start: only Save song's tail changes.
  assert.deepEqual(cropPlan(take, 360, undefined), { to: 2, starts: [0, 150, 250, 330], resume: undefined, prefix: 150 });
  // 'regenerate' there keeps the run-on up to the cut; the next take carries on from it.
  assert.deepEqual(cropPlan(take, 360, 'regenerate'), { to: 2, starts: [0, 150, 250, 330], resume: 360, prefix: 150 });
  assert.equal(cropPlan(take, 330, 'regenerate').resume, undefined, 'a cut at its start keeps none of it');
  // The last occurrence: moving on keeps it, cut short.
  assert.deepEqual(cropPlan({ ...take, to: 3 }, 360, 'advance').to, 3);
  assert.deepEqual(cropPlan({ ...take, to: 3 }, 360, 'advance').starts.length, 4);

  const rate = 48000, frames = rate * 2;
  const wav = Buffer.alloc(44 + frames * 4);
  wav.write('RIFF', 0); wav.writeUInt32LE(36 + frames * 4, 4); wav.write('WAVE', 8);
  wav.write('fmt ', 12); wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(2, 22);
  wav.writeUInt32LE(rate, 24); wav.writeUInt32LE(rate * 4, 28); wav.writeUInt16LE(4, 32); wav.writeUInt16LE(16, 34);
  wav.write('data', 36); wav.writeUInt32LE(frames * 4, 40);
  const cut = cropWav(wav, 0.5);
  assert.equal(cut.length, 44 + rate * 2);
  assert.equal(cut.readUInt32LE(40), rate * 2);
  assert.equal(cut.readUInt32LE(4), cut.length - 8);
  assert.equal(cropWav(wav, 5).length, wav.length);
});

test('a trimmed start drops the first seconds and fades in from silence', async () => {
  const { trimWavStart } = await import('./yue2Takes.js');
  const rate = 48000, frames = rate * 2;
  const wav = Buffer.alloc(44 + frames * 4);
  wav.write('RIFF', 0); wav.writeUInt32LE(36 + frames * 4, 4); wav.write('WAVE', 8);
  wav.write('fmt ', 12); wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(2, 22);
  wav.writeUInt32LE(rate, 24); wav.writeUInt32LE(rate * 4, 28); wav.writeUInt16LE(4, 32); wav.writeUInt16LE(16, 34);
  wav.write('data', 36); wav.writeUInt32LE(frames * 4, 40);
  for (let f = 0; f < frames; f++) { wav.writeInt16LE(10000, 44 + f * 4); wav.writeInt16LE(-10000, 46 + f * 4); }
  const out = trimWavStart(wav, 0.5, 0.5);
  assert.equal(out.readUInt32LE(40), rate * 1.5 * 4);
  assert.equal(out.readUInt32LE(4), out.length - 8);
  assert.equal(out.readInt16LE(44), 0, 'silent at the cut');
  assert.equal(out.readInt16LE(46), 0);
  assert.equal(out.readInt16LE(44 + (rate / 4) * 4), 5000, 'half way up the fade');
  assert.equal(out.readInt16LE(46 + (rate / 4) * 4), -5000);
  assert.equal(out.readInt16LE(44 + rate * 4), 10000, 'full level after it');
});

test('a repeated section is timed by its own copy of the lyrics', () => {
  const composed = composeArrangement(score, lyrics, [1, 2, 2]);
  const own = composed.occurrences.map(o => o.lyric);
  assert.ok(own[1] && own[2] && own[2][0] > own[1][1] - 1);
  const chars = Array.from(composed.lyrics);
  const find = (text: string, from: number) => {
    for (let i = from; i < chars.length; i++) if (chars.slice(i, i + text.length).join('') === text) return i;
    throw new Error(text);
  };
  // The verse, then words sung only in the second copy of the chorus.
  const sung: Yue2AlignWord[] = [
    { char0: find('first', 0), char1: 0, start: 2, end: 2.4, score: 0 },
    { char0: find('hook', own[2]![0]), char1: 0, start: 14, end: 14.4, score: 0 },
  ];
  const starts = sectionStarts(own, sung, 30 * YUE2_FPS);
  // The first copy was not heard, so the second starts just before its words.
  assert.equal(starts[1], null);
  assert.equal(starts[2], Math.round((14 - 0.5) * YUE2_FPS));
  // Heard in the first copy only: the repeat is not given its time.
  const early = sectionStarts(own, [sung[0], { ...sung[1], char0: find('hook', own[1]![0]), start: 8, end: 8.4 }], 30 * YUE2_FPS);
  assert.equal(early[1], Math.round(((2.4 + 8) / 2) * YUE2_FPS));
  assert.equal(early[2], null);
});

test('cover settings: rebuilt into params and request before the first keep, locked after it', async t => {
  const { applyBuildSettings, editBuildPlan } = await import('./yue2Takes.js');
  const { coverScoreChoices, transformCoverScore } = await import('./backends/yue2/coverScoreTransform.js');
  const previousDir = config.data.dir;
  const dir = fs.mkdtempSync(path.resolve('.test-yue2-settings-'));
  config.data.dir = dir;
  t.after(() => { config.data.dir = previousDir; fs.rmSync(dir, { recursive: true, force: true }); });
  const requests: Yue2SynthRequest[] = [];
  let hold = false, cancelled = false;
  t.mock.method(globalThis, 'fetch', async (input: string | URL | Request, init?: RequestInit) => {
    const url = new URL(String(input));
    if (url.pathname === '/yue2/synth') {
      const req = JSON.parse(String(init?.body)) as Yue2SynthRequest;
      requests.push(req);
      return Response.json({ id: String(requests.length) });
    }
    if (url.pathname === '/yue2/align') return Response.json({ words: [] });
    if (url.searchParams.has('cancel')) { cancelled = true; return Response.json({}); }
    if (url.searchParams.has('result')) return new Response(new Uint8Array(pcm(1)), { headers: { 'content-type': 'audio/wav' } });
    if (hold) return Response.json({ status: cancelled ? 'cancelled' : 'running', phase: 'semantic', phase_step: 1, phase_total: 4 });
    const req = requests[Number(url.searchParams.get('id')) - 1];
    return Response.json({ status: 'done', tracks: [{ song: 0, seed: req.seed!, noise_seed: req.seed!, variation: 0,
      semantic_ids: [...(req.codec_ids ?? []), ...Array<number>(400).fill(1)], frames: 400 }] });
  });
  const until = async (condition: () => boolean) => {
    const deadline = Date.now() + 5000;
    while (!condition()) { assert.ok(Date.now() < deadline, 'finished within deadline'); await new Promise(r => setTimeout(r, 5)); }
  };
  // What the route's envelope does with these params, in miniature: the cover
  // choices rewrite the score, and CFG and AR strength reach the request.
  const toRequest = (p: Record<string, unknown>) => {
    const choices = coverScoreChoices(p.yue2Cover as never);
    return { style: 'ambient', cot: 'full', lyrics: String(p.lyrics), abc: transformCoverScore(String(p.yue2Abc), choices).renderedAbc,
      cfg_scale: p.yue2CfgScale, lm_adapter_scale: (p.yue2Pick as { lmAdapterArScale: number }).lmAdapterArScale } as Yue2SynthRequest;
  };
  const cover = { sourceId: 'src', voices: 'both', keepChords: true, tempo: 'source', key: 'source', cfgScale: 1 };
  const params = { lyrics, yue2Abc: score, yue2Cover: cover, yue2CfgScale: 1, yue2Pick: { lmAdapterAr: 'a.safetensors', lmAdapterArScale: 0.6 } };
  const start = () => createBuildSession(params, toRequest(params), 'user');
  const settle = (id: string) => until(() => !getBuildSession(id, 'user').pending.length);
  const settings = { voices: 'vocal', tempo: 90, key: 'D', keyShift: 2, cfgScale: 1.2, arStrength: 0 };

  await t.test('each setting reaches the stored params and the next take\'s request', async () => {
    const s = start();
    generateBuildTakes(s.id, 'user', 0, 1, 0);
    await settle(s.id);
    const old = getBuildSession(s.id, 'user').takes[0];
    const next = applyBuildSettings(s.id, 'user', 0, settings, toRequest);
    assert.equal(next.revision, 1);
    assert.deepEqual(next.params.yue2Cover, { ...cover, voices: 'vocal', tempo: 90, key: 'D', keyShift: 2, cfgScale: 1.2 });
    assert.equal(next.params.yue2CfgScale, 1.2);
    assert.deepEqual(next.params.yue2Pick, { lmAdapterAr: 'a.safetensors', lmAdapterArScale: 0 });
    assert.equal(next.params.yue2Abc, score, 'the approved score stays; the request carries the rewritten one');
    assert.throws(() => keepBuildTake(s.id, 'user', old.id, 1), /before the last change/);
    // The same settings again change nothing.
    assert.equal(applyBuildSettings(s.id, 'user', 1, settings, toRequest).revision, 1);
    generateBuildTakes(s.id, 'user', 0, 1, 1);
    await settle(s.id);
    const sent = requests.at(-1)!;
    assert.match(sent.abc!, /^Q:1\/8=90$/m);
    assert.match(sent.abc!, /^K:D$/m);
    assert.equal(sent.cfg_scale, 1.2);
    assert.equal((sent as unknown as Record<string, unknown>).lm_adapter_scale, 0);
    // Back to the source key: keyShift goes, as the Prepare page sends it.
    const back = applyBuildSettings(s.id, 'user', 1, { ...settings, key: 'source', keyShift: undefined }, toRequest);
    assert.equal((back.params.yue2Cover as Record<string, unknown>).keyShift, undefined);
  });

  await t.test('the plan survives a settings change', () => {
    const s = start();
    const planned = editBuildPlan(s.id, 'user', 0, ['s1', 's2', 's2', 's3'], true);
    const next = applyBuildSettings(s.id, 'user', 1, settings, toRequest);
    assert.deepEqual(next.arrangement.plan, planned.arrangement.plan);
  });

  await t.test('bad values, stale revisions, pending takes and kept takes are refused', async () => {
    const s = start();
    assert.throws(() => applyBuildSettings(s.id, 'user', 0, { ...settings, arStrength: 11 }, toRequest), /AR strength/);
    assert.throws(() => applyBuildSettings(s.id, 'user', 0, { ...settings, tempo: 900 }, toRequest), /tempo must/);
    assert.throws(() => applyBuildSettings(s.id, 'user', 3, settings, toRequest), /changed since/);
    assert.throws(() => applyBuildSettings(s.id, 'other', 0, settings, toRequest), /not found/);
    assert.equal(getBuildSession(s.id, 'user').revision, 0);
    hold = true;
    generateBuildTakes(s.id, 'user', 0, 1, 0);
    await until(() => getBuildSession(s.id, 'user').pending[0]?.progress === 25);
    assert.throws(() => applyBuildSettings(s.id, 'user', 0, settings, toRequest), /pending/);
    await cancelBuildTakes(s.id, 'user');
    await settle(s.id);
    hold = false;
    generateBuildTakes(s.id, 'user', 0, 1, 0);
    await settle(s.id);
    keepBuildTake(s.id, 'user', getBuildSession(s.id, 'user').takes.at(-1)!.id, 0);
    assert.throws(() => applyBuildSettings(s.id, 'user', 0, settings, toRequest), /fresh build/);
    assert.deepEqual(getBuildSession(s.id, 'user').params.yue2Cover, cover);
  });
});

test('a Create build (no cover block) takes CFG and AR strength before its first keep', async t => {
  const { applyBuildSettings } = await import('./yue2Takes.js');
  const previousDir = config.data.dir;
  const dir = fs.mkdtempSync(path.resolve('.test-yue2-create-settings-'));
  config.data.dir = dir;
  t.after(() => { config.data.dir = previousDir; fs.rmSync(dir, { recursive: true, force: true }); });
  const toRequest = (p: Record<string, unknown>) => ({ style: 'ambient', cot: 'full', lyrics: String(p.lyrics), abc: String(p.yue2Abc),
    cfg_scale: p.yue2CfgScale, lm_adapter_scale: (p.yue2Pick as { lmAdapterArScale: number }).lmAdapterArScale } as Yue2SynthRequest);
  const params = { lyrics, yue2Abc: score, yue2CfgScale: 1.5, yue2Pick: { lmAdapterAr: 'a.safetensors', lmAdapterArScale: 0.6 } };
  const s = createBuildSession(params, toRequest(params), 'user');
  const next = applyBuildSettings(s.id, 'user', 0, { cfgScale: 2, arStrength: 0.2 }, toRequest);
  assert.equal(next.revision, 1);
  assert.equal(next.params.yue2CfgScale, 2);
  assert.deepEqual(next.params.yue2Pick, { lmAdapterAr: 'a.safetensors', lmAdapterArScale: 0.2 });
  assert.equal(next.params.yue2Cover, undefined, 'no cover block appears');
  assert.equal(applyBuildSettings(s.id, 'user', 1, { cfgScale: 2, arStrength: 0.2 }, toRequest).revision, 1, 'same settings change nothing');
  assert.throws(() => applyBuildSettings(s.id, 'user', 1, { cfgScale: 0, arStrength: 0.2 }, toRequest), /Condition strength/);
  assert.throws(() => applyBuildSettings(s.id, 'user', 1, { cfgScale: 2, arStrength: 11 }, toRequest), /AR strength/);
});

test('a build from a Prepare document checks it once and keeps the confirmed pair', async () => {
  const { structureHash } = await import('../contracts/yue2Prepare.js');
  const hash = structureHash(score);
  const prepare = { version: 2, sourceKey: 'src', abc: score, sectionIds: ['s1', 's2', 's3', 's4'], paste: '',
    blocks: [{ id: 'b1', text: 'first line here', tag: 'Verse' }, { id: 'b2', text: 'hook line now', tag: 'Chorus' }],
    assignments: { s1: { kind: 'instrumental' }, s2: { kind: 'words', blockIds: ['b1'] }, s3: { kind: 'words', blockIds: ['b2'] }, s4: { kind: 'instrumental' } },
    excluded: [], lyricRevision: 2, flagged: [], nextId: 7,
    confirmed: { structure: { structureHash: hash }, lyrics: { structureHash: hash, lyricRevision: 2 } } };
  const request = (text: string): Yue2SynthRequest => ({ style: 'ambient', cot: 'off', abc: score, lyrics: text });
  const s = createBuildSession({ yue2Abc: score, lyrics, yue2Prepare: prepare }, request(lyrics), 'user');
  assert.deepEqual(s.prepared, { structureHash: hash, lyricRevision: 2 });
  assert.equal(s.params.yue2Prepare, undefined, 'the session keeps the pair, not the document');
  assert.throws(() => createBuildSession({ yue2Abc: score, lyrics: '[verse]\nother', yue2Prepare: prepare }, request('[verse]\nother'), 'user'),
    /not the confirmed Prepare lyrics/);
  assert.throws(() => createBuildSession({ yue2Abc: score, lyrics, yue2Prepare: { ...prepare, lyricRevision: 3 } }, request(lyrics), 'user'),
    /Confirm the lyrics/);
  assert.throws(() => createBuildSession({ yue2Abc: score.replace('E6|', 'G6|'), lyrics, yue2Prepare: prepare }, request(lyrics), 'user'),
    /score changed/);
  // Instrumental mode must agree with the map, either way.
  assert.throws(() => createBuildSession({ yue2Abc: score, lyrics: '', instrumental: true, yue2Prepare: prepare }, request(''), 'user'),
    /have words; turn off Instrumental/);
  const quiet = { ...prepare, blocks: [], assignments: Object.fromEntries(prepare.sectionIds.map(id => [id, { kind: 'instrumental' }])) };
  assert.throws(() => createBuildSession({ yue2Abc: score, lyrics, yue2Prepare: quiet }, request(lyrics), 'user'), /turn on Instrumental/);
  assert.ok(createBuildSession({ yue2Abc: score, lyrics: '', instrumental: true, yue2Prepare: quiet }, request(''), 'user').prepared);
});
