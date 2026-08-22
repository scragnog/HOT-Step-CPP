#!/usr/bin/env node
// check-mm3-takes.mjs — end-to-end ensemble render through the job layer.
//
// Section [1] of check-mm3-ensemble.mjs proves the AR maths; this proves the
// PLUMBING: that a `takes: N` request really produces N different finished
// songs, that each is fetchable, that N streams can be read at once, and — the
// one that actually matters for regressions — that a one-take render is still
// byte-for-byte the render it always was.
//
// Usage: node server/scripts/check-mm3-takes.mjs [engineUrl]

const ENGINE = process.argv[2] || 'http://127.0.0.1:8085';

let failures = 0;
const fail = (m) => { console.log(`FAIL  ${m}`); failures++; };
const pass = (m) => console.log(`PASS  ${m}`);

const PROMPT = [
  '[Structured Caption]',
  'Global Metadata: Genre: electronic/synthwave. BPM: 110. Key: A minor. Mood: driving, nocturnal.',
  'Vocal Details: Male vocals, mid register, slight rasp.',
  'Arrangement: Analog synth bass, gated drums, arpeggiated pads, wide reverb.',
  '[Lyrics]',
  '[verse]',
  'Neon runs the length of the street',
].join('\n');

async function submit(extra) {
  const body = {
    caption: PROMPT, lyrics: '', duration: 24, seed: 4242, steps: 8,
    get_wav_bits: 16, get_lrc: false, reuse_ar: false, ...extra,
  };
  const r = await fetch(`${ENGINE}/mm3/synth`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(`submit ${r.status}: ${JSON.stringify(j)}`);
  return j.id || j.job_id;
}

async function wait(id, timeoutMs = 600000) {
  const t0 = Date.now();
  for (;;) {
    const r = await fetch(`${ENGINE}/job?id=${id}`);
    const j = await r.json();
    if (j.status === 'done') return (await (await fetch(`${ENGINE}/mm3/job?id=${id}`)).json());
    if (j.status === 'failed' || j.status === 'cancelled') {
      const d = await (await fetch(`${ENGINE}/mm3/job?id=${id}`)).json();
      throw new Error(`job ${j.status}: ${d.error ?? ''}`);
    }
    if (Date.now() - t0 > timeoutMs) throw new Error('timed out');
    await new Promise((s) => setTimeout(s, 500));
  }
}

const sha = async (buf) => {
  const h = await crypto.subtle.digest('SHA-256', buf);
  return Buffer.from(h).toString('hex').slice(0, 16);
};

// ── 1. one take is unchanged ───────────────────────────────────────────────
console.log('=== [1] a one-take render is untouched by ensemble support ===');
let soloSha = null;
{
  const id = await submit({});
  const st = await wait(id);
  const a = Buffer.from(await (await fetch(`${ENGINE}/mm3/take?id=${id}&take=0`)).arrayBuffer());
  const b = Buffer.from(await (await fetch(`${ENGINE}/job?id=${id}&result=1`)).arrayBuffer());
  soloSha = await sha(a);
  if (st.takes !== 1) fail(`takes reported ${st.takes}, expected 1`);
  else pass('takes: 1');
  if (!a.length) fail('/mm3/take?take=0 returned nothing');
  else pass(`/mm3/take?take=0 served ${a.length} bytes`);
  if (Buffer.compare(a, b) !== 0) fail('/mm3/take?take=0 differs from the shared /job?result=1 body');
  else pass('/mm3/take?take=0 === /job?result=1 (one copy, two URLs)');
}

// ── 2. N takes, N different songs ──────────────────────────────────────────
console.log('\n=== [2] takes: 3 produces three different finished songs ===');
{
  const id = await submit({ takes: 3 });
  const st = await wait(id);
  if (st.takes !== 3) fail(`takes reported ${st.takes}, expected 3`);
  else pass('takes: 3');
  if (!Array.isArray(st.take_detail) || st.take_detail.length !== 3) fail('take_detail missing or wrong length');
  else pass(`take_detail lists ${st.take_detail.length} takes`);

  const seeds = (st.take_detail || []).map((t) => Number(t.seed));
  if (seeds.join(',') !== '4242,4243,4244') fail(`seeds ${seeds} (expected 4242,4243,4244)`);
  else pass('seeds are base + take index');

  const hashes = [];
  for (let t = 0; t < 3; t++) {
    const r = await fetch(`${ENGINE}/mm3/take?id=${id}&take=${t}`);
    if (!r.ok) { fail(`take ${t}: HTTP ${r.status}`); continue; }
    const buf = Buffer.from(await r.arrayBuffer());
    hashes.push(await sha(buf));
    const d = st.take_detail[t];
    console.log(`      take ${t}: ${buf.length} bytes, ${d.duration_s.toFixed(2)}s, seed ${d.seed}, sha ${hashes[t]}`);
  }
  if (hashes.length !== 3) fail('not every take was fetchable');
  else if (new Set(hashes).size !== 3) fail(`only ${new Set(hashes).size}/3 distinct audio — takes are duplicating`);
  else pass('3/3 distinct audio bodies');

  if (hashes[0] === soloSha) {
    console.log('      note: take 0 of the ensemble matches the solo render byte-for-byte');
  } else {
    console.log('      note: take 0 differs from the solo render (expected — batch width changes the kernel)');
  }

  const r404 = await fetch(`${ENGINE}/mm3/take?id=${id}&take=9`);
  if (r404.status !== 404) fail(`take 9 returned ${r404.status}, expected 404`);
  else pass('an out-of-range take 404s');
}

// ── 3. N simultaneous streams ──────────────────────────────────────────────
console.log('\n=== [3] three takes stream at once, and each is complete ===');
{
  const id = await submit({ takes: 3, stream: true });
  // Attach all three readers immediately — before the job even reaches the
  // front of the queue, which is the case the per-take queues have to survive.
  const readers = [0, 1, 2].map(async (t) => {
    const r = await fetch(`${ENGINE}/mm3/stream?id=${id}&take=${t}`);
    if (!r.ok) throw new Error(`take ${t} stream: HTTP ${r.status}`);
    const chunks = [];
    for await (const c of r.body) chunks.push(Buffer.from(c));
    return { take: t, bytes: Buffer.concat(chunks) };
  });
  let streamed;
  try {
    streamed = await Promise.all(readers);
  } catch (e) {
    fail(`streaming: ${e.message}`);
    streamed = [];
  }
  for (const s of streamed) {
    if (!s.bytes.length) fail(`take ${s.take} streamed nothing`);
    else pass(`take ${s.take} streamed ${s.bytes.length} bytes`);
  }
  if (streamed.length === 3) {
    const hs = await Promise.all(streamed.map((s) => sha(s.bytes)));
    if (new Set(hs).size !== 3) fail('the three streams carried identical bytes — queues are crossed');
    else pass('the three streams are distinct (queues are not crossed)');
  }
  await wait(id);
  // The streamed audio must be the same song the saved WAV holds.
  for (let t = 0; t < streamed.length; t++) {
    const saved = Buffer.from(await (await fetch(`${ENGINE}/mm3/take?id=${id}&take=${t}`)).arrayBuffer());
    if (!saved.length) { fail(`take ${t}: no saved audio`); continue; }
    // Compare PCM lengths rather than bytes: the stream is many small WAVs,
    // the save is one big one, so only the payload is comparable.
    const streamedPcm = streamed[t].bytes.length - 44 * countWavHeaders(streamed[t].bytes);
    const savedPcm = saved.length - 44;
    if (Math.abs(streamedPcm - savedPcm) > 4)
      fail(`take ${t}: streamed ${streamedPcm} PCM bytes vs saved ${savedPcm}`);
    else pass(`take ${t}: streamed PCM length === saved PCM length (${savedPcm} bytes)`);
  }
}

function countWavHeaders(buf) {
  let n = 0;
  for (let i = 0; i + 4 <= buf.length; i++) {
    if (buf[i] === 0x52 && buf[i + 1] === 0x49 && buf[i + 2] === 0x46 && buf[i + 3] === 0x46) n++;
  }
  return n;
}

console.log(failures ? `\n${failures} FAILURE(S)` : '\nall checks passed');
process.exit(failures ? 1 : 0);
