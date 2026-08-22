// mm3-stream-node.mjs — end-to-end through the Node tier, plus cancel.
//
// The engine test proves the transport. This proves the parts only the server
// owns: the mm3Stream param mapping, the mm3_streaming flag on /status, the
// proxy at GET /api/generate/mm3/stream/:id, and that cancelling mid-stream
// closes the body instead of hanging the reader.

const NODE = 'http://127.0.0.1:3001';
const CAPTION = [
  'Global Metadata: genre: synthwave; mood: driving; bpm: 110; key: A minor.',
  'Vocal Details: instrumental.',
  'Arrangement: steady analog bass, gated pads, simple four-on-the-floor drums.',
].join('\n');

const sleep = ms => new Promise(r => setTimeout(r, ms));
const results = [];
const check = (name, ok, detail) => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
};

const token = (await (await fetch(`${NODE}/api/auth/auto`)).json()).token;

async function submit(extra = {}) {
  const r = await fetch(`${NODE}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({
      backend: 'minimax-m3', prompt: CAPTION, lyrics: '', instrumental: true,
      duration: 60, seed: 777, randomSeed: false,
      mm3Steps: 30, mm3ReuseAr: true, mm3Stream: true, skipLrc: true,
      ...extra,
    }),
  });
  if (!r.ok) throw new Error(`submit ${r.status}: ${await r.text()}`);
  return (await r.json()).jobId;
}

const status = async id => (await (await fetch(`${NODE}/api/generate/status/${id}`, {
  headers: { Authorization: `Bearer ${token}` },
})).json());

async function waitStreaming(id, timeoutMs = 120000) {
  const t0 = Date.now();
  for (;;) {
    const s = await status(id);
    if (s.mm3_streaming === true) return s;
    if (['succeeded', 'failed', 'cancelled'].includes(s.status)) return s;
    if (Date.now() - t0 > timeoutMs) return s;
    await sleep(500);
  }
}

// ── 1. Normal streamed render through Node ──
console.log('\n=== 1: /api/generate with mm3Stream, streamed through the Node proxy ===');
const t0 = Date.now();
const id1 = await submit();
console.log(`  job ${id1}`);
const s1 = await waitStreaming(id1);
check('/status reports mm3_streaming', s1.mm3_streaming === true, `status=${s1.status}`);
// The card's orange progress is (audio received / this). Without it the UI can
// only show a spinner, so it is a contract, not a nicety.
check('/status reports mm3_duration before any audio', typeof s1.mm3_duration === 'number' && s1.mm3_duration > 0,
  `mm3_duration=${s1.mm3_duration}`);


const res1 = await fetch(`${NODE}/api/generate/mm3/stream/${id1}`);
check('Node proxy serves the stream', res1.ok, `HTTP ${res1.status}`);
let bytes1 = 0, chunks1 = 0, first1 = 0;
if (res1.ok) {
  const rd = res1.body.getReader();
  let pending = Buffer.alloc(0);
  for (;;) {
    const { done, value } = await rd.read();
    if (done) break;
    if (!first1) first1 = Date.now();
    bytes1 += value.length;
    pending = Buffer.concat([pending, Buffer.from(value)]);
    for (;;) {
      if (pending.length < 44 || pending.toString('ascii', 0, 4) !== 'RIFF') break;
      const total = pending.readUInt32LE(4) + 8;
      if (pending.length < total) break;
      chunks1++; pending = pending.subarray(total);
    }
  }
  console.log(`  ${chunks1} windows, ${(bytes1 / 1048576).toFixed(1)} MB, first audio at ${((first1 - t0) / 1000).toFixed(1)}s`);
}
check('the proxied stream carries whole windows', chunks1 > 1 && bytes1 > 0, `${chunks1} windows`);
// Tri-state ON PURPOSE, and asserted only here: it is decided on the GPU worker
// thread after a VRAM check, so it is legitimately null between submit and the
// job reaching the front of the queue. The UI renders that null as "undecided"
// rather than as "serial", which is why the field is nullable rather than a
// boolean with a default.
const s1b = await status(id1);
check('/status settles mm3_interleaved once the job runs',
  s1b.mm3_interleaved === true || s1b.mm3_interleaved === false,
  `mm3_interleaved=${s1b.mm3_interleaved}`);

// The render must still finish and save normally — streaming is additive.
for (;;) {
  const s = await status(id1);
  if (s.status === 'succeeded') {
    check('the streamed render still saves a song', !!s.result?.audioUrls?.[0], s.result?.audioUrls?.[0] ?? 'no url');
    break;
  }
  if (['failed', 'cancelled'].includes(s.status)) { check('the streamed render still saves a song', false, s.status); break; }
  await sleep(1000);
}

// ── 2. Cancel mid-stream ──
console.log('\n=== 2: cancel mid-stream ===');
const id2 = await submit({ seed: 778 });
console.log(`  job ${id2}`);
const s2 = await waitStreaming(id2);
check('second job is streaming', s2.mm3_streaming === true, `status=${s2.status}`);

const res2 = await fetch(`${NODE}/api/generate/mm3/stream/${id2}`);
check('stream opened', res2.ok, `HTTP ${res2.status}`);
let chunks2 = 0, closedCleanly = false;
if (res2.ok) {
  const rd = res2.body.getReader();
  let pending = Buffer.alloc(0);
  let cancelled = false;
  const tStart = Date.now();
  for (;;) {
    const { done, value } = await rd.read();
    if (done) { closedCleanly = true; break; }
    pending = Buffer.concat([pending, Buffer.from(value)]);
    for (;;) {
      if (pending.length < 44 || pending.toString('ascii', 0, 4) !== 'RIFF') break;
      const total = pending.readUInt32LE(4) + 8;
      if (pending.length < total) break;
      chunks2++; pending = pending.subarray(total);
    }
    if (chunks2 >= 2 && !cancelled) {
      cancelled = true;
      console.log(`  cancelling after ${chunks2} windows (+${((Date.now() - tStart) / 1000).toFixed(1)}s)`);
      await fetch(`${NODE}/api/generate/cancel/${id2}`, { method: 'POST', headers: { Authorization: `Bearer ${token}` } });
    }
  }
}
check('cancel closes the stream body (no hang)', closedCleanly, `${chunks2} windows received`);
const s2f = await status(id2);
check('the cancelled job ends cancelled/failed', ['cancelled', 'failed'].includes(s2f.status), s2f.status);

const failed = results.filter(r => !r).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
