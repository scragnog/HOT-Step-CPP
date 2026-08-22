// mm3-stream-test.mjs — the validation bar for MM3 streaming.
//
// Two claims, both measurable, neither requiring ears:
//
//   1. NO REGRESSION WHEN OFF. Same seed, stream:false vs stream:true must
//      produce a BYTE-IDENTICAL WAV. Streaming moves the vocoder inside the
//      flow loop; if that changed a single sample, this catches it.
//   2. GAPLESS. The concatenated PCM of every streamed chunk must be
//      BYTE-IDENTICAL to the saved WAV's PCM. Any seam is an offset bug, and
//      it is measurable, so it gets measured rather than listened for.
//
// Usage: node mm3-stream-test.mjs [engineUrl]

const ENGINE = process.argv[2] || 'http://127.0.0.1:8085';

const CAPTION = [
  'Global Metadata: genre: synthwave; mood: driving; bpm: 110; key: A minor.',
  'Vocal Details: instrumental.',
  'Arrangement: steady analog bass, gated pads, simple four-on-the-floor drums.',
].join('\n');

const DURATION = 24;   // long enough for several windows, short enough to iterate
const SEED = 4242;
const STEPS = 8;

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function submit(stream) {
  const res = await fetch(`${ENGINE}/mm3/synth`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      caption: CAPTION, lyrics: '', duration: DURATION, seed: SEED,
      steps: STEPS, get_wav_bits: 16, get_lrc: false, reuse_ar: false, stream,
    }),
  });
  if (!res.ok) throw new Error(`submit failed ${res.status}: ${await res.text()}`);
  return res.json();
}

async function waitDone(id) {
  for (;;) {
    const r = await fetch(`${ENGINE}/job?id=${id}`);
    const j = await r.json();
    if (j.status === 'done') return j;
    if (j.status === 'failed' || j.status === 'cancelled') {
      const d = await (await fetch(`${ENGINE}/mm3/job?id=${id}`)).json();
      throw new Error(`job ${j.status}: ${d.error ?? ''}`);
    }
    await sleep(1000);
  }
}

async function getWav(id) {
  const r = await fetch(`${ENGINE}/job?id=${id}&result=1`);
  if (!r.ok) throw new Error(`result failed ${r.status}`);
  return Buffer.from(await r.arrayBuffer());
}

/** Split a stream of concatenated RIFF files into their PCM data payloads. */
function splitWavs(buf) {
  const out = [];
  let off = 0;
  while (off + 44 <= buf.length) {
    if (buf.toString('ascii', off, off + 4) !== 'RIFF') throw new Error(`no RIFF at ${off}`);
    const total = buf.readUInt32LE(off + 4) + 8;
    const dataSize = buf.readUInt32LE(off + 40);
    out.push({
      rate: buf.readUInt32LE(off + 24),
      header: buf.subarray(off, off + 44),
      data: buf.subarray(off + 44, off + 44 + dataSize),
    });
    off += total;
  }
  if (off !== buf.length) throw new Error(`trailing ${buf.length - off} bytes after the last WAV`);
  return out;
}

function firstDiff(a, b) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i;
  return a.length === b.length ? -1 : n;
}

const results = [];
const check = (name, ok, detail) => {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
};

(async () => {
  // ── Run A: streaming OFF (today's path) ──
  console.log('\n=== A: stream=false ===');
  const a0 = Date.now();
  const a = await submit(false);
  console.log(`  submitted ${a.job_id}, streaming=${a.streaming}, frames=${a.max_frames}`);
  check('A reports streaming:false', a.streaming === false, `got ${a.streaming}`);
  await waitDone(a.job_id);
  const wavA = await getWav(a.job_id);
  console.log(`  done in ${((Date.now() - a0) / 1000).toFixed(1)}s, ${wavA.length} bytes`);

  // ── Run B: streaming ON, same seed ──
  console.log('\n=== B: stream=true ===');
  const b0 = Date.now();
  const b = await submit(true);
  console.log(`  submitted ${b.job_id}, streaming=${b.streaming}`);
  check('B reports streaming:true', b.streaming === true, `got ${b.streaming}`);

  // Attach the reader straight away, exactly as the Node route does.
  const sres = await fetch(`${ENGINE}/mm3/stream?id=${b.job_id}`);
  check('GET /mm3/stream accepted', sres.ok, `HTTP ${sres.status}`);
  if (!sres.ok) { console.error(await sres.text()); process.exit(1); }

  const parts = [];
  let firstChunkAt = 0;
  const reader = sres.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!firstChunkAt) firstChunkAt = Date.now();
    parts.push(Buffer.from(value));
  }
  const streamed = Buffer.concat(parts);
  const totalMs = Date.now() - b0;
  console.log(`  stream closed: ${streamed.length} bytes; first byte at ${((firstChunkAt - b0) / 1000).toFixed(1)}s of ${(totalMs / 1000).toFixed(1)}s total`);

  await waitDone(b.job_id);
  const wavB = await getWav(b.job_id);

  // ── Claim 1: the restructure is numerically neutral ──
  const dAB = firstDiff(wavA, wavB);
  check('stream=false and stream=true are byte-identical', dAB === -1,
    dAB === -1 ? `${wavA.length} bytes` : `first difference at byte ${dAB} (A=${wavA.length}, B=${wavB.length})`);

  // ── Claim 2: the streamed concatenation IS the saved file ──
  const chunks = splitWavs(streamed);
  const pcm = Buffer.concat(chunks.map(c => c.data));
  const savedPcm = wavB.subarray(44, 44 + wavB.readUInt32LE(40));
  const dPCM = firstDiff(pcm, savedPcm);
  console.log(`  ${chunks.length} chunk(s), rates ${[...new Set(chunks.map(c => c.rate))].join('/')}, ` +
              `${chunks.map(c => c.data.length / 4).join(' + ')} frames`);
  check('streamed PCM === saved WAV PCM', dPCM === -1,
    dPCM === -1 ? `${pcm.length} bytes` : `first difference at byte ${dPCM} (streamed=${pcm.length}, saved=${savedPcm.length})`);
  check('every chunk carries the stream sample rate', new Set(chunks.map(c => c.rate)).size === 1,
    `${[...new Set(chunks.map(c => c.rate))].join(',')}`);

  // ── A second reader must be refused, not silently served half a song ──
  const dup = await fetch(`${ENGINE}/mm3/stream?id=${b.job_id}`);
  check('a finished job refuses a new reader', dup.status === 409, `HTTP ${dup.status}`);

  // ── A non-streaming job must say so rather than hang ──
  const off = await fetch(`${ENGINE}/mm3/stream?id=${a.job_id}`);
  check('a non-streaming job refuses the stream', off.status === 409, `HTTP ${off.status}`);

  const failed = results.filter(r => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error('\nERROR:', e.message); process.exit(1); });
