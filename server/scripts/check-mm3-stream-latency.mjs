// mm3-stream-latency.mjs — where does the first audio actually arrive?
//
// The phase-1 pipeline dispatches windows only once the AR stage is done, so
// time-to-first-audio == AR time. Two cases matter and they are very different:
//
//   AR cache MISS — the planner runs; first audio lands near the end.
//   AR cache HIT  — stage 1 is skipped outright (mm3-ar-cache.h), so the first
//                   window is one flow pass away. This is the app's DEFAULT
//                   (mm3ReuseAr defaults ON), so it is the case users see.
//
// It also exercises the plan's first trap: an AR-cache hit means there is no
// live AR loop, and window dispatch must not assume one.

const ENGINE = process.argv[2] || 'http://127.0.0.1:8085';
const CAPTION = [
  'Global Metadata: genre: synthwave; mood: driving; bpm: 110; key: A minor.',
  'Vocal Details: instrumental.',
  'Arrangement: steady analog bass, gated pads, simple four-on-the-floor drums.',
].join('\n');

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function run({ stream, reuse_ar, steps, duration, seed, label }) {
  const t0 = Date.now();
  const res = await fetch(`${ENGINE}/mm3/synth`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ caption: CAPTION, lyrics: '', duration, seed, steps, get_wav_bits: 16, get_lrc: false, reuse_ar, stream }),
  });
  if (!res.ok) throw new Error(`submit ${res.status}: ${await res.text()}`);
  const sub = await res.json();

  let firstByte = 0, bytes = 0, chunks = 0;
  let readerDone = Promise.resolve();
  if (stream) {
    const sres = await fetch(`${ENGINE}/mm3/stream?id=${sub.job_id}`);
    if (!sres.ok) throw new Error(`stream ${sres.status}: ${await sres.text()}`);
    const rd = sres.body.getReader();
    readerDone = (async () => {
      let pending = Buffer.alloc(0);
      for (;;) {
        const { done, value } = await rd.read();
        if (done) break;
        if (!firstByte) firstByte = Date.now();
        bytes += value.length;
        pending = Buffer.concat([pending, Buffer.from(value)]);
        // count complete RIFFs as they land
        for (;;) {
          if (pending.length < 44 || pending.toString('ascii', 0, 4) !== 'RIFF') break;
          const total = pending.readUInt32LE(4) + 8;
          if (pending.length < total) break;
          chunks++;
          const at = ((Date.now() - t0) / 1000).toFixed(1);
          console.log(`      window ${chunks} at +${at}s (${(pending.readUInt32LE(40) / 4 / 44100).toFixed(2)}s of audio)`);
          pending = pending.subarray(total);
        }
      }
    })();
  }

  for (;;) {
    const j = await (await fetch(`${ENGINE}/job?id=${sub.job_id}`)).json();
    if (j.status === 'done') break;
    if (j.status === 'failed' || j.status === 'cancelled') throw new Error(`job ${j.status}`);
    await sleep(500);
  }
  await readerDone;
  const total = (Date.now() - t0) / 1000;
  const d = await (await fetch(`${ENGINE}/mm3/job?id=${sub.job_id}`)).json();

  console.log(`  ${label}: total ${total.toFixed(1)}s` +
    (stream ? `, FIRST AUDIO at ${((firstByte - t0) / 1000).toFixed(1)}s (${chunks} windows, ${(bytes / 1048576).toFixed(1)} MB)` : '') +
    `  [ar_cached=${d.ar_cached}, ar ${Math.round(d.result?.ms?.ar ?? 0)}ms, flow ${Math.round(d.result?.ms?.flow ?? 0)}ms, voc ${Math.round(d.result?.ms?.voc ?? 0)}ms]`);
  return { total, firstAudio: (firstByte - t0) / 1000, detail: d };
}

(async () => {
  const cfg = { duration: 60, seed: 777, steps: 30 };
  console.log(`\n60 s clip, ${cfg.steps} steps, q8_0 — time to first audio\n`);

  console.log('  [1/2] priming the AR cache (stream off)…');
  await run({ ...cfg, stream: false, reuse_ar: true, label: 'prime (AR miss)' });

  console.log('\n  [2/2] same request, streaming on — the AR cache should hit');
  const hit = await run({ ...cfg, stream: true, reuse_ar: true, label: 'stream + AR HIT' });

  if (hit.detail.ar_cached !== true) {
    console.log('\n  NOTE: the AR cache did NOT hit, so this measured the miss path.');
  }
  console.log(`\n  Playback needs audio faster than realtime; this render produced ` +
              `${(hit.detail.result?.duration_sec ?? 0).toFixed(1)}s of audio in ${hit.total.toFixed(1)}s ` +
              `(${((hit.detail.result?.duration_sec ?? 0) / hit.total).toFixed(2)}x realtime).`);
})().catch(e => { console.error('ERROR:', e.message); process.exit(1); });
