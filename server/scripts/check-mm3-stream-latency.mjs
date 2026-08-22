// check-mm3-stream-latency.mjs — when does the first audio actually arrive?
//
// Three cases, and the difference between them is the whole feature:
//
//   INTERLEAVED, fresh plan  — the planner and the flow stage take turns, so the
//                              first window lands a few seconds in. Needs both
//                              model stacks co-resident (engine decides).
//   SERIAL, fresh plan       — the fallback when they will not co-reside: no
//                              window can exist until the planner is done, so
//                              first audio == the whole AR stage.
//   AR cache HIT             — stage 1 never runs at all; the first window is
//                              one flow pass away.
//
// It also prints the sustained rate, which is what decides whether playback
// stutters: audio must be produced faster than it is consumed.
//
// Usage: node server/scripts/check-mm3-stream-latency.mjs [engineUrl]

const ENGINE = process.argv[2] || 'http://127.0.0.1:8085';
const CAPTION = [
  'Global Metadata: genre: synthwave; mood: driving; bpm: 110; key: A minor.',
  'Vocal Details: instrumental.',
  'Arrangement: steady analog bass, gated pads, simple four-on-the-floor drums.',
].join('\n');

const DURATION = 60;
const STEPS = 30;
const sleep = ms => new Promise(r => setTimeout(r, ms));

async function run({ stream, reuse_ar, seed, label }) {
  const t0 = Date.now();
  const res = await fetch(`${ENGINE}/mm3/synth`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      caption: CAPTION, lyrics: '', duration: DURATION, seed,
      steps: STEPS, get_wav_bits: 16, get_lrc: false, reuse_ar, stream,
    }),
  });
  if (!res.ok) throw new Error(`submit ${res.status}: ${await res.text()}`);
  const sub = await res.json();

  let firstAudio = 0, chunks = 0, audioSec = 0;
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
        pending = Buffer.concat([pending, Buffer.from(value)]);
        for (;;) {
          if (pending.length < 44 || pending.toString('ascii', 0, 4) !== 'RIFF') break;
          const total = pending.readUInt32LE(4) + 8;
          if (pending.length < total) break;
          const secs = pending.readUInt32LE(40) / 4 / pending.readUInt32LE(24);
          chunks++; audioSec += secs;
          const at = (Date.now() - t0) / 1000;
          if (!firstAudio) firstAudio = at;
          // The margin is what matters once playing: audio in hand minus audio
          // consumed. Negative means the player has caught the renderer.
          const margin = audioSec - (at - firstAudio);
          console.log(`      window ${String(chunks).padStart(2)} at +${at.toFixed(1)}s  ` +
                      `(+${secs.toFixed(2)}s audio, ${audioSec.toFixed(1)}s total, buffer ${margin >= 0 ? '+' : ''}${margin.toFixed(1)}s)`);
          pending = pending.subarray(total);
        }
      }
    })();
  }

  for (;;) {
    const j = await (await fetch(`${ENGINE}/job?id=${sub.job_id}`)).json();
    if (j.status === 'done') break;
    if (j.status === 'failed' || j.status === 'cancelled') throw new Error(`job ${j.status}`);
    await sleep(400);
  }
  await readerDone;
  const total = (Date.now() - t0) / 1000;
  const d = await (await fetch(`${ENGINE}/mm3/job?id=${sub.job_id}`)).json();
  const dur = d.result?.duration_sec ?? 0;
  console.log(`  ${label}: total ${total.toFixed(1)}s` +
    (stream ? `, FIRST AUDIO at ${firstAudio.toFixed(1)}s` : '') +
    `  [interleaved=${d.stream_interleaved}, ar_cached=${d.ar_cached}, ` +
    `ar ${Math.round(d.result?.ms?.ar ?? 0)}ms, flow ${Math.round(d.result?.ms?.flow ?? 0)}ms]`);
  console.log(`      ${dur.toFixed(1)}s of audio in ${total.toFixed(1)}s = ${(dur / total).toFixed(2)}x realtime`);
  return { total, firstAudio, detail: d };
}

(async () => {
  console.log(`\n${DURATION} s clip, ${STEPS} steps — time to first audio\n`);

  console.log('  [1/2] fresh plan, streaming (interleaved if the stacks co-reside)');
  const fresh = await run({ stream: true, reuse_ar: true, seed: 31337, label: 'fresh plan' });
  if (fresh.detail.stream_interleaved !== true) {
    console.log('\n  NOTE: the engine declined co-residency, so this measured the SERIAL fallback.');
  }

  console.log('\n  [2/2] same request again — the AR cache should hit, so no planner at all');
  const hit = await run({ stream: true, reuse_ar: true, seed: 31337, label: 'AR cache hit' });
  if (hit.detail.ar_cached !== true) console.log('\n  NOTE: the AR cache did NOT hit.');

  console.log(`\n  first audio: ${fresh.firstAudio.toFixed(1)}s on a fresh plan, ` +
              `${hit.firstAudio.toFixed(1)}s on a cache hit`);
})().catch(e => { console.error('ERROR:', e.message); process.exit(1); });
