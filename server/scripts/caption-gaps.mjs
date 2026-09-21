#!/usr/bin/env node
// caption-gaps.mjs — fill missing captions across EVERY Training Studio dataset.
//
// For each dataset, finds tracks missing lyrics, an AS1.5 caption, a `.mm3.txt`
// or a `.yue2.txt`, and drives the app's own routes to fill them, in order:
//   1. no lyrics (and not instrumental) → POST /enhance/genius (fill_missing)
//   2. missing AS caption or .mm3.txt   → POST /enhance/caption (writes all three;
//      mergePolicy fill_missing, so an existing AS caption is never rewritten)
//   3. missing only .yue2.txt           → POST /enhance/yue2-caption (cheap text call)
//   4. POST /lyric-studio               → album (re)exported to Lyric Studio
// Datasets run one after another; each job is polled to completion. Re-running
// is safe — anything already on disk is skipped, so a killed run just resumes.
//
//   node server/scripts/caption-gaps.mjs                 # dry run: gap counts only
//   node server/scripts/caption-gaps.mjs --apply         # Gemini (default provider)
//   node server/scripts/caption-gaps.mjs --apply --provider moss
//   --model <id>   --yue2-provider <id>   --only <substring>   --limit <n>
//
// The app must be running (dev.bat / LAUNCH.bat). Port from HOTSTEP_PORT or 3001.

import fs from 'fs';
import path from 'path';

const argv = process.argv.slice(2);
const APPLY = argv.includes('--apply');
const opt = (n, d) => { const i = argv.indexOf(`--${n}`); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : d; };
const BASE = `http://127.0.0.1:${process.env.HOTSTEP_PORT || 3001}/api/training`;
const PROVIDER = opt('provider', 'gemini');
const MODEL = opt('model', '');
const YUE2_PROVIDER = opt('yue2-provider', PROVIDER === 'moss' ? 'gemini' : PROVIDER);
const ONLY = opt('only', '').toLowerCase();
const LIMIT = parseInt(opt('limit', '0'), 10) || 0;

const sleep = ms => new Promise(r => setTimeout(r, ms));
async function api(method, url, body) {
  const res = await fetch(BASE + url, {
    method, headers: { 'content-type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let json; try { json = JSON.parse(text); } catch { json = { error: text.slice(0, 200) }; }
  return { status: res.status, json };
}

/** Start a job, waiting out a 409 (something else owns the dataset), then poll it to the end. */
async function runJob(url, body, label) {
  let start;
  for (;;) {
    start = await api('POST', url, body);
    if (start.status !== 409) break;
    process.stdout.write(`    ${label}: dataset busy (${start.json.error}) — waiting\n`);
    await sleep(30_000);
  }
  if (start.status !== 202) throw new Error(`${label}: HTTP ${start.status} ${start.json.error || ''}`);
  const id = start.json.jobId;
  const t0 = Date.now();
  for (;;) {
    await sleep(5_000);
    const { json: job } = await api('GET', `/jobs/${id}`);
    if (!job || job.error === 'Job not found') throw new Error(`${label}: job ${id} vanished (cancelled?)`);
    if (['done', 'failed', 'cancelled'].includes(job.status)) {
      const mins = ((Date.now() - t0) / 60000).toFixed(1);
      process.stdout.write(`    ${label}: ${job.status} — ${job.done}/${job.total} done, ${job.failed} failed, ${mins} min\n`);
      return job;
    }
  }
}

const stemOf = p => p.replace(/\.[^.\\/]+$/, '');
const { json: list } = await api('GET', '/datasets');
let datasets = (list.datasets || []).filter(d => !ONLY || d.slug.toLowerCase().includes(ONLY));
datasets.sort((a, b) => a.slug.localeCompare(b.slug));
if (LIMIT) datasets = datasets.slice(0, LIMIT);

const totals = { datasets: 0, tracks: 0, needLyrics: 0, needCaption: 0, needYue2Only: 0, exported: 0, failed: 0 };
for (const ds of datasets) {
  const { status, json: detail } = await api('GET', `/datasets/${ds.id}`);
  if (status !== 200) { console.log(`${ds.slug}: cannot read (${detail.error})`); continue; }
  const samples = (detail.samples || []).filter(s => !s.excluded && !s.fileMissing);
  const captionIds = [], yue2Ids = [], geniusIds = [];
  for (const s of samples) {
    const stem = stemOf(s.audioPath);
    const hasMm3 = fs.existsSync(`${stem}.mm3.txt`);
    const hasYue2 = fs.existsSync(`${stem}.yue2.txt`);
    if (!String(s.lyrics || '').trim() && !s.isInstrumental) geniusIds.push(s.sampleId);
    if (!hasMm3 || !String(s.caption || '').trim()) captionIds.push(s.sampleId);
    else if (!hasYue2) yue2Ids.push(s.sampleId);
  }
  totals.tracks += samples.length;
  const needExport = !ds.lyricsSetId;
  if (!captionIds.length && !yue2Ids.length && !geniusIds.length && !needExport) continue;
  totals.datasets++;
  totals.needCaption += captionIds.length;
  totals.needYue2Only += yue2Ids.length;
  totals.needLyrics += geniusIds.length;
  console.log(`${ds.slug}: ${samples.length} tracks — ${geniusIds.length} need lyrics, ${captionIds.length} need captions, `
    + `${yue2Ids.length} need only .yue2.txt${needExport ? ', not in Lyric Studio' : ''}`);
  if (!APPLY) continue;

  try {
    // Lyrics first: the caption prompt carries an excerpt of them.
    if (geniusIds.length) {
      const job = await runJob(`/datasets/${ds.id}/enhance/genius`, {
        sampleIds: geniusIds, mergePolicy: 'fill_missing',
      }, 'genius').catch(err => { console.log(`    ${err.message}`); return { failed: 0 }; });
      totals.failed += job.failed;
    }
    if (captionIds.length) {
      const job = await runJob(`/datasets/${ds.id}/enhance/caption`, {
        sampleIds: captionIds, provider: PROVIDER, ...(MODEL ? { model: MODEL } : {}),
        mergePolicy: 'fill_missing', yue2Provider: YUE2_PROVIDER,
      }, 'caption');
      totals.failed += job.failed;
    }
    // Tracks the caption pass just handled got their .yue2.txt in the same
    // pass; re-check the disk so a failed one is retried here.
    const retry = samples.filter(s => !fs.existsSync(`${stemOf(s.audioPath)}.yue2.txt`)
      && fs.existsSync(`${stemOf(s.audioPath)}.mm3.txt`)).map(s => s.sampleId);
    if (retry.length) {
      const job = await runJob(`/datasets/${ds.id}/enhance/yue2-caption`, {
        sampleIds: retry, provider: YUE2_PROVIDER,
      }, 'yue2');
      totals.failed += job.failed;
    }
    // Lyric Studio export is an upsert (existing set reused, songs replaced),
    // so re-sending an album after new lyrics or captions is the right call.
    const exp = await api('POST', `/datasets/${ds.id}/lyric-studio`, {});
    if (exp.status === 200) { totals.exported++; process.stdout.write(`    lyric studio: exported (set ${exp.json.lyricsSetId ?? '?'})\n`); }
    else { totals.failed++; console.log(`    lyric studio: HTTP ${exp.status} ${exp.json.error || ''}`); }
  } catch (err) {
    totals.failed++;
    console.log(`    ${ds.slug}: ${err.message}`);
  }
}
console.log(`\n${APPLY ? 'done' : 'dry run'}: ${totals.datasets} datasets with gaps, ${totals.tracks} tracks scanned, `
  + `${totals.needLyrics} need lyrics, ${totals.needCaption} need captions, ${totals.needYue2Only} need only .yue2.txt, `
  + `${totals.exported} exported to Lyric Studio, ${totals.failed} failures`);
