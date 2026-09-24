// Render-based coherence judge: whisper the render, match the transcript to
// the lyric, split by song position. Late-song planner decay (loops, vocals
// vanish, smear) should show as the last third's match rate and word rate
// falling away from the first third's. Rob's idea (2026-09-24).
//
//   npx tsx server/scripts/yue2-coherence-judge.ts <study dir> [--only <group>] [--stems]
//
// --stems: whisper the SuperSep vocal stem (engine level 4, cached beside the
// track) instead of the mix, and write judge-stems.json. Rob's suggestion:
// whisper on the mix hears nothing on loud tracks.
//
// Reads <study>/study.json (tracks with file + group), the group's
// config.json (lyrics), writes <study>/judge.json {trackId: metrics} and, if
// scores.json exists, prints each track's metrics beside Rob's scores plus
// rank correlations. Resumable: judged tracks are skipped.
import fs from 'node:fs';
import path from 'node:path';
import { transcribeWithWhisper, stripSectionMarkers } from '../src/services/whisperTranscribe.js';

const study = process.argv[2];
const onlyIx = process.argv.indexOf('--only');
const only = onlyIx > 0 ? process.argv[onlyIx + 1] : '';
const useStems = process.argv.includes('--stems');
const ACE_URL = 'http://127.0.0.1:8085';

/** The vocal stem of `wav` through the engine's SuperSep (vocals-only level),
 *  cached at `<dir>/stems/<name>.vocals.wav`. Mirrors yue2Stems.separateOne. */
async function vocalStem(wav: string): Promise<string> {
  const out = path.join(path.dirname(wav), 'stems', path.basename(wav, '.wav') + '.vocals.wav');
  if (fs.existsSync(out) && fs.statSync(out).size > 0) return out;
  const started = await fetch(`${ACE_URL}/supersep/separate?level=4`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: fs.readFileSync(wav) });
  if (!started.ok) throw new Error(`separate: ${await started.text()}`);
  const { id } = await started.json() as { id: string };
  try {
    for (let i = 0; i < 2400; i++) {
      const p = await (await fetch(`${ACE_URL}/supersep/progress?id=${id}`)).json() as { status: string; error?: string };
      if (p.status === 'done') break;
      if (p.status === 'failed' || p.status === 'cancelled') throw new Error(p.error || `separation ${p.status}`);
      await new Promise(r => setTimeout(r, 500));
    }
    const { stems } = await (await fetch(`${ACE_URL}/supersep/result?id=${id}`)).json() as { stems: Array<{ name: string; index: number }> };
    const vocal = stems.find(s => ['vocals', 'vocal', 'lead vocals', 'lead_vocals'].includes(s.name.trim().toLowerCase()));
    fs.mkdirSync(path.dirname(out), { recursive: true });
    if (!vocal) { fs.writeFileSync(out, ''); return out; }  // no singing found: an empty marker
    const got = await fetch(`${ACE_URL}/supersep/serve?id=${id}&stem=${vocal.index}`);
    if (!got.ok) throw new Error(`serve: stem ${vocal.index}`);
    fs.writeFileSync(out, Buffer.from(await got.arrayBuffer()));
    return out;
  } finally {
    await fetch(`${ACE_URL}/supersep/release?id=${id}`, { method: 'POST' }).catch(() => {});
  }
}
const doc = JSON.parse(fs.readFileSync(path.join(study, 'study.json'), 'utf8')) as { tracks: Array<{ id: string; group: string; file: string; label: string }> };
const judgePath = path.join(study, useStems ? 'judge-stems.json' : 'judge.json');
const judged: Record<string, Metrics> = fs.existsSync(judgePath) ? JSON.parse(fs.readFileSync(judgePath, 'utf8')) : {};

interface Metrics { seconds: number; words: number; match: number; rate: number; thirds: Array<{ match: number; rate: number }>; lateMatch: number; lateRate: number }

const norm = (s: string) => s.toLowerCase().replace(/[^a-z0-9' ]+/g, ' ').split(/\s+/).filter(w => w.length >= 3);
const wavSeconds = (p: string): number => {
  const b = fs.readFileSync(p);
  // RIFF header: byte rate at 28, data size after the 'data' chunk id.
  const byteRate = b.readUInt32LE(28);
  const i = b.indexOf('data', 12, 'latin1');
  return i > 0 && byteRate > 0 ? b.readUInt32LE(i + 4) / byteRate : 0;
};

const lyricsFor: Record<string, Set<string>> = {};
for (const t of doc.tracks) {
  if (only && t.group !== only) continue;
  if (judged[t.id]) continue;
  const wav = path.join(study, t.file);
  if (!fs.existsSync(wav)) { console.log(`${t.id}: missing ${t.file}`); continue; }
  if (!lyricsFor[t.group]) {
    const cfg = JSON.parse(fs.readFileSync(path.join(study, t.group, 'config.json'), 'utf8')) as { lyrics: string };
    lyricsFor[t.group] = new Set(norm(stripSectionMarkers(cfg.lyrics)));
  }
  const vocab = lyricsFor[t.group];
  const seconds = wavSeconds(wav);
  const t0 = Date.now();
  let source = wav;
  if (useStems) {
    try { source = await vocalStem(wav); } catch (err) { console.log(`${t.id}: stems failed: ${(err as Error).message}`); continue; }
  }
  const r = source && fs.statSync(source).size > 0 ? await transcribeWithWhisper(source, '', { language: 'en', beamSize: 5 }) : null;
  const words = (r?.segments ?? []).flatMap(s => s.words?.length ? s.words : [{ word: s.text, start: s.start, end: s.end, probability: 1 }])
    .flatMap(w => norm(w.word).map(x => ({ w: x, t: w.start })));
  const thirds = [0, 1, 2].map(k => {
    const a = seconds * k / 3, b = seconds * (k + 1) / 3;
    const ws = words.filter(x => x.t >= a && x.t < b);
    return { match: ws.length ? ws.filter(x => vocab.has(x.w)).length / ws.length : 0, rate: ws.length / Math.max(1, b - a) };
  });
  const m: Metrics = { seconds, words: words.length, match: words.length ? words.filter(x => vocab.has(x.w)).length / words.length : 0,
    rate: words.length / Math.max(1, seconds), thirds, lateMatch: thirds[2].match - thirds[0].match, lateRate: thirds[2].rate - thirds[0].rate };
  judged[t.id] = m;
  fs.writeFileSync(judgePath, JSON.stringify(judged, null, 1));
  console.log(`${t.id.padEnd(30)} ${Math.round(seconds)}s ${String(m.words).padStart(4)} words  match ${m.match.toFixed(2)}  thirds ${thirds.map(x => x.match.toFixed(2)).join('/')}  rate ${thirds.map(x => x.rate.toFixed(2)).join('/')}  (${Math.round((Date.now() - t0) / 1000)} s)`);
}

const scoresPath = path.join(study, 'scores.json');
if (fs.existsSync(scoresPath)) {
  const scores = (JSON.parse(fs.readFileSync(scoresPath, 'utf8')) as { scores: Record<string, Record<string, number>> }).scores;
  const rows = Object.entries(judged).filter(([id]) => scores[id] && Number.isFinite(scores[id].coherence)).map(([id, m]) => ({ id, m, s: scores[id] }));
  const rank = (a: number[]) => { const s = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]); const r: number[] = []; s.forEach(([, i], k) => { r[i] = k; }); return r; };
  const spearman = (x: number[], y: number[]) => { const rx = rank(x), ry = rank(y), n = x.length, mx = (n - 1) / 2; let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < n; i++) { sxy += (rx[i] - mx) * (ry[i] - mx); sxx += (rx[i] - mx) ** 2; syy += (ry[i] - mx) ** 2; } return sxy / Math.sqrt(sxx * syy); };
  console.log(`\nSpearman over ${rows.length} scored tracks (metric vs ear):`);
  for (const crit of ['coherence', 'diction']) {
    const y = rows.map(r => r.s[crit] ?? NaN);
    if (!y.every(Number.isFinite)) continue;
    console.log(`  ${crit}: match ${spearman(rows.map(r => r.m.match), y).toFixed(2)}  match3 ${spearman(rows.map(r => r.m.thirds[2].match), y).toFixed(2)}  lateMatch ${spearman(rows.map(r => r.m.lateMatch), y).toFixed(2)}  rate3 ${spearman(rows.map(r => r.m.thirds[2].rate), y).toFixed(2)}  lateRate ${spearman(rows.map(r => r.m.lateRate), y).toFixed(2)}  words ${spearman(rows.map(r => r.m.words), y).toFixed(2)}`);
  }
}
