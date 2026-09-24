// Render-based coherence judge: whisper the render, match the transcript to
// the lyric, split by song position. Late-song planner decay (loops, vocals
// vanish, smear) should show as the last third's match rate and word rate
// falling away from the first third's. Rob's idea (2026-09-24).
//
//   npx tsx server/scripts/yue2-coherence-judge.ts <study dir> [--only <group>]
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
const doc = JSON.parse(fs.readFileSync(path.join(study, 'study.json'), 'utf8')) as { tracks: Array<{ id: string; group: string; file: string; label: string }> };
const judgePath = path.join(study, 'judge.json');
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
  const r = await transcribeWithWhisper(wav, '', { language: 'en', beamSize: 5 });
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
