// resolve-parity.mjs — does the server-side resolver produce what the browser
// produces today?
//
//   npx tsx scripts/resolve-parity.mjs [--fixtures <dir>]
//
// Two checks, both read-only:
//
// 1. Differential. The Node ports in src/services/generation/resolve are run
//    against the real UI modules (ui/src/utils/*, loaded under a localStorage
//    shim) on generated inputs covering every fallback branch, and must agree
//    exactly.
// 2. Browser fixtures. Every fixture the capture index marks useForCoverage is
//    paired with a resolution: written songs are resolved from the live song
//    row, album preset and caption sources and compared field by field; Create
//    fixtures are checked for the caption source pick, wildcard expansion and
//    duration rule that produced them.
//
// The database is opened through a temporary copy, so nothing the loaders
// touch (the album caption sync in getLyricsSet) can write to the real one.
// Prints a report and exits 1 on any mismatch.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const serverDir = path.resolve(here, '..');
const argDir = process.argv.indexOf('--fixtures');
const fixtureDir = argDir > 0 ? path.resolve(process.argv[argDir + 1]) : path.join(serverDir, 'data', 'dev-captures', 'generate');
const indexFile = path.join(fixtureDir, 'INDEX_BATCH1_2026_10_07.json');

// Point the server modules at a copy of the database before importing them.
const realData = path.join(serverDir, 'data');
const tmpData = fs.mkdtempSync(path.join(os.tmpdir(), 'resolve-parity-'));
fs.copyFileSync(path.join(realData, 'hotstep.db'), path.join(tmpData, 'hotstep.db'));
process.env.DATA_DIR = tmpData;
process.env.TRAINING_DIR = process.env.TRAINING_DIR || path.join(realData, 'training');

const store = new Map();
globalThis.localStorage = {
  getItem: k => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: k => store.delete(k),
  clear: () => store.clear(),
};

const ui = {
  caption: await import('../../ui/src/utils/captionForBackend.ts'),
  mm3: await import('../../ui/src/utils/mm3CaptionSource.ts'),
  yue2: await import('../../ui/src/utils/yue2CaptionSource.ts'),
  wild: await import('../../ui/src/utils/wildcardUtils.ts'),
  dur: await import('../../ui/src/utils/estimateDuration.ts'),
  key: await import('../../ui/src/utils/keyScale.ts'),
};
const node = {
  src: await import('../src/services/generation/resolve/captionSource.ts'),
  content: await import('../src/services/generation/resolve/content.ts'),
  intent: await import('../src/services/generation/resolve/resolveIntent.ts'),
  load: await import('../src/services/generation/resolve/loadIntentData.ts'),
  keyScale: (await import('../src/services/lireek/prompts.ts')).normalizeKeyScale,
  db: await import('../src/db/lireekDb.ts'),
  database: await import('../src/db/database.ts'),
  ds: await import('../src/services/training/yue2DatasetCaptions.ts'),
};

node.database.initDb();

const failures = [];
let checks = 0;
const same = (label, a, b) => {
  checks++;
  const ja = JSON.stringify(a);
  const jb = JSON.stringify(b);
  if (ja !== jb) failures.push(`${label}\n    browser: ${jb}\n    node:    ${ja}`);
};

/** The whole effective body, key by key. Documented exceptions:
 *  - `expectedBackend`: the resolver pins it to the engine (the browser sends
 *    none), so the body cannot run on another engine after a switch.
 *  - keys in `nondeterministic`: a fresh random draw on each side, checked by
 *    the caller as valid expansions instead. */
const exceptionsSeen = { expectedBackend: 0, nondeterministic: 0 };
function compareBody(label, resolved, captured, engine, nondeterministic = new Set()) {
  for (const k of new Set([...Object.keys(resolved), ...Object.keys(captured)])) {
    if (k === 'expectedBackend' && !(k in captured)) {
      exceptionsSeen.expectedBackend++;
      same(`${label}: expectedBackend pinned`, resolved[k], engine);
      continue;
    }
    if (nondeterministic.has(k)) { exceptionsSeen.nondeterministic++; continue; }
    // JSON drops undefined, so a key the resolver set to undefined equals an
    // absent one on the wire.
    same(`${label}: ${k}`, resolved[k] === undefined ? null : resolved[k], captured[k] === undefined ? null : captured[k]);
  }
}

// ── 1. Differential ─────────────────────────────────────────────────────────

let rngState = 12345;
const rnd = () => { rngState = (rngState * 1103515245 + 12345) % 2147483648; return rngState / 2147483648; };
const pick = arr => arr[Math.floor(rnd() * arr.length)];

const templates = ['plain', '{a|b|c}', 'x {a|{b|c}} y {d|e}', '{|}', '{one}', 'a {b| c |d} {e|f|g|h} [Verse]\n{la|da}', '{a|b}{c|d}{e|f}'];
for (const t of templates) {
  for (let i = 0; i < 200; i++) {
    const seed = pick([0, 1, 42, 2147483647, 4294967295, 4294967296, 9007199254740991, Math.floor(rnd() * 1e15)]);
    const slot = pick([0, 0, 1, 7]);
    same(`expandWildcards(${JSON.stringify(t)}, ${seed}, ${slot})`, node.content.expandWildcards(t, seed, slot), ui.wild.expandWildcards(t, seed, slot));
  }
  same(`hasWildcards(${JSON.stringify(t)})`, node.content.hasWildcards(t), ui.wild.hasWildcards(t));
}

const lyricSamples = ['', 'one line', '[Verse]\nl1\nl2\n\n[Chorus]\nc1', Array(120).fill('line').join('\n'), '[Intro]\n[Verse]\n' + Array(30).fill('x').join('\n')];
for (const ly of lyricSamples) {
  for (const bpm of [0, 20, 40, 90, 128, 300]) {
    same(`estimateDuration(${ly.length} chars, ${bpm})`, node.content.estimateDuration(ly, bpm), ui.dur.estimateDuration(ly, bpm));
    for (const llm of [undefined, 0, -5, 175]) {
      for (const useLlm of [true, false]) {
        localStorage.setItem(ui.dur.LLM_DURATION_KEY, JSON.stringify(useLlm));
        same(`resolveDuration(${llm}, ${ly.length} chars, ${bpm}, useLlm=${useLlm})`,
          node.content.resolveDuration(llm, ly, bpm, useLlm).value, ui.dur.resolveDuration(llm, ly, bpm));
      }
    }
  }
}
localStorage.removeItem(ui.dur.LLM_DURATION_KEY);
for (const k of ['E Major', 'c# minor', 'Bb major', 'F♯ Minor', 'nonsense', '', null, '  A  minor ']) {
  same(`normalizeKeyScale(${JSON.stringify(k)})`, node.keyScale(k), ui.key.normalizeKeyScale(k));
}

const mm3Tracks = [
  { title: 'A', bpm: 90, caption: 'cap A' }, { title: 'B', bpm: 120, caption: 'cap B' },
  { title: 'C', bpm: 120, caption: 'cap C' }, { title: 'D', caption: 'cap D' },
];
const yue2Tracks = [
  { name: 'a', caption: 'raw a', bpm: '~100', styled: 'sty a' }, { name: 'b', caption: 'raw b', bpm: '128.5' },
  { name: 'c', caption: 'raw c', bpm: 128.5, styled: 'sty c' }, { name: 'd', caption: '', styled: '' }, { name: 'e', caption: 'raw e' },
];
const mm3Sels = [undefined, { mode: 'auto' }, { mode: 'custom' }, { mode: 'track', selectedTitle: 'C' }, { mode: 'track', selectedTitle: 'Gone' }, { mode: 'track' }, { mode: 'bad' }];
const yue2Sels = [undefined, { mode: 'auto' }, { mode: 'custom' }, { mode: 'track', selectedName: 'e' }, { mode: 'track', selectedName: 'd' }, { mode: 'track', selectedName: 'gone' }, { mode: 'bad' }];
const songs = [
  { id: 11, bpm: 128, caption: ' ace ', caption_mm3: 'own mm3', caption_yue2: 'own yue2' },
  { id: 12, bpm: 119, caption: 'ace', caption_mm3: '', caption_yue2: '' },
  { id: 13, bpm: 0, caption: '', caption_mm3: '  ', caption_yue2: null },
  { id: 14, caption: 'ace only' },
];
const LYRICS_SET = 5;
const DATASET = 'ds-parity';
ui.mm3.cacheMm3SourceTracks(LYRICS_SET, mm3Tracks);
ui.yue2.cacheYue2SourceTracks(DATASET, yue2Tracks);

for (const gen of songs) {
  for (const engine of ['ace', 'minimax-m3', 'yue2']) {
    for (const withTracks of [true, false]) {
      for (const linked of [true, false]) {
        for (const renderingAs of [false, true]) {
          for (const adapterInForce of [true, false]) {
            for (const mm3Sel of engine === 'minimax-m3' ? mm3Sels : [undefined]) {
              for (const yueSel of engine === 'yue2' ? yue2Sels : [undefined]) {
                // Browser state for this case.
                store.clear();
                ui.mm3.cacheMm3SourceTracks(LYRICS_SET, withTracks ? mm3Tracks : []);
                ui.yue2.cacheYue2SourceTracks(DATASET, withTracks ? yue2Tracks : []);
                if (linked) localStorage.setItem(`hs-yue2DatasetForLyricsSet:${LYRICS_SET}`, JSON.stringify(DATASET));
                if (mm3Sel) localStorage.setItem(`hs-mm3CaptionSource:${gen.id}`, JSON.stringify(mm3Sel));
                if (yueSel) localStorage.setItem(`hs-yue2CaptionSource:ds:song:${gen.id}:${DATASET}`, JSON.stringify(yueSel));
                const browser = ui.caption.captionForBackend(gen, engine, LYRICS_SET, adapterInForce, renderingAs);
                // The same state as explicit inputs.
                const datasetId = linked ? DATASET : '';
                const r = node.src.captionForEngine(gen, engine, {
                  mm3: { tracks: withTracks ? mm3Tracks : [], selection: node.src.normalizeMm3Selection(mm3Sel) },
                  yue2: { datasetId, tracks: withTracks ? yue2Tracks : [], selection: node.src.effectiveYue2Selection(datasetId, yueSel, adapterInForce) },
                  renderingAs,
                });
                same(`captionForBackend(song ${gen.id}, ${engine}, tracks=${withTracks}, linked=${linked}, as=${renderingAs}, adapter=${adapterInForce}, mm3=${JSON.stringify(mm3Sel)}, yue2=${JSON.stringify(yueSel)})`,
                  r.caption, browser);
              }
            }
          }
        }
      }
    }
  }
}
store.clear();

// Create's own resolvers (no storage): resolveMm3Caption / resolveYue2Caption.
for (const bpm of [0, 90, 100, 119, 121, 128, 200]) {
  for (const own of ['', '  mine ']) {
    for (const sel of mm3Sels.filter(Boolean).filter(s => s.mode !== 'bad')) {
      for (const tracks of [mm3Tracks, []]) {
        same(`resolveMm3Caption(${bpm}, ${JSON.stringify(own)}, ${JSON.stringify(sel)}, ${tracks.length})`,
          node.src.resolveMm3Caption({ bpm, caption_mm3: own }, tracks, sel).caption, ui.mm3.resolveMm3Caption({ bpm, caption_mm3: own }, tracks, sel).caption);
      }
    }
    for (const sel of yue2Sels.filter(Boolean).filter(s => s.mode !== 'bad')) {
      for (const tracks of [yue2Tracks, []]) {
        const n = node.src.resolveYue2Caption(own, bpm, tracks, sel);
        const b = ui.yue2.resolveYue2Caption(own, bpm, tracks, sel);
        same(`resolveYue2Caption(${bpm}, ${JSON.stringify(own)}, ${JSON.stringify(sel)}, ${tracks.length})`, [n.caption, n.mode, n.fromTrack], [b.caption, b.mode, b.fromName]);
      }
    }
  }
}
const differentialChecks = checks;

// ── 2. Browser fixtures ─────────────────────────────────────────────────────

const report = [];
if (!fs.existsSync(indexFile)) {
  report.push(`fixtures: none at ${indexFile} (skipped)`);
} else {
  const index = JSON.parse(fs.readFileSync(indexFile, 'utf8'));
  const fixtures = index.fixtures.filter(f => f.useForCoverage);
  const sqlite = (await import('better-sqlite3')).default;
  const db = new sqlite(path.join(tmpData, 'hotstep.db'), { readonly: true });
  const presets = db.prepare('SELECT * FROM album_presets').all();
  const datasetCache = new Map();
  const datasetsFor = async () => {
    if (datasetCache.size) return datasetCache;
    for (const row of db.prepare('SELECT id FROM training_datasets').all()) {
      const ds = await node.ds.yue2DatasetCaptions({ dataset: row.id });
      if (ds.tracks.length) datasetCache.set(row.id, ds.tracks);
    }
    return datasetCache;
  };
  const albumMm3 = new Map();
  for (const row of db.prepare('SELECT id, songs FROM lyrics_sets').all()) {
    let songsJson = [];
    try { songsJson = JSON.parse(row.songs); } catch { /* skip */ }
    const t = node.src.collectMm3SourceTracks(songsJson);
    if (t.length) albumMm3.set(row.id, t);
  }

  for (const f of fixtures) {
    const fx = JSON.parse(fs.readFileSync(path.join(fixtureDir, f.file), 'utf8'));
    const body = fx.body;
    const engine = f.engine;
    const label = `[${f.engine}/${f.caller.source}] ${f.branch}`;
    const before = failures.length;
    try {
      if (f.caller.source === 'lyric-studio') {
        const gen = db.prepare('SELECT * FROM generations WHERE title = ? AND lyrics = ? ORDER BY id DESC').get(body.title, body.lyrics);
        if (!gen) { failures.push(`${label}: generation not found by title+lyrics`); continue; }
        const own = db.prepare('SELECT p.lyrics_set_id AS id FROM profiles p WHERE p.id = ?').get(gen.profile_id)?.id;
        // The album the song rendered as: the preset whose adapter the body carries.
        const byAdapter = presets.find(p => p.adapter_path && p.adapter_path === body.loraPath)
          ?? presets.find(p => body.yue2Pick && p.yue2_ar_adapter_path && p.yue2_ar_adapter_path === body.yue2Pick.lmAdapterAr);
        const target = /Render-as/.test(f.branch) ? (byAdapter?.lyrics_set_id ?? own) : own;
        // Snapshot = the submitted body; the resolver rewrites what the queue rewrites.
        const intent = {
          kind: 'written-song', engine, generationId: gen.id, lyricsSetId: target,
          ...(target !== own ? { sourceLyricsSetId: own } : {}), artistName: body.artist,
          params: { ...body }, yue2Pick: body.yue2Pick,
          settings: {
            useLlmDuration: true, useLmAdapter: typeof body.lmAdapter === 'string' && !!body.lmAdapter,
            triggerUseFilename: Array.isArray(body.triggerWords) && body.triggerWords.length > 0,
            triggerPlacement: body.triggerPlacement, randomizeTimbreRef: body.randomizeTimbreRef === true,
          },
        };
        const data = await node.load.loadWrittenSongData(intent, engine);
        // The capture round set these song captions temporarily and restored
        // them to empty afterwards (index findings), so the live row no longer
        // has them: put the captured one back for this resolution only.
        const savedOwn = /saved (song-specific|Custom) caption/.test(f.branch);
        if (savedOwn && engine === 'yue2') data.gen = { ...data.gen, caption_yue2: body.caption };
        if (savedOwn && engine === 'minimax-m3') data.gen = { ...data.gen, caption_mm3: body.caption };
        // Caption choice: whichever the captured caption shows was in force.
        if (savedOwn && engine === 'minimax-m3') intent.mm3Selection = { mode: 'custom' };
        else if (savedOwn && engine === 'yue2') {
          // "overrides selected Track": a Track pick is in force and must lose.
          const first = data.yue2Dataset?.tracks?.[0];
          intent.yue2Selection = first ? { mode: 'track', selectedName: first.name } : { mode: 'custom' };
        } else if (engine === 'minimax-m3') {
          const hit = (data.mm3Tracks ?? []).find(t => t.caption === body.caption);
          intent.mm3Selection = hit ? { mode: 'track', selectedTitle: hit.title } : { mode: 'custom' };
        }
        if (engine === 'yue2' && !savedOwn) {
          const hit = (data.yue2Dataset?.tracks ?? []).find(t => node.src.yue2TrackCaption(t) === body.caption);
          intent.yue2Selection = hit ? { mode: 'track', selectedName: hit.name } : { mode: 'custom' };
        }
        const r = node.intent.resolveWrittenSongIntent(intent, engine, data);
        compareBody(label, r.request, body, engine);
        // Auto branches: the pick the resolver makes unprompted must be the one captured.
        if (/Auto/.test(f.branch)) {
          const auto = node.intent.resolveWrittenSongIntent({ ...intent, mm3Selection: undefined, yue2Selection: undefined }, engine, data);
          same(`${label}: caption with the default (auto) selection`, auto.request.caption, body.caption);
        }
        report.push(`${failures.length === before ? 'PASS' : 'FAIL'} ${label} -> ${savedOwn ? 'captured song caption restored; ' : ''}gen ${gen.id}, album ${target}${target !== own ? ` (as, own ${own})` : ''}, caption ${r.provenance.caption.source}${r.provenance.caption.fromTrack ? ` "${r.provenance.caption.fromTrack}"` : ''}`);
      } else {
        // Create: rebuild the intent the panel would send, run the real loader
        // and resolver, compare the whole body.
        //
        // What the capture did not record is reconstructed, and each case says
        // which:
        //   typed caption  for a Custom box, the captured caption with the
        //                  compose trigger and beat tail removed; for a wildcard
        //                  case, the recorded template (index.supplement)
        //   compose        trigger "fixture-trigger" when the caption starts
        //                  with it; the beat tail and its bar count when present;
        //                  autoExpand for the expanded cases
        //   caption source the dataset or album whose track the caption is,
        //                  found by search; mode from the branch
        const notes = [];
        const trig = body.caption.startsWith('fixture-trigger, ') ? 'fixture-trigger' : '';
        const beat = / with a clean (\d+)-bar percussive intro and outro for DJ mixing$/.exec(body.caption);
        let core = trig ? body.caption.slice(trig.length + 2) : body.caption;
        if (beat) core = core.slice(0, core.length - beat[0].length - 1);
        const expanded = /expanded/.test(f.branch);
        const [capT, lyrT] = (index.supplement?.wildcardInputs ?? []).map(s => s.replace(/\\n/g, '\n'));
        const intent = {
          kind: 'create', engine,
          params: { ...body, caption: expanded ? capT : core, lyrics: expanded ? lyrT : body.lyrics },
          compose: { autoExpand: expanded, loraTrigger: trig, beatIntro: !!beat, ...(beat ? { introBars: Number(beat[1]) } : {}) },
        };
        notes.push(expanded ? 'typed=template' : 'typed=caption minus compose');
        if (engine === 'yue2' && /Auto|Track/.test(f.branch)) {
          const all = await datasetsFor();
          const hits = [...all].flatMap(([id, tracks]) => tracks.filter(t => node.src.yue2TrackCaption(t) === core).map(t => ({ id, t })));
          const mode = /Auto/.test(f.branch) ? 'auto' : 'track';
          // For auto, the dataset whose tempo pick is this track (several albums
          // can share a sidecar text).
          const hit = mode === 'auto'
            ? hits.find(h => node.src.resolveYue2Caption('', body.bpm, all.get(h.id), { mode: 'auto' }).caption === core)
            : hits[0];
          if (!hit) { failures.push(`${label}: no dataset ${mode === 'auto' ? 'auto-picks' : 'carries'} the captured caption`); report.push(`FAIL ${label}`); continue; }
          intent.captionSource = { engine: 'yue2', datasetId: hit.id, adapterInForce: true,
            selection: mode === 'auto' ? { mode: 'auto' } : { mode: 'track', selectedName: hit.t.name } };
          if (!expanded) { intent.params.caption = ''; notes[0] = 'typed=empty (box locked to the source)'; }
          notes.push(`yue2 ${mode} source by search`);
        }
        if (engine === 'minimax-m3' && /handoff/.test(f.branch)) {
          const mode = /Auto/.test(f.branch) ? 'auto' : /Track/.test(f.branch) ? 'track' : 'custom';
          const hits = [...albumMm3].filter(([, tracks]) => tracks.some(t => t.caption === core));
          const hit = mode === 'auto'
            ? hits.find(([, tracks]) => node.src.resolveMm3Caption({ bpm: body.bpm, caption_mm3: '' }, tracks, { mode: 'auto' }).caption === core)
            : hits[0];
          if (mode !== 'custom' && !hit) { failures.push(`${label}: no album ${mode === 'auto' ? 'auto-picks' : 'carries'} the captured caption`); report.push(`FAIL ${label}`); continue; }
          intent.captionSource = mode === 'custom'
            ? { engine: 'minimax-m3', customCaption: core, selection: { mode: 'custom' }, tracks: [] }
            : { engine: 'minimax-m3', customCaption: '', lyricsSetId: hit[0],
                selection: mode === 'auto' ? { mode: 'auto' } : { mode: 'track', selectedTitle: hit[1].find(t => t.caption === core).title } };
          if (mode !== 'custom') { intent.params.caption = ''; notes[0] = 'typed=empty (box locked to the source)'; }
          notes.push(`mm3 handoff ${mode}${mode === 'custom' ? '' : ' source by search'}`);
        }
        const parsed = (await import('../src/contracts/resolution.ts')).resolveIntentSchema.safeParse(intent);
        if (!parsed.success) { failures.push(`${label}: reconstructed intent fails the schema: ${parsed.error.message}`); report.push(`FAIL ${label}`); continue; }
        const data = await node.load.loadCreateData(parsed.data, engine);
        const r = node.intent.resolveCreateIntent(parsed.data, engine, data);
        // A random-seed expansion is a fresh draw on each side: those fields
        // must both be valid expansions, not equal.
        const random = expanded && body.randomSeed === true;
        const nondeterministic = new Set();
        if (random) {
          const opts = (t, wrap) => { const out = new Set(); for (let s = 0; s < 400; s++) out.add(wrap(node.content.expandWildcards(t, s))); return out; };
          const composeOpts = { loraTrigger: trig, beatIntro: !!beat, introBars: beat ? Number(beat[1]) : undefined };
          const captionSet = intent.captionSource ? new Set([r.request.caption]) : opts(capT, x => node.content.composeCreateCaption(x, composeOpts));
          const lyricSet = opts(lyrT, x => x);
          for (const [k, set] of [['caption', captionSet], ['lyrics', lyricSet]]) {
            checks += 2;
            if (!set.has(body[k])) failures.push(`${label}: captured ${k} is not a valid expansion`);
            if (!set.has(r.request[k])) failures.push(`${label}: resolved ${k} is not a valid expansion`);
            if (!(k === 'caption' && intent.captionSource)) nondeterministic.add(k);
          }
          notes.push(`random draw: ${[...nondeterministic].join(', ')} checked as valid expansions`);
        }
        compareBody(label, r.request, body, engine, nondeterministic);
        report.push(`${failures.length === before ? 'PASS' : 'FAIL'} ${label} -> ${notes.join('; ')}`);
      }
    } catch (err) {
      failures.push(`${label}: ${err?.stack || err}`);
      report.push(`FAIL ${label} -> error`);
    }
  }
  db.close();
}

console.log(`differential: ${differentialChecks} checks`);
console.log(report.join('\n'));
console.log(`\n${checks} checks, ${failures.length} mismatch(es); exceptions: expectedBackend pinned ${exceptionsSeen.expectedBackend}x, random-draw fields ${exceptionsSeen.nondeterministic}x`);
if (failures.length) console.log(failures.slice(0, 40).join('\n'));
try { fs.rmSync(tmpData, { recursive: true, force: true }); } catch { /* db handle still open on Windows */ }
process.exit(failures.length ? 1 : 0);
