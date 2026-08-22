// check-mm3-compose.ts — self-test for the deterministic MM3 caption composer
//
//   cd server && npx tsx scripts/check-mm3-compose.ts
//
// Covers both the composer service and the /api/lireek/mm3/* handlers without
// launching the app. The route layer is exercised by calling the registered
// handlers with mock req/res rather than binding a socket — a real listen()
// plus fetch() aborts on Node 24/Windows during teardown, and this script has
// to be usable from a shell script.
// Needs server/src/data/mm3-corpus.json — build it with:
//   node server/scripts/build-mm3-corpus.mjs

import type { Request, Response, Router } from 'express';
import { composeMm3Caption, parseBrief } from '../src/services/lireek/mm3Compose.js';
import { registerMm3Routes } from '../src/routes/lireek/mm3Routes.js';

let failures = 0;
function check(name: string, ok: boolean, extra = ''): void {
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? `  ${extra}` : ''}`);
}

// ── service level ───────────────────────────────────────────────────────────

console.log('── composer ──');

const BRIEFS: Array<{ brief: string; family: string }> = [
  { brief: 'a defiant hardcore punk song about queue-jumping at a processing station', family: 'pop-alternative-rock' },
  { brief: 'gritty delta blues with slide guitar and a growly old man voice', family: 'soul-blues-gospel' },
  { brief: 'epic symphonic metal with a choir and orchestra', family: 'metal-heavy-rock' },
  { brief: 'lo-fi hip hop for studying, rainy night', family: 'hip-hop-rap' },
  { brief: 'funky disco banger for a wedding dance floor', family: 'dance-pop-disco-funk' },
  { brief: 'a mandopop breakup ballad, female singer', family: 'east-asian-ballad-heritage' },
  // Genre-router rule 6: 'ballad' and 'happy' are modifiers, not genre evidence.
  { brief: 'a sad piano ballad about losing my dad', family: 'general-pop-ballad' },
  { brief: 'happy song about my dog', family: 'general-pop-ballad' },
];

for (const { brief, family } of BRIEFS) {
  const r = composeMm3Caption(brief, { seed: 1 });
  check(`routes "${brief.slice(0, 42)}…"`, r.family === family, `got ${r.family}`);
  check(`  format-valid`, r.validation.length === 0, r.validation.join(' | '));
  const words = r.caption.split(/\s+/).length;
  check(`  word count in band`, words >= 420 && words <= 700, `${words} words`);
  check(`  draws on several references`, r.sourceCount >= 5, `${r.sourceCount} templates`);
}

// Determinism and seed variation are both load-bearing: without the first the
// caption is not reproducible, without the second every user gets the same one.
const s7a = composeMm3Caption('a defiant hardcore punk song', { seed: 7 }).caption;
const s7b = composeMm3Caption('a defiant hardcore punk song', { seed: 7 }).caption;
const s8 = composeMm3Caption('a defiant hardcore punk song', { seed: 8 }).caption;
check('same seed is byte-identical', s7a === s7b);
check('different seed differs', s7a !== s8);

// The documented failure this whole approach exists to prevent: a punk brief
// rendering as southern rock because the caption used southern-rock vocabulary.
const WRONG_FAMILY = ['live-room', 'close-miked', 'galloping', 'garage', 'minimal polish'];
const punk = composeMm3Caption('a defiant hardcore punk song', { seed: 3 }).caption.toLowerCase();
check('no out-of-distribution vocabulary', !WRONG_FAMILY.some((w) => punk.includes(w)),
  WRONG_FAMILY.filter((w) => punk.includes(w)).join(', '));
check('carries the genre’s own dialect',
  ['wall of sound', 'heavily distorted', 'palm-muted', 'anthemic', 'power chord'].some((w) => punk.includes(w)));

// ── slot precedence ─────────────────────────────────────────────────────────

console.log('\n── slot resolution ──');

const brief = composeMm3Caption('a slow blues song with a female singer in Bb minor', { seed: 1 });
check('brief fills unset controls',
  brief.slots.bpm.source === 'brief' && brief.slots.key.value === 'Bb' &&
  brief.slots.scale.value === 'minor' && brief.slots.gender.value === 'female');

const ctrl = composeMm3Caption('a fast punk song with a female singer at 190 bpm', {
  controls: { bpm: 90, vocalGender: 'male' }, seed: 1,
});
check('control beats brief', ctrl.slots.bpm.value === 90 && ctrl.slots.gender.value === 'male');
check('conflicts are reported', ctrl.notes.filter((n) => n.includes('the prompt says')).length === 2,
  ctrl.notes.join(' / '));

const dflt = composeMm3Caption('a metalcore song', { seed: 1 });
check('corpus supplies genre-appropriate defaults',
  dflt.slots.bpm.source === 'corpus-default' && Number(dflt.slots.bpm.value) > 0);

const dropped = composeMm3Caption('a punk song', {
  controls: { timeSignature: '6/8', vocalLanguage: 'ja' }, seed: 1,
});
check('unexpressible controls are reported, not silently dropped',
  dropped.notes.some((n) => n.includes('time signature')) &&
  dropped.notes.some((n) => n.includes('language')));

check('parseBrief reads all four facts', (() => {
  const f = parseBrief('fast metalcore at 190 bpm, female vocals, in D minor');
  return f.bpm === 190 && f.gender === 'female' && f.key === 'D' && f.scale === 'minor';
})(), JSON.stringify(parseBrief('fast metalcore at 190 bpm, female vocals, in D minor')));

// "Bb" only parses as B-flat while the regex is case-sensitive on the note
// letter — with /i, [#b] also matches the letter B.
check('Bb parses as B-flat not B', parseBrief('a song in Bb major').key === 'Bb');

// ── route level ─────────────────────────────────────────────────────────────

console.log('\n── endpoints ──');

// Collect the handlers registered by registerMm3Routes, keyed "METHOD path".
type Handler = (req: Request, res: Response) => unknown;
const handlers = new Map<string, Handler>();
const fakeRouter = {
  post: (p: string, h: Handler) => handlers.set(`POST ${p}`, h),
  get: (p: string, h: Handler) => handlers.set(`GET ${p}`, h),
} as unknown as Router;
registerMm3Routes(fakeRouter);

/** Invokes one handler and captures whatever it sends. */
async function call(route: string, body: unknown): Promise<{ status: number; json: any }> {
  const handler = handlers.get(route);
  if (!handler) throw new Error(`no handler registered for ${route}`);
  let status = 200;
  let json: any;
  const res = {
    status(code: number) { status = code; return this; },
    json(payload: unknown) { json = payload; return this; },
  } as unknown as Response;
  await handler({ body } as Request, res);
  return { status, json };
}

check('all three routes registered', handlers.size === 3, [...handlers.keys()].join(', '));

const ok = await call('POST /mm3/compose', {
  brief: 'a defiant hardcore punk song',
  // deliberately a full params-shaped blob: the route must ignore what it does not read
  controls: { bpm: 172, keyScale: 'E minor', vocalGender: 'male', duration: 180, seed: 99, adapters: [] },
  seed: 3,
});
check('POST compose -> 200', ok.status === 200, `status=${ok.status}`);
check('  caption is well-formed', ok.json.caption?.startsWith('Global Metadata') && ok.json.validation.length === 0);
check('  unknown control fields ignored', ok.json.slots.bpm.value === 172);

const bad = await call('POST /mm3/compose', { brief: '   ' });
check('POST compose empty brief -> 400', bad.status === 400, `status=${bad.status}`);

const parsed = await call('POST /mm3/parse-brief', { brief: 'a duet at 140 bpm' });
check('POST parse-brief', parsed.json.bpm === 140 && parsed.json.gender === 'duet', JSON.stringify(parsed.json));

const info = (await call('GET /mm3/corpus-info', {})).json;
check('GET corpus-info', info.cards === 1000 && info.families === 18,
  `cards=${info.cards} families=${info.families}`);
// The thin-pool warning the UI relies on is only meaningful if this stays true.
check('  metal female pool is still tiny', info.byFamily['metal-heavy-rock'].female === 2,
  `female=${info.byFamily['metal-heavy-rock']?.female}`);


console.log(failures ? `\n${failures} CHECK(S) FAILED` : '\nall checks passed');
process.exitCode = failures ? 1 : 0;
