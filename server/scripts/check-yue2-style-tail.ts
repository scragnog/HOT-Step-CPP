// check-yue2-style-tail.ts — BPM and Key reach YuE2 through the style sentence.
// Run: npx tsx server/scripts/check-yue2-style-tail.ts
//
// There is no wire field for either. The trainer composed them into the tail
// of the style sentence (engine/src/train/yue2-sidecar.h), so that tail is the
// only channel they have. The critical property is the last one: a render that
// leaves them unset must compose EXACTLY what it composed before.

import assert from 'node:assert';
import { applyYue2StyleTemplate, splitYue2Tail, yue2StyleSelfCheck } from '../src/services/backends/yue2/style.js';

const trigger = 'an_artist';
const caption = 'theatrical chamber pop with sweeping strings.';
const base = { trigger, caption, template: 'upstream' as const };

// Unset: byte-identical to the pre-change behaviour.
assert.equal(
  applyYue2StyleTemplate({ ...base, bpm: '', key: '' }),
  applyYue2StyleTemplate(base),
  'blank bpm/key must compose no tail',
);
assert.equal(
  applyYue2StyleTemplate(base),
  'an_artist, in the style of an_artist. theatrical chamber pop with sweeping strings.',
);

// Set: the trained tail.
assert.equal(
  applyYue2StyleTemplate({ ...base, bpm: 142, key: 'C major' }),
  'an_artist, in the style of an_artist. theatrical chamber pop with sweeping strings. 142 BPM, key of C major.',
);
assert.equal(
  applyYue2StyleTemplate({ ...base, bpm: 142, key: '' }),
  'an_artist, in the style of an_artist. theatrical chamber pop with sweeping strings. 142 BPM.',
);
assert.equal(
  applyYue2StyleTemplate({ ...base, bpm: '', key: 'C major' }),
  'an_artist, in the style of an_artist. theatrical chamber pop with sweeping strings. key of C major.',
);

// Idempotent: re-submitting a composed caption must not double the tail.
const once = applyYue2StyleTemplate({ ...base, bpm: 142, key: 'C major' });
assert.equal(
  applyYue2StyleTemplate({ trigger, caption: once, template: 'upstream', bpm: 142, key: 'C major' }),
  once,
  'f(f(x)) === f(x)',
);

// ── splitYue2Tail: a dataset caption that already carries a tail ─────────────
// The real report: "…wide stereo reverb, 178 BPM" + a user BPM of 168 gave
// "…, 178 BPM 168 BPM, key of D minor."
const dataset = 'English, dark synth-pop, atmospheric synth pads, brooding and tense, '
  + 'modern spacious electronic mix with wide stereo reverb, 178 BPM';
let sp = splitYue2Tail(dataset);
assert.equal(sp.bpm, '178');
assert.equal(sp.key, '');
assert.ok(!/BPM/i.test(sp.caption), `tail left behind: ${sp.caption}`);
assert.ok(sp.caption.endsWith('wide stereo reverb'), sp.caption);

// Both fields, the full trained tail.
sp = splitYue2Tail('a caption, 178 BPM, key of D minor.');
assert.deepEqual([sp.caption, sp.bpm, sp.key], ['a caption', '178', 'D minor']);

// Either alone, and the reverse order.
assert.deepEqual(splitYue2Tail('a caption, key of D minor.').key, 'D minor');
assert.deepEqual(splitYue2Tail('a caption key of D minor, 178 BPM').bpm, '178');
assert.deepEqual(splitYue2Tail('a caption key of D minor, 178 BPM').key, 'D minor');

// No tail: untouched.
assert.deepEqual(splitYue2Tail('just a caption.'), { caption: 'just a caption.', bpm: '', key: '' });
// "BPM" as prose, not a value, is not a tail.
assert.equal(splitYue2Tail('a caption about BPM').bpm, '');

// End to end: the user's BPM replaces the dataset's, exactly once.
const composed = applyYue2StyleTemplate({
  trigger, template: 'upstream',
  caption: splitYue2Tail(dataset).caption,
  bpm: 168, key: 'D minor',
});
assert.equal((composed.match(/BPM/gi) || []).length, 1, `doubled tail: ${composed}`);
// No period is inserted after the caption — the engine assembles
// head + caption + ' ' + tail and nothing else (yue2StyleString's own note).
assert.ok(composed.endsWith('wide stereo reverb 168 BPM, key of D minor.'), composed);

// A field the user left unset keeps the caption's own value.
const kept = splitYue2Tail('a caption, 178 BPM, key of D minor.');
assert.equal(
  applyYue2StyleTemplate({ trigger, template: 'upstream', caption: kept.caption, bpm: 168, key: kept.key }),
  'an_artist, in the style of an_artist. a caption 168 BPM, key of D minor.',
);

// The literal port still matches the engine, branch for branch.
assert.deepEqual(yue2StyleSelfCheck(), [], 'style string diverged from the C++');

console.log('check-yue2-style-tail: all assertions passed');
