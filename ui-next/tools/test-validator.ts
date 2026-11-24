// The validator's own test: plausible mistakes a maintainer could make, one at a time, each on a fresh copy of
// ui-next, with what the type check and the validator say. Every mistake must be caught (a type error or a
// validator error), except the few this format has made impossible, which must now pass.
//
//   node tools/test-validator.ts          all of them; exit 1 if one is not caught
//   node tools/test-validator.ts M07 M12  only these
//
// The list started as the 40 of the format's review (definitions-review, work/mutate.py), rewritten for this
// format; a mistake found later belongs here too.

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TMP = path.join(os.tmpdir(), 'ui-next-mutations');
const D = 'src/definitions/';
const PARAMS = '  params: [\n';
type Edit = [file: string, old: string, now: string];
/** `shipped`: the copy records what it ships (npm run ship) before the mistake is made, as after a release. */
interface Mutation { id: string; says: string; edits: Edit[]; gone?: string; request?: true; core?: true; shipped?: true }

const M: Mutation[] = [
  { id: 'M01', says: 'a parameter added without `engines` beside its neighbours, drawn on YuE2 only', edits: [
    ['modules/planner.ts', PARAMS, PARAMS + "    { key: 'yp.minp', type: 'number', label: 'Plan min-P', range: [0, 1, 0.01], default: 0.05, wire: 'ext:yue2PlanTopP' },\n"],
    ['modules/planner.ts', "{ part: 'token', param: 'yp.topp', label: '--top-p' },", "{ part: 'token', param: 'yp.topp', label: '--top-p' }, { part: 'token', param: 'yp.minp', label: '--min-p' },"]] },
  { id: 'M02', says: 'a default off the step grid', edits: [
    ['modules/replan.ts', PARAMS, PARAMS + "    { key: 'yue.x', type: 'number', label: 'X', range: [0, 1, 0.1], default: 0.05, wire: 'ext:yue2ComposeRetries' },\n"],
    ['modules/replan.ts', "{ part: 'knob', param: 'yue.tries', label: 'Tries' },", "{ part: 'knob', param: 'yue.tries', label: 'Tries' }, { part: 'knob', param: 'yue.x' },"]] },
  { id: 'M03', says: 'a parameter nobody can change: listed only by a read-out', edits: [
    ['modules/replan.ts', PARAMS, PARAMS + "    { key: 'yue.x', type: 'number', label: 'X', range: [0, 1, 0.1], default: 0, wire: 'ext:yue2ComposeRetries' },\n"],
    ['modules/replan.ts', "reads: ['yue.tries'],", "reads: ['yue.tries', 'yue.x'],"]] },
  { id: 'M04', says: 'a parameter shown only by a node with tiers: []', edits: [
    ['modules/avoid.ts', PARAMS, PARAMS + "    { key: 'avoid.x', type: 'number', label: 'X', range: [0, 1, 0.1], default: 0, wire: 'param:cacheRatio' },\n"],
    ['modules/avoid.ts', '  face: [', "  face: [{ part: 'knob', param: 'avoid.x', tiers: [] }, "]] },
  { id: 'M05', says: 'a child tiered for a tier its parent never shows', edits: [
    ['modules/tempo.ts', "{ part: 'lcdButton', param: 'tempo.trim', label: 'AUTO-TRIM', tiers: ['lab'] },", "{ part: 'lcdButton', param: 'tempo.trim', label: 'AUTO-TRIM', tiers: ['quick'] },"]] },
  { id: 'M06', says: 'a misspelt real name', edits: [['modules/sampler.ts', "wire: 'param:inferenceSteps',", "wire: 'param:inferenceStep',"]] },
  { id: 'M07', says: 'two parameters on one engine sent to one real field', edits: [['pedals/lift.ts', "wire: 'param:slNoiseFloor',", "wire: 'param:slHfMix',"]] },
  { id: 'M08', says: 'invert on a number, scale on a switch', edits: [
    ['pedals/gain.ts', "wire: 'param:gainOffsetDb',", "wire: { to: 'param:gainOffsetDb', invert: true },"],
    ['pedals/ppvae.ts', "wire: 'param:ppVaeReencode'", "wire: { to: 'param:ppVaeReencode', scale: 1000 }"]] },
  { id: 'M09', says: 'an automatic parameter whose wire does not say what auto sends', edits: [['modules/tempo.ts', "        auto: 0,\n        note: 'Auto is 0. createContentParams", "        note: 'Auto is 0. createContentParams"]] },
  { id: 'M10', says: 'wire.auto on a parameter that has no auto', edits: [['pedals/gain.ts', "wire: 'param:gainOffsetDb',", "wire: { to: 'param:gainOffsetDb', auto: -1 },"]] },
  { id: 'M11', says: 'Quick choices off the step grid, and none equal to the default', edits: [['modules/sampler.ts', "choices: [['Loose', 5], ['Balanced', 9], ['Strict', 13]]", "choices: [['Loose', 5.55], ['Balanced', 8.123], ['Strict', 13]]"]] },
  { id: 'M12', says: 'Quick choices missing for one engine the row is on', edits: [['modules/sampler.ts', "        { engines: ['heartmula'], choices: [['Draft', 6], ['Standard', 10], ['High', 20]] },\n", '']] },
  { id: 'M13', says: 'two Quick choices with one word, two with one value', edits: [['modules/sampler.ts', "choices: [['Draft', 8], ['Standard', 12], ['High', 30]]", "choices: [['Draft', 8], ['Draft', 12], ['High', 12]]"]] },
  { id: 'M14', says: 'part options of the wrong type, an unknown look, a power chip\'s on outside the range', edits: [
    ['modules/guidance.ts', "opts: { look: 'apg', xl: 'momentum', yl: 'norm threshold' },", "opts: { look: 'purple', xl: 7, yl: ['a'] },"],
    ['modules/adhere.ts', 'opts: { off: 1, on: 1.5 }', 'opts: { off: 1, on: 99 }']] },
  { id: 'M15', says: 'an option and a read naming parameters that do not exist', edits: [
    ['modules/replan.ts', "opts: { offWhen: 'yue.replan' }", "opts: { offWhen: 'yue.nope' }"],
    ['units/wh.ts', "reads: ['pp.wh.model', 'pp.wh.lang']", "reads: ['pp.wh.modle', 'pp.wh.lang']"]] },
  { id: 'M16', says: 'a named action nothing declares', edits: [['pedals/vst.ts', "name: 'vstHost'", "name: 'vstHots'"]] },
  { id: 'M17', says: '"Insert example" on a parameter with no example; a set action far outside the range', edits: [
    ['modules/latent.ts', "do: 'example', param: 'lat.ts'", "do: 'example', param: 'lat.shift'"],
    ['pedals/tile.ts', "          param: 'samp.plugins',\n          value: true,", "          param: 'samp.steps',\n          value: 100000,"]] },
  { id: 'M18', says: 'conditions outside the range, contradictory, and circular', edits: [
    ['pedals/lift.ts', '      default: 6,', "      default: 6,\n      when: { param: 'pp.lift.den', above: 7, is: 5, not: 5 },"],
    ['pedals/lift.ts', '      default: 0.3,', "      default: 0.3,\n      when: { param: 'pp.lift.shim', above: 0 },"]] },
  { id: 'M19', says: 'a pedal\'s foot switch hidden by `when`', edits: [['pedals/ppvae.ts', "label: 'On', default: false,", "label: 'On', default: false, when: { param: 'pp.ppvae.blend', above: 0.5 },"]] },
  { id: 'M20', says: 'a range end whose partner has another range', edits: [['modules/adhere.ts', "      staysBelow: 'yue.pcfg2',", "      staysBelow: 'yue.pcfg',"]] },
  { id: 'M21', says: 'a range end that stays below itself', edits: [['modules/adhere.ts', "      staysBelow: 'yue.pcfg2',", "      staysBelow: 'yue.pcfg1',"]] },
  { id: 'M22', says: 'a key that collides with another parameter\'s auto switch (tempo.bpm.auto)', gone: 'auto switches are a map of their own, so tempo.bpm.auto is an ordinary key', edits: [
    ['modules/tempo.ts', PARAMS, PARAMS + "    { key: 'tempo.bpm.auto', type: 'boolean', label: 'Follow', default: false, engines: ['ace', 'minimax-m3', 'yue2'], wire: 'ui:x' },\n"],
    ['modules/tempo.ts', "            { part: 'tapTempo', param: 'tempo.bpm' },", "            { part: 'tapTempo', param: 'tempo.bpm' }, { part: 'lcdButton', param: 'tempo.bpm.auto' },"]] },
  { id: 'M23', says: 'two keys that give one translation key (perf.a-b and perf-a.b)', edits: [
    ['modules/perf.ts', PARAMS, PARAMS + "    { key: 'perf.a-b', type: 'boolean', label: 'One', default: false, wire: 'ui:x' },\n    { key: 'perf-a.b', type: 'boolean', label: 'Two', default: false, wire: 'ui:y' },\n"],
    ['modules/perf.ts', '  face: [\n', "  face: [\n    { part: 'pill', param: 'perf.a-b' }, { part: 'pill', param: 'perf-a.b' },\n"]] },
  { id: 'M24', says: 'two modules at one place', edits: [['modules/planner.ts', '  at: 60,', '  at: 80,']] },
  { id: 'M25', says: 'a Lab module under a module that is itself Lab-only', edits: [['modules/replan.ts', "labUnder: 'planner'", "labUnder: 'perf'"]] },
  { id: 'M26', says: 'a server-listed choice with a default the list cannot have', edits: [['modules/sampler.ts', "      default: 'euler',", "      default: 'eulr',"]] },
  { id: 'M27', says: 'a condition on a server-listed choice with a value it cannot have', edits: [['modules/sampler.ts', "when: { param: 'samp.sched', is: 'beta' },", "when: { param: 'samp.sched', is: 'Beta (Custom)' },"]] },
  { id: 'M28', says: 'a control part with nothing bound', edits: [['modules/replan.ts', "{ part: 'knob', param: 'yue.tries', label: 'Tries' },", "{ part: 'knob', param: 'yue.tries', label: 'Tries' }, { part: 'knob' }, { part: 'segmented', label: 'Mode' },"]] },
  { id: 'M29', says: 'terminal flags in another order than their controls', gone: 'a flag is its control\'s label, so it cannot be out of step with it', edits: [
    ['modules/planner.ts', "{ part: 'token', param: 'yp.temp', label: '--temp' },", "{ part: 'token', param: 'yp.temp', label: '--top-k' },"]] },
  { id: 'M30', says: 'kept per engine and saved as song content', edits: [['modules/planner.ts', "  owner: 'song',\n  stored: 'settings',", "  owner: 'engine',\n  stored: 'content',"]] },
  { id: 'M31', says: 'a definition on an engine where none of its parameters or nodes are', edits: [['modules/perf.ts', "engines: ['ace'],", "engines: ['ace', 'sa3'],"]] },
  { id: 'M32', says: 'a pedal thumbnail one unit wide with five knobs', gone: 'the width is worked out from the number of knobs; there is no field to get wrong', edits: [] },
  { id: 'M33', says: 'format words for a value it cannot hold, a template without {v}, negative decimals', edits: [['modules/timbre.ts', '      range: [0, 1, 0.05],\n      default: 1,', "      range: [0, 1, 0.05],\n      format: { words: [{ value: 55, text: 'never' }], template: 'lots', decimals: -3, perSecond: 0 },\n      default: 1,"]] },
  { id: 'M34', says: 'held on a pedal number with no words for the held note', edits: [['pedals/ss.ts', "      default: 8,\n      wire: 'param:stableStepSteps',", "      default: 8,\n      held: true,\n      engines: ['ace', 'minimax-m3', 'yue2', 'heartmula'],\n      per: [{ engines: ['heartmula'], range: [1, 32, 1] }],\n      wire: 'param:stableStepSteps',"]] },
  { id: 'M35', says: 'a strip that can never hold, with empty words', edits: [['modules/takes.ts', "when: { engine: ['sa3', 'heartmula'] }, says: 'One song per run · Generate again to queue another'", "when: { all: [{ engine: ['ace'] }, { engine: ['sa3'] }] }, says: ''"]] },
  { id: 'M36', says: 'an empty label; two options with one label', edits: [
    ['units/q.ts', "{ value: 'mastered', label: 'Mastered', short: 'Master' },", "{ value: 'mastered', label: 'Unmastered', short: 'Master' },"],
    ['units/q.ts', "label: 'Evaluate',", "label: '',"]] },
  { id: 'M37', says: 'a pedal\'s switch naming another pedal\'s parameter', edits: [['pedals/ppvae.ts', "on: 'pp.ppvae.on',", "on: 'pp.lift.on',"]] },
  { id: 'M38', says: 'a drop zone holding another module\'s state', edits: [['modules/timbre.ts', "      state: ['timbreFile'],\n      opts: { look: 'sleeve', empty:", "      state: ['masterRef'],\n      opts: { look: 'sleeve', empty:"]] },
  { id: 'M39', says: 'state declared and named by no node', edits: [['modules/perf.ts', PARAMS, "  state: [{ key: 'perfThing', kind: 'file', noun: 'the thing', owner: 'song', stored: 'settings', wire: 'ui:x' }],\n" + PARAMS]] },
  { id: 'M40', says: 'a range on one engine that leaves out the base default', edits: [['modules/sampler.ts', "{ engines: ['sa3'], range: [1, 64, 1], default: 8, wire: 'ext:sa3Steps' },", "{ engines: ['sa3'], range: [1, 6, 1], wire: 'ext:sa3Steps' },"]] },
  /* later ones */
  { id: 'M41', says: 'an engine added to a module, with a parameter that differs per engine left to inherit another engine\'s', edits: [['modules/takes.ts', "      engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],\n      held:", "      held:"]] },
  { id: 'M42', says: 'a read-out given a parameter to edit', edits: [['pedals/tile.ts', "{ part: 'lcdValue', reads: ['pp.tile.plugin'],", "{ part: 'lcdValue', param: 'pp.tile.plugin',"]] },
  { id: 'M43', says: 'a word slot that names nothing', edits: [['modules/guidance.ts', "says: '{engine} uses its own guidance'", "says: '{engin} uses its own guidance {value:samp.gide}'"]] },
  { id: 'M44', says: 'a transform name that does not exist', edits: [['modules/sampler.ts', "to: 'param:retakeSeed', via: 'int'", "to: 'param:retakeSeed', via: 'integer'"]] },
  { id: 'M45', says: 'a default that differs from the real app\'s, with no word why', edits: [['pedals/gain.ts', '      default: 0,', '      default: 3,']] },
  { id: 'M46', says: 'a factory revision setting a value its engine cannot take', edits: [['factory.ts', "vals: { 'takes.n': 4, 'samp.retake': 0.3 }", "vals: { 'takes.n': 12, 'samp.retake': 0.3 }"]] },
  /* the second review's (definitions-review2, work/mutate.mjs), rewritten for the files as they are now; `request` runs
     the request comparison too, `core` the derivation tests */
  { id: 'B01', says: 'a word line that changes range, default, unit and wire', edits: [['pedals/gain.ts', "      default: 0,\n", "      default: 0,\n      per: [{ when: { param: 'pp.on', is: false }, range: [0, 100, 1], default: 50, unit: '%', wire: 'param:lufsCeilingDb' }],\n"]] },
  { id: 'B02', says: 'a conditional parameter whose default is not the real one (crossover 300 Hz; real 250)', edits: [['pedals/ss.ts', '      default: 250,', '      default: 300,']] },
  { id: 'B03', says: 'the scale dropped from a wire, its note kept (StableStep strength sent as 30, not 0.30)', request: true, edits: [['pedals/ss.ts', "wire: { to: 'param:stableStepStrength', divide: 100, differs: ['range'],", "wire: { to: 'param:stableStepStrength', differs: ['range'],"]] },
  { id: 'B04', says: 'invert dropped from a switch whose wire has a note (Lyric timestamps)', request: true, edits: [['units/lrc.ts', "wire: { to: 'param:skipLrc', invert: true, note: 'skipLrc is the opposite.' },", "wire: { to: 'param:skipLrc', note: 'skipLrc is the opposite.' },"]] },
  { id: 'B05', says: 'a switch wired to a numeric field', edits: [['units/q.ts', "wire: 'param:qualityEvalEnabled' }", "wire: 'param:storkSubsteps' }"]] },
  { id: 'B06', says: 'an option value the server does not take (qualityEvalTarget "raw")', edits: [['units/q.ts', "{ value: 'unmastered', label: 'Unmastered', short: 'Raw' },", "{ value: 'raw', label: 'Unmastered', short: 'Raw' },"], ['units/q.ts', "default: 'unmastered',", "default: 'raw',"]] },
  { id: 'B07', says: 'two knobs with one label on one pedal', edits: [['pedals/ss.ts', "      label: 'Width',", "      label: 'Crossover',"]] },
  { id: 'B08', says: 'an option added to a mapped choice, the map forgotten', edits: [['pedals/norm.ts', "        { value: 'custom', label: 'Custom', lcd: 'CUSTOM', detail: 'set your own' },", "        { value: 'vinyl', label: 'Vinyl · -18', lcd: '-18 LUFS', detail: 'Vinyl' },\n        { value: 'custom', label: 'Custom', lcd: 'CUSTOM', detail: 'set your own' },"]] },
  { id: 'B09', says: 'a control shown at Studio only: nothing edits it at Lab', edits: [['modules/tempo.ts', "{ part: 'lcdButton', param: 'tempo.trim', label: 'AUTO-TRIM', tiers: ['lab'] },", "{ part: 'lcdButton', param: 'tempo.trim', label: 'AUTO-TRIM', tiers: ['studio'] },"]] },
  { id: 'B10', says: 'a text default that fails its own check', edits: [['pedals/ss.ts', "      default: '4242',", "      default: 'forty-two',"]] },
  { id: 'B11', says: 'a pedal in its neighbour\'s hue', edits: [['pedals/ppvae.ts', '  hue: 175,', '  hue: 140,']] },
  { id: 'B12', says: 'staysBelow put on the upper end', edits: [['modules/sampler.ts', "      staysBelow: 'samp.ce',\n", ''], ['modules/sampler.ts', "      engines: ['ace'],\n      wire: 'param:cfgIntervalEnd',", "      engines: ['ace'],\n      staysBelow: 'samp.cs',\n      wire: 'param:cfgIntervalEnd',"]] },
  { id: 'B13', says: 'two parameters that each stay below the other', edits: [['modules/sampler.ts', "      engines: ['ace'],\n      wire: 'param:cfgIntervalEnd',", "      engines: ['ace'],\n      staysBelow: 'samp.cs',\n      wire: 'param:cfgIntervalEnd',"]] },
  { id: 'B14', says: 'a factory revision with a step-off whole number, a parameter that does not apply and a malformed slot', edits: [['factory.ts', "vals: { 'samp.steps': 8, 'perf.cache': 0.3 }", "vals: { 'samp.steps': 8.5, 'perf.cache': 0.3, 'pp.ss.xhz': 61.3 }, state: { slots: 'tape-lofi-v2' }"]] },
  { id: 'B15', says: 'a "Show in Lab" action that goes to a tier where the control is not shown', edits: [['modules/sampler.ts', "actions: [{ do: 'tier', tier: 'lab', ring: ['lat.ts'], label: 'Show in Lab' }],", "actions: [{ do: 'tier', tier: 'studio', ring: ['lat.ts'], label: 'Show in Lab' }],"]] },
  { id: 'B16', says: 'words that name a value the engine does not have (a YuE2 note saying the APG momentum)', edits: [['modules/sampler.ts', "text: 'YuE2 guides inside its composer. The Planner holds its CFG.',", "text: 'YuE2 guides inside its composer at {value:apg.mom}. The Planner holds its CFG.',"]] },
  { id: 'B17', says: 'a pedal with no main knob and no Studio node', edits: [['pedals/ppvae.ts', "main: 'pp.ppvae.blend',", 'main: null,']] },
  { id: 'B18', says: 'a module whose owner is flipped (the Planner kept per engine): saved data keeps it where it was kept', shipped: true, edits: [['modules/planner.ts', "  owner: 'song',\n  stored: 'settings',", "  owner: 'engine',\n  stored: 'settings',"]] },
  { id: 'B19', says: 'song content left as a setting (the lyrics saved in revisions)', core: true, edits: [['modules/lyrics.ts', "  owner: 'song',\n  stored: 'content',", "  owner: 'song',\n  stored: 'settings',"]] },
  { id: 'B20', says: 'a read-out whose reads lost two of the three values its formula uses', edits: [['modules/perf.ts', "{ part: 'speedGauge', reads: ['perf.cfgcut', 'perf.cache', 'perf.lmcut'] },", "{ part: 'speedGauge', reads: ['perf.cfgcut'] },"]] },
  { id: 'B21', says: 'the same control twice in one face', edits: [['modules/perf.ts', "    { part: 'knob', param: 'perf.lmcut', label: 'LM cut' },", "    { part: 'knob', param: 'perf.lmcut', label: 'LM cut' },\n    { part: 'knob', param: 'perf.lmcut', label: 'LM cut again' },"]] },
  { id: 'B22', says: 'wire.auto of the wrong kind (BPM on auto sends the text "auto")', edits: [['modules/tempo.ts', "        auto: 0,\n        note: 'Auto is 0.", "        auto: 'auto',\n        note: 'Auto is 0."]] },
  { id: 'B23', says: 'an engine line whose `when` drops the parameter\'s own condition', edits: [['modules/sampler.ts', "          label: 'Schedule shift',", "          label: 'Schedule shift',\n          when: { engine: ['minimax-m3'] },"]] },
  { id: 'B24', says: 'a Quick row in one module bound to another module\'s parameter', edits: [['modules/avoid.ts', '  face: [', "  face: [{ part: 'choiceRow', param: 'samp.steps', label: 'Quality', choices: [['Draft', 8], ['Standard', 12]], tiers: ['quick'] }, "]] },
  { id: 'B25', says: 'a held note that does not say the limit in force (no {shown})', gone: 'cannot be caught: a held note may rightly not say the limit (the takes note, "Your {value} takes return on {engine}", is the concept\'s)', edits: [['modules/tempo.ts', "held: { note: 'Capped at {shown} here · your {value} returns on {engine}', toast: 'Length capped at {shown}' },", "held: { note: 'Your {value} returns on {engine}', toast: 'Length capped' },"]] },
  { id: 'B26', says: 'a transform beside scale (via does the whole change)', edits: [['pedals/ss.ts', "wire: { to: 'param:stableStepSeed', via: 'int' },", "wire: { to: 'param:stableStepSeed', via: 'int', scale: 1000 },"]] },
  { id: 'B27', says: 'a side unit with an empty face', edits: [['units/q.ts', "    { part: 'unitBadge', tip: 'The score each song gets, as its badge in the Library', text: '8.4' },\n", '']] },
  { id: 'B28', says: 'the key wheel bound to the time signature', edits: [['modules/tempo.ts', "params: ['tempo.key'],", "params: ['tempo.sig'],"]] },
  { id: 'B29', says: 'a strip action "Turn it on" that sets the value to off', gone: 'cannot be caught: whether words fit an action is a reading', edits: [['modules/build.ts', "actions: [{ do: 'set', param: 'lm.cotmode', value: 'full', label: 'Turn it on' }],", "actions: [{ do: 'set', param: 'lm.cotmode', value: 'off', label: 'Turn it on' }],"]] },
  { id: 'B30', says: 'an engine line whose label was copied from another engine (HeartMuLa steps called "ODE steps")', gone: 'cannot be caught: a label that is wrong for its engine is a reading', edits: [['modules/sampler.ts', "          label: 'Codec steps',", "          label: 'ODE steps',"]] },
  /* from the second review's trials */
  { id: 'T01', says: 'two options that claim one former value', edits: [['pedals/ss.ts', "{ value: 'crossover', label: 'Crossover', detail:", "{ value: 'crossover', label: 'Crossover', was: ['xo'], detail:"], ['pedals/ss.ts', "{ value: 'mix', label: 'Mix', detail:", "{ value: 'mix', label: 'Mix', was: ['xo'], detail:"]] },
  { id: 'T02', says: 'the five sends the second review found, put back: TensorRT as "trt", the High DCW knob to the low scaler, a phase window dropped, a HeartMuLa trigger, 9 takes', request: true, edits: [
    ['modules/flow.ts', "{ value: 'tensorrt', label: 'TensorRT' },", "{ value: 'trt', label: 'TensorRT' },"],
    ['modules/flow.ts', "when: { param: 'mm.rend', is: 'tensorrt' },", "when: { param: 'mm.rend', is: 'trt' },"]] },
  { id: 'T03', says: 'the High DCW knob wired to the low scaler', request: true, edits: [['modules/latent.ts', "        { to: 'param:dcwHighScaler', when: { param: 'dcw.mode', is: 'high' } },\n", ''], ['modules/latent.ts', "        { to: 'param:dcwLowScaler', when: { param: 'dcw.mode', not: 'high' } },", "        { to: 'param:dcwLowScaler' },"]] },
  { id: 'T04', says: 'a stacked adapter\'s phase window dropped on the way out', request: true, edits: [['transforms.ts', "return { path: s.file, scale: 1, ...(ph && (ph[0]! > 0 || ph[1]! < 100) ? { stepStart: +(1 - ph[1]! / 100).toFixed(4), stepEnd: +(1 - ph[0]! / 100).toFixed(4) } : {}) };", "void ph; return { path: s.file, scale: 1 };"]] },
  { id: 'T05', says: 'a typed trigger word sent on HeartMuLa', request: true, edits: [['modules/prompt.ts', "            none: 'HeartMuLa adds its adapter\\'s own trigger (withHeartmulaTrigger); the existing Create page sends no typed trigger word there.',\n          },", "            none: 'x',\n          },"], ['modules/prompt.ts', "          wire: {\n            none: 'x',\n          },", "          wire: 'intent:compose.loraTrigger',"]] },
  { id: 'T06', says: 'nine takes on ACE-Step, where the server runs eight', request: true, edits: [['modules/takes.ts', '      range: [1, 8, 1],', '      range: [1, 9, 1],']] },
  /* the third review's (definitions-review3): fourteen mistakes that passed every check (C3 and C3b are two), and
     those its trials found at run time; `request` runs the request comparison, `core` the derivation tests */
  { id: 'R1a', says: 'ACE-Step\'s Shift on Auto sends 0 instead of -1', request: true, edits: [['modules/sampler.ts', "wire: { to: 'param:shift', auto: -1,", "wire: { to: 'param:shift', auto: 0,"]] },
  { id: 'R2', says: 'the scheduler transform writes Beta\'s two numbers the other way round', request: true, edits: [['transforms.ts', "`beta:${f2(o['sch.alpha'], 0.5)}:${f2(o['sch.beta'], 0.7)}`", "`beta:${f2(o['sch.beta'], 0.7)}:${f2(o['sch.alpha'], 0.5)}`"]] },
  { id: 'R2b', says: 'the scheduler transform exchanges Composite\'s crossover and split', request: true, edits: [['transforms.ts', ":${f2(o['sch.x'], 0.5)}:${f2(o['sch.split'], 0.5)}`", ":${f2(o['sch.split'], 0.5)}:${f2(o['sch.x'], 0.5)}`"]] },
  { id: 'R4b', says: 'ACE-Step\'s guidance scale applies only under APG', request: true, edits: [['modules/sampler.ts', "      engines: ['ace', 'minimax-m3', 'sa3', 'heartmula'],\n      wire: 'param:guidanceScale',", "      engines: ['ace', 'minimax-m3', 'sa3', 'heartmula'],\n      wire: 'param:guidanceScale',\n      when: { param: 'samp.guide', is: 'apg' },"]] },
  { id: 'R5', says: 'Stable Audio 3\'s guidance starts at 7, not 1', request: true, edits: [['modules/sampler.ts', "          range: [1, 10, 0.1],\n          default: 1,", "          range: [1, 10, 0.1],\n          default: 7,"]] },
  { id: 'R6', says: 'HeartMuLa\'s adapter never sent with the request', request: true, edits: [['modules/adapters.ts', "            {\n              to: 'body:heartmulaAdapter',\n              via: 'firstPath',\n              note: 'Sent with the request so a queued job keeps its adapter.',\n            },\n", '']] },
  { id: 'R7', says: 'the trigger word needs two adapters', request: true, edits: [['modules/prompt.ts', "when: { any: [{ state: 'adapters', atLeast: 1 }, { param: 'ad.planner', not: '' }] },", "when: { any: [{ state: 'adapters', atLeast: 2 }, { param: 'ad.planner', not: '' }] },"]] },
  { id: 'R8', says: 'the lead sheet sent only while the planner writes nothing', request: true, edits: [['modules/score.ts', "when: { param: 'lm.cotmode', not: 'off' },", "when: { param: 'lm.cotmode', is: 'off' },"]] },
  { id: 'R10', says: 'the self-attention and cross-attention group scales exchange fields', request: true, edits: [['modules/adapters.ts', "wire: 'param:adapterGroupScales.self_attn',", "wire: 'param:adapterGroupScales.swap',"], ['modules/adapters.ts', "wire: 'param:adapterGroupScales.cross_attn',", "wire: 'param:adapterGroupScales.self_attn',"], ['modules/adapters.ts', "wire: 'param:adapterGroupScales.swap',", "wire: 'param:adapterGroupScales.cross_attn',"]] },
  { id: 'R13', says: 'the scheduler transform sends sgm_uniform as linear', request: true, edits: [['transforms.ts', "      return s;\n    },\n    read: real => {", "      return s === 'sgm_uniform' ? 'linear' : s;\n    },\n    read: real => {"]] },
  { id: 'C1', says: 'the denoiser strength\'s undo name misspelt', core: true, edits: [['modules/latent.ts', "undoName: 'Denoiser strength',", "undoName: 'Denoizer strenght',"]] },
  { id: 'C2b', says: 'a fixed option\'s label misspelt', core: true, edits: [['modules/latent.ts', "{ value: 'pix', label: 'Pixel-Space (No Wavelets)', short: 'Pixel' },", "{ value: 'pix', label: 'Pixle-Space (No Wavelets)', short: 'Pixel' },"]] },
  { id: 'C3', says: 'the engine switch toast counts each setting twice', core: true, edits: [['../derive/engine.ts', "' ')); else sets++;", "' ')); else sets += 2;"]] },
  { id: 'C3b', says: 'the engine switch toast counts one adapter too many', core: true, edits: [['../derive/engine.ts', "...(slots ? [msg('switch.adapters', { n: slots })] : []),", "...(slots ? [msg('switch.adapters', { n: slots + 1 })] : []),"]] },
  { id: 'C9', says: 'the Sampler\'s Power group shown while the schedule is Cosine', core: true, edits: [['modules/sampler.ts', "      opts: { title: 'Power', look: 'strip' },\n      engines: ['ace'],\n      when: { param: 'samp.sched', is: 'power' },", "      opts: { title: 'Power', look: 'strip' },\n      engines: ['ace'],\n      when: { param: 'samp.sched', is: ['power', 'cosine'] },"]] },
  { id: 'R12', says: 'an engine given to the Prompt without its description (a new engine with no prompt text)', edits: [['modules/prompt.ts', "      undoName: 'the description',\n      default: '',\n      engines: ['ace', 'minimax-m3', 'yue2', 'sa3', 'heartmula'],", "      undoName: 'the description',\n      default: '',\n      engines: ['ace', 'minimax-m3', 'yue2', 'sa3'],"]] },
  { id: 'R15', says: 'a value that is not a number taken from a control (NaN)', core: true, edits: [['../derive/values.ts', "  if (!Number.isFinite(raw)) throw new Error(`${raw} is not a value of ${key}`);", ""]] },
];

/* ---------------------------------------------------------------- the runner
   Each mistake is made on its own copy of ui-next, and several copies are checked at once (one per two cores, eight at
   most), so the whole list takes a few minutes; `--serial` checks one at a time. */
const NL = String.fromCharCode(10);
function fresh(dir: string): void {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  for (const x of ['src', 'tools', 'package.json', 'tsconfig.json']) fs.cpSync(path.join(ROOT, x), path.join(dir, x), { recursive: true });
  fs.symlinkSync(path.join(ROOT, 'node_modules'), path.join(dir, 'node_modules'), 'junction');
}
function run(dir: string, cmd: string, args: string[]): Promise<{ ok: boolean; out: string }> {
  return new Promise(done => execFile(cmd, args, { cwd: dir, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024, env: { ...process.env, HOTSTEP_APP: path.resolve(ROOT, '..') } },
    (err, stdout, stderr) => done({ ok: !err, out: `${stdout ?? ''}${stderr ?? ''}` })));
}
const tsc = path.join(ROOT, 'node_modules', 'typescript', 'bin', 'tsc');
async function check(m: Mutation, dir: string): Promise<{ caught: boolean; lines: string[] }> {
  fresh(dir);
  const lines: string[] = [];
  if (m.shipped) await run(dir, 'node', ['tools/validate-definitions.ts', '--ship']);
  for (const [file, old, now] of m.edits) {
    const p = path.join(dir, D, file), text = fs.readFileSync(p, 'utf8').replace(/\r\n/g, NL);
    if (!text.includes(old)) { lines.push(`the text to change is not in ${file}; update the mutation`); return { caught: false, lines }; }
    fs.writeFileSync(p, text.replace(old, now));
  }
  const [t, v] = await Promise.all([run(dir, 'node', [tsc, '-p', 'tsconfig.json']), run(dir, 'node', ['tools/validate-definitions.ts'])]);
  const r = m.request ? await run(dir, 'node', ['--no-warnings', '--import', './tools/real-app/load.mjs', 'tools/compare-request.ts', '--quick']) : { ok: true, out: '' };
  const c = m.core ? await run(dir, 'node', ['tools/test-core.ts']) : { ok: true, out: '' };
  const tErr = t.out.split(NL).filter(l => l.includes('error TS')).map(l => l.replace(/^.*?error /, 'type: '));
  const vErr = v.out.includes('errors (') ? v.out.slice(v.out.indexOf('errors (')).split(NL).slice(1).map(l => l.trim()).filter(Boolean) : [];
  const rErr = [...r.out.split(NL).filter(l => /new:|tools\/existing-ui|never/.test(l)), ...c.out.split(NL).filter(l => / [1-9]\d* differ($|;)|option labels: [^o]/.test(l))].map(l => l.trim());
  lines.push(...tErr.slice(0, 2), ...vErr.slice(0, 3), ...rErr.slice(0, 2));
  return { caught: !t.ok || !v.ok || !r.ok || !c.ok, lines };
}
const only = process.argv.slice(2).filter(a => !a.startsWith('--'));
const width = process.argv.includes('--serial') ? 1 : Math.max(1, Math.min(8, Math.floor(os.cpus().length / 2)));
const t0 = Date.now();
fresh(`${TMP}-base`);
const base = await Promise.all([run(`${TMP}-base`, 'node', [tsc, '-p', 'tsconfig.json']), run(`${TMP}-base`, 'node', ['tools/validate-definitions.ts'])]);
fs.rmSync(`${TMP}-base`, { recursive: true, force: true });
if (!base.every(r => r.ok)) { console.log('the unchanged copy does not pass:' + NL + base.map(r => r.out).join(NL)); process.exit(1); }
const todo = M.filter(x => !only.length || only.includes(x.id)), results = new Array<{ caught: boolean; lines: string[] }>(todo.length);
let next = 0;
await Promise.all(Array.from({ length: width }, async (_, slot) => {
  for (let i = next++; i < todo.length; i = next++) results[i] = await check(todo[i]!, `${TMP}-${slot}`);
  fs.rmSync(`${TMP}-${slot}`, { recursive: true, force: true });
}));
let bad = 0;
todo.forEach((m, i) => {
  const { caught, lines } = results[i]!;
  const verdict = m.gone ? (caught ? 'CAUGHT (expected to pass now)' : 'passes, as it should: ' + m.gone) : caught ? 'CAUGHT' : 'NOT CAUGHT';
  if (m.gone ? false : !caught) bad++;
  console.log(`${m.id} ${m.says}${NL}   -> ${verdict}`);
  lines.forEach(l => console.log('      ' + l.slice(0, 220)));
});
console.log(`${NL}${todo.length} mistakes in ${((Date.now() - t0) / 1000).toFixed(0)} s, ${width} at a time. ${bad ? `${bad} not caught.` : 'Every mistake caught.'}`);
process.exit(bad ? 1 : 0);
