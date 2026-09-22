// check-song-facts.ts — the details panel reads the right backend's parameters.
// Run: npx tsx ui/scripts/check-song-facts.ts
//
// The bug this guards: generation_params is the whole global params blob, so a
// YuE2 or MM3 row carries ditModel/inferMethod/guidanceScale/mm3Steps from
// panels the render never used. Showing those is what made every non-ACE song
// display AS1.5 numbers. Each case below deliberately includes the leftovers.

import assert from 'node:assert';
import { buildFactGroups, buildTrackChips, songBackend, formatDuration } from '../src/components/details/songFacts';
import type { Song } from '../src/types';

const song = (over: Record<string, unknown>) => over as unknown as Song;

/** Every label the panel would show, flattened. */
const labels = (s: Song) => buildFactGroups(s).flatMap(g => g.facts.map(f => `${g.title}/${f.label}`));
const valueOf = (s: Song, label: string) =>
  buildFactGroups(s).flatMap(g => g.facts).find(f => f.label === label)?.value;

// The AS1.5 leftovers every row carries regardless of which backend ran.
const LEFTOVERS = {
  ditModel: 'acestep-v15-merge-base-sft-turbo-xl-thirds-BF16.gguf',
  lmModel: 'acestep-5Hz-lm-4B-BF16.gguf',
  inferMethod: 'md_hamiltonian_v2',
  inferenceSteps: 50,
  guidanceScale: 20,
  scheduler: 'linear_quadratic',
  guidanceMode: 'dynamic_cfg',
  shift: -1,
  loraPath: 'M:\\Adapters\\dit-xl-thirds\\some_artist\\2026-09-03_R3',
  lmAdapter: 'M:\\Adapters\\lm-4b\\other_artist\\2026-08-13_04-33-46',
  mm3Steps: 20,
  mm3LmAdapterScaleMlp: 0.5,
};

// ── backend detection ────────────────────────────────────────────────────────
assert.equal(songBackend(song({ backend: 'yue2' })), 'yue2');
assert.equal(songBackend(song({ backend: 'minimax-m3' })), 'minimax-m3');
assert.equal(songBackend(song({ generationParams: { backend: 'yue2' } })), 'yue2');
assert.equal(songBackend(song({})), 'ace', 'unstamped rows are ACE-Step');

// ── YuE2: its own fields, none of the leftovers ──────────────────────────────
const yue2 = song({
  backend: 'yue2',
  duration: 195,
  bpm: 0,
  generationParams: {
    ...LEFTOVERS,
    backend: 'yue2',
    seed: 4971534777446333000,
    yue2Cot: 'full',
    yue2NarSolver: 'wasserstein',
    yue2NarScheduler: 'ht_v3',
    yue2OdeSteps: 64,
    yue2VaeVariant: 'standard',
    yue2: {
      ode_steps: 64,
      vae_variant: 'standard',
      end_reason: 'completed',
      stage_end_reasons: { plan: 'skipped', semantic: 'eos' },
      duration_s: 195.5,
      abc_supplied: true,
      score_health: { verdict: 'healthy', bars: 117, vocalShare: 0.795, tempo: 70, meter: '2/4' },
      auto_replan: { attempts: [{ verdict: 'runaway', reason: '267 bars with no vocal' }] },
    },
    yue2Request: { seed: 1, noise_seed: 2, ode_method: 'wasserstein' },
    yue2Adapters: {
      ar: { path: 'M:/Adapters/yue2/an_artist/2026-09-14_ar', trigger: 'an_artist', scales: { global: 1, attn: 1, mlp: 0.5 } },
      nar: { path: 'M:/Adapters/yue2/an_artist/2026-09-14_nar', trigger: 'an_artist', scales: { global: 0.8 } },
    },
  },
});
const yueLabels = labels(yue2);
for (const gone of ['DiT', 'Planner LM', 'DiT adapter', 'LM adapter', 'Flow CFG', 'Max frames']) {
  assert.ok(!yueLabels.some(l => l.endsWith(`/${gone}`)), `YuE2 must not show "${gone}": ${yueLabels}`);
}
assert.equal(valueOf(yue2, 'ODE steps'), '64');
assert.equal(valueOf(yue2, 'Solver'), 'Wasserstein');
assert.equal(valueOf(yue2, 'Schedule'), 'Ht V3');
assert.equal(valueOf(yue2, 'Ended'), 'Completed');
assert.equal(valueOf(yue2, 'Measured length'), '3:16');
assert.equal(valueOf(yue2, 'Score health'), 'Healthy');
assert.equal(valueOf(yue2, 'Bars'), '117 · 80% vocal');
assert.equal(valueOf(yue2, 'Auto-replan'), '1 attempt');
// "plan: skipped" next to "Score: Supplied (ABC)" is the same fact twice.
assert.equal(valueOf(yue2, 'Plan stage'), undefined, 'redundant plan row dropped');
assert.equal(valueOf(yue2, 'Semantic stage'), 'Ended naturally', 'eos in words');
assert.equal(valueOf(yue2, 'Seed'), '4971534777446333000');

assert.equal(valueOf(yue2, 'NAR adapter'), 'an_artist · 2026-09-14_nar');
assert.equal(valueOf(yue2, 'AR adapter'), 'an_artist · 2026-09-14_ar');
assert.equal(valueOf(yue2, 'NAR scale'), '0.8');
assert.equal(valueOf(yue2, 'AR mlp'), '0.5');
assert.equal(valueOf(yue2, 'AR scale'), undefined, 'a scale of 1 is not worth a row');

// bpm 0 and no key: chips must not invent them
const yueChips = buildTrackChips(yue2).map(c => c.label);
assert.deepEqual(yueChips, ['Length'], `YuE2 chips: ${yueChips}`);
assert.equal(buildTrackChips(yue2)[0].value, '3:15');

// ── MM3: its own fields, none of the leftovers ───────────────────────────────
const mm3 = song({
  backend: 'minimax-m3',
  duration: 180,
  bpm: 0,
  generationParams: {
    ...LEFTOVERS,
    backend: 'minimax-m3',
    seed: 12345,
    duration: -1,
    mm3Take: 1,
    mm3Takes: 3,
    mm3BaseSeed: 999,
    mm3LmAdapter: 'M:\\Adapters\\mm3-lm\\an_artist\\2026-09-01_run',
    mm3LmAdapterScale: 0.8,
    mm3: { max_frames: 2400, steps: 600, cfg_flow: 3, prompt_tokens: 128, instrumental: false, sample_rate: 44100 },
  },
});
const mmLabels = labels(mm3);
for (const gone of ['DiT', 'Planner LM', 'CFG scale', 'ODE steps', 'DiT adapter']) {
  assert.ok(!mmLabels.some(l => l.endsWith(`/${gone}`)), `MM3 must not show "${gone}": ${mmLabels}`);
}
assert.equal(valueOf(mm3, 'Steps'), '600', 'mm3.steps, not the ACE inferenceSteps of 50');
assert.equal(valueOf(mm3, 'Flow CFG'), '3');
assert.equal(valueOf(mm3, 'Sample rate'), '44 kHz');
assert.equal(valueOf(mm3, 'Length'), 'Model decides');
assert.equal(valueOf(mm3, 'Take'), '2 of 3');
assert.equal(valueOf(mm3, 'LM adapter'), 'an_artist · 2026-09-01_run');
assert.equal(valueOf(mm3, 'Vocals'), 'Sung');

// ── ACE: the leftovers are its real values ───────────────────────────────────
const ace = song({
  backend: 'ace',
  duration: 210,
  bpm: 138,
  generationParams: {
    ...LEFTOVERS,
    backend: 'ace',
    seed: 42,
    lmSeed: 43,
    keyScale: 'C major',
    loraScale: 0.9,
    useCotCaption: true,
  },
});
assert.equal(valueOf(ace, 'Steps'), '50');
assert.equal(valueOf(ace, 'Solver'), 'Md Hamiltonian V2');
assert.equal(valueOf(ace, 'Shift'), 'Auto', 'shift -1 means auto, not "-1"');
assert.equal(valueOf(ace, 'DiT adapter'), 'some_artist · 2026-09-03_R3');
assert.equal(valueOf(ace, 'LM adapter'), 'other_artist · 2026-08-13_04-33-46');
assert.equal(valueOf(ace, 'Thinking'), 'On');
const aceChips = buildTrackChips(ace).map(c => `${c.label}=${c.value}`);
assert.deepEqual(aceChips, ['Length=3:30', 'Tempo=138 bpm', 'Key=C major']);

// ── formatDuration ───────────────────────────────────────────────────────────
assert.equal(formatDuration(195.5), '3:16');
assert.equal(formatDuration(0), '');
assert.equal(formatDuration(undefined), '');
assert.equal(formatDuration('3:15'), '3:15');
assert.equal(formatDuration(65), '1:05');

console.log('check-song-facts: all assertions passed');
