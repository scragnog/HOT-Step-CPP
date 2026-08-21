// sendAuditionToCustomGen.ts — replay an audition side as a Custom-Gen setup
//
// Applies the MIRRORED-GENERATION RECIPE (parity proven 2026-08-12: identical
// CoT plan, identical codes, bit-identical audio) so that pressing Generate on
// the Create page reproduces the audition's DiT render exactly — until the
// user starts changing parameters, which is the point.
//
// The recipe is not "copy the audition fields and hope": every generation knob
// that would add something the audition never sent (guidance 9, APG, DCW,
// post-processing, codes-strength windowing, timbre reference, …) is
// explicitly reset to the engine-default-equivalent value. A single leftover
// slider silently breaks bit-parity, which is why this list is exhaustive
// rather than minimal.
//
// CAPTION ASYMMETRY (the one subtle part): the audition prompt was the TAGGED
// caption (dataset trigger applied server-side). A generation re-applies every
// loaded adapter's EMBEDDED trigger to the caption with no skip-if-present
// (translateParams:251-260), so:
//   • LM-adapter side → send the UNTAGGED caption; the adapter's embedded
//     trigger re-tags it into the identical string. Sending the tagged one
//     would double the tag.
//   • base-LM side → nothing injects; send the TAGGED caption verbatim.
// Previews recorded before captionInput existed fall back to the tagged
// caption for both sides (adapter side then risks a doubled tag — rerun the
// audition for an exact send).

import { useGlobalParamsStore, scopedKey } from '../../stores/globalParamsStore';
import { writePersistedState } from '../../hooks/usePersistedState';
import type { AuditionPreview, AuditionSideResult } from '../../services/trainingApi';

/** Which half of the render matrix to reproduce: the bare-DiT render or the
 *  DiT-adapter one. 'adapter' is only meaningful when the preview carries
 *  renderDitAdapter. */
export type AuditionRenderCell = 'bare' | 'adapter';

export function sendAuditionToCustomGen(
  preview: AuditionPreview,
  side: AuditionSideResult,
  cell: AuditionRenderCell,
): void {
  const gps = useGlobalParamsStore.getState();
  const write = (key: string, value: unknown) => writePersistedState(key, value);

  // ── Content (CreatePanel state) ─────────────────────────────────────────
  const caption = side.lmAdapter
    ? (preview.captionInput ?? preview.caption)
    : preview.caption;
  write('hs-caption', caption || '');
  write('hs-lyrics', preview.lyrics || '');
  write('hs-instrumental', false);
  write('hs-negative-prompt', '');          // audition sent none — engine default
  write('hs-lora-trigger', '');             // embedded triggers only, no manual override
  write('hs-beat-intro', false);            // compose-time caption mutators off
  write('hs-bpm', preview.bpm ?? 0);
  write('hs-keyScale', preview.keyscale ?? '');
  write('hs-timeSignature', preview.timesignature ?? '');
  write('hs-duration', preview.durationSec || 180);
  write('hs-vocalLanguage', 'en');          // both audition passes pin en
  write('hs-sourceLatentUrl', '');          // plain text2music, never a cover source

  // ── Models ──────────────────────────────────────────────────────────────
  // The render DiT when the preview has one (pinned to the adapter's training
  // base when adapter renders ran), else the detok DiT the codes family owns.
  gps.setDitModel(preview.renderDitModel || preview.ditModel || '');
  gps.setLmModel(preview.lmModel || '');
  gps.setVaeModel(preview.vaeModel || '');
  gps.setEmbeddingModel('');                // audition render used the engine default

  // ── Adapters ────────────────────────────────────────────────────────────
  gps.setLmAdapter(side.lmAdapter || '');
  gps.setLmAdapterScale(side.lmAdapterScale ?? 1.0);
  const ditAdapter = cell === 'adapter' ? (preview.renderDitAdapter || '') : '';
  gps.setAdapter(ditAdapter);
  gps.setAdapterScale(1.0);
  gps.setAdapterStack([]);                  // a persisted Advanced stack must not override
  gps.setAdvancedAdapters(false);
  gps.setAdapterMode('runtime');            // what the adapter render sent
  // The audition render sent NO group scales → engine default is ALL 1.0 —
  // including time_embed/proj_in, which the UI's own default zeroes.
  gps.setAdapterGroupScales({
    self_attn: 1.0, cross_attn: 1.0, mlp: 1.0, cond_embed: 1.0, time_embed: 1.0, proj_in: 1.0,
  });
  gps.setRebaseSource('');

  // ── Seeds / batch ───────────────────────────────────────────────────────
  gps.setSeed(preview.seed);
  gps.setRandomSeed(false);
  gps.setLmSeedFollowsDit(true);            // audition set lm_seed = seed
  gps.setBatchSize(1);

  // ── DiT sampling — engine-default-equivalent values (the recipe) ────────
  gps.setInferenceSteps(preview.renderSteps || 8);
  gps.setGuidanceScale(1.0);                // 0/absent = auto = 1.0; explicit 1.0 matches
  gps.setShift(0);                          // 0 = engine auto (what the render used)
  gps.setInferMethod('euler');              // engine default solver
  gps.setScheduler('linear');
  gps.setGuidanceMode('apg');               // engine default; a no-op at guidance 1.0
  gps.setApgMomentum(0.75);                 // apg sub-params back to defaults —
  gps.setApgNormThreshold(2.5);             // sent whenever guidanceMode is apg
  gps.setCfgCutoffRatio(1.0);
  gps.setLmCfgCutoffRatio(1.0);
  gps.setCacheRatio(0);
  gps.setCustomTimesteps('');
  gps.setDcwEnabled(false);
  gps.setLatentShift(0);
  gps.setLatentRescale(1);
  gps.setDenoiseStrength(0);
  gps.setLssStrength(0);
  // Solver/scheduler plugin params back to declared defaults.
  useGlobalParamsStore.setState({ pluginParams: {} });
  // scopedKey(): pluginParams is per backend, so clear the ACTIVE backend's
  // copy — the bare key belongs to ACE-Step.
  try { localStorage.setItem(scopedKey('hs-pluginParams'), '{}'); } catch { /* full */ }

  // ── LM / Thinking — exactly what buildLmRequest sent ────────────────────
  gps.setSkipLm(false);
  gps.setUseCotCaption(true);
  gps.setLmTemperature(preview.lmTemperature ?? 0.85);
  gps.setLmTopP(preview.lmTopP ?? 0.9);
  gps.setLmCfgScale(preview.lmCfgScale ?? 2.0);
  gps.setLmRepPenalty(preview.lmRepPenalty ?? 1.1);
  // The audition sent no mode/window → engine defaults (presence/64). Both ARE
  // sent by a generation whenever the penalty is > 1.0, so pin them.
  gps.setLmRepMode('presence');
  gps.setLmRepWindow(64);
  gps.setLmTopK(0);
  // Absent and 'NO USER INPUT' are the same phase-2 uncond (prompt.h:446,
  // exonerated 2026-08-12) — but a CUSTOM negative prompt is not. Reset to
  // the default so a leftover experiment can't shift the codes.
  gps.setLmNegativePrompt('NO USER INPUT');
  gps.setLmCodesStrength(1.0);              // no audio_cover_strength windowing
  gps.setLmCodesMode('ratio');

  // ── Everything downstream of the raw render: OFF ────────────────────────
  gps.setPostProcessingEnabled(false);
  gps.setStableStepOn(false);
  gps.setTimbreReference(false);
  gps.setTimbreAudioPath('');               // sent unconditionally when set — must clear
  gps.setAutoTrimEnabled(false);
  gps.setSkipLrc(true);
  gps.setCoverArtEnabled(false);
  gps.setQualityEvalEnabled(false);
  gps.setWhisperLyricsEnabled(false);

  // ── App settings: LM codes cache off (recipe) ───────────────────────────
  // computeLmCacheKey is a manual whitelist ([[lm-echo-sideband-gotcha]]) — a
  // hit keyed on a stale subset of knobs would silently substitute wrong
  // codes, so the replay always runs the LM for real.
  try {
    const raw = localStorage.getItem('ace-settings');
    const settings = raw ? JSON.parse(raw) : {};
    settings.cacheLmCodes = false;
    localStorage.setItem('ace-settings', JSON.stringify(settings));
  } catch { /* unreadable settings — the cache stays as it was */ }

  console.log(
    `[TrainingStudio] Send to Custom-Gen: seed ${preview.seed}, ` +
    `lmAdapter=${side.lmAdapter || '(base)'}, ditAdapter=${ditAdapter || '(none)'}, ` +
    `dit=${preview.renderDitModel || preview.ditModel || '(default)'}`);

  // ── Navigate to the Create page (same mechanism as Lyric Studio's send) ─
  window.history.pushState({}, '', '/');
  window.dispatchEvent(new PopStateEvent('popstate'));
}
