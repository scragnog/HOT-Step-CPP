import { engineReady } from '../../../engineState.js';
import { isEngineSuspended } from '../../aceEngineProcess.js';
import { heartmulaClient } from './client.js';
import { heartmulaModels, heartmulaSelection, selectHeartmulaModel } from './models.js';
import { mapHeartmulaRequest, runHeartmulaGeneration } from './generate.js';
import type { EngineBackend, BackendExtensionParam } from '../types.js';

const extensions: BackendExtensionParam[] = [
  // LM sampling. Defaults are heartlib's (temperature 1.0, top-k 50, CFG 1.5).
  { key: 'heartmulaTemperature', group: 'lm', label: 'Temperature', hint: 'Sampling temperature for all eight audio codebooks.', type: 'slider', min: 0.01, max: 4, step: 0.01, default: 1 },
  { key: 'heartmulaTopK', group: 'lm', label: 'Top K', hint: 'Number of candidate tokens sampled for each codebook.', type: 'slider', min: 1, max: 8197, step: 1, default: 50 },
  { key: 'heartmulaCfgScale', group: 'lm', label: 'CFG scale', hint: 'LM guidance. A value of 1 uses a single branch and less KV memory.', type: 'slider', min: 1, max: 10, step: 0.05, default: 1.5 },
  { key: 'heartmulaCodecSteps', label: 'Codec steps', hint: 'Euler steps for each HeartCodec audio chunk.', type: 'slider', min: 1, max: 100, step: 1, default: 10 },
  { key: 'heartmulaCodecGuidance', label: 'Codec guidance', hint: 'Guidance scale for HeartCodec flow decoding.', type: 'slider', min: 1, max: 10, step: 0.05, default: 1.25 },
  { key: 'samplerPluginsEnabled', type: 'toggle', label: 'Sampler Plugins', default: false,
    hint: 'Drive the HeartCodec flow decoder with the shared Lua solver, scheduler and guidance plugins instead of its own Euler loop. '
      + 'Off is the reference default; on is experimental, since the plugins were tuned against the ACE-Step DiT. '
      + 'Guidance plugins only run when Codec guidance is above 1.' },
  { key: 'heartmulaFlowShift', label: 'Schedule shift', hint: 'Shift handed to a scheduler plugin. 1 keeps its curve as written.',
    type: 'slider', min: 0.1, max: 10, step: 0.05, default: 1, visible_when: { key: 'samplerPluginsEnabled', equals: 'true' } },
  // Adapter dials, YuE2's grouping. Effective scale = strength x (attention or MLP) x the layer's depth third.
  { key: 'heartmulaAdapterStrength', group: 'adapter', label: 'Strength', hint: 'Overall strength of the selected HeartMuLa LoRA. Zero uses the base model.', type: 'slider', min: 0, max: 2, step: 0.05, default: 1 },
  { key: 'heartmulaAdapterScaleAttn', group: 'adapter', label: 'Attention', hint: 'Scales the adapter on the attention projections (q, k, v, output).', type: 'slider', min: 0, max: 2, step: 0.05, default: 1 },
  { key: 'heartmulaAdapterScaleMlp', group: 'adapter', label: 'MLP', hint: 'Scales the adapter on the feed-forward layers. Only changes adapters trained with "Train MLP layers".', type: 'slider', min: 0, max: 2, step: 0.05, default: 1 },
  { key: 'heartmulaAdapterRebase', group: 'adapter', label: 'Re-base strength (β)', type: 'slider', min: 0, max: 1, step: 0.05, default: 0.75,
    hint: 'For an adapter trained on another variant: blends the weights it targets toward that home model, which needs to be installed. '
      + '1 runs those layers as the home model, 0 applies the adapter to the loaded model unchanged.' },
  { key: 'heartmulaAdapterScaleEarly', group: 'adapter', section: 'Depth thirds', section_hint: 'Scale the adapter by layer depth: early, middle and late thirds of the 28 backbone layers.', label: 'Early', hint: 'Backbone layers 0 to 8.', type: 'slider', min: 0, max: 2, step: 0.05, default: 1 },
  { key: 'heartmulaAdapterScaleMid', group: 'adapter', section: 'Depth thirds', label: 'Mid', hint: 'Backbone layers 9 to 17.', type: 'slider', min: 0, max: 2, step: 0.05, default: 1 },
  { key: 'heartmulaAdapterScaleLate', group: 'adapter', section: 'Depth thirds', label: 'Late', hint: 'Backbone layers 18 to 27.', type: 'slider', min: 0, max: 2, step: 0.05, default: 1 },
];

export const heartmulaBackend: EngineBackend = {
  id: 'heartmula', displayName: 'HeartMuLa', resourcePool: 'gpu',
  lifecycle: {
    async start() {}, async stop() { await heartmulaClient.unload(); },
    status() {
      return isEngineSuspended() ? 'suspended' : engineReady && heartmulaClient.cachedProps()?.synth_ready ? 'ready' : 'down';
    },
  },
  async capabilities() {
    const { props, stale } = await heartmulaClient.props();
    return {
      backend: 'heartmula', up: engineReady && !isEngineSuspended() && props?.synth_ready === true,
      core: {
        duration: { max: 240, auto: true }, bpm: false, keyscale: false, negativePrompt: false,
        batch: { max: 1 }, seed: true, sampleRate: 48000, captionSource: 'none',
        propsStale: stale, modelsMissing: !stale && props != null && !props.synth_ready,
      },
      features: {
        models: true, lm: false, plugins: false, samplerPlugins: true, adapters: false,
        lmAdapters: true, lmAdapterSelectable: true, postProcess: true, stableStep: true,
        whisper: false, lyricTimestamps: false, forcedAlignment: false, cover: true,
        repaint: false, lego: false, extract: false, streaming: false, training: false,
        midi: false, stems: false, understand: false, conceptSteering: false,
      },
      extensions,
    };
  },
  models: heartmulaModels, selectModel: selectHeartmulaModel,
  operations: ['text2music', 'cover'],
  resolveRequest(submission) {
    const models = heartmulaSelection();
    for (const [key, target] of Object.entries({
      heartmulaVariant: 'variant', heartmulaGlobalModel: 'global', heartmulaLocalModel: 'local',
      heartmulaFlowModel: 'flow', heartmulaScalarDecoderModel: 'scalar_decoder', heartmulaAdapter: 'lmAdapter',
    })) {
      if (submission[key] !== undefined) {
        if (typeof submission[key] !== 'string') throw new Error(`${key} must be a string`);
        models[target] = submission[key];
      }
    }
    const wire = mapHeartmulaRequest(submission, models);
    return {
      operation: typeof submission.taskType === 'string' ? submission.taskType : 'text2music',
      common: { caption: wire.tags, lyrics: wire.lyrics, duration: wire.duration_s,
        ...(typeof wire.seed === 'number' ? { seed: wire.seed } : {}) },
      models, options: wire as unknown as Record<string, unknown>,
      policy: { retry: { maxAttempts: 1, reseedOnRetry: false } },
    };
  },
  generate: runHeartmulaGeneration,
  releaseVram: () => heartmulaClient.unload(),
};
