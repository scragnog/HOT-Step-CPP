// How the existing Create page holds each intent the definitions can express: the second statement the request
// comparison (tools/compare-request.ts) checks the wires against. It is written from the existing UI's controls
// (ui/src), not from the wires, so a wire that sends the wrong thing shows as a difference.
//
// A parameter is listed here when its wire is more than a plain name (a scale, an inversion, a map, a condition, a
// transform, several fields, auto), and for its state when it has state. A parameter that is not listed is held as
// its wire says, unchanged: `param:x` in the store's `x`, `ext:x` in the store's backendParams, `content:x` in the
// Create form's `x`, `intent:compose.x` in CreatePanel's switch. The comparison refuses a wire that is more than a
// plain name and is not listed, so a new transforming wire needs its line here.
//
// Targets:  store:<field>         the global params store, set through its own setter (setLufsPreset sets the target)
//           store:backendParams.<key>   a backend knob (setBackendParam)
//           content:<field>       CreatePanel's content (CreateContent)
//           compose:<field>       CreatePanel's compose switches (autoExpand, loraTrigger, beatIntro, introBars)
//           selection:<bucket>    the model the server has selected (the engine panel's choice)

import type { EngineId } from '../src/definitions/engines.ts';
import type { Condition, Json, Value } from '../src/definitions/types.ts';

export interface Hold {
  to: `store:${string}` | `content:${string}` | `compose:${string}` | `selection:${string}`;
  times?: number;
  invert?: true;
  /** The existing control's value for each of the definition's; a value missing here is one it cannot hold. */
  map?: { [value: string]: Json };
  /** Held as text (the existing control is a text box), or as a whole number typed as text. */
  as?: 'text' | 'number';
  /** What the existing control holds for "auto"; absent: it has no auto, and its own default stays. */
  auto?: Json;
  /** Held only while this holds on the desk; otherwise `else`, or nothing. */
  when?: Condition;
  else?: Json;
  /** Values the existing page holds by leaving its control alone (a model at "Engine default"). */
  leave?: Value[];
  /** Hold that parameter's value here instead (turning the denoiser on sets its strength). */
  from?: string;
}
export type Holding = Hold | Hold[] | { engines: EngineId[]; hold: Hold | Hold[] }[];

export const HOLDINGS: { [key: string]: Holding } = {
  /* DCW: in High mode the existing page's one "Scaler" slider is dcwHighScaler (GenerationDropdown.tsx) */
  'dcw.low': [{ to: 'store:dcwLowScaler', when: { param: 'dcw.mode', not: 'high' } }, { to: 'store:dcwHighScaler', when: { param: 'dcw.mode', is: 'high' } }],
  /* The caption source is CreatePanel's own state, put in the intent whole (CAPTION_SOURCE below) */
  'prompt.text': [{ engines: ['minimax-m3'], hold: { to: 'content:caption' } }],
  'cap.src': [],
  'cap.ds': [],
  /* Tempo: BPM and length are 0 and -1 on auto (MetadataSection) */
  'tempo.bpm': { to: 'content:bpm', auto: 0 },
  'tempo.dur': { to: 'content:duration', auto: -1 },
  /* YuE2's longest length is minutes (Yue2 controls); auto-trim fade is milliseconds (GenerationDropdown) */
  'tempo.max': { to: 'store:backendParams.yue2MaxMinutes', times: 1 / 60 },
  'tempo.trimfade': { to: 'store:autoTrimFadeMs', times: 1000 },
  /* Adapters: Simple mode's strength and the stack's first entry are separate store fields (AdaptersDropdown) */
  s0: [{ engines: ['ace'], hold: [{ to: 'store:adapterScale' }, { to: 'store:adapterStack[0].scale' }] }],
  'ad.adv': { to: 'store:advancedAdapters', map: { simple: false, advanced: true } },
  'ad.mq': { to: 'store:adapterMergeLowVram', map: { hq: false, low: true } },
  /* Planner: "Skip LM" is the existing switch */
  'lm.on': { to: 'store:skipLm', invert: true },
  /* YuE2's composer CFG is a text box, empty for the checkpoint's own */
  'yue.cfg': { to: 'store:backendParams.yue2CfgScale', as: 'text', auto: '' },
  /* Shift on ACE-Step: the existing slider's Auto button sets -1 (GenerationDropdown.tsx, "gp.setShift(-1)") */
  'samp.shift': [{ engines: ['ace'], hold: { to: 'store:shift', auto: -1 } }],
  /* The retake seed is a number field, -1 for a fresh one */
  'samp.rseed': { to: 'store:retakeSeed', as: 'number' },
  /* The denoiser and LSS have no switch: on is a strength above 0, off is 0 (LatentDropdown) */
  'den.on': { to: 'store:denoiseStrength', from: 'den.str', when: { param: 'den.on', is: true }, else: 0 },
  'den.str': { to: 'store:denoiseStrength', when: { param: 'den.on', is: true }, else: 0 },
  'lss.on': { to: 'store:lssStrength', from: 'lss.str', when: { param: 'lss.on', is: true }, else: 0 },
  'lss.str': { to: 'store:lssStrength', when: { param: 'lss.on', is: true }, else: 0 },
  /* The tiled decoder: the existing page starts with no plugin chosen, so turning it on is choosing one too */
  'pp.tile.on': [{ to: 'store:postprocessEnabled' }, { to: 'store:postprocessPlugin', from: 'pp.tile.plugin', when: { param: 'pp.tile.on', is: true } }],
  /* Seed: "Random" is the existing switch */
  'seed.lock': { to: 'store:randomSeed', invert: true },
  /* Percent knobs over 0..1 store fields (PostProcessingDropdown) */
  'pp.refiner.mix': { to: 'store:mm3RefinerMix', times: 0.01 },
  'pp.ss.str': { to: 'store:stableStepStrength', times: 0.01 },
  'pp.ss.seed': { to: 'store:stableStepSeed', as: 'number' },
  /* The normalizer's preset is a store field whose setter sets the target (setLufsPreset) */
  'pp.norm.target': { to: 'store:lufsPreset' },
  /* ...and its Custom slider is drawn only on Custom (PostProcessingDropdown.tsx, lufsPreset === 'custom') */
  'pp.norm.custom': { to: 'store:lufsTarget', when: { param: 'pp.norm.target', is: 'custom' } },
  /* Lyric timestamps: "Skip LRC" is the existing switch */
  'pp.lrc.on': { to: 'store:skipLrc', invert: true },
  /* HeartMuLa's models are the engine panel's selection; "Engine default" leaves it as the server has it */
  'model.var': { to: 'selection:heartmula.variant', leave: [''] },
  'model.dit': { to: 'selection:heartmula.global', leave: [''] },
  'model.lm': { to: 'selection:heartmula.local', leave: [''] },
  'model.x': { to: 'selection:heartmula.flow', leave: [''] },
};

/** Fields several parameters make together, as the existing control writes them. */
export const PACKED: { engines: EngineId[]; keys: string[]; to: `store:${string}`; make(v: { [key: string]: Value | undefined }): Json | undefined }[] = [
  {
    engines: ['ace', 'minimax-m3', 'sa3', 'heartmula'],
    /* GenerationDropdown.tsx: setScheduler(`beta:${a.toFixed(2)}:${b.toFixed(2)}`), `power:${v.toFixed(2)}`,
       `composite:${a}+${b}:${c.toFixed(2)}:${s.toFixed(2)}`, or the schedule's id */
    keys: ['samp.sched', 'sch.alpha', 'sch.beta', 'sch.exp', 'sch.a', 'sch.b', 'sch.x', 'sch.split'],
    to: 'store:scheduler',
    make: v => {
      const s = v['samp.sched'], n = (k: string, d: number) => (typeof v[k] === 'number' ? (v[k] as number) : d).toFixed(2);
      if (s === undefined) return undefined;
      if (s === 'beta') return `beta:${n('sch.alpha', 0.5)}:${n('sch.beta', 0.7)}`;
      if (s === 'power') return `power:${n('sch.exp', 2)}`;
      if (s === 'composite') return `composite:${v['sch.a'] ?? 'bong_tangent'}+${v['sch.b'] ?? 'linear'}:${n('sch.x', 0.5)}:${n('sch.split', 0.5)}`;
      return s as string;
    },
  },
];

interface Slot { file: string; type?: string; ph?: [number, number] }
/** Module state, as the existing page holds it. Each returns the targets to set. */
export const STATE_HOLDINGS: { [key: string]: (v: Json, e: EngineId) => [Hold['to'], Json][] } = {
  /* ACE-Step: Simple mode's adapter, and the stack with each entry's timestep window (AdaptersDropdown.tsx
     setWindowPct: stepStart = 1 - end%, stepEnd = 1 - start%; a full window is none) */
  slots: (v, e) => {
    const slots = (Array.isArray(v) ? v : []) as unknown as Slot[];
    if (e === 'ace') {
      return [['store:adapter', slots[0]?.file ?? ''], ['store:adapterStack', slots.map(s => ({
        path: s.file, scale: 1,
        ...(s.ph && (s.ph[0] > 0 || s.ph[1] < 100) ? { stepStart: +(1 - s.ph[1] / 100).toFixed(4), stepEnd: +(1 - s.ph[0] / 100).toFixed(4) } : {}),
      }))]];
    }
    if (e === 'minimax-m3') return slots.length ? [['store:backendParams.mm3LmAdapter', slots[0]!.file]] : [];
    if (e === 'heartmula') return [['selection:heartmula.lmAdapter', slots[0]?.file ?? '']];
    return [];
  },
  /* setSeed keeps a safe integer as a number (utils/seed.ts normalizeSeed) */
  seed: v => (typeof v === 'number' ? [['store:seed', v]] : []),
  timbreFile: v => (typeof v === 'string' && v ? [['store:timbreAudioPath', v]] : []),
  /* CreatePanel's imported latent (hs-sourceLatentUrl); the concept imports one on ACE-Step only */
  latent: (v, e) => (e === 'ace' && typeof v === 'string' && v ? [['content:sourceLatentUrl', v]] : []),
  masterRef: v => (typeof v === 'string' && v ? [['store:masteringReference', v]] : []),
};

/** The caption source CreatePanel puts in the intent while there are tracks to pick from (CreatePanel.tsx
 *  createIntent): the handoff's song caption and tracks on MiniMax-Music3, the dataset on YuE2. The source chooser's
 *  value is the mode, or a track's name. */
export const CAPTION_SOURCE = (e: EngineId, v: { source: string; dataset: string; caption: string; tracks: { title: string; bpm?: number; caption: string }[]; adapter: boolean }): Json | undefined => {
  if (!v.tracks.length) return undefined;
  const mode = v.source === 'custom' || v.source === 'auto' ? v.source : 'track';
  if (e === 'minimax-m3') return { engine: 'minimax-m3', customCaption: v.caption, selection: mode === 'track' ? { mode, selectedTitle: v.source } : { mode }, tracks: v.tracks };
  if (e === 'yue2') return { engine: 'yue2', datasetId: v.dataset, adapterInForce: v.adapter, selection: mode === 'track' ? { mode, selectedName: v.source } : { mode } };
  return undefined;
};

/** YuE2's continuation is the recording and the part of it, together (Yue2 continuation panel). */
export const CONTINUATION = (file: Json | undefined, span: Json | undefined): Json | undefined => {
  const s = (span ?? {}) as { start?: number; end?: number };
  return typeof file === 'string' && file ? { sourceAudioUrl: file, start: s.start ?? 0, end: s.end ?? 30 } : undefined;
};
