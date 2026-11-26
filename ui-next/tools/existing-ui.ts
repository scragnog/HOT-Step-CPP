// Which control of the existing Create page a user would use for each parameter: the second statement the request
// comparison (tools/compare-request.ts) checks the wires against. Each line names a control by where it is and what
// it says on the existing page ("Generation > Shift", "Post-Processing > StableStep > Refine strength"), as
// tools/real-app/page.ts reads the page; the comparison then uses that control through the page's own handler, and
// the page's own code decides which store field changes and what the request carries. Nothing here names a request
// field, so a wire that sends a value to the wrong field, in the wrong unit or under the wrong condition shows as a
// difference. Written from the existing page as it is drawn, not from the wires.
//
// A parameter with no line here is not driven: the comparison reports it as having no counterpart on the existing
// page, and its fields as "new, no reference". So a new parameter needs no line; a parameter for something the
// existing page has a control for gets one, which the comparison asks for when its field differs.
//
// What cannot be set through a control in Node is restated below, and counted as restated: a file or a seed chosen
// in a dialog, models chosen in the engine panel, the caption source's tracks (state the page fetches).

import type { EngineId } from '../src/definitions/engines.ts';
import type { Condition, Json, Value } from '../src/definitions/types.ts';

/** How the existing page's control is used for a parameter's value. */
export interface Link {
  /** The control, by its place and words on the existing page. Words after " — " in a label (a live read-out) are not
   *  part of it. */
  control?: string;
  /** Pressed rather than set: a button for a switch (pressed when the value is on), or the button for each value. */
  press?: string | { [value: string]: string };
  /** A button that shows the control, pressed first when the page does not show it (MiniMax-Music3's depth thirds). */
  reveal?: string;
  /** The existing control's value for the definition's: a factor, an inversion, a map, as text. */
  times?: number;
  invert?: true;
  map?: { [value: string]: Json };
  as?: 'text' | 'number';
  /** On auto: the value the control is set to (BPM 0), or nothing (the control is left). */
  auto?: Json;
  /** Used only while this holds on the desk; otherwise `else` is set, or the control is left. */
  when?: Condition;
  else?: Json;
  /** Values the existing page holds by leaving the control alone. */
  leave?: Value[];
  /** Set this parameter's value instead (turning the denoiser on sets its strength). */
  from?: string;
}
type One = string | Link;
/** A line: one control on every engine, several, or one per engine. */
export type Holding = One | One[] | { on: { [e in EngineId]?: One | One[] } };

const PP = 'Post-Processing';
const SS = `${PP} > StableStep`;
export const LINKS: { [key: string]: Holding } = {
  /* the Create form (CreatePanel, ContentSection, MetadataSection) */
  'prompt.text': { on: { ace: 'Create > Style Description', 'minimax-m3': 'Create > Style Description', yue2: 'Create > Style Description', sa3: 'Create > Style Description', heartmula: 'Create > Tags' } },
  'wild.auto': { on: { ace: { press: 'Create > ⚄ auto' }, 'minimax-m3': { press: 'Create > ⚄ auto' }, sa3: { press: 'Create > ⚄ auto' }, heartmula: { press: 'Create > ⚄ auto' } } },
  'ad.trigger': 'Create > LoRA',
  'song.artist': 'Create > Artist',
  'song.title': 'Create > Title',
  'song.subject': 'Create > Subject',
  'lyrics.text': 'Create > Lyrics',
  'lyrics.instr': 'Create > Instrumental (no vocals)',
  'lyrics.lang': { on: { ace: 'Create > Vocal Language', 'minimax-m3': 'Create > Lyrics Language', yue2: 'Create > Vocal Language', heartmula: 'Create > Vocal Language' } },
  'tempo.bpm': { control: 'Create > BPM', auto: 0 },
  'tempo.key': 'Create > Key',
  'tempo.sig': 'Create > Time Sig',
  'tempo.dur': { control: 'Create > Duration', auto: -1 },
  'tempo.beatio': { press: 'Create > Beat I/O' },
  'tempo.beatbars': { press: { 1: 'Create > 1', 2: 'Create > 2', 4: 'Create > 4', 8: 'Create > 8' } },
  'avoid.text': 'Create > Negative Prompt',
  /* the bar's Generation panel */
  'tempo.trim': 'Generation > Auto-Trim Endings',
  'tempo.trimbuf': 'Generation > Auto-Trim Endings > Duration Buffer (seconds)',
  'tempo.trimfade': 'Generation > Auto-Trim Endings > Fade-Out (seconds)',
  'tempo.max': { control: 'Generation > Max Song Length (min)', times: 1 / 60 },
  'tempo.incap': 'Generation > Include BPM/key in caption',
  'timbre.str': 'Generation > Reference Strength',
  'samp.steps': { on: { ace: 'Generation > Inference Steps', 'minimax-m3': 'Generation > Flow Steps', yue2: 'Generation > ODE Steps', sa3: 'Generation > Steps', heartmula: 'Generation > Codec steps' } },
  'samp.cfg': { on: { ace: 'Generation > Guidance Scale', 'minimax-m3': 'Generation > Flow Guidance (CFG)', heartmula: 'Generation > Codec guidance' } },
  /* Shift on ACE-Step: the slider under "Shift", and its Auto button (-1) */
  'samp.shift': { on: { ace: { control: 'Generation > Shift', auto: -1 }, 'minimax-m3': 'Generation > Schedule Shift', heartmula: 'Generation > Schedule shift' } },
  'samp.plugins': 'Generation > Sampler Plugins',
  'samp.solver': { on: { ace: 'Generation > Solver', 'minimax-m3': 'Generation > Solver', yue2: 'Generation > NAR Solver', sa3: 'Generation > Solver', heartmula: 'Generation > Solver' } },
  'samp.sched': { on: { ace: 'Generation > Schedule', 'minimax-m3': 'Generation > Schedule', yue2: 'Generation > NAR Scheduler', sa3: 'Generation > Schedule', heartmula: 'Generation > Schedule' } },
  'samp.guide': 'Generation > Guidance',
  'sch.alpha': 'Generation > Alpha (α)',
  'sch.beta': 'Generation > Beta (β)',
  'sch.exp': 'Generation > Exponent',
  'sch.a': 'Generation > Stage A',
  'sch.b': 'Generation > Stage B',
  'sch.x': 'Generation > Crossover',
  'sch.split': 'Generation > Split',
  'samp.cs': 'Generation > CFG interval start',
  'samp.ce': 'Generation > CFG interval end',
  'samp.retake': 'Generation > Retake variance',
  'samp.rseed': { control: 'Generation > Retake seed', as: 'number' },
  /* APG's two settings: ACE-Step's Generation panel only */
  'apg.mom': { on: { ace: 'Generation > Momentum' } },
  'apg.norm': { on: { ace: 'Generation > Norm Threshold' } },
  'perf.cfgcut': 'Generation > CFG Cutoff',
  'perf.lmcut': 'Generation > LM CFG Cutoff',
  'perf.cache': 'Generation > Step Cache',
  'lat.ts': 'Generation > Custom Timesteps',
  'lat.shift': 'Generation > Latent Shift',
  'lat.rescale': 'Generation > Latent Rescale',
  'dcw.on': 'Generation > DCW Correction',
  'dcw.mode': 'Generation > DCW Correction > Correction Mode',
  /* the first DCW knob is the low scaler, or in High mode the one "Scaler" the page shows */
  'dcw.low': [
    { control: 'Generation > DCW Correction > Low-Freq Scaler', when: { param: 'dcw.mode', is: 'double' } },
    { control: 'Generation > DCW Correction > Scaler', when: { param: 'dcw.mode', not: 'double' } },
  ],
  'dcw.high': 'Generation > DCW Correction > High-Freq Scaler',
  /* the denoiser and LSS have a switch whose "on" sets a strength; a strength set while off is never typed */
  'den.on': 'Generation > Denoiser',
  'den.str': { control: 'Generation > Denoiser > Strength', when: { param: 'den.on', is: true } },
  'den.smooth': 'Generation > Denoiser > Smoothing',
  'den.mix': 'Generation > Denoiser > Mix',
  'lss.on': 'Generation > LSS',
  'lss.str': { control: 'Generation > LSS > Strength', when: { param: 'lss.on', is: true } },
  'lss.var': 'Generation > LSS > Var Threshold',
  'lss.dc': 'Generation > LSS > DC Remove',
  'seed.lock': { control: 'Generation > Random', invert: true },
  'takes.n': { on: { ace: 'Generation > Batch Size', 'minimax-m3': 'Generation > Variations Per Render', yue2: 'Generation > Batch Size' } },
  /* the bar's LM / Thinking panel (ACE-Step) */
  'lm.on': 'LM / Thinking > LM / Thinking',
  'lm.cot': 'LM / Thinking > Chain-of-Thought Caption',
  'lm.temp': 'LM / Thinking > Temperature',
  'lm.cfg': 'LM / Thinking > CFG Scale',
  'lm.topk': 'LM / Thinking > Top-K',
  'lm.topp': 'LM / Thinking > Top-P',
  'lm.minp': 'LM / Thinking > Min-P',
  'lm.plaus': 'LM / Thinking > CFG Plausibility',
  'lm.resc': 'LM / Thinking > CFG Rescale',
  'lm.logp': 'LM / Thinking > CFG in Log-Probabilities',
  'lm.rep': 'LM / Thinking > Repetition Penalty',
  'lm.repmode': 'LM / Thinking > Rep. Mode',
  'lm.win': 'LM / Thinking > Rep. Window (codes)',
  'lm.drybase': 'LM / Thinking > DRY Base',
  'lm.drymin': 'LM / Thinking > DRY Min Match',
  'lm.neg': 'LM / Thinking > Negative Prompt',
  'lm.codesmode': { control: 'LM / Thinking > Step Count', map: { steps: true, ratio: false } },
  'lm.codesteps': 'LM / Thinking > Codes Steps',
  'lm.codes': 'LM / Thinking > Strength',
  'lm.seedlock': 'LM / Thinking > Use DiT Seed',
  /* YuE2's declared knobs, drawn by the page from the server's manifest */
  'lm.cotmode': 'Generation > Chain of Thought',
  'lm.preview': 'Generation > Preview the score first',
  'yue.cfg': { control: 'Generation > CFG Scale', as: 'text', auto: '' },
  'yp.temp': 'LM / Thinking > Plan Temperature',
  'yp.topp': 'LM / Thinking > Plan Top-P',
  'yp.topk': 'LM / Thinking > Plan Top-K',
  'yp.rep': 'LM / Thinking > Plan Repetition Penalty',
  'yp.win': 'LM / Thinking > Plan Repetition Window',
  'yc.temp': 'LM / Thinking > Composer Temperature',
  'yc.topp': 'LM / Thinking > Composer Top-P',
  'yc.topk': 'LM / Thinking > Composer Top-K',
  'yc.minp': 'LM / Thinking > Composer Min-P',
  'yc.rep': 'LM / Thinking > Composer Repetition Penalty',
  'yc.win': 'LM / Thinking > Composer Repetition Window',
  'yue.abc': 'Generation > Lead sheet (ABC)',
  'yue.timing': 'Generation > Lyric Timing (C6)',
  'yue.replan': 'Generation > Re-plan runaway scores',
  'yue.replanflag': 'Generation > Re-plan flagged scores',
  'yue.tries': 'Generation > Re-plan tries',
  'yue.recompose': 'Generation > Recompose runaway songs',
  'yue.pws': 'Generation > Plan Strength (start)',
  'yue.pwe': 'Generation > Plan Strength (end)',
  'yue.pwc': 'Generation > Plan Strength Curve',
  'yue.pcfg': 'Generation > Plan CFG',
  'yue.pcfg1': 'Generation > Plan CFG From t',
  'yue.pcfg2': 'Generation > Plan CFG To t',
  'yue.pcfgr': 'Generation > Plan CFG Rescale',
  'yue.pcfglm': 'Generation > Plan CFG Level Match',
  'yue.cads': 'Generation > Plan Conditioning Noise (CADS)',
  'yue.cads1': 'Generation > CADS Clean Below t',
  'yue.cads2': 'Generation > CADS Full Noise Above t',
  'yue.cadsr': 'Generation > CADS Rescale',
  'yue.chunk': 'Generation > NAR Chunk Length (s)',
  'yue.coalesce': 'Generation > Batch Queued Songs',
  'yue.noisevar': 'Generation > Noise Variations',
  'yue.narcache': 'Generation > NAR Cache Ratio',
  'yn.wassersteinflow.0': 'Generation > Proximal Scale',
  'yn.wassersteinflow.1': 'Generation > Element Energy Weight',
  'yn.wassersteinflow.2': 'Generation > RMS Weight',
  'yn.wassersteinflow.3': 'Generation > RMS Target (0 = Auto)',
  'yn.wassersteinflow.4': 'Generation > Sigma Gate',
  'yn.wassersteinflow.5': 'Generation > Proximal Iterations',
  'yn.wassersteinflow.6': 'Generation > Transverse Projection',
  'yn.wassersteinflow.7': 'Generation > Auto Target RMS',
  'yn.latentchanneleq.0': 'Generation > Channel Group',
  'yn.latentchanneleq.1': 'Generation > Gain',
  'yn.latentchanneleq.2': 'Generation > EQ From t',
  'yn.latentchanneleq.3': 'Generation > EQ To t',
  'yn.latentchanneleq.4': 'Generation > Group Gains',
  'yn.latentchanneleq.5': 'Generation > Section Only',
  'yn.timesteplie.0': 'Generation > Lie (t offset)',
  'yn.timesteplie.1': 'Generation > Lie Curve',
  'yn.timesteplie.2': 'Generation > Lie Window t1',
  'yn.timesteplie.3': 'Generation > Lie Window t2',
  'yn.timesteplie.4': 'Generation > Section Lies',
  'yn.planoverdrive.0': 'Generation > Drive (peak plan strength)',
  'yn.planoverdrive.1': 'Generation > Shape',
  'yn.planoverdrive.2': 'Generation > Base Strength',
  'yn.planoverdrive.3': 'Generation > Pulse Period (steps)',
  'yn.htv3.0': 'Generation > HT Kinetic Energy',
  'yn.htv3.1': 'Generation > HT Damping Friction',
  'yn.htv3.2': 'Generation > HT Critical Sigma',
  'yn.htv3.3': 'Generation > HT Phase Intensity',
  'yn.htv3.4': 'Generation > HT Well Width',
  'yn.htv3.5': 'Generation > HT Density Floor',
  'yn.htv3.6': 'Generation > HT SNR Space',
  'yn.htv3.7': 'Generation > HT Poly Slope',
  'yn.htv3.8': 'Generation > HT Uniform Blend',
  'yn.htv3.9': 'Generation > HT Smoothing Window',
  'yn.htv3.10': 'Generation > HT CDF Resolution',
  'yn.htv3.11': 'Generation > HT Shift Warp',
  /* MiniMax-Music3's declared knobs, and its renderer row */
  'mm.temp': 'LM / Thinking > LM Temperature',
  'mm.topk': 'LM / Thinking > LM Top-K',
  'mm.topp': 'LM / Thinking > LM Top-P',
  'mm.rep': 'LM / Thinking > LM Repetition Penalty',
  'mm.repmode': 'LM / Thinking > Repetition Mode',
  'mm.win': 'LM / Thinking > Repetition Window',
  'mm.seed': 'LM / Thinking > Planner Seed',
  'mm.reuse': 'LM / Thinking > Reuse Planner Output',
  'mm.savear': 'LM / Thinking > Save AR Plan (Plank)',
  'mm.plank': 'LM / Thinking > Replay AR Plan',
  'mm.savehid': 'LM / Thinking > Save Plan To Disk',
  'mm.hidname': 'LM / Thinking > Plan Name',
  'mm.hidpath': 'LM / Thinking > Replay Saved Plan',
  'mm.hop': 'Generation > Window Hop',
  'mm.dcache': 'Generation > CFG Delta Cache',
  'mm.lowstep': 'Generation > Low-Step Compensation',
  'mm.fused': 'Generation > Fused Depth Decoder',
  'mm.stream': 'Generation > Play While Rendering',
  'mm.reqend': 'Generation > Require Natural Ending',
  'mm.minlen': 'Generation > Reject Short Renders',
  'mm.basehand': 'Generation > Base Model Ending',
  'mm.kvq8': 'Generation > Compact Planner Cache',
  /* Stable Audio 3's and HeartMuLa's declared knobs */
  'sa3.sampler': 'Generation > Sampler',
  'sa3.lufs': 'Generation > Loudness Normalize',
  'sa3.lufst': 'Generation > LUFS Target',
  'hm.temp': 'LM / Thinking > Temperature',
  'hm.topk': 'LM / Thinking > Top K',
  'hm.cfg': 'LM / Thinking > CFG scale',
  /* the Adapters panel, once adapters are loaded (the slots are restated below) */
  /* a stacked adapter's dial says Strength alone, Weight in Blend and Adapter Scale in Sum; Simple mode's is Adapter Scale */
  s0: { on: { ace: [
    { control: 'Adapters > Adapter Scale', when: { param: 'ad.adv', is: 'simple' } },
    { control: 'Adapters > Strength', when: { all: [{ param: 'ad.adv', is: 'advanced' }, { not: { state: 'adapters', atLeast: 2 } }] } },
    { control: 'Adapters > Weight', when: { all: [{ param: 'ad.adv', is: 'advanced' }, { state: 'adapters', atLeast: 2 }, { param: 'ad.stack', is: 'blend' }] } },
    { control: 'Adapters > Adapter Scale', when: { all: [{ param: 'ad.adv', is: 'advanced' }, { state: 'adapters', atLeast: 2 }, { param: 'ad.stack', is: 'sum' }] } },
  ], 'minimax-m3': 'Adapters > Strength', heartmula: 'Adapters > Strength' } },
  s1: [{ control: 'Adapters > Weight #2', when: { param: 'ad.stack', is: 'blend' } }, { control: 'Adapters > Adapter Scale #2', when: { param: 'ad.stack', is: 'sum' } }],
  s2: [{ control: 'Adapters > Weight #3', when: { param: 'ad.stack', is: 'blend' } }, { control: 'Adapters > Adapter Scale #3', when: { param: 'ad.stack', is: 'sum' } }],
  'ad.mode': { on: { ace: { press: { merge: 'Adapters > Merge', runtime: 'Adapters > Runtime ⚡', runtime_lowrank: 'Adapters > Low-Rank 🪶' } }, 'minimax-m3': { press: { runtime: 'Adapters > Runtime', merge: 'Adapters > Merge' } } } },
  'ad.dora': 'Adapters > Runtime DoRA magnitudes',
  'ad.plannerstr': 'Adapters > Planner Strength',
  'ad.adv': { press: { simple: 'Adapters > Simple', advanced: 'Adapters > Advanced' } },
  'ad.stack': { press: { blend: 'Adapters > Blend', sum: 'Adapters > Sum' } },
  'ad.sigma': 'Adapters > Combined Strength (Σ)',
  'ad.rq': { press: { bf16: 'Adapters > Full', q8_0: 'Adapters > Q8 ½', q4_0: 'Adapters > Q4 ¼' } },
  'ad.mq': { press: { hq: 'Adapters > HQ', low: 'Adapters > Low ¼' } },
  'ad.ref': { press: 'Adapters > No-adapter reference' },
  'ad.home': 'Adapters > Adapter trained on (home base)',
  'ad.align': 'Adapters > Alignment Timing',
  'ad.beta': { on: { ace: 'Adapters > Re-base Strength (β)', heartmula: 'Adapters > Re-base strength (β)' } },
  'ag.self': 'Adapters > Self-Attn',
  'ag.cross': 'Adapters > Cross-Attn',
  'ag.mlp': 'Adapters > MLP',
  'ag.cond': 'Adapters > Conditioning',
  'ag.time': 'Adapters > Timestep',
  'ag.proj': 'Adapters > Proj-In',
  'ad.att': 'Adapters > Attention',
  'ad.mlp': 'Adapters > MLP',
  'ad.plan': 'Adapters > Plan',
  'ad.early': { on: { 'minimax-m3': { control: 'Adapters > Early third', reveal: 'Adapters > Depth thirds (advanced)' }, heartmula: 'Adapters > Early' } },
  'ad.mid': { on: { 'minimax-m3': { control: 'Adapters > Middle third', reveal: 'Adapters > Depth thirds (advanced)' }, heartmula: 'Adapters > Mid' } },
  'ad.late': { on: { 'minimax-m3': { control: 'Adapters > Late third', reveal: 'Adapters > Depth thirds (advanced)' }, heartmula: 'Adapters > Late' } },
  'ad.gpu': 'Adapters > GPU merging',
  /* the bar's Post-Processing panel */
  'pp.on': `${PP} > ${PP}`,
  'pp.lrc.on': { control: `${PP} > Lyric Timestamps (LRC)` },
  'pp.fa.on': `${PP} > Lyric Timestamps (Forced Alignment)`,
  'pp.wh.on': `${PP} > Whisper Lyrics`,
  'pp.wh.model': `${PP} > Whisper Lyrics > Model`,
  'pp.wh.lang': `${PP} > Whisper Lyrics > Language`,
  'pp.wh.beam': `${PP} > Whisper Lyrics > Beam Size`,
  'pp.wh.iso': `${PP} > Whisper Lyrics > Isolate vocals first`,
  'pp.refiner.on': `${PP} > MM3 Latent Refiner`,
  'pp.refiner.steps': `${PP} > MM3 Latent Refiner > Steps`,
  'pp.refiner.win': `${PP} > MM3 Latent Refiner > Window seconds`,
  'pp.refiner.ov': `${PP} > MM3 Latent Refiner > Overlap seconds`,
  'pp.refiner.mix': { control: `${PP} > MM3 Latent Refiner > Refined mix`, times: 0.01 },
  'pp.ppvae.on': `${PP} > PP-VAE Re-encode`,
  'pp.ppvae.blend': `${PP} > PP-VAE Re-encode > Original Blend`,
  'pp.ss.on': SS,
  'pp.ss.str': { control: `${SS} > Refine strength`, times: 0.01 },
  'pp.ss.steps': `${SS} > Refine steps`,
  'pp.ss.vox': `${SS} > Vocal level`,
  'pp.ss.dyn': `${SS} > Preserve source dynamics`,
  'pp.ss.ppv': `${SS} > Re-encode vocals through PP-VAE`,
  'pp.ss.blend': { press: { off: `${SS} > Off`, crossover: `${SS} > Crossover`, mix: `${SS} > Mix` } },
  'pp.ss.xhz': `${SS} > Crossover frequency`,
  'pp.ss.xw': `${SS} > Crossover width`,
  'pp.ss.mix': `${SS} > SA3 amount`,
  'pp.ss.seedf': `${SS} > Follow generation seed`,
  'pp.ss.seed': { control: `${SS} > Fixed SA3 refine seed`, as: 'number' },
  'pp.lift.on': `${PP} > Spectral Lifter`,
  'pp.lift.den': `${PP} > Spectral Lifter > Denoise Strength`,
  'pp.lift.floor': `${PP} > Spectral Lifter > Noise Floor`,
  'pp.lift.hf': `${PP} > Spectral Lifter > HF Extension`,
  'pp.lift.tr': `${PP} > Spectral Lifter > Transient Boost`,
  'pp.lift.shim': `${PP} > Spectral Lifter > Shimmer Reduction`,
  'pp.nat.on': `${PP} > Vocal Naturalizer`,
  'pp.nat.amt': `${PP} > Vocal Naturalizer > Amount`,
  'pp.nat.vr': `${PP} > Vocal Naturalizer > Vibrato Rate`,
  'pp.nat.vd': `${PP} > Vocal Naturalizer > Vibrato Depth`,
  'pp.nat.fm': `${PP} > Vocal Naturalizer > Formant Humanize`,
  'pp.nat.mt': `${PP} > Vocal Naturalizer > Metallic Cut`,
  'pp.nat.qm': `${PP} > Vocal Naturalizer > Quantization Mask`,
  'pp.nat.ts': `${PP} > Vocal Naturalizer > Transition Smooth`,
  'pp.gain': `${PP} > Pre-VST Gain Offset > Offset`,
  'pp.master.on': `${PP} > Mastering`,
  'pp.master.timbre': `${PP} > Mastering > Also use as timbre reference`,
  'pp.norm.on': `${PP} > Final Normalizer`,
  'pp.norm.target': `${PP} > Final Normalizer > Target Preset`,
  'pp.norm.custom': { control: `${PP} > Final Normalizer > Custom Target`, when: { param: 'pp.norm.target', is: 'custom' } },
  'pp.norm.ceil': `${PP} > Final Normalizer > Peak Ceiling`,
  'pp.art.on': `${PP} > Cover Art`,
  'pp.art.subject': 'Create > Image Subject',
  'pp.q.on': `${PP} > Quality Evaluator`,
  'pp.q.eval': { press: { unmastered: `${PP} > Quality Evaluator > Unmastered`, mastered: `${PP} > Quality Evaluator > Mastered`, both: `${PP} > Quality Evaluator > Both` } },
  /* the tiled decoder, and its plugin picker (shown while it is on; its words are the plugin's) */
  'pp.tile.on': { on: { ace: `${PP} > Tiled Decoder` } },
  'pp.tile.plugin': { on: { ace: `${PP} > Tiled Decoder > *StyledSelect` } },
};

/* ---------------------------------------------------------------- restated: what a control in Node cannot set */

/** Store fields set directly, as the existing page sets them when the user chooses a file, a model or a seed in a
 *  dialog, or as its engine panel does. Each says where the existing page does it; the comparison counts these
 *  pairs as restated, not driven. */
export type Target = `store:${string}` | `selection:${string}`;
export const RESTATED: { [key: string]: { why: string; set(v: Value, e: EngineId, desk: { vals: { [key: string]: Value } }): [Target, Json][] } } = {
  /* the engine panel's model rows are engine state, chosen through POST /api/backends/models */
  'model.var': { why: 'the engine panel posts the model choice', set: v => (v === '' ? [] : [['selection:heartmula.variant', v as string]]) },
  'model.dit': { why: 'the engine panel posts the model choice', set: (v, e) => (e === 'heartmula' ? (v === '' ? [] : [['selection:heartmula.global', v as string]]) : e === 'ace' ? [['store:ditModel', v as string]] : []) },
  'model.lm': { why: 'the engine panel posts the model choice', set: (v, e) => (e === 'heartmula' ? (v === '' ? [] : [['selection:heartmula.local', v as string]]) : e === 'ace' ? [['store:lmModel', v as string]] : []) },
  'model.x': { why: 'the engine panel posts the model choice', set: v => (v === '' ? [] : [['selection:heartmula.flow', v as string]]) },
  'model.vae': { why: 'the engine panel posts the model choice', set: (v, e) => (e === 'ace' ? [['store:vaeModel', v as string]] : []) },
  /* MiniMax-Music3's renderer is a row of the engine panel's MiniMax section */
  'mm.rend': { why: 'Mm3RendererRow posts it with the model choice', set: v => [['store:backendParams.mm3DitBackend', v as string]] },
  /* the planner adapter is picked from the list of planner adapters the server has */
  'ad.planner': { why: 'picked from the server list of planner adapters', set: v => [['store:lmAdapter', v as string]] },
  /* the precision row is shown only on TensorRT */
  'mm.prec': { why: 'Mm3RendererRow, drawn from the server model catalogue', set: (v, _e, desk) => (desk.vals['mm.rend'] === 'tensorrt' ? [['store:backendParams.mm3DitPrecision', v as string]] : []) },
};

interface Slot { file: string; type?: string; ph?: [number, number] }
/** Module state, as the existing page holds it once a file is chosen in its dialog. Each returns the targets to set. */
export const STATE_HOLDINGS: { [key: string]: (v: Json, e: EngineId) => [Target, Json][] } = {
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

  timbreFile: v => (typeof v === 'string' && v ? [['store:timbreAudioPath', v]] : []),
  masterRef: v => (typeof v === 'string' && v ? [['store:masteringReference', v]] : []),
};

/** CreatePanel state that comes from somewhere a control in Node cannot reach: the imported latent (LatentImport
 *  uploads a file), YuE2's continuation (Yue2Continuation uploads a recording). Set on the intent's params as
 *  CreatePanel puts them there. */
export const CREATE_STATE: { [key: string]: (v: Json, e: EngineId, all: { [key: string]: Json | undefined }) => [field: string, value: Json][] } = {
  latent: (v, e) => (e === 'ace' && typeof v === 'string' && v ? [['sourceLatentUrl', v]] : []),
  contFile: (v, e, all) => {
    if (e !== 'yue2' || typeof v !== 'string' || !v) return [];
    const s = (all.contSel ?? {}) as { start?: number; end?: number };
    return [['yue2Continuation', { sourceAudioUrl: v, start: s.start ?? 0, end: s.end ?? 30 }]];
  },
};

/** The caption source CreatePanel puts in the intent while there are tracks to pick from (CreatePanel.tsx
 *  createIntent): the handoff's song caption and tracks on MiniMax-Music3, the dataset on YuE2. The tracks come from
 *  Lyric Studio or a dataset the page fetches, so this is restated. The source chooser's value is the mode, or a
 *  track's name. */
export const CAPTION_SOURCE = (e: EngineId, v: { source: string; dataset: string; caption: string; tracks: { title: string; bpm?: number; caption: string }[]; adapter: boolean }): Json | undefined => {
  if (!v.tracks.length) return undefined;
  const mode = v.source === 'custom' || v.source === 'auto' ? v.source : 'track';
  if (e === 'minimax-m3') return { engine: 'minimax-m3', customCaption: v.caption, selection: mode === 'track' ? { mode, selectedTitle: v.source } : { mode }, tracks: v.tracks };
  if (e === 'yue2') return { engine: 'yue2', datasetId: v.dataset, adapterInForce: v.adapter, selection: mode === 'track' ? { mode, selectedName: v.source } : { mode } };
  return undefined;
};
