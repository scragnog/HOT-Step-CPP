// songFacts.ts — what a song's stored parameters actually mean, per backend.
//
// THE TRAP: generation_params is a copy of the whole global params blob at
// submit time, not the subset the backend used. A YuE2 row carries ditModel,
// inferMethod, guidanceScale, mm3Steps — every one of them a leftover from a
// panel the render never touched. Reading those generically showed AS1.5
// numbers on MM3 and YuE2 songs.
//
// So: pick the backend first, then read only that backend's own keys.

import type { Song } from '../../types';
import { formatDitModel, formatLmModel } from '../global-bar/modelLabels';
import { seedText } from '../../utils/seed';

export type Backend = 'ace' | 'minimax-m3' | 'yue2' | 'sa3';

export interface Fact {
  label: string;
  value: string;
  /** Render in a monospace face — seeds, paths, frame counts. */
  mono?: boolean;
  /** Full text on hover, when the value is abbreviated. */
  title?: string;
  /** Draws attention: a warning or a notable outcome. */
  tone?: 'warn' | 'good';
}

export interface FactGroup {
  title: string;
  facts: Fact[];
}

export const BACKEND_LABELS: Record<Backend, string> = {
  'ace': 'ACE-Step 1.5',
  'minimax-m3': 'MiniMax-Music3',
  'yue2': 'YuE2',
  'sa3': 'Stable Audio 3',
};

export function songBackend(song: Song): Backend {
  const gp = params(song);
  const raw = String(song.backend || gp?.backend || 'ace');
  if (raw === 'minimax-m3' || raw === 'mm3' || raw === 'minimax') return 'minimax-m3';
  if (raw === 'yue2') return 'yue2';
  if (raw === 'sa3' || raw === 'stable-audio-3') return 'sa3';
  return 'ace';
}

function params(song: Song): any {
  return (song.generationParams || song.generation_params) as any;
}

// ── formatting ───────────────────────────────────────────────────────────────

/** Seconds to m:ss. Strings pass through — some rows store them pre-formatted. */
export function formatDuration(value: unknown): string {
  if (value === undefined || value === null || value === '') return '';
  if (typeof value === 'string' && value.includes(':')) return value;
  const secs = Number(value);
  if (!Number.isFinite(secs) || secs <= 0) return '';
  const m = Math.floor(secs / 60);
  const s = Math.round(secs % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

/** snake_case and kebab-case engine names to something readable. */
function pretty(value: string): string {
  return value
    .split(':')[0]
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

/** Last path segment of an adapter or model path. */
function basename(p: string): string {
  return (p.split(/[\\/]/).filter(Boolean).pop() || p).replace(/^acestep-/, '');
}

/** The adapter's own name, not its timestamped run folder. */
function adapterName(p: string): string {
  const parts = p.split(/[\\/]/).filter(Boolean);
  const last = parts[parts.length - 1] || p;
  // Runs are stored as <adapter>/<YYYY-MM-DD_run>, so the parent is the name.
  return /^\d{4}-\d{2}-\d{2}/.test(last) && parts.length > 1
    ? `${parts[parts.length - 2]} · ${last}`
    : last;
}

/** The engine's per-stage terminators, in words. "skipped" in particular does
 *  not mean something went wrong: the plan stage reports it when chain-of-
 *  thought is off, or when the app planned the ABC score itself and handed the
 *  engine a finished one (engine/src/yue2/yue2-pipeline.h). */
const STAGE_REASON: Record<string, { text: string; tone?: Fact['tone'] }> = {
  eos: { text: 'Ended naturally', tone: 'good' },
  limit_hit: { text: 'Hit the length cap', tone: 'warn' },
  preview_limit: { text: 'Stopped at the preview limit', tone: 'warn' },
  skipped: { text: 'Not run' },
};

function num(value: unknown): string {
  if (value === undefined || value === null || value === '') return '';
  // Seeds past 2^53: print the exact integer, not "...570000".
  return typeof value === 'number' ? seedText(value) : String(value);
}

/** Push a fact only when it has a value — keeps every builder below flat. */
function add(facts: Fact[], label: string, value: string, extra: Partial<Fact> = {}): void {
  if (value === '' || value === undefined || value === null) return;
  facts.push({ label, value, ...extra });
}

// ── per-backend builders ─────────────────────────────────────────────────────

function aceFacts(gp: any): FactGroup[] {
  const groups: FactGroup[] = [];

  const models: Fact[] = [];
  if (gp.ditModel) add(models, 'DiT', formatDitModel(gp.ditModel), { title: gp.ditModel });
  if (gp.lmModel) add(models, 'Planner LM', formatLmModel(gp.lmModel), { title: gp.lmModel });
  if (gp.vaeModel) add(models, 'VAE', basename(gp.vaeModel), { title: gp.vaeModel });
  if (models.length) groups.push({ title: 'Models', facts: models });

  const sampling: Fact[] = [];
  add(sampling, 'Steps', num(gp.inferenceSteps));
  add(sampling, 'CFG scale', num(gp.guidanceScale));
  if (gp.inferMethod) add(sampling, 'Solver', pretty(gp.inferMethod));
  if (gp.scheduler) add(sampling, 'Schedule', pretty(gp.scheduler));
  if (gp.guidanceMode) add(sampling, 'Guidance', pretty(gp.guidanceMode));
  if (gp.shift !== undefined) add(sampling, 'Shift', gp.shift < 0 ? 'Auto' : String(gp.shift));
  if (sampling.length) groups.push({ title: 'Sampling', facts: sampling });

  const lm: Fact[] = [];
  if (gp.skipLm) add(lm, 'Planner', 'Skipped');
  add(lm, 'LM CFG', num(gp.lmCfgScale));
  add(lm, 'Temperature', num(gp.lmTemperature));
  add(lm, 'Top P', num(gp.lmTopP));
  if (gp.lmTopK) add(lm, 'Top K', num(gp.lmTopK));
  if (gp.useCotCaption !== undefined) add(lm, 'Thinking', gp.useCotCaption ? 'On' : 'Off');
  if (lm.length) groups.push({ title: 'Planner', facts: lm });

  const adapters = adapterFacts(gp);
  if (adapters.length) groups.push({ title: 'Adapters', facts: adapters });

  const seeds: Fact[] = [];
  add(seeds, 'Seed', num(gp.seed), { mono: true });
  add(seeds, 'LM seed', num(gp.lmSeed), { mono: true });
  if (gp.batchSize > 1) add(seeds, 'Batch', num(gp.batchSize));
  if (seeds.length) groups.push({ title: 'Reproduction', facts: seeds });

  return groups;
}

/** DiT and planner adapters — ACE-Step only; the other backends key them
 *  separately and a shared reader is how stale names leaked in. */
function adapterFacts(gp: any): Fact[] {
  const facts: Fact[] = [];
  const stack = Array.isArray(gp.loraStack) ? gp.loraStack : [];

  if (stack.length > 1) {
    add(facts, 'DiT stack', `${stack.length} adapters`,
      { title: stack.map((a: any) => a?.path || '').join('\n') });
    if (gp.adapterStackMode) add(facts, 'Stack mode', pretty(gp.adapterStackMode));
    if (gp.adapterStackBudget !== undefined) add(facts, 'Stack budget', num(gp.adapterStackBudget));
  } else {
    const dit = gp.loraPath || gp.adapter || stack[0]?.path || '';
    if (dit) {
      add(facts, 'DiT adapter', adapterName(dit), { title: dit });
      if (gp.loraScale !== undefined && gp.loraScale !== 1) add(facts, 'Scale', num(gp.loraScale));
    }
  }

  if (gp.lmAdapter) {
    add(facts, 'LM adapter', adapterName(gp.lmAdapter), { title: gp.lmAdapter });
    if (gp.lmAdapterScale !== undefined && gp.lmAdapterScale !== 1) {
      add(facts, 'LM scale', num(gp.lmAdapterScale));
    }
  }
  if (facts.length && gp.adapterMode) add(facts, 'Apply', pretty(gp.adapterMode));
  return facts;
}

function mm3Facts(gp: any): FactGroup[] {
  const mm3 = gp.mm3 || {};
  const req = gp.mm3Request || {};
  const groups: FactGroup[] = [];

  const render: Fact[] = [];
  add(render, 'Steps', num(mm3.steps ?? gp.mm3Steps));
  add(render, 'Flow CFG', num(mm3.cfg_flow));
  if (mm3.max_frames) add(render, 'Max frames', num(mm3.max_frames), { mono: true });
  if (mm3.prompt_tokens) add(render, 'Prompt tokens', num(mm3.prompt_tokens), { mono: true });
  if (mm3.sample_rate) add(render, 'Sample rate', `${Math.round(Number(mm3.sample_rate) / 1000)} kHz`);
  if (mm3.instrumental !== undefined) add(render, 'Vocals', mm3.instrumental ? 'Instrumental' : 'Sung');
  if (gp.duration === -1) add(render, 'Length', 'Model decides');
  if (render.length) groups.push({ title: 'Render', facts: render });

  const adapter: Fact[] = [];
  if (gp.mm3LmAdapter) {
    add(adapter, 'LM adapter', adapterName(String(gp.mm3LmAdapter)), { title: String(gp.mm3LmAdapter) });
    if (gp.mm3LmAdapterMode) add(adapter, 'Apply', pretty(String(gp.mm3LmAdapterMode)));
    if (gp.mm3LmAdapterScale !== undefined) add(adapter, 'Scale', num(gp.mm3LmAdapterScale));
    if (gp.mm3LmAdapterScaleAttn !== undefined && gp.mm3LmAdapterScaleAttn !== 1) {
      add(adapter, 'Attention', num(gp.mm3LmAdapterScaleAttn));
    }
    if (gp.mm3LmAdapterScaleMlp !== undefined && gp.mm3LmAdapterScaleMlp !== 1) {
      add(adapter, 'MLP', num(gp.mm3LmAdapterScaleMlp));
    }
  }
  if (adapter.length) groups.push({ title: 'Adapter', facts: adapter });

  // Cover Studio (code forcing): which plank, and how hard it was held.
  const cover: Fact[] = [];
  const plank = gp.mm3PlankPath || gp.mm3CoverPlank;
  if (plank) {
    // Cover modes (Cover Studio): Pin forces the plank; the others follow its guide.
    const MODES: Record<string, string> = { follow: 'Follow source', 'follow-depth': 'Follow source, depth fix',
      'follow-stream': 'Follow source, stream fix', product: 'Product rule', blend: 'State blend' };
    add(cover, 'Mode', MODES[String(gp.mm3CoverMode)] ?? 'Pin');
    add(cover, 'Plank', String(plank).replace(/\.mm3plank$/, '').slice(0, 8), { mono: true, title: String(plank) });
    if (gp.mm3PlankLaundered !== undefined) {
      add(cover, 'Laundered', gp.mm3PlankLaundered ? 'Yes' : 'No',
        gp.mm3PlankLaunderModels ? { title: String(gp.mm3PlankLaunderModels) } : undefined);
    }
    if (gp.mm3PlankLaunderSteps !== undefined) add(cover, 'Launder steps', num(gp.mm3PlankLaunderSteps));
    if (gp.mm3PlankLaunderSeed !== undefined) add(cover, 'Launder seed', String(gp.mm3PlankLaunderSeed));
    if (gp.mm3PlankLaunderCaption !== undefined) add(cover, 'Source caption', String(gp.mm3PlankLaunderCaption));
    if (gp.mm3CoverMode) {
      if (gp.mm3GuideProduct !== undefined) add(cover, 'Weight', num(gp.mm3GuideProduct));
      else add(cover, 'Strength', num(gp.mm3GuideScale));
      add(cover, 'Candidates', num(gp.mm3GuideK));
      const FIX = ['Off', 'Depth', 'Stream 0', 'Both'];
      if (gp.mm3GuideStateFix !== undefined) add(cover, 'State fix', FIX[Number(gp.mm3GuideStateFix)] ?? String(gp.mm3GuideStateFix));
      if (gp.mm3BlendAmount !== undefined) add(cover, 'Blend', `${Math.round(Number(gp.mm3BlendAmount) * 100)}%`);
      if (Number(gp.mm3GuideSeconds) > 0) add(cover, 'Trim', `${num(gp.mm3GuideSeconds)} s`);
    } else add(cover, 'Forcing', gp.mm3PlankSemanticOnly ? 'Semantic only' : 'Full replay');
    if (gp.mm3PlankSemanticOnly && !gp.mm3CoverMode) {
      add(cover, 'Bias', gp.mm3ForceBias === '' || gp.mm3ForceBias === undefined ? 'Hard pin' : num(gp.mm3ForceBias));
      const p = gp.mm3ForcePattern;
      if (p) {
        const rules = [p.prefixSeconds && `prefix ${p.prefixSeconds} s`, p.anchorEvery && `every ${p.anchorEvery} frames`,
          p.pinnedRanges?.length && p.pinnedRanges.map((r: number[]) => `${r[0]}-${r[1]}`).join(', ')].filter(Boolean);
        add(cover, 'Pattern', `${rules.join(' + ')}${p.invert ? ' (inverted)' : ''}`);
      }
    }
    if (gp.mm3PlankTrimSeconds && !gp.mm3CoverMode) add(cover, 'Trim', `${num(gp.mm3PlankTrimSeconds)} s`);
  }
  // Sound: the source's latents blended into the flow (audio to audio).
  if (Number(gp.mm3A2aStrength) > 0) {
    add(cover, 'Sound', num(gp.mm3A2aStrength));
    if (Number(gp.mm3A2aHold) > 0) add(cover, 'Hold', num(gp.mm3A2aHold));
    if (Number(gp.mm3A2aSourceSeconds) > 0) add(cover, 'Source trim', `${num(gp.mm3A2aSourceSeconds)} s`);
  }
  if (cover.length) groups.push({ title: 'Cover', facts: cover });

  const seeds: Fact[] = [];
  add(seeds, 'Seed', num(gp.seed), { mono: true });
  if (gp.mm3ArSeed !== undefined && gp.mm3ArSeed !== '') add(seeds, 'Planner seed', num(gp.mm3ArSeed), { mono: true });
  if (gp.mm3Takes > 1) {
    add(seeds, 'Take', `${Number(gp.mm3Take) + 1} of ${gp.mm3Takes}`);
    add(seeds, 'Base seed', num(gp.mm3BaseSeed), { mono: true });
  }
  if (req.model) add(seeds, 'Model', String(req.model), { title: String(req.model) });
  if (seeds.length) groups.push({ title: 'Reproduction', facts: seeds });

  return groups;
}

function yue2Facts(gp: any): FactGroup[] {
  const y = gp.yue2 || {};
  const req = gp.yue2Request || {};
  const cot = req.cot ?? gp.yue2Cot;
  const groups: FactGroup[] = [];

  const plan: Fact[] = [];
  if (cot) add(plan, 'Chain of thought', pretty(String(cot)));
  if (y.abc_supplied !== undefined) add(plan, 'Score', y.abc_supplied ? 'Supplied (ABC)' : 'Planned by LM');
  const health = y.score_health;
  if (health?.verdict) {
    add(plan, 'Score health', pretty(String(health.verdict)),
      { tone: health.verdict === 'healthy' ? 'good' : 'warn', title: health.reason });
  }
  if (health?.bars) {
    add(plan, 'Bars', `${health.bars}${health.vocalShare !== undefined
      ? ` · ${Math.round(Number(health.vocalShare) * 100)}% vocal` : ''}`);
  }
  if (health?.tempo) add(plan, 'Planned tempo', `${health.tempo} bpm`);
  if (health?.meter) add(plan, 'Meter', String(health.meter));
  const replans = Array.isArray(y.auto_replan?.attempts) ? y.auto_replan.attempts.length : 0;
  if (replans) {
    add(plan, 'Auto-replan', `${replans} attempt${replans === 1 ? '' : 's'}`,
      { tone: 'warn', title: y.auto_replan.attempts.map((a: any) => `${a.verdict}: ${a.reason}${Array.isArray(a.flags) && a.flags.length ? ` (${a.flags.join('; ')})` : ''}`).join('\n') });
  }
  // Legibility flags on the plan that was rendered (2026-09-27), and whether
  // the re-plan loop ran out without a clean one.
  const planFlags: string[] = Array.isArray(health?.legibility?.flags) ? health.legibility.flags : [];
  if (planFlags.length) add(plan, 'Plan flags', planFlags.join('; '), { tone: 'warn' });
  if (y.auto_replan && y.auto_replan.clean === false) {
    add(plan, 'Plan quality', `No clean plan in ${replans} tries; rendered the least-flagged`, { tone: 'warn' });
  }
  if (plan.length) groups.push({ title: 'Plan', facts: plan });

  const render: Fact[] = [];
  add(render, 'ODE steps', num(y.ode_steps ?? gp.yue2OdeSteps ?? req.ode_steps));
  if (gp.yue2NarSolver || req.ode_method) add(render, 'Solver', pretty(String(gp.yue2NarSolver || req.ode_method)));
  if (gp.yue2NarScheduler || req.scheduler) add(render, 'Schedule', pretty(String(gp.yue2NarScheduler || req.scheduler)));
  add(render, 'CFG scale', num(y.cfg_scale ?? gp.yue2CfgScale));
  if (gp.yue2NarCacheRatio) add(render, 'NAR cache', num(gp.yue2NarCacheRatio));
  if (y.vae_variant || gp.yue2VaeVariant) add(render, 'VAE', pretty(String(y.vae_variant || gp.yue2VaeVariant)));
  if (y.instrumental !== undefined) add(render, 'Vocals', y.instrumental ? 'Instrumental' : 'Sung');
  if (render.length) groups.push({ title: 'Render', facts: render });

  const outcome: Fact[] = [];
  if (y.end_reason) {
    add(outcome, 'Ended', pretty(String(y.end_reason)),
      { tone: y.end_reason === 'completed' ? 'good' : 'warn' });
  }
  for (const [stage, reason] of Object.entries(y.stage_end_reasons || {})) {
    // "plan: skipped" alongside "Score: Supplied (ABC)" is the same fact twice,
    // and the second phrasing reads like a failure. Drop it.
    if (stage === 'plan' && reason === 'skipped' && y.abc_supplied) continue;
    // An app-planned score (auto re-plan, instrumental move) also reaches the
    // engine finished, so the engine says "skipped" though a plan did run.
    if (stage === 'plan' && reason === 'skipped' && y.abc_planned) {
      add(outcome, 'Plan stage', 'Planned by the app');
      continue;
    }
    // The orchestrator can plan first, then render with that ABC. The render's
    // skipped result describes only its own job; abc_supplied records whether
    // the score came from the caller before that separate planning pass.
    if (stage === 'plan' && reason === 'skipped' && y.abc_supplied === false
      && req.abc && cot !== 'off') {
      add(outcome, 'Plan stage', 'Planned by LM');
      continue;
    }
    const mapped = STAGE_REASON[String(reason)];
    add(outcome, `${pretty(stage)} stage`, mapped?.text ?? pretty(String(reason)),
      mapped?.tone ? { tone: mapped.tone } : {});
  }
  add(outcome, 'Measured length', formatDuration(y.duration_s));
  if (outcome.length) groups.push({ title: 'Outcome', facts: outcome });

  // Stamped by the backend at render time. The AR/NAR pick is engine state
  // rather than a request field, so nothing else in gp records it — and the
  // gp.lmAdapter sitting alongside belongs to ACE-Step's 4B planner.
  const adapters: Fact[] = [];
  for (const slot of ['nar', 'ar'] as const) {
    const half = gp.yue2Adapters?.[slot];
    if (!half?.path) continue;
    const label = slot.toUpperCase();
    add(adapters, `${label} adapter`, adapterName(String(half.path)), { title: String(half.path) });
    if (half.trigger) add(adapters, `${label} trigger`, String(half.trigger), { mono: true });
    const sc = half.scales || {};
    for (const [key, name] of [['global', 'Scale'], ['attn', 'Attention'], ['mlp', 'MLP'],
                               ['early', 'Early'], ['mid', 'Mid'], ['late', 'Late']] as const) {
      if (sc[key] !== undefined && sc[key] !== 1) add(adapters, `${label} ${name.toLowerCase()}`, num(sc[key]));
    }
  }
  if (adapters.length) groups.push({ title: 'Adapters', facts: adapters });

  const seeds: Fact[] = [];
  const seed = gp.seed ?? req.seed;
  const noise = y.noise_seed ?? req.noise_seed;
  // baseSeed is what the user typed; seed is what rendered (base + variation
  // index, or a re-plan seed). Rows before baseSeed existed show only one.
  const base = gp.baseSeed;
  add(seeds, 'Seed', num(base ?? seed), { mono: true });
  if (base !== undefined && seed !== undefined && num(base) !== num(seed)) {
    add(seeds, 'Rendered seed', num(seed), { mono: true });
  }
  // Only worth a line when it is actually a second seed.
  if (noise !== undefined && num(noise) !== num(seed)) {
    add(seeds, 'Noise seed', num(noise), { mono: true });
  }
  if (y.song !== undefined) add(seeds, 'Variation', `${Number(y.song) + 1}.${Number(y.variation) + 1}`);
  if (gp.yue2Variations > 1) add(seeds, 'Variations', num(gp.yue2Variations));
  if (seeds.length) groups.push({ title: 'Reproduction', facts: seeds });

  return groups;
}

/** Stable Audio 3. gp is the whole global blob, so ditModel, lmModel,
 *  inferenceSteps and friends are ACE leftovers. Only sa3Request (the wire as
 *  sent) and sa3 (render stamp) are truth. The solver/scheduler/guidance
 *  pickers only reach the wire when plugins were enabled; when absent the
 *  render ran the native sampler, and on rows that predate sa3Request we say
 *  "Unknown" rather than guess from the ACE keys. */
function sa3Facts(gp: any): FactGroup[] {
  const req = gp.sa3Request;
  const s = gp.sa3 || {};
  const groups: FactGroup[] = [];

  const models: Fact[] = [];
  add(models, 'Model', 'Stable Audio 3');
  const adapters: any[] = Array.isArray(req?.adapters) ? req.adapters : [];
  for (const a of adapters) {
    if (!a?.name) continue;
    add(models, 'Adapter', `${basename(String(a.name))}${a.scale !== undefined && a.scale !== 1 ? ` @ ${a.scale}` : ''}`,
      { title: String(a.name) });
  }
  groups.push({ title: 'Models', facts: models });

  const sampling: Fact[] = [];
  if (!req) {
    add(sampling, 'Parameters', 'Unknown (older render)');
  } else {
    add(sampling, 'Steps', num(req.steps ?? s.steps));
    const native = !req.solver && !req.scheduler && !req.guidanceMode;
    add(sampling, 'Sampler', native ? pretty(String(req.sampler ?? s.sampler ?? 'pingpong')) : 'Plugins');
    if (req.solver) add(sampling, 'Solver', pretty(String(req.solver)));
    if (req.scheduler) add(sampling, 'Schedule', pretty(String(req.scheduler)));
    if (req.guidanceMode) add(sampling, 'Guidance', pretty(String(req.guidanceMode)));
    if (req.guidanceScale !== undefined) add(sampling, 'CFG scale', num(req.guidanceScale));
    if (req.postprocessPlugin) add(sampling, 'Decode', pretty(String(req.postprocessPlugin)));
    if (req.nTokens ?? s.prompt_tokens) add(sampling, 'Prompt tokens', num(req.nTokens ?? s.prompt_tokens), { mono: true });
  }
  if (s.sample_rate) add(sampling, 'Sample rate', `${(Number(s.sample_rate) / 1000).toFixed(1)} kHz`);
  if (sampling.length) groups.push({ title: 'Sampling', facts: sampling });

  const seeds: Fact[] = [];
  add(seeds, 'Seed', num(req?.seed ?? gp.seed), { mono: true });
  if (seeds.length) groups.push({ title: 'Reproduction', facts: seeds });
  return groups;
}

/** Everything the panel shows under "How it was made", for this song's backend. */
export function buildFactGroups(song: Song): FactGroup[] {
  const gp = params(song);
  if (!gp) return [];
  const backend = songBackend(song);
  const cover = coverFacts(gp, backend);
  let groups: FactGroup[];
  switch (backend) {
    case 'minimax-m3': groups = mm3Facts(gp); break;
    case 'yue2': groups = yue2Facts(gp); break;
    case 'sa3': groups = sa3Facts(gp); break;
    default: groups = aceFacts(gp);
  }
  return cover.length ? [{ title: 'Cover Studio', facts: cover }, ...groups] : groups;
}

function coverFacts(gp: any, backend: Backend): Fact[] {
  const cover = backend === 'yue2' ? gp.yue2Cover : undefined;
  if (!cover && gp.source !== 'cover-studio') return [];
  const facts: Fact[] = [];
  add(facts, 'Workflow', 'Cover Studio');
  if (gp.artistName) add(facts, 'Covering band', String(gp.artistName));
  if (gp.sourceArtist) add(facts, 'Source artist', String(gp.sourceArtist));
  if (cover) {
    add(facts, 'Source type', cover.sourceKind === 'musicxml' ? 'MusicXML' : 'Audio');
    if (cover.sourceLabel) add(facts, 'Source', String(cover.sourceLabel));
    if (cover.sourceId) add(facts, 'Source ID', String(cover.sourceId), { mono: true });
    if (cover.voices) add(facts, 'Score voices', cover.voices === 'both' ? 'Vocal and accompaniment' : 'Vocal');
    if (cover.keepChords !== undefined) add(facts, 'Keep chords', cover.keepChords ? 'On' : 'Off');
    if (cover.tempo !== undefined) add(facts, 'Tempo', typeof cover.tempo === 'number' ? `${cover.tempo} bpm` : pretty(String(cover.tempo)));
    if (cover.key) add(facts, 'Key', String(cover.key));
    if (cover.keyShift !== undefined) add(facts, 'Key shift', `${cover.keyShift} semitones`);
    add(facts, 'Cover CFG', num(cover.cfgScale));
  } else {
    if (gp.sourceAudioUrl) add(facts, 'Source ID', String(gp.sourceAudioUrl), { mono: true });
    add(facts, 'Structure fidelity', num(gp.audioCoverStrength));
    add(facts, 'Source preservation', num(gp.coverNoiseStrength));
  }
  return facts;
}

/** The handful of musical facts shown as chips above the parameters. Only the
 *  backends that actually measure them store them — MM3 and YuE2 write 0 and
 *  '' into bpm/key/time_signature, so those are dropped rather than shown as
 *  "0" and blank. */
export function buildTrackChips(song: Song): Fact[] {
  const gp = params(song) || {};
  const chips: Fact[] = [];
  add(chips, 'Length', formatDuration(song.duration));

  // The song's own columns are the only per-render truth. gp.bpm/keyScale are
  // the global panel's values, which ACE-Step consumes as inputs and the other
  // backends ignore entirely — reading them showed a YuE2 track as "138 bpm,
  // C major" while its own plan said 70 bpm in 2/4.
  const ace = songBackend(song) === 'ace';
  const bpm = Number(song.bpm || (ace ? gp.bpm : 0) || 0);
  if (bpm > 0) add(chips, 'Tempo', `${bpm} bpm`);
  const key = String(song.key_scale || (ace ? gp.keyScale : '') || '').trim();
  if (key) add(chips, 'Key', key);
  const sig = String(song.time_signature || (ace ? gp.timeSignature : '') || '').trim();
  if (sig) add(chips, 'Time', sig);
  return chips;
}
