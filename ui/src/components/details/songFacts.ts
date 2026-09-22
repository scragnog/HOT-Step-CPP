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

export type Backend = 'ace' | 'minimax-m3' | 'yue2';

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
};

export function songBackend(song: Song): Backend {
  const gp = params(song);
  const raw = String(song.backend || gp?.backend || 'ace');
  if (raw === 'minimax-m3' || raw === 'mm3' || raw === 'minimax') return 'minimax-m3';
  if (raw === 'yue2') return 'yue2';
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
  return value === undefined || value === null || value === '' ? '' : String(value);
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

  const seeds: Fact[] = [];
  add(seeds, 'Seed', num(gp.seed), { mono: true });
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
  const groups: FactGroup[] = [];

  const plan: Fact[] = [];
  if (gp.yue2Cot || req.cot) add(plan, 'Chain of thought', pretty(String(gp.yue2Cot || req.cot)));
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
      { tone: 'warn', title: y.auto_replan.attempts.map((a: any) => `${a.verdict}: ${a.reason}`).join('\n') });
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
  add(seeds, 'Seed', num(seed), { mono: true });
  // Only worth a line when it is actually a second seed.
  if (noise !== undefined && String(noise) !== String(seed)) {
    add(seeds, 'Noise seed', num(noise), { mono: true });
  }
  if (y.song !== undefined) add(seeds, 'Variation', `${Number(y.song) + 1}.${Number(y.variation) + 1}`);
  if (gp.yue2Variations > 1) add(seeds, 'Variations', num(gp.yue2Variations));
  if (seeds.length) groups.push({ title: 'Reproduction', facts: seeds });

  return groups;
}

/** Everything the panel shows under "How it was made", for this song's backend. */
export function buildFactGroups(song: Song): FactGroup[] {
  const gp = params(song);
  if (!gp) return [];
  switch (songBackend(song)) {
    case 'minimax-m3': return mm3Facts(gp);
    case 'yue2': return yue2Facts(gp);
    default: return aceFacts(gp);
  }
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
