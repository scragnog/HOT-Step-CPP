// DriftPanel.tsx -- Directed Drift controls for StormPage (Slice 3)
// MDMAchine / A&E Concepts 2026

import { useState, useEffect, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useQuery } from '@tanstack/react-query';
import { usePersistedState } from '@/hooks/usePersistedState';
import { stormDriftOptions } from '@/queries/shell-queue/queue';
import { apiClient } from '@/services/httpClient';
import type { useStreamAudio } from '@/hooks/useStreamAudio';
import { Toggle } from '@/components/shared/controls/Toggle';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { HintButton } from '@/components/shared/controls/HintButton';
import { QueryError } from '@/components/shared/feedback/QueryError';
import { clamp01 } from '@/utils/format/clamp';
import { formatDecimal } from '@/utils/format/format';
import { isPresent } from '@/utils/guards';

// ---------------------------------------------------------------------------
// Types mirrored from server/src/services/storm/drift.ts
// ---------------------------------------------------------------------------

type SeedMode = 'lock' | 'walk' | 'jump';

interface DriftPressure {
  targets: Record<string, number>;
  targetCaption: string[];
  captionPool: string[];
  pull: number;
  rate: number;
  jitter: number;
  locks: Record<string, boolean>;
  seedMode: SeedMode;
  pluginDrift?: boolean;
  customTagSets?: Record<string, string[]>;
}

interface ParamSpec {
  key: string;
  min: number;
  max: number;
  step: number;
  int?: boolean;
}

// ---------------------------------------------------------------------------
// Param spec tables (mirrors server/src/services/storm/drift.ts)
// ---------------------------------------------------------------------------

const ACE_PARAM_SPECS: ParamSpec[] = [
  { key: 'guidanceScale',  min: 1,  max: 15,  step: 0.1  },
  { key: 'lssStrength',    min: 0,  max: 1,   step: 0.05 },
  { key: 'inferenceSteps', min: 10, max: 100, step: 1, int: true },
  { key: 'bpm',            min: 60, max: 220, step: 1, int: true },
  { key: 'duration',       min: 15, max: 120, step: 1, int: true },
];

const YUE2_PARAM_SPECS: ParamSpec[] = [
  { key: 'yue2CfgScale',      min: 1,  max: 25,  step: 0.1  },
  { key: 'yue2OdeSteps',      min: 8,  max: 64,  step: 1, int: true },
  { key: 'yue2NarCacheRatio', min: 0,  max: 0.9, step: 0.05 },
];

// Mirrors server/src/services/storm/drift.ts SA3_PARAM_SPECS
const SA3_PARAM_SPECS: ParamSpec[] = [
  { key: 'guidanceScale', min: 1,  max: 15,  step: 0.1       },
  { key: 'sa3Steps',      min: 4,  max: 64,  step: 1, int: true },
  { key: 'duration',      min: 5,  max: 600, step: 1, int: true },
];

// Display labels for param keys
const PARAM_LABELS: Record<string, string> = {
  guidanceScale:      'Guidance',
  lssStrength:        'LSS Strength',
  inferenceSteps:     'Steps',
  bpm:                'BPM',
  duration:           'Duration',
  yue2CfgScale:       'CFG Scale',
  yue2OdeSteps:       'ODE Steps',
  yue2NarCacheRatio:  'NAR Cache',
  sa3Steps:           'SA3 Steps',
};

const PARAM_INFO: Record<string, string> = {
  guidanceScale: 'Target prompt guidance for future slots. Higher values follow the caption more strongly.',
  lssStrength: 'Target latent self-similarity strength for future slots.',
  inferenceSteps: 'Target diffusion step count. More steps take longer to render.',
  bpm: 'Target tempo in beats per minute for future slots.',
  duration: 'Target slot length in seconds. Longer slots take longer to render.',
  yue2CfgScale: 'Target YuE2 guidance scale for future slots. Higher values follow the caption more strongly.',
  yue2OdeSteps: 'Target YuE2 NAR solver step count. More steps take longer to render.',
  yue2NarCacheRatio: 'Target YuE2 NAR cache ratio. Higher values reuse more cached computation.',
  sa3Steps: 'Target Stable Audio 3 diffusion step count. More steps take longer to render.',
};

// ---------------------------------------------------------------------------
// Mood presets (mirrors server/src/services/storm/drift.ts)
// ---------------------------------------------------------------------------

const MOOD_PRESETS: Record<string, string[]> = {
  dark:    ['dark', 'melancholic', 'atmospheric', 'minor key', 'slow'],
  bright:  ['bright', 'uplifting', 'major key', 'energetic', 'upbeat'],
  chill:   ['chill', 'relaxed', 'ambient', 'smooth', 'lo-fi'],
  intense: ['intense', 'powerful', 'driving', 'aggressive', 'fast'],
  dreamy:  ['dreamy', 'ethereal', 'floating', 'reverb', 'lush'],
};

// ---------------------------------------------------------------------------
// Default pressure
// ---------------------------------------------------------------------------

function defaultPressure(): DriftPressure {
  return {
    targets:       {},
    targetCaption: [],
    captionPool:   [],
    pull:          0.5,
    rate:          0.3,
    jitter:        0.1,
    locks:         {},
    seedMode:      'walk',
    pluginDrift:   false,
    customTagSets: {},
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function denormalize(spec: ParamSpec, u: number): number {
  const raw = spec.min + u * (spec.max - spec.min);
  if (spec.int) return Math.round(raw);
  return Math.round(raw / spec.step) * spec.step;
}

function fmtVal(spec: ParamSpec, u: number): string {
  const v = denormalize(spec, clamp01(u));
  return spec.int ? String(v) : v.toFixed(2);
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SliderRow({
  label,
  info,
  value,
  min = 0,
  max = 1,
  step = 0.01,
  onChange,
  fmt,
}: {
  label: string;
  info: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
  fmt?: (v: number) => string;
}) {
  return (
    <div className="flex items-center gap-2 min-w-0">
      <ParamLabel label={label} info={info} className="text-[11px] text-zinc-400" rootClassName="!w-20 shrink-0" />
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="flex-1 h-1 accent-purple-500"
      />
      <span className="text-[11px] text-zinc-300 w-10 text-right tabular-nums shrink-0">
        {fmt ? fmt(value) : formatDecimal(value, 2)}
      </span>
    </div>
  );
}

function TagWeightBar({ tag, weight }: { tag: string; weight: number }) {
  const pct = Math.round(clamp01(weight) * 100);
  const active = weight > 0.5;
  return (
    <div className="flex items-center gap-1.5 text-[10px]">
      <span className={`truncate w-24 ${active ? 'text-purple-300' : 'text-zinc-600'}`}>{tag}</span>
      <div className="flex-1 h-1 rounded bg-zinc-800">
        <div
          className={`h-full rounded ${active ? 'bg-purple-500' : 'bg-zinc-600'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-zinc-500 w-7 text-right tabular-nums">{pct}%</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// DriftPanel
// ---------------------------------------------------------------------------

export interface DriftPanelProps {
  sa: ReturnType<typeof useStreamAudio>;
  isYue2: boolean;
  isSa3?: boolean;
  streamId?: string;
}

export function DriftPanel({ sa, isYue2, isSa3, streamId = 'default' }: DriftPanelProps) {
  const { t } = useTranslation();
  const specs = isYue2 ? YUE2_PARAM_SPECS : isSa3 ? SA3_PARAM_SPECS : ACE_PARAM_SPECS;

  // Persisted pressure
  const [pressure, setPressure] = usePersistedState<DriftPressure>(
    'hs-storm-drift-pressure',
    defaultPressure(),
  );

  // Live snapshot from GET /storm/control, polled only while streaming.
  const driftQuery = useQuery({
    ...stormDriftOptions(streamId),
    enabled: sa.isPlaying,
    refetchInterval: 2000,
    refetchIntervalInBackground: true,
  });
  const snapshot = sa.isPlaying ? driftQuery.data?.drift ?? null : null;

  // Custom tag set editing
  const [newSetName, setNewSetName] = useState('');
  const [newSetTags, setNewSetTags] = useState('');

  // Target caption textarea
  const [captionInput, setCaptionInput] = useState(pressure.targetCaption.join(', '));

  // Debounce POST
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const postPressure = useCallback((p: DriftPressure) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      // Silent: a dropped control tick is replaced by the next one; a toast per
      // failure while dragging a slider would be noise.
      void apiClient.post('/generate/storm/control', { streamId, drift: p }, { silent: true }).catch(() => {});
    }, 400);
  }, [streamId]);

  // Ref for one-slot pull restore: stores prevPull + slot counter at press time.
  // Cleared after the restore fires.
  const pullRestoreRef = useRef<{ prevPull: number; slotAtPress: number } | null>(null);

  // Pull-now: set pull=1 locally and on server for one slot, then restore.
  // One-slot restore: watches the snapshot slot counter -- when it advances
  // past slotAtPress the pull is restored. Chosen over the simple local-only
  // approach because the snapshot exposes a slot counter.
  const pullNow = useCallback(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    const prevPull = pressure.pull;
    const slotAtPress = snapshot?.slot ?? -1;
    // Update local state so UI and server agree immediately.
    setPressure(prev => ({ ...prev, pull: 1 }));
    void apiClient.post('/generate/storm/control', { streamId, drift: { pull: 1 } }, { silent: true }).catch(() => {});
    pullRestoreRef.current = { prevPull, slotAtPress };
  }, [streamId, pressure, snapshot, setPressure]);

  // Watch snapshot slot counter; restore pull after the slot at press advances.
  // Guard: if slotAtPress is -1 (no snapshot existed at press time), wait until
  // the slot counter is at least 1 before restoring -- otherwise slot 0 fires
  // the restore immediately and the pull-now has no effect on any real slot.
  useEffect(() => {
    if (!snapshot || !pullRestoreRef.current) return;
    const { prevPull, slotAtPress } = pullRestoreRef.current;
    const threshold = slotAtPress < 0 ? 0 : slotAtPress;
    if (snapshot.slot > threshold) {
      pullRestoreRef.current = null;
      setPressure(prev => ({ ...prev, pull: prevPull }));
      void apiClient.post('/generate/storm/control', { streamId, drift: { pull: prevPull } }, { silent: true }).catch(() => {});
    }
  }, [snapshot, streamId, setPressure]);

  // Update pressure + trigger debounced POST
  const update = useCallback((patch: Partial<DriftPressure>) => {
    setPressure(prev => {
      const next = { ...prev, ...patch };
      postPressure(next);
      return next;
    });
  }, [setPressure, postPressure]);

  // Sync caption input -> pressure when caption changes externally
  useEffect(() => {
    setCaptionInput(pressure.targetCaption.join(', '));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // only on mount

  // Commit caption input on blur or Enter
  const commitCaption = useCallback(() => {
    const tags = captionInput.split(',').map(t => t.trim()).filter(isPresent);
    update({ targetCaption: tags, captionPool: Array.from(new Set([...pressure.captionPool, ...tags])) });
  }, [captionInput, pressure.captionPool, update]);

  // Anchor: snapshot current live values into targets
  const anchor = useCallback(() => {
    if (!snapshot) return;
    const newTargets: Record<string, number> = {};
    for (const spec of specs) {
      if (snapshot.pos[spec.key] !== undefined) {
        newTargets[spec.key] = snapshot.pos[spec.key];
      }
    }
    const newCaption = snapshot.captionWeights
      ? Object.entries(snapshot.captionWeights).filter(([, w]) => w > 0.5).map(([t]) => t)
      : pressure.targetCaption;
    update({ targets: newTargets, targetCaption: newCaption });
    setCaptionInput(newCaption.join(', '));
  }, [snapshot, specs, pressure.targetCaption, update]);

  // Energy macro: push guidance + bpm up, add energetic tags
  const applyEnergy = useCallback(() => {
    const cfgKey = isYue2 ? 'yue2CfgScale' : 'guidanceScale';
    const bpmKey = 'bpm';
    const spec = specs.find(s => s.key === cfgKey);
    const bpmSpec = specs.find(s => s.key === bpmKey);
    const targets = { ...pressure.targets };
    if (spec) targets[cfgKey] = Math.min(1, (targets[cfgKey] ?? 0.5) + 0.25);
    if (bpmSpec) targets[bpmKey] = Math.min(1, (targets[bpmKey] ?? 0.5) + 0.2);
    const energyTags = ['energetic', 'driving', 'powerful', 'fast'];
    const merged = Array.from(new Set([...pressure.targetCaption, ...energyTags]));
    update({ targets, targetCaption: merged });
    setCaptionInput(merged.join(', '));
  }, [isYue2, specs, pressure, update]);

  // Mood macro: replace target caption with preset tags
  const applyMood = useCallback((preset: string) => {
    const all = MOOD_PRESETS[preset] ?? [];
    const custom = (pressure.customTagSets ?? {})[preset] ?? [];
    const merged = Array.from(new Set([...all, ...custom]));
    update({ targetCaption: merged, captionPool: Array.from(new Set([...pressure.captionPool, ...merged])) });
    setCaptionInput(merged.join(', '));
  }, [pressure, update]);

  // Genre macro: set caption to whatever is typed (user sets targetCaption manually)
  const [genreInput, setGenreInput] = useState('');
  const applyGenre = useCallback(() => {
    if (!genreInput.trim()) return;
    const tags = genreInput.split(',').map(t => t.trim()).filter(isPresent);
    update({ targetCaption: tags, captionPool: Array.from(new Set([...pressure.captionPool, ...tags])) });
    setCaptionInput(tags.join(', '));
  }, [genreInput, pressure.captionPool, update]);

  // Add custom tag set
  const addCustomSet = useCallback(() => {
    if (!newSetName.trim() || !newSetTags.trim()) return;
    const tags = newSetTags.split(',').map(t => t.trim()).filter(isPresent);
    const customTagSets = { ...(pressure.customTagSets ?? {}), [newSetName.trim()]: tags };
    update({ customTagSets });
    setNewSetName('');
    setNewSetTags('');
  }, [newSetName, newSetTags, pressure, update]);

  // Target slider change
  const setTarget = useCallback((key: string, u: number) => {
    update({ targets: { ...pressure.targets, [key]: u } });
  }, [pressure, update]);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  const sectionCls = 'space-y-2 bg-zinc-900/50 border border-zinc-800 rounded-lg p-3';
  const labelCls   = 'text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-1.5';

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* Single-column narrow / two-column wide */}
      <div className="flex-1 overflow-y-auto p-4">
        {sa.isPlaying && driftQuery.isError && (
          <QueryError error={driftQuery.error} onRetry={() => void driftQuery.refetch()} className="mb-3 max-w-4xl mx-auto" />
        )}
        <div className="grid gap-4 grid-cols-1 min-[900px]:grid-cols-2 max-w-4xl mx-auto">

          {/* ── Column 1 ── */}
          <div className="space-y-4">

            {/* Target Caption */}
            <div className={sectionCls}>
              <ParamLabel label="Target Caption" info="Comma-separated tags to steer future slots toward. Tags gain weight as the stream drifts." className={labelCls} />
              <textarea
                rows={3}
                aria-label={t('storm.drift.targetCaption', 'Target caption')}
                value={captionInput}
                onChange={e => setCaptionInput(e.target.value)}
                onBlur={commitCaption}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); commitCaption(); } }}
                placeholder="Tags that describe where the stream should head..."
                className="w-full bg-zinc-800/60 border border-zinc-700 rounded-md px-2 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 resize-none focus:outline-none focus:border-purple-600"
              />
              <div className="flex gap-2 mt-1">
                <button
                  onClick={() => { commitCaption(); pullNow(); }}
                  className="flex-1 text-xs bg-purple-700 hover:bg-purple-600 text-white rounded-md py-1 transition-colors"
                >
                  Pull now
                </button>
                <HintButton
                  onClick={anchor}
                  disabled={!snapshot}
                  label={t('storm.drift.anchor', 'Anchor')}
                  info={t('storm.drift.anchorInfo', 'Sets the drift targets to the current live values.')}
                  ariaLabel={t('storm.drift.anchor', 'Anchor')}
                  className="text-xs bg-zinc-700 hover:bg-zinc-600 disabled:opacity-40 text-white rounded-md px-3 py-1 transition-colors"
                >
                  Anchor
                </HintButton>
              </div>
            </div>

            {/* Macros */}
            <div className={sectionCls}>
              <div className={labelCls}>Macros</div>
              {/* Energy */}
              <div className="mb-2">
                <button
                  onClick={applyEnergy}
                  className="w-full text-xs bg-orange-800/60 hover:bg-orange-700/70 text-orange-200 rounded-md py-1.5 transition-colors"
                >
                  Energy boost
                </button>
              </div>
              {/* Mood presets */}
              <div className="grid grid-cols-3 gap-1.5 mb-2">
                {Object.keys(MOOD_PRESETS).map(preset => (
                  <button
                    key={preset}
                    onClick={() => applyMood(preset)}
                    className="text-[11px] bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded-md py-1 capitalize transition-colors"
                  >
                    {preset}
                  </button>
                ))}
              </div>
              {/* Genre */}
              <ParamLabel label="Genre tags" info="Comma-separated tags. Genre replaces the target caption with these tags." className="text-[11px] text-zinc-400" />
              <div className="flex gap-2">
                <input
                  type="text"
                  value={genreInput}
                  aria-label={t('storm.drift.genreTags', 'Genre tags')}
                  onChange={e => setGenreInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') applyGenre(); }}
                  placeholder="Genre tags..."
                  className="flex-1 bg-zinc-800/60 border border-zinc-700 rounded-md px-2 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-purple-600"
                />
                <button
                  onClick={applyGenre}
                  className="text-xs bg-zinc-700 hover:bg-zinc-600 text-white rounded-md px-3 py-1 transition-colors"
                >
                  Genre
                </button>
              </div>
            </div>

            {/* Custom tag sets */}
            <div className={sectionCls}>
              <ParamLabel label="Custom tag sets" info="Save named groups of comma-separated tags. Click a saved set to use it as the target caption." className={labelCls} />
              {Object.entries(pressure.customTagSets ?? {}).map(([name, tags]) => (
                <div key={name} className="flex items-center gap-2 mb-1">
                  <button
                    onClick={() => applyMood(name)}
                    className="text-[11px] bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded px-2 py-0.5 capitalize"
                  >
                    {name}
                  </button>
                  <span className="text-[10px] text-zinc-600 truncate flex-1">{tags.join(', ')}</span>
                  <button
                    onClick={() => {
                      const s = { ...(pressure.customTagSets ?? {}) };
                      delete s[name];
                      update({ customTagSets: s });
                    }}
                    className="text-[10px] text-zinc-600 hover:text-red-400"
                  >
                    x
                  </button>
                </div>
              ))}
              <div className="flex gap-1.5 mt-1">
                <input
                  type="text"
                  value={newSetName}
                  aria-label={t('storm.drift.tagSetName', 'Tag set name')}
                  onChange={e => setNewSetName(e.target.value)}
                  placeholder="Name"
                  className="w-20 bg-zinc-800/60 border border-zinc-700 rounded-md px-1.5 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-purple-600"
                />
                <input
                  type="text"
                  value={newSetTags}
                  aria-label={t('storm.drift.tagSetTags', 'Tag set tags')}
                  onChange={e => setNewSetTags(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') addCustomSet(); }}
                  placeholder="tag1, tag2..."
                  className="flex-1 bg-zinc-800/60 border border-zinc-700 rounded-md px-1.5 py-1 text-xs text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-purple-600"
                />
                <button
                  onClick={addCustomSet}
                  className="text-xs bg-zinc-700 hover:bg-zinc-600 text-white rounded-md px-2 py-1 transition-colors"
                >
                  +
                </button>
              </div>
            </div>

          </div>

          {/* ── Column 2 ── */}
          <div className="space-y-4">

            {/* Drift controls */}
            <div className={sectionCls}>
              <div className={labelCls}>Drift controls</div>
              <div className="space-y-2.5">
                <SliderRow
                  label="Pull"
                  info="How strongly each slot moves toward your targets. Zero lets it wander; one pulls directly toward the targets."
                  value={pressure.pull}
                  onChange={v => update({ pull: v })}
                  fmt={v => formatDecimal(v, 2)}
                />
                <SliderRow
                  label="Rate"
                  info="How far parameters and caption weights move per slot. Higher values change the stream faster."
                  value={pressure.rate}
                  onChange={v => update({ rate: v })}
                  fmt={v => formatDecimal(v, 2)}
                />
                <SliderRow
                  label="Jitter"
                  info="Random variation added to each slot's drift. Higher values make the path less predictable."
                  value={pressure.jitter}
                  onChange={v => update({ jitter: v })}
                  fmt={v => formatDecimal(v, 2)}
                />
              </div>
            </div>

            {/* Per-param targets + locks */}
            <div className={sectionCls}>
              <div className={labelCls}>Param targets</div>
              <div className="space-y-2">
                {specs.map(spec => {
                  const locked = !!pressure.locks[spec.key];
                  const targetU = pressure.targets[spec.key] ?? 0.5;
                  const currentU = snapshot?.pos[spec.key];
                  return (
                    <div key={spec.key} className="space-y-0.5">
                      <div className="flex items-center gap-2">
                        <ParamLabel
                          label={PARAM_LABELS[spec.key] ?? spec.key}
                          info={`${PARAM_INFO[spec.key]} Turn on the lock to exclude this parameter from drift and use its stream setting.`}
                          meta={`Range ${spec.min} to ${spec.max}`}
                          rootClassName="!w-24 shrink-0"
                          className={`text-[11px] ${locked ? 'text-zinc-600' : 'text-zinc-300'}`}
                        />
                        <Toggle size="sm" accent="amber" checked={locked}
                          onChange={checked => update({ locks: { ...pressure.locks, [spec.key]: checked } })}
                          aria-label={`Lock ${PARAM_LABELS[spec.key] ?? spec.key}`}
                        />
                        {!locked && (
                          <input
                            type="range"
                            aria-label={`${PARAM_LABELS[spec.key] ?? spec.key} target`}
                            min={0}
                            max={1}
                            step={0.01}
                            value={targetU}
                            onChange={e => setTarget(spec.key, Number(e.target.value))}
                            className="flex-1 h-1 accent-purple-500"
                          />
                        )}
                        {!locked && (
                          <span className="text-[11px] text-zinc-400 w-10 text-right tabular-nums shrink-0">
                            {fmtVal(spec, targetU)}
                          </span>
                        )}
                      </div>
                      {/* Live current -> target readout */}
                      {!locked && currentU !== undefined && (
                        <div className="flex items-center gap-1 pl-26 ml-[6.5rem]">
                          <span className="text-[10px] text-zinc-600 tabular-nums">
                            now {fmtVal(spec, currentU)}
                          </span>
                          <span className="text-[10px] text-zinc-700">-&gt;</span>
                          <span className="text-[10px] text-purple-500 tabular-nums">
                            {fmtVal(spec, targetU)}
                          </span>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Seed mode */}
            <div className={sectionCls}>
              <ParamLabel label="Seed evolution" info="Lock leaves the stream seed unchanged. Walk advances it each slot, with larger jumps at higher jitter. Jump occasionally picks a random seed; higher jitter makes jumps more likely." className={labelCls} />
              <div className="flex gap-2">
                {(['lock', 'walk', 'jump'] as SeedMode[]).map(m => (
                  <button
                    key={m}
                    onClick={() => update({ seedMode: m })}
                    className={`flex-1 text-xs rounded-md py-1.5 capitalize transition-colors ${
                      pressure.seedMode === m
                        ? 'bg-purple-700 text-white'
                        : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
              {snapshot && (
                <div className="mt-1.5 text-[10px] text-zinc-600 tabular-nums">
                  seed {snapshot.seed} &nbsp; slot {snapshot.slot}
                </div>
              )}
            </div>

            {/* Plugin drift toggle */}
            <div className={sectionCls}>
              <div className="flex items-center gap-3">
                <Toggle size="sm" accent="purple" checked={!!pressure.pluginDrift}
                  onChange={checked => update({ pluginDrift: checked })}
                  label="Drift plugin choice (solver/scheduler/guidance)"
                  info="Allow random solver, scheduler or guidance changes on supported backends. The chance per slot increases with jitter and rate."
                />
              </div>
              {pressure.pluginDrift && (
                <p className="text-[10px] text-zinc-600 mt-1">
                  Solver, scheduler, or guidance may jump at low probability per slot (p = 0.1 x jitter x rate).
                </p>
              )}
            </div>

          </div>

        </div>

        {/* ── Caption tag weight bars (full width below columns) ── */}
        {snapshot && Object.keys(snapshot.captionWeights).length > 0 && (
          <div className={`${sectionCls} max-w-4xl mx-auto mt-4`}>
            <div className={labelCls}>Caption tag weights (live)</div>
            <div className="grid grid-cols-1 min-[600px]:grid-cols-2 gap-1">
              {Object.entries(snapshot.captionWeights)
                .sort((a, b) => b[1] - a[1])
                .map(([tag, w]) => (
                  <TagWeightBar key={tag} tag={tag} weight={w} />
                ))}
            </div>
          </div>
        )}

        {/* Not playing notice */}
        {!sa.isPlaying && (
          <div className="max-w-4xl mx-auto mt-6 text-center text-sm text-zinc-600">
            Start the stream above to enable drift. Pressure settings are saved and will apply when the stream starts.
          </div>
        )}
      </div>
    </div>
  );
}
