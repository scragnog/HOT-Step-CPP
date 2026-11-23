// StormLiveControls.tsx — Shared streaming live controls panel
// Used by both ContentSection (sidebar) and StormPage (full-width)
// MDMAchine / A&E Concepts 2026

import React from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2 } from 'lucide-react';
import { HintButton } from '@/components/shared/controls/HintButton';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { runLlmInspire } from '@/services/inspireApi';
import { AiContinuePresetModal, loadTemplate, DEFAULT_TEMPLATE, type AiPreset, type PresetCategory } from './AiContinuePresetModal';
import { SeedManagerDrawer } from '@/components/global-bar/SeedManagerDrawer';
import { expandWildcards, expandInPlace, hasWildcards, randomWildcardSeed } from '@/utils/wildcardUtils';
import type { useStreamAudio } from '@/hooks/useStreamAudio';
import { formatClock } from '@/utils/format/format';
import { readJsonString } from '@/utils/storage/storageJson';
import { formatPercent } from '@/utils/format/format';
import { errorMessage } from '@/utils/format/errorMessage';
import { isPresent } from '@/utils/guards';

// ── Shared types (re-exported for consumers) ──────────────────────────────────
export type LyricsMode = 'loop' | 'cycle' | 'shuffle';

function parseLyricSections(lyrics: string): string[] {
  const byHeader = lyrics.split(/\n(?=\[)/).map(s => s.trim()).filter(isPresent);
  if (byHeader.length > 1) return byHeader;
  const byBlank = lyrics.split(/\n\n+/).map(s => s.trim()).filter(isPresent);
  if (byBlank.length > 1) return byBlank;
  return [lyrics.trim()].filter(isPresent);
}

const fmtT = (s: number) => formatClock(s);

// ── Next-slot param definitions ───────────────────────────────────────────────
const NEXT_SLOT_PARAMS = [
  { label: 'Guidance', min: 0.5, max: 15,  step: 0.1, key: 'guidance_scale',  fmt: (v: number) => v.toFixed(1),         emoji: '📐', color: 'text-blue-400'   },
  { label: 'Steps',    min: 4,   max: 50,  step: 1,   key: 'inference_steps', fmt: (v: number) => String(Math.round(v)), emoji: '⚡', color: 'text-yellow-400' },
  { label: 'Duration', min: 10,  max: 300, step: 5,   key: 'duration',        fmt: (v: number) => `${Math.round(v)}s`,  emoji: '⏱', color: 'text-green-400'  },
  { label: 'BPM',      min: 60,  max: 200, step: 1,   key: 'bpm',             fmt: (v: number) => String(Math.round(v)), emoji: '🥁', color: 'text-orange-400' },
] as const;

type SlotKey = (typeof NEXT_SLOT_PARAMS)[number]['key'];

interface PendingItem {
  id: string;
  label: string;
  value: string;   // short pill text
  detail?: string; // full untruncated text for expanded view
  emoji: string;
  color: string;
  targetSlot: number;
}

// Metadata stamped when a slot transitions to PLAYING
export interface SlotMeta {
  seed?: number;
  detectedKey?: string;
  detectedBpm?: number;
  guidance?: number;
  steps?: number;
  duration?: number;
  bpm?: number;
  style?: string;   // full style text sent
  lyrics?: string;  // full lyrics text sent
  solver?: string;
  scheduler?: string;
  guidanceMode?: string;
}

// ── Props ─────────────────────────────────────────────────────────────────────
type StreamAudioReturn = ReturnType<typeof useStreamAudio>;

export interface StormLiveControlsProps {
  sa: StreamAudioReturn;
  // Main content (for AI + verse cycling)
  lyrics: string;
  caption: string;
  // Next-slot fields (managed by parent)
  streamPrompt: string; onStreamPromptChange: (v: string) => void;
  streamLyrics: string; onStreamLyricsChange: (v: string) => void;
  persistStyle: boolean; onPersistStyleChange: (v: boolean) => void;
  persistLyrics: boolean; onPersistLyricsChange: (v: boolean) => void;
  streamSeed: number; onStreamSeedChange: (v: number) => void;
  streamSeedLock: boolean; onStreamSeedLockChange: (v: boolean) => void;
  onStreamSend: () => void;
  // Slider defaults
  gpGuidanceScale?: number; gpInferenceSteps?: number; gpBpm?: number; gpDuration?: number;
  // Layout
  showMiniPlayer?: boolean;  // true = continuous mode (show progress bar + vol)
  alwaysShowControls?: boolean;
  onParamsChange?: (p: {guidance_scale:number, inference_steps:number, duration:number, bpm:number}) => void;
  onStop: () => void;
  // Verse cycling (from parent — buttons may live in parent's lyrics section)
  lyricsMode?: LyricsMode;
  streamSlotPlaying?: number; // redundant with sa.playingSlot but kept for clarity
  // Active engine config — stamped into slot metadata for timeline display
  activeSolver?: string;
  activeScheduler?: string;
  activeGuidanceMode?: string;
  onSlotMetaUpdate?: (meta: Map<number, SlotMeta>) => void;
  /** When true, hides ACE-only controls (Steps, Key, BPM). Pass for YuE2. */
  hideAceControls?: boolean;
}

// ── Component ─────────────────────────────────────────────────────────────────
export const StormLiveControls: React.FC<StormLiveControlsProps> = ({
  sa,
  lyrics, caption,
  streamPrompt, onStreamPromptChange,
  streamLyrics, onStreamLyricsChange,
  persistStyle, onPersistStyleChange,
  persistLyrics, onPersistLyricsChange,
  streamSeed, onStreamSeedChange,
  streamSeedLock, onStreamSeedLockChange,
  onStreamSend,
  gpGuidanceScale = 7, gpInferenceSteps = 8, gpBpm = 120, gpDuration = 120,
  showMiniPlayer = false,
  alwaysShowControls = false,
  onParamsChange,
  onStop,
  lyricsMode = 'loop',
  activeSolver,
  activeScheduler,
  activeGuidanceMode,
  hideAceControls = false,
  onSlotMetaUpdate,
}) => {
  const { t } = useTranslation();
  const received = sa.currentSlot || 0;
  const playing  = sa.playingSlot || 0;

  // ── Slider state ─────────────────────────────────────────────────────────
  const initSliderVals = () => ({
    guidance_scale: gpGuidanceScale, inference_steps: gpInferenceSteps,
    duration: gpDuration, bpm: gpBpm,
  });
  const [sliderVals, setSliderVals] = React.useState<Record<SlotKey, number>>(initSliderVals);
  // Report params to parent on mount + change
  React.useEffect(() => { onParamsChange?.(sliderVals as any); }, [sliderVals]);
  const [editingKey,  setEditingKey]  = React.useState<SlotKey | null>(null);
  const [keyMode,     setKeyMode]     = React.useState<'auto'|'manual'>('auto');
  const [manualKey,   setManualKey]   = React.useState('');
  const [bpmMode,     setBpmMode]     = React.useState<'auto'|'manual'>('manual');
  const [editRaw,    setEditRaw]    = React.useState('');
  const [seedEditRaw, setSeedEditRaw] = React.useState('');
  const [editingSeed,      setEditingSeed]      = React.useState(false);
  const [seedDrawerOpen,   setSeedDrawerOpen]   = React.useState(false);
  const [autoExpand,       setAutoExpand]       = React.useState(true);
  const styleTextareaRef  = React.useRef<HTMLTextAreaElement>(null);
  const lyricsTextareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Reset sliders when stream starts
  // Sliders initialize from props on mount only — no reset on Start

  // ── Pending items ─────────────────────────────────────────────────────────
  const [pendingItems,  setPendingItems]  = React.useState<PendingItem[]>([]);
  const [, setAppliedItems] = React.useState<PendingItem[]>([]); // history kept for display
  const [slotMeta,      setSlotMeta]      = React.useState<Map<number, SlotMeta>>(new Map());
  const [sendFlash,     setSendFlash]     = React.useState<'idle' | 'sent'>('idle');
  const debounceTimers  = React.useRef<Record<string, ReturnType<typeof setTimeout>>>({});
  const [appliedFlash, setAppliedFlash] = React.useState<number | null>(null);
  const [stuckStyle,   setStuckStyle]   = React.useState<string | null>(null);
  const [stuckLyrics,  setStuckLyrics]  = React.useState<string | null>(null);
  const [promptTab,      setPromptTab]      = React.useState<'style' | 'lyrics'>('style');
  const [loraTrigger,    setLoraTrigger]    = React.useState('');
  const [beatIntroOutro, setBeatIntroOutro] = React.useState(false);
  const [introOutroBars, setIntroOutroBars] = React.useState(2);

  const prevPlayingSlot = React.useRef(0);
  React.useEffect(() => {
    const ps = playing;
    if (ps > prevPlayingSlot.current) {
      const nowApplied = pendingItems.filter(i => i.targetSlot <= ps);
      if (nowApplied.length > 0) {
        setAppliedFlash(ps);
        setTimeout(() => setAppliedFlash(null), 2000);
        setAppliedItems(prev => [...prev, ...nowApplied].slice(-20));
      }
      // Stamp slot metadata when it becomes PLAYING
      setSlotMeta(prev => {
        const next = new Map(prev);
        next.set(ps, {
          seed:         streamSeed + ps,
          detectedKey:  sa.detectedKey || undefined,
          detectedBpm:  sa.detectedBpm > 0 ? sa.detectedBpm : undefined,
          guidance:     sliderVals['guidance_scale'],
          steps:        sliderVals['inference_steps'],
          duration:     sliderVals['duration'],
          bpm:          sliderVals['bpm'],
          // pull style/lyrics from most recent applied item for this slot
          style:  nowApplied.find(i => i.id === 'style')?.detail,
          lyrics: nowApplied.find(i => i.id === 'lyrics')?.detail,
          solver:       activeSolver   || undefined,
          scheduler:    activeScheduler || undefined,
          guidanceMode: activeGuidanceMode || undefined,
        });
        // Keep only last 20 slots
        if (next.size > 20) next.delete(Math.min(...next.keys()));
        onSlotMetaUpdate?.(next);
        return next;
      });
      setPendingItems(p => p.filter(i => i.targetSlot > ps));
      prevPlayingSlot.current = ps;
    }
  }, [playing]);

  React.useEffect(() => {
    if (!sa.isPlaying) {
      setPendingItems([]); setAppliedItems([]); setSlotMeta(new Map()); prevPlayingSlot.current = 0;
      setStuckStyle(null); setStuckLyrics(null);
    }
  }, [sa.isPlaying]);

  // ── Verse cycling (auto-send on slot advance) ─────────────────────────────
  const [lyricSectionIdx, setLyricSectionIdx] = React.useState(0);
  const lyricSections = React.useMemo(() => parseLyricSections(lyrics), [lyrics]);
  const prevPlayingForCycle = React.useRef(0);

  React.useEffect(() => {
    if (!sa.isPlaying || lyricsMode === 'loop' || lyricSections.length <= 1) return;
    if (playing <= prevPlayingForCycle.current) return;
    prevPlayingForCycle.current = playing;
    let nextIdx: number;
    if (lyricsMode === 'cycle') {
      nextIdx = (lyricSectionIdx + 1) % lyricSections.length;
    } else {
      const options = lyricSections.map((_, i) => i).filter(i => i !== lyricSectionIdx);
      nextIdx = options[Math.floor(Math.random() * options.length)] ?? 0;
    }
    setLyricSectionIdx(nextIdx);
    sa.sendControl('lyrics', expandWildcards(lyricSections[nextIdx], streamSeed ^ (playing || 0), playing || 0));
  }, [playing]);

  React.useEffect(() => {
    if (!sa.isPlaying) { setLyricSectionIdx(0); prevPlayingForCycle.current = 0; }
  }, [sa.isPlaying]);
  React.useEffect(() => { setLyricSectionIdx(0); }, [lyrics]);

  // ── AI Continuation ───────────────────────────────────────────────────────
  const [aiContinueEnabled,  setAiContinueEnabled]  = React.useState(false);
  const [aiStatus,           setAiStatus]           = React.useState<{msg:string, type:'idle'|'pending'|'ok'|'err'|'warn'}>({msg:'', type:'idle'});
  const [aiContinueInterval, setAiContinueInterval] = React.useState(1);
  const [aiContinueDir,      setAiContinueDir]      = React.useState('');
  const [aiContinuePending,  setAiContinuePending]  = React.useState(false);
  const [aiWarnDisabled,     setAiWarnDisabled]      = React.useState(false);
  const [presetModalOpen,    setPresetModalOpen]     = React.useState(false);
  const [aiTemplate,         setAiTemplate]          = React.useState(() => loadTemplate());
  const isCustomTemplate = aiTemplate.trim() !== DEFAULT_TEMPLATE.trim();
  // AI can optionally prefix response with [meta: bpm=128, key=Cm, duration=90, seed=42069]
  const [recentPresets,      setRecentPresets]       = React.useState<AiPreset[]>([]);
  const aiFailuresRef   = React.useRef(0);
  const slotsSinceAiRef = React.useRef(0);
  const prevPlayingForAi = React.useRef(0);

  const triggerAiContinuation = React.useCallback(async (targetSlot: number) => {
    const provider = readJsonString('hs-customgen-ai-provider');
    const model    = readJsonString('hs-customgen-ai-model');
    if (!provider) {
      setAiStatus({ msg: 'No AI provider configured — set one in AI Generate settings', type: 'warn' });
      console.warn('[AI ↻] no provider configured');
      return;
    }

    // Use most recently played slot lyrics as context — the static `lyrics` prop
    // is the original compose textarea and goes stale immediately after stream start.
    const recentSlotLyrics = (() => {
      let latest: string | undefined; let latestSlot = -1;
      for (const [slot, meta] of slotMeta) {
        if (meta.lyrics && slot > latestSlot) { latestSlot = slot; latest = meta.lyrics; }
      }
      return latest;
    })();
    const contextLyrics = recentSlotLyrics || lyrics;

    if (!contextLyrics.trim()) {
      setAiStatus({ msg: 'No lyrics to continue — add lyrics first', type: 'warn' });
      return;
    }
    console.log(`[AI ↻] firing for slot ${targetSlot}, provider=${provider}, lyrics from ${recentSlotLyrics ? `slot meta` : 'compose'}`);
    setAiStatus({ msg: `Generating… (slot ${targetSlot})`, type: 'pending' });
    setAiContinuePending(true);
    try {
      const result = await Promise.race([
        runLlmInspire({
          provider, model: model || undefined,
          genres: caption ? [caption] : ['any genre'],
          subject: caption || 'song continuation',
          language: 'en',
          systemPrompt: aiTemplate
            .replace('{direction}', aiContinueDir ? `\n\nDirection: ${aiContinueDir}.` : '')
            .replace('{lyrics}', contextLyrics)
            .replace('{style}', caption),
        }),
        new Promise<never>((_, r) => setTimeout(() => r(new Error('AI timeout')), 30000)),
      ]);
      if (result?.lyrics) {
        // Parse optional meta tags: [meta: bpm=128, key=Cm, duration=90, seed=42069]
        let lyricsText: string = result.lyrics;
        const metaMatch = lyricsText.match(/^\[meta:\s*([^\]]+)\]\s*\n?/i);
        if (metaMatch) {
          lyricsText = lyricsText.slice(metaMatch[0].length).trim();
          const metaStr = metaMatch[1];
          const parseMeta = (key: string) => { const m = metaStr.match(new RegExp(`${key}\\s*=\\s*([\\w#.]+)`, 'i')); return m ? m[1] : null; };
          const mBpm = parseMeta('bpm');
          const mKey = parseMeta('key');
          const mDur = parseMeta('duration');
          const mSeed = parseMeta('seed');
          if (mBpm && !isNaN(Number(mBpm))) { sa.sendControl('bpm', Number(mBpm)); console.log('[AI ↻] meta bpm:', mBpm); }
          if (mKey) { sa.sendControl('key', mKey); console.log('[AI ↻] meta key:', mKey); }
          if (mDur && !isNaN(Number(mDur))) { sa.sendControl('next_duration', Number(mDur)); console.log('[AI ↻] meta duration:', mDur); }
          if (mSeed && !isNaN(Number(mSeed))) { sa.sendControl('seed', Number(mSeed)); console.log('[AI ↻] meta seed:', mSeed); }
        }
        console.log('[AI ↻] got lyrics, queuing for slot', targetSlot);
        sa.sendControl('lyrics', lyricsText);
        setAiStatus({ msg: `✓ lyrics queued for slot ${targetSlot}`, type: 'ok' });
        setTimeout(() => setAiStatus({ msg: '', type: 'idle' }), 4000);
        setPendingItems(prev => [...prev.filter(i => i.id !== 'ai-lyrics'), {
          id: 'ai-lyrics', label: 'AI Lyrics', value: 'AI continuation…',
          emoji: '🤖', color: 'text-violet-400', targetSlot,
        }]);
        aiFailuresRef.current = 0;
      } else {
        console.warn('[AI ↻] result had no lyrics field — full result:', JSON.stringify(result));
        setAiStatus({ msg: 'AI returned no lyrics — check console', type: 'err' });
        setTimeout(() => setAiStatus({ msg: '', type: 'idle' }), 5000);
      }
    } catch (e: unknown) {
      const msg = errorMessage(e);
      console.warn('[AI ↻] failed:', msg);
      setAiStatus({ msg: `Failed: ${msg.slice(0, 40)}`, type: 'err' });
      setTimeout(() => setAiStatus({ msg: '', type: 'idle' }), 5000);
      aiFailuresRef.current += 1;
      if (aiFailuresRef.current >= 3) {
        setAiContinueEnabled(false); setAiWarnDisabled(true); aiFailuresRef.current = 0;
        setAiStatus({ msg: '3 failures — re-enable to retry', type: 'err' });
      }
    } finally { setAiContinuePending(false); }
  }, [lyrics, caption, aiContinueDir, sa, slotMeta]);

  React.useEffect(() => {
    if (!sa.isPlaying || !aiContinueEnabled || aiWarnDisabled) return;
    if (playing <= prevPlayingForAi.current) return;
    prevPlayingForAi.current = playing;
    slotsSinceAiRef.current += 1;
    if (slotsSinceAiRef.current >= aiContinueInterval) {
      slotsSinceAiRef.current = 0;
      if (!aiContinuePending) {
        triggerAiContinuation(received + 2);
      } else { console.log('[AI ↻] interval hit but LLM still pending — skipping'); }
    }
  }, [playing]);

  React.useEffect(() => {
    if (!sa.isPlaying) { slotsSinceAiRef.current = 0; prevPlayingForAi.current = 0; setAiWarnDisabled(false); aiFailuresRef.current = 0; }
  }, [sa.isPlaying]);

  const handlePresetFire = React.useCallback((preset: AiPreset, category: PresetCategory) => {
    setRecentPresets(prev => [preset, ...prev.filter(p => p.id !== preset.id)].slice(0, 3));
    const targetSlot = received + 2;
    const provider = readJsonString('hs-customgen-ai-provider');
    const model    = readJsonString('hs-customgen-ai-model');
    if (!provider) return;
    // Use most recent slot lyrics for lyric presets, same as triggerAiContinuation
    const recentSlotLyrics = (() => {
      let latest: string | undefined; let latestSlot = -1;
      for (const [slot, meta] of slotMeta) {
        if (meta.lyrics && slot > latestSlot) { latestSlot = slot; latest = meta.lyrics; }
      }
      return latest;
    })();
    const contextLyrics = category === 'lyric' ? (recentSlotLyrics || lyrics) : caption;
    const continuationPrompt = aiTemplate
      .replace('{direction}', `\n\nDirection: ${preset.value}.`)
      .replace('{lyrics}', contextLyrics)
      .replace('{style}', caption);
    setAiContinuePending(true);
    Promise.race([
      runLlmInspire({
        provider, model: model || undefined,
        genres: caption ? [caption] : ['any genre'],
        subject: caption || 'song continuation',
        language: 'en',
        systemPrompt: continuationPrompt,
      }),
      new Promise<never>((_, r) => setTimeout(() => r(new Error('timeout')), 30000)),
    ]).then(result => {
      if (result?.lyrics) {
        sa.sendControl(category === 'lyric' ? 'lyrics' : 'prompt', result.lyrics);
        setPendingItems(prev => [...prev.filter(i => i.id !== 'ai-preset'), {
          id: 'ai-preset', label: preset.label, value: preset.label,
          emoji: category === 'style' ? '🎨' : '🎤', color: 'text-violet-400', targetSlot,
        }]);
      }
    }).catch(() => {}).finally(() => setAiContinuePending(false));
  }, [aiTemplate, lyrics, caption, sa, received, slotMeta]);

  // ── Handlers ──────────────────────────────────────────────────────────────
  const fireControl = (key: SlotKey, val: number) => {
    // Debounce sendControl — update pill immediately, but only POST after 150ms idle
    const meta = NEXT_SLOT_PARAMS.find(x => x.key === key)!;
    const targetSlot = received + 2;
    setPendingItems(prev => {
      const filtered = prev.filter(i => !(i.id === key && i.targetSlot === targetSlot));
      return [...filtered, { id: key, label: meta.label, value: meta.fmt(val), emoji: meta.emoji, color: meta.color, targetSlot }];
    });
    if (debounceTimers.current[key]) clearTimeout(debounceTimers.current[key]);
    debounceTimers.current[key] = setTimeout(() => {
      sa.sendControl(key, val);
    }, 150);
  };

  const handleSend = () => {
    const targetSlot = received + 2;
    const newItems: PendingItem[] = [];
    const expandSeed = streamSeed ^ (playing || 0);
    if (streamPrompt.trim()) {
      const beatText = beatIntroOutro ? `, with a clean ${introOutroBars}-bar percussive intro and outro for DJ mixing` : '';
      const loraText = loraTrigger.trim() ? `${loraTrigger.trim()}, ` : '';
      const resolvedPrompt = autoExpand ? expandWildcards(streamPrompt, expandSeed, playing || 0) : streamPrompt;
      const fullPrompt = `${loraText}${resolvedPrompt}${beatText}`;
      sa.sendControl('prompt', fullPrompt);
      // Auto-stick when pin is ON — pin controls persistence
      if (persistStyle) {
        sa.sendControl('stick_prompt', fullPrompt);
        setStuckStyle(streamPrompt);
      }
      newItems.push({ id: 'style', label: 'Style', value: (loraTrigger.trim() ? '🔌 ' : '') + (beatIntroOutro ? '🥁 ' : '') + (persistStyle ? '📌 ' : '') + streamPrompt.slice(0, 12) + (streamPrompt.length > 12 ? '…' : ''), detail: streamPrompt, emoji: '🎨', color: 'text-pink-400', targetSlot });
    }
    // Only send lyrics if the user explicitly typed something in the live lyrics field.
    // Do NOT fall back to main compose lyrics — causes unintended lyric pushes on style-only Sends.
    const rawLyrics = streamLyrics.trim();
    const lyricsToSend = rawLyrics && autoExpand ? expandWildcards(rawLyrics, expandSeed, playing || 0) : rawLyrics;
    if (lyricsToSend) {
      sa.sendControl('lyrics', lyricsToSend);
      if (persistLyrics) {
        sa.sendControl('stick_lyrics', lyricsToSend);
        setStuckLyrics(lyricsToSend.split('\n')[0].slice(0, 20));
      }
      const preview = (persistLyrics ? '📌 ' : '') + lyricsToSend.split('\n')[0].slice(0, 16) + (lyricsToSend.length > 16 ? '…' : '');
      newItems.push({ id: 'lyrics', label: 'Lyrics', value: preview, detail: lyricsToSend, emoji: '🎤', color: 'text-purple-400', targetSlot });
    }
    if (newItems.length) setPendingItems(prev => [...prev.filter(i => !newItems.find(n => n.id === i.id && n.targetSlot === i.targetSlot)), ...newItems]);
    setSendFlash('sent');
    setTimeout(() => setSendFlash('idle'), 1500);
    if (!persistStyle)  onStreamPromptChange('');
    if (!persistLyrics) onStreamLyricsChange('');
    onStreamSend();
  };

  const handleUnstick = (field: 'style' | 'lyrics') => {
    if (field === 'style') { sa.sendControl('stick_prompt', null as any); setStuckStyle(null); }
    else                   { sa.sendControl('stick_lyrics', null as any); setStuckLyrics(null); }
  };

  const commitEdit = (key: SlotKey, raw: string, min: number, max: number, step: number) => {
    const parsed = parseFloat(raw);
    if (!isNaN(parsed)) {
      const snapped = Math.round(Math.min(max, Math.max(min, parsed)) / step) * step;
      setSliderVals(p => ({ ...p, [key]: snapped }));
      fireControl(key, snapped);
    }
    setEditingKey(null);
  };

  const commitSeedEdit = (raw: string) => {
    const parsed = parseInt(raw, 10);
    if (!isNaN(parsed) && parsed >= 0) { onStreamSeedChange(parsed); sa.sendControl('seed', parsed); }
    setEditingSeed(false);
  };

  const randomizeSeed = () => {
    const s = randomWildcardSeed();
    onStreamSeedChange(s); sa.sendControl('seed', s);
  };

  const handleSeedLoad = React.useCallback((seed: number) => {
    onStreamSeedChange(seed); sa.sendControl('seed', seed);
    setSeedDrawerOpen(false);
  }, [onStreamSeedChange, sa]);

  const handleSeedLoadRandom = React.useCallback((seed: number) => {
    onStreamSeedChange(seed); sa.sendControl('seed', seed);
  }, [onStreamSeedChange, sa]);

  const handleExpandWildcards = React.useCallback(
    (field: 'style' | 'lyrics') => {
      const expandSeed = streamSeed ^ (playing || 0);
      const ref = field === 'style' ? styleTextareaRef : lyricsTextareaRef;
      if (ref.current) {
        const { value, selectionStart, selectionEnd } = expandInPlace(ref.current, expandSeed, playing || 0);
        if (field === 'style') {
          onStreamPromptChange(value);
          requestAnimationFrame(() => { ref.current?.setSelectionRange(selectionStart, selectionEnd); ref.current?.focus(); });
        } else {
          onStreamLyricsChange(value);
          requestAnimationFrame(() => { ref.current?.setSelectionRange(selectionStart, selectionEnd); ref.current?.focus(); });
        }
      } else {
        if (field === 'style') onStreamPromptChange(expandWildcards(streamPrompt, expandSeed, playing || 0));
        else onStreamLyricsChange(expandWildcards(streamLyrics, expandSeed, playing || 0));
      }
    },
    [streamSeed, playing, streamPrompt, streamLyrics, onStreamPromptChange, onStreamLyricsChange],
  );

  // ── Render ────────────────────────────────────────────────────────────────
  if (!alwaysShowControls && !sa.isPlaying) return null;

  return (
    <div className="space-y-2">

      {/* Mini player — continuous mode only */}
      {sa.isPlaying && showMiniPlayer && (
        <div className="bg-zinc-900/80 rounded-lg p-2.5 space-y-2 border border-zinc-700/50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              <span className="text-xs text-zinc-300 font-medium">
                {appliedFlash ? `✓ applied at slot ${appliedFlash}` : `Slot ${playing || received || 0}`}
              </span>
              {sa.detectedKey && (
                <ParamLabel label={t('storm.live.detectedKey', 'Detected key')}
                  info={t('storm.live.detectedKeyInfo', 'Key detected from the audio in the current slot.')}>
                  <span className="text-[9px] text-violet-400 font-medium">🎹 {sa.detectedKey}</span>
                </ParamLabel>
              )}
              {sa.detectedBpm > 0 && (
                <ParamLabel label={t('storm.live.measuredBpm', 'Measured BPM')}
                  info={t('storm.live.measuredBpmInfo', 'Tempo measured from the audio in the current slot.')}>
                  <span className="text-[9px] text-orange-400 tabular-nums">♩{sa.detectedBpm}</span>
                </ParamLabel>
              )}
              {/* Record button */}
              <button onClick={sa.isRecording ? sa.stopRecording : sa.startRecording}
                className={`flex items-center gap-0.5 text-[9px] px-1.5 py-0.5 rounded font-medium transition-all ${
                  sa.isRecording ? 'bg-red-600 text-white animate-pulse' : 'bg-zinc-800 text-zinc-500 hover:text-red-400 border border-zinc-700'}`}>
                {sa.isRecording
                  ? `⏹ ${Math.floor((sa.recordingTime||0)/60)}:${String((sa.recordingTime||0)%60).padStart(2,'0')}`
                  : '🔴 REC'}
              </button>
            </div>
            <span className="text-[10px] text-zinc-500 tabular-nums">{fmtT(sa.currentTime)} / {fmtT(sa.bufferedTime)}</span>
          </div>
          <div className="w-full h-1 rounded-full bg-zinc-800 overflow-hidden">
            <div className="h-full rounded-full bg-gradient-to-r from-red-600 to-pink-500 transition-all"
              style={{ width: `${sa.bufferedTime ? Math.min(100, (sa.currentTime / sa.bufferedTime) * 100) : 0}%` }} />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-zinc-500 w-8">Vol</span>
            <input type="range" min={0} max={1} step={0.05} value={sa.volume}
              onChange={e => sa.setVolume(Number(e.target.value))} className="flex-1 accent-red-500 h-1" />
            <span className="text-[10px] text-zinc-400 w-8 text-right tabular-nums">{formatPercent(sa.volume)}</span>
          </div>
        </div>
      )}

      {/* Live controls panel */}
      <div className="bg-zinc-900/80 rounded-lg p-2.5 space-y-2.5 border border-zinc-700/50">

        {/* Sequential slot header (no mini player) */}
        {sa.isPlaying && !showMiniPlayer && (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              <span className="text-xs text-zinc-300 font-medium">
                {appliedFlash ? `✓ applied at slot ${appliedFlash}` : `Slot ${playing || received || 0}`}
              </span>
              {sa.detectedKey && <span className="text-[9px] text-violet-400">🎹 {sa.detectedKey}</span>}
              {sa.detectedBpm > 0 && <span className="text-[9px] text-orange-400 tabular-nums">♩{sa.detectedBpm}</span>}
            </div>
            <button onClick={sa.isRecording ? sa.stopRecording : sa.startRecording}
              className={`text-[9px] px-1.5 py-0.5 rounded font-medium transition-all ${
                sa.isRecording ? 'bg-red-600 text-white animate-pulse' : 'bg-zinc-800 text-zinc-500 hover:text-red-400 border border-zinc-700'}`}>
              {sa.isRecording ? `⏹ ${Math.floor((sa.recordingTime||0)/60)}:${String((sa.recordingTime||0)%60).padStart(2,'0')}` : '🔴 REC'}
            </button>
          </div>
        )}

        {/* Next-slot sliders */}
        {NEXT_SLOT_PARAMS.filter(s => s.key !== 'bpm' && !(hideAceControls && s.key === 'inference_steps')).map(s => (
          <div key={s.key} className="flex items-center gap-2">
            <span className={`text-[10px] w-14 ${s.color}`}>{s.emoji} {s.label}</span>
            <input type="range" min={s.min} max={s.max} step={s.step}
              value={sliderVals[s.key]}
              onChange={e => { const v = Number(e.target.value); setSliderVals(p => ({ ...p, [s.key]: v })); fireControl(s.key, v); }}
              className="flex-1 accent-pink-500 h-1" />
            {editingKey === s.key ? (
              <input type="number" min={s.min} max={s.max} step={s.step}
                value={editRaw} onChange={e => setEditRaw(e.target.value)}
                onBlur={() => commitEdit(s.key, editRaw, s.min, s.max, s.step)}
                onKeyDown={e => { if (e.key === 'Enter') commitEdit(s.key, editRaw, s.min, s.max, s.step); if (e.key === 'Escape') setEditingKey(null); }}
                autoFocus className="w-12 text-right text-[10px] bg-zinc-700 border border-pink-500/60 rounded px-1 py-0.5 text-zinc-200 outline-none tabular-nums" />
            ) : (
              <span title="Click to type"
                onClick={() => { setEditingKey(s.key); setEditRaw(String(sliderVals[s.key])); }}
                className={`w-12 text-right text-[10px] tabular-nums cursor-text rounded px-1 py-0.5 transition-colors select-none ${
                  pendingItems.some(i => i.id === s.key) ? `${s.color} font-semibold` : 'text-zinc-300 hover:text-white hover:bg-zinc-700'}`}>
                {s.fmt(sliderVals[s.key])}
              </span>
            )}
          </div>
        ))}

        {/* Seed row */}
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-zinc-500 w-14">🎲 Seed</span>
          {editingSeed ? (
            <input type="number" min={0} max={2147483647}
              value={seedEditRaw} onChange={e => setSeedEditRaw(e.target.value)}
              onBlur={() => commitSeedEdit(seedEditRaw)}
              onKeyDown={e => { if (e.key === 'Enter') commitSeedEdit(seedEditRaw); if (e.key === 'Escape') setEditingSeed(false); }}
              autoFocus className="flex-1 text-[10px] bg-zinc-700 border border-pink-500/60 rounded px-1.5 py-0.5 text-zinc-200 outline-none tabular-nums" />
          ) : (
            <span title="Click to set seed"
              onClick={() => { setEditingSeed(true); setSeedEditRaw(String(streamSeed)); }}
              className="flex-1 text-[10px] tabular-nums cursor-text rounded px-1.5 py-0.5 transition-colors select-none">
              <span className="text-zinc-400">{streamSeed}</span>
              {sa.isPlaying && playing > 0 && (
                <span className="text-zinc-600 ml-1">→ <span className="text-green-400/80">{streamSeed + playing}</span></span>
              )}
            </span>
          )}
          <HintButton onClick={randomizeSeed}
            label={t('storm.live.randomizeSeed', 'Randomize seed')}
            info={t('storm.live.randomizeSeedInfo', 'Picks a new random seed for the next slot.')}
            ariaLabel={t('storm.live.randomizeSeed', 'Randomize seed')}
            className="text-[11px] px-1.5 py-0.5 rounded hover:bg-zinc-700 text-zinc-400 hover:text-white transition-colors">🎲</HintButton>
          <HintButton onClick={() => setSeedDrawerOpen(v => !v)}
            label={t('storm.live.seedManager', 'Seed manager')}
            info={t('storm.live.seedManagerInfo', 'Save the current stream seed or load a saved one.')}
            ariaLabel={t('storm.live.seedManager', 'Seed manager')}
            className={`text-[11px] px-1.5 py-0.5 rounded transition-colors ${seedDrawerOpen ? 'bg-amber-600/30 text-amber-300' : 'text-zinc-600 hover:bg-zinc-700 hover:text-zinc-400'}`}>
            💾
          </HintButton>
          <button
            onClick={() => { const v = !streamSeedLock; onStreamSeedLockChange(v); sa.sendControl('seed_lock', v); }}
            title={streamSeedLock ? 'Seed locked' : 'Seed unlocked'}
            className={`text-[11px] px-1.5 py-0.5 rounded transition-colors ${streamSeedLock ? 'bg-pink-600/30 text-pink-300' : 'text-zinc-600 hover:bg-zinc-700 hover:text-zinc-400'}`}>
            {streamSeedLock ? '🔒' : '🔓'}
          </button>
        </div>
        <SeedManagerDrawer
          isOpen={seedDrawerOpen}
          onClose={() => setSeedDrawerOpen(false)}
          currentSeed={streamSeed}
          onLoad={handleSeedLoad}
          onLoadRandom={handleSeedLoadRandom}
        />

        {/* Key control row — ACE only */}
        {!hideAceControls && <div className="flex items-center gap-2">
          <span className="text-[10px] text-zinc-500 w-14">🎹 Key</span>
          <button onClick={() => setKeyMode(keyMode === 'auto' ? 'manual' : 'auto')}
            className={`text-[8px] px-1.5 py-0.5 rounded font-medium transition-colors shrink-0 ${
              keyMode === 'auto' ? 'bg-violet-600/40 text-violet-300' : 'bg-zinc-800 text-zinc-500 hover:text-zinc-300'}`}>
            {keyMode === 'auto' ? 'AUTO' : 'MAN'}
          </button>
          {keyMode === 'auto' ? (
            <span className={`flex-1 text-[10px] tabular-nums px-1.5 py-0.5 rounded ${sa.detectedKey ? 'text-violet-400' : 'text-zinc-600 italic'}`}>
              {sa.detectedKey || (sa.isPlaying ? 'detecting…' : '—')}
            </span>
          ) : (
            <input type="text" value={manualKey} onChange={e => setManualKey(e.target.value)}
              onBlur={() => { if (manualKey.trim()) sa.sendControl('key', manualKey.trim()); }}
              onKeyDown={e => { if (e.key === 'Enter' && manualKey.trim()) sa.sendControl('key', manualKey.trim()); }}
              placeholder="e.g. Cm, F#, Bbm"
              className="flex-1 text-[10px] bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-zinc-200 placeholder:text-zinc-600 outline-none focus:border-violet-500/40 transition-colors" />
          )}
        </div>}

        {/* BPM auto row — ACE only */}
        {!hideAceControls && <div className="flex items-center gap-2">
          <span className="text-[10px] text-zinc-500 w-14">🥁 BPM</span>
          <button onClick={() => setBpmMode(bpmMode === 'auto' ? 'manual' : 'auto')}
            className={`text-[8px] px-1.5 py-0.5 rounded font-medium transition-colors shrink-0 ${
              bpmMode === 'auto' ? 'bg-orange-600/40 text-orange-300' : 'bg-zinc-800 text-zinc-500 hover:text-zinc-300'}`}>
            {bpmMode === 'auto' ? 'AUTO' : 'MAN'}
          </button>
          {bpmMode === 'auto' ? (
            <span className={`flex-1 text-[10px] tabular-nums px-1.5 py-0.5 rounded ${sa.detectedBpm > 0 ? 'text-orange-400' : 'text-zinc-600 italic'}`}>
              {sa.detectedBpm > 0 ? `♩${sa.detectedBpm}` : (sa.isPlaying ? 'detecting…' : '—')}
            </span>
          ) : (
            <>
              <input type="range" min={60} max={200} step={1}
                value={sliderVals['bpm']}
                onChange={e => { const v=Number(e.target.value); setSliderVals(p=>({...p,bpm:v})); fireControl('bpm',v); sa.setBpm(v); }}
                className="flex-1 accent-orange-500 h-1" />
              <span className="text-[10px] text-orange-400 w-8 text-right tabular-nums">{sliderVals['bpm']}</span>
            </>
          )}
        </div>}

        {/* AI Continuation */}
        <div className="space-y-1">
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => { setAiContinueEnabled(!aiContinueEnabled); setAiWarnDisabled(false); aiFailuresRef.current = 0; }}
              title={aiWarnDisabled ? '3 failures — click to re-enable' : aiContinueEnabled ? 'Disable AI continuation' : 'Enable AI lyric continuation'}
              className={`flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded font-medium transition-colors ${
                aiWarnDisabled      ? 'bg-amber-900/50 text-amber-400 border border-amber-700/60'
                : aiContinueEnabled ? 'bg-violet-600 text-white'
                : 'text-zinc-600 hover:text-zinc-400 hover:bg-zinc-800'}`}>
              {aiContinuePending ? <Loader2 size={9} className="animate-spin" /> : aiWarnDisabled ? '⚠' : '🤖'} AI ↻
            </button>
            {aiContinueEnabled && (
              <>
                <span className="text-[9px] text-zinc-600">every</span>
                {([1, 2, 4] as const).map(n => (
                  <button key={n} onClick={() => setAiContinueInterval(n)}
                    className={`text-[9px] w-5 h-4 rounded font-medium transition-colors ${
                      aiContinueInterval === n ? 'bg-violet-500 text-white' : 'text-zinc-600 hover:bg-zinc-800 hover:text-zinc-400'}`}>
                    {n}
                  </button>
                ))}
                <span className="text-[9px] text-zinc-600">slots</span>
              </>
            )}
            <HintButton onClick={() => setPresetModalOpen(true)}
              label={t('storm.live.aiPresets', 'AI presets & template')}
              info={t('storm.live.aiPresetsInfo', 'Manage the built-in and saved continuation presets, and edit the prompt template sent to the LLM.')}
              ariaLabel={t('storm.live.aiPresets', 'AI presets & template')}
              rootClassName="ml-auto"
              className={`text-[9px] px-1.5 py-0.5 rounded transition-colors flex items-center gap-0.5 ${ isCustomTemplate ? 'text-violet-400 hover:text-violet-300 hover:bg-zinc-800' : 'text-zinc-600 hover:text-zinc-300 hover:bg-zinc-800'}`}>
              ⚙{isCustomTemplate && <span className="text-[7px] text-violet-400 font-bold">CUSTOM</span>}
            </HintButton>
          </div>
          {aiContinueEnabled && (
            <input type="text" value={aiContinueDir} onChange={e => setAiContinueDir(e.target.value)}
              placeholder="direction hint: darker, bridge, outro…"
              className="w-full px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-[9px] text-zinc-400 placeholder:text-zinc-700 outline-none focus:border-violet-500/50 transition-colors" />
          )}
          {recentPresets.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {recentPresets.map(p => (
                <button key={p.id}
                  onClick={() => handlePresetFire(p, p.id.startsWith('bl-') || p.id.includes('lyric') ? 'lyric' : 'style')}
                  className="text-[8px] px-1.5 py-0.5 rounded bg-zinc-800 text-violet-400 hover:bg-violet-700 hover:text-white transition-colors font-medium">
                  {p.label}
                </button>
              ))}
              <button onClick={() => setPresetModalOpen(true)}
                className="text-[8px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-600 hover:text-zinc-400 transition-colors">more…</button>
            </div>
          )}
          {recentPresets.length === 0 && (
            <button onClick={() => setPresetModalOpen(true)} className="text-[8px] text-zinc-700 hover:text-zinc-500 transition-colors">
              ⚙ presets & template
            </button>
          )}
          {/* AI status feedback */}
          {aiStatus.type !== 'idle' && (
            <div className={`text-[9px] px-2 py-0.5 rounded leading-tight ${
              aiStatus.type === 'ok'      ? 'text-green-400 bg-green-900/20' :
              aiStatus.type === 'err'     ? 'text-red-400 bg-red-900/20' :
              aiStatus.type === 'warn'    ? 'text-amber-400 bg-amber-900/20' :
              aiStatus.type === 'pending' ? 'text-violet-400 bg-violet-900/20 animate-pulse' :
              'text-zinc-500'}`}>
              {aiStatus.msg}
            </div>
          )}
        </div>

        {/* Style / Lyrics tabs */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-1">
            <button onClick={() => setPromptTab('style')}
              className={`text-[10px] px-2 py-0.5 rounded-md font-medium transition-colors ${promptTab === 'style' ? 'bg-pink-600 text-white' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800'}`}>
              🎨 Style
            </button>
            <button onClick={() => onPersistStyleChange(!persistStyle)}
              title={persistStyle ? 'Style persists after Send' : 'Style clears after Send'}
              className={`text-[9px] px-1 py-0.5 rounded transition-colors ${persistStyle ? 'text-pink-400' : 'text-zinc-700 hover:text-zinc-500'}`}>
              {persistStyle ? '📌' : '·'}
            </button>
            <div className="w-px h-3 bg-zinc-700 mx-0.5" />
            <button onClick={() => setPromptTab('lyrics')}
              className={`text-[10px] px-2 py-0.5 rounded-md font-medium transition-colors ${promptTab === 'lyrics' ? 'bg-purple-600 text-white' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800'}`}>
              🎤 Lyrics
            </button>
            <button onClick={() => onPersistLyricsChange(!persistLyrics)}
              title={persistLyrics ? 'Lyrics persist after Send' : 'Lyrics clear after Send'}
              className={`text-[9px] px-1 py-0.5 rounded transition-colors ${persistLyrics ? 'text-purple-400' : 'text-zinc-700 hover:text-zinc-500'}`}>
              {persistLyrics ? '📌' : '·'}
            </button>
          </div>

          {/* Wildcard controls */}
          <div className="flex items-center gap-1.5 min-h-[18px]">
            {promptTab === 'style' && hasWildcards(streamPrompt) && (
              <HintButton onClick={() => handleExpandWildcards('style')}
                label={t('storm.live.expandStyle', 'Expand wildcards in style')}
                info={t('storm.live.expandStyleInfo', 'Resolves every {A|B} choice in the style once, using the current slot seed.')}
                ariaLabel={t('storm.live.expandStyle', 'Expand wildcards in style')}
                className="text-[9px] px-1.5 py-0.5 rounded bg-pink-900/40 text-pink-300 hover:bg-pink-700/60 hover:text-pink-100 transition-colors font-mono">
                {'{·}'} expand
              </HintButton>
            )}
            {promptTab === 'lyrics' && hasWildcards(streamLyrics) && (
              <HintButton onClick={() => handleExpandWildcards('lyrics')}
                label={t('storm.live.expandLyrics', 'Expand wildcards in lyrics')}
                info={t('storm.live.expandLyricsInfo', 'Resolves every {A|B} choice in the lyrics once, using the current slot seed.')}
                ariaLabel={t('storm.live.expandLyrics', 'Expand wildcards in lyrics')}
                className="text-[9px] px-1.5 py-0.5 rounded bg-purple-900/40 text-purple-300 hover:bg-purple-700/60 hover:text-purple-100 transition-colors font-mono">
                {'{·}'} expand
              </HintButton>
            )}
            <button onClick={() => setAutoExpand(v => !v)}
              title={autoExpand ? 'Auto-expand ON: wildcards resolved before each Send' : 'Auto-expand OFF'}
              className={`text-[9px] px-1.5 py-0.5 rounded transition-colors ml-auto ${
                autoExpand ? 'bg-amber-600/30 text-amber-300 hover:bg-amber-600/50' : 'text-zinc-700 hover:text-zinc-500 hover:bg-zinc-800'}`}>
              ⚄ auto
            </button>
          </div>

          {/* LoRA trigger word */}
          {promptTab === 'style' && (
            <div className="flex items-center gap-1.5">
              <span className="text-[9px] text-zinc-600 shrink-0">🔌 LoRA</span>
              <input type="text" value={loraTrigger} onChange={e => setLoraTrigger(e.target.value)}
                placeholder="trigger word (prepended to style)"
                className="flex-1 px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-[9px] text-zinc-400 placeholder:text-zinc-700 outline-none focus:border-pink-500/40 transition-colors" />
            </div>
          )}

          {/* Beat intro/outro for DJ crossfading */}
          {promptTab === 'style' && (
            <div className="flex items-center gap-1.5">
              <button onClick={() => setBeatIntroOutro(!beatIntroOutro)}
                className={`text-[9px] px-1.5 py-0.5 rounded font-medium transition-colors shrink-0 ${
                  beatIntroOutro ? 'bg-orange-600/40 text-orange-300' : 'text-zinc-600 hover:bg-zinc-800 hover:text-zinc-400'}`}>
                🥁 Beat I/O
              </button>
              {beatIntroOutro && (
                <>
                  <span className="text-[9px] text-zinc-600">bars:</span>
                  {([1, 2, 4, 8] as const).map(n => (
                    <button key={n} onClick={() => setIntroOutroBars(n)}
                      className={`text-[9px] w-5 h-4 rounded font-medium transition-colors ${
                        introOutroBars === n ? 'bg-orange-500 text-white' : 'text-zinc-600 hover:bg-zinc-800'}`}>
                      {n}
                    </button>
                  ))}
                </>
              )}
            </div>
          )}

          {stuckStyle && promptTab === 'style' && (
            <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-pink-500/10 border border-pink-500/20">
              <span className="text-[9px] text-pink-400 flex-1 truncate">📌 stuck: {stuckStyle.slice(0, 30)}{stuckStyle.length > 30 ? '…' : ''}</span>
              <button onClick={() => handleUnstick('style')} className="text-[9px] text-zinc-600 hover:text-red-400 transition-colors shrink-0">✕</button>
            </div>
          )}
          {stuckLyrics && promptTab === 'lyrics' && (
            <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-purple-500/10 border border-purple-500/20">
              <span className="text-[9px] text-purple-400 flex-1 truncate">📌 stuck: {stuckLyrics.split('\n')[0].slice(0, 30)}…</span>
              <button onClick={() => handleUnstick('lyrics')} className="text-[9px] text-zinc-600 hover:text-red-400 transition-colors shrink-0">✕</button>
            </div>
          )}

          {promptTab === 'style' ? (
            <textarea ref={styleTextareaRef} value={streamPrompt} onChange={e => onStreamPromptChange(e.target.value)}
              placeholder={t('storm.live.newStylePlaceholder', 'New style for next slot…')} rows={2}
              className="w-full px-2 py-1.5 rounded-md bg-zinc-800 border border-zinc-700 text-xs text-zinc-200 placeholder:text-zinc-600 outline-none resize-none focus:border-pink-500/50 transition-colors" />
          ) : (
            <textarea ref={lyricsTextareaRef} value={streamLyrics} onChange={e => onStreamLyricsChange(e.target.value)}
              placeholder={t('storm.live.newLyricsPlaceholder', 'Lyrics for next slot…')} rows={2}
              className="w-full px-2 py-1.5 rounded-md bg-zinc-800 border border-zinc-700 text-xs text-zinc-200 placeholder:text-zinc-600 outline-none resize-none focus:border-purple-500/50 transition-colors" />
          )}

          <div className="flex gap-1">
            <button onClick={handleSend}
              className={`flex-1 px-2.5 py-1.5 rounded-md text-white text-[10px] font-medium transition-all ${sendFlash === 'sent' ? 'bg-green-600 scale-[0.98]' : 'bg-pink-600 hover:bg-pink-500'}`}>
              {sendFlash === 'sent' ? `✓ slot ${received + 2}` : 'Send'}
            </button>
            {/* Stick button removed — pin (📌) auto-sticks on Send */}
          </div>
        </div>
      </div>

      {/* STOP */}
      {sa.isPlaying && (
        <button onClick={onStop}
          className="w-full py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white text-sm font-medium transition-colors">
          STOP
        </button>
      )}

      {/* AI Preset Modal */}
      <AiContinuePresetModal
        isOpen={presetModalOpen}
        onClose={() => setPresetModalOpen(false)}
        onPresetFire={handlePresetFire}
        onTemplateChange={setAiTemplate}
      />
    </div>
  );
};