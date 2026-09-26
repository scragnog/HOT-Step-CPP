// StormPage.tsx — Full-width STORM Streaming performance page
// MDMAchine / A&E Concepts 2026

import React, { useState, useEffect } from 'react';
import { Zap, Square, Play, Music } from 'lucide-react';
import { useGlobalParams } from '../../context/GlobalParamsContext';
import { usePersistedState } from '../../hooks/usePersistedState';
import { useStreamAudio } from '../../hooks/useStreamAudio';
import { StormLiveControls, type LyricsMode, type SlotMeta } from './StormLiveControls';
import { expandWildcards, hasWildcards } from '../../utils/wildcardUtils';
import { BackendCapabilityGate } from '../shared/BackendCapabilityGate';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import { ParamLabel } from '../shared/ParamLabel';
import type { GenerationParams } from '../../types';

// ── Camelot wheel ─────────────────────────────────────────────────────────────
const KEY_TO_CAMELOT: Record<string, string> = {
  'C':'8B','G':'9B','D':'10B','A':'11B','E':'12B','B':'1B',
  'F#':'2B','C#':'3B','G#':'4B','D#':'5B','A#':'6B','F':'7B',
  'Am':'8A','Em':'9A','Bm':'10A','F#m':'11A','C#m':'12A','G#m':'1A',
  'D#m':'2A','A#m':'3A','Fm':'4A','Cm':'5A','Gm':'6A','Dm':'7A',
};
const CAMELOT_HUE: Record<string, string> = {
  '1':'#e74c3c','2':'#e67e22','3':'#f1c40f','4':'#2ecc71',
  '5':'#1abc9c','6':'#3498db','7':'#9b59b6','8':'#e91e63',
  '9':'#ff5722','10':'#4caf50','11':'#00bcd4','12':'#673ab7',
};
function camelotColor(c: string) { return CAMELOT_HUE[c?.replace(/[AB]/,'')] || '#555'; }
function keyCompat(a: string, b: string): 'compat'|'adjacent'|'clash' {
  if (!a||!b) return 'clash';
  const an=parseInt(a), am=a.slice(-1), bn=parseInt(b), bm=b.slice(-1);
  if (an===bn&&am===bm) return 'compat';
  if (am===bm&&(Math.abs(an-bn)===1||Math.abs(an-bn)===11)) return 'compat';
  if (an===bn) return 'adjacent';
  if (am===bm&&(Math.abs(an-bn)===2||Math.abs(an-bn)===10)) return 'adjacent';
  return 'clash';
}

function fmtT(s: number) { const m=Math.floor(s/60); return `${m}:${String(Math.floor(s%60)).padStart(2,'0')}`; }
function fmtBuf(s: number) { return s>=3600?'∞':s>=60?`${Math.round(s/60)}m`:`${s}s`; }

type StormMode = 'sequential'|'continuous'|'dj'|'drift';
type BoxState = 'past'|'playing'|'rendered'|'rendering'|'queued';
function boxState(slot: number, playing: number, received: number): BoxState {
  if (slot<playing) return 'past';
  if (slot===playing) return 'playing';
  if (slot<=received) return 'rendered';
  if (slot===received+1) return 'rendering';
  return 'queued';
}
const BOX_CLS: Record<BoxState,string> = {
  past:'bg-zinc-950 border-zinc-800/40 opacity-25',
  playing:'bg-red-500/15 border-red-400 ring-1 ring-red-400/40',
  rendered:'bg-pink-950/40 border-pink-600/40',
  rendering:'bg-amber-950/40 border-amber-500/50',
  queued:'bg-zinc-800/50 border-zinc-600/60',
};
const BOX_LBL: Record<BoxState,string> = {
  past:'',playing:'PLAYING ▶',rendered:'RENDERED ✓',rendering:'RENDERING ⚡',queued:'QUEUED',
};
const BOX_CLR: Record<BoxState,string> = {
  past:'text-zinc-700',playing:'text-red-300 font-semibold',rendered:'text-pink-400',
  rendering:'text-amber-400',queued:'text-zinc-400',
};

const NEXT_PARAMS = [
  {label:'Guidance',min:0.5,max:15,step:0.1,key:'guidance_scale',fmt:(v:number)=>v.toFixed(1),color:'text-blue-400',
    info:'Slider for the next slot\'s CFG scale — how closely it follows the style description. Higher sticks closer to the caption; lower gives more freedom, which can sound looser.'},
  {label:'Steps',min:4,max:50,step:1,key:'inference_steps',fmt:(v:number)=>String(Math.round(v)),color:'text-yellow-400',
    info:'Slider for the next slot\'s inference step count. More steps can sound cleaner but render slower; fewer steps render faster but can sound rougher.'},
  {label:'Duration',min:10,max:300,step:5,key:'duration',fmt:(v:number)=>`${Math.round(v)}s`,color:'text-green-400',
    info:'Slider for the next slot\'s length, in seconds.'},
  {label:'BPM',min:60,max:200,step:1,key:'bpm',fmt:(v:number)=>String(Math.round(v)),color:'text-orange-400',
    info:'Sends a fixed BPM target for the next rendered slot.'},
];
type NKey = (typeof NEXT_PARAMS)[number]['key'];

// ── DeckPanel (DJ mode) ───────────────────────────────────────────────────────
interface DeckProps {
  id: 'deck-a'|'deck-b';
  label: string;
  otherCamelot: string;
  crossfadeGain: number;
  onRegister?: (sa: ReturnType<typeof useStreamAudio>) => void;
}

const DeckPanel: React.FC<DeckProps> = ({ id, label, otherCamelot, crossfadeGain, onRegister }) => {
  const gp = useGlobalParams();
  const sa = useStreamAudio(id);
  useEffect(() => { onRegister?.(sa); }, [sa.isPlaying, sa.detectedKey]); // re-register on play state / key detection
  const [caption, setCaption] = usePersistedState(`hs-storm-${id}-caption`, '');
  const [lyrics, setLyrics]   = usePersistedState(`hs-storm-${id}-lyrics`, '');
  const [bpm]                 = usePersistedState(`hs-storm-${id}-bpm`, 120);
  const [sliderVals, setSliderVals] = useState<Record<NKey,number>>({
    guidance_scale: gp.guidanceScale ?? 7,
    inference_steps: gp.inferenceSteps ?? 8,
    duration: 120, bpm: 120,
  });
  const [nextStyle, setNextStyle] = useState('');
  const [sendFlash, setSendFlash] = useState(false);

  // Apply crossfade gain whenever it changes
  useEffect(() => { sa.setCrossfadeGain(crossfadeGain); }, [crossfadeGain]);

  const camelot = KEY_TO_CAMELOT[sa.detectedKey || ''] || '';
  const compat  = keyCompat(camelot, otherCamelot);
  const compatEl = camelot && otherCamelot ? (
    <span className={`text-[9px] font-medium ${compat==='compat'?'text-green-400':compat==='adjacent'?'text-yellow-400':'text-red-400'}`}>
      {compat==='compat'?'✓':compat==='adjacent'?'~':'✗'}
    </span>
  ) : null;

  const startDeck = () => {
    sa.start({
      ...gp.getGlobalParams(),
      caption, lyrics: lyrics || '[Instrumental]',
      bpm, taskType: 'text2music',
    });
  };

  const handleSend = () => {
    if (nextStyle.trim()) sa.sendControl('prompt', nextStyle);
    setSendFlash(true);
    setTimeout(() => setSendFlash(false), 1200);
  };

  const received  = sa.currentSlot || 0;
  const playing   = sa.playingSlot || 0;
  const slotCount = Math.max(received + 3, 6);
  const allSlots  = Array.from({ length: slotCount }, (_, i) => i);

  return (
    <div className={`flex flex-col flex-1 min-w-0 rounded-xl border p-3 gap-2.5 overflow-y-auto hide-scrollbar ${sa.isPlaying ? 'border-red-500/40 bg-zinc-900/60' : 'border-zinc-700 bg-zinc-900/30'}`}>
      {/* Deck header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-zinc-300">{label}</span>
          {camelot && (
            <span className="flex items-center gap-1">
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded" style={{background: camelotColor(camelot)+'30', color: camelotColor(camelot), border: `1px solid ${camelotColor(camelot)}60`}}>
                {camelot}
              </span>
              {compatEl}
            </span>
          )}
          {sa.detectedBpm > 0 && <span className="text-[9px] text-orange-400 tabular-nums">♩{sa.detectedBpm}</span>}
        </div>
        {/* Start / Stop */}
        <button
          onClick={sa.isPlaying ? sa.stop : startDeck}
          className={`flex items-center gap-1 text-[10px] px-2.5 py-1 rounded-lg font-medium transition-all ${
            sa.isPlaying ? 'bg-red-600 hover:bg-red-500 text-white' : 'bg-green-600 hover:bg-green-500 text-white'}`}>
          {sa.isPlaying ? <><Square size={10}/> Stop</> : <><Play size={10}/> Start</>}
        </button>
      </div>

      {/* Caption */}
      <textarea value={caption} onChange={e => setCaption(e.target.value)} rows={2}
        placeholder="Style description…"
        className="w-full px-2 py-1.5 rounded-lg bg-zinc-800 border border-zinc-700 text-xs text-zinc-200 placeholder:text-zinc-600 outline-none resize-none focus:border-pink-500/40 transition-colors" />

      {/* Lyrics (collapsed by default) */}
      <textarea value={lyrics} onChange={e => setLyrics(e.target.value)} rows={2}
        placeholder="Lyrics (leave blank for instrumental)…"
        className="w-full px-2 py-1.5 rounded-lg bg-zinc-800 border border-zinc-700 text-xs text-zinc-200 placeholder:text-zinc-600 outline-none resize-none focus:border-purple-500/40 transition-colors" />

      {/* Slot timeline — always visible */}
      <div className="flex gap-0.5">
        {allSlots.map(slot => {
          const s = boxState(slot, playing, received);
          return (
            <div key={slot} className={`flex-1 rounded px-0.5 py-1 border text-center ${BOX_CLS[s]}`}>
              <div className="text-[8px] text-zinc-600 tabular-nums">{slot}</div>
              <div className={`text-[7px] leading-tight ${BOX_CLR[s]}`}>{BOX_LBL[s]}</div>
              {s==='rendering'&&<div className="flex justify-center mt-0.5"><span className="w-1 h-1 rounded-full bg-amber-400 animate-pulse"/></div>}
            </div>
          );
        })}
      </div>

      {/* Mini player */}
      {sa.isPlaying && (
        <div className="space-y-1.5">
          {/* Progress */}
          <div className="w-full h-0.5 rounded-full bg-zinc-800 overflow-hidden">
            <div className="h-full rounded-full bg-gradient-to-r from-red-600 to-pink-500"
              style={{width:`${sa.bufferedTime?Math.min(100,(sa.currentTime/sa.bufferedTime)*100):0}%`}} />
          </div>
          <div className="flex items-center justify-between text-[9px] text-zinc-500">
            <span>Slot {playing}/{received}</span>
            <span className="tabular-nums">{fmtT(sa.currentTime)} / {fmtT(sa.bufferedTime)}</span>
          </div>
          {/* Volume */}
          <div className="flex items-center gap-2">
            <ParamLabel label="Vol" className="text-[9px] text-zinc-600" rootClassName="w-6"
              info="Playback volume for this deck's monitor output. Does not change the render." />
            <input type="range" min={0} max={1} step={0.05} value={sa.volume}
              onChange={e => sa.setVolume(Number(e.target.value))}
              className="flex-1 accent-red-500 h-1" />
            <span className="text-[9px] text-zinc-400 w-8 text-right tabular-nums">{Math.round(sa.volume*100)}%</span>
          </div>
        </div>
      )}

      {/* Next-slot params (when playing) */}
      {sa.isPlaying && (
        <div className="space-y-1.5">
          {NEXT_PARAMS.map(p => (
            <div key={p.key} className="flex items-center gap-2">
              <ParamLabel label={p.label} info={p.info} className={`text-[9px] ${p.color}`} rootClassName="w-14" />
              <input type="range" min={p.min} max={p.max} step={p.step}
                value={sliderVals[p.key]}
                onChange={e => {
                  const v = Number(e.target.value);
                  setSliderVals(prev => ({...prev, [p.key]: v}));
                  sa.sendControl(p.key, v);
                }}
                className="flex-1 accent-pink-500 h-0.5" />
              <span className={`text-[9px] w-10 text-right tabular-nums ${p.color}`}>{p.fmt(sliderVals[p.key])}</span>
            </div>
          ))}
          {/* Next style */}
          <div className="flex gap-1">
            <input type="text" value={nextStyle} onChange={e => setNextStyle(e.target.value)}
              placeholder="Next style…"
              className="flex-1 px-2 py-0.5 rounded bg-zinc-800 border border-zinc-700 text-[9px] text-zinc-300 placeholder:text-zinc-600 outline-none focus:border-pink-500/40" />
            <button onClick={handleSend}
              className={`text-[9px] px-2 rounded font-medium transition-all ${sendFlash?'bg-green-600 text-white':'bg-pink-600 hover:bg-pink-500 text-white'}`}>
              {sendFlash?'✓':'Send'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

// ── StormPage ─────────────────────────────────────────────────────────────────
interface StormPageProps {
  onGenerate: (params: Partial<GenerationParams>) => void;
  activeJobCount: number;
}

export const StormPage: React.FC<StormPageProps> = ({ onGenerate, activeJobCount: _activeJobCount }) => {
  const gp = useGlobalParams();
  const [mode, setMode] = usePersistedState<StormMode>('hs-storm-page-mode', 'continuous');

  // Seq/Cont stream state
  const sa = useStreamAudio('storm-main');

  // Stop stream when leaving the page — prevents orphan GPU renders
  React.useEffect(() => {
    return () => { sa.stop(); };
  }, []);
  const [streamPrompt,   setStreamPrompt]   = React.useState('');
  const [streamLyricsNS, setStreamLyricsNS] = React.useState('');
  const [persistStyle,   setPersistStyle]   = React.useState(true);
  const [persistLyrics,  setPersistLyrics]  = React.useState(true);
  const [streamSeed,     setStreamSeed]     = usePersistedState('hs-storm-seed', Math.floor(Math.random()*2147483647));
  const [streamSeedLock, setStreamSeedLock] = usePersistedState('hs-storm-seed-lock', false);
  const [caption,       setCaption]       = usePersistedState('hs-storm-caption', '');
  const [lyrics,        setLyrics]        = usePersistedState('hs-storm-lyrics', '');
  const [negPrompt,     setNegPrompt]     = usePersistedState('hs-storm-neg', '');
  const [autoExpand,    setAutoExpand]    = usePersistedState('hs-storm-auto-expand', true);
  const [loraTrigger,   setLoraTrigger]   = usePersistedState('hs-storm-lora', '');
  const [beatIntro,     setBeatIntro]     = usePersistedState('hs-storm-beat-intro', false);
  const [introBars,     setIntroBars]     = usePersistedState('hs-storm-intro-bars', 2);
  const [instrumental,  setInstrumental]  = usePersistedState('hs-storm-instrumental', false);
  const [bpm]                             = usePersistedState('hs-storm-bpm', 120);
  const [duration]                        = usePersistedState('hs-storm-duration-v2', 120);
  const [lyricsMode,    setLyricsMode]    = React.useState<LyricsMode>('loop');
  const [lssStrength,         setLssStrength]         = usePersistedState('hs-storm-lss', 0.65);
  const [xfadeBeats,          setXfadeBeats]          = usePersistedState('hs-storm-xfade', 4);
  const [maxBuffer,           setMaxBufferState]       = usePersistedState('hs-storm-maxbuf', 900);
  const [bufMode,             setBufMode]             = React.useState<'time'|'slots'>('slots');
  const [bufSlots,            setBufSlots]            = React.useState(2);
  // Real STORM plugin params matching storm_sampler_core.lua
  const [stiffnessThreshold,  setStiffnessThreshold]  = usePersistedState('hs-storm-stiffness',  0.15);
  const [lookBackLambda,      setLookBackLambda]      = usePersistedState('hs-storm-lb-lambda',  0.15);
  const [lookBackSnrPower,    setLookBackSnrPower]    = usePersistedState('hs-storm-lb-snr',     1.5);
  const [rkOrder,             setRkOrder]             = usePersistedState('hs-storm-rk-order',   'auto');
  const [cacheDepth,          setCacheDepth]          = usePersistedState('hs-storm-cache-depth', 5);
  // C++ side params (top-level, not plugin_params)
  const [cacheRatio,          setCacheRatio]          = usePersistedState('hs-storm-cache-ratio', 0);
  const [cfgCutoffRatio,      setCfgCutoffRatio]      = usePersistedState('hs-storm-cfg-cutoff',  1.0);
  // Live solver / scheduler / guider — mirrors GlobalParams at start, overridable mid-stream
  const [liveInferMethod,  setLiveInferMethod]  = usePersistedState('hs-storm-infer-method', gp.inferMethod ?? '');
  const [liveScheduler,    setLiveScheduler]    = usePersistedState('hs-storm-scheduler', gp.scheduler ?? '');
  const [liveGuidanceMode, setLiveGuidanceMode] = usePersistedState('hs-storm-guidance-mode', gp.guidanceMode ?? '');
  // Plugin registry for dropdowns
  const [pluginRegistry, setPluginRegistry] = React.useState<{
    solvers:   { name: string; display: string }[];
    schedulers:{ name: string; display: string }[];
    guidance:  { name: string; display: string }[];
  }>({ solvers: [], schedulers: [], guidance: [] });
  React.useEffect(() => {
    fetch('/api/plugins').then(r => r.json()).then(data => {
      setPluginRegistry({
        solvers:    (data.solvers    ?? []).map((p: any) => ({ name: p.name, display: p.display ?? p.name })),
        schedulers: (data.schedulers ?? []).map((p: any) => ({ name: p.name, display: p.display ?? p.name })),
        guidance:   (data.guidance   ?? []).map((p: any) => ({ name: p.name, display: p.display ?? p.name })),
      });
    }).catch(() => {}); // silent fail — dropdowns just stay empty until engine is up
  }, []);
  // Live params from StormLiveControls (shared between seq + cont)
  const [liveParams, setLiveParams] = React.useState({ guidance_scale: gp.guidanceScale??7, inference_steps: gp.inferenceSteps??20, duration: 120, bpm: 120 });
  const [selectedSlot, setSelectedSlot] = React.useState<number | null>(null);
  const [slotMeta, setSlotMeta] = React.useState<Map<number, SlotMeta>>(new Map());
  const [extraSolverParams,    setExtraSolverParams]    = usePersistedState<Record<string, number>>('hs-storm-solver-extra', {});
  const [extraSchedulerParams, setExtraSchedulerParams] = usePersistedState<Record<string, number|string>>('hs-storm-scheduler-extra', {});
  const [extraGuiderParams,    setExtraGuiderParams]    = usePersistedState<Record<string, number|string>>('hs-storm-guider-extra', {});

  // DJ deck refs (for nudge + quantized start)
  const deckARef = React.useRef<ReturnType<typeof useStreamAudio>|null>(null);
  const deckBRef = React.useRef<ReturnType<typeof useStreamAudio>|null>(null);
  // DJ crossfader
  const [xfader, setXfader] = useState(0.5); // 0=all A, 1=all B
  const gainA = Math.cos(xfader * Math.PI / 2);
  const gainB = Math.sin(xfader * Math.PI / 2);

  // DJ deck camelot
  const [deckAKey, setDeckAKey] = useState('');
  const [deckBKey, setDeckBKey] = useState('');
  const camelotA = KEY_TO_CAMELOT[deckAKey] || '';
  const camelotB = KEY_TO_CAMELOT[deckBKey] || '';
  const djCompat = keyCompat(camelotA, camelotB);

  const handleStart = () => {
    const beatText = beatIntro ? `, with a clean ${introBars}-bar percussive intro and outro for DJ mixing` : '';
    const loraText = loraTrigger.trim() ? `${loraTrigger.trim()}, ` : '';
    const expandedCaption = autoExpand && hasWildcards(caption) ? expandWildcards(caption, streamSeed, 0) : caption;
    const fullCaption = `${loraText}${expandedCaption}${beatText}`;
    const params = {
      ...gp.getGlobalParams(),
      caption: fullCaption,
      lyrics: instrumental ? '[Instrumental]' : (autoExpand && hasWildcards(lyrics) ? expandWildcards(lyrics, streamSeed, 0) : lyrics),
      ...(negPrompt.trim() ? {negative_prompt: autoExpand && hasWildcards(negPrompt) ? expandWildcards(negPrompt.trim(), streamSeed, 0) : negPrompt.trim()} : {}),
      ...(lssStrength > 0 ? { lssStrength } : {}),
      seed: streamSeed,
      randomSeed: false,
      bpm,
      duration,
      taskType: 'text2music',
      // Bake current live values into baseParams — ctrl overrides will layer on top
      ...(liveInferMethod  ? { inferMethod:  liveInferMethod  } : {}),
      ...(liveScheduler    ? { scheduler:    liveScheduler    } : {}),
      ...(liveGuidanceMode ? { guidanceMode: liveGuidanceMode } : {}),
      pluginParams: {
        ...gp.pluginParams,
        ...extraSolverParams,
        ...extraSchedulerParams,
        ...extraGuiderParams,
        stiffness_threshold: stiffnessThreshold,
        look_back_lambda:    lookBackLambda,
        look_back_snr_power: lookBackSnrPower,
        rk_order:            rkOrder,
        cache_depth:         cacheDepth,
      },
      ...(cacheRatio > 0       ? { cacheRatio }      : {}),
      ...(cfgCutoffRatio < 1.0 ? { cfgCutoffRatio }  : {}),
      coResident: (() => {
        try { return JSON.parse(localStorage.getItem('ace-settings') || '{}').coResident === true; }
        catch { return false; }
      })(),
    };
    if (mode === 'sequential') {
      onGenerate({ ...params, guidanceScale: liveParams.guidance_scale, inferenceSteps: liveParams.inference_steps, duration: liveParams.duration, bpm: liveParams.bpm });
    } else if (mode === 'continuous') {
      sa.setXfadeBeats(xfadeBeats); // sync xfadeBeatsRef before stream starts
      sa.start(params);
    }
  };

  const handleStop = async () => { await sa.stop(); };

  // Solver-aware plugin params — storm included so all solvers use one render path
  const SOLVER_EXTRA_PARAMS: Record<string, Array<{key:string,label:string,min:number,max:number,step:number,default:number,fmt:(v:number)=>string,color:string,title?:string,meta?:string}>> = {
    storm: [
      {key:'stiffness_threshold', label:'Detail Sens',  min:0.05, max:0.50, step:0.01, default:0.15, fmt:v=>v.toFixed(2),          color:'text-pink-400',   title:'Stiffness threshold — lower = more detail on transients'},
      {key:'look_back_lambda',    label:'Coherence',    min:0,    max:1,    step:0.01, default:0.15, fmt:v=>v.toFixed(2),          color:'text-violet-400', title:'Look-back smoothing — 0=off, higher=smoother output'},
      {key:'look_back_snr_power', label:'Early Focus',  min:0.5,  max:3,    step:0.1,  default:1.5,  fmt:v=>v.toFixed(1),          color:'text-cyan-400',   title:'Concentrates smoothing on early structure steps'},
      {key:'cache_depth',         label:'Cache Depth',  min:2,    max:10,   step:1,    default:5,    fmt:v=>String(Math.round(v)), color:'text-blue-400',   title:'Steps of solver history — more=smoother blending'},
    ],
    pingpong: [
      {key:'look_back_lambda',    label:'Coherence',  min:0,   max:1,   step:0.01, default:0.15, fmt:v=>v.toFixed(2), color:'text-violet-400',
        title:'Blends each step with recent steps to smooth the trajectory and reduce harmonic hum, fading out at low sigma. 0 turns it off; the plugin’s own tuned values run around 0.55 for shorter runs (~25 steps) and 0.35 for longer ones (~35 steps).', meta:'default 0.15 · range 0–1'},
      {key:'look_back_snr_power', label:'SNR Power',  min:0.5, max:3,   step:0.1,  default:1.5,  fmt:v=>v.toFixed(1), color:'text-cyan-400',
        title:'Falloff exponent for how fast that look-back smoothing fades as sigma drops. Higher values fade it out sooner, concentrating the smoothing on the early, high-sigma steps.', meta:'default 1.5 · range 0.5–3'},
    ],
    hap: [
      {key:'kinetic_energy',   label:'Kinetic E',  min:0, max:5,  step:0.05, default:1.0, fmt:v=>v.toFixed(2), color:'text-cyan-400',
        title:'Sent to the hap solver as kinetic_energy when that solver is selected. No hap solver plugin ships in this build, so it currently has no effect.', meta:'default 1.0 · range 0–5'},
      {key:'damping_friction', label:'Friction',   min:0, max:8,  step:0.1,  default:0.5, fmt:v=>v.toFixed(1), color:'text-blue-400',
        title:'Sent to the hap solver as damping_friction when that solver is selected. No hap solver plugin ships in this build, so it currently has no effect.', meta:'default 0.5 · range 0–8'},
    ],
    otto: [
      {key:'combustion_point',  label:'Combustion', min:0.1, max:0.9, step:0.01, default:0.4,  fmt:v=>v.toFixed(2), color:'text-orange-400',
        title:'Sent to the otto solver as combustion_point when that solver is selected. No otto solver plugin ships in this build, so it currently has no effect.', meta:'default 0.4 · range 0.1–0.9'},
      {key:'compression_ratio', label:'Compress',   min:1,   max:20,  step:0.5,  default:10.0, fmt:v=>v.toFixed(1), color:'text-red-400',
        title:'Sent to the otto solver as compression_ratio when that solver is selected. No otto solver plugin ships in this build, so it currently has no effect.', meta:'default 10.0 · range 1–20'},
      {key:'adiabatic_index',   label:'Adiabatic',  min:0.5, max:5,   step:0.1,  default:2.0,  fmt:v=>v.toFixed(1), color:'text-amber-400',
        title:'Sent to the otto solver as adiabatic_index when that solver is selected. No otto solver plugin ships in this build, so it currently has no effect.', meta:'default 2.0 · range 0.5–5'},
    ],
    brayton: [
      {key:'pressure_ratio', label:'Pressure', min:1, max:30, step:0.5, default:10.0, fmt:v=>v.toFixed(1), color:'text-sky-400',
        title:'Sent to the brayton solver as pressure_ratio when that solver is selected. No brayton solver plugin ships in this build, so it currently has no effect.', meta:'default 10.0 · range 1–30'},
    ],
    causal: [
      {key:'lina_shift', label:'LINA Shift', min:0.1, max:3.0, step:0.05, default:1.2, fmt:v=>v.toFixed(2), color:'text-teal-400',
        title:'Time-axis warp on the step schedule — below 1 front-loads steps toward high sigma, above 1 back-loads them toward low sigma, 1.0 is no warp. Same lina_shift knob the causal scheduler uses; no causal solver plugin ships in this build, so it currently has no effect here.', meta:'default 1.2 · range 0.1–3'},
    ],
    omni: [
      {key:'relational_weight', label:'Relational W', min:0, max:1, step:0.05, default:0.5, fmt:v=>v.toFixed(2), color:'text-pink-400',
        title:'Base weight of the relational blend term at high sigma, fading toward 0 as sigma drops. 0 keeps the solver at plain Euler throughout; 0.5 balances it during the structure phase.', meta:'default 0.5 · range 0–1'},
      {key:'sigma_power',       label:'Sigma Power',  min:0.25, max:4, step:0.25, default:1.0, fmt:v=>v.toFixed(2), color:'text-violet-400',
        title:'How fast the relational weight fades with sigma. 1.0 fades linearly, 2.0 fades faster (quadratic) so the effect concentrates on early steps, below 1.0 fades more slowly.', meta:'default 1.0 · range 0.25–4'},
    ],
  };
  // Match selected solver dropdown to a param key. Blank/unrecognised = show storm params.
  const activeSolverKey = liveInferMethod
    ? (Object.keys(SOLVER_EXTRA_PARAMS).find(k => liveInferMethod.toLowerCase().includes(k)) ?? null)
    : 'storm';
  const activeSolverParams = activeSolverKey ? (SOLVER_EXTRA_PARAMS[activeSolverKey] ?? []) : [];

  type ParamOpt = {value: string, label: string};
  type ParamDef = {
    type?: 'slider'|'select';
    key: string; label: string; color: string; title?: string; meta?: string;
    // slider fields
    min?: number; max?: number; step?: number; default?: number|string; fmt?: (v:number)=>string;
    // select fields
    options?: ParamOpt[];
  };
  const SCHEDULER_PARAMS: Record<string, ParamDef[]> = {
    hap: [
      {key:'kinetic_energy',   label:'Kinetic E',  min:0, max:5,  step:0.05, default:1.5, fmt:v=>v.toFixed(2), color:'text-cyan-400',   title:'Hamiltonian kinetic energy'},
      {key:'damping_friction', label:'Friction',   min:0, max:8,  step:0.1,  default:3.0, fmt:v=>v.toFixed(1), color:'text-blue-400',   title:'Damping friction coefficient'},
    ],
    ht_scheduler: [
      {key:'kinetic_energy',   label:'Kinetic E',  min:0,   max:2,  step:0.05, default:0.3, fmt:v=>v.toFixed(2), color:'text-cyan-400',
        title:'HAP leading-edge sharpness in the timestep schedule. 0 is uniform spacing; raising it front-loads more steps at high sigma.', meta:'default 0.3 · range 0–2'},
      {key:'damping_friction', label:'Friction',   min:0,   max:5,  step:0.1,  default:2.2, fmt:v=>v.toFixed(1), color:'text-blue-400',
        title:'HAP tail compression in the timestep schedule. Higher values cluster steps toward the front and away from the end of the run.', meta:'default 2.2 · range 0–5'},
      {key:'critical_temp',    label:'Crit Temp',  min:0.1, max:1,  step:0.05, default:0.6, fmt:v=>v.toFixed(2), color:'text-orange-400',
        title:'Sigma fraction where the TPT phase transition centers — steps cluster around this point in the schedule.', meta:'default 0.6 · range 0.1–1'},
      {key:'phase_intensity',  label:'Phase Int',  min:0,   max:3,  step:0.1,  default:1.0, fmt:v=>v.toFixed(1), color:'text-violet-400',
        title:'TPT clustering strength added on top of the HAP curve. 0 is pure HAP (off), 1 is moderate, higher values cluster steps more strongly around Crit Temp.', meta:'default 1.0 · range 0–3'},
    ],
    otto: [
      {key:'combustion_point',  label:'Combustion', min:0.1, max:0.9, step:0.01, default:0.4,  fmt:v=>v.toFixed(2), color:'text-orange-400',
        title:'Sent to the otto scheduler as combustion_point when that scheduler is selected. No otto scheduler plugin ships in this build, so it currently has no effect.', meta:'default 0.4 · range 0.1–0.9'},
      {key:'compression_ratio', label:'Compress',   min:1,   max:20,  step:0.5,  default:10.0, fmt:v=>v.toFixed(1), color:'text-red-400',
        title:'Sent to the otto scheduler as compression_ratio when that scheduler is selected. No otto scheduler plugin ships in this build, so it currently has no effect.', meta:'default 10.0 · range 1–20'},
      {key:'adiabatic_index',   label:'Adiabatic',  min:0.5, max:5,   step:0.1,  default:2.0,  fmt:v=>v.toFixed(1), color:'text-amber-400',
        title:'Sent to the otto scheduler as adiabatic_index when that scheduler is selected. No otto scheduler plugin ships in this build, so it currently has no effect.', meta:'default 2.0 · range 0.5–5'},
    ],
    brayton: [
      {key:'combustion_start',    label:'C Start', min:0.01, max:0.9,  step:0.01, default:0.2,  fmt:v=>v.toFixed(2), color:'text-orange-400',
        title:'Sent to the brayton scheduler as combustion_start when that scheduler is selected. No brayton scheduler plugin ships in this build, so it currently has no effect.', meta:'default 0.2 · range 0.01–0.9'},
      {key:'combustion_duration', label:'C Dur',   min:0.05, max:0.99, step:0.01, default:0.5,  fmt:v=>v.toFixed(2), color:'text-amber-400',
        title:'Sent to the brayton scheduler as combustion_duration when that scheduler is selected. No brayton scheduler plugin ships in this build, so it currently has no effect.', meta:'default 0.5 · range 0.05–0.99'},
      {key:'pressure_ratio',      label:'Pressure', min:1,    max:30,  step:0.5,  default:8.0,  fmt:v=>v.toFixed(1), color:'text-sky-400',
        title:'Sent to the brayton scheduler as pressure_ratio when that scheduler is selected. No brayton scheduler plugin ships in this build, so it currently has no effect.', meta:'default 8.0 · range 1–30'},
      {key:'spool_speed',         label:'Spool',    min:1,    max:50,  step:1,    default:20.0, fmt:v=>v.toFixed(0), color:'text-teal-400'},
    ],
    causal: [
      {type:'select', key:'mode', label:'Mode', default:'polynomial', color:'text-teal-400',
        title:'Base timestep curve before the LINA warp is applied — Karras, linear, polynomial, beta, AYS and others. Changes the shape of the step schedule the other causal knobs (Rho, Power, Blend, …) warp on top of.', meta:'default polynomial', options:[
        {value:'karras',             label:'Karras (rho)'},
        {value:'simple',             label:'Simple (Smoothstep)'},
        {value:'linear',             label:'Linear'},
        {value:'exponential',        label:'Exponential'},
        {value:'polynomial',         label:'Polynomial'},
        {value:'beta',               label:'Beta'},
        {value:'ays',                label:'AYS'},
        {value:'bong',               label:'Bong (Tangent)'},
        {value:'linear_quadratic',   label:'Linear-Quadratic'},
        {value:'ddim_uniform',       label:'DDIM Uniform'},
        {value:'sgm_uniform',        label:'SGM Uniform'},
        {value:'blended',            label:'Blended (Karras+Lin)'},
        {value:'variance_preserving',label:'Variance Preserving'},
        {value:'kl_optimal',         label:'KL Optimal'},
      ]},
      {type:'slider', key:'lina_shift',   label:'LINA Shift', min:0.1, max:3.0, step:0.05, default:1.0, fmt:v=>v.toFixed(2), color:'text-teal-400'},
      {type:'slider', key:'rho',          label:'Rho',        min:1,   max:15,  step:0.5,  default:7.0, fmt:v=>v.toFixed(1), color:'text-cyan-400', title:'Karras rho — higher=more steps at low sigma'},
      {type:'slider', key:'blend_factor', label:'Blend',      min:0,   max:1,   step:0.05, default:0.5, fmt:v=>v.toFixed(2), color:'text-violet-400',
        title:'Blend between the Karras curve (0) and the Linear curve (1). Only active when Mode is set to Blended.', meta:'default 0.5 · range 0–1'},
    ],
    hyper: [
      {key:'warp_factor', label:'Warp', min:0.1, max:3.0, step:0.05, default:1.2, fmt:v=>v.toFixed(2), color:'text-pink-400',
        title:'Sent to the hyper scheduler as warp_factor. No hyper scheduler plugin ships in this build, so it currently has no effect.', meta:'default 1.2 · range 0.1–3'},
    ],
    pam: [
      {key:'focal_sigma', label:'Focal σ',   min:0.01, max:0.99, step:0.01, default:0.5, fmt:v=>v.toFixed(2), color:'text-pink-400',
        title:'Sent to the pam scheduler as focal_sigma. No pam scheduler plugin ships in this build, so it currently has no effect.', meta:'default 0.5 · range 0.01–0.99'},
      {key:'bandwidth',   label:'Bandwidth', min:0.05, max:2,    step:0.05, default:0.5, fmt:v=>v.toFixed(2), color:'text-violet-400',
        title:'Sent to the pam scheduler as bandwidth. No pam scheduler plugin ships in this build, so it currently has no effect.', meta:'default 0.5 · range 0.05–2'},
      {key:'intensity',   label:'Intensity', min:0,    max:10,   step:0.1,  default:3.0, fmt:v=>v.toFixed(1), color:'text-amber-400',
        title:'Sent to the pam scheduler as intensity. No pam scheduler plugin ships in this build, so it currently has no effect.', meta:'default 3.0 · range 0–10'},
    ],
    noise_decay: [
      {type:'select', key:'algorithm', label:'Algorithm', default:'polynomial', color:'text-amber-400',
        title:'Sent to the noise_decay scheduler as algorithm. No noise_decay scheduler plugin ships in this build, so it currently has no effect.', meta:'default polynomial', options:[
        {value:'polynomial',  label:'Polynomial'},
        {value:'sigmoidal',   label:'Sigmoidal'},
        {value:'piecewise',   label:'Piecewise'},
        {value:'fourier',     label:'Fourier'},
        {value:'exponential', label:'Exponential'},
        {value:'gaussian',    label:'Gaussian'},
      ]},
      {type:'slider', key:'decay_exponent', label:'Exponent', min:0.1, max:8,  step:0.1, default:2.0, fmt:v=>v.toFixed(1), color:'text-amber-400', title:'Poly=power, Sigmoid=steepness, Fourier=freq, Exp=fall rate'},
      {type:'slider', key:'smooth_window',  label:'Smooth',   min:1,   max:9,  step:1,   default:2,   fmt:v=>String(Math.round(v)), color:'text-zinc-400', title:'Moving average window — 1=off'},
      {type:'slider', key:'shift',          label:'Shift',    min:0.5, max:8,  step:0.1, default:1.0, fmt:v=>v.toFixed(1), color:'text-sky-400', title:'HOT-Step shift warp applied after curve'},
    ],
  };

  const GUIDER_PARAMS: Record<string, ParamDef[]> = {
    apg: [
      {key:'eta',             label:'Eta',       min:0,  max:1,   step:0.01, default:0.0,  fmt:v=>v.toFixed(2), color:'text-pink-400',   title:'Momentum blend — 0=off'},
      {key:'momentum',        label:'Momentum',  min:0,  max:1,   step:0.01, default:0.5,  fmt:v=>v.toFixed(2), color:'text-violet-400', title:'EMA momentum for projection'},
      {key:'norm_threshold',  label:'Norm Thr',  min:0,  max:2,   step:0.05, default:0.0,  fmt:v=>v.toFixed(2), color:'text-cyan-400',   title:'Projection norm threshold — 0=always project'},
      {key:'warmup_steps',    label:'Warmup',    min:0,  max:10,  step:1,    default:0,    fmt:v=>String(Math.round(v)), color:'text-blue-400', title:'Steps before guidance activates'},
    ],
    adg: [
      {key:'max_angle_deg', label:'Max Angle', min:5,  max:90,  step:1,    default:60.0, fmt:v=>v.toFixed(0), color:'text-orange-400',
        title:'Sent to the adg guider as max_angle_deg. No adg guidance plugin ships in this build, so it currently has no effect.', meta:'default 60 · range 5–90'},
      {key:'apg_blend',     label:'APG Blend', min:0,  max:1,   step:0.05, default:0.5,  fmt:v=>v.toFixed(2), color:'text-pink-400',
        title:'Sent to the adg guider as apg_blend. No adg guidance plugin ships in this build, so it currently has no effect.', meta:'default 0.5 · range 0–1'},
      {key:'warmup_steps',  label:'Warmup',    min:0,  max:10,  step:1,    default:2,    fmt:v=>String(Math.round(v)), color:'text-blue-400'},
    ],
    pmg: [
      {key:'strength',    label:'Strength',    min:0, max:3, step:0.05, default:1.0, fmt:v=>v.toFixed(2), color:'text-pink-400',
        title:'Sent to the pmg guider as strength. No pmg guidance plugin ships in this build, so it currently has no effect.', meta:'default 1.0 · range 0–3'},
      {key:'glide_power', label:'Glide Power', min:0, max:3, step:0.1,  default:1.0, fmt:v=>v.toFixed(1), color:'text-violet-400',
        title:'Sent to the pmg guider as glide_power. No pmg guidance plugin ships in this build, so it currently has no effect.', meta:'default 1.0 · range 0–3'},
    ],
    spectral: [
      {key:'entropy_floor',       label:'Ent Floor',  min:0,  max:2,  step:0.05, default:0.5, fmt:v=>v.toFixed(2), color:'text-cyan-400',
        title:'Sent to the spectral guider as entropy_floor. No spectral guidance plugin ships in this build, so it currently has no effect.', meta:'default 0.5 · range 0–2'},
      {key:'entropy_sensitivity', label:'Ent Sens',   min:0,  max:2,  step:0.05, default:0.8, fmt:v=>v.toFixed(2), color:'text-teal-400',
        title:'Sent to the spectral guider as entropy_sensitivity. No spectral guidance plugin ships in this build, so it currently has no effect.', meta:'default 0.8 · range 0–2'},
      {key:'aos_strength',        label:'AOS Str',    min:0,  max:1,  step:0.05, default:0.5, fmt:v=>v.toFixed(2), color:'text-violet-400',
        title:'Sent to the spectral guider as aos_strength. No spectral guidance plugin ships in this build, so it currently has no effect.', meta:'default 0.5 · range 0–1'},
      {key:'spectral_threshold',  label:'Spec Thr',   min:0,  max:1,  step:0.05, default:0.3, fmt:v=>v.toFixed(2), color:'text-blue-400',
        title:'Sent to the spectral guider as spectral_threshold. No spectral guidance plugin ships in this build, so it currently has no effect.', meta:'default 0.3 · range 0–1'},
    ],
    storm_guidance: [
      {key:'cfg_knee',        label:'CFG Knee',   min:0,    max:1,  step:0.05, default:0.75, fmt:v=>v.toFixed(2), color:'text-pink-400',
        title:'Progress fraction of the schedule where guidance starts to roll off toward the floor. Guidance stays flat at full scale before this point; raising the knee delays the rolloff to later steps.', meta:'default 0.75 · range 0–1'},
      {key:'cfg_floor',       label:'CFG Floor',  min:0,    max:1,  step:0.05, default:0.70, fmt:v=>v.toFixed(2), color:'text-violet-400',
        title:'Minimum guidance scale during the rolloff, as a fraction of the set guidance scale. Raise it if detail collapses late in the schedule; 0 lets guidance fall to nothing.', meta:'default 0.70 · range 0–1'},
      {key:'cfg_tail_power',  label:'Tail Pow',   min:0.5,  max:5,  step:0.1,  default:2.0,  fmt:v=>v.toFixed(1), color:'text-cyan-400',
        title:'Shape of the guidance rolloff curve. 1.0 is a linear fade, 2.0 a smooth quadratic fade, higher values fall off more sharply near the end.', meta:'default 2.0 · range 0.5–5'},
      {key:'nag_clamp_intensity', label:'NAG Clamp', min:0, max:1,  step:0.05, default:0.20, fmt:v=>v.toFixed(2), color:'text-amber-400',
        title:'How strongly spiking latent elements get pulled toward the mean each step, suppressing attention/LoRA resonance buildup. 0 disables it; the plugin’s own recommended range for audio LoRA is 0.15–0.25.', meta:'default 0.20 · range 0–1'},
    ],
    hyperguider: [
      {key:'eta',            label:'Eta',         min:0,    max:1,   step:0.01, default:0.0,  fmt:v=>v.toFixed(2), color:'text-pink-400'},
      {key:'gate_strength',  label:'Gate Str',    min:0,    max:1,   step:0.05, default:0.5,  fmt:v=>v.toFixed(2), color:'text-violet-400'},
      {key:'tv_floor',       label:'TV Floor',    min:0,    max:0.5, step:0.01, default:0.01, fmt:v=>v.toFixed(2), color:'text-cyan-400'},
      {key:'contrast_floor', label:'Contrast',    min:0,    max:0.5, step:0.01, default:0.1,  fmt:v=>v.toFixed(2), color:'text-teal-400'},
    ],
    weyl: [
      {key:'sliding_current',    label:'Sliding',    min:0, max:1,  step:0.05, default:0.3, fmt:v=>v.toFixed(2), color:'text-pink-400',
        title:'Sent to the weyl guider as sliding_current. No weyl guidance plugin ships in this build, so it currently has no effect.', meta:'default 0.3 · range 0–1'},
      {key:'dissipation_factor', label:'Dissipation',min:0, max:1,  step:0.05, default:0.3, fmt:v=>v.toFixed(2), color:'text-violet-400',
        title:'Sent to the weyl guider as dissipation_factor. No weyl guidance plugin ships in this build, so it currently has no effect.', meta:'default 0.3 · range 0–1'},
      {key:'weyl_chirality',     label:'Chirality',  min:-1,max:1,  step:0.1,  default:0.0, fmt:v=>v.toFixed(1), color:'text-cyan-400',
        title:'Sent to the weyl guider as weyl_chirality. No weyl guidance plugin ships in this build, so it currently has no effect.', meta:'default 0.0 · range −1–1'},
    ],
  };

  const activeSchedulerKey = liveScheduler
    ? (Object.keys(SCHEDULER_PARAMS).find(k => liveScheduler.toLowerCase().includes(k)) ?? null)
    : null;
  const activeSchedulerParams = activeSchedulerKey ? (SCHEDULER_PARAMS[activeSchedulerKey] ?? []) : [];

  const activeGuiderKey = liveGuidanceMode
    ? (Object.keys(GUIDER_PARAMS).find(k => liveGuidanceMode.toLowerCase().includes(k)) ?? null)
    : null;
  const activeGuiderParams = activeGuiderKey ? (GUIDER_PARAMS[activeGuiderKey] ?? []) : [];

  const modeBtn = (m: StormMode, label: string, emoji: string) => (
    <button onClick={() => setMode(m)}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
        mode === m ? 'bg-red-600 text-white' : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700'}`}>
      <span>{emoji}</span>{label}
    </button>
  );

  return (
    <BackendCapabilityGate feature="streaming">
    <div className="h-full flex flex-col bg-zinc-950 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-zinc-800 shrink-0">
        <div className="flex items-center gap-3">
          <Zap size={18} className="text-red-400" />
          <span className="text-sm font-bold text-white tracking-wide">STORM</span>
        </div>
        {/* Mode tabs */}
        <div className="flex gap-1">
          {modeBtn('sequential',  'Sequential',  '📋')}
          {modeBtn('continuous',  'Continuous',  '🔄')}
          {modeBtn('dj',          'DJ',          '🎧')}
          {modeBtn('drift',       'Drift',       '🌊')}
        </div>
        {/* Global start/stop for seq/cont */}
        {/* Always reserve right-side space so tabs stay centered in all modes */}
        <div className="w-[120px] flex justify-end">
          {(mode==='sequential'||mode==='continuous') && (
            <button
              onClick={sa.isPlaying ? handleStop : handleStart}
              disabled={!caption.trim() && !lyrics.trim() && !instrumental}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all disabled:opacity-40 ${
                sa.isPlaying ? 'bg-red-600 hover:bg-red-500 text-white' : 'bg-gradient-to-r from-pink-600 to-purple-600 hover:from-pink-500 hover:to-purple-500 text-white'}`}>
              {sa.isPlaying ? <><Square size={14}/> Stop</> : mode === 'sequential' ? <><Zap size={14}/> Generate</> : <><Zap size={14}/> Start</>}
            </button>
          )}
        </div>
      </div>

      {/* ── Sequential / Continuous ── */}
      {(mode==='sequential'||mode==='continuous') && (() => {
        const received  = sa.currentSlot  || 0;
        const playing   = sa.playingSlot  || 0;
        const slotCount = Math.max(received + 3, 7);
        const allSlots  = Array.from({ length: slotCount }, (_, i) => i);
        return (
          <>
            {/* ── Timeline strip ──────────────────────────────────── */}
            <div className="shrink-0 border-b border-zinc-800/60 bg-zinc-950/80">
              <div className="flex gap-1.5 px-4 py-2">
                {allSlots.map(slot => {
                  const s = boxState(slot, playing, received);
                  const isSelected = selectedSlot === slot;
                  return (
                    <button
                      key={slot}
                      onClick={() => {
                        setSelectedSlot(isSelected ? null : slot);
                        // TODO: implement seek_slot server-side
                        if (!isSelected && sa.isPlaying) sa.sendControl('seek_slot' as any, slot);
                      }}
                      className={`flex-1 rounded-md px-1 py-1.5 border transition-all text-center focus:outline-none ${BOX_CLS[s]} ${isSelected ? 'ring-2 ring-white/30 scale-[1.04]' : 'hover:brightness-125'}`}>
                      <div className="text-[9px] text-zinc-500 tabular-nums">{slot}</div>
                      <div className={`text-[8px] leading-tight font-medium ${BOX_CLR[s]}`}>{BOX_LBL[s]}</div>
                      {s==='rendering' && <div className="flex justify-center mt-0.5"><span className="w-1 h-1 rounded-full bg-amber-400 animate-pulse"/></div>}
                    </button>
                  );
                })}
              </div>
              {/* Slot metadata panel — expands on click */}
              {selectedSlot !== null && (() => {
                const meta = slotMeta.get(selectedSlot);
                const slotState = boxState(selectedSlot, playing, received);
                return (
                  <div className="flex flex-wrap items-start gap-x-4 gap-y-1 px-4 py-2 border-t border-zinc-800/60 bg-zinc-900/60 text-[11px]">
                    <span className="text-zinc-500 shrink-0 font-medium">Slot {selectedSlot}</span>
                    <span className={`font-medium shrink-0 ${BOX_CLR[slotState]}`}>{BOX_LBL[slotState] || 'waiting'}</span>
                    {meta ? (<>
                      {meta.detectedBpm  && <span className="text-orange-400 tabular-nums">♩ {Math.round(meta.detectedBpm)} BPM</span>}
                      {meta.detectedKey  && <span className="text-cyan-400">{meta.detectedKey} · {KEY_TO_CAMELOT[meta.detectedKey] || '—'}</span>}
                      {meta.seed         !== undefined && <span className="text-zinc-500 tabular-nums font-mono text-[10px]">🎲 {meta.seed}</span>}
                      {meta.guidance     !== undefined && <span className="text-blue-400">cfg {meta.guidance.toFixed(1)}</span>}
                      {meta.steps        !== undefined && <span className="text-yellow-400">steps {meta.steps}</span>}
                      {meta.duration     !== undefined && <span className="text-green-400">{Math.round(meta.duration)}s</span>}
                      {meta.solver       && <span className="text-cyan-400 truncate max-w-[140px]" title={meta.solver}>⚙ {meta.solver}</span>}
                      {meta.scheduler    && <span className="text-teal-400 truncate max-w-[140px]" title={meta.scheduler}>📅 {meta.scheduler}</span>}
                      {meta.guidanceMode && <span className="text-indigo-400 truncate max-w-[140px]" title={meta.guidanceMode}>🎯 {meta.guidanceMode}</span>}
                      {meta.style        && <span className="text-pink-400 truncate max-w-[200px]" title={meta.style}>🎨 {meta.style.slice(0, 60)}{meta.style.length > 60 ? '…' : ''}</span>}
                      {meta.lyrics       && <span className="text-purple-400 truncate max-w-[200px]" title={meta.lyrics}>🎤 {meta.lyrics.split('\n')[0].slice(0, 50)}</span>}
                    </>) : (
                      <span className="text-zinc-600 text-[10px]">no metadata yet — plays first</span>
                    )}
                    <button onClick={() => setSelectedSlot(null)} className="text-zinc-600 hover:text-zinc-400 ml-auto shrink-0">✕</button>
                  </div>
                );
              })()}
            </div>

            {/* ── Three-column body ───────────────────────────────── */}
            <div className="flex-1 flex overflow-hidden">

              {/* Left — Compose */}
              <div className="w-[380px] shrink-0 overflow-y-auto hide-scrollbar p-4 space-y-3 border-r border-zinc-800/60">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <ParamLabel label="Style" className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
                      info="The style caption for the stream. Supports {A|B|C} wildcard syntax, expandable in place with the wildcard button. Combined with the LoRA trigger word and beat intro/outro text before sending." />
                    <div className="flex items-center gap-1">
                      {hasWildcards(caption) && (
                        <button onClick={()=>setCaption(expandWildcards(caption, streamSeed, 0))}
                          className="text-[9px] px-1.5 py-0.5 rounded font-medium bg-amber-600/30 text-amber-300 hover:bg-amber-600/50 transition-colors"
                          title="Expand wildcards in-place">{'{·}'} expand</button>
                      )}
                      <button onClick={()=>setAutoExpand(v=>!v)}
                        title={autoExpand ? 'Auto-expand ON: wildcards resolve on Start' : 'Auto-expand OFF'}
                        className={`text-[9px] px-1.5 py-0.5 rounded font-medium transition-colors ${autoExpand ? 'bg-amber-600/30 text-amber-300 hover:bg-amber-600/50' : 'text-zinc-600 hover:text-zinc-400 hover:bg-zinc-800'}`}>
                        auto
                      </button>
                    </div>
                  </div>
                  <textarea value={caption} onChange={e=>setCaption(e.target.value)} rows={4}
                    placeholder="Dreamy indie folk, warm acoustic guitar…"
                    className="w-full px-2.5 py-2 rounded-lg bg-zinc-900 border border-zinc-700 text-xs text-zinc-200 placeholder:text-zinc-600 outline-none focus:border-pink-500/40 transition-colors"
                    style={{resize:'vertical', minHeight:'80px'}}/>
                </div>
                {/* LoRA + Beat */}
                <div className="flex flex-col gap-1.5">
                  <div className="flex items-center gap-2">
                    <ParamLabel label="LoRA" className="text-[10px] text-zinc-500" rootClassName="shrink-0 w-10"
                      info="A trigger word prepended to the style caption. Leave blank to render with no adapter styling." />
                    <input type="text" value={loraTrigger} onChange={e=>setLoraTrigger(e.target.value)}
                      placeholder="trigger word"
                      className="flex-1 px-2 py-1 rounded bg-zinc-900 border border-zinc-700 text-[11px] text-zinc-200 placeholder:text-zinc-600 outline-none focus:border-pink-500/40 transition-colors"/>
                  </div>
                  <div className="flex items-center gap-2">
                    <button onClick={()=>setBeatIntro(!beatIntro)}
                      className={`text-[10px] px-2 py-0.5 rounded font-medium transition-colors shrink-0 ${beatIntro ? 'bg-orange-600/30 text-orange-300 border border-orange-600/40' : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 border border-zinc-700'}`}>
                      Beat I/O
                    </button>
                    {beatIntro && (
                      <div className="flex items-center gap-1">
                        <span className="text-[9px] text-zinc-600">bars:</span>
                        {([1,2,4,8]).map(n => (
                          <button key={n} onClick={()=>setIntroBars(n)}
                            className={`text-[9px] w-5 h-4 rounded font-medium transition-colors ${introBars===n ? 'bg-orange-500 text-white' : 'text-zinc-600 hover:bg-zinc-800'}`}>
                            {n}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
                {/* Instrumental */}
                <div className="flex items-center gap-2">
                  <Music size={13} className="text-zinc-500"/>
                  <Toggle
                    accent="pink"
                    size="sm"
                    checked={instrumental}
                    onChange={setInstrumental}
                    label="Instrumental"
                    info="Skips lyrics and forces an instrumental generation. On: hides the Lyrics field and sends [Instrumental] instead, so only the caption shapes the arrangement. Off: uses whatever is written in Lyrics."
                  />
                </div>
                {/* Lyrics */}
                {!instrumental && (
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <ParamLabel label="Lyrics" className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
                        info="Lyrics for the stream. The loop/cycle/shuffle buttons control how sections advance automatically as slots play: loop repeats the same lyrics, cycle steps through sections in order, shuffle picks a random unseen section each slot. Ignored when Instrumental is on." />
                      <div className="flex items-center gap-1">
                        {hasWildcards(lyrics) && (
                          <button onClick={()=>setLyrics(expandWildcards(lyrics, streamSeed, 0))}
                            className="text-[9px] px-1.5 py-0.5 rounded font-medium bg-amber-600/30 text-amber-300 hover:bg-amber-600/50 transition-colors"
                            title="Expand wildcards in-place">{'{·}'} expand</button>
                        )}
                        <button onClick={()=>setAutoExpand(v=>!v)}
                          title={autoExpand ? 'Auto-expand ON' : 'Auto-expand OFF'}
                          className={`text-[9px] px-1.5 py-0.5 rounded font-medium transition-colors ${autoExpand ? 'bg-amber-600/30 text-amber-300 hover:bg-amber-600/50' : 'text-zinc-600 hover:text-zinc-400 hover:bg-zinc-800'}`}>
                          auto
                        </button>
                      <div className="flex gap-0.5">
                        {(['loop','cycle','shuffle'] as LyricsMode[]).map(m => (
                          <button key={m} onClick={()=>setLyricsMode(m)}
                            className={`text-[9px] px-1.5 py-0.5 rounded font-medium transition-colors ${lyricsMode===m ? 'bg-purple-600 text-white' : 'text-zinc-600 hover:text-zinc-400 hover:bg-zinc-800'}`}>
                            {m==='loop'?'↻':m==='cycle'?'→':'⇌'}
                          </button>
                        ))}
                      </div>
                      </div>
                    </div>
                    <textarea value={lyrics} onChange={e=>setLyrics(e.target.value)} rows={6}
                      placeholder={'[verse 1]\nYour lyrics here…'}
                      className="w-full px-2.5 py-2 rounded-lg bg-zinc-900 border border-zinc-700 text-xs text-zinc-200 placeholder:text-zinc-600 outline-none focus:border-pink-500/40 transition-colors"
                      style={{resize:'vertical', minHeight:'120px'}}/>
                  </div>
                )}
                {/* Negative prompt */}
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <ParamLabel label="Negative Prompt" className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider"
                      info="Free-text negative prompt, sent with every slot. Optional; leave blank to skip." />
                    <div className="flex items-center gap-1">
                      {hasWildcards(negPrompt) && (
                        <button onClick={()=>setNegPrompt(expandWildcards(negPrompt, streamSeed, 0))}
                          className="text-[9px] px-1.5 py-0.5 rounded font-medium bg-amber-600/30 text-amber-300 hover:bg-amber-600/50 transition-colors"
                          title="Expand wildcards in-place">{'{·}'} expand</button>
                      )}
                      <button onClick={()=>setAutoExpand(v=>!v)}
                        title={autoExpand ? 'Auto-expand ON' : 'Auto-expand OFF'}
                        className={`text-[9px] px-1.5 py-0.5 rounded font-medium transition-colors ${autoExpand ? 'bg-amber-600/30 text-amber-300 hover:bg-amber-600/50' : 'text-zinc-600 hover:text-zinc-400 hover:bg-zinc-800'}`}>
                        auto
                      </button>
                    </div>
                  </div>
                  <textarea value={negPrompt} onChange={e=>setNegPrompt(e.target.value)} rows={2}
                    placeholder="jazz, acoustic, slow…"
                    className="w-full px-2.5 py-2 rounded-lg bg-zinc-900 border border-zinc-700 text-xs text-zinc-200 placeholder:text-zinc-600 outline-none focus:border-red-500/40 transition-colors"
                    style={{resize:'vertical', minHeight:'52px'}}/>
                </div>
              </div>

              {/* Middle — Sampler */}
              <div className="w-[310px] shrink-0 overflow-y-auto hide-scrollbar p-4 space-y-2.5 border-r border-zinc-800/60">
                <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Sampler</span>
                {/* Unified solver params — swaps automatically based on dropdown selection */}
                {activeSolverParams.map(s => {
                  const isStormParam = activeSolverKey === 'storm';
                  const stormSetters: Record<string, (v: number) => void> = {
                    stiffness_threshold: setStiffnessThreshold,
                    look_back_lambda:    setLookBackLambda,
                    look_back_snr_power: setLookBackSnrPower,
                    cache_depth:         setCacheDepth,
                  };
                  const stormVals: Record<string, number> = {
                    stiffness_threshold: stiffnessThreshold,
                    look_back_lambda:    lookBackLambda,
                    look_back_snr_power: lookBackSnrPower,
                    cache_depth:         cacheDepth,
                  };
                  const val    = isStormParam ? (stormVals[s.key] ?? s.default) : (extraSolverParams[s.key] ?? s.default);
                  const setVal = (v: number) => {
                    if (isStormParam && stormSetters[s.key]) stormSetters[s.key](v);
                    else setExtraSolverParams(p => ({...p, [s.key]: v}));
                    if (sa.isPlaying) sa.sendControl('plugin_params', {[s.key]: v});
                  };
                  return (
                    <div key={s.key} className="flex items-center gap-2">
                      <ParamLabel label={s.label} info={s.title} meta={s.meta} className="text-[10px] text-zinc-500" rootClassName="w-20 shrink-0" />
                      <input type="range" min={s.min} max={s.max} step={s.step} value={val}
                        onChange={e => setVal(Number(e.target.value))}
                        className="flex-1 accent-red-500 h-1"/>
                      <span className={`text-[11px] w-9 text-right tabular-nums ${s.color}`}>{s.fmt(val)}</span>
                    </div>
                  );
                })}
                {activeSolverKey === null && (
                  <p className="text-[10px] text-zinc-600 py-1">No extra params for this solver.</p>
                )}
                {/* Precision */}
                <div className="flex items-center gap-2">
                  <ParamLabel label="Precision" className="text-[10px] text-zinc-500" rootClassName="w-20 shrink-0"
                    info="RK solver order (auto, 2 to 5). Auto ramps up as the STORM cache fills. Applies from the next Start, not live." />
                  <div className="flex gap-0.5 flex-1">
                    {(['auto','2','3','4','5']).map(o => (
                      <button key={o} onClick={()=>setRkOrder(o)}
                        className={`flex-1 text-[9px] py-0.5 rounded font-medium transition-colors ${rkOrder===o ? 'bg-red-600 text-white' : 'bg-zinc-800 text-zinc-500 hover:bg-zinc-700 hover:text-zinc-300'}`}>
                        {o==='auto'?'auto':`RK${o}`}
                      </button>
                    ))}
                  </div>
                </div>
                {/* C++ params divider */}
                <div className="border-t border-zinc-800/60 pt-2 space-y-2">
                  {([
                    { label:'Cache Ratio', min:0,   max:0.9,  step:0.05, val:cacheRatio,     set:setCacheRatio,     fmt:(v:number)=>v>0?v.toFixed(2):'off', title:'Skip redundant DiT steps. 0=off. Restart to apply.' },
                    { label:'CFG Cutoff',  min:0.1, max:1.0,  step:0.05, val:cfgCutoffRatio, set:setCfgCutoffRatio, fmt:(v:number)=>v<1?v.toFixed(2):'off',  title:'Skip unconditional pass after ratio. 1.0=off. Restart to apply.' },
                  ]).map(s => (
                    <div key={s.label} className="flex items-center gap-2">
                      <ParamLabel label={s.label} info={s.title} className="text-[10px] text-zinc-500" rootClassName="w-20 shrink-0" />
                      <input type="range" min={s.min} max={s.max} step={s.step} value={s.val}
                        onChange={e=>s.set(Number(e.target.value))}
                        className="flex-1 accent-red-500 h-1"/>
                      <span className="text-[11px] text-zinc-300 w-9 text-right tabular-nums">{s.fmt(s.val)}</span>
                    </div>
                  ))}
                  <div className="flex items-center gap-2">
                    <ParamLabel label="LSS" className="text-[10px] text-zinc-500" rootClassName="w-20 shrink-0"
                      info="Latent self-similarity strength, live-adjustable while streaming." />
                    <input type="range" min={0} max={1} step={0.05} value={lssStrength}
                      onChange={e=>{ const v=Number(e.target.value); setLssStrength(v); if (sa.isPlaying) sa.sendControl('lss_strength',v); }}
                      className="flex-1 accent-red-500 h-1"/>
                    <span className="text-[11px] text-zinc-300 w-9 text-right tabular-nums">{lssStrength.toFixed(2)}</span>
                  </div>
                </div>
                {(cacheRatio>0||cfgCutoffRatio<1.0)&&!sa.isPlaying&&(
                  <p className="text-[8px] text-amber-500/70">Cache Ratio / CFG Cutoff apply on next Start</p>
                )}
                {/* Solver / Scheduler / Guider */}
                <div className="border-t border-zinc-800/60 pt-2 space-y-1.5">
                  {([
                    { label:'Solver',    val:liveInferMethod,  set:setLiveInferMethod,  ctrlKey:'infer_method',  opts:pluginRegistry.solvers    },
                    { label:'Scheduler', val:liveScheduler,    set:setLiveScheduler,    ctrlKey:'scheduler',     opts:pluginRegistry.schedulers },
                    { label:'Guider',    val:liveGuidanceMode, set:setLiveGuidanceMode, ctrlKey:'guidance_mode', opts:pluginRegistry.guidance   },
                  ]).map(row => (
                    <div key={row.label} className="flex items-center gap-1.5">
                      <ParamLabel label={row.label} className="text-[10px] text-zinc-500" rootClassName="w-14 shrink-0"
                        info={`Overrides the stream's solver, noise scheduler or guidance mode. "— inherit global —" uses whatever is set in the global bar. Changing this while a stream plays sends the change live.`} />
                      <StyledSelect
                        accent="pink"
                        size="sm"
                        value={row.val}
                        onChange={v=>{ row.set(v); if (sa.isPlaying&&v) sa.sendControl(row.ctrlKey,v); }}
                        title={row.val || '— inherit global —'}
                        options={[{value:'', label:'— inherit global —'}, ...row.opts.map(p=>({value:p.name, label:p.display}))]}
                        className="flex-1 min-w-0"/>
                      {sa.isPlaying&&row.val&&<span className="text-[8px] text-pink-400 shrink-0">live</span>}
                    </div>
                  ))}
                </div>
                {/* Scheduler params */}
                {activeSchedulerParams.length > 0 && (
                  <div className="border-t border-zinc-800/60 pt-2 space-y-2">
                    <span className="text-[9px] text-zinc-600 uppercase tracking-wider">Scheduler params</span>
                    {activeSchedulerParams.map(s => (
                      <div key={s.key} className="flex items-center gap-2">
                        <ParamLabel label={s.label} info={s.title} meta={s.meta} className="text-[10px] text-zinc-500" rootClassName="w-20 shrink-0" />
                        {s.type === 'select' ? (
                          <StyledSelect
                            accent="pink"
                            size="sm"
                            value={String(extraSchedulerParams[s.key] ?? s.default)}
                            onChange={v => { setExtraSchedulerParams(p=>({...p,[s.key]:v})); if (sa.isPlaying) sa.sendControl('plugin_params',{[s.key]:v}); }}
                            options={(s.options ?? []).map(o => ({value:o.value, label:o.label}))}
                            className="flex-1 min-w-0"/>
                        ) : (
                          <>
                            <input type="range" min={s.min} max={s.max} step={s.step}
                              value={Number(extraSchedulerParams[s.key] ?? s.default)}
                              onChange={e => { const v=Number(e.target.value); setExtraSchedulerParams(p=>({...p,[s.key]:v})); if (sa.isPlaying) sa.sendControl('plugin_params',{[s.key]:v}); }}
                              className="flex-1 accent-red-500 h-1"/>
                            <span className={`text-[11px] w-9 text-right tabular-nums ${s.color}`}>{(s.fmt ?? String)(Number(extraSchedulerParams[s.key] ?? s.default))}</span>
                          </>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                {/* Guider params */}
                {activeGuiderParams.length > 0 && (
                  <div className="border-t border-zinc-800/60 pt-2 space-y-2">
                    <span className="text-[9px] text-zinc-600 uppercase tracking-wider">Guider params</span>
                    {activeGuiderParams.map(s => (
                      <div key={s.key} className="flex items-center gap-2">
                        <ParamLabel label={s.label} info={s.title} meta={s.meta} className="text-[10px] text-zinc-500" rootClassName="w-20 shrink-0" />
                        {s.type === 'select' ? (
                          <StyledSelect
                            accent="pink"
                            size="sm"
                            value={String(extraGuiderParams[s.key] ?? s.default)}
                            onChange={v => { setExtraGuiderParams(p=>({...p,[s.key]:v})); if (sa.isPlaying) sa.sendControl('plugin_params',{[s.key]:v}); }}
                            options={(s.options ?? []).map(o => ({value:o.value, label:o.label}))}
                            className="flex-1 min-w-0"/>
                        ) : (
                          <>
                            <input type="range" min={s.min} max={s.max} step={s.step}
                              value={Number(extraGuiderParams[s.key] ?? s.default)}
                              onChange={e => { const v=Number(e.target.value); setExtraGuiderParams(p=>({...p,[s.key]:v})); if (sa.isPlaying) sa.sendControl('plugin_params',{[s.key]:v}); }}
                              className="flex-1 accent-red-500 h-1"/>
                            <span className={`text-[11px] w-9 text-right tabular-nums ${s.color}`}>{(s.fmt ?? String)(Number(extraGuiderParams[s.key] ?? s.default))}</span>
                          </>
                        )}
                      </div>
                    ))}
                  </div>
                )}
                {/* XFade */}
                <div className="flex items-center gap-2">
                  <ParamLabel label="XFade" className="text-[10px] text-zinc-500" rootClassName="w-20 shrink-0"
                    info="Crossfade length between slots, in beats." />
                  <input type="range" min={0} max={64} step={1} value={xfadeBeats}
                    onChange={e=>{ const v=Number(e.target.value); setXfadeBeats(v); sa.setXfadeBeats(v); }}
                    className="flex-1 accent-red-500 h-1"/>
                  <span className="text-[11px] text-zinc-300 w-9 text-right tabular-nums">{xfadeBeats}b</span>
                </div>
                {/* Max Buf */}
                <div className="flex items-center gap-2">
                  <label className="text-[10px] text-zinc-500 w-20 shrink-0 flex items-center gap-1">
                    <ParamLabel label="MaxBuf" className="text-[10px] text-zinc-500"
                      info="How far ahead of playback the client is allowed to buffer, in minutes or in slot multiples (toggle with the clock/note icon). The engine pauses rendering once the buffer fills and resumes once playback catches up." />
                    {sa.bufferPaused&&<span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse inline-block"/>}
                    <span className="flex gap-0.5 ml-0.5">
                      <button onClick={()=>setBufMode('time')} className={`text-[8px] px-0.5 rounded transition-colors ${bufMode==='time'?'bg-zinc-600 text-zinc-200':'text-zinc-700'}`}>⏱</button>
                      <button onClick={()=>setBufMode('slots')} className={`text-[8px] px-0.5 rounded transition-colors ${bufMode==='slots'?'bg-zinc-600 text-zinc-200':'text-zinc-700'}`}>🎵</button>
                    </span>
                  </label>
                  {bufMode==='time' ? (
                    <>
                      <input type="range" min={60} max={3600} step={60} value={maxBuffer}
                        onChange={e=>{ const v=Number(e.target.value); setMaxBufferState(v); sa.setMaxBuffer(v); }}
                        className="flex-1 accent-red-500 h-1"/>
                      <span className="text-[11px] text-zinc-300 w-9 text-right tabular-nums">{fmtBuf(maxBuffer)}</span>
                    </>
                  ) : (
                    <>
                      <input type="range" min={1} max={8} step={1} value={bufSlots}
                        onChange={e=>{ const v=Number(e.target.value); setBufSlots(v); const s=v*(sa.estimatedSlotDuration||180); setMaxBufferState(s); sa.setMaxBuffer(s); }}
                        className="flex-1 accent-red-500 h-1"/>
                      <span className="text-[11px] text-zinc-300 w-9 text-right tabular-nums">{bufSlots}✕/{fmtBuf(bufSlots*(sa.estimatedSlotDuration||180))}</span>
                    </>
                  )}
                </div>
              </div>

              {/* Right — Live Controls */}
              <div className="flex-1 min-w-0 overflow-y-auto hide-scrollbar">
                <StormLiveControls
                  sa={sa}
                  lyrics={lyrics}
                  caption={caption}
                  streamPrompt={streamPrompt} onStreamPromptChange={setStreamPrompt}
                  streamLyrics={streamLyricsNS} onStreamLyricsChange={setStreamLyricsNS}
                  persistStyle={persistStyle} onPersistStyleChange={setPersistStyle}
                  persistLyrics={persistLyrics} onPersistLyricsChange={setPersistLyrics}
                  streamSeed={streamSeed} onStreamSeedChange={setStreamSeed}
                  streamSeedLock={streamSeedLock} onStreamSeedLockChange={setStreamSeedLock}
                  onStreamSend={() => {}}
                  gpGuidanceScale={gp.guidanceScale??7}
                  gpInferenceSteps={gp.inferenceSteps??8}
                  gpBpm={bpm} gpDuration={duration}
                  showMiniPlayer={mode==='continuous'}
                  alwaysShowControls={true}
                  lyricsMode={lyricsMode}
                  onParamsChange={setLiveParams}
                  onStop={handleStop}
                  activeSolver={liveInferMethod?(pluginRegistry.solvers.find(p=>p.name===liveInferMethod)?.display??liveInferMethod.replace(/^md_/,'').replace(/_/g,' ')):undefined}
                  activeScheduler={liveScheduler?(pluginRegistry.schedulers.find(p=>p.name===liveScheduler)?.display??liveScheduler.replace(/^md_/,'').replace(/_/g,' ')):undefined}
                  activeGuidanceMode={liveGuidanceMode?(pluginRegistry.guidance.find(p=>p.name===liveGuidanceMode)?.display??liveGuidanceMode.replace(/^md_/,'').replace(/_/g,' ')):undefined}
                  onSlotMetaUpdate={setSlotMeta}
                />
              </div>

            </div>
          </>
        );
      })()}

      {/* ── DJ Mode ── */}
      {mode==='dj' && (
        <div className="flex-1 flex overflow-hidden p-4">
          <div className="flex gap-4 w-full h-full">
            {/* Deck A */}
            <DeckPanel id="deck-a" label="Deck A" otherCamelot={camelotB} crossfadeGain={gainA}
              onRegister={sa => { deckARef.current = sa; setDeckAKey(sa.detectedKey); }} />

            {/* Crossfader center column */}
            <div className="flex flex-col items-center justify-start pt-4 gap-4 w-48 shrink-0">
              {/* Camelot compatibility */}
              {camelotA && camelotB && (
                <div className={`text-center px-3 py-2 rounded-lg border text-xs font-medium ${
                  djCompat==='compat'?'border-green-500/40 bg-green-500/10 text-green-400':
                  djCompat==='adjacent'?'border-yellow-500/40 bg-yellow-500/10 text-yellow-400':
                  'border-red-500/40 bg-red-500/10 text-red-400'}`}>
                  {camelotA} ↔ {camelotB}
                  <div className="text-[9px] mt-0.5">
                    {djCompat==='compat'?'✓ Compatible':djCompat==='adjacent'?'~ Adjacent':'✗ Clash'}
                  </div>
                </div>
              )}

              {/* Crossfader */}
              <div className="flex flex-col items-center gap-2 w-full">
                <div className="flex justify-between w-full text-[9px] text-zinc-500 px-1">
                  <span>A</span><span>B</span>
                </div>
                <input type="range" min={0} max={1} step={0.01} value={xfader}
                  onChange={e => setXfader(Number(e.target.value))}
                  className="w-full accent-pink-500"/>
                <div className="text-[9px] text-zinc-500 tabular-nums">
                  {Math.round(gainA*100)}% A / {Math.round(gainB*100)}% B
                </div>
              </div>

              {/* Center cut buttons */}
              <div className="flex gap-1">
                <button onClick={()=>setXfader(0)} className="text-[9px] px-2 py-1 rounded bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors">A</button>
                <button onClick={()=>setXfader(0.5)} className="text-[9px] px-2 py-1 rounded bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors">⊙</button>
                <button onClick={()=>setXfader(1)} className="text-[9px] px-2 py-1 rounded bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-colors">B</button>
              </div>
              <div className="space-y-1.5">
                <div className="text-[8px] text-zinc-600 text-center uppercase tracking-wider">Sync</div>
                <button
                  onClick={() => {
                    // Quantized start: start deck B on next beat boundary of deck A
                    const saA = deckARef.current;
                    const saB = deckBRef.current;
                    if (!saA || !saB || !saA.isPlaying) return;
                    const nextBeat = saA.getNextBeatTime(saA.detectedBpm > 0 ? saA.detectedBpm : bpm);
                    // shiftNextSlot on B to align with A's next beat
                    saB.shiftNextSlot((nextBeat - (Date.now() / 1000)) * 1000);
                  }}
                  className="w-full text-[9px] py-0.5 rounded bg-zinc-800 text-zinc-400 hover:bg-violet-900/40 hover:text-violet-300 border border-zinc-700 transition-colors">
                  ⊙ Quantize A→B
                </button>
              </div>
              <div className="text-[9px] text-zinc-600 text-center">Nudge</div>
              {/* Nudge — deck A */}
              <div className="space-y-1.5">
                <div className="text-[8px] text-zinc-600 text-center uppercase tracking-wider">Nudge A</div>
                <div className="flex gap-1 justify-center">
                  <button onClick={() => deckARef.current?.shiftNextSlot(-200)}
                    className="text-[9px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700 transition-colors">◂◂</button>
                  <button onClick={() => deckARef.current?.shiftNextSlot(-50)}
                    className="text-[9px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700 transition-colors">◂</button>
                  <button onClick={() => deckARef.current?.shiftNextSlot(50)}
                    className="text-[9px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700 transition-colors">▸</button>
                  <button onClick={() => deckARef.current?.shiftNextSlot(200)}
                    className="text-[9px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700 transition-colors">▸▸</button>
                </div>
              </div>
              {/* Nudge — deck B */}
              <div className="space-y-1.5">
                <div className="text-[8px] text-zinc-600 text-center uppercase tracking-wider">Nudge B</div>
                <div className="flex gap-1 justify-center">
                  <button onClick={() => deckBRef.current?.shiftNextSlot(-200)}
                    className="text-[9px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700 transition-colors">◂◂</button>
                  <button onClick={() => deckBRef.current?.shiftNextSlot(-50)}
                    className="text-[9px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700 transition-colors">◂</button>
                  <button onClick={() => deckBRef.current?.shiftNextSlot(50)}
                    className="text-[9px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700 transition-colors">▸</button>
                  <button onClick={() => deckBRef.current?.shiftNextSlot(200)}
                    className="text-[9px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700 transition-colors">▸▸</button>
                </div>
              </div>
            </div>

            {/* Deck B */}
            <DeckPanel id="deck-b" label="Deck B" otherCamelot={camelotA} crossfadeGain={gainB}
              onRegister={sa => { deckBRef.current = sa; setDeckBKey(sa.detectedKey); }} />
          </div>
        </div>
      )}

      {/* ── Drift placeholder ── */}
      {mode==='drift' && (
        <div className="flex-1 flex overflow-hidden p-4">
          <div className="flex-1 flex items-start justify-center pt-16">
            <div className="text-center space-y-3 max-w-md px-6">
              <div className="text-4xl">🌊</div>
              <h2 className="text-xl font-bold text-white">Directed Drift</h2>
              <p className="text-sm text-zinc-400 leading-relaxed">
                The stream never stops. It evolves forever. Inputs are pressure — not commands.
                You're not programming it, you're steering it.
              </p>
              <p className="text-xs text-zinc-600">Coming soon — Plank H</p>
            </div>
          </div>
        </div>
      )}
    </div>
    </BackendCapabilityGate>
  );
};