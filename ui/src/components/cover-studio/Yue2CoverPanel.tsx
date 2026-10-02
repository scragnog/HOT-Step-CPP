import React from 'react';
import { Loader2 } from 'lucide-react';
import { ParamLabel } from '../shared/ParamLabel';
import { StyledSelect } from '../shared/StyledSelect';
import { Toggle } from '../shared/Toggle';
import type { Yue2SourceTrack } from '../../utils/yue2CaptionSource';

interface Props {
  caption: string;
  onCaptionChange: (value: string) => void;
  captionMode: string;
  onCaptionMode: (value: string) => void;
  captionTracks: Yue2SourceTrack[];
  resolvedCaption: string;
  instrumental: boolean;
  onInstrumentalChange: (value: boolean) => void;
  pairMode: 'base' | 'pair';
  onPairMode: (value: 'base' | 'pair') => void;
  ar: string;
  nar: string;
  onAr: (value: string) => void;
  onNar: (value: string) => void;
  arOptions: Array<{ value: string; label: string }>;
  narOptions: Array<{ value: string; label: string }>;
  cot: 'melody' | 'full';
  onCot: (value: 'melody' | 'full') => void;
  canGenerate: boolean;
  isGenerating: boolean;
  genProgress: number;
  genStage: string;
  onGenerate: () => void;
  onCancel: () => void;
}

export const Yue2CoverPanel: React.FC<Props> = p => (
  <div className="w-[420px] flex-shrink-0 overflow-y-auto scrollbar-hide p-4 space-y-5">
    <div className="space-y-2">
      <ParamLabel label="YuE2 style caption" info="Describe the target genre, instruments, mood and vocal style. An adapter can also use a caption from its training dataset." />
      {p.captionTracks.length > 0 && p.pairMode === 'pair' && (
        <>
          <ParamLabel label="Caption source" info="Use your own caption, the training track nearest the detected BPM, or a named training track." />
          <StyledSelect accent="pink" value={p.captionMode} onChange={p.onCaptionMode}
            options={[
              { value: 'custom', label: 'Custom caption' },
              { value: 'auto', label: 'Automatic from dataset' },
              ...p.captionTracks.map(track => ({ value: `track:${track.name}`, label: `Track: ${track.name}` })),
            ]} />
        </>
      )}
      <textarea value={p.caption} onChange={e => p.onCaptionChange(e.target.value)}
        placeholder="Describe the target sound"
        className="w-full h-24 resize-none rounded-xl bg-white dark:bg-black/20 border border-zinc-200 dark:border-white/10 px-3 py-2 text-xs text-zinc-900 dark:text-white" />
      {p.captionMode !== 'custom' && p.captionTracks.length > 0 && (
        <p className="text-xs text-zinc-500">Render caption: {p.resolvedCaption || 'No caption available'}</p>
      )}
    </div>

    <div className="flex items-center justify-between">
      <ParamLabel label="Instrumental" info="Skip lyrics and render the approved melody without vocals." />
      <Toggle accent="cyan" checked={p.instrumental} onChange={p.onInstrumentalChange} aria-label="Instrumental" />
    </div>

    <div className="space-y-2">
      <ParamLabel label="Adapter pair" info="Base uses neither YuE2 adapter. Choose the AR composer and NAR renderer separately for an adapter cover." />
      <StyledSelect accent="pink" value={p.pairMode} onChange={v => p.onPairMode(v as 'base' | 'pair')}
        options={[{ value: 'base', label: 'Base YuE2' }, { value: 'pair', label: 'Choose AR/NAR pair' }]} />
      {p.pairMode === 'pair' && (
        <div className="space-y-2">
          <ParamLabel label="AR composer" info="The AR adapter steers the musical plan and semantic tokens." />
          <StyledSelect accent="pink" value={p.ar} onChange={p.onAr} options={p.arOptions} />
          <ParamLabel label="NAR renderer" info="The NAR adapter steers the audio rendering stage." />
          <StyledSelect accent="pink" value={p.nar} onChange={p.onNar} options={p.narOptions} />
        </div>
      )}
    </div>

    <div className="space-y-2">
      <ParamLabel label="Score conditioning" info="Melody follows the reviewed tune and leaves harmony open. Full score also follows the chords you add to the ABC." />
      <StyledSelect accent="pink" value={p.cot} onChange={v => p.onCot(v as 'melody' | 'full')}
        options={[{ value: 'melody', label: 'Melody only' }, { value: 'full', label: 'Full score (fixed harmony)' }]} />
      {p.cot === 'full' && <p className="text-xs text-zinc-500">Add chord symbols to the ABC before approval to fix the harmony.</p>}
    </div>

    {p.isGenerating && (
      <div className="space-y-2 text-xs">
        <p>{p.genStage || 'Generating'} · {p.genProgress}%</p>
        <div className="h-2 bg-zinc-800 rounded-full"><div className="h-full bg-pink-500 rounded-full" style={{ width: `${p.genProgress}%` }} /></div>
        <button onClick={p.onCancel} className="text-red-400">Cancel current render</button>
      </div>
    )}
    <button onClick={p.onGenerate} disabled={!p.canGenerate}
      className="w-full rounded-xl bg-gradient-to-r from-pink-600 to-purple-600 py-3 text-sm font-bold text-white disabled:opacity-40 disabled:cursor-not-allowed">
      {p.isGenerating ? <><Loader2 className="inline w-4 h-4 mr-1 animate-spin" /> Add to queue</> : 'Generate YuE2 cover'}
    </button>
    {!p.canGenerate && <p className="text-xs text-zinc-500">Choose audio, approve an ABC score, and enter lyrics or select Instrumental.</p>}
  </div>
);
