// MidiExportPanel.tsx — tempo, meter and bar corrections for a transcription
// that has a beat grid, plus quantize / gap fill for notation software.
//
// Rewrites the job's .mid on the server (POST /:jobId/export) from the kept
// note events and tracked beats: grid maths only, no model, about a second.
// The options are MuScriptor v0.4's export options (ace-midi --help).

import React, { useState } from 'react';
import { ChevronLeft, ChevronRight, Copy, Download, Loader2, RotateCcw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { useMutation, useQuery } from '@tanstack/react-query';
import { midiNotesOptions } from '@/queries/studios/midi';
import { channelLabel, getMidiAbc, type MidiExportOptions, type MidiGrid } from '@/services/midiStudioApi';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { StyledSelect } from '@/components/shared/controls/StyledSelect';
import { Toggle } from '@/components/shared/controls/Toggle';
import { QueryError } from '@/components/shared/feedback/QueryError';
import { saveText } from '@/utils/storage/saveFile';
import { formatDecimal } from '@/utils/format/format';

interface Props {
  jobId: string;
  applied: MidiExportOptions;
  apply: (next: MidiExportOptions, onSuccess?: () => void) => void;
  busy: boolean;
}

const input = 'w-full px-2.5 py-1.5 rounded-lg text-xs bg-zinc-100 dark:bg-black/30 border border-zinc-200 dark:border-white/10 text-zinc-900 dark:text-white placeholder-zinc-400 outline-none focus:border-purple-500/40';

export function MidiDownbeatShift({ grid, busy, onShift }: {
  grid: MidiGrid;
  busy: boolean;
  onShift: (downbeat: number) => void;
}) {
  const { t } = useTranslation();
  const period = 60 / grid.bpm;
  const anchor = grid.firstDownbeat ?? (grid.bars[0] ?? grid.beats[0] ?? 0) - grid.onsetDelay;
  const shift = (direction: -1 | 1) => {
    if (!grid.steady) {
      // A tempo map needs the adjacent tracked beat, even when the local beat
      // spacing differs considerably from the song's average BPM.
      const bar = grid.bars.find(b => {
        const index = grid.beats.indexOf(b);
        return index >= 0 && grid.beats[index + direction] !== undefined;
      });
      if (bar !== undefined) {
        onShift(grid.beats[grid.beats.indexOf(bar) + direction] - grid.onsetDelay);
        return;
      }
    }
    onShift(anchor + direction * period);
  };
  return (
    <div className="mt-3 flex items-center gap-2 text-xs">
      <ParamLabel label={t('midiStudio.export.downbeat')} info={t('midiStudio.export.downbeatInfo')} />
      {([-1, 1] as const).map(direction => {
        const label = t(direction < 0 ? 'midiStudio.export.downbeatEarlier' : 'midiStudio.export.downbeatLater');
        const Icon = direction < 0 ? ChevronLeft : ChevronRight;
        return <button key={direction} type="button" disabled={busy} aria-label={label} title={label}
          className="rounded border border-zinc-200 p-1 hover:text-purple-500 disabled:opacity-50 dark:border-white/10"
          onClick={() => shift(direction)}><Icon size={15} /></button>;
      })}
    </div>
  );
}

export const MidiExportPanel: React.FC<Props> = ({ jobId, applied, apply, busy }) => {
  const { t } = useTranslation();
  const gridQuery = useQuery(midiNotesOptions(jobId));
  const grid = gridQuery.data?.grid;
  const [draft, setDraft] = useState<MidiExportOptions>({});
  const opts = { ...applied, ...draft };

  if (!grid) {
    return gridQuery.isError
      ? <QueryError error={gridQuery.error} onRetry={() => void gridQuery.refetch()} />
      : null;
  }

  const set = (patch: MidiExportOptions) => setDraft(o => ({ ...o, ...patch }));
  const num = (v: string) => (v.trim() === '' ? undefined : Number(v));

  const summary = [
    t('midiStudio.export.bpm', { bpm: grid.bpm.toFixed(1) }),
    grid.beatsPerBar ? `${grid.beatsPerBar}/4` : t('midiStudio.export.meterUnknown'),
    grid.steady ? t('midiStudio.export.fixedTempo') : t('midiStudio.export.tempoMap'),
  ].join(' · ');

  return (
    <div className="mt-2 rounded-lg border border-zinc-200 dark:border-white/5 bg-white dark:bg-black/20 p-3">
      <div className="flex items-center gap-2 flex-wrap mb-2.5">
        <span className="text-xs font-semibold text-zinc-800 dark:text-zinc-200">{t('midiStudio.export.title')}</span>
        <span className="text-[11px] text-zinc-500 dark:text-zinc-400">{summary}</span>
      </div>

      <div className="grid grid-cols-2 min-[1024px]:grid-cols-4 gap-x-3 gap-y-2.5">
        <label className="flex flex-col gap-1">
          <ParamLabel label={t('midiStudio.export.bpmLabel')} meta={t('midiStudio.export.blankDetected')}
            info={t('midiStudio.export.bpmInfo')} />
          <input className={input} type="number" min={20} max={400} step={0.1} placeholder={formatDecimal(grid.bpm, 1)}
            value={opts.bpm ?? ''} onChange={e => set({ bpm: num(e.target.value) })} />
        </label>
        <label className="flex flex-col gap-1">
          <ParamLabel label={t('midiStudio.export.beatsPerBar')} meta={t('midiStudio.export.blankDetected')}
            info={t('midiStudio.export.beatsPerBarInfo')} />
          <input className={input} type="number" min={1} max={16} step={1} placeholder={grid.beatsPerBar ? String(grid.beatsPerBar) : '?'}
            value={opts.beatsPerBar ?? ''} onChange={e => set({ beatsPerBar: num(e.target.value) })} />
        </label>
        <div className="flex flex-col gap-1">
          <ParamLabel label={t('midiStudio.export.tempo')} info={t('midiStudio.export.tempoInfo')} />
          <StyledSelect size="sm" accent="purple" value={opts.tempoMode ?? 'auto'}
            onChange={v => set({ tempoMode: v })}
            options={[
              { value: 'auto', label: t('midiStudio.export.tempoAuto') },
              { value: 'map', label: t('midiStudio.export.tempoMap') },
              { value: 'fixed', label: t('midiStudio.export.fixedTempo') },
            ]} />
        </div>
        <div className="flex flex-col gap-1">
          <ParamLabel label={t('midiStudio.export.beatScale')} info={t('midiStudio.export.beatScaleInfo')} />
          <StyledSelect size="sm" accent="purple" value={opts.beatScale ?? 1}
            onChange={v => set({ beatScale: v === 1 ? undefined : v })}
            options={[
              { value: 0.5, label: t('midiStudio.export.halfTime') },
              { value: 1, label: t('midiStudio.export.asDetected') },
              { value: 2, label: t('midiStudio.export.doubleTime') },
            ]} />
        </div>
      </div>

      {grid.beats.length > 1 && (
        <MidiDownbeatShift grid={grid} busy={busy}
          onShift={firstDownbeat => apply({
            ...applied, firstDownbeat, beatsPerBar: applied.beatsPerBar ?? (grid.beatsPerBar ? undefined : 4),
          })} />
      )}

      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 mt-3">
        <Toggle size="sm" accent="purple" checked={!!opts.quantize} onChange={v => set({ quantize: v })}
          label={t('midiStudio.export.quantize')} info={t('midiStudio.export.quantizeInfo')} />
        {opts.quantize && (
          <>
            <div className="flex items-center gap-2">
              <ParamLabel label={t('midiStudio.export.grid')} info={t('midiStudio.export.gridInfo')} />
              <StyledSelect size="sm" accent="purple" value={opts.grid ?? 0} className="w-36"
                onChange={v => set({ grid: v || undefined })}
                options={[
                  { value: 0, label: grid.subdivision ? t('midiStudio.export.gridDetected', { n: grid.subdivision }) : t('midiStudio.export.gridAuto') },
                  { value: 2, label: t('midiStudio.export.grid8th') },
                  { value: 3, label: t('midiStudio.export.grid8thTriplet') },
                  { value: 4, label: t('midiStudio.export.grid16th') },
                  { value: 6, label: t('midiStudio.export.grid16thTriplet') },
                  { value: 8, label: t('midiStudio.export.grid32nd') },
                ]} />
            </div>
            <div className="flex items-center gap-2">
              <ParamLabel label={t('midiStudio.export.strength')} info={t('midiStudio.export.strengthInfo')} />
              <StyledSelect size="sm" accent="purple" value={opts.strength ?? 1} className="w-24"
                onChange={v => set({ strength: v === 1 ? undefined : v })}
                options={[1, 0.75, 0.5, 0.25].map(s => ({ value: s, label: `${s * 100}%` }))} />
            </div>
          </>
        )}
        <Toggle size="sm" accent="purple" checked={!!opts.fillGaps} onChange={v => set({ fillGaps: v ? 0.5 : undefined })}
          label={t('midiStudio.export.fillGaps')} info={t('midiStudio.export.fillGapsInfo')} />
        <Toggle size="sm" accent="purple" checked={!!opts.trimLeadingBars} onChange={v => set({ trimLeadingBars: v })}
          label={t('midiStudio.export.trimLeading')} info={t('midiStudio.export.trimLeadingInfo')} />
      </div>

      <div className="flex items-center gap-3 mt-3">
        <button onClick={() => apply(opts, () => setDraft({}))} disabled={busy}
          className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-50 transition-colors flex items-center gap-1.5">
          {busy && <Loader2 size={12} className="animate-spin" />} {t('midiStudio.export.apply')}
        </button>
        <button onClick={() => apply({
          fullVelocity: applied.fullVelocity, programOverrides: applied.programOverrides,
        }, () => setDraft({}))} disabled={busy}
          className="px-3 py-1.5 rounded-lg text-xs font-medium text-zinc-500 dark:text-zinc-400 hover:text-purple-600 dark:hover:text-purple-400 border border-zinc-200 dark:border-white/10 transition-colors flex items-center gap-1.5">
          <RotateCcw size={11} /> {t('midiStudio.export.reset')}
        </button>
      </div>

      <AbcRow jobId={jobId} />
    </div>
  );
};

// The transcription as a YuE2 lead sheet: copy it into Cover Studio's ABC
// box, or save it. Uses the grid the MIDI was last written on.
const AbcRow: React.FC<{ jobId: string }> = ({ jobId }) => {
  const { t } = useTranslation();
  const channelsQuery = useQuery(midiNotesOptions(jobId));
  const channels = channelsQuery.data?.channels ?? [];
  const [ins, setIns] = useState(-1);
  const [chords, setChords] = useState(true);
  const [note, setNote] = useState('');
  const programs = [...new Set(channels.filter(c => !c.isDrums).map(c => c.sourceProgram ?? c.program).filter(p => p !== 52))];

  const { mutate: makeAbc, isPending: busy, error: abcError } = useMutation({
    mutationFn: async (action: 'copy' | 'save') => {
      const r = await getMidiAbc(jobId, { ins: ins >= 0 ? ins : undefined, melodyOnly: !chords });
      if (action === 'copy') await navigator.clipboard.writeText(r.abc);
      else {
        saveText(r.abc, 'lead-sheet.abc', 'text/plain');
      }
      return { action, summary: r.summary };
    },
    onSuccess: ({ action, summary }) => {
      setNote(t(action === 'copy' ? 'midiStudio.abc.copied' : 'midiStudio.abc.saved', {
        vocal: summary.vocal_notes, ins: summary.ins.replace(/_/g, ' '), insNotes: summary.ins_notes, key: summary.key.replace(':', ' '), chords: summary.chords,
      }) + (summary.vocal_notes < 40 ? ' ' + t('midiStudio.abc.fewVocal') : ''));
    },
  });
  const error = abcError?.message ?? '';

  return (
    <div className="mt-3 pt-3 border-t border-zinc-200 dark:border-white/5 flex flex-wrap items-center gap-x-4 gap-y-2">
      {channelsQuery.isError && (
        <QueryError error={channelsQuery.error} onRetry={() => void channelsQuery.refetch()} compact className="basis-full" />
      )}
      <ParamLabel label={t('midiStudio.abc.title')} info={t('midiStudio.abc.info')} />
      <div className="flex items-center gap-2">
        <ParamLabel label={t('midiStudio.abc.ins')} info={t('midiStudio.abc.insInfo')} />
        <StyledSelect size="sm" accent="purple" value={ins} className="w-40" onChange={setIns}
          options={[{ value: -1, label: t('midiStudio.abc.insAuto') },
            ...programs.map(p => ({ value: p, label: channelLabel({ channel: 0, program: p, isDrums: false, noteCount: 0 }) + ` (${p})` }))]} />
      </div>
      <Toggle size="sm" accent="purple" checked={chords} onChange={setChords}
        label={t('midiStudio.abc.chords')} info={t('midiStudio.abc.chordsInfo')} />
      <button onClick={() => makeAbc('copy')} disabled={busy}
        className="px-3 py-1.5 rounded-lg text-xs font-medium text-purple-600 dark:text-purple-300 border border-purple-500/40 hover:bg-purple-500/10 disabled:opacity-50 transition-colors flex items-center gap-1.5">
        {busy ? <Loader2 size={11} className="animate-spin" /> : <Copy size={11} />} {t('midiStudio.abc.copy')}
      </button>
      <button onClick={() => makeAbc('save')} disabled={busy}
        className="text-xs text-zinc-500 dark:text-zinc-400 hover:text-purple-600 dark:hover:text-purple-400 hover:underline flex items-center gap-1">
        <Download size={11} /> {t('midiStudio.abc.save')}
      </button>
      {note && <span className="text-[11px] text-zinc-500 dark:text-zinc-400 basis-full">{note}</span>}
      {error && <span className="text-[11px] text-red-500 dark:text-red-400 basis-full">{error}</span>}
    </div>
  );
};
