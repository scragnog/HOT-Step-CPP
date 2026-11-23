import React, { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { StyledSelect } from '@/components/shared/controls/StyledSelect';
import { importMusicXml, type MusicXmlPart, type MusicXmlResult } from '@/components/cover-studio/musicxml';

interface Props {
  abc: string;
  disabled: boolean;
  /** `imported` carries the score's title and any lyrics, which the native score itself leaves out. */
  onAbcChange: (abc: string, imported?: { title: string; lyrics: string; chords: MusicXmlResult['chords'] }) => void;
  children: React.ReactNode;
  /** Shown at the right-hand end of the import button's row. */
  action?: React.ReactNode;
  /** Right-aligned over the left column (the ABC box) when `wide`. */
  toggle?: React.ReactNode;
  /** The row splits into the same two columns as the sheet below it. */
  wide?: boolean;
}

const isMusicXml = (name: string) => /\.(musicxml|xml|mxl)$/i.test(name);

export const MusicXmlImport: React.FC<Props> = ({ abc, disabled, onAbcChange, children, action, toggle, wide }) => {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const abcRef = useRef(abc);
  const disabledRef = useRef(disabled);
  useEffect(() => { abcRef.current = abc; disabledRef.current = disabled; }, [abc, disabled]);
  const requestRef = useRef(0);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [warnings, setWarnings] = useState<string[]>([]);
  const [parts, setParts] = useState<MusicXmlPart[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [selectedPart, setSelectedPart] = useState('');
  const [selectedStaff, setSelectedStaff] = useState('1');
  const [selectedVerse, setSelectedVerse] = useState('');

  useEffect(() => () => { requestRef.current++; }, []);

  const convert = async (source: File, partId?: string, staff?: number, verse?: string) => {
    const request = ++requestRef.current;
    const startingAbc = abcRef.current;
    setLoading(true);
    setError('');
    setWarnings([]);
    try {
      const result = await importMusicXml(source, source.name, partId ? { partId, staff, verse } : undefined);
      if (request !== requestRef.current) return;
      if (disabledRef.current) {
        setError(t('cover.musicXml.busy'));
        return;
      }
      if (!partId && (result.parts.length > 1 || result.parts[0].staves > 1 || result.parts[0].verses.length > 1)) {
        setFile(source);
        setParts(result.parts);
        setSelectedPart(result.parts[0].id);
        setSelectedStaff('1');
        setSelectedVerse(result.parts[0].verses[0] ?? '');
        setWarnings(result.warnings);
        return;
      }
      if (!result.abc) throw new Error(t('cover.musicXml.empty'));
      if (abcRef.current !== startingAbc) {
        setError(t('cover.musicXml.changed'));
        return;
      }
      if (startingAbc.trim() && !window.confirm(t('cover.musicXml.replace'))) return;
      onAbcChange(result.abc, { title: result.title, lyrics: result.lyrics, chords: result.chords });
      setWarnings(result.warnings);
      setParts([]);
      setFile(null);
    } catch (cause) {
      if (request === requestRef.current) setError(cause instanceof Error ? cause.message : t('cover.musicXml.failed'));
    } finally {
      if (request === requestRef.current) setLoading(false);
    }
  };

  const openFile = (source?: File) => {
    if (!source || disabled || loading) return;
    if (!isMusicXml(source.name)) {
      setError(t('cover.musicXml.fileType'));
      return;
    }
    setParts([]);
    setFile(null);
    void convert(source);
  };

  const choice = parts.find(part => part.id === selectedPart);

  return (
    <div className="space-y-2">
      <div className={wide ? 'grid grid-cols-2 gap-3 items-center' : 'flex items-center gap-2'}>
      <div className="flex min-w-0 items-center gap-2">
        <button type="button" disabled={disabled || loading} onClick={() => inputRef.current?.click()}
          className="whitespace-nowrap rounded-lg bg-cyan-500/20 px-3 py-1.5 text-xs text-cyan-700 dark:text-cyan-300 disabled:opacity-40">
          {loading ? t('cover.musicXml.importing') : t('cover.musicXml.import')}
        </button>
        <span className="truncate text-xs text-zinc-500">.musicxml, .xml, .mxl</span>
        <input ref={inputRef} type="file" accept=".musicxml,.xml,.mxl" className="hidden"
          aria-label={t('cover.musicXml.chooseFile')} onChange={event => {
            openFile(event.target.files?.[0]);
            event.target.value = '';
          }} />
        {toggle && <div className="ml-auto">{toggle}</div>}
      </div>
        {action && <div className={wide ? 'flex justify-end' : ''}>{action}</div>}
      </div>
      <div onDragEnter={event => { event.preventDefault(); if (!disabled) setDragging(true); }}
        onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = disabled || loading ? 'none' : 'copy'; }}
        onDragLeave={event => { event.preventDefault(); if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false); }}
        onDrop={event => { event.preventDefault(); setDragging(false); openFile(event.dataTransfer.files[0]); }}
        className={dragging ? 'rounded-lg ring-2 ring-cyan-500' : ''}>
        {children}
      </div>
      {parts.length > 0 && file && (
        <div className="space-y-2 rounded-lg border border-zinc-200 p-2 dark:border-white/10">
          <p className="text-xs text-zinc-700 dark:text-zinc-300">{t('cover.musicXml.choosePart')}</p>
          <StyledSelect value={selectedPart} onChange={id => { setSelectedPart(id); setSelectedStaff('1'); setSelectedVerse(parts.find(p => p.id === id)?.verses[0] ?? ''); }} accent="cyan" size="sm"
            options={parts.map(part => ({ value: part.id,
              label: t('cover.musicXml.part', { name: part.name, count: part.staves,
                staff: t(part.staves === 1 ? 'cover.musicXml.staff' : 'cover.musicXml.staves'),
                lyrics: t(part.hasLyrics ? 'cover.musicXml.yes' : 'cover.musicXml.no') }) }))} />
          {choice && choice.staves > 1 && <>
            <p className="text-xs text-zinc-700 dark:text-zinc-300">{t('cover.musicXml.chooseStaff')}</p>
            <StyledSelect value={selectedStaff} onChange={setSelectedStaff} accent="cyan" size="sm"
              options={Array.from({ length: choice.staves }, (_, i) => ({ value: String(i + 1), label: `${t('cover.musicXml.staff')} ${i + 1}` }))} />
          </>}
          {choice && choice.verses.length > 1 && <>
            <p className="text-xs text-zinc-700 dark:text-zinc-300">{t('cover.musicXml.chooseVerse')}</p>
            <StyledSelect value={selectedVerse} onChange={setSelectedVerse} accent="cyan" size="sm"
              options={choice.verses.map((number, i) => ({ value: number, label: t('cover.musicXml.verse', { number: number || i + 1 }) }))} />
          </>}
          <div className="flex gap-2">
            <button type="button" disabled={disabled || loading || !selectedPart} onClick={() => void convert(file, selectedPart, Number(selectedStaff), selectedVerse)}
              className="rounded-lg bg-cyan-500/20 px-3 py-1.5 text-xs text-cyan-700 dark:text-cyan-300 disabled:opacity-40">{t('cover.musicXml.importPart')}</button>
            <button type="button" onClick={() => { requestRef.current++; setParts([]); setFile(null); setWarnings([]); setError(''); }}
              className="text-xs text-zinc-500">{t('cover.musicXml.cancel')}</button>
          </div>
        </div>
      )}
      {error && <p role="alert" className="text-xs text-red-600 dark:text-red-400">{error}</p>}
      {warnings.length > 0 && <div role="status" className="text-xs text-amber-700 dark:text-amber-300">
        <p>{t('cover.musicXml.checkScore')}</p>
        <ul className="list-disc pl-5">{warnings.map((warning, index) => <li key={`${index}-${warning}`}>{warning}</li>)}</ul>
      </div>}
    </div>
  );
};
