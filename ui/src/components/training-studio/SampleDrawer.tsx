// SampleDrawer.tsx — full caption / lyrics editor with audio preview
//
// Same optimistic + debounced edit path as the grid cells. The <audio>
// element is local to the drawer: preview never touches the global playback
// store because these files are not library songs.

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, Headphones, Loader2, X, XCircle } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { getSampleMm3, getSampleYue2, sampleAudioUrl, saveSampleMm3, saveSampleYue2 } from '../../services/trainingApi';
import { mergedSample, useTrainingStore } from '../../stores/trainingStore';
import { AuditionPlayer } from './AuditionPlayer';
import { ParamLabel } from '../shared/ParamLabel';
import { Toggle } from '../shared/Toggle';

interface SampleDrawerProps {
  sampleId: string;
}

export const SampleDrawer: React.FC<SampleDrawerProps> = ({ sampleId }) => {
  const { t } = useTranslation();
  const datasetId = useTrainingStore(s => s.selectedDatasetId);
  const sampleRaw = useTrainingStore(s => s.samplesById[sampleId]);
  const pending = useTrainingStore(s => s.pendingEdits[sampleId]);
  const sampleOrder = useTrainingStore(s => s.sampleOrder);
  const saving = useTrainingStore(s => s.savingSampleIds.has(sampleId));
  const editSample = useTrainingStore(s => s.editSample);
  const flushSample = useTrainingStore(s => s.flushSample);
  const revertSampleField = useTrainingStore(s => s.revertSampleField);
  const setOpenSampleId = useTrainingStore(s => s.setOpenSampleId);
  const auditionSample = useTrainingStore(s => s.auditionSample);
  const preview = useTrainingStore(s => s.samplePreviews[sampleId]);
  const previewSource = useTrainingStore(s => s.samplePreviewSources[sampleId]);
  const previewPending = useTrainingStore(s => !!s.samplePreviewPending[sampleId]);
  const previewError = useTrainingStore(s => s.samplePreviewErrors[sampleId]);

  const sample = useMemo(() => mergedSample(sampleRaw, pending), [sampleRaw, pending]);
  const idx = sampleOrder.indexOf(sampleId);

  // MM3 Structured Caption — a separate file (<stem>.mm3.txt), not a sidecar
  // field, so it does not ride the store's optimistic edit path. Fetched when
  // the drawer opens on a sample; saved on blur; Escape reverts to last saved.
  const [mm3, setMm3] = useState<string | null>(null);
  const [mm3Error, setMm3Error] = useState<string | null>(null);
  const mm3SavedRef = useRef('');
  useEffect(() => {
    let stale = false;
    setMm3(null);
    setMm3Error(null);
    if (!datasetId || !sampleId) return;
    getSampleMm3(datasetId, sampleId)
      .then(({ text }) => { if (!stale) { mm3SavedRef.current = text; setMm3(text); } })
      .catch((err: Error) => { if (!stale) { setMm3(''); setMm3Error(err.message); } });
    return () => { stale = true; };
  }, [datasetId, sampleId]);
  const flushMm3 = () => {
    if (!datasetId || mm3 === null || mm3 === mm3SavedRef.current) return;
    const text = mm3;
    saveSampleMm3(datasetId, sampleId, text)
      .then(() => { mm3SavedRef.current = text; setMm3Error(null); })
      .catch((err: Error) => setMm3Error(err.message));
  };

  // YuE2 planner caption — the third format, same file-not-field arrangement
  // as MM3 (<stem>.yue2.txt), so the same fetch-on-open / save-on-blur path.
  const [yue2, setYue2] = useState<string | null>(null);
  const [yue2Error, setYue2Error] = useState<string | null>(null);
  const yue2SavedRef = useRef('');
  useEffect(() => {
    let stale = false;
    setYue2(null);
    setYue2Error(null);
    if (!datasetId || !sampleId) return;
    getSampleYue2(datasetId, sampleId)
      .then(({ text }) => { if (!stale) { yue2SavedRef.current = text; setYue2(text); } })
      .catch((err: Error) => { if (!stale) { setYue2(''); setYue2Error(err.message); } });
    return () => { stale = true; };
  }, [datasetId, sampleId]);
  const flushYue2 = () => {
    if (!datasetId || yue2 === null || yue2 === yue2SavedRef.current) return;
    const text = yue2;
    saveSampleYue2(datasetId, sampleId, text)
      .then(() => { yue2SavedRef.current = text; setYue2Error(null); })
      .catch((err: Error) => setYue2Error(err.message));
  };

  if (!sample || !datasetId) return null;

  const readOnly = sample.labelStatus === 'processing' || sample.fileMissing;
  const close = () => { void flushSample(sampleId); flushMm3(); flushYue2(); setOpenSampleId(null); };
  const goto = (delta: number) => {
    const next = sampleOrder[idx + delta];
    if (!next) return;
    void flushSample(sampleId);
    flushMm3();
    flushYue2();
    setOpenSampleId(next);
  };

  const area = `w-full rounded-lg px-3 py-2 text-sm bg-zinc-100 dark:bg-black/20 border border-zinc-300 dark:border-white/10 text-zinc-800 dark:text-zinc-200 placeholder-zinc-400 dark:placeholder-zinc-600 focus:outline-none focus:border-amber-500 disabled:opacity-50`;

  return (
    <div className="fixed inset-0 z-[140] flex justify-end">
      <div className="absolute inset-0 bg-black/30 dark:bg-black/60 backdrop-blur-sm" onClick={close} />
      <div className="relative w-[560px] max-w-full h-full overflow-y-auto bg-white dark:bg-suno-card border-l border-zinc-200 dark:border-white/10 shadow-2xl flex flex-col gap-4 p-5">
        {/* Header */}
        <div className="flex items-center gap-2">
          <div className="flex-1 min-w-0">
            <div className="text-sm font-bold text-zinc-900 dark:text-white truncate" title={sample.relPath}>{sample.filename}</div>
            <div className="text-[11px] font-mono text-zinc-500 truncate">{sample.relPath}</div>
          </div>
          {saving && <Loader2 size={14} className="animate-spin text-amber-500" />}
          <button
            onClick={() => goto(-1)}
            disabled={idx <= 0}
            title={t('trainingStudio.drawer.prev')}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-800 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-30 transition-colors"
          >
            <ChevronLeft size={16} />
          </button>
          <button
            onClick={() => goto(1)}
            disabled={idx < 0 || idx >= sampleOrder.length - 1}
            title={t('trainingStudio.drawer.next')}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-800 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10 disabled:opacity-30 transition-colors"
          >
            <ChevronRight size={16} />
          </button>
          <button
            onClick={close}
            title={t('trainingStudio.drawer.close')}
            className="p-1.5 rounded-lg text-zinc-500 hover:text-zinc-800 dark:hover:text-white hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Preview */}
        {!sample.fileMissing && (
          <audio src={sampleAudioUrl(datasetId, sampleId)} controls className="w-full h-9" preload="none" />
        )}

        {/* Codes audition — the plan next to the truth, two players, one panel.
            The button is NOT gated on `hasAudioCodes`: that flag is the legacy
            /understand payload only, and the primary codes source is the
            variant's lm_codes.jsonl, which the browser cannot see. The server
            owns the precedence and answers 409 when neither exists. */}
        <div className="flex flex-col gap-2">
          <button
            onClick={() => void auditionSample(sampleId)}
            disabled={previewPending}
            className="flex items-center gap-2 w-fit px-3 py-1.5 rounded-lg text-[11px] font-semibold bg-amber-500/10 border border-amber-500/25 text-amber-600 dark:text-amber-400 hover:bg-amber-500/20 disabled:opacity-40 transition-colors"
          >
            {previewPending
              ? <Loader2 size={12} className="animate-spin" />
              : <Headphones size={12} />}
            {t('trainingStudio.audition.sampleButton')}
          </button>

          {previewError && (
            <div className="flex items-start gap-2 px-3 py-2 rounded-lg border border-red-500/25 bg-red-500/10 text-[11px] text-red-500 dark:text-red-400">
              <XCircle size={12} className="mt-0.5 flex-shrink-0" />
              <span className="min-w-0 break-words">{previewError}</span>
            </div>
          )}

          {preview?.sides[0] && (
            <div className="flex flex-col gap-1">
              <span className="text-[10px] text-zinc-500">
                {t(previewSource === 'label'
                  ? 'trainingStudio.audition.source.label'
                  : 'trainingStudio.audition.source.lmCodes')}
              </span>
              <AuditionPlayer side={preview.sides[0]} />
            </div>
          )}
        </div>

        {/* Caption */}
        <label className="flex flex-col gap-1.5">
          <ParamLabel
            label={t('trainingStudio.drawer.caption')}
            info={t('trainingStudio.drawer.captionInfo', 'The style description this sample trains with, same text as the caption column in the grid. Blank falls back to the filename when the dataset is built. Saves automatically a moment after you stop typing, or right away when you leave the box.')}
            className="text-xs font-semibold text-zinc-600 dark:text-zinc-400"
          />
          <textarea
            rows={5}
            value={sample.caption}
            disabled={readOnly}
            placeholder={t('trainingStudio.drawer.captionPlaceholder')}
            onChange={(e) => void editSample(sampleId, { caption: e.target.value })}
            onBlur={() => void flushSample(sampleId)}
            onKeyDown={(e) => { if (e.key === 'Escape') revertSampleField(sampleId, 'caption'); }}
            className={area}
          />
        </label>

        {/* MM3 Structured Caption — the second of the two caption formats a
            MOSS labeling run writes. Shown beside the AS1.5 caption so the two
            can be compared and edited without leaving the app. */}
        <label className="flex flex-col gap-1.5">
          <ParamLabel
            label={t('trainingStudio.drawer.mm3Caption')}
            info={t('trainingStudio.drawer.mm3CaptionInfo', "MiniMax-Music3's own caption format (Global Metadata / Vocal Details / Arrangement), stored in a separate <stem>.mm3.txt file next to the audio rather than in the dataset row. Only an MM3 planner adapter trains on it. Saves when you leave the box; Escape reverts to the last saved text.")}
            className="text-xs font-semibold text-zinc-600 dark:text-zinc-400"
          />
          {mm3 === null ? (
            <div className="flex items-center gap-2 px-3 py-2 text-[11px] text-zinc-500">
              <Loader2 size={12} className="animate-spin" />
            </div>
          ) : (
            <>
              {mm3 === '' && mm3SavedRef.current === '' && !mm3Error && (
                <span className="text-[11px] text-zinc-500">{t('trainingStudio.drawer.mm3Missing')}</span>
              )}
              <textarea
                rows={10}
                value={mm3}
                disabled={readOnly}
                placeholder={t('trainingStudio.drawer.mm3Placeholder')}
                onChange={(e) => setMm3(e.target.value)}
                onBlur={flushMm3}
                onKeyDown={(e) => { if (e.key === 'Escape') setMm3(mm3SavedRef.current); }}
                className={`${area} font-mono text-xs leading-relaxed`}
              />
              {mm3Error && (
                <span className="text-[11px] text-red-500 dark:text-red-400 break-words">{mm3Error}</span>
              )}
            </>
          )}
        </label>

        {/* YuE2 planner caption — one sentence, fixed part order. Written in the
            same labeling pass as the other two (enhanceService), editable here
            because the planner is prompted with exactly this text. */}
        <label className="flex flex-col gap-1.5">
          <ParamLabel
            label={t('trainingStudio.drawer.yue2Caption', 'YuE2 caption')}
            info={t('trainingStudio.drawer.yue2CaptionInfo', 'The one-sentence caption YuE2’s planner is prompted with directly, stored in <stem>.yue2.txt next to the audio. Written by the same labeling pass as the other two captions; editing it here changes exactly what the planner sees during training. Saves when you leave the box; Escape reverts to the last saved text.')}
            className="text-xs font-semibold text-zinc-600 dark:text-zinc-400"
          />
          {yue2 === null ? (
            <div className="flex items-center gap-2 px-3 py-2 text-[11px] text-zinc-500">
              <Loader2 size={12} className="animate-spin" />
            </div>
          ) : (
            <>
              {yue2 === '' && yue2SavedRef.current === '' && !yue2Error && (
                <span className="text-[11px] text-zinc-500">{t('trainingStudio.drawer.yue2Missing', 'No YuE2 caption yet — run the caption step (or Enhance → YuE2 caption) to write one.')}</span>
              )}
              <textarea
                rows={4}
                value={yue2}
                disabled={readOnly}
                placeholder={t('trainingStudio.drawer.yue2Placeholder', 'One sentence: language, genre, vocal, instruments, mood, production, BPM…')}
                onChange={(e) => setYue2(e.target.value)}
                onBlur={flushYue2}
                onKeyDown={(e) => { if (e.key === 'Escape') setYue2(yue2SavedRef.current); }}
                className={`${area} font-mono text-xs leading-relaxed`}
              />
              {yue2Error && (
                <span className="text-[11px] text-red-500 dark:text-red-400 break-words">{yue2Error}</span>
              )}
            </>
          )}
        </label>

        {/* Instrumental */}
        <Toggle
          accent="amber"
          checked={sample.isInstrumental}
          disabled={readOnly}
          onChange={(checked) => { void editSample(sampleId, { isInstrumental: checked }); void flushSample(sampleId); }}
          label={t('trainingStudio.drawer.instrumental')}
          info={t('trainingStudio.drawer.instrumentalInfo', 'Marks this sample as having no vocals, so training treats it as instrumental and skips the lyrics text. Off: the sample trains with its lyrics as written.')}
        />

        {/* Lyrics */}
        <label className="flex flex-col gap-1.5 flex-1">
          <ParamLabel
            label={t('trainingStudio.drawer.lyrics')}
            info={t('trainingStudio.drawer.lyricsInfo', 'Lyrics text this sample trains with, alongside its caption. Leave blank for instrumental tracks; use the Instrumental toggle above rather than an empty box, so the sample is flagged correctly everywhere else it is used.')}
            className="text-xs font-semibold text-zinc-600 dark:text-zinc-400"
          />
          <textarea
            rows={16}
            value={sample.lyrics}
            disabled={readOnly}
            placeholder={t('trainingStudio.drawer.lyricsPlaceholder')}
            onChange={(e) => void editSample(sampleId, { lyrics: e.target.value })}
            onBlur={() => void flushSample(sampleId)}
            onKeyDown={(e) => { if (e.key === 'Escape') revertSampleField(sampleId, 'lyrics'); }}
            className={`${area} font-mono text-xs leading-relaxed`}
          />
        </label>

        {/* Meta strip */}
        <div className="flex items-center gap-3 flex-wrap text-[11px] text-zinc-500">
          {sample.bpm != null && <span>{sample.bpm} BPM</span>}
          {sample.key && <span>{sample.key}</span>}
          {sample.signature && <span>{sample.signature}</span>}
          {sample.language && <span>{sample.language}</span>}
          {sample.labeledAt && <span>{t('trainingStudio.drawer.saved')} · {sample.labeledAt.slice(0, 10)}</span>}
        </div>
      </div>
    </div>
  );
};

export default SampleDrawer;
