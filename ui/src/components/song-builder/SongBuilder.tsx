// SongBuilder.tsx — Udio-style section-by-section song assembly
//
// Build a song one section at a time. Each section generates N candidate
// variants via the normal generate pipeline (text2music for the first section,
// outpaint-repaint extending the previously chosen variant for every section
// after). The user auditions the variants, picks one, and that pick becomes the
// source for the next section. Sections can also be prepended (e.g. add an intro
// in front of an already-built verse).
//
// The engine already does all the heavy lifting via task_type:'repaint' with an
// out-of-bounds region (negative start = prepend, end > duration = append). This
// studio is the orchestration layer + bookkeeping (builder_projects/sections).

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  Plus, Loader2, Play, Pause, Check, Trash2, Music, Layers,
  ChevronLeft, ArrowRightToLine, ArrowLeftToLine, Sparkles, MapPin,
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useGlobalParamsStore } from '../../context/GlobalParamsContext';
import { generateApi, builderApi } from '../../services/api';
import type { BuilderProject, BuilderSection, BuilderDirection } from '../../services/api';
import { play, playFromList, togglePlay, usePlaybackSelector, songToTrack } from '../../stores/playbackStore';
import type { Song } from '../../types';

// ── Lyrics helpers ───────────────────────────────────────────────────────────

/** Format one section's contribution to the cumulative lyric sheet. */
function fmtBlock(label: string, lyrics: string): string {
  const body = (lyrics || '').trim();
  const tag = label ? `[${label}]` : '';
  if (!body) return tag; // instrumental / structural section → just the tag
  return tag ? `${tag}\n${body}` : body;
}

const SECTION_PRESETS = ['Intro', 'Verse', 'Pre-Chorus', 'Chorus', 'Bridge', 'Outro', 'Solo', 'Instrumental'];

// ── Main studio ──────────────────────────────────────────────────────────────

export const SongBuilder: React.FC = () => {
  const { token } = useAuth();
  const gp = useGlobalParamsStore();

  // Previews route through the global play bar (full scrub/transport for free).
  const playingTrackId = usePlaybackSelector(s => s.currentTrack?.id ?? null);
  const isPlaying = usePlaybackSelector(s => s.isPlaying);
  const playheadTime = usePlaybackSelector(s => s.currentTime);

  // Play a song in the global bar within a navigable list; clicking the already-
  // playing track toggles pause. Returns nothing — state lives in the play store.
  const playSong = useCallback((song: Song | null, list: (Song | null)[]) => {
    if (!song) return;
    if (playingTrackId === song.id) { togglePlay(); return; }
    const tracks = list.filter(Boolean).map(s => songToTrack(s as Song));
    playFromList(songToTrack(song), tracks, 'direct');
  }, [playingTrackId]);

  // Project list / current project
  const [projects, setProjects] = useState<BuilderProject[]>([]);
  const [project, setProject] = useState<BuilderProject | null>(null);
  const [sections, setSections] = useState<BuilderSection[]>([]);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState('');

  // New-project form
  const [newTitle, setNewTitle] = useState('');
  const [newStyle, setNewStyle] = useState('');

  // Composer (next section to generate)
  const [direction, setDirection] = useState<BuilderDirection>('first');
  const [nextLabel, setNextLabel] = useState('Intro');
  const [nextLyrics, setNextLyrics] = useState('');
  const [nextLength, setNextLength] = useState(30);
  // How many seconds of the existing audio the extension overwrites at the seam.
  // The overwritten zone is regenerated as a transition, so the prior section's
  // clean resolution (or song-start, for prepend) is replaced by a continuation.
  // 0 = pure append/prepend (preserve everything, hard seam).
  const [transitionOverlap, setTransitionOverlap] = useState(4);
  // Clip point on the "song so far" where the extension attaches: the extend-from
  // point for append (content after it is regenerated/discarded), or the connect-at
  // point for prepend (content before it is regenerated). null = default (very end
  // for append, very start for prepend). Captured from the global play bar playhead.
  const [clipPoint, setClipPoint] = useState<number | null>(null);
  // Off by default: intermediate sections skip the mastering/PP chain entirely
  // and only run it on the finished track. Opt in for a per-section preview.
  const [previewMastering, setPreviewMastering] = useState(false);

  // Generation tracking
  const [isGenerating, setIsGenerating] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [genProgress, setGenProgress] = useState(0);
  const [genStage, setGenStage] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const showToast = (m: string) => { setToast(m); setTimeout(() => setToast(''), 4000); };

  // ── Derived: head (current full canvas) + the section being auditioned ──
  const chosenSections = useMemo(
    () => sections.filter(s => s.chosen && s.status === 'chosen'),
    [sections],
  );
  const head = useMemo(() => {
    if (!chosenSections.length) return null;
    return chosenSections.reduce((a, b) => (a.created_at >= b.created_at ? a : b));
  }, [chosenSections]);
  const headSong = head?.chosen ?? null;
  const headDuration = headSong?.duration ?? 0;

  // The most recent section that has variants but no pick yet → audition target
  const auditionSection = useMemo(
    () => sections.filter(s => s.status === 'ready').sort((a, b) => b.created_at.localeCompare(a.created_at))[0] ?? null,
    [sections],
  );

  // Timeline = chosen sections in musical (position) order
  const timeline = useMemo(
    () => chosenSections.slice().sort((a, b) => a.position - b.position),
    [chosenSections],
  );

  // ── Load project list on mount ──
  useEffect(() => {
    if (!token) return;
    builderApi.listProjects(token).then(r => setProjects(r.projects)).catch(() => {});
  }, [token]);

  // When no head exists, the only valid direction is 'first'
  useEffect(() => {
    if (!headSong && direction !== 'first') setDirection('first');
    if (headSong && direction === 'first') setDirection('append');
  }, [headSong, direction]);

  // A clip point is only meaningful against the current head + direction — reset
  // it whenever either changes so a stale point can't leak into a new extension.
  useEffect(() => { setClipPoint(null); }, [head?.id, direction]);

  // ── Project open / create / delete ──
  const openProject = useCallback(async (id: string) => {
    if (!token) return;
    setLoading(true);
    try {
      const { project: p, sections: secs } = await builderApi.getProject(id, token);
      setProject(p); setSections(secs); setNewStyle(p.style);
    } catch (e: any) { showToast(`Failed to open: ${e.message}`); }
    finally { setLoading(false); }
  }, [token]);

  const createProject = useCallback(async () => {
    if (!token) return;
    try {
      const { project: p } = await builderApi.createProject(
        { title: newTitle || 'Untitled Song', style: newStyle, variantCount: 4, sectionLength: 30 }, token);
      setProject(p); setSections([]); setNewTitle('');
      setProjects(prev => [p, ...prev]);
    } catch (e: any) { showToast(`Create failed: ${e.message}`); }
  }, [token, newTitle, newStyle]);

  const backToList = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    setProject(null); setSections([]); setIsGenerating(false); setActiveJobId(null);
    if (token) builderApi.listProjects(token).then(r => setProjects(r.projects)).catch(() => {});
  }, [token]);

  const refresh = useCallback(async () => {
    if (!project || !token) return;
    const { project: p, sections: secs } = await builderApi.getProject(project.id, token);
    setProject(p); setSections(secs);
  }, [project, token]);

  // ── Build the cumulative lyric sheet for the section about to be generated ──
  const cumulativeLyrics = useMemo(() => {
    const blocks = timeline.map(s => fmtBlock(s.label, s.chosen?.lyrics ?? s.lyrics));
    const newBlock = fmtBlock(nextLabel, nextLyrics);
    const all = direction === 'prepend' ? [newBlock, ...blocks] : [...blocks, newBlock];
    return all.filter(Boolean).join('\n\n') || '[Instrumental]';
  }, [timeline, nextLabel, nextLyrics, direction]);

  // ── Generate the next section ──
  const handleGenerate = useCallback(async () => {
    if (!token || !project) return;
    if (direction !== 'first' && !headSong) { showToast('Choose a section first to extend from'); return; }

    setIsGenerating(true);
    setGenProgress(0);
    setGenStage('Queued…');
    try {
      const engineParams = gp.getGlobalParams();
      const params: Record<string, any> = {
        ...engineParams,
        customMode: true,
        source: 'builder',
        title: `${project.title} — ${nextLabel || direction}`,
        style: project.style || (engineParams as any).style || '',
        lyrics: cumulativeLyrics,
        batchSize: project.variant_count || 4,
        randomSeed: true,
      };

      // ── Builder pipeline tuning ──────────────────────────────────────────
      // 1. Keep models resident across every variant and every section — no
      //    load/free churn between generations (sets keep_loaded=1 engine-side).
      params.coResident = true;
      // 2. Bypass the whole cosmetic post-processing chain for intermediate
      //    sections — mastering/PP-VAE/spectral/LUFS run only on the finished
      //    track (or when the user opts into a per-section preview). The timbre
      //    reference is loaded in the synth phase regardless, so it still feeds
      //    the DiT as generation conditioning.
      if (!previewMastering) {
        params.postProcessingEnabled = false;
        params.masteringEnabled = false;
        params.ppVaeReencode = false;
        params.spectralLifterEnabled = false;
        params.lufsEnabled = false;
      }
      // These enrichment steps are never useful mid-build — always skip them.
      params.coverArtEnabled = false;
      params.parallelCoverArt = false;
      params.whisperLyricsEnabled = false;
      params.qualityEvalEnabled = false;
      params.parallelQualityEval = false;
      params.autoTrimEnabled = false;
      params.skipLrc = true;
      if (project.bpm) params.bpm = project.bpm;
      if (project.key_scale) params.keyScale = project.key_scale;
      if (project.time_signature) params.timeSignature = project.time_signature;
      if (project.vocal_language) params.vocalLanguage = project.vocal_language;

      if (direction === 'first') {
        params.duration = nextLength;
      } else {
        // Overwrite `overlap` seconds at the seam so the prior section's clean
        // resolution (append) or song-start (prepend) is regenerated as a
        // transition rather than preserved verbatim. Clamp so we never overwrite
        // the whole source or more than the new section is long.
        const overlap = Math.max(0, Math.min(transitionOverlap, headDuration - 1, nextLength));
        params.taskType = 'repaint';
        params.duration = 0; // engine derives from source canvas
        if (direction === 'prepend') {
          // connect-at point: content before it (the unwanted head) is regenerated
          // as part of the intro lead-in. Default 0 = overwrite only the seam.
          const at = clipPoint ?? 0;
          params.repaintingStart = -nextLength;          // pad/generate before the song
          params.repaintingEnd = at + overlap;           // overwrite [0, at+overlap]
        } else {
          // extend-from point: content after it (the unwanted tail) is fully inside
          // the regenerated region and gets discarded. Default = the very end.
          const from = clipPoint ?? headDuration;
          params.repaintingStart = from - overlap;       // overwrite back from the attach point
          // Ensure the whole tail past `from` is regenerated (never preserve a
          // sliver of old tail beyond the new content when the user trims).
          params.repaintingEnd = Math.max(from + nextLength, headDuration);
        }
        params.repaintInjectionRatio = 0.5;
        params.repaintCrossfadeFrames = 10;
        const latent = (headSong as any).latentUrl || (headSong as any).latent_url;
        const audio = (headSong as any).audioUrl || (headSong as any).audio_url;
        if (latent) params.sourceLatentUrl = latent;
        if (audio) params.sourceAudioUrl = audio;
      }

      const { jobId } = await generateApi.submit(params as any, token);
      setActiveJobId(jobId);

      // Record the section immediately so a refresh/reload survives the in-flight job.
      const { section } = await builderApi.createSection(project.id, {
        label: nextLabel,
        lyrics: nextLyrics,
        direction,
        sectionLength: nextLength,
        jobId,
        status: 'generating',
      }, token);
      setSections(prev => [...prev, section]);

      pollJob(jobId, section.id);
    } catch (e: any) {
      showToast(`Generation failed: ${e.message}`);
      setIsGenerating(false);
    }
  }, [token, project, direction, headSong, headDuration, gp, nextLabel, nextLyrics, nextLength, cumulativeLyrics, previewMastering, transitionOverlap, clipPoint]);

  const pollJob = useCallback((jobId: string, sectionId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const s = await generateApi.status(jobId);
        const rawProg = (s as any).progress;
        if (rawProg != null) setGenProgress(Math.min(100, Math.round(rawProg > 1 ? rawProg : rawProg * 100)));
        if ((s as any).stage) setGenStage((s as any).stage);

        if (s.status === 'succeeded') {
          clearInterval(pollRef.current!);
          const songIds: string[] = (s as any).result?.songIds || [];
          await builderApi.updateSection(sectionId, { candidateSongIds: songIds, status: 'ready' }, token!);
          setIsGenerating(false);
          setActiveJobId(null);
          setGenProgress(100);
          setGenStage('Pick a variant');
          await refresh();
        } else if (s.status === 'failed') {
          clearInterval(pollRef.current!);
          await builderApi.updateSection(sectionId, { status: 'failed' }, token!).catch(() => {});
          setIsGenerating(false);
          setActiveJobId(null);
          setGenStage('');
          showToast(`Failed: ${(s as any).error || 'Unknown error'}`);
          await refresh();
        }
      } catch { /* transient poll error — keep polling */ }
    }, 2000);
  }, [token, refresh]);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  // ── Choose a variant for a section ──
  const chooseVariant = useCallback(async (sectionId: string, songId: string) => {
    if (!token) return;
    try {
      await builderApi.updateSection(sectionId, { chosenSongId: songId, status: 'chosen' }, token);
      // Advance composer: default to appending the next section.
      setDirection('append');
      setNextLabel('Verse');
      setNextLyrics('');
      await refresh();
    } catch (e: any) { showToast(`Could not select: ${e.message}`); }
  }, [token, refresh]);

  const deleteSection = useCallback(async (sectionId: string) => {
    if (!token) return;
    await builderApi.deleteSection(sectionId, token).catch(() => {});
    await refresh();
  }, [token, refresh]);

  const cancelGen = useCallback(async () => {
    if (activeJobId) { try { await generateApi.cancel(activeJobId); } catch {} }
    if (pollRef.current) clearInterval(pollRef.current);
    setIsGenerating(false); setActiveJobId(null); setGenProgress(0); setGenStage('');
  }, [activeJobId]);

  // ── Render: project list (no project open) ──
  if (!project) {
    return (
      <div className="flex flex-col w-full h-full bg-zinc-50 dark:bg-suno overflow-y-auto p-6">
        {toast && <Toast msg={toast} />}
        <div className="max-w-3xl mx-auto w-full">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-violet-500/15 flex items-center justify-center">
              <Layers size={20} className="text-violet-400" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Song Builder</h1>
              <p className="text-xs text-zinc-500">Build a song section by section — generate options, pick the best, extend.</p>
            </div>
          </div>

          {/* New project */}
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 mb-6">
            <h2 className="text-sm font-semibold text-white mb-3">New song</h2>
            <div className="space-y-2">
              <input
                value={newTitle} onChange={e => setNewTitle(e.target.value)}
                placeholder="Song title"
                className="w-full px-3 py-2 rounded-lg bg-black/20 border border-white/10 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500"
              />
              <input
                value={newStyle} onChange={e => setNewStyle(e.target.value)}
                placeholder="Style / caption (e.g. dreamy synthpop, 90 bpm, female vocals)"
                className="w-full px-3 py-2 rounded-lg bg-black/20 border border-white/10 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500"
              />
              <button
                onClick={createProject}
                className="w-full px-3 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium flex items-center justify-center gap-2 transition-colors"
              >
                <Plus size={16} /> Start building
              </button>
            </div>
          </div>

          {/* Existing projects */}
          <h2 className="text-sm font-semibold text-zinc-400 mb-2">Your songs</h2>
          {projects.length === 0 ? (
            <p className="text-xs text-zinc-600 py-6 text-center">No songs yet — start one above.</p>
          ) : (
            <div className="space-y-2">
              {projects.map(p => (
                <button
                  key={p.id} onClick={() => openProject(p.id)}
                  className="w-full text-left px-4 py-3 rounded-xl bg-white/[0.03] border border-white/5 hover:border-violet-500/30 hover:bg-white/[0.05] transition-colors flex items-center gap-3"
                >
                  <Music size={16} className="text-violet-400 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-white truncate">{p.title}</div>
                    <div className="text-[11px] text-zinc-500 truncate">{p.style || 'No style set'}</div>
                  </div>
                  <span className="text-[11px] text-zinc-600">{p.section_count ?? 0} section{(p.section_count ?? 0) === 1 ? '' : 's'}</span>
                  <Trash2
                    size={14} className="text-zinc-600 hover:text-red-400"
                    onClick={async (e) => { e.stopPropagation(); if (token) { await builderApi.deleteProject(p.id, token); setProjects(prev => prev.filter(x => x.id !== p.id)); } }}
                  />
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── Render: open project ──
  const canPrepend = !!headSong;
  return (
    <div className="flex flex-col w-full h-full bg-zinc-50 dark:bg-suno overflow-hidden">
      {toast && <Toast msg={toast} />}

      {/* Header */}
      <div className="flex-shrink-0 flex items-center gap-3 px-5 py-3 border-b border-white/5">
        <button onClick={backToList} className="text-zinc-500 hover:text-white"><ChevronLeft size={18} /></button>
        <Layers size={18} className="text-violet-400" />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-white truncate">{project.title}</div>
          <div className="text-[11px] text-zinc-500 truncate">{project.style || 'No style set'}</div>
        </div>
        {headSong && (
          <div className="text-[11px] text-zinc-400">
            Song so far: <span className="text-violet-300 font-medium">{Math.round(headDuration)}s</span> · {timeline.length} section{timeline.length === 1 ? '' : 's'}
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-5 space-y-6">
        {/* ── Timeline of committed sections ── */}
        <div>
          <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wide mb-2">Timeline</h3>
          {timeline.length === 0 ? (
            <p className="text-xs text-zinc-600 py-4">No sections yet. Generate your first section below.</p>
          ) : (
            <div className="flex items-stretch gap-2 overflow-x-auto pb-2">
              {timeline.map((s, i) => {
                const isHead = s.id === head?.id;
                const isThis = playingTrackId === s.chosen?.id;
                return (
                  <div key={s.id}
                    className={`flex-shrink-0 w-40 rounded-xl border p-3 ${isHead ? 'border-violet-500/50 bg-violet-500/10' : 'border-white/10 bg-white/[0.03]'}`}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] uppercase tracking-wide text-zinc-500">#{i + 1}{isHead ? ' · head' : ''}</span>
                      <Trash2 size={12} className="text-zinc-600 hover:text-red-400 cursor-pointer" onClick={() => deleteSection(s.id)} />
                    </div>
                    <div className="text-sm text-white font-medium truncate">{s.label || 'Section'}</div>
                    <div className="text-[11px] text-zinc-500 mb-2">{Math.round(s.chosen?.duration || 0)}s · {s.direction}</div>
                    <button
                      onClick={() => playSong(s.chosen, timeline.map(t => t.chosen))}
                      className="w-full px-2 py-1.5 rounded-lg bg-black/20 hover:bg-black/30 text-xs text-white flex items-center justify-center gap-1.5"
                    >
                      {isThis && isPlaying ? <Pause size={12} /> : <Play size={12} />}
                      {isThis && isPlaying ? 'Pause' : 'Play'}
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* ── Audition: variants awaiting a pick ── */}
        {auditionSection && (
          <div className="rounded-2xl border border-violet-500/30 bg-violet-500/[0.06] p-4">
            <h3 className="text-sm font-semibold text-white mb-1 flex items-center gap-2">
              <Sparkles size={15} className="text-violet-400" /> Pick a variant for “{auditionSection.label || 'section'}”
            </h3>
            <p className="text-[11px] text-zinc-500 mb-3">All variants share the same prior song and differ only in this new section.</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {auditionSection.candidates.map((c, i) => {
                const isThis = playingTrackId === c.id;
                return (
                  <div key={c.id} className="rounded-xl border border-white/10 bg-white/[0.03] p-3 flex items-center gap-3">
                    <button onClick={() => playSong(c, auditionSection.candidates)} className="w-9 h-9 rounded-full bg-violet-600 hover:bg-violet-500 flex items-center justify-center flex-shrink-0" title="Play in the bar below — scrub to audition">
                      {isThis && isPlaying ? <Pause size={15} className="text-white" /> : <Play size={15} className="text-white" />}
                    </button>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-white">Option {i + 1}</div>
                      <div className="text-[11px] text-zinc-500">{Math.round(c.duration || 0)}s</div>
                    </div>
                    <button
                      onClick={() => chooseVariant(auditionSection.id, c.id)}
                      className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium flex items-center gap-1.5"
                    >
                      <Check size={13} /> Use
                    </button>
                  </div>
                );
              })}
            </div>
            <button onClick={() => deleteSection(auditionSection.id)} className="mt-3 text-[11px] text-zinc-500 hover:text-red-400">Discard these variants</button>
          </div>
        )}

        {/* ── Composer: generate the next section ── */}
        {!auditionSection && (
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
            <h3 className="text-sm font-semibold text-white mb-3">
              {direction === 'first' ? 'Generate first section' : direction === 'prepend' ? 'Prepend a section' : 'Extend with next section'}
            </h3>

            {/* Direction */}
            {headSong && (
              <div className="flex gap-2 mb-3">
                <DirBtn active={direction === 'append'} onClick={() => setDirection('append')} icon={<ArrowRightToLine size={13} />} label="Append" />
                <DirBtn active={direction === 'prepend'} onClick={() => setDirection('prepend')} icon={<ArrowLeftToLine size={13} />} label="Prepend" disabled={!canPrepend} />
              </div>
            )}

            {/* Label presets */}
            <div className="flex flex-wrap gap-1.5 mb-3">
              {SECTION_PRESETS.map(p => (
                <button key={p} onClick={() => setNextLabel(p)}
                  className={`px-2.5 py-1 rounded-full text-[11px] ${nextLabel === p ? 'bg-violet-600 text-white' : 'bg-white/5 text-zinc-400 hover:text-white'}`}>
                  {p}
                </button>
              ))}
            </div>

            <input
              value={nextLabel} onChange={e => setNextLabel(e.target.value)}
              placeholder="Section label (e.g. Chorus)"
              className="w-full px-3 py-2 mb-2 rounded-lg bg-black/20 border border-white/10 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500"
            />
            <textarea
              value={nextLyrics} onChange={e => setNextLyrics(e.target.value)}
              placeholder="Lyrics for this section (leave empty for instrumental)"
              rows={4}
              className="w-full px-3 py-2 mb-2 rounded-lg bg-black/20 border border-white/10 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-violet-500 resize-y"
            />

            <div className="flex items-center gap-3 mb-3">
              <label className="text-xs text-zinc-400">Length</label>
              <input type="range" min={5} max={60} step={1} value={nextLength} onChange={e => setNextLength(Number(e.target.value))} className="flex-1 accent-violet-500" />
              <span className="text-xs text-violet-300 w-10 text-right">{nextLength}s</span>
            </div>

            {direction !== 'first' && (
              <div className="flex items-center gap-3 mb-1" title="Seconds of the existing audio the extension overwrites at the seam, regenerated as a transition. Higher = smoother blend but replaces more of the neighbouring section. 0 = hard seam.">
                <label className="text-xs text-zinc-400 whitespace-nowrap">Transition blend</label>
                <input type="range" min={0} max={10} step={1} value={transitionOverlap} onChange={e => setTransitionOverlap(Number(e.target.value))} className="flex-1 accent-violet-500" />
                <span className="text-xs text-violet-300 w-10 text-right">{transitionOverlap}s</span>
              </div>
            )}

            {/* Clip point — set where the extension attaches by scrubbing the global bar */}
            {direction !== 'first' && headSong && (
              <div className="mb-3 rounded-lg border border-white/10 bg-black/20 p-2.5">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] text-zinc-400">
                    {direction === 'prepend' ? 'Connect intro at' : 'Extend from'}:{' '}
                    <span className="text-violet-300 font-mono">{(clipPoint ?? (direction === 'prepend' ? 0 : headDuration)).toFixed(1)}s</span>
                    {clipPoint == null && <span className="text-zinc-600"> (default {direction === 'prepend' ? 'start' : 'end'})</span>}
                  </span>
                  {clipPoint != null && (
                    <button onClick={() => setClipPoint(null)} className="text-[11px] text-zinc-500 hover:text-zinc-300">Reset</button>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => play(songToTrack(headSong))}
                    className="px-2.5 py-1 rounded-lg bg-white/5 hover:bg-white/10 text-[11px] text-white flex items-center gap-1.5"
                  >
                    <Play size={11} /> Load song so far
                  </button>
                  <button
                    onClick={() => setClipPoint(Math.max(0, Math.min(playheadTime, headDuration)))}
                    disabled={playingTrackId !== headSong.id}
                    className="px-2.5 py-1 rounded-lg bg-violet-600/80 hover:bg-violet-500 disabled:opacity-40 text-[11px] text-white flex items-center gap-1.5"
                    title="Scrub the play bar below to the spot, then capture it here"
                  >
                    <MapPin size={11} /> Set to playhead{playingTrackId === headSong.id ? ` (${playheadTime.toFixed(1)}s)` : ''}
                  </button>
                </div>
                <p className="text-[10px] text-zinc-600 mt-1.5">
                  {direction === 'prepend'
                    ? 'Scrub the bar to where the intro should meet the song — anything before it becomes lead-in.'
                    : 'Scrub the bar to where the next section should start — anything after it is replaced.'}
                </p>
              </div>
            )}

            <label className="flex items-center gap-2 mb-3 text-[11px] text-zinc-400 cursor-pointer select-none">
              <input type="checkbox" checked={previewMastering} onChange={e => setPreviewMastering(e.target.checked)} className="accent-violet-500" />
              Apply mastering/post-processing to these variants (preview only — off by default for speed; runs only on the finished track)
            </label>

            {/* Generate */}
            {isGenerating ? (
              <div className="space-y-2">
                <div className="h-2 rounded-full bg-black/30 overflow-hidden">
                  <div className="h-full bg-violet-500 transition-all" style={{ width: `${genProgress}%` }} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-zinc-500">{genStage || 'Generating…'}</span>
                  <button onClick={cancelGen} className="text-[11px] text-zinc-500 hover:text-red-400">Cancel</button>
                </div>
              </div>
            ) : (
              <button
                onClick={handleGenerate}
                className="w-full px-3 py-2.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium flex items-center justify-center gap-2 transition-colors"
              >
                <Sparkles size={16} /> Generate {project.variant_count || 4} variant{(project.variant_count || 4) === 1 ? '' : 's'}
              </button>
            )}

            {/* Cumulative lyric preview */}
            {direction !== 'first' && (
              <details className="mt-3">
                <summary className="text-[11px] text-zinc-500 cursor-pointer hover:text-zinc-300">Preview cumulative lyrics sent to the model</summary>
                <pre className="mt-2 p-2 rounded-lg bg-black/30 text-[11px] text-zinc-400 whitespace-pre-wrap max-h-40 overflow-y-auto">{cumulativeLyrics}</pre>
              </details>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// ── Small presentational helpers ─────────────────────────────────────────────

const Toast: React.FC<{ msg: string }> = ({ msg }) => (
  <div className="absolute top-16 right-6 z-50 px-4 py-2 rounded-xl bg-zinc-900 text-white text-sm shadow-xl border border-white/10">
    {msg}
  </div>
);

const DirBtn: React.FC<{ active: boolean; onClick: () => void; icon: React.ReactNode; label: string; disabled?: boolean }> =
  ({ active, onClick, icon, label, disabled }) => (
    <button onClick={onClick} disabled={disabled}
      className={`flex-1 px-3 py-1.5 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition-colors disabled:opacity-40 ${active ? 'bg-violet-600 text-white' : 'bg-white/5 text-zinc-400 hover:text-white'}`}>
      {icon} {label}
    </button>
  );

export default SongBuilder;
