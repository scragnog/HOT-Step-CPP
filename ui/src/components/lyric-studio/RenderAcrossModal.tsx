/**
 * RenderAcrossModal.tsx — render a written song under ANOTHER album.
 *
 * A render takes all of its settings from one album: the preset (ACE, LM,
 * MM3 and YuE2 adapters, reference track) and the caption source. The song
 * only brings its lyrics, captions, tempo and key. So both directions here
 * come down to the same pair: a song, and the album it renders AS.
 *
 *   as    the song is fixed (a card in this album); pick the target album.
 *   from  the target is fixed (this album); pick a song from another album.
 *
 * The recording is listed under the target album, not the song's own.
 */

import React, { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { X, Shuffle, Play, Send } from 'lucide-react';
import { lireekApi } from '../../services/lireekApi';
import type { Generation, LyricsSet, AlbumPreset } from '../../services/lireekApi';
import { StyledSelect } from '../shared/StyledSelect';
import { ParamLabel } from '../shared/ParamLabel';
import { useDisguiseMode } from '../../hooks/useDisguiseMode';

export interface RenderTarget {
  lyricsSetId: number;
  artistId: number;
  artistName: string;
  artistImageUrl?: string;
}

interface RenderAcrossModalProps {
  /** 'as' needs `gen`; 'from' needs `target`. */
  mode: 'as' | 'from' | null;
  gen?: Generation | null;
  target?: RenderTarget | null;
  /** The album being viewed: left out of the album lists. */
  currentLyricsSetId: number;
  onClose: () => void;
  onRender: (gen: Generation, target: RenderTarget) => void;
  onSendToCreate?: (gen: Generation, target: RenderTarget) => void;
}

function presetHint(p: AlbumPreset | undefined): string {
  if (!p) return 'No album preset: base models';
  const parts: string[] = [];
  if (p.adapter_path) parts.push('ACE adapter');
  if (p.lm_adapter_path) parts.push('LM adapter');
  if (p.mm3_adapter_path) parts.push('MM3 adapter');
  if (p.yue2_ar_adapter_path || p.yue2_nar_adapter_path) parts.push('YuE2 adapter');
  if (p.reference_track_path) parts.push('reference track');
  return parts.length ? parts.join(' · ') : 'No adapters set';
}

export const RenderAcrossModal: React.FC<RenderAcrossModalProps> = ({
  mode, gen, target, currentLyricsSetId, onClose, onRender, onSendToCreate,
}) => {
  const { t } = useTranslation();
  const { disguiseArtist } = useDisguiseMode();
  const [albums, setAlbums] = useState<LyricsSet[]>([]);
  const [presets, setPresets] = useState<Map<number, AlbumPreset>>(new Map());
  const [allGens, setAllGens] = useState<Generation[]>([]);
  const [albumId, setAlbumId] = useState<number | ''>('');
  const [genId, setGenId] = useState<number | ''>('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!mode) return;
    setAlbumId(''); setGenId('');
    let live = true;
    setLoading(true);
    Promise.all([
      lireekApi.listLyricsSets(),
      lireekApi.listAllPresets().catch(() => ({ presets: [] as AlbumPreset[] })),
      mode === 'from' ? lireekApi.listAllGenerations() : Promise.resolve(null),
    ]).then(([ls, pr, gens]) => {
      if (!live) return;
      setAlbums(ls.lyrics_sets);
      setPresets(new Map(pr.presets.map(p => [p.lyrics_set_id, p])));
      // Server hands back a raw array; the declared type says { generations }.
      if (gens) setAllGens(Array.isArray(gens) ? gens : (gens.generations || []));
    }).catch(() => {}).finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [mode]);

  const albumLabel = (a: LyricsSet) => `${disguiseArtist(a.artist_name)} — ${a.album || 'Top Songs'}`;

  // 'as': every other album. 'from': albums that have written songs.
  const albumOptions = useMemo(() => {
    const withSongs = new Set(allGens.map(g => g.lyrics_set_id));
    return albums
      .filter(a => a.id !== currentLyricsSetId)
      .filter(a => mode !== 'from' || withSongs.has(a.id))
      .sort((a, b) => albumLabel(a).localeCompare(albumLabel(b)))
      .map(a => ({ value: a.id, label: albumLabel(a), hint: mode === 'as' ? presetHint(presets.get(a.id)) : undefined }));
  }, [albums, presets, allGens, currentLyricsSetId, mode]); // eslint-disable-line react-hooks/exhaustive-deps

  const songOptions = useMemo(() => allGens
    .filter(g => g.lyrics_set_id === albumId)
    .map(g => ({ value: g.id, label: g.title || 'Untitled', hint: [g.bpm ? `${g.bpm} BPM` : '', g.key || '', g.subject || ''].filter(Boolean).join(' · ') })),
  [allGens, albumId]);

  if (!mode) return null;

  // Resolve the pair this render would use.
  let song: Generation | null = null;
  let dest: RenderTarget | null = null;
  if (mode === 'as') {
    song = gen ?? null;
    const a = albums.find(x => x.id === albumId);
    if (a) dest = { lyricsSetId: a.id, artistId: a.artist_id, artistName: a.artist_name, artistImageUrl: a.image_url };
  } else {
    song = allGens.find(g => g.id === genId) ?? null;
    dest = target ?? null;
  }
  const ready = !!song && !!dest;

  const run = (fn: (g: Generation, d: RenderTarget) => void) => {
    if (!song || !dest) return;
    fn(song, dest);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/30 dark:bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-white/10 rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-200 dark:border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-pink-500/10 flex items-center justify-center">
              <Shuffle className="w-4 h-4 text-pink-400" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">
                {mode === 'as' ? t('lyric.renderAs', 'Render as another album') : t('lyric.renderFrom', 'Render a song from another album')}
              </h2>
              <p className="text-xs text-zinc-500">
                {mode === 'as'
                  ? (gen?.title || 'Untitled')
                  : `${t('lyric.renderFromInto', 'Renders as')} ${target ? disguiseArtist(target.artistName) : ''}`}
              </p>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-white/5 text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-4">
          <label className="flex flex-col gap-1">
            <ParamLabel
              label={mode === 'as' ? t('lyric.renderAsAlbum', 'Render as') : t('lyric.renderFromAlbum', 'Source album')}
              info={mode === 'as'
                ? t('lyric.renderAsAlbumInfo', "The album whose settings this song renders with: its preset's adapters and reference track, and its own tracks as the caption source on MM3 and YuE2. The song keeps its lyrics, tempo and key. The recording is listed under this album, not the song's own.")
                : t('lyric.renderFromAlbumInfo', "The album the song was written for. Only the song's lyrics, captions, tempo and key come from it; the render uses this album's preset and caption source, and the recording is listed here.")}
              className="text-sm text-zinc-700 dark:text-zinc-300"
            />
            <StyledSelect
              accent="pink"
              value={albumId}
              onChange={v => { setAlbumId(v); setGenId(''); }}
              options={albumOptions}
              placeholder={loading ? t('common.loading', 'Loading…') : t('lyric.pickAlbum', 'Pick an album')}
              searchable
              className="w-full"
            />
          </label>

          {mode === 'from' && (
            <label className="flex flex-col gap-1">
              <ParamLabel
                label={t('lyric.renderFromSong', 'Song')}
                info={t('lyric.renderFromSongInfo', "A written song from the source album. Its pinned caption track, if it had one, belongs to its own album and is not used here; the caption source falls back to this album's tracks by tempo.")}
                className="text-sm text-zinc-700 dark:text-zinc-300"
              />
              <StyledSelect
                accent="pink"
                value={genId}
                onChange={setGenId}
                options={songOptions}
                placeholder={albumId === '' ? t('lyric.pickAlbumFirst', 'Pick an album first') : t('lyric.pickSong', 'Pick a song')}
                disabled={albumId === ''}
                searchable
                className="w-full"
              />
            </label>
          )}

          <div className="flex items-center gap-2 pt-2">
            <button
              onClick={() => run(onRender)}
              disabled={!ready}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-pink-500/30 to-purple-500/30 text-white hover:from-pink-500/40 hover:to-purple-500/40 text-sm font-semibold transition-all border border-pink-500/20 disabled:opacity-40"
            >
              <Play className="w-3.5 h-3.5" />
              {t('lyric.generateAudio')}
            </button>
            {onSendToCreate && (
              <button
                onClick={() => run(onSendToCreate)}
                disabled={!ready}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-500/20 text-amber-300 hover:bg-amber-500/30 text-sm font-medium transition-colors border border-amber-500/10 disabled:opacity-40"
              >
                <Send className="w-3.5 h-3.5" />
                {t('lyric.sendToCreate')}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
