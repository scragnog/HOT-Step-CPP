import { useQuery } from '@tanstack/react-query';
import { studioRecentSongsOptions } from '@/queries/studios/resources';
// SourceSelector.tsx — Upload or pick existing track for stem extraction
import React, { useState, useCallback, useRef } from 'react';
import { Upload, Music, Search, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { ParamLabel } from '@/components/shared/controls/ParamLabel';
import { useAuth } from '@/context/AuthContext';
import { useUpload } from '@/queries/writes';

interface SourceSelectorProps {
  sourceAudioUrl: string;
  sourceFileName: string;
  onSourceChange: (url: string, fileName: string, meta?: { style?: string; lyrics?: string }) => void;
}

const SOURCE_FILTERS = [
  { value: 'all', label: 'All' },
  { value: 'create', label: 'Create' },
  { value: 'lyric-studio', label: 'Lyric Studio' },
  { value: 'cover-studio', label: 'Cover Studio' },
];

export const SourceSelector: React.FC<SourceSelectorProps> = ({ sourceAudioUrl, sourceFileName, onSourceChange }) => {
  const { token, sessionEpoch } = useAuth();
  void sessionEpoch;
  const { t } = useTranslation();
  const [uploadDragging, setUploadDragging] = useState(false);
  const { mutate: uploadAudio, isPending: uploading } = useUpload<{ audio_url?: string }>();
  const [showPicker, setShowPicker] = useState(false);
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  const songsQuery = useQuery({
    ...studioRecentSongsOptions(filter === 'all' ? 'all' : filter),
    enabled: showPicker && !!token,
  });
  const songs = songsQuery.data?.songs ?? [];

  const filteredSongs = songs.filter(s =>
    !search || s.title.toLowerCase().includes(search.toLowerCase()) ||
    (s.artist_name || '').toLowerCase().includes(search.toLowerCase())
  );

  const handleUpload = useCallback((file: File) => {
    uploadAudio({
      path: '/upload/audio',
      files: [{ field: 'audio', file }],
      onSuccess: data => { if (data.audio_url) onSourceChange(data.audio_url, file.name); },
    });
  }, [onSourceChange, uploadAudio]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setUploadDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
  };

  return (
    <div style={styles.container}>
      {(songsQuery.error) && <div role="alert" className="text-sm text-red-500">{(songsQuery.error)?.message}</div>}
      <h3 style={styles.sectionTitle}>{t('stem.sourceAudio')}</h3>

      {/* Current selection */}
      {sourceAudioUrl && (
        <div style={styles.selectedSource}>
          <Music size={14} style={{ color: '#a78bfa', flexShrink: 0 }} />
          <span style={styles.selectedName}>{sourceFileName}</span>
          <button onClick={() => onSourceChange('', '')} style={styles.clearButton}>
            <X size={12} />
          </button>
        </div>
      )}

      {/* Upload zone */}
      <div
        style={{
          ...styles.uploadZone,
          borderColor: uploadDragging ? '#a78bfa' : 'rgba(255,255,255,0.1)',
          background: uploadDragging ? 'rgba(167,139,250,0.08)' : 'rgba(255,255,255,0.03)',
        }}
        onDragOver={(e) => { e.preventDefault(); setUploadDragging(true); }}
        onDragLeave={() => setUploadDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
      >
        <input ref={fileRef} type="file" accept="audio/*" style={{ display: 'none' }} onChange={handleFileInput} />
        {uploading ? (
          <span style={styles.uploadText}>Uploading...</span>
        ) : (
          <>
            <Upload size={20} style={{ color: '#888' }} />
            <span style={styles.uploadText}>{t('stem.dropAudioOrUpload')}</span>
            <span style={styles.uploadHint}>MP3, WAV, FLAC, AIFF, OGG</span>
          </>
        )}
      </div>

      {/* Divider */}
      <div style={styles.divider}>
        <span style={styles.dividerText}>{t('stem.orPickFromLibrary')}</span>
      </div>

      {/* Song picker toggle */}
      <button onClick={() => setShowPicker(!showPicker)} style={styles.pickerToggle}>
        <Music size={14} />
        {showPicker ? t('stem.hideLibrary') : t('stem.browseLibrary')}
      </button>

      {/* Song picker */}
      {showPicker && (
        <div style={styles.picker}>
          {/* Filters */}
          <div style={styles.filterRow}>
            {SOURCE_FILTERS.map(f => (
              <button
                key={f.value}
                onClick={() => setFilter(f.value)}
                style={{
                  ...styles.filterBtn,
                  background: filter === f.value ? '#a78bfa' : 'rgba(255,255,255,0.06)',
                  color: filter === f.value ? '#000' : '#aaa',
                }}
              >
                {f.label}
              </button>
            ))}
          </div>

          {/* Search */}
          <div style={styles.searchRow}>
            <Search size={13} style={{ color: '#666' }} />
            <input
              type="text"
              placeholder="Search..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={styles.searchInput}
            />
          </div>

          {/* Song list */}
          <div style={styles.songList}>
            {filteredSongs.length === 0 && (
              <div style={styles.emptyMsg}>{t('stem.noTracksFound')}</div>
            )}
            {filteredSongs.map(song => {
              const hasMastered = !!song.mastered_audio_url;
              return (
              <div
                key={song.id}
                style={{
                  ...styles.songItem,
                  background: sourceAudioUrl === song.audio_url || sourceAudioUrl === song.mastered_audio_url
                    ? 'rgba(167,139,250,0.15)' : 'transparent',
                }}
              >
                <button
                  onClick={() => {
                    onSourceChange(song.audio_url, song.title, {
                      style: song.caption || '',
                      lyrics: song.lyrics || '',
                    });
                    setShowPicker(false);
                  }}
                  style={styles.songSelectBtn}
                >
                  <span style={styles.songTitle}>{song.title}</span>
                  {song.artist_name && <span style={styles.songArtist}>{song.artist_name}</span>}
                  <span style={styles.songSource}>{song.source}</span>
                </button>
                {hasMastered && (
                  <div style={styles.versionToggle}>
                    <ParamLabel label={t('stem.source.raw', 'Raw')}
                      info={t('stem.source.rawInfo', 'Uses the unmastered render as the source for stem separation.')}
                      rootClassName="inline-flex">
                      <button
                        onClick={() => {
                          onSourceChange(song.audio_url, song.title, {
                            style: song.caption || '',
                            lyrics: song.lyrics || '',
                          });
                          setShowPicker(false);
                        }}
                        style={{
                          ...styles.versionBtn,
                          background: sourceAudioUrl === song.audio_url ? '#a78bfa' : 'transparent',
                          color: sourceAudioUrl === song.audio_url ? '#000' : '#888',
                        }}
                        aria-pressed={sourceAudioUrl === song.audio_url}
                      >Raw</button>
                    </ParamLabel>
                    <ParamLabel label={t('stem.source.mastered', 'Mastered')}
                      info={t('stem.source.masteredInfo', 'Uses the mastered render as the source for stem separation.')}
                      rootClassName="inline-flex">
                      <button
                        onClick={() => {
                          onSourceChange(song.mastered_audio_url!, `${song.title} (Mastered)`, {
                            style: song.caption || '',
                            lyrics: song.lyrics || '',
                          });
                          setShowPicker(false);
                        }}
                        style={{
                          ...styles.versionBtn,
                          background: sourceAudioUrl === song.mastered_audio_url ? '#a78bfa' : 'transparent',
                          color: sourceAudioUrl === song.mastered_audio_url ? '#000' : '#888',
                        }}
                        aria-pressed={sourceAudioUrl === song.mastered_audio_url}
                      >Mastered</button>
                    </ParamLabel>
                  </div>
                )}
              </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Optional: style hint + lyrics */}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  sectionTitle: {
    margin: 0,
    fontSize: 14,
    fontWeight: 600,
    color: '#d4d4d4',
    letterSpacing: '0.02em',
  },
  selectedSource: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '8px 12px',
    borderRadius: 8,
    background: 'rgba(167,139,250,0.1)',
    border: '1px solid rgba(167,139,250,0.2)',
  },
  selectedName: {
    fontSize: 13,
    color: '#d4d4d4',
    flex: 1,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  clearButton: {
    background: 'none',
    border: 'none',
    color: '#888',
    cursor: 'pointer',
    padding: 4,
    borderRadius: 4,
    display: 'flex',
    alignItems: 'center',
  },
  uploadZone: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    padding: '20px 16px',
    borderRadius: 10,
    border: '1.5px dashed',
    cursor: 'pointer',
    transition: 'all 0.15s ease',
  },
  uploadText: {
    fontSize: 13,
    color: '#888',
    fontWeight: 500,
  },
  uploadHint: {
    fontSize: 11,
    color: '#555',
  },
  divider: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  dividerText: {
    fontSize: 11,
    color: '#555',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    width: '100%',
    textAlign: 'center',
  },
  pickerToggle: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '8px 12px',
    borderRadius: 8,
    border: '1px solid rgba(255,255,255,0.1)',
    background: 'rgba(255,255,255,0.04)',
    color: '#aaa',
    cursor: 'pointer',
    fontSize: 13,
    fontWeight: 500,
    transition: 'all 0.15s ease',
  },
  picker: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    padding: 10,
    borderRadius: 10,
    background: 'rgba(0,0,0,0.2)',
    border: '1px solid rgba(255,255,255,0.06)',
  },
  filterRow: {
    display: 'flex',
    gap: 4,
    flexWrap: 'wrap',
  },
  filterBtn: {
    padding: '4px 10px',
    borderRadius: 6,
    border: 'none',
    fontSize: 11,
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.1s ease',
  },
  searchRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '6px 10px',
    borderRadius: 6,
    background: 'rgba(255,255,255,0.04)',
    border: '1px solid rgba(255,255,255,0.06)',
  },
  searchInput: {
    flex: 1,
    background: 'transparent',
    border: 'none',
    color: '#d4d4d4',
    fontSize: 12,
    outline: 'none',
  },
  songList: {
    display: 'flex',
    flexDirection: 'column',
    maxHeight: 200,
    overflowY: 'auto',
  },
  emptyMsg: {
    padding: 16,
    textAlign: 'center',
    color: '#555',
    fontSize: 12,
  },
  songItem: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '4px 8px',
    borderRadius: 6,
    transition: 'background 0.1s ease',
  },
  songSelectBtn: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '4px 0',
    border: 'none',
    background: 'none',
    cursor: 'pointer',
    textAlign: 'left',
    minWidth: 0,
  },
  songTitle: {
    flex: 1,
    fontSize: 12,
    color: '#d4d4d4',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  songArtist: {
    fontSize: 11,
    color: '#888',
  },
  songSource: {
    fontSize: 10,
    padding: '1px 6px',
    borderRadius: 4,
    background: 'rgba(255,255,255,0.06)',
    color: '#666',
    fontWeight: 600,
    textTransform: 'uppercase',
    flexShrink: 0,
  },
  versionToggle: {
    display: 'flex',
    gap: 2,
    borderRadius: 6,
    border: '1px solid rgba(255,255,255,0.08)',
    overflow: 'hidden',
    flexShrink: 0,
  },
  versionBtn: {
    padding: '2px 7px',
    border: 'none',
    fontSize: 10,
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.1s ease',
  } as React.CSSProperties,
};
