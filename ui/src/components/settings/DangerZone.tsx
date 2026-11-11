// DangerZone.tsx — the settings storage tab's destructive actions: nuke
// generated audio, generated lyrics, or artist profiles. Each is a two-click
// confirm; failures are toasted by the transport's one notifier, so there is
// no local error line (a "Failed:" copy of the same message was a double toast).

import React, { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { AlertTriangle, Loader2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { songApi } from '../../services/api';
import { lireekApi } from '../../services/lireekApi';
import { invalidateEvents } from '../../queries/invalidation';

interface DangerZoneProps {
  onNukeComplete?: () => void;
}

export const DangerZone: React.FC<DangerZoneProps> = ({ onNukeComplete }) => {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [nukeConfirm, setNukeConfirm] = useState(false);
  const [nukeLyricsConfirm, setNukeLyricsConfirm] = useState(false);
  const [nukeProfilesConfirm, setNukeProfilesConfirm] = useState(false);
  const [nukeResult, setNukeResult] = useState<string | null>(null);

  // A nuke dirties every cached view of songs (the full nuke, the Lyric Studio
  // area too), so each one invalidates through the manager — these handlers
  // used to skip it entirely (F4#11).
  const nukeMut = useMutation({
    mutationFn: () => songApi.nukeGenerations(),
    onMutate: () => { setNukeResult(null); },
    onSuccess: res => {
      setNukeResult(
        `Nuked ${res.songsDeleted} songs, ${res.filesDeleted} audio files, ${res.lireekAudioGensDeleted} Lyric Studio entries.`
      );
      setNukeConfirm(false);
      onNukeComplete?.();
      void invalidateEvents(queryClient, ['song:written', 'cover-lyrics:lireek-purged']);
    },
  });
  const nukeRunning = nukeMut.isPending;
  // Named zero-arg handlers so the buttons can pass them straight to onClick —
  // mutate itself takes `void` variables, which a click event is not.
  const nuke = () => nukeMut.mutate();

  const nukeLyricsMut = useMutation({
    mutationFn: () => lireekApi.purgeGenerations(),
    onMutate: () => { setNukeResult(null); },
    onSuccess: res => {
      setNukeResult(`Deleted ${res.generations_deleted} generated lyrics.`);
      setNukeLyricsConfirm(false);
      void invalidateEvents(queryClient, ['cover-lyrics:lireek-purged']);
    },
  });
  const nukeLyricsRunning = nukeLyricsMut.isPending;
  const nukeLyrics = () => nukeLyricsMut.mutate();

  const nukeProfilesMut = useMutation({
    mutationFn: () => lireekApi.purgeProfiles(),
    onMutate: () => { setNukeResult(null); },
    onSuccess: res => {
      setNukeResult(`Deleted ${res.profiles_deleted} profiles and ${res.generations_deleted} dependent lyrics.`);
      setNukeProfilesConfirm(false);
      void invalidateEvents(queryClient, ['cover-lyrics:lireek-purged']);
    },
  });
  const nukeProfilesRunning = nukeProfilesMut.isPending;
  const nukeProfiles = () => nukeProfilesMut.mutate();

  return (
    <div className="settings-section" style={{ borderColor: 'rgba(239, 68, 68, 0.2)' }}>
      <div className="settings-section-header">
        <AlertTriangle size={16} style={{ color: '#ef4444' }} />
        <span className="settings-section-title" style={{ color: '#ef4444' }}>{t('settings.storage.dangerZone')}</span>
      </div>

      <div className="setting-row setting-row--danger">
        <div className="setting-info">
          <div className="setting-label">{t('settings.storage.nukeGenerations')}</div>
          <div className="setting-description">
            Permanently delete <strong>all</strong> generated audio — songs from the library, audio files from disk,
            and all Lyric Studio audio generation entries. Lyrics, profiles, and artist data are preserved.
          </div>
        </div>
        <div className="settings-danger-actions">
          {!nukeConfirm ? (
            <button
              id="nuke-generations-btn"
              onClick={() => setNukeConfirm(true)}
              disabled={nukeRunning}
              className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
              style={{
                background: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                cursor: 'pointer',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.25)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.5)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.15)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.3)'; }}
            >
              ☢️ {t('settings.storage.nukeGenerations')}
            </button>
          ) : (
            <div className="settings-actions">
              <span className="text-xs text-red-400" style={{ whiteSpace: 'nowrap' }}>{t('settings.storage.areYouSure')}</span>
              <button
                id="nuke-confirm-btn"
                onClick={nuke}
                disabled={nukeRunning}
                className="px-4 py-2 rounded-lg text-sm font-bold transition-all duration-200"
                style={{
                  background: '#dc2626',
                  border: '1px solid #ef4444',
                  color: 'white',
                  cursor: nukeRunning ? 'wait' : 'pointer',
                  opacity: nukeRunning ? 0.6 : 1,
                }}
              >
                {nukeRunning ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Loader2 size={14} className="animate-spin" /> {t('settings.storage.nuking')}
                  </span>
                ) : (
                  t('settings.storage.confirmNuke')
                )}
              </button>
              <button
                onClick={() => setNukeConfirm(false)}
                disabled={nukeRunning}
                className="px-3 py-2 rounded-lg text-sm text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors"
                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', cursor: 'pointer' }}
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Nuke Lyrics */}
      <div className="setting-row setting-row--danger">
        <div className="setting-info">
          <div className="setting-label">{t('settings.storage.nukeLyrics')}</div>
          <div className="setting-description">
            Permanently delete <strong>all</strong> generated/written song lyrics across all artists.
            Profiles and source lyrics are preserved.
          </div>
        </div>
        <div className="settings-danger-actions">
          {!nukeLyricsConfirm ? (
            <button
              id="nuke-lyrics-btn"
              onClick={() => setNukeLyricsConfirm(true)}
              disabled={nukeLyricsRunning}
              className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
              style={{
                background: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                cursor: 'pointer',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.25)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.5)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.15)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.3)'; }}
            >
              🗑️ {t('settings.storage.nukeLyrics')}
            </button>
          ) : (
            <div className="settings-actions">
              <span className="text-xs text-red-400" style={{ whiteSpace: 'nowrap' }}>{t('settings.storage.areYouSure')}</span>
              <button
                id="nuke-lyrics-confirm-btn"
                onClick={nukeLyrics}
                disabled={nukeLyricsRunning}
                className="px-4 py-2 rounded-lg text-sm font-bold transition-all duration-200"
                style={{
                  background: '#dc2626',
                  border: '1px solid #ef4444',
                  color: 'white',
                  cursor: nukeLyricsRunning ? 'wait' : 'pointer',
                  opacity: nukeLyricsRunning ? 0.6 : 1,
                }}
              >
                {nukeLyricsRunning ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Loader2 size={14} className="animate-spin" /> {t('settings.storage.nuking')}
                  </span>
                ) : (
                  t('settings.storage.confirmNukeLyrics')
                )}
              </button>
              <button
                onClick={() => setNukeLyricsConfirm(false)}
                disabled={nukeLyricsRunning}
                className="px-3 py-2 rounded-lg text-sm text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors"
                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', cursor: 'pointer' }}
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Nuke Profiles */}
      <div className="setting-row setting-row--danger">
        <div className="setting-info">
          <div className="setting-label">{t('settings.storage.nukeProfiles')}</div>
          <div className="setting-description">
            Permanently delete <strong>all</strong> artist profiles and their dependent written lyrics.
            Source lyrics and artist data are preserved.
          </div>
        </div>
        <div className="settings-danger-actions">
          {!nukeProfilesConfirm ? (
            <button
              id="nuke-profiles-btn"
              onClick={() => setNukeProfilesConfirm(true)}
              disabled={nukeProfilesRunning}
              className="px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200"
              style={{
                background: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                cursor: 'pointer',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.25)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.5)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(239, 68, 68, 0.15)'; e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.3)'; }}
            >
              🧬 {t('settings.storage.nukeProfiles')}
            </button>
          ) : (
            <div className="settings-actions">
              <span className="text-xs text-red-400" style={{ whiteSpace: 'nowrap' }}>{t('settings.storage.lyricsAlsoDeleted')}</span>
              <button
                id="nuke-profiles-confirm-btn"
                onClick={nukeProfiles}
                disabled={nukeProfilesRunning}
                className="px-4 py-2 rounded-lg text-sm font-bold transition-all duration-200"
                style={{
                  background: '#dc2626',
                  border: '1px solid #ef4444',
                  color: 'white',
                  cursor: nukeProfilesRunning ? 'wait' : 'pointer',
                  opacity: nukeProfilesRunning ? 0.6 : 1,
                }}
              >
                {nukeProfilesRunning ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Loader2 size={14} className="animate-spin" /> {t('settings.storage.nuking')}
                  </span>
                ) : (
                  t('settings.storage.confirmNukeProfiles')
                )}
              </button>
              <button
                onClick={() => setNukeProfilesConfirm(false)}
                disabled={nukeProfilesRunning}
                className="px-3 py-2 rounded-lg text-sm text-zinc-600 dark:text-zinc-400 hover:text-white transition-colors"
                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', cursor: 'pointer' }}
              >
                Cancel
              </button>
            </div>
          )}
        </div>
      </div>
      {nukeResult && (
        <div
          className="text-sm px-4 py-3 rounded-lg"
          style={{
            background: nukeResult.startsWith('Failed')
              ? 'rgba(239, 68, 68, 0.1)' : 'rgba(34, 197, 94, 0.1)',
            border: `1px solid ${nukeResult.startsWith('Failed')
              ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)'}`,
            color: nukeResult.startsWith('Failed') ? '#fca5a5' : '#86efac',
            marginTop: '8px',
          }}
        >
          {nukeResult}
        </div>
      )}
    </div>
  );
};
