// ShellHost.tsx — mounts everything a shell skin must not mount itself: the global-params
// provider, the app runtime (queue, live takes, song-cache events, offline detection) and the
// offline, restart and update overlays. Then it renders the chosen shell inside a
// `[data-shell="<id>"]` root that owns font, margin and colour-scheme (docs/dev/skins.md).
// A skin can't forget the runtime or mount it twice. Classic does not use the host.

import React, { Suspense, useCallback, useEffect, useMemo } from 'react';
import { useAuth } from '../context/AuthContext';
import { GlobalParamsProvider } from '../context/GlobalParamsContext';
import { ErrorBoundary } from '../components/shared/ErrorBoundary';
import { UpdateReconnectOverlay, UpdateRestartContext } from '../components/settings/UpdateReconnectOverlay';
import { enqueueSimpleGen } from '../stores/audioGenQueueStore';
import { requestNotifyPermissionFromGesture } from '../hooks/useJobProgress';
import { useTheme } from '../hooks/useTheme';
import { RestartingOverlay, ServerOfflineOverlay } from './ServerOverlays';
import { useAppRuntime } from './useAppRuntime';
import { setActiveRuntime, type SkinRuntime } from './skinContext';

const Runtime: React.FC<{ shellId: string; children: React.ReactNode }> = ({ shellId, children }) => {
  const { token } = useAuth();
  const { isOffline, isRestarting, updateAttempt, setUpdateAttempt, handleSongCreated, enrichCreateParams } = useAppRuntime();
  const { theme } = useTheme(); // mounting it makes the DOM match hs-theme

  const generate = useCallback((content: Record<string, unknown>) => {
    if (!token) return;
    void requestNotifyPermissionFromGesture(); // first Generate click is a user gesture
    void enqueueSimpleGen(enrichCreateParams(content), token, handleSongCreated);
  }, [token, enrichCreateParams, handleSongCreated]);
  const runtime = useMemo<SkinRuntime>(() => ({ shellId, generate }), [shellId, generate]);
  useEffect(() => { setActiveRuntime(runtime); return () => setActiveRuntime(null); }, [runtime]);
  const restartCtx = useMemo(() => ({ attempt: updateAttempt, setAttempt: setUpdateAttempt }), [updateAttempt, setUpdateAttempt]);
  // The host owns the document root's colour scheme; the shell root inherits it.
  useEffect(() => { document.documentElement.style.colorScheme = theme; }, [theme]);

  // Same precedence as classic: update recovery owns its health polling, then restart, then offline.
  if (updateAttempt) return <div className="fixed inset-0 bg-white dark:bg-black"><UpdateReconnectOverlay attempt={updateAttempt} /></div>;
  if (isRestarting) return <RestartingOverlay />;
  if (isOffline) return <ServerOfflineOverlay />;

  return (
          <UpdateRestartContext.Provider value={restartCtx}>
        {/* min-h so the shell may size itself; margin and font are the host's, not the shell's */}
        <div data-shell={shellId} className="min-h-screen m-0 font-sans" style={{ colorScheme: theme }}>
          <ErrorBoundary>
            <Suspense fallback={null}>{children}</Suspense>
          </ErrorBoundary>
        </div>
      </UpdateRestartContext.Provider>
  );
};

export const ShellHost: React.FC<{ shellId: string; children: React.ReactNode }> = ({ shellId, children }) => (
  <GlobalParamsProvider>
    <Runtime shellId={shellId}>{children}</Runtime>
  </GlobalParamsProvider>
);
