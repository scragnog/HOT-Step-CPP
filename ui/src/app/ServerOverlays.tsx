// ServerOverlays.tsx — the restarting and offline screens. Both are full-page
// liveness states, kept out of App.tsx so the app shell reads as routing.

import React, { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import serverOfflineImage from '../assets/server-offline.webp?inline';
import { useServerHealthReload } from './useServerHealthReload';

/** Restarting overlay — polls /api/health and reloads when the server is back */
export const RestartingOverlay: React.FC = () => {
  const { t } = useTranslation();
  const [status, setStatus] = useState(t('app.restarting.stopping'));
  const [dots, setDots] = useState('');

  // Animate dots
  useEffect(() => {
    const iv = setInterval(() => {
      setDots(prev => prev.length >= 3 ? '' : prev + '.');
    }, 500);
    return () => clearInterval(iv);
  }, []);

  // Keep polling, but say what to do once a restart has clearly stalled.
  const [stalled, setStalled] = useState(false);
  useEffect(() => {
    const id = window.setTimeout(() => setStalled(true), 90_000);
    return () => window.clearTimeout(id);
  }, []);

  const onWaiting = useCallback(() => setStatus(t('app.restarting.waiting')), [t]);
  const onReconnected = useCallback(() => setStatus(t('app.restarting.reconnected')), [t]);
  useServerHealthReload(3000, onWaiting, onReconnected);

  return (
    <div className="fixed inset-0 z-[200] flex flex-col items-center justify-center bg-white dark:bg-black text-zinc-900 dark:text-white">
      {/* Animated spinner */}
      <div className="relative mb-8">
        <div className="w-16 h-16 rounded-full border-4 border-amber-500/20" />
        <div className="absolute inset-0 w-16 h-16 rounded-full border-4 border-transparent border-t-amber-400 animate-spin" />
      </div>
      <h1 className="text-2xl font-bold mb-2">{t('app.restarting.title')}</h1>
      <p className="text-zinc-600 dark:text-zinc-400 text-lg">{status}{dots}</p>
      <p className="text-zinc-600 text-sm mt-4">{t('app.restarting.message')}</p>
      {stalled && (
        <p className="text-amber-600 dark:text-amber-400 text-sm mt-4 max-w-md text-center">
          {t('app.restarting.stalled', 'The server has not come back after 90 seconds. Check the HOT-Step Server window for errors. If it has closed, start HOT-Step again; this page reloads by itself once the server answers.')}
        </p>
      )}
    </div>
  );
};

export const ServerOfflineOverlay: React.FC = () => {
  useServerHealthReload(0);
  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black">
      <img src={serverOfflineImage} alt="Server offline. Please restart or close tab" className="max-w-[90vw] max-h-[90vh] object-contain" />
    </div>
  );
};

