import { createContext, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { RefreshCw } from 'lucide-react';
import type { UpdateApplyResult, UpdateRestartAttempt } from '../../services/updatesApi';
import { updateReconnectOptions } from '../../queries/shell-queue/settings';
import { updateVersion } from './updatesViewState';

// App owns the attempt so replacing Settings with a liveness screen cannot
// discard it. Settings starts the attempt only after apply returns 202.
export const UpdateRestartContext = createContext<{
  attempt: UpdateRestartAttempt | null;
  setAttempt: (attempt: UpdateRestartAttempt) => void;
} | null>(null);

export function UpdateOutcome({ result }: { result: UpdateApplyResult }) {
  const { t } = useTranslation();
  return <>
    <p>{t(`settings.updates.outcome.${result.outcome}`, { version: updateVersion(result) })}</p>
    {result.message && <p className="setting-description">{result.message}</p>}
    <time className="setting-description" dateTime={result.at}>{new Date(result.at).toLocaleString()}</time>
  </>;
}

export function UpdateReconnectOverlay({ attempt }: { attempt: UpdateRestartAttempt }) {
  const { t } = useTranslation();
  const dialog = useRef<HTMLDialogElement>(null);
  const result = useQuery(updateReconnectOptions(attempt)).data;
  const [overdue, setOverdue] = useState(false);
  useEffect(() => {
    const el = dialog.current!;
    el.showModal();
    const timer = window.setTimeout(() => setOverdue(true), Math.max(0, 300_000 - (Date.now() - attempt.startedAt)));
    return () => { clearTimeout(timer); el.close(); };
  }, [attempt.startedAt]);

  return createPortal(<dialog ref={dialog} className="settings-update-overlay" aria-labelledby="update-reconnect-title"
    onCancel={e => e.preventDefault()}>
    <div className="settings-update-overlay-card" role="status" aria-live="polite">
      <RefreshCw size={24} className={result ? '' : 'animate-spin'} aria-hidden="true" />
      <h2 id="update-reconnect-title">{t(result ? 'settings.updates.finished' : 'settings.updates.restarting')}</h2>
      {result ? <>
        <UpdateOutcome result={result} />
        <button type="button" className="settings-save-btn" onClick={() => window.location.reload()} autoFocus>{t('settings.updates.reload')}</button>
      </> : <p>{t(overdue ? 'settings.updates.reconnectTimeout' : 'settings.updates.reconnectWaiting')}</p>}
    </div>
  </dialog>, document.body);
}
