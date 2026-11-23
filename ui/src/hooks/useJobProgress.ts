/**
 * Generation progress surfaces: tab title, badge data and the
 * "job finished while you were away" notification.
 *
 * Permission is only requested from a user gesture (first Generate click, or
 * the opt-in toggle), never on load.
 */
import { useEffect, useRef } from 'react';
import { useAudioGenQueue, useAudioGenQueueSelector } from '@/stores/audioGenQueueStore';
import { summarizeJobs, formatTitle, formatBadge, newlySucceeded, statusSnapshot, type JobSummary } from '@/utils/jobBadge';
import { emit } from '@/utils/appEvents';

const ASKED_KEY = 'hotstep.notify.asked';

export function notificationsSupported(): boolean {
  return typeof window !== 'undefined' && 'Notification' in window;
}

/** Call from a click handler. Asks once (or always when `force`, for the toggle). */
export async function requestNotifyPermissionFromGesture(force = false): Promise<NotificationPermission | 'unsupported'> {
  if (!notificationsSupported()) return 'unsupported';
  if (Notification.permission !== 'default') return Notification.permission;
  if (!force && localStorage.getItem(ASKED_KEY)) return Notification.permission;
  localStorage.setItem(ASKED_KEY, '1');
  try { return await Notification.requestPermission(); } catch { return Notification.permission; }
}

/** Badge text ("2 · 43%") for the sidebar / mobile header, or null when idle. */
export function useJobBadge(): string | null {
  const key = useAudioGenQueueSelector(s => {
    const sum = summarizeJobs(s.items);
    return `${sum.running}|${sum.activePct ?? ''}`;
  });
  const [r, p] = key.split('|');
  const sum: JobSummary = { running: Number(r), activePct: p === '' ? null : Number(p) };
  return formatBadge(sum);
}

async function showDoneNotification(title: string, body: string, songId?: string): Promise<void> {
  if (!notificationsSupported() || Notification.permission !== 'granted') return;
  const options: NotificationOptions = { body, icon: '/logo.webp', tag: songId ? `song-${songId}` : undefined, data: { songId } };
  try {
    const reg = 'serviceWorker' in navigator ? await navigator.serviceWorker.getRegistration() : undefined;
    if (reg) { await reg.showNotification(title, options); return; }
  } catch { /* fall through */ }
  try {
    const n = new Notification(title, options);
    n.onclick = () => {
      window.focus();
      if (songId) emit('hotstep:open-song', { songId });
      n.close();
    };
  } catch { /* some mobile browsers only allow SW notifications */ }
}

/** Mount once (App). Drives document.title and the completion notification. */
export function useJobProgressWatcher(doneTitle: string, doneBody: (name: string) => string): void {
  const state = useAudioGenQueue();
  const baseTitle = useRef<string>(typeof document !== 'undefined' ? document.title : 'HOT-Step');
  const prev = useRef<Map<string, string> | null>(null);

  useEffect(() => {
    const sum = summarizeJobs(state.items);
    document.title = formatTitle(sum, baseTitle.current);
    if (prev.current && typeof document !== 'undefined' && document.hidden) {
      for (const it of newlySucceeded(prev.current, state.items)) {
        const name = (it.generation as { title?: string } | undefined)?.title || 'Song';
        void showDoneNotification(doneTitle, doneBody(name), it.songId);
      }
    }
    prev.current = statusSnapshot(state.items);
  }, [state.items, doneTitle, doneBody]);

  useEffect(() => () => { document.title = baseTitle.current; }, []);

  // Service worker relays notification clicks as a message.
  useEffect(() => {
    if (!('serviceWorker' in navigator)) return;
    const timers = new Set<ReturnType<typeof setTimeout>>();
    // AppContent installs its song listener in a separate effect.
    const afterMount = (fn: () => void) => {
      const timer = setTimeout(() => { timers.delete(timer); fn(); }, 0);
      timers.add(timer);
    };
    const onMsg = (e: MessageEvent) => {
      if (e.data?.type === 'notification-ready') {
        afterMount(() => e.ports[0]?.postMessage('ready'));
      }
      if (e.data?.type === 'open-song' && typeof e.data.songId === 'string' && e.data.songId) {
        emit('hotstep:open-song', { songId: e.data.songId });
      }
    };
    navigator.serviceWorker.addEventListener('message', onMsg);
    const url = new URL(window.location.href);
    const songId = url.searchParams.get('open-song');
    if (songId) afterMount(() => {
      emit('hotstep:open-song', { songId });
      url.searchParams.delete('open-song');
      history.replaceState(history.state, '', url);
    });
    return () => {
      navigator.serviceWorker.removeEventListener('message', onMsg);
      timers.forEach(clearTimeout);
    };
  }, []);
}
