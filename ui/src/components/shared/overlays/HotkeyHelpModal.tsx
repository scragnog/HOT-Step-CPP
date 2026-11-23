import React, { useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { HOTKEY_HELP } from '@/utils/hotkeys';
import { useFocusTrap } from '@/hooks/useFocusTrap';

/** Small cheat sheet for the global keyboard shortcuts ("?"). */
export const HotkeyHelpModal: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const { t } = useTranslation();
  const dialogRef = useRef<HTMLDivElement>(null);
  useFocusTrap(dialogRef, true, { onEscape: onClose, consumeEscape: true });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === '?') { e.preventDefault(); onClose(); } };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        ref={dialogRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label={t('hotkeys.title')}
        className="w-80 max-w-[90vw] rounded-xl bg-white dark:bg-suno-panel border border-zinc-200 dark:border-zinc-800 p-4 shadow-xl"
        onClick={e => e.stopPropagation()}
      >
        <h2 className="text-sm font-semibold mb-3">{t('hotkeys.title')}</h2>
        <table className="w-full text-xs">
          <tbody>
            {HOTKEY_HELP.map(row => (
              <tr key={row.keys} className="border-t border-zinc-100 dark:border-zinc-800">
                <td className="py-1.5 pr-3 whitespace-nowrap"><kbd className="px-1.5 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 font-mono">{row.keys}</kbd></td>
                <td className="py-1.5 text-zinc-600 dark:text-zinc-300">{t(`hotkeys.actions.${row.action}`)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-3 text-[11px] text-zinc-500">{t('hotkeys.note')}</p>
      </div>
    </div>
  );
};
