// LyricEditorRepairPopover.tsx — Repair timings, and the choices it runs with.
//
// The button used to do exactly one thing, with its rules hard-coded. Those
// rules are not right for every track: a chorus that comes back wants the repeat
// pass, a legato vocal wants its long tails left alone, a spoken verse wants
// onsets pulled onto consonants but nothing copied, and a track someone has
// already hand-checked wants the order repair kept away from its anchors.
//
// The panel is the selection. Each switch is one pass, each has the house hover
// card explaining what it will and will not touch, and Apply runs the pass with
// those switches and shows the diff as usual — nothing is written until the
// preview is accepted.
//
// The panel is portalled and fixed-positioned for the same reason ParamLabel's
// card is: the row sits inside the modal's `overflow-hidden` box, so an
// absolutely-positioned panel would be clipped exactly when it matters. It opens
// upward from the trigger because the trigger is near the bottom of the window.

import React, { useLayoutEffect, useRef, useState } from 'react';
import ReactDOM from 'react-dom';
import { ChevronDown, ChevronUp, Wand2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { Toggle } from '@/components/shared/controls/Toggle';
import { BTN } from './editorStyles';
import type { RealignSettings } from '@/storage/lyricEditorSettings';
import { useClickOutside } from '@/hooks/useClickOutside';
import { useEscapeKey } from '@/hooks/useEscapeKey';
import { useGlobalEvent } from '@/hooks/useGlobalEvent';

const PANEL_W = 340;
const GAP = 6;

const APPLY_BTN = 'rounded-md border border-pink-500 bg-pink-500 px-2 py-0.5 text-[11px] font-semibold '
  + 'text-white hover:bg-pink-400 disabled:pointer-events-none disabled:opacity-40';

export interface LyricEditorRepairPopoverProps {
  settings: RealignSettings;
  onChange: (next: RealignSettings) => void;
  /** Runs the pass with the current settings and closes the panel. */
  onApply: () => void;
  disabled?: boolean;
}

export const LyricEditorRepairPopover: React.FC<LyricEditorRepairPopoverProps> = ({
  settings, onChange, onApply, disabled,
}) => {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; bottom: number; maxHeight: number } | null>(null);
  const anchorRef = useRef<HTMLSpanElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  const measure = () => {
    const rect = anchorRef.current?.getBoundingClientRect();
    if (!rect) return;
    setPos({
      left: Math.max(8, Math.min(rect.left, window.innerWidth - PANEL_W - 8)),
      bottom: window.innerHeight - rect.top + GAP,
      // Open upward; never taller than the room above the trigger.
      maxHeight: Math.max(180, rect.top - GAP * 2),
    });
  };

  useLayoutEffect(() => { if (open) measure(); }, [open]);
  useGlobalEvent('scroll', measure, { enabled: open, capture: true });
  useGlobalEvent('resize', measure, { enabled: open });

  useClickOutside([panelRef, anchorRef], () => setOpen(false), open, 'pointerdown');
  // Capture, and stop the event: Escape closes the panel, not the editor.
  useEscapeKey((e) => { e.stopPropagation(); setOpen(false); }, open, true);

  const set = (key: keyof RealignSettings, value: boolean) => {
    onChange({ ...settings, [key]: value });
  };

  return (
    <span ref={anchorRef} className="inline-flex">
      <button
        type="button"
        className={BTN}
        onClick={() => setOpen(value => !value)}
        disabled={disabled}
        aria-haspopup="dialog"
        aria-expanded={open}
      >
        <Wand2 size={12} /> {t('lyrics.editor.repairTimings', 'Repair timings')}
        {open ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
      </button>

      {open && pos && ReactDOM.createPortal(
        <div
          ref={panelRef}
          role="dialog"
          aria-label={t('lyrics.editor.repairSettings', 'Repair timings settings')}
          className="fixed z-[16000] rounded-xl border border-zinc-200 bg-white p-3
                     shadow-2xl shadow-black/25 dark:border-white/10 dark:bg-zinc-900"
          style={{
            left: pos.left,
            bottom: pos.bottom,
            width: PANEL_W,
            maxHeight: pos.maxHeight,
            overflowY: 'auto',
          }}
        >
          <div className="text-xs font-bold text-zinc-800 dark:text-zinc-100">
            {t('lyrics.editor.repairSettings', 'Repair timings settings')}
          </div>
          <p className="mt-1 text-[10px] leading-relaxed text-zinc-500 dark:text-zinc-400">
            {t(
              'lyrics.editor.repairSettingsIntro',
              'Choose which repairs this pass may make. It still previews the diff first: nothing is written until you accept it.',
            )}
          </p>

          <div className="mt-2.5 flex flex-col gap-2">
            <Toggle
              size="sm"
              accent="pink"
              checked={settings.repeats}
              onChange={value => set('repeats', value)}
              label={t('lyrics.editor.repairRepeats', 'Reuse repeated lines')}
              info={t(
                'lyrics.editor.repairRepeatsInfo',
                'Copies a line’s word offsets onto every later occurrence of the same words, exactly or nearly so inside one section. A chorus fixed by ear becomes the source for the choruses after it. Off: every occurrence keeps the timing the aligner gave it.',
              )}
              disabled={disabled}
            />
            <Toggle
              size="sm"
              accent="pink"
              checked={settings.cutBack}
              onChange={value => set('cutBack', value)}
              label={t('lyrics.editor.repairCutBack', 'Trim swallowed tails')}
              info={t(
                'lyrics.editor.repairCutBackInfo',
                'Shortens a word whose tail runs into the next onset, or whose tail sits over silence instead of a held note. A tail that keeps sounding is a melisma and is left alone. Off: long tails are reported as flagged but not changed.',
              )}
              disabled={disabled}
            />
            <Toggle
              size="sm"
              accent="pink"
              checked={settings.monotonic}
              onChange={value => set('monotonic', value)}
              label={t('lyrics.editor.repairMonotonic', 'Fix out-of-order words')}
              info={t(
                'lyrics.editor.repairMonotonicInfo',
                'Pushes a word that starts before the previous word in its vocal layer has ended, keeping its duration and stopping at the next onset. Hand-ticked lines are anchors and are never moved either way. Off: an overlap is left exactly as the aligner wrote it.',
              )}
              disabled={disabled}
            />
            <Toggle
              size="sm"
              accent="pink"
              checked={settings.snap}
              onChange={value => set('snap', value)}
              label={t('lyrics.editor.repairSnap', 'Snap onsets to the waveform')}
              info={t(
                'lyrics.editor.repairSnapInfo',
                'Moves a word with low confidence, or one left long after the tail test, onto a voiced rise within 120 ms that keeps a plausible duration. A correct onset stays where the aligner put it. Off: flagged onsets are not moved.',
              )}
              disabled={disabled}
            />
          </div>

          <div className="mt-3 flex items-center justify-end gap-1.5">
            <button type="button" className={BTN} onClick={() => setOpen(false)}>
              {t('common.cancel', 'Cancel')}
            </button>
            <button
              type="button"
              className={APPLY_BTN}
              onClick={() => { onApply(); setOpen(false); }}
              disabled={disabled}
            >
              {t('lyrics.editor.repairApply', 'Apply & run')}
            </button>
          </div>
        </div>,
        document.body,
      )}
    </span>
  );
};

export default LyricEditorRepairPopover;
