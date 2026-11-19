// LyricPeaks.tsx — the waveform above the timeline, drawn for the visible window.
//
// The canvas and its wrapper are own refs here rather than the modal's, because
// the only thing that needs them is the hook that paints into them. Sizing and
// sampling belong to `useVisiblePeaks`; this file is the strip it paints on.
//
// Dragging across it is the shortest way to say "this part of the song". The
// band stays on screen after the pointer is lifted, so the selection is visible
// while the rows below are checked; either edge resizes it, and the grip on its
// bottom edge moves it. Anywhere else draws a new section — including over the old
// band's body, because pressing the waveform and dragging has to mean "select
// this", never "the band that was already there is in the way". The two
// buttons that come with it replay it or run Whisper and the aligner over the
// lines it covers. A drag shorter than a few pixels is a slip of the hand, not a
// selection, so it changes nothing.
//
// The explanation lives behind the info icon in the corner, in the same hover
// card as every other control here: a native `title` looked like nothing else
// in the app and never appeared on touch (rule 3 of docs/dev/ui-design.md).

import React, { useRef, useState } from 'react';
import { Info } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { HintButton } from '@/components/shared/controls/HintButton';
import { BTN, ICON_BTN } from './editorStyles';
import { MIN_SPAN_SEC, PEAKS_H } from './timingModel';
import { useVisiblePeaks, type PeaksLike } from './useVisiblePeaks';

/** Below this many pixels a drag is a click, and a click selects nothing. */
const DRAG_MIN_PX = 6;
/** How close to an edge the pointer counts as grabbing that edge. */
const HANDLE_PX = 8;
/** The grip at the band's bottom centre: the only part that moves the band. */
const MOVE_GRIP_W = 26;
const MOVE_GRIP_H = 12;
/** A readable surface for controls that sit over the waveform. */
const OVERLAY_SURFACE = 'bg-white/95 shadow-sm backdrop-blur-sm dark:bg-zinc-900/95';

type Section = { start: number; end: number };
/** A press either starts a section, moves it, or grabs one of its two edges. */
type GestureMode = 'create' | 'move' | 'start' | 'end';

export interface LyricPeaksProps {
  peaks: PeaksLike | null | undefined;
  total: number;
  pxPerSec: number;
  scrollLeft: number;
  /** The section selected on the waveform, in seconds, when there is one. */
  span?: Section | null;
  /** A section drawn, moved or resized; null when it is thrown away. */
  onSpanChange?: (span: Section | null) => void;
  /**
   * The timeline pane this strip is the map of. Its live offset is read while a
   * gesture runs: the `scrollLeft` prop only catches up on the next scroll
   * event, and a stale offset put a press seconds away from where it looked.
   */
  scrollerRef?: React.RefObject<HTMLDivElement | null>;
  /** Whether the section loop is armed, and the toggle that arms it. */
  looping?: boolean;
  onLoop?: () => void;
  /** Runs Whisper and the aligner over the lines the section covers. */
  onRealign?: () => void;
  realignDisabled?: boolean;
  /** Turns the section into its own song by transcribing a YuE2 ABC lead sheet. */
  onMelody?: () => void;
  melodyBusy?: boolean;
}

export const LyricPeaks: React.FC<LyricPeaksProps> = ({
  peaks, total, pxPerSec, scrollLeft, span, onSpanChange, scrollerRef, looping, onLoop, onRealign,
  realignDisabled, onMelody, melodyBusy,
}) => {
  const { t } = useTranslation();
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // `px` is the zoom the press started at: a zoom that lands mid-gesture must
  // not rescale the seconds already crossed.
  const [gesture, setGesture] = useState<{
    mode: GestureMode; origin: number; base: Section | null; px: number;
  } | null>(null);
  const [preview, setPreview] = useState<Section | null>(null);
  /** While a gesture runs its span is the truth; otherwise the prop is. */
  const drawn = preview ?? span ?? null;
  const interactive = !!onSpanChange && pxPerSec > 0;

  useVisiblePeaks({
    wrapRef, canvasRef, peaks, total, pxPerSec, scrollLeft, height: PEAKS_H,
    span: drawn, dragging: !!preview,
  });

  /** The pane's offset as it is now, not as of the last render. */
  const scrollNow = () => scrollerRef?.current?.scrollLeft ?? scrollLeft;

  /** Pointer x to seconds, in the window the canvas is showing. */
  const timeAt = (clientX: number, element: HTMLElement, px: number): number => {
    const rect = element.getBoundingClientRect();
    const seconds = (scrollNow() + Math.max(0, clientX - rect.left)) / (px || 1);
    return Math.max(0, Math.min(total || seconds, seconds));
  };

  /**
   * The section a gesture would be at `at` seconds.
   *
   * Shared by the move and the release: a fast drag can lift the pointer well
   * past the last move event, and committing the move's position instead of the
   * release's is what made a selection sometimes end short of where it looked.
   */
  const spanAt = (
    mode: GestureMode, origin: number, base: Section | null, at: number, fallback: Section | null,
  ): Section | null => {
    if (mode === 'create') return { start: Math.min(origin, at), end: Math.max(origin, at) };
    if (!base) return fallback;
    if (mode === 'move') {
      const duration = base.end - base.start;
      const limit = total > 0 ? Math.max(0, total - duration) : Number.POSITIVE_INFINITY;
      const start = Math.max(0, Math.min(limit, base.start + (at - origin)));
      return { start, end: start + duration };
    }
    if (mode === 'start') {
      return { start: Math.max(0, Math.min(at, base.end - MIN_SPAN_SEC)), end: base.end };
    }
    const limit = total > 0 ? total : at;
    return { start: base.start, end: Math.max(base.start + MIN_SPAN_SEC, Math.min(limit, at)) };
  };

  const begin = (event: React.PointerEvent<HTMLDivElement>) => {
    // A right or middle press is not a selection: it used to start one, which
    // is how a stray context-menu press could replace the band.
    if (!interactive || event.button !== 0) return;
    const px = pxPerSec || 1;
    const at = timeAt(event.clientX, event.currentTarget, px);
    const rect = event.currentTarget.getBoundingClientRect();
    let mode: GestureMode = 'create';
    let base: Section | null = null;
    if (span) {
      // Hit-tested on the band's own pixels rather than on child elements, so a
      // press anywhere lands in exactly one gesture. The offset and the scale
      // are the live ones: reading a stale scroll offset put the press inside
      // where the band used to be drawn.
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      // Seconds to pixels first, then the pane's offset (which is pixels):
      // `(start - scrollNow()) * px` mixes the two units and puts the band's
      // edges tens of thousands of pixels away once the view has been panned.
      const left = span.start * px - scrollNow();
      const right = span.end * px - scrollNow();
      // Edges resize and the grip moves; everything else — the body included —
      // draws a new section. Moving only from the grip keeps a press-drag on the
      // waveform meaning what it looks like. The grip is offered only while the
      // band fits in the window: at high zoom it is off the visible strip.
      const fits = left >= -HANDLE_PX && right <= rect.width + HANDLE_PX;
      const gripLeft = (left + right) / 2 - MOVE_GRIP_W / 2;
      if (left >= -HANDLE_PX && Math.abs(x - left) <= HANDLE_PX) mode = 'start';
      else if (right <= rect.width + HANDLE_PX && Math.abs(x - right) <= HANDLE_PX) mode = 'end';
      else if (fits && y >= rect.height - MOVE_GRIP_H && x >= gripLeft && x <= gripLeft + MOVE_GRIP_W) {
        mode = 'move';
      }
      if (mode !== 'create') base = span;
    }
    // Every other pointer gesture in the editor prevents the browser's own
    // selection. Without this, a press can start a text selection across the
    // toolbar and arrive as a `pointercancel`, which throws the gesture away.
    event.preventDefault();
    setGesture({ mode, origin: at, base, px });
    setPreview(mode === 'create' ? { start: at, end: at } : null);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const move = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!gesture) return;
    const at = timeAt(event.clientX, event.currentTarget, gesture.px);
    setPreview(spanAt(gesture.mode, gesture.origin, gesture.base, at, null));
  };

  /** Commit on release, using the release's own position as the final edge. */
  const finish = (event?: React.PointerEvent<HTMLDivElement>) => {
    const current = gesture;
    const next = current && event
      ? spanAt(current.mode, current.origin, current.base, timeAt(event.clientX, event.currentTarget, current.px), preview)
      : preview;
    setGesture(null);
    setPreview(null);
    if (!current || !next) return;
    // A click (or a slip) is not a selection, and leaving the last one standing
    // is what makes a stray press harmless.
    if (current.mode === 'create' && Math.abs(next.end - next.start) * pxPerSec < DRAG_MIN_PX) return;
    if (next.end - next.start < MIN_SPAN_SEC) return;
    onSpanChange?.({ start: next.start, end: next.end });
  };

  // Seconds to pixels first, then the pane's pixel offset: the band has to sit
  // where the same stretch of waveform is painted, at any scroll position.
  const left = drawn ? drawn.start * pxPerSec - scrollLeft : 0;
  const width = drawn ? Math.max(0, (drawn.end - drawn.start) * pxPerSec) : 0;

  return (
    <div
      ref={wrapRef}
      className={`relative shrink-0 select-none overflow-hidden border-b border-zinc-200 bg-zinc-100 dark:border-white/10 dark:bg-black/40 ${
        interactive ? 'cursor-crosshair touch-none' : ''
      }`}
      style={{ height: PEAKS_H }}
      onPointerDown={begin}
      onPointerMove={move}
      onPointerUp={finish}
      // A cancelled pointer never selected anything, so it must not commit the
      // half-drawn band it happened to be showing.
      onPointerCancel={() => { setGesture(null); setPreview(null); }}
    >
      <canvas ref={canvasRef} className="block" />
      {drawn && (
        // The band is the selection: it stays after the drag, so the stretch a
        // run or a loop will act on is visible instead of remembered.
        <div
          aria-hidden
          // The tint itself is painted into the canvas by `useVisiblePeaks`, so
          // it is clipped to the window and shows at any zoom; this element only
          // supplies the edges, the handles and the hit area. It stays
          // transparent because a filtered layer this wide (thousands of pixels
          // at high zoom) is what stopped the pink from painting.
          className={`absolute inset-y-0 border-x-2 border-pink-400 ${
            interactive ? 'cursor-crosshair' : 'pointer-events-none'
          }`}
          style={{ left, width }}
        >
          <span className={`absolute inset-y-0 left-0 w-[6px] bg-pink-400/90 ${interactive ? 'cursor-ew-resize' : ''}`} />
          <span className={`absolute inset-y-0 right-0 w-[6px] bg-pink-400/90 ${interactive ? 'cursor-ew-resize' : ''}`} />
          {interactive && (
            // The band's only move target: a visible grip rather than a hidden
            // hotspot, so the body can stay a drawing surface. It sits at the
            // bottom because the strip's top edge carries the section buttons.
            <span
              className="absolute bottom-0 left-1/2 h-[10px] w-[26px] -translate-x-1/2 cursor-grab rounded-t border border-b-0 border-pink-400 bg-pink-400/90"
              title={t('lyrics.editor.moveSection', 'Drag to move the selected section')}
            />
          )}
        </div>
      )}
      {/* The surface is only on the buttons: a pointer-events-none container
          leaves the padding and gaps as waveform, so a drag started beside an
          icon is still a selection instead of a dead spot. */}
      <div className="pointer-events-none absolute right-1 top-1 z-20" onPointerDown={e => e.stopPropagation()}>
        <HintButton
          label={t('lyrics.editor.selectSection', 'Select a section')}
          info={t(
            'lyrics.editor.selectSectionInfo',
            'Drag across the waveform to choose a stretch of the song. The band stays on screen so the selection is visible; drag the grip on its bottom edge to move it, or either edge to make it longer or shorter. Dragging anywhere else draws a new section. Loop replays that stretch, and Re-align runs Whisper and the aligner over the lines it covers. A drag shorter than a few pixels is a slip of the hand and selects nothing.',
          )}
          className={`${ICON_BTN} ${OVERLAY_SURFACE}`}
          rootClassName="pointer-events-auto"
          ariaLabel={t('lyrics.editor.selectSection', 'Select a section')}
        >
          <Info size={12} />
        </HintButton>
      </div>
      {(span || looping) && (
        // The cluster stops its own presses so arming a loop never starts a
        // section drag under the pointer.
        <div className="pointer-events-none absolute left-1 top-1 z-20 flex items-center gap-1" onPointerDown={e => e.stopPropagation()}>
          {onLoop && (
            <HintButton
              label={t('lyrics.editor.loopSection', 'Loop section')}
              info={t(
                'lyrics.editor.loopSectionInfo',
                'Replays the stretch of waveform you selected until it is switched off. Arming it seeks just before the start, and the loop restarts whenever playback passes the end, so a phrase can be heard as many times as it takes. It stays here while it is on, even if the selection is cleared.',
              )}
              className={`${BTN} ${OVERLAY_SURFACE} ${looping ? 'border-pink-400 text-pink-400' : ''}`}
              rootClassName="pointer-events-auto"
              onClick={onLoop}
              disabled={!span && !looping}
              ariaPressed={looping}
            >
              {t('lyrics.editor.loopSection', 'Loop section')}
            </HintButton>
          )}
          {onMelody && (
            <HintButton
              label={t('lyrics.editor.melodySection', 'Convert to melody')}
              info={t(
                'lyrics.editor.melodySectionInfo',
                'Cuts this stretch out as its own track and transcribes it into a YuE2 ABC lead sheet. The new song is named after the current one with the moment it was made, so a verse or a chorus can be transcribed without the rest of the arrangement. It appears in the library when the transcription finishes.',
              )}
              className={`${BTN} ${OVERLAY_SURFACE}`}
              rootClassName="pointer-events-auto"
              onClick={onMelody}
              disabled={melodyBusy || !span}
            >
              {melodyBusy
                ? t('lyrics.editor.melodySectionBusy', 'Transcribing…')
                : t('lyrics.editor.melodySection', 'Convert to melody')}
            </HintButton>
          )}
          {onRealign && (
            <HintButton
              label={t('lyrics.editor.realignSection', 'Re-align section')}
              info={t(
                'lyrics.editor.realignSectionInfo',
                'Runs Whisper over this audio and then the CTC forced aligner over the lines this section covers, replacing only those lines. Nothing outside it is re-timed, and the result is a preview you review before applying. The full run is the same thing with the whole track selected.',
              )}
              className={`${BTN} ${OVERLAY_SURFACE}`}
              rootClassName="pointer-events-auto"
              onClick={onRealign}
              disabled={realignDisabled || !span}
            >
              {t('lyrics.editor.realignSection', 'Re-align section')}
            </HintButton>
          )}
        </div>
      )}
    </div>
  );
};

export default LyricPeaks;
