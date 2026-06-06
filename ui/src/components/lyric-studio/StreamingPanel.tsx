import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2, ChevronDown, ChevronUp, Terminal, SkipForward } from 'lucide-react';

interface StreamingPanelProps {
  visible: boolean;
  streamText: string;
  phase: string;
  done: boolean;
  onSkipThinking?: () => void;
}

export const StreamingPanel: React.FC<StreamingPanelProps> = ({
  visible, streamText, phase, done, onSkipThinking,
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const { t } = useTranslation();
  const [skipRequested, setSkipRequested] = useState(false);
  const preRef = useRef<HTMLPreElement>(null);
  const scrollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Only render the last chunk of text to avoid DOM/layout explosion
  const MAX_DISPLAY = 50_000; // ~50KB visible in the panel
  const displayText = useMemo(() => {
    if (streamText.length <= MAX_DISPLAY) return streamText;
    return '…(earlier output trimmed)…\n' + streamText.slice(-MAX_DISPLAY);
  }, [streamText]);

  // Debounced auto-scroll — avoids forcing layout reflow on every chunk
  useEffect(() => {
    if (collapsed) return;
    if (scrollTimerRef.current) clearTimeout(scrollTimerRef.current);
    scrollTimerRef.current = setTimeout(() => {
      if (preRef.current) preRef.current.scrollTop = preRef.current.scrollHeight;
    }, 100);
  }, [displayText, collapsed]);

  // Reset skip state when a new stream starts
  useEffect(() => {
    if (!done && streamText === '') {
      setSkipRequested(false);
    }
  }, [done, streamText]);

  if (!visible) return null;

  // Detect if model is currently inside a thinking block.
  // Only scan the tail of the text — we just need the last open/close tag.
  const isThinking = useMemo(() => {
    if (done || skipRequested) return false;
    // Check last 500 chars for unclosed thinking tags
    const tail = streamText.slice(-500);
    const lastThinkOpen = tail.lastIndexOf('<think>');
    const lastThinkClose = tail.lastIndexOf('</think>');
    const lastChannelOpen = tail.lastIndexOf('<|channel>thought');
    const lastChannelClose = tail.lastIndexOf('<channel|>');
    return (lastThinkOpen > lastThinkClose) || (lastChannelOpen > lastChannelClose);
  }, [streamText, done, skipRequested]);

  const handleSkip = () => {
    setSkipRequested(true);
    onSkipThinking?.();
  };

  return (
    <div className="rounded-xl overflow-hidden transition-all bg-zinc-100/60 dark:bg-zinc-900/60 border border-zinc-200 dark:border-white/5">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center gap-2 text-xs font-medium text-zinc-600 dark:text-zinc-400 hover:text-zinc-800 dark:text-zinc-200 transition-colors"
        >
          <Terminal className="w-3 h-3 text-pink-400" />
          {t('lyric.llmOutput')}
          {phase && (
            <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-pink-500 text-white">
              {phase}
            </span>
          )}
          {!done && (
            <Loader2 className="w-3 h-3 animate-spin text-pink-400" />
          )}
          {collapsed ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
        </button>

        {/* Skip Thinking button */}
        {isThinking && onSkipThinking && (
          <button
            onClick={handleSkip}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold bg-pink-500 text-white hover:bg-pink-600 transition-colors"
            title="Stop the model's chain-of-thought and produce output immediately"
          >
            <SkipForward className="w-3 h-3" />
            {t('lyric.skipThinking')}
          </button>
        )}
        {skipRequested && !done && (
          <span className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium text-pink-400">
            <Loader2 className="w-3 h-3 animate-spin" />
            {t('lyric.skipping')}
          </span>
        )}
      </div>

      {/* Content */}
      {!collapsed && (
        <pre
          ref={preRef}
          className="px-4 pb-3 text-xs leading-relaxed overflow-y-auto whitespace-pre-wrap break-words text-zinc-600 dark:text-zinc-400 scrollbar-hide"
          style={{
            maxHeight: '300px',
            fontFamily: 'ui-monospace, "Cascadia Code", "Fira Code", Menlo, monospace',
          }}
        >
          {displayText || (done ? t('lyric.noOutput') : t('lyric.waitingForLlm'))}
        </pre>
      )}
    </div>
  );
};
