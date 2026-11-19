import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * "Copied" flash for a copy button or row.
 *
 * Six surfaces tracked this by hand with a boolean (or the copied path) plus a
 * setTimeout they never cleared: a copy followed by unmount set state on a dead
 * component, and a second click left two timers racing to reset the label.
 *
 * Returns `[copied, markCopied, clearCopied]`. The flag clears itself after
 * `resetMs`; clearCopied is for callers that hide the button first.
 */
export function useCopiedFlag(resetMs = 1500): [boolean, () => void, () => void] {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const stop = useCallback(() => {
    if (timer.current !== null) { clearTimeout(timer.current); timer.current = null; }
  }, []);
  const clearCopied = useCallback(() => { stop(); setCopied(false); }, [stop]);
  const markCopied = useCallback(() => {
    stop();
    setCopied(true);
    timer.current = setTimeout(() => { timer.current = null; setCopied(false); }, resetMs);
  }, [stop, resetMs]);
  useEffect(() => stop, [stop]);
  return [copied, markCopied, clearCopied];
}

/** Keyed variant for lists: remembers which path was copied. */
export function useCopiedKey(resetMs = 1500): [string, (key: string) => void, () => void] {
  const [copied, setCopied] = useState('');
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const stop = useCallback(() => {
    if (timer.current !== null) { clearTimeout(timer.current); timer.current = null; }
  }, []);
  const clearCopied = useCallback(() => { stop(); setCopied(''); }, [stop]);
  const markCopied = useCallback((key: string) => {
    stop();
    setCopied(key);
    timer.current = setTimeout(() => { timer.current = null; setCopied(''); }, resetMs);
  }, [stop, resetMs]);
  useEffect(() => stop, [stop]);
  return [copied, markCopied, clearCopied];
}
