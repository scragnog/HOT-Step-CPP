// Fold.tsx — a titled collapsible section in the Song Info style whose
// open/closed state survives page loads (stored under `storageKey`).
import React from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { usePersistedState } from '@/hooks/usePersistedState';

export const Fold: React.FC<{
  title: React.ReactNode;
  /** localStorage key for the open state. */
  storageKey: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}> = ({ title, storageKey, defaultOpen = false, children }) => {
  const [open, setOpen] = usePersistedState(storageKey, defaultOpen);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs font-medium text-zinc-500 uppercase tracking-wider hover:text-zinc-700 dark:hover:text-zinc-300 transition-colors mb-1.5"
      >
        {open ? <ChevronDown size={12} className="text-zinc-500" /> : <ChevronRight size={12} className="text-zinc-500" />}
        {title}
      </button>
      {open && children}
    </div>
  );
};
