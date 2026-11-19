// QueryError.tsx — the inline read-failure line for direct renderers.
//
// GET failures never toast (only writes do), so a panel whose empty state is
// also its failure state would otherwise look like "no data". Direct renderers
// show this line instead, with a retry that re-runs the failed read.

import React from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';

interface Props {
  error?: unknown;
  onRetry?: () => void;
  /** Override the default "Could not load this data." line. */
  message?: string;
  className?: string;
  compact?: boolean;
}

export const QueryError: React.FC<Props> = ({ error, onRetry, message, className = '', compact = false }) => {
  const detail = error instanceof Error ? error.message : typeof error === 'string' ? error : '';
  return (
    <div
      role="alert"
      className={`flex items-center gap-2 rounded-lg border border-red-500/20 bg-red-500/10 px-2.5 py-1.5 text-[11px] text-red-300 ${compact ? '' : 'my-1'} ${className}`}
    >
      <AlertTriangle size={12} className="flex-shrink-0" />
      <span className="min-w-0 flex-1 truncate" title={detail || message}>
        {message || 'Could not load this data.'}{detail ? ` ${detail}` : ''}
      </span>
      {onRetry && (
        <button
          type="button"
          onClick={() => onRetry()}
          className="flex-shrink-0 inline-flex items-center gap-1 text-red-300/80 hover:text-red-200 transition-colors"
        >
          <RotateCcw size={11} />
          Retry
        </button>
      )}
    </div>
  );
};
