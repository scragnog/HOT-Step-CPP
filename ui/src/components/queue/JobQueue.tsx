// JobQueue.tsx — Display active generation jobs with progress
// Ported to Tailwind styling matching hot-step-9000.

import React from 'react';
import { X, Loader2, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';
import type { GenerationJob } from '../../types';

interface JobQueueProps {
  jobs: GenerationJob[];
  onCancel: (jobId: string) => void;
  onClearCompleted: () => void;
  onRemove: (jobId: string) => void;
}

const statusIcons: Record<string, React.ReactNode> = {
  pending: <Loader2 size={14} className="spinner text-zinc-400" />,
  lm_running: <Loader2 size={14} className="spinner text-purple-400" />,
  synth_running: <Loader2 size={14} className="spinner text-pink-400" />,
  succeeded: <CheckCircle size={14} className="text-green-400" />,
  failed: <XCircle size={14} className="text-red-400" />,
  cancelled: <AlertTriangle size={14} className="text-yellow-400" />,
};

export const JobQueue: React.FC<JobQueueProps> = ({ jobs, onCancel, onClearCompleted, onRemove }) => {
  if (jobs.length === 0) return null;

  const hasCompleted = jobs.some(j =>
    ['succeeded', 'failed', 'cancelled'].includes(j.status)
  );

  return (
    <div className="px-4 pt-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-zinc-300">Generation Queue</h3>
        {hasCompleted && (
          <button
            className="text-xs text-zinc-500 hover:text-white transition-colors"
            onClick={onClearCompleted}
          >
            Clear
          </button>
        )}
      </div>

      <div className="space-y-2">
        {jobs.map(job => (
          <div
            key={job.jobId}
            className={`
              rounded-xl border px-3 py-2.5 transition-all
              ${job.status === 'succeeded' ? 'bg-green-500/5 border-green-500/20' :
                job.status === 'failed' ? 'bg-red-500/5 border-red-500/20' :
                'bg-zinc-800/50 border-white/5'}
            `}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {statusIcons[job.status] || null}
                <span className="text-sm text-zinc-300 capitalize">
                  {job.stage || job.status.replace('_', ' ')}
                </span>
              </div>
              <button
                className="p-1 rounded-lg text-zinc-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                onClick={() => {
                  if (['pending', 'lm_running', 'synth_running'].includes(job.status)) {
                    onCancel(job.jobId);
                  } else {
                    onRemove(job.jobId);
                  }
                }}
              >
                <X size={14} />
              </button>
            </div>

            {job.progress !== undefined && job.progress > 0 && job.progress < 100 && (
              <div className="mt-2 h-1 bg-zinc-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-pink-500 to-purple-500 rounded-full transition-all duration-300"
                  style={{ width: `${job.progress}%` }}
                />
              </div>
            )}

            {job.status === 'failed' && job.error && (
              <div className="mt-2 text-xs text-red-400 break-words">{job.error}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
