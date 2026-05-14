// RecentExtractions.tsx — List of past extraction jobs (right sidebar)
import React, { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { Trash2, ChevronRight, Clock } from 'lucide-react';
import { listJobs, deleteJob, type ExtractJobSummary } from '../../services/stemStudioApi';

interface RecentExtractionsProps {
  onSelectJob: (jobId: string) => void;
  activeJobId?: string;
  refreshTrigger?: number;  // bump to force refresh
}

function timeAgo(dateStr: string): string {
  // SQLite datetime('now') stores UTC without 'Z' suffix — force UTC interpretation
  const normalized = dateStr.includes('T') || dateStr.includes('Z') ? dateStr : dateStr.replace(' ', 'T') + 'Z';
  const diff = Date.now() - new Date(normalized).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export const RecentExtractions: React.FC<RecentExtractionsProps> = ({ onSelectJob, activeJobId, refreshTrigger }) => {
  const [jobs, setJobs] = useState<ExtractJobSummary[]>([]);
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      const data = await listJobs();
      setJobs(data);
    } catch (err) {
      console.warn('Failed to load recent extractions:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh, refreshTrigger]);

  const handleDelete = async (e: React.MouseEvent, jobId: string) => {
    e.stopPropagation();
    if (!confirm('Delete this extraction and all its stems?')) return;
    try {
      await deleteJob(jobId);
      setJobs(prev => prev.filter(j => j.id !== jobId));
    } catch (err) {
      console.error('Delete failed:', err);
    }
  };

  return (
    <div style={styles.container}>

      {loading && jobs.length === 0 && (
        <div style={styles.emptyMsg}>Loading...</div>
      )}

      {!loading && jobs.length === 0 && (
        <div style={styles.emptyMsg}>
          {t('stem.noExtractionsYet')}
          <br />
          <span style={{ fontSize: 11, color: '#555' }}>{t('stem.extractToSeeHere')}</span>
        </div>
      )}

      <div style={styles.list}>
        {jobs.map(job => (
          <button
            key={job.id}
            onClick={() => onSelectJob(job.id)}
            style={{
              ...styles.jobItem,
              background: activeJobId === job.id ? 'rgba(167,139,250,0.12)' : 'rgba(255,255,255,0.02)',
              borderColor: activeJobId === job.id ? 'rgba(167,139,250,0.25)' : 'rgba(255,255,255,0.05)',
            }}
          >
            <div style={styles.jobHeader}>
              <span style={styles.jobName}>{job.sourceFileName || 'Unknown'}</span>
              <ChevronRight size={12} style={{ color: '#555', flexShrink: 0 }} />
            </div>
            <div style={styles.jobMeta}>
              {job.type === 'supersep' ? (
                <span style={styles.typeBadgeSep}>SuperSep</span>
              ) : (
                <span style={styles.typeBadgeExtract}>Extract</span>
              )}
              <span style={styles.stemCount}>{job.completedStems?.length || 0} stems</span>
              <span style={styles.jobTime}>
                <Clock size={10} /> {timeAgo(job.createdAt)}
              </span>
              <button
                onClick={(e) => handleDelete(e, job.id)}
                style={styles.deleteBtn}
                title={t('stem.deleteExtraction')}
              >
                <Trash2 size={12} />
              </button>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    height: '100%',
  },
  title: {
    margin: 0,
    fontSize: 13,
    fontWeight: 600,
    color: '#a3a3a3',
    letterSpacing: '0.02em',
    textTransform: 'uppercase',
    padding: '0 4px',
  },
  emptyMsg: {
    padding: 20,
    textAlign: 'center',
    color: '#666',
    fontSize: 12,
    lineHeight: 1.6,
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
    flex: 1,
    overflowY: 'auto',
  },
  jobItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
    padding: '8px 10px',
    borderRadius: 8,
    border: '1px solid',
    cursor: 'pointer',
    textAlign: 'left',
    transition: 'all 0.15s ease',
    width: '100%',
  },
  jobHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 4,
  },
  jobName: {
    fontSize: 12,
    fontWeight: 600,
    color: '#d4d4d4',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    flex: 1,
  },
  jobMeta: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  stemCount: {
    fontSize: 10,
    color: '#a78bfa',
    fontWeight: 600,
    padding: '1px 6px',
    borderRadius: 4,
    background: 'rgba(167,139,250,0.1)',
  },
  typeBadgeExtract: {
    fontSize: 9,
    color: '#a78bfa',
    fontWeight: 700,
    padding: '1px 5px',
    borderRadius: 3,
    background: 'rgba(167,139,250,0.1)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.04em',
  },
  typeBadgeSep: {
    fontSize: 9,
    color: '#22c55e',
    fontWeight: 700,
    padding: '1px 5px',
    borderRadius: 3,
    background: 'rgba(34,197,94,0.1)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.04em',
  },
  jobTime: {
    fontSize: 10,
    color: '#666',
    display: 'flex',
    alignItems: 'center',
    gap: 3,
    flex: 1,
  },
  deleteBtn: {
    background: 'none',
    border: 'none',
    color: '#555',
    cursor: 'pointer',
    padding: 4,
    borderRadius: 4,
    display: 'flex',
    alignItems: 'center',
    transition: 'color 0.1s ease',
  },
};
