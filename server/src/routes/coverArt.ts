// coverArt.ts — API routes for AI cover art generation
//
// Endpoints:
//   GET    /api/cover-art/status          — Installation/download status
//   POST   /api/cover-art/download        — Start first-use download
//   POST   /api/cover-art/download/cancel — Cancel active download
//   POST   /api/cover-art/generate        — Generate cover art for a song
//   GET    /api/cover-art/generate/:jobId — Poll generation job status

import { Router } from 'express';
import { getUserId } from './auth.js';
import { generateCoverArt, getCoverArtReadiness, type CoverArtResult } from '../services/coverArt/coverArtService.js';
import { coverArtDownloader } from '../services/coverArt/coverArtDownloader.js';

const router = Router();

// ── In-memory job tracking ──────────────────────────────────────────────

interface CoverArtJob {
  id: string;
  songId: string;
  status: 'pending' | 'running' | 'succeeded' | 'failed';
  result?: CoverArtResult;
  error?: string;
  createdAt: number;
}

const jobs = new Map<string, CoverArtJob>();

// Clean up old jobs periodically (keep last 100)
function cleanupJobs(): void {
  if (jobs.size <= 100) return;
  const sorted = Array.from(jobs.entries())
    .sort((a, b) => a[1].createdAt - b[1].createdAt);
  const toRemove = sorted.slice(0, sorted.length - 100);
  for (const [id] of toRemove) jobs.delete(id);
}

// ── GET /status — Installation status ───────────────────────────────────

router.get('/status', (_req, res) => {
  const readiness = getCoverArtReadiness();
  const downloadStatus = coverArtDownloader.getStatus();

  res.json({
    installed: readiness.installed,
    missingFiles: readiness.missingFiles,
    dir: readiness.dir,
    download: {
      phase: downloadStatus.phase,
      files: downloadStatus.files,
      totalBytes: downloadStatus.totalBytes,
      downloadedBytes: downloadStatus.downloadedBytes,
      overallProgress: downloadStatus.overallProgress,
      sdCliMissing: downloadStatus.sdCliMissing,
    },
  });
});

// ── POST /download — Start first-use download ───────────────────────────

router.post('/download', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const status = coverArtDownloader.getStatus();
  if (status.phase === 'downloading') {
    res.json({ ok: true, message: 'Download already in progress' });
    return;
  }

  // Fire and forget — client polls /status for progress
  coverArtDownloader.startDownload().catch(err => {
    console.error('[CoverArt] Download failed:', err.message);
  });

  res.json({ ok: true, message: 'Download started' });
});

// ── POST /download/cancel — Cancel active download ──────────────────────

router.post('/download/cancel', (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  coverArtDownloader.cancelDownload();
  res.json({ ok: true, message: 'Download cancelled' });
});

// ── SSE /download/progress — Stream download progress ───────────────────

router.get('/download/progress', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const sendProgress = () => {
    const status = coverArtDownloader.getStatus();
    res.write(`data: ${JSON.stringify(status)}\n\n`);
  };

  // Send initial state
  sendProgress();

  // Subscribe to progress updates
  coverArtDownloader.on('progress', sendProgress);

  // Cleanup on disconnect
  req.on('close', () => {
    coverArtDownloader.removeListener('progress', sendProgress);
  });
});

// ── POST /generate — Generate cover art for a song ──────────────────────

router.post('/generate', async (req, res) => {
  const userId = getUserId(req);
  if (!userId) { res.status(401).json({ error: 'Unauthorized' }); return; }

  const { songId, title, style, lyrics, subject } = req.body;
  if (!songId) {
    res.status(400).json({ error: 'songId is required' });
    return;
  }

  // Check readiness
  const readiness = getCoverArtReadiness();
  if (!readiness.installed) {
    res.status(503).json({
      error: 'Cover art not installed',
      missingFiles: readiness.missingFiles,
    });
    return;
  }

  // Create job
  const jobId = `ca-${Date.now().toString(36)}`;
  const job: CoverArtJob = {
    id: jobId,
    songId,
    status: 'pending',
    createdAt: Date.now(),
  };
  jobs.set(jobId, job);
  cleanupJobs();

  // Return immediately, run generation async
  res.json({ jobId, status: 'pending' });

  // Fire generation
  job.status = 'running';
  try {
    const result = await generateCoverArt({
      songId,
      title: title || '',
      style: style || '',
      lyrics: lyrics || '',
      subject: subject || '',
    });
    job.status = 'succeeded';
    job.result = result;
  } catch (err: any) {
    job.status = 'failed';
    job.error = err.message;
    console.error(`[CoverArt] Generation failed for song ${songId}:`, err.message);
  }
});

// ── GET /generate/:jobId — Poll generation status ───────────────────────

router.get('/generate/:jobId', (req, res) => {
  const job = jobs.get(req.params.jobId);
  if (!job) {
    res.status(404).json({ error: 'Job not found' });
    return;
  }

  res.json({
    jobId: job.id,
    songId: job.songId,
    status: job.status,
    result: job.result,
    error: job.error,
  });
});

export default router;
