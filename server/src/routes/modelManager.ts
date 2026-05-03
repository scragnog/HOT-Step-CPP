// modelManager.ts — Model download and management API routes
//
// GET  /api/model-manager/registry         — registry + installed status
// POST /api/model-manager/download         — start download { fileId }
// GET  /api/model-manager/downloads        — SSE stream of download progress
// POST /api/model-manager/download/:id/cancel  — cancel download
// POST /api/model-manager/download/:id/resume  — resume download
// DELETE /api/model-manager/files/:filename    — delete installed model

import { Router } from 'express';
import { modelDownloadService } from '../services/modelDownloadService.js';

const router = Router();

// GET /api/model-manager/registry
router.get('/registry', (_req, res) => {
  try {
    const data = modelDownloadService.getRegistry();
    res.json(data);
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/model-manager/download
router.post('/download', (req, res) => {
  try {
    const { fileId } = req.body;
    if (!fileId) {
      res.status(400).json({ error: 'fileId is required' });
      return;
    }
    const jobId = modelDownloadService.startDownload(fileId);
    res.json({ jobId });
  } catch (err: any) {
    res.status(400).json({ error: err.message });
  }
});

// GET /api/model-manager/downloads — SSE stream
router.get('/downloads', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();

  // Send initial state
  const sendUpdate = () => {
    const jobs = modelDownloadService.getJobs();
    res.write(`data: ${JSON.stringify({ jobs })}\n\n`);
  };

  sendUpdate();

  // Send updates on progress events
  const onProgress = () => sendUpdate();
  modelDownloadService.on('progress', onProgress);

  // Also send periodic updates (in case events are missed)
  const interval = setInterval(sendUpdate, 1000);

  // Cleanup on disconnect
  req.on('close', () => {
    modelDownloadService.off('progress', onProgress);
    clearInterval(interval);
  });
});

// POST /api/model-manager/download/:jobId/cancel
router.post('/download/:jobId/cancel', (req, res) => {
  const ok = modelDownloadService.cancelDownload(req.params.jobId);
  res.json({ ok });
});

// POST /api/model-manager/download/:jobId/resume
router.post('/download/:jobId/resume', (req, res) => {
  try {
    const jobId = modelDownloadService.resumeDownload(req.params.jobId);
    res.json({ jobId });
  } catch (err: any) {
    res.status(400).json({ error: err.message });
  }
});

// DELETE /api/model-manager/files/:filename
router.delete('/files/:filename', (req, res) => {
  try {
    const ok = modelDownloadService.deleteFile(req.params.filename);
    res.json({ ok });
  } catch (err: any) {
    res.status(400).json({ error: err.message });
  }
});

export default router;
