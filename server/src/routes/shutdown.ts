// shutdown.ts — Graceful shutdown endpoint
//
// POST /api/shutdown — kills ace-server child process and exits Node.js

import { Router } from 'express';

const router = Router();

// POST /api/shutdown — terminate everything
router.post('/', (_req, res) => {
  console.log('[Server] Shutdown requested via API');
  res.json({ success: true, message: 'Shutting down...' });

  // Give the response time to send, then exit
  setTimeout(() => {
    process.emit('SIGINT');
  }, 500);
});

export default router;
