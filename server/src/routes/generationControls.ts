import { Router, type Request } from 'express';
import { GENERATION_DEFAULTS } from '../contracts/generationControls.js';
import { CORE_GENERATION_PARAM_DESCRIPTORS } from '../contracts/generationControlDescriptors.js';
import { getUserId } from './auth.js';

export function createGenerationControlsRouter(userId: (req: Request) => string | null = getUserId): Router {
  const router = Router();
  router.get('/defaults', (req, res) => {
    if (!userId(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
    res.json(GENERATION_DEFAULTS);
  });
  router.get('/descriptors', (req, res) => {
    if (!userId(req)) { res.status(401).json({ error: 'Unauthorized' }); return; }
    res.json(CORE_GENERATION_PARAM_DESCRIPTORS);
  });
  return router;
}

export default createGenerationControlsRouter();
