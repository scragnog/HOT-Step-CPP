// lireek/mm3Routes.ts — deterministic MiniMax-Music3 caption composition
//
// No LLM, no network, no provider selection: these routes are pure functions
// over MiniMax's own 1,000-caption reference corpus. See
// services/lireek/mm3Compose.ts for why the composer is not a model call.
// Registered on the parent router by lireek.ts.

import type { Router, Request, Response } from 'express';
import { composeMm3Caption, parseBrief, type Mm3Controls } from '../../services/lireek/mm3Compose.js';
import { getMm3Corpus } from '../../services/lireek/mm3Corpus.js';

/** Accepts only the fields the composer reads, so a full params blob is safe to post. */
function pickControls(raw: unknown): Mm3Controls {
  const c = (raw ?? {}) as Record<string, unknown>;
  const gender = c.vocalGender;
  return {
    bpm: Number(c.bpm) || 0,
    keyScale: typeof c.keyScale === 'string' ? c.keyScale : '',
    timeSignature: typeof c.timeSignature === 'string' ? c.timeSignature : '',
    duration: Number(c.duration ?? -1),
    vocalLanguage: typeof c.vocalLanguage === 'string' ? c.vocalLanguage : '',
    vocalGender:
      gender === 'male' || gender === 'female' || gender === 'duet' ? gender : '',
  };
}

export function registerMm3Routes(router: Router): void {
  /**
   * POST /api/lireek/mm3/compose
   * body: { brief, controls?, seed?, sections?, neighbourhood? }
   *
   * Returns the caption plus its provenance, resolved slots and notes. The
   * notes carry control/prompt conflicts and the controls MM3 cannot express —
   * the UI is expected to show them, not swallow them.
   */
  router.post('/mm3/compose', (req: Request, res: Response) => {
    try {
      const brief = typeof req.body?.brief === 'string' ? req.body.brief.trim() : '';
      if (!brief) {
        res.status(400).json({ error: 'brief required' });
        return;
      }

      const seedRaw = Number(req.body?.seed);
      const nRaw = Number(req.body?.neighbourhood);
      const result = composeMm3Caption(brief, {
        controls: pickControls(req.body?.controls),
        seed: Number.isFinite(seedRaw) && seedRaw > 0 ? Math.floor(seedRaw) : 1,
        sections: Array.isArray(req.body?.sections) ? req.body.sections.map(String) : undefined,
        neighbourhood: Number.isFinite(nRaw) && nRaw >= 4 && nRaw <= 200 ? Math.floor(nRaw) : undefined,
      });

      res.json(result);
    } catch (err) {
      console.error('[MM3] compose failed:', err);
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  /**
   * POST /api/lireek/mm3/parse-brief
   * body: { brief }
   *
   * The bpm / key / scale / gender the composer would read out of the user's own
   * prose. Lets the UI pre-fill or highlight the metadata controls without
   * committing to a full compose.
   */
  router.post('/mm3/parse-brief', (req: Request, res: Response) => {
    const brief = typeof req.body?.brief === 'string' ? req.body.brief : '';
    res.json(parseBrief(brief));
  });

  /**
   * GET /api/lireek/mm3/corpus-info
   * Corpus health, and the per-family gender pools — the UI uses the thin ones
   * to warn that e.g. female metal has very few references to draw on.
   */
  router.get('/mm3/corpus-info', (_req: Request, res: Response) => {
    try {
      const { cards, families } = getMm3Corpus();
      const byFamily: Record<string, { total: number; male: number; female: number; duet: number }> = {};
      for (const f of families) byFamily[f] = { total: 0, male: 0, female: 0, duet: 0 };
      for (const c of cards) {
        const row = byFamily[c.family];
        if (!row) continue;
        row.total++;
        if (c.gender === 'male' || c.gender === 'female' || c.gender === 'duet') row[c.gender]++;
      }
      res.json({ cards: cards.length, families: families.length, byFamily });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });
}
