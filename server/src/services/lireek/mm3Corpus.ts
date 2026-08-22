// mm3Corpus.ts — loader for MiniMax's compiled Structured Caption corpus
//
// The corpus is 1,000 reference captions, each parsed into 12 labelled columns
// and tagged with the family/style/tempo/key/gender metadata MiniMax publishes
// in its own index cards. Built by server/scripts/build-mm3-corpus.mjs; see
// that file for the source layout and how to refresh it.
//
// Why a compiled JSON rather than the 1,019 upstream files: `.claude/` does not
// ship in a portable release, and one 4 MB read at first use beats a thousand.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { PORTABLE_MODE, PROJECT_ROOT } from '../../config.js';

/** One reference caption plus the metadata the composer selects on. */
export interface Mm3Card {
  /** Template slug, e.g. `pop-punk-alternative-rock_0002`. */
  id: string;
  /** One of the 18 routing families. */
  family: string;
  /** MiniMax's own genre string, always a slash pair or triple. */
  style: string;
  /** Family this card declares as its secondary route, or '' for none. */
  secondary: string;
  bpm: number;
  key: string;
  scale: string;
  gender: 'male' | 'female' | 'duet' | 'unknown';
  /** label -> section words that label's prose mentions (normalised, singular). */
  sections: Record<string, string[]>;
  /** label -> the reference prose for that column. */
  fields: Record<string, string>;
}

export interface Mm3Corpus {
  labels: string[];
  families: string[];
  cards: Mm3Card[];
}

const corpusPath = PORTABLE_MODE
  ? path.join(PROJECT_ROOT, 'server', 'data', 'mm3-corpus.json')
  : path.join(path.dirname(fileURLToPath(import.meta.url)), '..', '..', 'data', 'mm3-corpus.json');

let cached: Mm3Corpus | null = null;

/** Loads (once) and returns the compiled corpus. Throws if it is missing. */
export function getMm3Corpus(): Mm3Corpus {
  if (cached) return cached;
  if (!fs.existsSync(corpusPath)) {
    throw new Error(
      `MM3 caption corpus not found at ${corpusPath} — run: node server/scripts/build-mm3-corpus.mjs`,
    );
  }
  cached = JSON.parse(fs.readFileSync(corpusPath, 'utf-8')) as Mm3Corpus;
  return cached;
}
